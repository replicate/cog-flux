from contextlib import contextmanager
import os
import time
from typing import List, Tuple

from einops import rearrange

from cog import Path
from flux.modules.image_embedders import CannyImageEncoder
from flux.sampling import (
    denoise,
    get_noise,
    get_schedule,
    prepare,
    prepare_redux,
    unpack,
)
import torch
from torch import Tensor
from torchvision import transforms
from PIL import Image
import numpy as np

from flux.util import (
    load_ae,
    load_clip,
    load_depth_encoder,
    load_flow_model,
    load_redux,
    load_t5,
)
from fp8.flux_pipeline import FluxPipeline
from fp8.lora_loading import load_lora, load_loras, unload_loras
from fp8.util import LoadedModels
from weights import WeightsDownloadCache

FLUX_DEV = "flux-dev"
FLUX_SCHNELL = "flux-schnell"


class LoraMixin:
    """
    Handles lora loading + extra lora for flux models using BFL code.
    Merges lora weights with base weights for inference; results in faster inference than unmerged loras.
    Set store_clones=True to persist copies of unmerged base weights. Consumes extra memory; store_clones=False may degrade performance over time.
    """

    def __init__(
        self,
        weights_cache: WeightsDownloadCache,
        scale_multiplier=1.0,
        store_clones=False,
    ):
        self.lora = None
        self.lora_scale = None
        # we apply a 1.5x multiplier to fp8 loras by default
        self.lora_scale_multiplier = scale_multiplier
        self.extra_lora = None
        self.extra_lora_scale = None
        self.weights_cache = weights_cache
        self.store_clones = store_clones

    def handle_loras(
        self,
        lora_weights: str | None = None,
        lora_scale: float = 1.0,
        extra_lora_weights: str | None = None,
        extra_lora_scale: float = 1.0,
    ):
        loading = "loading"
        if not lora_weights and extra_lora_weights:
            print(
                f"extra_lora_weights {extra_lora_weights} were found, and lora_weights were None! This shouldn't happen. Setting lora_weights to {extra_lora_weights} and lora_scale to extra_lora_scale: {extra_lora_scale} and running."
            )
            lora_weights = extra_lora_weights
            lora_scale = extra_lora_scale
            extra_lora_weights = None

        lora_scale = lora_scale * self.lora_scale_multiplier
        cur_lora = self.lora
        cur_extra_lora = self.extra_lora

        # don't assume loading succeeds
        self.lora = loading
        self.extra_lora = loading
        model = self.model

        if lora_weights:
            # since we merge weights, need to reload for change in scale. auto-reloading for extra weights
            if (
                lora_weights != cur_lora
                or lora_scale != self.lora_scale
                or extra_lora_weights != cur_extra_lora
                or extra_lora_scale != self.extra_lora_scale
            ):
                if self.lora or self.extra_lora:
                    unload_loras(model)
                lora_path = self.weights_cache.ensure(lora_weights)
                if extra_lora_weights:
                    extra_lora_path = self.weights_cache.ensure(extra_lora_weights)
                    load_loras(
                        model,
                        [lora_path, extra_lora_path],
                        [lora_scale, extra_lora_scale],
                        self.store_clones,
                    )
                else:
                    load_lora(model, lora_path, lora_scale, self.store_clones)
            else:
                print(f"Lora {lora_weights} already loaded")
                if extra_lora_weights:
                    print(f"Extra lora {extra_lora_weights} already loaded")

        elif self.lora:
            unload_loras(model)

        self.lora = lora_weights
        self.lora_scale = lora_scale
        self.extra_lora = extra_lora_weights
        self.extra_lora_scale = extra_lora_scale


class BflBf16Predictor(LoraMixin):
    """Base bf16 inference model. Supports loras w/LoraMixin"""

    def __init__(
        self,
        flow_model_name: str,
        loaded_models: LoadedModels | None = None,
        device: str = "cuda",
        offload: bool = False,
        weights_download_cache: WeightsDownloadCache | None = None,
        restore_lora_from_cloned_weights: bool = False,
    ):
        super().__init__(
            weights_cache=weights_download_cache,
            store_clones=restore_lora_from_cloned_weights,
        )
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")
        self.offload = offload

        max_length = 256 if self.flow_model_name == FLUX_SCHNELL else 512
        self.t5 = (
            loaded_models.t5
            if loaded_models and loaded_models.t5
            else load_t5(device, max_length=max_length)
        )
        self.clip = (
            loaded_models.clip
            if loaded_models and loaded_models.clip
            else load_clip(device)
        )
        self.ae = (
            loaded_models.ae
            if loaded_models and loaded_models.ae
            else load_ae(self.flow_model_name, device="cpu" if self.offload else device)
        )
        self.model = load_flow_model(
            self.flow_model_name, device="cpu" if self.offload else device
        )

        self.num_steps = 4 if self.flow_model_name == FLUX_SCHNELL else 28
        self.shift = self.flow_model_name != FLUX_SCHNELL
        self.compile_run = False

        self.vae_scale_factor = 8
        return

    def get_shared_models(self):
        return LoadedModels(
            flow=None, ae=self.ae, clip=self.clip, t5=self.t5, config=None
        )

    def prepare(self, x, prompt):
        return prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

    @contextmanager
    def maybe_offload_ae(self):
        if self.offload:
            self.ae.encoder.to(torch.device("cuda"))
        try:
            yield
        finally:
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

    def prepare_init_image(self, image_path: Path) -> tuple[torch.Tensor, int, int]:
        """prepares image for img2img inference using flux dev"""
        torch_device = torch.device("cuda")

        print("Image detected - setting to img2img mode")
        init_image = load_image_tensor(image_path).to(torch_device)
        init_image, width, height = maybe_scale_to_closest_multiple(
            init_image, multiple=16
        )

        with self.maybe_offload_ae():
            init_image = self.ae.encode(init_image)

        return init_image, width, height

    def prepare_legacy_mask(
        self,
        mask_path: Path,
        init_image: Tensor,
        noise: Tensor,
        width: int,
        height: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Preprocesses mask for inpainting using flux-dev, NOT using flux-fill"""
        image = Image.open(mask_path).convert("L")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        img: torch.Tensor = transform(image)
        img[img < 0.5] = 0
        img[img > 0.5] = 1
        mask = img[None, ...]

        mask_height = int(height) // self.vae_scale_factor
        mask_width = int(width) // self.vae_scale_factor
        mask = torch.nn.functional.interpolate(mask, size=(mask_height, mask_width))
        mask = mask.to(device=torch.device("cuda"), dtype=torch.bfloat16)

        def pack_img(img):
            return rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        mask = pack_img(mask.repeat(1, 16, 1, 1))
        noise = pack_img(noise)
        image_latents = pack_img(init_image.to(dtype=torch.bfloat16))

        return mask, noise, image_latents

    def prepare_conditioning(self):
        return None

    @torch.inference_mode()
    def predict(
        self,
        prompt: str,
        num_outputs: int,
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        prompt_strength: float = 0.8,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        legacy_image_path: Path | None = None,  # img2img for flux-dev
        legacy_mask_path: Path = None,  # inpainting for hotswap
        conditioning_kwargs: dict = {},
        prepare_kwargs: dict = {},
    ) -> tuple[List[Image.Image], List[np.ndarray]]:
        torch_device = torch.device("cuda")
        init_image = None
        img_cond = None

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if conditioning_kwargs:
            img_cond = self.prepare_conditioning(
                height=height, width=width, **conditioning_kwargs
            )

        # used for flux-dev & hotswap img2img
        if legacy_image_path is not None and img_cond is None:
            # For backwards compatibility, we still preserve width and
            # height for init_images, as opposed to using megapixels
            # with a "match_input" value.
            init_image, width, height = self.prepare_init_image(legacy_image_path)

        x = get_noise(
            num_outputs,
            height,
            width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        timesteps = get_schedule(
            num_inference_steps,
            (x.shape[-1] * x.shape[-2]) // 4,  # equivalent to inp["img"].shape[1] below
            shift=self.shift,
        )

        # used for flux-dev & hotswap img2img
        if init_image is not None:
            t_idx = int((1.0 - prompt_strength) * num_inference_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)

        inp = self.prepare(x, [prompt] * num_outputs, **prepare_kwargs)

        # fill/controlnets
        if img_cond is not None:
            inp["img_cond"] = img_cond

        # hotswap inpainting
        if legacy_mask_path:
            assert init_image is not None, "Init image is not set when mask is set"
            inp["mask"], inp["noise"], inp["image_latents"] = self.prepare_legacy_mask(
                mask_path=legacy_mask_path,
                init_image=init_image,
                noise=x,
                width=width,
                height=height,
            )

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(torch_device)

        x, flux = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            compile_run=self.compile_run,
        )

        if self.compile_run:
            self.compile_run = False
            self.model = flux

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        np_images = [
            (127.5 * (rearrange(x[i], "c h w -> h w c").clamp(-1, 1) + 1.0))
            .cpu()
            .byte()
            .numpy()
            for i in range(num_outputs)
        ]
        images = [Image.fromarray(img) for img in np_images]
        return images, np_images


class BflReduxPredictor(BflBf16Predictor):
    """
    Works for dev and schnell.
    To use, pass path to redux image into predict as a prepare_kwargs - e.g.:
        redux_predictor.predict(..., prepare_kwargs={"redux_img_path": redux_image}
    """

    def __init__(
        self,
        flow_model_name: str,
        loaded_models: LoadedModels | None = None,
        device: str = "cuda",
        offload: bool = False,
        weights_download_cache: WeightsDownloadCache | None = None,
    ):
        super().__init__(
            flow_model_name,
            loaded_models,
            device=device,
            offload=offload,
            weights_download_cache=weights_download_cache,
        )
        self.redux_image_encoder = load_redux(device="cuda")

    def prepare(self, x, prompt, redux_img_path=None):
        """Overrides prepare in order to properly preprocess redux image"""
        return prepare_redux(
            self.t5,
            self.clip,
            x,
            prompt=prompt,
            encoder=self.redux_image_encoder,
            img_cond_path=redux_img_path,
        )


class BflFillFlux(BflBf16Predictor):
    """
    Works for flux fill.
    To use, pass image and mask into predict as conditioning_kwargs - e.g.:
        fill_predictor.predict(..., conditioning_kwargs={"image_path": image, "mask_path": mask},)
    """

    def prepare_conditioning(
        self,
        image_path: Path,
        mask_path: Path,
        width: int,
        height: int,
    ) -> torch.Tensor:
        torch_device = torch.device("cuda")

        image_pil = load_image(image_path)
        image = maybe_scale_to_size_and_convert_to_tensor(image_pil, width, height).to(
            torch_device
        )
        mask_pil = load_image(mask_path, grayscale=True)
        mask = maybe_scale_to_size_and_convert_to_tensor(
            mask_pil, width, height, grayscale=True
        ).to(torch_device)

        # TODO(andreas): support image inputs with alpha channels
        with torch.no_grad():
            img_cond = image
            img_cond = img_cond * (1 - mask)

            with self.maybe_offload_ae():
                img_cond = self.ae.encode(img_cond)

            mask = mask[:, 0, :, :]
            mask = mask.to(torch.bfloat16)
            mask = rearrange(
                mask,
                "b (h ph) (w pw) -> b (ph pw) h w",
                ph=8,
                pw=8,
            )
            mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_cond = img_cond.to(torch.bfloat16)
        img_cond = rearrange(
            img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
        )
        return torch.cat((img_cond, mask), dim=-1)


class BflControlNetFlux(BflBf16Predictor):
    """
    Works for flux canny & depth.
    To use, pass image and mask into predict as conditioning_kwargs - e.g.:
        control_predictor.predict(...,conditioning_kwargs={"image_path": control_image},)
    """

    def __init__(
        self,
        flow_model_name: str,
        loaded_models: LoadedModels | None = None,
        device: str = "cuda",
        offload: bool = False,
        weights_download_cache: WeightsDownloadCache | None = None,
    ):
        super().__init__(
            flow_model_name,
            loaded_models,
            device=device,
            offload=offload,
            weights_download_cache=weights_download_cache,
        )
        # should be able to add new controlnets here as they come out as long as they pass conditioning images in the same way
        if self.flow_model_name == "flux-depth-dev":
            self.control_image_embedder = load_depth_encoder("cuda")
        elif self.flow_model_name == "flux-canny-dev":
            self.control_image_embedder = CannyImageEncoder(torch.device("cuda"))
        else:
            raise ValueError(f"flux model {flow_model_name} is not a controlnet model")

    def prepare_conditioning(
        self,
        image_path: Path,
        width: int,
        height: int,
    ) -> torch.Tensor:
        image_pil = load_image(image_path)
        image = maybe_scale_to_size_and_convert_to_tensor(image_pil, width, height)

        with torch.no_grad():
            img_cond = self.control_image_embedder(image)
            with self.maybe_offload_ae():
                img_cond = self.ae.encode(img_cond)

        img_cond = img_cond.to(torch.bfloat16)
        return rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


class BflFp8Flux(LoraMixin):
    """
    Fp8 support for dev and schnell.
    Supports loras and torch compilation.

    To download and use prequantized weights, pass in `flux-dev/schnell-fp8`; otherwise pass in `flux-dev/schnell`.
    Configs are in fp8/configs
    """

    def __init__(
        self,
        flow_model_name: str,
        loaded_models: LoadedModels | None,
        torch_compile: bool = False,
        compilation_aspect_ratios: dict[str, Tuple[int, int]] = None,
        offload: bool = False,
        weights_download_cache: WeightsDownloadCache | None = None,
        restore_lora_from_cloned_weights: bool = False,
    ):
        super().__init__(
            weights_cache=weights_download_cache,
            scale_multiplier=1.5,
            store_clones=restore_lora_from_cloned_weights,
        )
        self.offload = offload

        if torch_compile:
            extra_args = {
                "compile_whole_model": True,
                "compile_extras": True,
                "compile_blocks": True,
            }
        else:
            extra_args = {
                "compile_whole_model": False,
                "compile_extras": False,
                "compile_blocks": False,
            }

        if self.offload:
            extra_args |= {
                "offload_text_encoder": True,
                "offload_vae": True,
                "offload_flow": True,
            }
        self.fp8_pipe = FluxPipeline.load_pipeline_from_config_path(
            f"fp8/configs/config-1-{flow_model_name}-h100.json",
            shared_models=loaded_models,
            **extra_args,  # type: ignore
        )
        self.num_steps = 4 if "schnell" in flow_model_name else 28

        # hack to expose this for lora loading mixin
        self.model = self.fp8_pipe.model

        if torch_compile:
            print("compiling fp8 model")
            st = time.time()
            self.fp8_pipe.generate(
                prompt="a cool dog",
                width=1344,
                height=768,
                num_steps=self.num_steps,
                guidance=3,
                seed=123,
                compiling=True,
            )

            # need to pre-warm cudnn kernels or else we take a 2 sec latency hit
            for k, v in compilation_aspect_ratios.items():
                print(f"warming kernel for {k}")
                width, height = v
                self.fp8_pipe.generate(
                    prompt="godzilla!",
                    width=width,
                    height=height,
                    num_steps=4,
                    guidance=3,
                )
                self.fp8_pipe.generate(
                    prompt="godzilla!",
                    width=width // 2,
                    height=height // 2,
                    num_steps=4,
                    guidance=3,
                )

            print("compiled in ", time.time() - st)

    def predict(
        self,
        prompt: str,
        num_outputs: int,
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        image: Path | None = None,  # img2img for flux-dev
        prompt_strength: float = 0.8,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs,  # noqa: ARG002
    ) -> tuple[List[Image.Image], List[np.ndarray]]:
        """Run a single prediction on the model"""
        print("running quantized prediction")

        return self.fp8_pipe.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            init_image=image,
            strength=prompt_strength,
            num_images=num_outputs,
        )


###
# util functions
###


def load_image_tensor(image_path: Path) -> Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )
    img: Tensor = transform(image)  # type: ignore
    return img[None, ...]


def maybe_scale_to_closest_multiple(
    image: Tensor, multiple: int, max_image_size: int = 1440
) -> tuple[Tensor, int, int]:
    width = image.shape[-1]
    height = image.shape[-2]
    print(f"Input image size: {width}x{height}")

    # Calculate the scaling factor if the image exceeds max_image_size
    scale = min(max_image_size / width, max_image_size / height, 1)
    if scale < 1:
        width = int(width * scale)
        height = int(height * scale)
        print(f"Scaling image down to {width}x{height}")

    # Round image width and height to nearest multiple of 16
    width = round(width / multiple) * multiple
    height = round(height / multiple) * multiple
    print(f"Input image size set to: {width}x{height}")

    # Resize
    image = torch.nn.functional.interpolate(image, (height, width))

    return image, width, height


def load_image(image_path: Path, grayscale: bool = False) -> Image.Image:
    return Image.open(image_path).convert("L" if grayscale else "RGB")


def maybe_scale_to_size_and_convert_to_tensor(
    image: Image.Image, width: int, height: int, grayscale: bool = False
) -> Tensor:
    if grayscale:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    if image.size == (width, height):
        return transform(image)[None, ...]  # type: ignore

    # Resize with Lanczos
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    return transform(resized)[None, ...]  # type: ignore


def maybe_crop_to_size_and_convert_to_tensor(
    image: Image.Image, width: int, height: int, grayscale: bool = False
) -> Tensor:
    if grayscale:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )

    if image.size == (width, height):
        return transform(image)[None, ...]  # type: ignore

    # Calculate crop box for center crop
    img_width, img_height = image.size
    left = (img_width - width) // 2
    top = (img_height - height) // 2
    right = left + width
    bottom = top + height

    # Center crop
    cropped = image.crop((left, top, right, bottom))
    return transform(cropped)[None, ...]  # type: ignore
