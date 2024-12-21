from contextlib import contextmanager
import os
import time
from typing import Any, Tuple

import torch
from torch import Tensor

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20
import logging


from dataclasses import dataclass
from flux.sampling import (
    denoise,
    get_noise,
    get_schedule,
    prepare,
    prepare_redux,
    unpack,
)
from fp8.flux_pipeline import FluxPipeline
from fp8.util import LoadedModels
from fp8.lora_loading import load_lora, load_loras, unload_loras

import numpy as np
from einops import rearrange
from PIL import Image
from typing import List
from torchvision import transforms
from cog import BasePredictor, Input, Path  # type: ignore
from flux.util import (
    load_ae,
    load_clip,
    load_depth_encoder,
    load_flow_model,
    load_redux,
    load_t5,
    download_weights,
)
from flux.modules.image_embedders import (
    ImageEncoder,
    CannyImageEncoder,
)

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import (
    CLIPImageProcessor,
    AutoModelForImageClassification,
    ViTImageProcessor,
)
from weights import WeightsDownloadCache

SAFETY_CACHE = Path("./safety-cache")
FEATURE_EXTRACTOR = Path("./feature-extractor")
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
MAX_IMAGE_SIZE = 1440

FALCON_MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_CACHE = Path("./falcon-cache")
FALCON_MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)

FLUX_DEV = "flux-dev"
FLUX_SCHNELL = "flux-schnell"

# Suppress diffusers nsfw warnings
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


@dataclass(frozen=True)
class Inputs:
    prompt = Input(description="Prompt for generated image")
    aspect_ratio = Input(
        description="Aspect ratio for the generated image",
        choices=list(ASPECT_RATIOS.keys()),
        default="1:1",
    )
    num_outputs = Input(
        description="Number of outputs to generate", default=1, le=4, ge=1
    )
    seed = Input(
        description="Random seed. Set for reproducible generation", default=None
    )
    output_format = Input(
        description="Format of the output images",
        choices=["webp", "jpg", "png"],
        default="webp",
    )
    output_quality = Input(
        description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
        default=80,
        ge=0,
        le=100,
    )
    disable_safety_checker = Input(
        description="Disable safety checker for generated images.",
        default=False,
    )
    lora_weights: Input = Input(
        description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
        default=None,
    )
    lora_scale = Input(
        description="Determines how strongly the main LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
        default=1.0,
        le=3.0,
        ge=-1.0,
    )
    megapixels = Input(
        description="Approximate number of megapixels for generated image",
        choices=["1", "0.25"],
        default="1",
    )
    megapixels_with_match_input = Input(
        description="Approximate number of megapixels for generated image. Use match_input to match the size of the input (with an upper limit of 1440x1440 pixels)",
        choices=["1", "0.25", "match_input"],
        default="1",
    )

    @staticmethod
    def go_fast_with_default(default: bool) -> Input:
        return Input(
            description="Run faster predictions with model optimized for speed (currently fp8 quantized); disable to run in original bf16",
            default=default,
        )

    @staticmethod
    def guidance_with(default: float, le: float) -> Input:
        return Input(
            description="Guidance for generated image", ge=0, le=le, default=default
        )

    @staticmethod
    def num_inference_steps_with(
        default: int, le: int, recommended: int | tuple[int, int]
    ) -> Input:
        description = "Number of denoising steps. "
        if isinstance(recommended, tuple):
            description += f"Recommended range is {recommended[0]}-{recommended[1]}, and lower number of steps produce lower quality outputs, faster."
        else:
            description += f"{recommended} is recommended, and lower number of steps produce lower quality outputs, faster."

        return Input(
            description=description,
            ge=1,
            le=le,
            default=default,
        )


class Predictor(BasePredictor):
    def setup(self) -> None:
        return

    def lora_setup(self):
        self.weights_cache = WeightsDownloadCache()
        self.bf16_lora = None
        self.bf16_lora_scale = None
        self.bf16_extra_lora = None
        self.bf16_extra_lora_scale = None
        self.fp8_lora = None
        self.fp8_lora_scale = None
        self.fp8_lora_scale_multiplier = 1.5
        self.fp8_extra_lora = None
        self.fp8_extra_lora_scale = None

    def base_setup(
        self,
        flow_model_name: str,
        compile_fp8: bool = False,
        compile_bf16: bool = False,
        disable_fp8: bool = False,
        t5=None,
        clip=None,
        ae=None,
    ) -> None:
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")

        gpu_name = (
            os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
            .read()
            .strip()
        )
        print("Detected GPU:", gpu_name)

        if not SAFETY_CACHE.exists():
            download_weights(SAFETY_URL, SAFETY_CACHE)
        print("Loading Safety Checker to GPU")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")  # type: ignore
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Falcon safety checker...")
        if not FALCON_MODEL_CACHE.exists():
            download_weights(FALCON_MODEL_URL, FALCON_MODEL_CACHE)
        self.falcon_model = AutoModelForImageClassification.from_pretrained(
            FALCON_MODEL_NAME,
            cache_dir=FALCON_MODEL_CACHE,
        )
        self.falcon_processor = ViTImageProcessor.from_pretrained(FALCON_MODEL_NAME)

        # need > 48 GB of ram to store all models in VRAM
        total_mem = torch.cuda.get_device_properties(0).total_memory
        self.offload = total_mem < 48 * 1024**3
        if self.offload:
            print("GPU memory is:", total_mem / 1024**3, ", offloading models")
            compile_fp8 = False

        device = "cuda"
        max_length = 256 if self.flow_model_name == FLUX_SCHNELL else 512
        if t5:
            self.t5 = t5
        else:
            self.t5 = load_t5(device, max_length=max_length)
        if clip:
            self.clip = clip
        else:
            self.clip = load_clip(device)
        self.flux = load_flow_model(
            self.flow_model_name, device="cpu" if self.offload else device
        )
        self.flux = self.flux.eval()
        if ae:
            self.ae = ae
        else:
            self.ae = load_ae(
                self.flow_model_name, device="cpu" if self.offload else device
            )

        self.num_steps = 4 if self.flow_model_name == FLUX_SCHNELL else 28
        self.shift = self.flow_model_name != FLUX_SCHNELL
        self.compile_run = False

        shared_models = LoadedModels(
            flow=None, ae=self.ae, clip=self.clip, t5=self.t5, config=None
        )

        self.vae_scale_factor = 8
        self.disable_fp8 = disable_fp8

        if not self.disable_fp8:
            if compile_fp8:
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
                shared_models=shared_models,
                **extra_args,  # type: ignore
            )

            if compile_fp8:
                self.compile_fp8()

        if compile_bf16:
            self.compile_bf16()

    def compile_fp8(self):
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

        for k in ASPECT_RATIOS:
            print(f"warming kernel for {k}")
            width, height = self.aspect_ratio_to_width_height(k)
            self.fp8_pipe.generate(
                prompt="godzilla!", width=width, height=height, num_steps=4, guidance=3
            )
            self.fp8_pipe.generate(
                prompt="godzilla!",
                width=width // 2,
                height=height // 2,
                num_steps=4,
                guidance=3,
            )

        print("compiled in ", time.time() - st)

    def aspect_ratio_to_width_height(self, aspect_ratio: str) -> tuple[int, int]:
        return ASPECT_RATIOS[aspect_ratio]

    def prepare_legacy_mask(
        self,
        mask_path: Path,
        init_image: Tensor,
        noise: Tensor,
        width: int,
        height: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
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

    # TODO(andreas): make this an abstract class
    def predict(self):
        raise Exception("You need to instantiate a predictor for a specific flux model")

    def prepare(self, x, prompt):
        return prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

    @torch.inference_mode()
    def handle_loras(
        self,
        go_fast: bool,
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

        if go_fast:
            model = self.fp8_pipe.model
            cur_lora = self.fp8_lora
            lora_scale = lora_scale * self.fp8_lora_scale_multiplier
            cur_scale = self.fp8_lora_scale
            cur_extra_lora = self.fp8_extra_lora
            cur_extra_lora_scale = self.fp8_extra_lora_scale

            self.fp8_lora = loading
            self.fp8_extra_lora = loading

        else:
            model = self.flux
            cur_lora = self.bf16_lora
            cur_scale = self.bf16_lora_scale
            cur_extra_lora = self.bf16_extra_lora
            cur_extra_lora_scale = self.bf16_extra_lora_scale

            self.bf16_lora = loading
            self.bf16_extra_lora = loading

        if lora_weights:
            # since we merge weights, need to reload for change in scale. auto-reloading for extra weights
            if (
                lora_weights != cur_lora
                or lora_scale != cur_scale
                or extra_lora_weights != cur_extra_lora
                or extra_lora_scale != cur_extra_lora_scale
            ):
                if cur_lora or cur_extra_lora:
                    unload_loras(model)
                lora_path = self.weights_cache.ensure(lora_weights)
                if extra_lora_weights:
                    extra_lora_path = self.weights_cache.ensure(extra_lora_weights)
                    load_loras(
                        model,
                        [lora_path, extra_lora_path],
                        [lora_scale, extra_lora_scale],
                    )
                else:
                    load_lora(model, lora_path, lora_scale)
            else:
                print(f"Lora {lora_weights} already loaded")
                if extra_lora_weights:
                    print(f"Extra lora {extra_lora_weights} already loaded")

        elif cur_lora:
            unload_loras(model)

        if go_fast:
            self.fp8_lora = lora_weights
            self.fp8_lora_scale = lora_scale
            self.fp8_extra_lora = extra_lora_weights
            self.fp8_extra_lora_scale = extra_lora_scale

        else:
            self.bf16_lora = lora_weights
            self.bf16_lora_scale = lora_scale
            self.bf16_extra_lora = extra_lora_weights
            self.bf16_extra_lora_scale = extra_lora_scale

    def size_from_aspect_megapixels(
        self, aspect_ratio: str, megapixels: str = "1"
    ) -> Tuple[int, int]:
        width, height = ASPECT_RATIOS[aspect_ratio]
        if megapixels == "0.25":
            width, height = width // 2, height // 2

        return (width, height)

    # TODO(andreas): This is getting messy, with bf16_predict, shared_predict,
    # and lots of model-specific switching. Refactor.
    @torch.inference_mode()
    def bf16_predict(
        self,
        prompt: str,
        num_outputs: int,
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        image_path: Path | None = None,  # img2img for flux-dev
        mask_path: Path | None = None,  # mask for flux-dev-fill
        prompt_strength: float = 0.8,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        legacy_mask_path: Path = None,  # inpainting for hotswap
        control_image_embedder: ImageEncoder | None = None,
    ) -> tuple[List[Image.Image], List[np.ndarray]]:
        """Run a single prediction on the model"""
        torch_device = torch.device("cuda")
        init_image = None
        img_cond = None

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if mask_path:
            assert image_path is not None
            img_cond = self.prepare_img_cond(
                image_path, mask_path, width=width, height=height
            )

        elif control_image_embedder is not None:
            img_cond = self.prepare_control(
                image_path=image_path,
                image_embedder=control_image_embedder,
                width=width,
                height=height,
            )

        # img2img
        elif image_path is not None:
            # For backwards compatibility, we still preserve width and
            # height for init_images, as opposed to using megapixels
            # with a "match_input" value.
            init_image, width, height = self.prepare_init_image(image_path)

        # prepare input
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
            # equivalent to inp["img"].shape[1], needs to be here for prompt strength in img2img
            (x.shape[-1] * x.shape[-2]) // 4,
            shift=self.shift,
        )

        if init_image is not None:
            t_idx = int((1.0 - prompt_strength) * num_inference_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)

        inp = self.prepare(x, [prompt] * num_outputs)

        if img_cond is not None:
            inp["img_cond"] = img_cond

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
            self.flux = self.flux.to(torch_device)

        x, flux = denoise(
            self.flux,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            compile_run=self.compile_run,
        )

        if self.compile_run:
            self.compile_run = False
            self.flux = flux

        if self.offload:
            self.flux.cpu()
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

    def fp8_predict(
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

    def postprocess(
        self,
        images: List[Image.Image],
        disable_safety_checker: bool,
        output_format: str,
        output_quality: int,
        np_images: List[np.ndarray],
    ) -> List[Path]:
        has_nsfw_content = [False] * len(images)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(images, np_images)

        output_paths = []
        for i, (img, is_nsfw) in enumerate(zip(images, has_nsfw_content)):
            if is_nsfw:
                try:
                    falcon_is_safe = self.run_falcon_safety_checker(img)
                except Exception as e:
                    print(f"Error running safety checker: {e}")
                    falcon_is_safe = False
                if not falcon_is_safe:
                    print(f"NSFW content detected in image {i}")
                    continue

            output_path = f"out-{i}.{output_format}"
            save_params = (
                {"quality": output_quality, "optimize": True}
                if output_format != "png"
                else {}
            )
            img.save(output_path, **save_params)
            output_paths.append(Path(output_path))

        if not output_paths:
            raise Exception(
                "All generated images contained NSFW content. Try running it again with a different prompt."
            )

        print(f"Total safe images: {len(output_paths)} out of {len(images)}")
        return output_paths

    def run_safety_checker(self, images, np_images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(  # type: ignore
            "cuda"
        )
        image, has_nsfw_concept = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def run_falcon_safety_checker(self, image):
        with torch.no_grad():
            inputs = self.falcon_processor(images=image, return_tensors="pt")  # type: ignore
            outputs = self.falcon_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]

        return result == "normal"

    def shared_predict(
        self,
        go_fast: bool,
        prompt: str,
        num_outputs: int,
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        image: Path | None = None,  # img2img for flux-dev
        mask: Path | None = None,  # for flux-dev-fill
        prompt_strength: float = 0.8,
        seed: int | None = None,
        width: int = 1024,
        height: int = 1024,
        control_image_embedder: ImageEncoder | None = None,
        legacy_mask_path: Path | None = None,
    ) -> tuple[List[Image.Image], List[np.ndarray]]:
        if go_fast and not self.disable_fp8:
            assert image is None
            assert mask is None

            return self.fp8_predict(
                prompt=prompt,
                num_outputs=num_outputs,
                num_inference_steps=num_inference_steps,
                guidance=guidance,
                image=image,
                prompt_strength=prompt_strength,
                seed=seed,
                width=width,
                height=height,
            )
        if self.disable_fp8:
            print("running bf16 model, fp8 disabled")
        return self.bf16_predict(
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            image_path=image,
            mask_path=mask,
            prompt_strength=prompt_strength,
            seed=seed,
            width=width,
            height=height,
            legacy_mask_path=legacy_mask_path,
            control_image_embedder=control_image_embedder,
        )

    def prepare_init_image(self, image_path: Path) -> tuple[torch.Tensor, int, int]:
        torch_device = torch.device("cuda")

        print("Image detected - setting to img2img mode")
        init_image = load_image_tensor(image_path).to(torch_device)
        init_image, width, height = maybe_scale_to_closest_multiple(
            init_image, multiple=16
        )

        with self.maybe_offload_ae():
            init_image = self.ae.encode(init_image)

        return init_image, width, height

    def prepare_control(
        self,
        image_path: Path,
        image_embedder: ImageEncoder,
        width: int,
        height: int,
    ) -> torch.Tensor:
        image_pil = load_image(image_path)
        image = maybe_scale_to_size_and_convert_to_tensor(image_pil, width, height)

        with torch.no_grad():
            img_cond = image_embedder(image)
            with self.maybe_offload_ae():
                img_cond = self.ae.encode(img_cond)

        img_cond = img_cond.to(torch.bfloat16)
        return rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    def prepare_img_cond(
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

    def size_maybe_match_input(
        self, image_path: Path, megapixels: str
    ) -> tuple[int, int]:
        image = Image.open(image_path)
        width, height = image.size

        # 32 since that's what fill is using
        def round_to_nearest_multiple_of_32(width: int, height: int) -> tuple[int, int]:
            return int(width / 32) * 32, int(height / 32) * 32

        if megapixels == "match_input":
            # scale down if needed to fit within MAX_IMAGE_SIZE
            scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
            if scale < 1:
                width = int(width * scale)
                height = int(height * scale)
            return round_to_nearest_multiple_of_32(width, height)

        target_pixels = int(float(megapixels) * 1024 * 1024)
        current_pixels = width * height
        scale = (target_pixels / current_pixels) ** 0.5

        width = int(width * scale)
        height = int(height * scale)

        return round_to_nearest_multiple_of_32(width, height)

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


class SchnellPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup(FLUX_SCHNELL, compile_fp8=True)

    def predict(
        self,
        prompt: str = Inputs.prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=4, default=4, recommended=4
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = Inputs.go_fast_with_default(True),
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class DevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup(FLUX_DEV, compile_fp8=True)

    def predict(
        self,
        prompt: str = Inputs.prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        image: Path = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.80,
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=50, default=28, recommended=(28, 50)
        ),
        guidance: float = Inputs.guidance_with(default=3, le=10),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = Inputs.go_fast_with_default(True),
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        if image and go_fast:
            print("img2img not supported with fp8 quantization; running with bf16")
            go_fast = False
        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance,
            image=image,
            prompt_strength=prompt_strength,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class SchnellLoraPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup(FLUX_SCHNELL, compile_fp8=True)
        self.lora_setup()

    def predict(
        self,
        prompt: str = Inputs.prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=4, default=4, recommended=4
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = Inputs.go_fast_with_default(True),
        lora_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        self.handle_loras(go_fast, lora_weights, lora_scale)

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class DevLoraPredictor(Predictor):
    def setup(self, t5=None, clip=None, ae=None) -> None:
        self.base_setup(FLUX_DEV, compile_fp8=True, t5=t5, clip=clip, ae=ae)
        self.lora_setup()

    def predict(
        self,
        prompt: str = Inputs.prompt,
        aspect_ratio: str = Inputs.aspect_ratio,
        image: Path = Input(
            description="Input image for image to image mode. The aspect ratio of your output will match this image",
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.80,
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=50, default=28, recommended=(28, 50)
        ),
        guidance: float = Inputs.guidance_with(default=3, le=10),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = Inputs.go_fast_with_default(True),
        lora_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        if image and go_fast:
            print("img2img not supported with fp8 quantization; running with bf16")
            go_fast = False

        self.handle_loras(go_fast, lora_weights, lora_scale)

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance,
            image=image,
            prompt_strength=prompt_strength,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class _ReduxPredictor(Predictor):
    def redux_setup(self):
        self.redux_image_encoder = load_redux(device="cuda")
        self.cur_prediction_redux_img = None
        # TODO: hopefully temporary
        self.disable_fp8 = True

    def prepare(self, x, prompt):
        return prepare_redux(
            self.t5,
            self.clip,
            x,
            prompt=prompt,
            encoder=self.redux_image_encoder,
            # TODO(andreas): Refactor this so we don't have to ignore the type here
            img_cond_path=self.cur_prediction_redux_img,  # type: ignore
        )


class SchnellReduxPredictor(_ReduxPredictor):
    def setup(self):
        self.base_setup("flux-schnell", compile_fp8=False)
        self.redux_setup()

    def predict(
        self,
        redux_image: Path = Input(
            description="Input image to condition your output on. This replaces prompt for FLUX.1 Redux models",
        ),
        aspect_ratio: str = Inputs.aspect_ratio,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Input(
            description="Number of denoising steps. 4 is recommended, and lower number of steps produce lower quality outputs, faster.",
            ge=1,
            le=4,
            default=4,
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        go_fast = False
        prompt = ""

        # TODO: don't love passing this via a class variable, but it's better than totally breaking our abstractions.
        self.cur_prediction_redux_img = redux_image

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class DevReduxPredictor(_ReduxPredictor):
    def setup(self):
        self.base_setup("flux-dev", compile_fp8=False)
        self.redux_setup()

    def predict(
        self,
        redux_image: Path = Input(
            description="Input image to condition your output on. This replaces prompt for FLUX.1 Redux models",
        ),
        aspect_ratio: str = Inputs.aspect_ratio,
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended range is 28-50",
            ge=1,
            le=50,
            default=28,
        ),
        guidance: float = Input(
            description="Guidance for generated image", ge=0, le=10, default=3
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        megapixels: str = Inputs.megapixels,
    ) -> List[Path]:
        go_fast = False
        prompt = ""

        # TODO: don't love passing this via a class variable, but it's better than totally breaking our abstractions.
        self.cur_prediction_redux_img = redux_image

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class FillDevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-fill-dev", disable_fp8=True)

    def predict(
        self,
        prompt: str = Inputs.prompt,
        image: Path = Input(
            description=f"The image to inpaint. Can contain alpha mask. If the image width or height are not multiples of 32, they will be scaled to the closest multiple of 32. If the image dimensions don't fit within {MAX_IMAGE_SIZE}x{MAX_IMAGE_SIZE}, it will be scaled down to fit."
        ),
        mask: Path = Input(
            description="A black-and-white image that describes the part of the image to inpaint. Black areas will be preserved while white areas will be inpainted.",
            default=None,
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=50, default=28, recommended=(28, 50)
        ),
        guidance: float = Inputs.guidance_with(default=30, le=100),
        seed: int = Inputs.seed,
        megapixels: str = Inputs.megapixels_with_match_input,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        # go_fast: bool = Inputs.go_fast,
    ) -> List[Path]:
        # TODO(andreas): This means we're reading the image twice
        # which is a bit inefficient.
        width, height = self.size_maybe_match_input(image, megapixels)

        imgs, np_imgs = self.shared_predict(
            go_fast=False,
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            image=image,
            mask=mask,
            seed=seed,
            width=width,
            height=height,
        )
        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class HotswapPredictor(BasePredictor):
    def setup(self) -> None:
        self.schnell_lora = SchnellLoraPredictor()
        self.schnell_lora.setup()

        self.dev_lora = DevLoraPredictor()
        self.dev_lora.setup(
            t5=self.schnell_lora.t5,
            clip=self.schnell_lora.clip,
            ae=self.schnell_lora.ae,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for generated image. If you include the `trigger_word` used in the training process you are more likely to activate the trained object, style, or concept in the resulting image."
        ),
        image: Path = Input(
            description="Input image for image to image or inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
            default=None,
        ),
        mask: Path = Input(
            description="Image mask for image inpainting mode. If provided, aspect_ratio, width, and height inputs are ignored.",
            default=None,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image. If custom is selected, uses height and width below & will run in bf16 mode",
            choices=list(ASPECT_RATIOS.keys()) + ["custom"],
            default="1:1",
        ),
        height: int = Input(
            description="Height of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
            ge=256,
            le=1440,
            default=None,
        ),
        width: int = Input(
            description="Width of generated image. Only works if `aspect_ratio` is set to custom. Will be rounded to nearest multiple of 16. Incompatible with fast generation",
            ge=256,
            le=1440,
            default=None,
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.80,
        ),
        model: str = Input(
            description="Which model to run inference with. The dev model performs best with around 28 inference steps but the schnell model only needs 4 steps.",
            choices=["dev", "schnell"],
            default="dev",
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Input(
            description="Number of denoising steps. More steps can give more detailed images, but take longer.",
            ge=1,
            le=50,
            default=28,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for the diffusion process. Lower values can give more realistic images. Good values to try are 2, 2.5, 3 and 3.5",
            ge=0,
            le=10,
            default=3,
        ),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        go_fast: bool = Inputs.go_fast_with_default(False),
        megapixels: str = Inputs.megapixels,
        replicate_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
        extra_lora: str = Input(
            description="Load LoRA weights. Supports Replicate models in the format <owner>/<username> or <owner>/<username>/<version>, HuggingFace URLs in the format huggingface.co/<owner>/<model-name>, CivitAI URLs in the format civitai.com/models/<id>[/<model-name>], or arbitrary .safetensors URLs from the Internet. For example, 'fofr/flux-pixar-cars'",
            default=None,
        ),
        extra_lora_scale: float = Input(
            description="Determines how strongly the extra LoRA should be applied. Sane results between 0 and 1 for base inference. For go_fast we apply a 1.5x multiplier to this value; we've generally seen good performance when scaling the base value by that amount. You may still need to experiment to find the best value for your particular lora.",
            default=1.0,
            le=3.0,
            ge=-1,
        ),
    ) -> List[Path]:
        # so you're basically gonna just call the model.
        model = self.dev_lora if model == "dev" else self.schnell_lora

        if aspect_ratio == "custom":
            if go_fast:
                print(
                    "Custom aspect ratios not supported with fast fp8 inference; will run in bf16"
                )
                go_fast = False
        else:
            width, height = model.size_from_aspect_megapixels(
                aspect_ratio, megapixels=megapixels
            )

        model.handle_loras(
            go_fast, replicate_weights, lora_scale, extra_lora, extra_lora_scale
        )

        if image and go_fast:
            print(
                "Img2img and inpainting not supported with fast fp8 inference; will run in bf16"
            )
            go_fast = False

        imgs, np_imgs = model.shared_predict(
            go_fast,
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance_scale,
            image=image,
            prompt_strength=prompt_strength,
            seed=seed,
            width=width,
            height=height,
            legacy_mask_path=mask,
        )

        return model.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class CannyDevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-canny-dev", disable_fp8=True)
        self.control_image_embedder = CannyImageEncoder(torch.device("cuda"))

    def predict(
        self,
        prompt: str = Inputs.prompt,
        control_image: Path = Input(
            description="Image used to control the generation. The canny edge detection will be automatically generated."
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=50, default=28, recommended=(28, 50)
        ),
        guidance: float = Inputs.guidance_with(default=30, le=100),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        megapixels: str = Inputs.megapixels_with_match_input,
    ) -> List[Path]:
        # TODO(andreas): This means we're reading the image twice
        # which is a bit inefficient.
        width, height = self.size_maybe_match_input(control_image, megapixels)

        imgs, np_imgs = self.shared_predict(
            go_fast=False,
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            image=control_image,
            control_image_embedder=self.control_image_embedder,
        )
        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class DepthDevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-depth-dev", disable_fp8=True)
        self.control_image_embedder = load_depth_encoder("cuda")

    def predict(
        self,
        prompt: str = Inputs.prompt,
        control_image: Path = Input(
            description="Image used to control the generation. The depth map will be automatically generated."
        ),
        num_outputs: int = Inputs.num_outputs,
        num_inference_steps: int = Inputs.num_inference_steps_with(
            le=50, default=28, recommended=(28, 50)
        ),
        guidance: float = Inputs.guidance_with(default=10, le=100),
        seed: int = Inputs.seed,
        output_format: str = Inputs.output_format,
        output_quality: int = Inputs.output_quality,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
        megapixels: str = Inputs.megapixels_with_match_input,
    ) -> List[Path]:
        # TODO(andreas): This means we're reading the image twice
        # which is a bit inefficient.
        width, height = self.size_maybe_match_input(control_image, megapixels)

        imgs, np_imgs = self.shared_predict(
            go_fast=False,
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            image=control_image,
            control_image_embedder=self.control_image_embedder,
        )
        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class TestPredictor(Predictor):
    def setup(self) -> None:
        self.num = 3

    def predict(self, how_many: int = Input(description="how many", ge=0)) -> Any:
        return self.num + how_many


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
    image: Tensor, multiple: int
) -> tuple[Tensor, int, int]:
    width = image.shape[-1]
    height = image.shape[-2]
    print(f"Input image size: {width}x{height}")

    # Calculate the scaling factor if the image exceeds MAX_IMAGE_SIZE
    scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
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
