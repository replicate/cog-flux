import io
import math
import os
import random
import shutil
import warnings
from typing import TYPE_CHECKING, Callable, List

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import torch
from einops import rearrange, repeat

from pybase64 import standard_b64decode
from torch._dynamo import config
from torch._inductor import config as ind_config

config.cache_size_limit = 10000000000
ind_config.shape_padding = True
import platform

#from loguru import logger
from torchvision.transforms import functional as TF
from tqdm import tqdm

import fp8.lora_loading as lora_loading
from fp8.image_encoder import ImageEncoder
from fp8.util import (
    LoadedModels,
    ModelSpec,
    ModelVersion,
    into_device,
    into_dtype,
    load_config_from_path,
    load_models_from_config,
)

if platform.system() == "Windows":
    MAX_RAND = 2**16 - 1
else:
    MAX_RAND = 2**32 - 1


if TYPE_CHECKING:
    from fp8.modules.autoencoder import AutoEncoder
    from fp8.modules.conditioner import HFEmbedder
    from fp8.modules.flux_model import Flux


class FluxPipeline:
    """
    FluxPipeline is a class that provides a pipeline for generating images using the Flux model.
    It handles input preparation, timestep generation, noise generation, device management
    and model compilation.
    """

    def __init__(
        self,
        name: str,
        offload: bool = False,
        clip: "HFEmbedder" = None,
        t5: "HFEmbedder" = None,
        model: "Flux" = None,
        ae: "AutoEncoder" = None,
        dtype: torch.dtype = torch.float16,
        verbose: bool = False,
        flux_device: torch.device | str = "cuda:0",
        ae_device: torch.device | str = "cuda:1",
        clip_device: torch.device | str = "cuda:1",
        t5_device: torch.device | str = "cuda:1",
        config: ModelSpec = None,
        debug: bool = False,
    ):
        """
        Initialize the FluxPipeline class.

        This class is responsible for preparing input tensors for the Flux model, generating
        timesteps and noise, and handling device management for model offloading.
        """

        if config is None:
            raise ValueError("ModelSpec config is required!")

        self.debug = debug
        self.name = name
        self.device_flux = into_device(flux_device)
        self.device_ae = into_device(ae_device)
        self.device_clip = into_device(clip_device)
        self.device_t5 = into_device(t5_device)
        self.dtype = into_dtype(dtype)
        self.offload = offload
        self.clip: "HFEmbedder" = clip
        self.t5: "HFEmbedder" = t5
        self.model: "Flux" = model
        self.ae: "AutoEncoder" = ae
        self.rng = torch.Generator(device="cpu")
        self.img_encoder = ImageEncoder()
        self.verbose = verbose
        self.ae_dtype = torch.bfloat16
        self.config = config
        self.offload_text_encoder = config.offload_text_encoder
        self.offload_vae = config.offload_vae
        self.offload_flow = config.offload_flow
        # If models are not offloaded, move them to the appropriate devices

        if not self.offload_flow:
            self.model.to(self.device_flux)
        if not self.offload_vae:
            self.ae.to(self.device_ae)
        if not self.offload_text_encoder:
            self.clip.to(self.device_clip)
            self.t5.to(self.device_t5)

        # compile the model if needed
        if config.compile_blocks or config.compile_extras or config.compile_whole_model:
            self.compile()

    def set_seed(
        self, seed: int | None = None, seed_globally: bool = False
    ) -> torch.Generator:
        if isinstance(seed, (int, float)):
            seed = int(abs(seed)) % MAX_RAND
            cuda_generator = torch.Generator("cuda").manual_seed(seed)
        elif isinstance(seed, str):
            try:
                seed = abs(int(seed)) % MAX_RAND
            except Exception as e:
                print(
                    f"Recieved string representation of seed, but was not able to convert to int: {seed}, using random seed"
                )
                seed = abs(self.rng.seed()) % MAX_RAND
            cuda_generator = torch.Generator("cuda").manual_seed(seed)
        else:
            seed = abs(self.rng.seed()) % MAX_RAND
            cuda_generator = torch.Generator("cuda").manual_seed(seed)

        if seed_globally:
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
        return cuda_generator, seed

    def load_lora(self, lora_path: str, scale: float):
        """
        Loads a LoRA checkpoint into the Flux flow transformer.

        Currently supports LoRA checkpoints from either diffusers checkpoints which usually start with transformer.[...],
        or loras which contain keys which start with lora_unet_[...].

        Args:
            lora_path (str): Path to the LoRA checkpoint.
            scale (float): Scaling factor for the LoRA weights.

        """
        self.model = lora_loading.apply_lora_to_model_and_optionally_store_clones(self.model, lora_path, scale)

    @torch.inference_mode()
    def compile(self):
        """
        Compiles the model and extras.

        First, if:

        - A) Checkpoint which already has float8 quantized weights and tuned input scales.
        In which case, it will not run warmups since it assumes the input scales are already tuned.

        - B) Checkpoint which has not been quantized, in which  case it will be quantized
        and the input scales will be tuned. via running a warmup loop.
            - If the model is flux-schnell, it will run 3 warmup loops since each loop is 4 steps.
            - If the model is flux-dev, it will run 1 warmup loop for 12 steps.

        """

        # Run warmups if the checkpoint is not prequantized
        if not self.config.prequantized_flow:
            print("Running warmups for compile...")
            warmup_dict = dict(
                prompt="A beautiful test image used to solidify the fp8 nn.Linear input scales prior to compilation 😉",
                height=768,
                width=768,
                num_steps=12,
                guidance=3.5,
                seed=10,
            )
            if self.config.version == ModelVersion.flux_schnell:
                warmup_dict["num_steps"] = 4
                for _ in range(3):
                    self.generate(**warmup_dict)
            else:
                self.generate(**warmup_dict)

        # Compile the model and extras
        to_gpu_extras = [
            "vector_in",
            "img_in",
            "txt_in",
            "time_in",
            "guidance_in",
            "final_layer",
            "pe_embedder",
        ]

        if self.config.compile_whole_model:
            # we're not doing the compilation here in this case
            return

        if self.config.compile_blocks:
            for block in self.model.double_blocks:
                block.compile()
            for block in self.model.single_blocks:
                block.compile()
        if self.config.compile_extras:
            for extra in to_gpu_extras:
                getattr(self.model, extra).compile()

    @torch.inference_mode()
    def prepare(
        self,
        img: torch.Tensor,
        prompt: str | list[str],
        target_device: torch.device = torch.device("cuda:0"),
        target_dtype: torch.dtype = torch.float16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for the Flux model.

        This function processes the input image and text prompt, converting them into
        the appropriate format and embedding representations required by the model.

        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
            prompt (str | list[str]): Text prompt or list of prompts guiding the image generation.
            target_device (torch.device, optional): The target device for the output tensors.
                Defaults to torch.device("cuda:0").
            target_dtype (torch.dtype, optional): The target data type for the output tensors.
                Defaults to torch.float16.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - img: Processed image tensor.
                - img_ids: Image position IDs.
                - vec: Clip text embedding vector.
                - txt: T5 text embedding hidden states.
                - txt_ids: Text position IDs.

        Note:
            This function handles the necessary device management for text encoder offloading
            if enabled in the configuration.
        """
        bs, c, h, w = img.shape
        if bs == 1 and not isinstance(prompt, str):
            bs = len(prompt)
        img = img.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 1, 4, 5)
        img = img.reshape(img.shape[0], -1, img.shape[3] * img.shape[4] * img.shape[5])
        assert img.shape == (
            bs,
            (h // 2) * (w // 2),
            c * 2 * 2,
        ), f"{img.shape} != {(bs, (h//2)*(w//2), c*2*2)}"
        if img.shape[0] == 1 and bs > 1:
            img = img[None].repeat_interleave(bs, dim=0)

        img_ids = torch.zeros(
            h // 2, w // 2, 3, device=target_device, dtype=target_dtype
        )
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.arange(h // 2, device=target_device, dtype=target_dtype)[:, None]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.arange(w // 2, device=target_device, dtype=target_dtype)[None, :]
        )

        img_ids = img_ids[None].repeat(bs, 1, 1, 1).flatten(1, 2)
        if self.offload_text_encoder:
            self.clip.to(device=self.device_clip)
            self.t5.to(device=self.device_t5)

        # get the text embeddings
        if isinstance(prompt, str):
            prompt = [prompt]
        txt = self.t5(prompt)
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3, dtype=target_dtype, device=target_device)

        vec = self.clip(prompt)
        if vec.shape[0] == 1 and bs > 1:
            vec = repeat(vec, "1 ... -> bs ...", bs=bs)

        # offload text encoder to cpu if needed
        if self.offload_text_encoder:
            self.clip.to("cpu")
            self.t5.to("cpu")
            torch.cuda.empty_cache()
        return img, img_ids, vec, txt, txt_ids

    @torch.inference_mode()
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @torch.inference_mode()
    def get_schedule(
        self,
        num_steps: int,
        image_seq_len: int,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        shift: bool = True,
    ) -> list[float]:
        """Generates a schedule of timesteps for the given number of steps and image sequence length."""
        # extra step for zero
        timesteps = torch.linspace(1, 0, num_steps + 1)

        # shifting the schedule to favor high timesteps for higher signal images
        if shift:
            # eastimate mu based on linear estimation between two points
            mu = self.get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
            timesteps = self.time_shift(mu, 1.0, timesteps)

        return timesteps.tolist()

    @torch.inference_mode()
    def get_noise(
        self,
        num_samples: int,
        height: int,
        width: int,
        generator: torch.Generator,
        dtype=None,
        device=None,
    ) -> torch.Tensor:
        """Generates a latent noise tensor of the given shape and dtype on the given device."""
        if device is None:
            device = self.device_flux
        if dtype is None:
            dtype = self.dtype
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=False,
        )

    @torch.inference_mode()
    def into_bytes(self, x: torch.Tensor, jpeg_quality: int = 99) -> io.BytesIO:
        """Converts the image tensor to bytes."""
        # bring into PIL format and save
        num_images = x.shape[0]
        images: List[torch.Tensor] = []
        for i in range(num_images):
            x = (
                x[i]
                .clamp(-1, 1)
                .add(1.0)
                .mul(127.5)
                .clamp(0, 255)
                .contiguous()
                .type(torch.uint8)
            )
            images.append(x)
        if len(images) == 1:
            im = images[0]
        else:
            im = torch.vstack(images)

        im = self.img_encoder.encode_torch(im, quality=jpeg_quality)
        images.clear()
        return im

    @torch.inference_mode()
    def as_img_tensor(self, x: torch.Tensor) -> tuple[List[Image.Image], List[np.ndarray]]:
        """Converts the image tensor to PIL images and np.ndarrays."""
        # bring into PIL format and save
        num_images = x.shape[0]
        np_images = []
        pil_images = []
        for i in range(num_images):
            im = (
                rearrange(x[i], "c h w -> h w c")
                .clamp(-1, 1)
                .add(1.0)
                .mul(127.5)
                .clamp(0, 255)
                .contiguous()
                .type(torch.uint8)
                .cpu()
                .numpy()
            )
            np_images.append(im)
            pil_images.append(Image.fromarray(im))

        return pil_images, np_images

    @torch.inference_mode()
    def load_init_image_if_needed(
        self, init_image: torch.Tensor | str | Image.Image | np.ndarray
    ) -> torch.Tensor:
        """
        Loads the initial image if it is a string, numpy array, or PIL.Image,
        if torch.Tensor, expects it to be in the correct format and returns it as is.
        """
        if isinstance(init_image, str):
            try:
                init_image = Image.open(init_image)
            except Exception as e:
                init_image = Image.open(
                    io.BytesIO(standard_b64decode(init_image.split(",")[-1]))
                )
            init_image = torch.from_numpy(np.array(init_image)).type(torch.uint8)
        elif isinstance(init_image, np.ndarray):
            init_image = torch.from_numpy(init_image).type(torch.uint8)
        elif isinstance(init_image, Image.Image):
            init_image = torch.from_numpy(np.array(init_image)).type(torch.uint8)

        return init_image

    @torch.inference_mode()
    def vae_decode(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Decodes the latent tensor to the pixel space."""
        if self.offload_vae:
            self.ae.to(self.device_ae)
            x = x.to(self.device_ae)
        else:
            x = x.to(self.device_ae)
        x = self.unpack(x.float(), height, width)
        with torch.autocast(
            device_type=self.device_ae.type, dtype=torch.bfloat16, cache_enabled=False
        ):
            x = self.ae.decode(x)
        if self.offload_vae:
            self.ae.to("cpu")
            torch.cuda.empty_cache()
        return x

    def unpack(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @torch.inference_mode()
    def resize_center_crop(
        self, img: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Resizes and crops the image to the given height and width."""
        img = TF.resize(img, min(width, height))
        img = TF.center_crop(img, (height, width))
        return img

    @torch.inference_mode()
    def preprocess_latent(
        self,
        init_image: torch.Tensor | np.ndarray = None,
        height: int = 720,
        width: int = 1024,
        num_steps: int = 20,
        strength: float = 1.0,
        generator: torch.Generator = None,
        num_images: int = 1,
    ) -> tuple[torch.Tensor, List[float]]:
        """
        Preprocesses the latent tensor for the given number of steps and image sequence length.
        Also, if an initial image is provided, it is vae encoded and injected with the appropriate noise
        given the strength and number of steps replacing the latent tensor.
        """
        # prepare input

        if init_image is not None:
            if isinstance(init_image, np.ndarray):
                init_image = torch.from_numpy(init_image)

            init_image = (
                init_image.permute(2, 0, 1)
                .contiguous()
                .to(self.device_ae, dtype=self.ae_dtype)
                .div(127.5)
                .sub(1)[None, ...]
            )
            init_image = self.resize_center_crop(init_image, height, width)
            with torch.autocast(
                device_type=self.device_ae.type,
                dtype=torch.bfloat16,
                cache_enabled=False,
            ):
                if self.offload_vae:
                    self.ae.to(self.device_ae)
                init_image = (
                    self.ae.encode(init_image)
                    .to(dtype=self.dtype, device=self.device_flux)
                    .repeat(num_images, 1, 1, 1)
                )
                if self.offload_vae:
                    self.ae.to("cpu")
                    torch.cuda.empty_cache()

        x = self.get_noise(
            num_images,
            height,
            width,
            device=self.device_flux,
            dtype=self.dtype,
            generator=generator,
        )
        timesteps = self.get_schedule(
            num_steps=num_steps,
            image_seq_len=x.shape[-1] * x.shape[-2] // 4,
            shift=(self.name != "flux-schnell"),
        )
        if init_image is not None:
            t_idx = int((1 - strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image
        return x, timesteps

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        width: int = 720,
        height: int = 1024,
        num_steps: int = 24,
        guidance: float = 3.5,
        seed: int | None = None,
        init_image: torch.Tensor | str | Image.Image | np.ndarray | None = None,
        strength: float = 1.0,
        silent: bool = False,
        num_images: int = 1,
        jpeg_quality: int = 99,
        compiling: bool = False,
    ) -> tuple[List[Image.Image], List[np.ndarray]]:
        """
        Generate images based on the given prompt and parameters.

        Args:
            prompt `(str)`: The text prompt to guide the image generation.

            width `(int, optional)`: Width of the generated image. Defaults to 720.

            height `(int, optional)`: Height of the generated image. Defaults to 1024.

            num_steps `(int, optional)`: Number of denoising steps. Defaults to 24.

            guidance `(float, optional)`: Guidance scale for text-to-image generation. Defaults to 3.5.

            seed `(int | None, optional)`: Random seed for reproducibility. If None, a random seed is used. Defaults to None.

            init_image `(torch.Tensor | str | Image.Image | np.ndarray | None, optional)`: Initial image for image-to-image generation. Defaults to None.

                -- note: if the image's height/width do not match the height/width of the generated image, the image is resized and centered cropped to match the height/width arguments.

                -- If a string is provided, it is assumed to be either a path to an image file or a base64 encoded image.

                -- If a numpy array is provided, it is assumed to be an RGB numpy array of shape (height, width, 3) and dtype uint8.

                -- If a PIL.Image is provided, it is assumed to be an RGB PIL.Image.

                -- If a torch.Tensor is provided, it is assumed to be a torch.Tensor of shape (height, width, 3) and dtype uint8 with range [0, 255].

            strength `(float, optional)`: Strength of the init_image in image-to-image generation. Defaults to 1.0.

            silent `(bool, optional)`: If True, suppresses progress bar. Defaults to False.

            num_images `(int, optional)`: Number of images to generate. Defaults to 1.

            return_seed `(bool, optional)`: If True, returns the seed along with the generated image. Defaults to False.

            jpeg_quality `(int, optional)`: Quality of the JPEG compression. Defaults to 99.

        Returns:
            io.BytesIO: Generated image(s) in bytes format.
            int: Seed used for generation (only if return_seed is True).
        """
        init_image = self.load_init_image_if_needed(init_image)

        # allow for packing and conversion to latent space
        height = 16 * (height // 16)
        width = 16 * (width // 16)

        generator, seed = self.set_seed(seed)

        print(f"Using seed: {seed}")

        # preprocess the latent
        img, timesteps = self.preprocess_latent(
            init_image=init_image,
            height=height,
            width=width,
            num_steps=num_steps,
            strength=strength,
            generator=generator,
            num_images=num_images,
        )

        # prepare inputs
        img, img_ids, vec, txt, txt_ids = map(
            lambda x: x, # x.contiguous(),
            self.prepare(
                img=img,
                prompt=prompt,
                target_device=self.device_flux,
                target_dtype=self.dtype,
            ),
        )

        # dispatch to gpu if offloaded
        if self.offload_flow:
            self.model.to(self.device_flux)

        # perform the denoising loop
        batch_size = img.shape[0]
        output_imgs = []

        for i in range(batch_size):
            denoised_img = self.denoise_single_item(
                img[i],
                img_ids[i],
                txt[i],
                txt_ids[i],
                vec[i],
                timesteps,
                guidance,
                compiling
            )
            output_imgs.append(denoised_img)
            compiling = False

        img = torch.cat(output_imgs)

        # offload the model to cpu if needed
        if self.offload_flow:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        # decode latents to pixel space
        img = self.vae_decode(img, height, width)

        return self.as_img_tensor(img)

    def denoise_single_item(self,
                            img,
                            img_ids,
                            txt,
                            txt_ids,
                            vec,
                            timesteps,
                            guidance,
                            compiling
                            ):

        img = img.unsqueeze(0)
        img_ids = img_ids.unsqueeze(0)
        txt = txt.unsqueeze(0)
        txt_ids = txt_ids.unsqueeze(0)
        vec = vec.unsqueeze(0)

        guidance_vec = torch.full(
            (img.shape[0],), guidance, device=self.device_flux, dtype=self.dtype
        )
        t_vec = None

        for t_curr, t_prev in tqdm(
            zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1
        ):
            if t_vec is None:
                t_vec = torch.full(
                    (img.shape[0],),
                    t_curr,
                    dtype=self.dtype,
                    device=self.device_flux,
                )
            else:
                t_vec = t_vec.reshape((img.shape[0],)).fill_(t_curr)
            if compiling:
                torch._dynamo.mark_dynamic(img, 1, min=256, max=8100)
                torch._dynamo.mark_dynamic(img_ids, 1, min=256, max=8100)
                self.model = torch.compile(self.model)
                compiling = False

            pred = self.model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred
        return img

    @classmethod
    def load_pipeline_from_config_path(
        cls, path: str, flow_model_path: str = None, debug: bool = False, shared_models: LoadedModels = None, **kwargs
    ) -> "FluxPipeline":
        with torch.inference_mode():
            config = load_config_from_path(path)
            if flow_model_path:
                config.ckpt_path = flow_model_path
            for k, v in kwargs.items():
                if hasattr(config, k):
                    print(
                        f"Overriding config {k}:{getattr(config, k)} with value {v}"
                    )
                    setattr(config, k, v)
            return cls.load_pipeline_from_config(config, debug=debug, shared_models=shared_models)

    @classmethod
    def load_pipeline_from_config(
        cls, config: ModelSpec, debug: bool = False, shared_models: LoadedModels = None
    ) -> "FluxPipeline":
        from fp8.float8_quantize import quantize_flow_transformer_and_dispatch_float8

        with torch.inference_mode():
            if debug:
                print(
                    f"Loading as prequantized flow transformer? {config.prequantized_flow}"
                )

            models = load_models_from_config(config, shared_models)
            config = models.config
            flux_device = into_device(config.flux_device)
            ae_device = into_device(config.ae_device)
            clip_device = into_device(config.text_enc_device)
            t5_device = into_device(config.text_enc_device)
            flux_dtype = into_dtype(config.flow_dtype)
            flow_model = models.flow

            if not config.prequantized_flow:
                flow_model = quantize_flow_transformer_and_dispatch_float8(
                    flow_model,
                    flux_device,
                    offload_flow=config.offload_flow,
                    swap_linears_with_cublaslinear=flux_dtype == torch.float16,
                    flow_dtype=flux_dtype,
                    quantize_modulation=config.quantize_modulation,
                    quantize_flow_embedder_layers=config.quantize_flow_embedder_layers,
                )
            else:
                flow_model.eval().requires_grad_(False)

        return cls(
            name=config.version,
            clip=models.clip,
            t5=models.t5,
            model=flow_model,
            ae=models.ae,
            dtype=flux_dtype,
            verbose=False,
            flux_device=flux_device,
            ae_device=ae_device,
            clip_device=clip_device,
            t5_device=t5_device,
            config=config,
            debug=debug,
        )
