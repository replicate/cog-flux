import os
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

import torch
import numpy as np
from einops import rearrange
from PIL import Image
from cog import BasePredictor, Input, Path, emit_metric
from flux.util import load_ae, load_clip, load_flow_model, load_t5, download_weights

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        gpu_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits").read().strip()
        print("Detected GPU:", gpu_name)

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        # need > 48 GB of ram to store all models in VRAM
        self.offload = "A40" in gpu_name

        self.flow_model_name = os.getenv("FLUX_MODEL", "flux-schnell")
        print(f"Booting model {self.flow_model_name}")

        device = "cuda" 
        max_length = 256 if self.flow_model_name == "flux-schnell" else 512
        self.t5 = load_t5(device, max_length=max_length)
        self.clip = load_clip(device)
        self.flux = load_flow_model(self.flow_model_name, device="cpu" if self.offload else device)
        self.ae = load_ae(self.flow_model_name, device="cpu" if self.offload else device)

        self.num_steps = 4 if self.flow_model_name == "flux-schnell" else 50
        self.shift = self.flow_model_name != "flux-schnell"

    
    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Prompt for generated image"),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        # guidance: float = Input(description="Guidance for generated image. Ignored for flux-schnell", ge=0, le=10, default=3.5),
        # num_outputs: int = Input(description="Number of outputs to generate", default=1, le=4, ge=1),
        seed: int = Input(description="Random seed. Set for reproducible generation", default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        torch_device = "cuda"
        num_outputs = 1
        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        
        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        
        x = get_noise(num_outputs, height, width, device=torch_device, dtype=torch.bfloat16, seed=seed)

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)

        inp = prepare(self.t5, self.clip, x, prompt=prompt)
        timesteps = get_schedule(self.num_steps, inp["img"].shape[1], shift=self.shift)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            self.safety_checker = self.safety_checker.cpu()
            torch.cuda.empty_cache()
            self.flux = self.flux.to(torch_device)

        # handling api mismatch for dev/schnell
        if "guidance" not in locals():
            guidance = 3.5

        x = denoise(self.flux, **inp, timesteps=timesteps, guidance=guidance)

        if self.offload:
            self.flux.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
        
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        # bring into PIL format and save
        x = rearrange(x[0], "c h w -> h w c").clamp(-1, 1)
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        if not disable_safety_checker:
            if self.offload:
                self.safety_checker = self.safety_checker.to("cuda")
            _, has_nsfw_content = self.run_safety_checker(img)
            if has_nsfw_content:
                raise Exception(f"NSFW content detected. Try running it again, or try a different prompt.")

        output_path = f"out-0.{output_format}"
        if output_format != 'png':
            img.save(output_path, quality=output_quality, optimize=True)
        else:
            img.save(output_path)

        emit_metric("num_images", num_outputs)
        return Path(output_path)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = np.array(image)
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
