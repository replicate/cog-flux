import os
from typing import Optional

from attr import dataclass
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

import torch
import numpy as np
from einops import rearrange
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path, emit_metric
from flux.util import load_ae, load_clip, load_flow_model, load_t5, download_weights

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"

@dataclass
class SharedInputs:
    prompt: Input = Input(description="Prompt for generated image")
    aspect_ratio: Input = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1")
    num_outputs: Input = Input(description="Number of outputs to generate", default=1, le=10, ge=1)
    seed: Input = Input(description="Random seed. Set for reproducible generation", default=None)
    output_format: Input = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        )
    output_quality: Input = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        )
    disable_safety_checker: Input = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
    )

SHARED_INPUTS = SharedInputs()


class Predictor(BasePredictor):
    def setup(self) -> None:
        return

    def base_setup(self, flow_model_name) -> None:
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")

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
    
    def predict():
        raise Exception("You need to instantiate a predictor for a specific flux model")

    def base_predict(
        self,
        prompt: str,
        aspect_ratio: str,
        num_outputs: int,
        output_format: str,
        output_quality: int,
        disable_safety_checker: bool,
        guidance: float = 3.5, # schnell ignores guidance within the model, fine to have default
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Run a single prediction on the model"""
        torch_device = "cuda"
        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        
        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        
        x = get_noise(num_outputs, height, width, device=torch_device, dtype=torch.bfloat16, seed=seed)

        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)

        timesteps = get_schedule(self.num_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=self.shift)
        inp = prepare(self.t5, self.clip, x, prompt=[prompt]*num_outputs)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            self.safety_checker = self.safety_checker.cpu()
            torch.cuda.empty_cache()
            self.flux = self.flux.to(torch_device)

        x = denoise(self.flux, **inp, timesteps=timesteps, guidance=guidance)

        if self.offload:
            self.flux.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)
        
        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=torch_device, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        output_paths = []
        for i in range(num_outputs):
            # bring into PIL format and save
            img = rearrange(x[i], "c h w -> h w c").clamp(-1, 1)
            img = Image.fromarray((127.5 * (img + 1.0)).cpu().byte().numpy())

            if not disable_safety_checker:
                if self.offload:
                    self.safety_checker = self.safety_checker.to("cuda")
                _, has_nsfw_content = self.run_safety_checker(img)
                if has_nsfw_content[0]:
                    raise Exception(f"NSFW content detected in image {i+1}. Try running it again, or try a different prompt.")

            output_path = f"out-{i}.{output_format}"
            if output_format != 'png':
                img.save(output_path, quality=output_quality, optimize=True)
            else:
                img.save(output_path)
            output_paths.append(Path(output_path))

        emit_metric("num_images", num_outputs)
        return output_paths

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
    

class SchnellPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-schnell")
    
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = SHARED_INPUTS.prompt,
        aspect_ratio: str = SHARED_INPUTS.aspect_ratio,
        num_outputs: int = SHARED_INPUTS.num_outputs,
        seed: int = SHARED_INPUTS.seed,
        output_format: str = SHARED_INPUTS.output_format,
        output_quality: int = SHARED_INPUTS.output_quality,
        disable_safety_checker: bool = SHARED_INPUTS.disable_safety_checker,
    ) -> List[Path]:

        return self.base_predict(prompt, aspect_ratio, num_outputs, output_format, output_quality, disable_safety_checker, seed=seed)
    

class DevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-dev")
    
    @torch.inference_mode()
    def predict(
        self,
        prompt: str = SHARED_INPUTS.prompt,
        aspect_ratio: str = SHARED_INPUTS.aspect_ratio,
        num_outputs: int = SHARED_INPUTS.num_outputs,
        guidance: float = Input(description="Guidance for generated image. Ignored for flux-schnell", ge=0, le=10, default=3.5),
        seed: int = SHARED_INPUTS.seed,
        output_format: str = SHARED_INPUTS.output_format,
        output_quality: int = SHARED_INPUTS.output_quality,
        disable_safety_checker: bool = SHARED_INPUTS.disable_safety_checker,
    ) -> List[Path]:

        return self.base_predict(prompt, aspect_ratio, num_outputs, output_format, output_quality, disable_safety_checker, guidance=guidance, seed=seed)
