import os
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

import torch
from einops import rearrange
from PIL import Image
from cog import BasePredictor, Input, Path, emit_metric
from flux.util import load_ae, load_clip, load_flow_model, load_t5


class Predictor(BasePredictor):
    def setup(self) -> None:
        gpu_name = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits").read().strip()
        print("Detected GPU:", gpu_name)

        # need > 48 GB of ram to store all models in VRAM
        self.offload = "A40" in gpu_name

        self.flow_model_name = os.getenv("FLUX_MODEL", "flux-schnell")

        device = "cuda" 
        self.t5 = load_t5(device)
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
        guidance: int = Input(description="Guidance for generated image. Ignored for flux-schnell", ge=0, le=10, default=3.5),
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
        timesteps = get_schedule(self.num_steps, inp["img"].shape[0], shift=self.shift)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
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

        # bring into PIL format and save
        x = rearrange(x[0], "c h w -> h w c").clamp(-1, 1)
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        output_path = f"out-0.{output_format}"
        if output_format != 'png':
            img.save(output_path, quality=output_quality, optimize=True)
        else:
            img.save(output_path)

        emit_metric("num_images", num_outputs)
        return Path(output_path)

