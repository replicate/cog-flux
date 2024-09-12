import cProfile
import os
import pickle
import time
import logging
import statistics
from typing import Optional

from attr import dataclass
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack

import torch
import numpy as np
from einops import rearrange
from PIL import Image
from typing import List
from einops import rearrange
from torchvision import transforms
from cog import BasePredictor, Input, Path
from flux.util import load_ae, load_clip, load_flow_model, load_t5, download_weights

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import (
    CLIPImageProcessor,
    AutoModelForImageClassification,
    ViTImageProcessor,
)

SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "/src/feature-extractor"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
MAX_IMAGE_SIZE = 1440

FALCON_MODEL_NAME = "Falconsai/nsfw_image_detection"
FALCON_MODEL_CACHE = "falcon-cache"
FALCON_MODEL_URL = (
    "https://weights.replicate.delivery/default/falconai/nsfw-image-detection.tar"
)

# Suppress diffusers nsfw warnings
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)


@dataclass
class SharedInputs:
    prompt: Input = Input(description="Prompt for generated image")
    aspect_ratio: Input = Input(
        description="Aspect ratio for the generated image",
        choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
        default="1:1",
    )
    num_outputs: Input = Input(
        description="Number of outputs to generate", default=1, le=4, ge=1
    )
    seed: Input = Input(
        description="Random seed. Set for reproducible generation", default=None
    )
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
        description="Disable safety checker for generated images.",
        default=False,
    )


SHARED_INPUTS = SharedInputs()


class Predictor(BasePredictor):
    def setup(self) -> None:
        return

    def base_setup(self, flow_model_name: str, compile: bool) -> None:
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")

        gpu_name = (
            os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
            .read()
            .strip()
        )
        print("Detected GPU:", gpu_name)

        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        print("Loading Safety Checker to GPU")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        print("Loading Falcon safety checker...")
        if not os.path.exists(FALCON_MODEL_CACHE):
            download_weights(FALCON_MODEL_URL, FALCON_MODEL_CACHE)
        self.falcon_model = AutoModelForImageClassification.from_pretrained(
            FALCON_MODEL_NAME,
            cache_dir=FALCON_MODEL_CACHE,
        )
        self.falcon_processor = ViTImageProcessor.from_pretrained(FALCON_MODEL_NAME)

        # need > 48 GB of ram to store all models in VRAM
        self.offload = "A40" in gpu_name

        device = "cuda"
        max_length = 256 if self.flow_model_name == "flux-schnell" else 512
        self.t5 = load_t5(device, max_length=max_length)
        self.clip = load_clip(device)
        self.flux = load_flow_model(
            self.flow_model_name, device="cpu" if self.offload else device
        )
        self.flux = self.flux.eval()
        self.ae = load_ae(
            self.flow_model_name, device="cpu" if self.offload else device
        )

        self.num_steps = 4 if self.flow_model_name == "flux-schnell" else 28
        self.shift = self.flow_model_name != "flux-schnell"
        self.compile_run = False
        if compile:
            torch._inductor.config.fallback_random = True
            self.compile_run = True
            self.predict(
                prompt="a cool dog",
                aspect_ratio="1:1",
                image=None,
                prompt_strength=1,
                num_outputs=1,
                num_inference_steps=self.num_steps,
                guidance=3.5,
                output_format="png",
                output_quality=80,
                disable_safety_checker=True,
                seed=123,
            )

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

    def get_image(self, image: str):
        if image is None:
            return None
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]

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
        num_inference_steps: int,
        guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
        image: Path = None,  # img2img for flux-dev
        prompt_strength: float = 0.8,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Run a single prediction on the model"""
        torch_device = torch.device("cuda")
        init_image = None
        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # img2img only works for flux-dev
        if image:
            print("Image detected - settting to img2img mode")
            init_image = self.get_image(image)
            width = init_image.shape[-1]
            height = init_image.shape[-2]
            print(f"Input image size: {width}x{height}")

            # Calculate the scaling factor if the image exceeds MAX_IMAGE_SIZE
            scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
            if scale < 1:
                width = int(width * scale)
                height = int(height * scale)
                print(f"Scaling image down to {width}x{height}")

            # Round image width and height to nearest multiple of 16
            width = round(width / 16) * 16
            height = round(height / 16) * 16
            print(f"Input image size set to: {width}x{height}")

            # Resize
            init_image = init_image.to(torch_device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae.encoder.to(torch_device)
            init_image = self.ae.encode(init_image)
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

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
            num_inference_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=self.shift
        )

        if init_image is not None:
            t_idx = int((1.0 - prompt_strength) * num_inference_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=[prompt] * num_outputs)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.flux = self.flux.to(torch_device)

        if self.compile_run:
            print("Compiling")
            st = time.time()

        x, flux = denoise(
            self.flux,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            compile_run=self.compile_run,
        )

        if self.compile_run:
            print(f"Compiled in {time.time() - st}")
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

        images = [
            Image.fromarray(
                (127.5 * (rearrange(x[i], "c h w -> h w c").clamp(-1, 1) + 1.0))
                .cpu()
                .byte()
                .numpy()
            )
            for i in range(num_outputs)
        ]
        has_nsfw_content = [False] * len(images)
        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(images)  # always on gpu

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

    @torch.inference_mode()
    def simple_predict(self, prompt: str) -> List[Path]:
        """Run a single prediction on the model"""
        torch_device = torch.device("cuda")

        # prepare input
        x = get_noise(
            num_samples=1,
            height=1024,
            width=1024,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=1,
        )
        timesteps = get_schedule(4, (x.shape[-1] * x.shape[-2]) // 4)

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=[prompt])

        x, flux = denoise(self.flux, **inp, timesteps=timesteps)

        x = unpack(x.float(), 1024, 1024)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        gpu_image = 127.5 * (rearrange(x[0], "c h w -> h w c").clamp(-1, 1) + 1.0)
        cpu_image = gpu_image.cpu()
        image = Image.fromarray(cpu_image.byte().numpy())

        _, has_nsfw_content = self.run_safety_checker([image])

        if has_nsfw_content[0]:
            print("NSFW content detected")
            return
        output_path = "out.webp"
        image.save(output_path)
        return Path(output_path)

    def run_safety_checker(self, images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )
        np_images = [np.array(img) for img in images]
        image, has_nsfw_concept = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def run_falcon_safety_checker(self, image):
        with torch.no_grad():
            inputs = self.falcon_processor(images=image, return_tensors="pt")
            outputs = self.falcon_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]

        return result == "normal"


class SchnellPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-schnell", compile=False)

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

        return self.base_predict(
            prompt,
            aspect_ratio,
            num_outputs,
            output_format,
            output_quality,
            disable_safety_checker,
            num_inference_steps=self.num_steps,
            seed=seed,
        )


class DevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup("flux-dev", compile=True)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = SHARED_INPUTS.prompt,
        aspect_ratio: str = SHARED_INPUTS.aspect_ratio,
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
        num_outputs: int = SHARED_INPUTS.num_outputs,
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended range is 28-50",
            ge=1,
            le=50,
            default=28,
        ),
        guidance: float = Input(
            description="Guidance for generated image", ge=0, le=10, default=3
        ),
        seed: int = SHARED_INPUTS.seed,
        output_format: str = SHARED_INPUTS.output_format,
        output_quality: int = SHARED_INPUTS.output_quality,
        disable_safety_checker: bool = SHARED_INPUTS.disable_safety_checker,
    ) -> List[Path]:

        return self.base_predict(
            prompt,
            aspect_ratio,
            num_outputs,
            output_format,
            output_quality,
            disable_safety_checker,
            guidance=guidance,
            image=image,
            prompt_strength=prompt_strength,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

if __name__ == "__main__":
    p = SchnellPredictor()
    p.setup()
    p.simple_predict("hi")
    #p.base_predict("hi", "1:1", 1, "webp", "", False, num_inference_steps=4)
    times = []
    with cProfile.Profile() as prof:
        for i in range(10):
            st = time.time()
            #p.base_predict("hi", "1:1", 1, "webp", "", False)
            p.simple_predict("hi")
            times.append(time.time() - st)
        prof.dump_stats("prof")
    print(statistics.mean(times), statistics.median(times))

