from abc import ABC, abstractmethod
import os
from typing import Tuple

import torch

from bfl_predictor import (
    BflBf16Predictor,
    BflControlNetFlux,
    BflFillFlux,
    BflFp8Flux,
    BflReduxPredictor,
)
from diffusers_predictor import DiffusersFlux
from flux.modules.conditioner import PreLoadedHFEmbedder
from fp8.util import LoadedModels

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20
import logging


from dataclasses import dataclass

import numpy as np
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path  # type: ignore
from flux.util import (
    download_weights,
    load_ae,
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
FLUX_DEV_FP8 = "flux-dev-fp8"
FLUX_SCHNELL = "flux-schnell"
FLUX_SCHNELL_FP8 = "flux-schnell-fp8"

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


class Predictor(BasePredictor, ABC):
    """
    Base object with shared flux pre & post-processing functionality (e.g. safety checking) that doesn't depend on implementation.
    The goal here, broadly, is to decouple the Cog and therefore Replicate API for flux (defined by this object and its child classes)
    from the actual inference implementation (defined by the various objects that the child classes instantiate in their setup() fucntions).

    This enables decoupling bf16 implementations from fp8 implementations, and hosting multiple implementations in the same model.

    All child objects need to implement the Cog `setup` and `predict` functions.
    """

    def setup(self) -> None:
        return

    def base_setup(
        self,
    ) -> None:
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

    def should_offload(self):
        # need > 48 GB of ram to store all models in VRAM
        total_mem = torch.cuda.get_device_properties(0).total_memory
        self.offload = total_mem < 48 * 1024**3
        if self.offload:
            print("GPU memory is:", total_mem / 1024**3, ", offloading models")
        return self.offload

    @abstractmethod
    def predict(self):
        pass

    def size_from_aspect_megapixels(
        self, aspect_ratio: str, megapixels: str = "1"
    ) -> Tuple[int, int]:
        width, height = ASPECT_RATIOS[aspect_ratio]
        if megapixels == "0.25":
            width, height = width // 2, height // 2

        return (width, height)

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


class SchnellPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup()
        self.bf16_model = BflBf16Predictor(FLUX_SCHNELL, offload=self.should_offload())
        self.fp8_model = BflFp8Flux(
            FLUX_SCHNELL_FP8,
            loaded_models=self.bf16_model.get_shared_models(),
            torch_compile=True,
            compilation_aspect_ratios=ASPECT_RATIOS,
            offload=self.should_offload(),
        )

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
        model = self.fp8_model if go_fast else self.bf16_model

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = model.predict(
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
        self.base_setup()
        self.bf16_model = BflBf16Predictor(FLUX_DEV, offload=self.should_offload())
        self.fp8_model = BflFp8Flux(
            FLUX_DEV_FP8,
            loaded_models=self.bf16_model.get_shared_models(),
            torch_compile=True,
            compilation_aspect_ratios=ASPECT_RATIOS,
            offload=self.should_offload(),
        )

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
        model = self.fp8_model if go_fast else self.bf16_model
        imgs, np_imgs = model.predict(
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance,
            legacy_image_path=image,
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
        self.base_setup()
        cache = WeightsDownloadCache()
        self.bf16_model = BflBf16Predictor(
            FLUX_SCHNELL,
            offload=self.should_offload(),
            weights_download_cache=cache,
            restore_lora_from_cloned_weights=True,
        )
        self.fp8_model = BflFp8Flux(
            FLUX_SCHNELL_FP8,
            loaded_models=self.bf16_model.get_shared_models(),
            torch_compile=True,
            compilation_aspect_ratios=ASPECT_RATIOS,
            offload=self.should_offload(),
            weights_download_cache=cache,
            restore_lora_from_cloned_weights=True,
        )

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
        model = self.fp8_model if go_fast else self.bf16_model
        model.handle_loras(lora_weights, lora_scale)

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = model.predict(
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
    def setup(self) -> None:
        self.base_setup()
        cache = WeightsDownloadCache()
        self.bf16_model = BflBf16Predictor(
            FLUX_DEV,
            offload=self.should_offload(),
            weights_download_cache=cache,
            restore_lora_from_cloned_weights=True,
        )
        self.fp8_model = BflFp8Flux(
            FLUX_DEV_FP8,
            loaded_models=self.bf16_model.get_shared_models(),
            torch_compile=True,
            compilation_aspect_ratios=ASPECT_RATIOS,
            offload=self.should_offload(),
            weights_download_cache=cache,
            restore_lora_from_cloned_weights=True,
        )

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

        model = self.fp8_model if go_fast else self.bf16_model
        model.handle_loras(lora_weights, lora_scale)

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = model.predict(
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance,
            legacy_image_path=image,
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


class SchnellReduxPredictor(Predictor):
    def setup(self):
        self.base_setup()
        self.model = BflReduxPredictor(FLUX_SCHNELL, offload=self.should_offload())

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
        prompt = ""

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.model.predict(
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            seed=seed,
            width=width,
            height=height,
            prepare_kwargs={"redux_img_path": redux_image},
        )

        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class DevReduxPredictor(Predictor):
    def setup(self):
        self.base_setup()
        self.model = BflReduxPredictor(FLUX_DEV, offload=self.should_offload())

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
        prompt = ""

        width, height = self.size_from_aspect_megapixels(aspect_ratio, megapixels)
        imgs, np_imgs = self.model.predict(
            prompt,
            num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            prepare_kwargs={"redux_img_path": redux_image},
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
        self.base_setup()
        cache = WeightsDownloadCache()
        self.model = BflFillFlux(
            "flux-fill-dev",
            offload=self.should_offload(),
            weights_download_cache=cache,
            restore_lora_from_cloned_weights=True,
        )

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
        lora_weights: str = Inputs.lora_weights,
        lora_scale: float = Inputs.lora_scale,
        disable_safety_checker: bool = Inputs.disable_safety_checker,
    ) -> List[Path]:
        # TODO(andreas): This means we're reading the image twice
        # which is a bit inefficient.
        width, height = self.size_maybe_match_input(image, megapixels)

        self.model.handle_loras(lora_weights, lora_scale)

        imgs, np_imgs = self.model.predict(
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            conditioning_kwargs={"image_path": image, "mask_path": mask},
        )
        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


class HotswapPredictor(Predictor):
    def setup(self, torch_compile=True) -> None:
        self.base_setup()
        shared_cache = WeightsDownloadCache()

        self.bf16_dev = DiffusersFlux(FLUX_DEV, shared_cache)
        shared_models = self.bf16_dev.get_models()

        # hack to get around delta in vae code
        bfl_ae = load_ae(FLUX_DEV)

        shared_models_for_fp8 = LoadedModels(
            ae=bfl_ae,
            clip=PreLoadedHFEmbedder(
                True, 77, shared_models.tokenizer, shared_models.text_encoder
            ),
            t5=PreLoadedHFEmbedder(
                False, 512, shared_models.tokenizer_2, shared_models.text_encoder_2
            ),
            flow=None,
            config=None,
        )
        self.fp8_dev = BflFp8Flux(
            FLUX_DEV_FP8,
            shared_models_for_fp8,
            torch_compile=torch_compile,
            compilation_aspect_ratios=ASPECT_RATIOS,
            weights_download_cache=shared_cache,
            restore_lora_from_cloned_weights=True,
        )

        self.bf16_schnell = DiffusersFlux(FLUX_SCHNELL, shared_cache, shared_models)
        shared_models_for_fp8.t5 = PreLoadedHFEmbedder(
            False, 256, shared_models.tokenizer_2, shared_models.text_encoder_2
        )

        self.fp8_schnell = BflFp8Flux(
            FLUX_SCHNELL_FP8,
            shared_models_for_fp8,
            torch_compile=torch_compile,
            compilation_aspect_ratios=ASPECT_RATIOS,
            weights_download_cache=shared_cache,
            restore_lora_from_cloned_weights=True,
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
        if aspect_ratio == "custom":
            if go_fast:
                print(
                    "Custom aspect ratios not supported with fast fp8 inference; will run in bf16"
                )
                go_fast = False
            width = make_multiple_of_16(width)
            height = make_multiple_of_16(height)
        else:
            width, height = self.size_from_aspect_megapixels(
                aspect_ratio, megapixels=megapixels
            )

        if image and go_fast:
            print(
                "Img2img and inpainting not supported with fast fp8 inference; will run in bf16"
            )
            go_fast = False

        if model == "dev":
            model = self.fp8_dev if go_fast else self.bf16_dev
        else:
            model = self.fp8_schnell if go_fast else self.bf16_schnell

        model.handle_loras(replicate_weights, lora_scale, extra_lora, extra_lora_scale)

        imgs, np_imgs = model.predict(
            prompt,
            num_outputs,
            num_inference_steps,
            guidance=guidance_scale,
            legacy_image_path=image,
            legacy_mask_path=mask,
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


class CannyDevPredictor(Predictor):
    def setup(self) -> None:
        self.base_setup()
        self.model = BflControlNetFlux("flux-canny-dev", offload=self.should_offload())

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

        imgs, np_imgs = self.model.predict(
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            conditioning_kwargs={"image_path": control_image},
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
        self.base_setup()
        self.model = BflControlNetFlux("flux-depth-dev", offload=self.should_offload())

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

        imgs, np_imgs = self.model.predict(
            prompt=prompt,
            num_outputs=num_outputs,
            num_inference_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            width=width,
            height=height,
            conditioning_kwargs={"image_path": control_image},
        )
        return self.postprocess(
            imgs,
            disable_safety_checker,
            output_format,
            output_quality,
            np_images=np_imgs,
        )


def make_multiple_of_16(n):
    # Rounds up to the next multiple of 16, or returns n if already a multiple of 16
    return ((n + 15) // 16) * 16
