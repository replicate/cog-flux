import os
import subprocess
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import logging
from PIL import Image
from pathlib import Path
from diffusers.pipelines import (
    FluxPipeline,
    FluxInpaintPipeline,
    FluxImg2ImgPipeline,
)

from weights import WeightsDownloadCache

from lora_loading_patch import load_lora_into_transformer

MODEL_URL_DEV = (
    "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
)
MODEL_URL_SCHNELL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-schnell/slim.tar"

FLUX_DEV_PATH = "./model-cache/FLUX.1-dev"
FLUX_SCHNELL_PATH = "./model-cache/FLUX.1-schnell"
MODEL_CACHE = "./model-cache/"

MAX_IMAGE_SIZE = 1440


@dataclass
class FluxConfig:
    url: str
    path: str
    download_path: str  # this only exists b/c flux-dev needs a different donwload_path from "path" based on how we're storing weights.
    num_steps: int
    max_sequence_length: int


CONFIGS = {
    "flux-schnell": FluxConfig(
        MODEL_URL_SCHNELL, FLUX_SCHNELL_PATH, FLUX_SCHNELL_PATH, 4, 256
    ),
    "flux-dev": FluxConfig(MODEL_URL_DEV, FLUX_DEV_PATH, MODEL_CACHE, 28, 512),
}

# Suppress diffusers nsfw warnings
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)


@dataclass
class LoadedLoRAs:
    main: str | None
    extra: str | None


from diffusers import AutoencoderKL
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5TokenizerFast


@dataclass
class ModelHolster:
    vae: AutoencoderKL
    text_encoder: CLIPTextModel
    text_encoder_2: T5EncoderModel
    tokenizer: CLIPTokenizer
    tokenizer_2: T5TokenizerFast


class DiffusersFlux:
    """
    Wrapper to map diffusers flux pipeline to the methods we need to serve these models in predict.py
    """

    def __init__(
        self,
        model_name: str,
        weights_cache: WeightsDownloadCache,
        shared_models: ModelHolster | None = None,
    ) -> None:  # pyright: ignore
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()

        # Don't pull weights
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        config = CONFIGS[model_name]
        model_path = config.path

        self.default_num_steps = config.num_steps
        self.max_sequence_length = config.max_sequence_length

        # dependency injection hell yeah it's java time baybee
        self.weights_cache = weights_cache

        if not os.path.exists(model_path):  # noqa: PTH110
            print("Model path not found, downloading models")
            # TODO: download everything separately; it will suck less.
            download_base_weights(config.url, config.download_path)

        print("Loading pipeline")
        if shared_models:
            txt2img_pipe = FluxPipeline.from_pretrained(
                model_path,
                vae=shared_models.vae,
                text_encoder=shared_models.text_encoder,
                text_encoder_2=shared_models.text_encoder_2,
                tokenizer=shared_models.tokenizer,
                tokenizer_2=shared_models.tokenizer_2,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        else:
            txt2img_pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
        txt2img_pipe.__class__.load_lora_into_transformer = classmethod(
            load_lora_into_transformer
        )
        self.txt2img_pipe = txt2img_pipe

        # Load img2img pipelines
        img2img_pipe = FluxImg2ImgPipeline(
            transformer=txt2img_pipe.transformer,
            scheduler=txt2img_pipe.scheduler,
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
        ).to("cuda")
        img2img_pipe.__class__.load_lora_into_transformer = classmethod(
            load_lora_into_transformer
        )

        self.img2img_pipe = img2img_pipe

        # Load inpainting pipelines
        inpaint_pipe = FluxInpaintPipeline(
            transformer=txt2img_pipe.transformer,
            scheduler=txt2img_pipe.scheduler,
            vae=txt2img_pipe.vae,
            text_encoder=txt2img_pipe.text_encoder,
            text_encoder_2=txt2img_pipe.text_encoder_2,
            tokenizer=txt2img_pipe.tokenizer,
            tokenizer_2=txt2img_pipe.tokenizer_2,
        ).to("cuda")
        inpaint_pipe.__class__.load_lora_into_transformer = classmethod(
            load_lora_into_transformer
        )

        self.inpaint_pipe = inpaint_pipe

        self.loaded_lora_urls = LoadedLoRAs(main=None, extra=None)
        self.lora_scale = 1.0
        print("setup took: ", time.time() - start)

    def get_models(self):
        return ModelHolster(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
        )

    def handle_loras(self, lora_weights, lora_scale, extra_lora, extra_lora_scale):
        # all pipes share the same weights, can do this to any of them
        pipe = self.txt2img_pipe

        if lora_weights:
            start_time = time.time()
            if extra_lora:
                self.lora_scale = 1.0
                self.load_multiple_loras(lora_weights, extra_lora)
                pipe.set_adapters(
                    ["main", "extra"], adapter_weights=[lora_scale, extra_lora_scale]
                )
            else:
                self.load_single_lora(lora_weights)
                pipe.set_adapters(["main"], adapter_weights=[lora_scale])
                self.lora_scale = lora_scale
            print(f"Loaded LoRAs in {time.time() - start_time:.2f}s")
        else:
            pipe.unload_lora_weights()
            self.loaded_lora_urls = LoadedLoRAs(main=None, extra=None)
            self.lora_scale = 1.0

    @torch.inference_mode()
    def predict(  # pyright: ignore
        self,
        prompt: str,
        num_outputs: int = 1,
        num_inference_steps: int | None = None,
        legacy_image_path: Path = None,
        legacy_mask_path: Path = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float = 3.5,
        prompt_strength: float = 0.8,
        seed: int | None = None,
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None or seed < 0:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        is_img2img_mode = legacy_image_path is not None and legacy_mask_path is None
        is_inpaint_mode = legacy_image_path is not None and legacy_mask_path is not None

        flux_kwargs = {}
        print(f"Prompt: {prompt}")

        if is_img2img_mode or is_inpaint_mode:
            input_image = Image.open(legacy_image_path).convert("RGB")
            original_width, original_height = input_image.size

            width = original_width
            height = original_height
            print(f"Input image size: {width}x{height}")

            # Calculate the scaling factor if the image exceeds max_image_size
            scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
            if scale < 1:
                width = int(width * scale)
                height = int(height * scale)

            # Calculate dimensions that are multiples of 16
            target_width = make_multiple_of_16(width)
            target_height = make_multiple_of_16(height)
            target_size = (target_width, target_height)

            print(
                f"[!] Resizing input image from {original_width}x{original_height} to {target_width}x{target_height}"
            )

            # We're using highest quality settings; if you want to go fast, you're not running this code.
            input_image = input_image.resize(target_size, Image.LANCZOS)
            flux_kwargs["image"] = input_image

            # Set width and height to match the resized input image
            flux_kwargs["width"], flux_kwargs["height"] = target_size

            if is_img2img_mode:
                print("[!] img2img mode")
                pipe = self.img2img_pipe
            else:  # is_inpaint_mode
                print("[!] inpaint mode")
                mask_image = Image.open(legacy_mask_path).convert("RGB")
                mask_image = mask_image.resize(target_size, Image.NEAREST)
                flux_kwargs["mask_image"] = mask_image
                pipe = self.inpaint_pipe

            flux_kwargs["strength"] = prompt_strength

        else:  # is_txt2img_mode
            print("[!] txt2img mode")
            pipe = self.txt2img_pipe
            flux_kwargs["width"] = width
            flux_kwargs["height"] = height

        max_sequence_length = self.max_sequence_length

        generator = torch.Generator(device="cuda").manual_seed(seed)

        if self.loaded_lora_urls.main is not None:
            # this sets lora scale for prompt encoding, weirdly enough. it does not actually do anything with attention processing anymore.
            flux_kwargs["joint_attention_kwargs"] = {"scale": self.lora_scale}

        common_args = {
            "prompt": [prompt] * num_outputs,
            "guidance_scale": guidance,
            "generator": generator,
            "num_inference_steps": num_inference_steps
            if num_inference_steps
            else self.default_num_steps,
            "max_sequence_length": max_sequence_length,
            "output_type": "pil",
        }

        output = pipe(**common_args, **flux_kwargs)

        return output.images, [np.array(img) for img in output.images]

    def load_single_lora(self, lora_url: str):
        # If no change, skip
        if lora_url == self.loaded_lora_urls.main:
            print("Weights already loaded")
            return

        pipe = self.txt2img_pipe
        pipe.unload_lora_weights()
        lora_path = self.weights_cache.ensure(lora_url)
        pipe.load_lora_weights(lora_path, adapter_name="main")
        self.loaded_lora_urls = LoadedLoRAs(main=lora_url, extra=None)
        pipe = pipe.to("cuda")

    def load_multiple_loras(self, main_lora_url: str, extra_lora_url: str):
        pipe = self.txt2img_pipe

        # If no change, skip
        if (
            main_lora_url == self.loaded_lora_urls.main
            and extra_lora_url == self.loaded_lora_urls.extra
        ):
            print("Weights already loaded")
            return

        # We always need to load both?
        pipe.unload_lora_weights()

        main_lora_path = self.weights_cache.ensure(main_lora_url)
        pipe.load_lora_weights(main_lora_path, adapter_name="main")

        extra_lora_path = self.weights_cache.ensure(extra_lora_url)
        pipe.load_lora_weights(extra_lora_path, adapter_name="extra")

        self.loaded_lora_urls = LoadedLoRAs(main=main_lora_url, extra=extra_lora_url)
        pipe = pipe.to("cuda")


def download_base_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def make_multiple_of_16(n):
    # Rounds up to the next multiple of 16, or returns n if already a multiple of 16
    return ((n + 15) // 16) * 16
