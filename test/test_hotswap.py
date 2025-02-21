import inspect
import os
import time

import pydantic
from predict import HotswapPredictor
from PIL import Image
from utils import get_params

@pytest.fixture
def uncompiled_hotswap_predictor():
    p = HotswapPredictor()
    p.setup(torch_compile=False)
    return p

def test_functionality(uncompiled_hotswap_predictor):
    # base prediction
    p.predict(
        prompt="a selfie, style of 80s cyberpunk",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=0.5,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=False,
        megapixels="1",
        replicate_weights="fofr/flux-80s-cyberpunk",
        lora_scale=1.1,
        extra_lora=None,
        extra_lora_scale=None,
    )
    os.system("mv out-0.png 80s-cyberpunk.png")


    # test inpainting
    p.predict(
        prompt="a green cat sitting on a park bench",
        image="test/inpainting-img.png",
        mask="test/inpainting-mask.png",
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=1.0,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=True,
        megapixels="1",
        replicate_weights="fofr/flux-80s-cyberpunk",
        lora_scale=1.1,
        extra_lora=None,
        extra_lora_scale=None,
    )
    os.system("mv out-0.png cat_on_bench.png")

def test_multi_lora(uncompiled_hotswap_predictor):
    st = time.time()
    p.predict(
        prompt="A portrait photo of MNALSA woman sitting at a party table with a selection of bad 70s food",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=1.0,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=True,
        megapixels="1",
        replicate_weights="fofr/flux-bad-70s-food",
        lora_scale=0.85,
        extra_lora="fofr/flux-mona-lisa",
        extra_lora_scale=1.0,
    )
    os.system("mv out-0.png mona-lisa-fast.png")
    print("fast lora prediction took ", time.time() - st)

    st = time.time()
    p.predict(
        prompt="A portrait photo of MNALSA woman sitting at a party table with a selection of bad 70s food",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=1.0,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=False,
        megapixels="1",
        replicate_weights="fofr/flux-bad-70s-food",
        lora_scale=0.85,
        extra_lora="fofr/flux-mona-lisa",
        extra_lora_scale=0.8,
    )
    os.system("mv out-0.png mona-lisa-slow.png")
    print("slow lora prediction took ", time.time() - st)

    st = time.time()
    p.predict(
        prompt="A portrait photo of MNALSA woman sitting at a party table with a selection of bad 70s food",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=1.0,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=True,
        megapixels="1",
        replicate_weights=None,
        lora_scale=0.85,
        extra_lora=None,
        extra_lora_scale=0.8,
    )
    os.system("mv out-0.png no-mona-lisa-fast.png")
    print("fast no lora prediction took ", time.time() - st)

    st = time.time()
    p.predict(
        prompt="A portrait photo of MNALSA woman sitting at a party table with a selection of bad 70s food",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=1.0,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=False,
        megapixels="1",
        replicate_weights=None,
        lora_scale=0.85,
        extra_lora=None,
        extra_lora_scale=0.8,
    )
    os.system("mv out-0.png no-mona-lisa-slow.png")
    print("slow no lora prediction took ", time.time() - st)

def test_resizing(uncompiled_hotswap_predictor):
    test_inputs = {
        "image": "./test/resources/2048_x_2048.jpg",
        "model": "dev",
        "prompt": "a cool dog"
        "go_fast": False,
        "output_format": "png",
        "guidance_scale": 3,
        "prompt_strength": 0.35,
        "lora_scale": 1,
        "replicate_weights": "fofr/flux-mona-lisa"
    }
    inputs = get_params(p, test_inputs)
    p.predict(**inputs)
    os.system("mv out-0.png right_size.png")
    img = Image.open("right_size.png")
    print(img.size)
    assert img.size[0] <= 1440
    assert img.size[1] <= 1440

def test_text_encoder_loras(uncompiled_hotswap_predictor):
    # base prediction
    p.predict(
        prompt="cy04, a book titled Did I Leave The Oven On?, an illustration of a man sitting at work, looking worried, thought bubble above his head with an oven in it",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=0.5,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=1234,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=False,
        megapixels="1",
        replicate_weights="huggingface.co/Purz/choose-your-own-adventure",
        lora_scale=1.1,
        extra_lora=None,
        extra_lora_scale=None,
    )
    os.system("mv out-0.png cyo.png")

    p.predict(
        prompt="cy04, a book titled Did I Leave The Oven On?, an illustration of a man sitting at work, looking worried, thought bubble above his head with an oven in it",
        image=None,
        mask=None,
        aspect_ratio="1:1",
        height=None,
        width=None,
        prompt_strength=0.5,
        model="dev",
        num_outputs=1,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=1234,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=False,
        megapixels="1",
        replicate_weights="huggingface.co/Purz/choose-your-own-adventure",
        lora_scale=1.1,
        extra_lora=None,
        extra_lora_scale=None,
    )
    os.system("mv out-0.png cyo2.png")

if __name__ == "__main__":
    p = HotswapPredictor()
    p.setup(torch_compile=False)
    test_text_encoder(p)
