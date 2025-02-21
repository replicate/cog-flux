import hashlib
import os
import pytest
from predict import DevLoraPredictor
from utils import get_params
from PIL import Image


@pytest.fixture()
def uncompiled_dev_lora_predictor():
    p = DevLoraPredictor()
    p.setup(torch_compile=False)
    return p

@pytest.fixture()
def dev_lora_predictor():
    p = DevLoraPredictor()
    p.setup()
    return p

def test_reload_scale(uncompiled_dev_lora_predictor):
    p = uncompiled_dev_lora_predictor
    p.predict(
        prompt="a cool dog",
        aspect_ratio="1:1",
        image=None,
        prompt_strength=0.5,
        num_outputs=1,
        num_inference_steps=28,
        guidance=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=True,
        lora_weights="fofr/flux-80s-cyberpunk",
        lora_scale=1.1,
    )

    p.predict(
        prompt="a cool dog",
        aspect_ratio="1:1",
        image=None,
        prompt_strength=0.5,
        num_outputs=1,
        num_inference_steps=28,
        guidance=3.5,
        seed=None,
        output_format="png",
        output_quality=80,
        disable_safety_checker=False,
        go_fast=True,
        lora_weights="fofr/flux-80s-cyberpunk",
        lora_scale=1.5,
    )
    assert True


def test_text_encoder_lora(uncompiled_dev_lora_predictor):
    p = uncompiled_dev_lora_predictor
    base_lora_prediction = {
        "prompt": "a cool dog in the style of 80s cyberpunk",
        "aspect_ratio":"1:1",
        "go_fast": False,
        "lora_weights": "fofr/flux-80s-cyberpunk",
        "lora_scale":1.5,
        "output_format": "png",
        "seed": 42
    }

    text_encoder_lora_prediction = {
        "prompt": "cy04, a book titled Did I Leave The Oven On?, an illustration of a man sitting at work, looking worried, thought bubble above his head with an oven in it",
        "num_inference_steps": 28,
        "guidance": 3.5,
        "seed": 1234,
        "go_fast": False,
        "lora_weights": "huggingface.co/Purz/choose-your-own-adventure",
        "output_format": "png"
    }
    base_in = get_params(p, base_lora_prediction)
    te_in = get_params(p, text_encoder_lora_prediction)

    p.predict(
        **base_in
    )
    os.system("mv out-0.png cyberdog_1.png")

    p.predict(
        **te_in
    )
    os.system("mv out-0.png book_1.png")

    p.predict(
        **base_in
    )
    os.system("mv out-0.png cyberdog_2.png")

    p.predict(
        **te_in
    )
    os.system("mv out-0.png book_2.png")

    assert_same_imgs("cyberdog_1.png", "cyberdog_2.png")
    assert_same_imgs("book_1.png", "book_2.png")

    # # run same predictions in fp8, assert that nothing's changed
    base_in['go_fast'] = True
    te_in['go_fast'] = True
    
    p.predict(**base_in)
    os.system("mv out-0.png fast_dog_1.png")

    p.predict(**te_in)
    os.system("mv out-0.png fast_book_1.png")

    p.predict(**base_in)
    os.system("mv out-0.png fast_dog_2.png")

    p.predict(**te_in)
    os.system("mv out-0.png fast_book_2.png")

    assert_same_imgs("fast_dog_1.png", "fast_dog_2.png")
    assert_same_imgs("fast_book_1.png", "fast_book_2.png")

    te_in['go_fast'] = False
    base_in['go_fast'] = False

    p.predict(
        **te_in
    )
    os.system("mv out-0.png book_3.png")

    p.predict(
        **base_in
    )
    os.system("mv out-0.png cyberdog_3.png")

    assert_same_imgs("cyberdog_1.png", "cyberdog_3.png")
    assert_same_imgs("book_1.png", "book_3.png")

def assert_same_imgs(path_1, path_2):
    img_1 = Image.open(path_1)
    img_2 = Image.open(path_2)

    assert img_1.size == img_2.size

    # Check if the images are the same
    img_one_hash = hashlib.md5(img_1.tobytes()).hexdigest()
    img_two_hash = hashlib.md5(img_2.tobytes()).hexdigest()

    assert img_one_hash == img_two_hash
