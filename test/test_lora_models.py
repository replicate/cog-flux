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

    dog_one = Image.open("cyberdog_1.png")
    dog_two = Image.open("cyberdog_2.png")

    book_one = Image.open("book_1.png")
    book_two = Image.open("book_2.png")

    assert dog_one.size == dog_two.size
    assert book_one.size == book_two.size

    # Check if the images are the same
    # TODO - may need to replace with a fuzzy similarity check
    dog_one_hash = hashlib.md5(dog_one.tobytes()).hexdigest()
    dog_two_hash = hashlib.md5(dog_two.tobytes()).hexdigest()
    book_one_hash = hashlib.md5(book_one.tobytes()).hexdigest()
    book_two_hash = hashlib.md5(book_two.tobytes()).hexdigest()

    assert dog_one_hash == dog_two_hash
    assert book_one_hash == book_two_hash
