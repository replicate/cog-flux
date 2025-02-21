import pytest
from predict import DevLoraPredictor
from utils import get_params


@pytest.fixture
def uncompiled_dev_lora_predictor():
    p = DevLoraPredictor()
    p.setup(torch_compile=False)
    return p

@pytest.fixture
def dev_lora_predictor():
    p = DevLoraPredictor()
    p.setup()
    return p

def test_reload_scale(uncompiled_dev_lora_predictor):
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

    }

    text_encoder_lora_prediction = {

    }
    base_in = get_params(p, base_lora_prediction)
    te_in = get_params(p, text_encoder_lora_prediction)
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
