import argparse
import os
from predict import DevPredictor, SchnellPredictor
from safetensors.torch import save_file

"""
Code to prequantize and save fp8 weights for Dev or Schnell. Pattern should work for other models.
Note - for this code to work, you'll need to tweak the config of the fp8 flux models in `predict.py` s.t. they load and quantize models.
in practice, this just means eliminating the '-fp8' suffix on the model names.
"""


def generate_dev_img(p, img_name="cool_dog_1234.png"):
    p.predict("a cool dog", "1:1", None, 0, 1, 28, 3, 1234, "png", 100, True, True, "1")
    os.system(f"mv out-0.png {img_name}")


def save_dev_fp8():
    p = DevPredictor()
    p.setup()

    fp8_weights_path = "model-cache/dev-fp8"
    if not os.path.exists(fp8_weights_path):  # noqa: PTH110
        os.makedirs(fp8_weights_path)  # noqa: PTH103

    generate_dev_img(p)
    print(
        "scale initialized: ",
        p.fp8_model.fp8_pipe.model.double_blocks[0].img_mod.lin.input_scale_initialized,
    )
    sd = p.fp8_model.fp8_pipe.model.state_dict()
    to_trim = "_orig_mod."
    sd_to_save = {k[len(to_trim) :]: v for k, v in sd.items()}
    save_file(sd_to_save, fp8_weights_path + "/" + "dev-fp8.safetensors")


def test_dev_fp8():
    p = DevPredictor()
    p.setup()
    generate_dev_img(p, "cool_dog_1234_loaded_from_compiled.png")


def generate_schnell_img(p, img_name="fast_dog_1234.png"):
    p.predict("a cool dog", "1:1", 1, 4, 1234, "png", 100, True, True, "1")
    os.system(f"mv out-0.png {img_name}")


def save_schnell_fp8():
    p = SchnellPredictor()
    p.setup()

    fp8_weights_path = "model-cache/schnell-fp8"
    if not os.path.exists(fp8_weights_path):  # noqa: PTH110
        os.makedirs(fp8_weights_path)  # noqa: PTH103

    generate_schnell_img(p)
    print(
        "scale initialized: ",
        p.fp8_model.fp8_pipe.model.double_blocks[0].img_mod.lin.input_scale_initialized,
    )
    sd = p.fp8_model.fp8_pipe.model.state_dict()
    to_trim = "_orig_mod."
    sd_to_save = {k[len(to_trim) :]: v for k, v in sd.items()}
    save_file(sd_to_save, fp8_weights_path + "/" + "schnell-fp8.safetensors")


def test_schnell_fp8():
    p = SchnellPredictor()
    p.setup()
    generate_schnell_img(p, "fast_dog_1234_loaded_from_compiled.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run image generation tests from YAML file"
    )
    parser.add_argument("flux_model", help="schnell, dev, or all")
    args = parser.parse_args()
    if args.flux_model == "dev" or args.flux_model == "all":
        save_dev_fp8()
    if args.flux_model == "schnell" or args.flux_model == "all":
        save_schnell_fp8()
    else:
        print("testing I guess")
        # test_dev_fp8()
        test_schnell_fp8()
