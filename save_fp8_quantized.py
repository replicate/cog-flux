# right. so what should this do? 
# basically we need to load the model and quantize
# then persist it. 

import os
from predict import DevPredictor
from fp8.util import ModelVersion



def generate_dev_img(p, img_name='cool_dog_1234.png'):
    p.predict('a cool dog', '1:1', None, 0, 1, 28, 3, 1234, 'png', 100, True, True, "1")
    os.system(f'mv out-0.png {img_name}')


def save_dev_fp8():
    p = DevPredictor()
    p.base_setup("flux-dev", compile_fp8=True)

    from safetensors.torch import save_file

    fp8_weights_path = 'model-cache/dev-fp8'
    if not os.path.exists(fp8_weights_path):
        os.makedirs(fp8_weights_path)

    generate_dev_img(p)
    print("scale initialized: ", p.fp8_pipe.model.double_blocks[0].img_mod.lin.input_scale_initialized)
    sd = p.fp8_pipe.model.state_dict()
    to_trim = '_orig_mod.'
    sd_to_save = {k[len(to_trim):] : v for k, v in sd.items()}
    save_file(sd_to_save, fp8_weights_path + '/' + 'dev_fp8.safetensors')


def test_dev_fp8():
    p = DevPredictor()
    p.base_setup("flux-dev-fp8", compile_fp8=True)
    generate_dev_img(p, 'cool_dog_1234_loaded_from_compiled.png')

if __name__ == '__main__':
    save_dev_fp8()
    test_dev_fp8()