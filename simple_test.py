import torch_tensorrt
import torch
from diffusers import FluxPipeline


pipe = FluxPipeline.from_pretrained(
    # "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, device_map=None, enable_low_cpu_mem_usage=False
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
)
inputs = {
    "prompt": "dog",
    # fake, real is 1024x1024
    "height": 512,
    "width": 512,
    "num_inference_steps": 4,
    # "guidance_scale": 3.5,
}
z = False
if z:
    mod = torch.compile(pipe, backend="tensorrt")#, debug=True)
    output = mod(**inputs)
