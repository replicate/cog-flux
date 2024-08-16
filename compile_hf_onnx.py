import inspect
from typing import List, Optional, Union
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel, T5EncoderModel, CLIPTokenizer, T5Tokenizer
import torch
import numpy as np


def run_pipeline():
    max_length = 256

    clip_encoder = CLIPTextModel.from_pretrained("model-cache/clip/model", torch_dtype=torch.bfloat16)
    clip_tokenizer = CLIPTokenizer.from_pretrained("model-cache/clip/tokenizer", max_length=max_length)
    t5_model = T5EncoderModel.from_pretrained("model-cache/t5/model", torch_dtype=torch.bfloat16)
    t5_tokenizer = T5Tokenizer.from_pretrained("model-cache/t5/tokenizer", max_length=max_length)
    vae = AutoencoderKL.from_pretrained("model-cache/ae-huggingface", torch_dtype=torch.bfloat16)
    flux = FluxTransformer2DModel.from_pretrained("model-cache/schnell-huggingface", torch_dtype=torch.bfloat16, use_safetensors=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("model-cache/scheduler-huggingface")


    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=clip_encoder,
        tokenizer=clip_tokenizer,
        text_encoder_2=t5_model,
        tokenizer_2=t5_tokenizer,
        transformer=flux
    )

    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()

    out = pipe("a cool dog")
    for i, img in enumerate(out.images):
        img.save(f"img{i}.png")

# great, this works. 

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def get_latent_dims():
    max_length = 256

    clip_encoder = CLIPTextModel.from_pretrained("model-cache/clip/model", torch_dtype=torch.bfloat16)
    clip_tokenizer = CLIPTokenizer.from_pretrained("model-cache/clip/tokenizer", max_length=max_length)
    t5_model = T5EncoderModel.from_pretrained("model-cache/t5/model", torch_dtype=torch.bfloat16)
    t5_tokenizer = T5Tokenizer.from_pretrained("model-cache/t5/tokenizer", max_length=max_length)
    vae = AutoencoderKL.from_pretrained("model-cache/ae-huggingface", torch_dtype=torch.bfloat16)
    flux = FluxTransformer2DModel.from_pretrained("model-cache/schnell-huggingface", torch_dtype=torch.bfloat16, use_safetensors=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("model-cache/scheduler-huggingface")


    pipe = FluxPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=clip_encoder,
        tokenizer=clip_tokenizer,
        text_encoder_2=t5_model,
        tokenizer_2=t5_tokenizer,
        transformer=flux
    )

    pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()

    # out = pipe("a cool dog")
    # for i, img in enumerate(out.images):
    #     img.save(f"img{i}.png")

    height = 1024
    width = 1024

    # 1. Check inputs. Raise error if not correct
    prompt="A cool dog"
    prompt_2=None
    num_images_per_prompt=1
    max_sequence_length=256
    num_inference_steps=28

    batch_size = 1
    device = "cuda"

    lora_scale = None
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    generator = None

    # 4. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, latent_image_ids = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    # 5. Prepare timesteps
    timesteps=None
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    # num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    # self._num_timesteps = len(timesteps)

    # 6. Denoising loop

    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
    t = timesteps[0]
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

    # handle guidance
    if flux.config.guidance_embeds:
        guidance = torch.tensor([3.5], device=device)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    def print_it(name, tensor):
        print(name)
        if tensor is not None:
            print(tensor.shape)
        else:
            print("None")
    
    print_it("latents", latents) # batch, channels, height, width
    print_it("guidance", guidance) # singleton
    print_it("timestep", timestep) # also singleton, perhaps with batch 
    print_it("pooled_prompt_embeds", pooled_prompt_embeds) # batch, 768 (d_clip)
    print_it("propmt_embeds", prompt_embeds) # batch, seq_len, 4096 (d_t5)
    print_it("text_ids", text_ids) # batch, d_text, empty
    print_it("img_ids", latent_image_ids) # batch, channels, height, width. - but if so, why

# great - so they're the same as previously. can take this and go on from there. 
# latents
# torch.Size([1, 4096, 64])
# guidance
# None
# timestep
# torch.Size([1])
# pooled_prompt_embeds
# torch.Size([1, 768])
# propmt_embeds
# torch.Size([1, 256, 4096])
# text_ids
# torch.Size([1, 256, 3])
# img_ids
# torch.Size([1, 4096, 3])

def compile_onnx():
    flux = FluxTransformer2DModel.from_pretrained("model-cache/schnell-huggingface", torch_dtype=torch.bfloat16, use_safetensors=True)
    flux = flux.to("cuda")

    def get_rand_inputs(max_length=256, guidance=None):
        # second img and img_id dimension is dynamic (=h * w / 256), everything else is fixed 
        hidden_states = torch.randn((1, 4096, 64), dtype=torch.bfloat16, device="cuda")
        img_ids = torch.rand((1, 4096, 3), dtype=torch.float32, device="cuda")

        txt = torch.randn((1, max_length, 4096), dtype=torch.bfloat16, device="cuda")
        txt_ids = torch.rand((1, max_length, 3), dtype=torch.float32, device="cuda")

        vec = torch.randn((1, 768), dtype=torch.bfloat16, device="cuda")
        if guidance:
            guidance = torch.randn((1), dtype=torch.bfloat16, device="cuda") 
        timestep = torch.randn((1), dtype=torch.bfloat16, device="cuda")

        # don't have to be passed as dict, can also just do basic args in this order: 
        # img: Tensor,
        # img_ids: Tensor,
        # txt: Tensor,
        # txt_ids: Tensor,
        # timesteps: Tensor,
        # y: Tensor, - this is "vec"
        # guidance: Tensor | None = None,
        # might need to have hidden_states as an input argument, i.e. (hidden_states, {rest_of_dictionary}). TBD. 
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": txt,
            "pooled_projections": vec,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance
        }


    with torch.inference_mode():
        inputs = get_rand_inputs()

        input_names = ["hidden_states", "encoder_hidden_states", "pooled_projections", "timestep", "img_ids", "txt_ids", "guidance"]
        output_names = ["prediction"]
        dynamic_axes = {"hidden_states": {1: "h*w"}, "img_ids": {1: "h*w"}, "prediction": {1: "h*w"}}
        path = "./out/test.onnx"
        print(flux.device)
        for input in inputs.values():
            if input is not None:
                print(input.device)

        torch.onnx.export(flux,
                        inputs,
                        path,
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes
                        )
        
if __name__ == '__main__':
    compile_onnx()