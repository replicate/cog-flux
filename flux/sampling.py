import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor
from tqdm.auto import tqdm

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(
    t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]
) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise_single_item(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    compile_run: bool = False,
):
    img = img.unsqueeze(0)
    img_ids = img_ids.unsqueeze(0)
    txt = txt.unsqueeze(0)
    txt_ids = txt_ids.unsqueeze(0)
    vec = vec.unsqueeze(0)
    guidance_vec = torch.full((1,), guidance, device=img.device, dtype=img.dtype)



    if compile_run:
        import torch_tensorrt
        from torch_tensorrt import Input
        bf16 = torch.bfloat16
        f32 = torch.float32
        inputs = dict(
            img=Input(
                min_shape=(1, 3808, 64), opt_shape=(1, 4096, 64), max_shape=(1, 4096, 64), dtype=bf16
            ),
            img_ids=Input(
                min_shape=(1, 3808, 3), opt_shape=(1, 4096, 3), max_shape=(1, 4096, 3), dtype=f32
            ),
            txt=Input((1, 512, 4096), dtype=bf16),
            txt_ids=Input((1, 512, 3), dtype=f32),
            timesteps=Input((1,), dtype=bf16),
            y=Input((1, 768), dtype=bf16),
        )
        # compile checks for (arg_inputs or inputs) and doesn't allow passing all inputs via kw_inputs
        img_input = inputs.pop("img")

        compile_run = False
        # torch._dynamo.mark_dynamic(img, 1, min=256, max=8100)  # needs at least torch 2.4
        # torch._dynamo.mark_dynamic(img_ids, 1, min=256, max=8100)
        # options = {"timing_cache_path": "."}
        # input = [img, img_ids, txt, txt_ids, t_vec, vec, guidance_vec]
        model = torch_tensorrt.compile(model, ir="dynamo", arg_inputs=[img_input], kwarg_inputs=inputs, debug=True)
        torch_tensorrt.save(model, "flux-trt.ep", arg_inputs=[img_input], kwarg_inputs=inputs)

    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((1,), t_curr, dtype=img.dtype, device=img.device)

        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=t_vec,
            y=vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred.squeeze(0)

    return img, model


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    compile_run: bool = False,
):
    batch_size = img.shape[0]
    output_imgs = []

    for i in range(batch_size):
        denoised_img, model = denoise_single_item(
            model,
            img[i],
            img_ids[i],
            txt[i],
            txt_ids[i],
            vec[i],
            timesteps,
            guidance,
            compile_run,
        )
        compile_run = False
        output_imgs.append(denoised_img)

    return torch.cat(output_imgs), model


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
