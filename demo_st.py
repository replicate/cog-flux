import os
import time
from glob import glob
from io import BytesIO

import streamlit as st
import torch
from einops import rearrange
from fire import Fire
from PIL import Image
from st_keyup import st_keyup

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5


@st.cache_resource()
def get_models(name: str, device: torch.device, offload: bool, quantize_flow: bool):
    t5 = load_t5(device)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device, quantize=quantize_flow)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


@torch.inference_mode()
def main(
    quantize_flow: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    offload: bool = False,
    output_dir: str = "output",
):
    torch_device = torch.device(device)
    names = list(configs.keys())
    name = st.selectbox("Which model to load?", names)
    if name is None or not st.checkbox("Load model", False):
        return

    model, ae, t5, clip = get_models(
        name,
        device=torch_device,
        offload=offload,
        quantize_flow=quantize_flow,
    )
    is_schnell = name == "flux-schnell"

    # allow for packing and conversion to latent space
    width = int(16 * (st.number_input("Width", min_value=128, max_value=8192, value=1024) // 16))
    height = int(16 * (st.number_input("Height", min_value=128, max_value=8192, value=1024) // 16))
    num_steps = int(st.number_input("Number of steps?", min_value=1, value=(4 if is_schnell else 50)))
    guidance = float(st.number_input("Guidance", min_value=1.0, value=3.5, disabled=is_schnell))
    seed = int(st.number_input("Seed (-1 to disable)", min_value=-1, value=-1, disabled=is_schnell))
    if seed == -1:
        seed = None
    save_samples = st.checkbox("Save samples?", not is_schnell)

    default_prompt = (
        "a photo of a forest with mist swirling around the tree trunks. The word "
        '"FLUX" is painted over it in big, red brush strokes with visible texture'
    )
    prompt = st_keyup("Enter a prompt", value=default_prompt, debounce=300, key="interactive_text")

    output_name = os.path.join(output_dir, "img_{idx}.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        idx = 0
    elif len(os.listdir(output_dir)) > 0:
        fns = glob(output_name.format(idx="*"))
        idx = max(int(fn.split("_")[-1].split(".")[0]) for fn in fns) + 1
    else:
        idx = 0

    rng = torch.Generator(device="cpu")

    if "seed" not in st.session_state:
        st.session_state.seed = rng.seed()

    def increment_counter():
        st.session_state.seed += 1

    def decrement_counter():
        if st.session_state.seed > 0:
            st.session_state.seed -= 1

    opts = SamplingOptions(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if name == "flux-schnell":
        cols = st.columns([5, 1, 1, 5])
        with cols[1]:
            st.button("↩", on_click=increment_counter)
        with cols[2]:
            st.button("↪", on_click=decrement_counter)
    if is_schnell or st.button("Sample"):
        if is_schnell:
            opts.seed = st.session_state.seed
        elif opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp = prepare(t5=t5, clip=clip, img=x, prompt=opts.prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(not is_schnell))

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        # denoise initial noise
        x = denoise(model, **inp, timesteps=timesteps, guidance=opts.guidance)

        # offload model, load autoencoder to gpu
        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = ae.decode(x)
        t1 = time.perf_counter()

        fn = output_name.format(idx=idx)
        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format and save
        x = rearrange(x[0], "c h w -> h w c").clamp(-1, 1)
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        if save_samples:
            print(f"Saving {fn}")
            img.save(fn)
            idx += 1

        st.session_state["samples"] = {
            "prompt": opts.prompt,
            "img": img,
            "seed": opts.seed,
        }
        opts.seed = None

    samples = st.session_state.get("samples", None)
    if samples is not None:
        st.image(samples["img"], caption=samples["prompt"])
        if "bytes" not in samples:
            img: Image.Image = samples["img"]
            buffer = BytesIO()
            img.save(buffer, format="png")
            samples["bytes"] = buffer.getvalue()
        st.download_button(
            "Download full-resolution",
            samples["bytes"],
            file_name="generated.png",
            mime="image/png",
        )
        st.write(f"Seed: {samples['seed']}")


def app():
    Fire(main)


if __name__ == "__main__":
    app()
