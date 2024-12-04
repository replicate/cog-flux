import os
from dataclasses import dataclass
import subprocess
import time

import torch
from safetensors.torch import load_file as load_sft

from flux.model import Flux, FluxParams
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from flux.modules.conditioner import HFEmbedder
from flux.modules.image_embedders import DepthImageEncoder, ReduxImageEncoder
from flux.modules.quantize import replace_linear_weight_only_int8_per_channel
from huggingface_hub import hf_hub_download
from pathlib import Path


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ckpt_url: str | None
    ae_path: str | None
    ae_url: str | None

T5_URL = "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar"
T5_CACHE = "./model-cache/t5"
CLIP_URL = "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar"
CLIP_CACHE = "./model-cache/clip"
SCHNELL_CACHE = "./model-cache/schnell/schnell.sft"
SCHNELL_URL = "https://weights.replicate.delivery/default/official-models/flux/schnell/schnell.sft"
DEV_CACHE = "./model-cache/dev/dev.sft"
DEV_URL = "https://weights.replicate.delivery/default/official-models/flux/dev/dev.sft"
DEV_CANNY_CACHE = "./model-cache/dev-canny/dev-canny.safetensors"
DEV_CANNY_URL = "https://weights.replicate.delivery/default/black-forest-labs/ctrl-n-fill/flux1-canny-dev.safetensors"
DEV_DEPTH_CACHE = "./model-cache/dev-depth/dev-depth.safetensors"
DEV_DEPTH_URL = "https://weights.replicate.delivery/default/black-forest-labs/ctrl-n-fill/flux1-depth-dev.safetensors"
DEV_INPAINTING_CACHE = "./model-cache/dev-inpainting/dev-inpainting.safetensors"
DEV_INPAINTING_URL = "https://weights.replicate.delivery/default/black-forest-labs/ctrl-n-fill/flux1-fill-dev.safetensors"
AE_CACHE = "./model-cache/ae/ae.sft"
AE_URL = "https://weights.replicate.delivery/default/official-models/flux/ae/ae.sft"
SIGLIP_URL = "https://weights.replicate.delivery/default/google/siglip-so400m-patch14-384/model-bf16.tar"
SIGLIP_CACHE = "./model-cache/siglip"
REDUX_URL = "https://weights.replicate.delivery/default/black-forest-labs/ctrl-n-fill/flux1-redux-dev.safetensors"
REDUX_CACHE = "./model-cache/redux/flux1-redux-dev.safetensors"
DEPTH_URL = "https://weights.replicate.delivery/default/liheyoung/depth-anything-large/flux-depth-model.tar"
DEPTH_CACHE = "./model-cache/depth"

configs = {
    "flux-dev": ModelSpec(
        ckpt_path=DEV_CACHE,
        ckpt_url=DEV_URL,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=AE_CACHE,
        ae_url=AE_URL,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": ModelSpec(
        ckpt_path=SCHNELL_CACHE,
        ckpt_url=SCHNELL_URL,
        params=FluxParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        ae_path=AE_CACHE,
        ae_url=AE_URL,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-canny-dev": ModelSpec(
        ckpt_path=DEV_CANNY_CACHE,
        ckpt_url=DEV_CANNY_URL,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=AE_CACHE,
        ae_url=AE_URL,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-depth-dev": ModelSpec(
        ckpt_path=DEV_DEPTH_CACHE,
        ckpt_url=DEV_DEPTH_URL,
        params=FluxParams(
            in_channels=128,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=AE_CACHE,
        ae_url=AE_URL,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-fill-dev": ModelSpec(
        ckpt_path=DEV_INPAINTING_CACHE,
        ckpt_url=DEV_INPAINTING_URL,
        params=FluxParams(
            in_channels=384,
            out_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=AE_CACHE,
        ae_url=AE_URL,
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


def load_flow_model(name: str, device: str | torch.device = "cuda", quantize: bool = False):
    # Loading Flux
    print("Init model")
    ckpt_path = configs[name].ckpt_path
    ckpt_url = configs[name].ckpt_url

    if not os.path.exists(ckpt_path):
        download_weights(ckpt_url, ckpt_path)

    with torch.device("meta" if ckpt_path is not None else device):
        model = Flux(configs[name].params).to(torch.bfloat16)

        if quantize:
            replace_linear_weight_only_int8_per_channel(model)

    if quantize and ckpt_path is not None:
        ckpt_path = Path(ckpt_path).stem + "_quantized.sft"
        print(f"Quantized checkpoint path: {ckpt_path}")

    print("Loading checkpoint")
    # load_sft doesn't support torch.device
    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(device))
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return model


def load_t5(device: str | torch.device = "cuda", max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    if not os.path.exists(T5_CACHE):
        download_weights(T5_URL, T5_CACHE)
    device = torch.device(device)
    return HFEmbedder(T5_CACHE, max_length=max_length, torch_dtype=torch.bfloat16).to(device)


def load_clip(device: str | torch.device = "cuda") -> HFEmbedder:
    if not os.path.exists(CLIP_CACHE):
        download_weights(CLIP_URL, CLIP_CACHE)
    device = torch.device(device)
    return HFEmbedder(CLIP_CACHE, max_length=77, is_clip=True, torch_dtype=torch.bfloat16).to(device)


def load_ae(name: str, device: str | torch.device = "cuda") -> AutoEncoder:
    # Loading the autoencoder
    print("Init AE")
    with torch.device("meta" if configs[name].ae_path is not None else device):
        ae = AutoEncoder(configs[name].ae_params)

    ae_path = configs[name].ae_path
    ae_url = configs[name].ae_url
    if not os.path.exists(ae_path):
        download_weights(ae_url, ae_path)

    if configs[name].ae_path is not None:
        sd = load_sft(configs[name].ae_path, device=str(device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    return ae

def load_redux(device: str | torch.device = "cuda") -> ReduxImageEncoder:
    if not os.path.exists(SIGLIP_CACHE):
        download_weights(SIGLIP_URL, SIGLIP_CACHE)
    if not os.path.exists(REDUX_CACHE):
        download_weights(REDUX_URL, REDUX_CACHE)

    return ReduxImageEncoder(device, redux_path = REDUX_CACHE, siglip_path=SIGLIP_CACHE, dtype=torch.bfloat16)

def load_depth_encoder(device: str | torch.device = "cuda") -> DepthImageEncoder:
    if not os.path.exists(DEPTH_CACHE):
        download_weights(DEPTH_URL, DEPTH_CACHE)

    return DepthImageEncoder(device, DEPTH_CACHE)


def download_ckpt_from_hf(
    repo_id: str,
    ckpt_name: str = "flux.safetensors",
    ae_name: str | None = None,
    **kwargs,
) -> tuple[Path, Path | None]:
    ckpt_path = hf_hub_download(repo_id, ckpt_name, **kwargs)
    ae_path = hf_hub_download(repo_id, ae_name, **kwargs) if ae_name else None
    return Path(ckpt_path).resolve(), Path(ae_path).resolve() if ae_path else None


def download_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if url.endswith("tar"):
        subprocess.check_call(["pget", "--log-level=WARNING", "-x", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", "--log-level=WARNING", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)
