from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, Optional

import torch
from fp8.modules.autoencoder import AutoEncoder, AutoEncoderParams
from fp8.modules.conditioner import HFEmbedder
from fp8.modules.flux_model import Flux, FluxParams
from safetensors.torch import load_file as load_sft

try:
    from enum import StrEnum
except:
    from enum import Enum

    class StrEnum(str, Enum):
        pass


from pydantic import BaseModel, Field, validator
from loguru import logger


class ModelVersion(StrEnum):
    flux_dev = "flux-dev"
    flux_schnell = "flux-schnell"
    flux_krea_dev = "flux-krea-dev"
    flux_krea_dev_fp8 = "flux-krea-dev-fp8"


class QuantizationDtype(StrEnum):
    qfloat8 = "qfloat8"
    qint2 = "qint2"
    qint4 = "qint4"
    qint8 = "qint8"


class ModelSpec(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
    version: ModelVersion
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    text_enc_max_length: int = 512
    text_enc_path: str | None
    text_enc_device: str | torch.device | None = "cuda:0"
    ae_device: str | torch.device | None = "cuda:0"
    flux_device: str | torch.device | None = "cuda:0"
    flux_url: str | None
    flow_dtype: str = "float16"
    ae_dtype: str = "bfloat16"
    text_enc_dtype: str = "bfloat16"
    # unused / deprecated
    num_to_quant: Optional[int] = 20
    quantize_extras: bool = False
    compile_whole_model: bool = False
    compile_extras: bool = False
    compile_blocks: bool = False
    flow_quantization_dtype: Optional[QuantizationDtype] = QuantizationDtype.qfloat8
    text_enc_quantization_dtype: Optional[QuantizationDtype] = QuantizationDtype.qfloat8
    ae_quantization_dtype: Optional[QuantizationDtype] = None
    clip_quantization_dtype: Optional[QuantizationDtype] = None
    offload_text_encoder: bool = False
    offload_vae: bool = False
    offload_flow: bool = False
    prequantized_flow: bool = False

    # Improved precision via not quanitzing the modulation linear layers
    quantize_modulation: bool = True
    # Improved precision via not quanitzing the flow embedder layers
    quantize_flow_embedder_layers: bool = False

def load_models(config: ModelSpec) -> tuple[Flux, AutoEncoder, HFEmbedder, HFEmbedder]:
    flow = load_flow_model(config)
    ae = load_autoencoder(config)
    clip, t5 = load_text_encoders(config)
    return flow, ae, clip, t5


def parse_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        return torch.device("cuda:0")


def into_dtype(dtype: str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype}")


def into_device(device: str | torch.device | None) -> torch.device:
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    elif isinstance(device, int):
        return torch.device(f"cuda:{device}")
    else:
        return torch.device("cuda:0")


def load_config(
    name: ModelVersion = ModelVersion.flux_dev,
    flux_path: str | None = None,
    ae_path: str | None = None,
    text_enc_path: str | None = None,
    text_enc_device: str | torch.device | None = None,
    ae_device: str | torch.device | None = None,
    flux_device: str | torch.device | None = None,
    flow_dtype: str = "float16",
    ae_dtype: str = "bfloat16",
    text_enc_dtype: str = "bfloat16",
    num_to_quant: Optional[int] = 20,
    compile_extras: bool = False,
    compile_blocks: bool = False,
    offload_text_enc: bool = False,
    offload_ae: bool = False,
    offload_flow: bool = False,
    quant_text_enc: Optional[Literal["float8", "qint2", "qint4", "qint8"]] = None,
    quant_ae: bool = False,
    prequantized_flow: bool = False,
    quantize_modulation: bool = True,
    quantize_flow_embedder_layers: bool = False,
) -> ModelSpec:
    """
    Load a model configuration using the passed arguments.
    """
    text_enc_device = str(parse_device(text_enc_device))
    ae_device = str(parse_device(ae_device))
    flux_device = str(parse_device(flux_device))
    return ModelSpec(
        version=name,
        repo_id=(
            "black-forest-labs/FLUX.1-dev"
            if name == ModelVersion.flux_dev
            else "black-forest-labs/FLUX.1-schnell"
        ),
        repo_flow=(
            "flux1-dev.sft" if name == ModelVersion.flux_dev else "flux1-schnell.sft"
        ),
        repo_ae="ae.sft",
        ckpt_path=flux_path,
        params=FluxParams(
            in_channels=64,
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
            guidance_embed=name == ModelVersion.flux_dev,
        ),
        ae_path=ae_path,
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
        text_enc_path=text_enc_path,
        text_enc_device=text_enc_device,
        ae_device=ae_device,
        flux_device=flux_device,
        flow_dtype=flow_dtype,
        ae_dtype=ae_dtype,
        text_enc_dtype=text_enc_dtype,
        text_enc_max_length=512 if name == ModelVersion.flux_dev else 256,
        num_to_quant=num_to_quant,
        compile_extras=compile_extras,
        compile_blocks=compile_blocks,
        offload_flow=offload_flow,
        offload_text_encoder=offload_text_enc,
        offload_vae=offload_ae,
        text_enc_quantization_dtype={
            "float8": QuantizationDtype.qfloat8,
            "qint2": QuantizationDtype.qint2,
            "qint4": QuantizationDtype.qint4,
            "qint8": QuantizationDtype.qint8,
        }.get(quant_text_enc, None),
        ae_quantization_dtype=QuantizationDtype.qfloat8 if quant_ae else None,
        prequantized_flow=prequantized_flow,
        quantize_modulation=quantize_modulation,
        quantize_flow_embedder_layers=quantize_flow_embedder_layers,
    )


def load_config_from_path(path: str) -> ModelSpec:
    path_path = Path(path)
    if not path_path.exists():
        raise ValueError(f"Path {path} does not exist")
    if not path_path.is_file():
        raise ValueError(f"Path {path} is not a file")
    return ModelSpec(**json.loads(path_path.read_text()))


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        logger.warning("\n" + "-" * 79 + "\n")
        logger.warning(
            f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected)
        )
    elif len(missing) > 0:
        logger.warning(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        logger.warning(
            f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected)
        )
        
def download_weights(url: str, dest: Path):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    if url.endswith("tar"):
        subprocess.check_call(["pget", "--log-level=WARNING", "-x", url, dest], close_fds=False)
    else:
        subprocess.check_call(["pget", "--log-level=WARNING", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


def load_flow_model(config: ModelSpec) -> Flux:
    ckpt_path = config.ckpt_path
    if not os.path.exists(ckpt_path):
        flux_url = config.flux_url
        download_weights(flux_url, ckpt_path)

    FluxClass = Flux

    with torch.device("meta"):
        model = FluxClass(config, dtype=into_dtype(config.flow_dtype))
        if not config.prequantized_flow:
            model.type(into_dtype(config.flow_dtype))

    if ckpt_path is not None:
        # load_sft doesn't support torch.device
        sd = load_sft(ckpt_path, device="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
        if not config.prequantized_flow:
            model.type(into_dtype(config.flow_dtype))
    return model


def load_text_encoders(config: ModelSpec) -> tuple[HFEmbedder, HFEmbedder]:
    clip = HFEmbedder(
        "openai/clip-vit-large-patch14",
        max_length=77,
        torch_dtype=into_dtype(config.text_enc_dtype),
        device=into_device(config.text_enc_device).index or 0,
        quantization_dtype=config.clip_quantization_dtype,
    )
    t5 = HFEmbedder(
        config.text_enc_path,
        max_length=config.text_enc_max_length,
        torch_dtype=into_dtype(config.text_enc_dtype),
        device=into_device(config.text_enc_device).index or 0,
        quantization_dtype=config.text_enc_quantization_dtype,
    )
    return clip, t5


def load_autoencoder(config: ModelSpec) -> AutoEncoder:
    ckpt_path = config.ae_path
    with torch.device("meta" if ckpt_path is not None else config.ae_device):
        ae = AutoEncoder(config.ae_params).to(into_dtype(config.ae_dtype))

    if ckpt_path is not None:
        sd = load_sft(ckpt_path, device=str(config.ae_device))
        missing, unexpected = ae.load_state_dict(sd, strict=False, assign=True)
        print_load_warning(missing, unexpected)
    ae.to(device=into_device(config.ae_device), dtype=into_dtype(config.ae_dtype))
    if config.ae_quantization_dtype is not None:
        from fp8.float8_quantize import recursive_swap_linears

        recursive_swap_linears(ae)
    if config.offload_vae:
        ae.to("cpu")
        torch.cuda.empty_cache()
    return ae

@dataclass
class LoadedModels():
    flow: Optional[Flux]
    ae: Any
    clip: Any 
    t5: Any 
    config: Optional[ModelSpec]


def load_models_from_config_path(
    path: str,
) -> LoadedModels:
    config = load_config_from_path(path)
    clip, t5 = load_text_encoders(config)
    return LoadedModels(
        flow=load_flow_model(config),
        ae=load_autoencoder(config),
        clip=clip,
        t5=t5,
        config=config,
    )


def load_models_from_config(config: ModelSpec, shared_models: LoadedModels = None) -> LoadedModels:
    if shared_models:
        clip = shared_models.clip
        t5 = shared_models.t5
        ae = shared_models.ae
    else:
        clip, t5 = load_text_encoders(config)
        ae = load_autoencoder(config)
        
    flow = load_flow_model(config)

    return LoadedModels(
        flow=flow,
        ae=ae,
        clip=clip,
        t5=t5,
        config=config,
    )
