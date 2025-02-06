from pathlib import Path
import re
import time
import torch
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm

try:
    from cublas_ops import CublasLinear
except Exception as e:
    CublasLinear = type(None)
from fp8.float8_quantize import F8Linear
from fp8.modules.flux_model import Flux


class F8LinearClone:
    def __init__(self, module):
        self.float8_data = module.float8_data
        self.scale = module.scale
        self.input_scale = module.input_scale
        self.scale_reciprocal = module.scale_reciprocal
        self.input_scale_reciprocal = module.input_scale_reciprocal

class LinearClone: 
    def __init__(self, module):
        self.weight = module.weight.data


def swap_scale_shift(weight):
    scale, shift = weight.chunk(2, dim=0)
    new_weight = torch.cat([shift, scale], dim=0)
    return new_weight


def check_if_lora_exists(state_dict, lora_name):
    subkey = lora_name.split(".lora_A")[0].split(".lora_B")[0].split(".weight")[0]
    for key in state_dict.keys():
        if subkey in key:
            return subkey
    return False


def convert_if_lora_exists(new_state_dict, state_dict, lora_name, flux_layer_name):
    if (original_stubkey := check_if_lora_exists(state_dict, lora_name)) != False:
        weights_to_pop = [k for k in state_dict.keys() if original_stubkey in k]
        for key in weights_to_pop:
            key_replacement = key.replace(
                original_stubkey, flux_layer_name.replace(".weight", "")
            )
            new_state_dict[key_replacement] = state_dict.pop(key)
        return new_state_dict, state_dict
    else:
        return new_state_dict, state_dict


def convert_diffusers_to_flux_transformer_checkpoint(
    diffusers_state_dict,
    num_layers,
    num_single_layers,
    has_guidance=True,
    prefix="",
):
    original_state_dict = {}

    # time_text_embed.timestep_embedder -> time_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    )
    # time_text_embed.text_embedder -> vector_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_1.weight",
        "vector_in.in_layer.weight",
    )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_2.weight",
        "vector_in.out_layer.weight",
    )

    if has_guidance:
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.weight",
        )

    # context_embedder -> txt_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}context_embedder.weight",
        "txt_in.weight",
    )

    # x_embedder -> img_in
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}x_embedder.weight",
        "img_in.weight",
    )
    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        # norms
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1.linear.weight",
            f"double_blocks.{i}.img_mod.lin.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm1_context.linear.weight",
            f"double_blocks.{i}.txt_mod.lin.weight",
        )

        if f"{prefix}{block_prefix}attn.to_q.lora_A.weight" in diffusers_state_dict:
            sample_q_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_q.lora_A.weight"
            )
            sample_q_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_q.lora_B.weight"
            )

            sample_k_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_k.lora_A.weight"
            )
            sample_k_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_k.lora_B.weight"
            )

            sample_v_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_v.lora_A.weight"
            )
            sample_v_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.to_v.lora_B.weight"
            )

            original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_A.weight"] = (
                torch.cat([sample_q_A, sample_k_A, sample_v_A], dim=1)
            )
            original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_B.weight"] = (
                torch.cat([sample_q_B, sample_k_B, sample_v_B], dim=0)
            )

        if f"{prefix}{block_prefix}attn.add_q_proj.lora_A.weight" in diffusers_state_dict:

            context_q_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_q_proj.lora_A.weight"
            )
            context_q_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_q_proj.lora_B.weight"
            )

            context_k_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_k_proj.lora_A.weight"
            )
            context_k_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_k_proj.lora_B.weight"
            )
            context_v_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_v_proj.lora_A.weight"
            )
            context_v_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}attn.add_v_proj.lora_B.weight"
            )


            original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_A.weight"] = (
                torch.cat([context_q_A, context_k_A, context_v_A], dim=1)
            )
            original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_B.weight"] = (
                torch.cat([context_q_B, context_k_B, context_v_B], dim=0)
            )

        # qk_norm
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_q.weight",
            f"double_blocks.{i}.img_attn.norm.query_norm.scale",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_k.weight",
            f"double_blocks.{i}.img_attn.norm.key_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_q.weight",
            f"double_blocks.{i}.txt_attn.norm.query_norm.scale",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.norm_added_k.weight",
            f"double_blocks.{i}.txt_attn.norm.key_norm.scale",
        )

        # ff img_mlp

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.0.proj.weight",
            f"double_blocks.{i}.img_mlp.0.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff.net.2.weight",
            f"double_blocks.{i}.img_mlp.2.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.0.proj.weight",
            f"double_blocks.{i}.txt_mlp.0.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}ff_context.net.2.weight",
            f"double_blocks.{i}.txt_mlp.2.weight",
        )
        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_out.0.weight",
            f"double_blocks.{i}.img_attn.proj.weight",
        )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}attn.to_add_out.weight",
            f"double_blocks.{i}.txt_attn.proj.weight",
        )

    # single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        # norm.linear -> single_blocks.0.modulation.lin
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}norm.linear.weight",
            f"single_blocks.{i}.modulation.lin.weight",
        )

        # Q, K, V, mlp - note that we don't support only tuning a subset of Q, K, V at the moment. 
        if f"{prefix}{block_prefix}attn.to_q.lora_A.weight" in diffusers_state_dict:
            q_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
            q_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
            k_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
            k_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
            v_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
            v_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")

            # some loras don't tune mlp_A or mlp_B - default value makes loading these a no-op. 
            rank = q_A.shape[0]
            mlp_A = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}proj_mlp.lora_A.weight",
                torch.zeros(rank, 3072, device=q_A.device, dtype=q_A.dtype)
            )
            mlp_B = diffusers_state_dict.pop(
                f"{prefix}{block_prefix}proj_mlp.lora_B.weight",
                torch.zeros(12288, rank, device=q_A.device, dtype=q_A.dtype)
            )
            original_state_dict[f"single_blocks.{i}.linear1.lora_A.weight"] = torch.cat(
                [q_A, k_A, v_A, mlp_A], dim=1
            )
            original_state_dict[f"single_blocks.{i}.linear1.lora_B.weight"] = torch.cat(
                [q_B, k_B, v_B, mlp_B], dim=0
            )

        # output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}{block_prefix}proj_out.weight",
            f"single_blocks.{i}.linear2.weight",
        )

    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.weight",
        "final_layer.linear.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.bias",
        "final_layer.linear.bias",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    )
    if len(list(diffusers_state_dict.keys())) > 0:
        logger.warning("Unexpected keys:", diffusers_state_dict.keys())

    return original_state_dict


def convert_from_original_flux_checkpoint(
    original_state_dict,
):
    sd = {
        k.replace("lora_unet_", "")
        .replace("double_blocks_", "double_blocks.")
        .replace("single_blocks_", "single_blocks.")
        .replace("_img_attn_", ".img_attn.")
        .replace("_txt_attn_", ".txt_attn.")
        .replace("_img_mod_", ".img_mod.")
        .replace("_txt_mod_", ".txt_mod.")
        .replace("_img_mlp_", ".img_mlp.")
        .replace("_txt_mlp_", ".txt_mlp.")
        .replace("_linear1", ".linear1")
        .replace("_linear2", ".linear2")
        .replace("_modulation_", ".modulation.")
        .replace("lora_up", "lora_B")
        .replace("lora_down", "lora_A"): v
        for k, v in original_state_dict.items()
        if "lora" in k and "_te1" not in k
    }
    if len(original_state_dict) != len(sd):
        print("Warning - loading loras that fine-tune the text encoder is not supported at present, text encoder weights will be ignored")
    return sd


def get_module_for_key(
    key: str, model: Flux
) -> F8Linear | torch.nn.Linear | CublasLinear:
    parts = key.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def get_lora_for_key(
    key: str, lora_weights: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = key.split(".lora")[0]
    lora_A = lora_weights[f"{prefix}.lora_A.weight"]
    lora_B = lora_weights[f"{prefix}.lora_B.weight"]
    alpha = lora_weights.get(f"{prefix}.alpha", None)
    return lora_A, lora_B, alpha


@torch.inference_mode()
def apply_linear1_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int | None = None,
    lora_scale: float = 1.0,
):
    lora_A, lora_B, alpha = lora_weights

    if rank is None:
        rank = lora_B.shape[1]
    else:
        rank = rank
    if alpha is None:
        alpha = rank
    else:
        alpha = alpha
    w_orig = module_weight
    w_up = lora_A  
    w_down = lora_B  

    if alpha != rank:
        w_up = w_up * alpha / rank

    q = w_orig[:3072]
    k = w_orig[3072 : 3072 * 2]
    v = w_orig[3072 * 2 : 3072 * 3]
    mlp = w_orig[3072 * 3 :]
    q_up = w_up[:, :3072]
    k_up = w_up[:, 3072 : 3072 * 2]
    v_up = w_up[:, 3072 * 2 : 3072 * 3]
    mlp_up = w_up[:, 3072 * 3 :]
    q_down = w_down[:3072]
    k_down = w_down[3072 : 3072 * 2]
    v_down = w_down[3072 * 2 : 3072 * 3]
    mlp_down = w_down[3072 * 3 :]

    q = (q.float() + lora_scale * torch.mm(q_down, q_up)).to(torch.bfloat16)
    k = (k.float() + lora_scale * torch.mm(k_down, k_up)).to(torch.bfloat16)
    v = (v.float() + lora_scale * torch.mm(v_down, v_up)).to(torch.bfloat16)
    mlp = (mlp.float() + lora_scale * torch.mm(mlp_down, mlp_up)).to(torch.bfloat16)

    fused_weight = torch.cat([q, k, v, mlp], dim=0)
    return fused_weight  


@torch.inference_mode()
def apply_attn_qkv_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int | None = None,
    lora_scale: float = 1.0,
):
    lora_A, lora_B, alpha = lora_weights

    if rank is None:
        rank = lora_B.shape[1]
    else:
        rank = rank
    if alpha is None:
        alpha = rank
    else:
        alpha = alpha
    w_orig = module_weight
    w_up = lora_A  
    w_down = lora_B  

    if alpha != rank:
        w_up = w_up * alpha / rank

    q = w_orig[:3072]
    k = w_orig[3072 : 3072 * 2]
    v = w_orig[3072 * 2 : 3072 * 3]
    q_up = w_up[:, :3072]
    k_up = w_up[:, 3072 : 3072 * 2]
    v_up = w_up[:, 3072 * 2 : 3072 * 3]
    q_down = w_down[:3072]
    k_down = w_down[3072 : 3072 * 2]
    v_down = w_down[3072 * 2 : 3072 * 3]

    q = (q.float() + lora_scale * torch.mm(q_down, q_up)).to(torch.bfloat16)
    k = (k.float() + lora_scale * torch.mm(k_down, k_up)).to(torch.bfloat16)
    v = (v.float() + lora_scale * torch.mm(v_down, v_up)).to(torch.bfloat16)

    fused_weight = torch.cat([q, k, v], dim=0)
    return fused_weight 


@torch.inference_mode()
def apply_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: int | None = None,
    lora_scale: float = 1.0,
):
    lora_A, lora_B, alpha = lora_weights

    uneven_rank = lora_B.shape[1] != lora_A.shape[0]
    rank_diff = lora_A.shape[0] / lora_B.shape[1]

    if rank is None:
        rank = lora_B.shape[1]
    else:
        rank = rank
    if alpha is None:
        alpha = rank
    else:
        alpha = alpha

    w_orig = module_weight
    w_up = lora_A  
    w_down = lora_B  

    if alpha != rank:
        w_up = w_up * alpha / rank
    if uneven_rank:
        w_down = w_down.repeat_interleave(int(rank_diff), dim=1)

    fused_lora = lora_scale * torch.mm(w_down, w_up)
    fused_weight = (w_orig.float() + fused_lora).to(torch.bfloat16)
    return fused_weight  

def convert_lora_weights(lora_path: str | Path, has_guidance: bool):
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights = load_file(lora_path, device="cuda")
    is_kohya = any(".lora_down.weight" in k for k in lora_weights)

    # converting to diffusers to convert from diffusers is a bit circuitous at the moment but it works 
    if is_kohya:
        lora_weights = _convert_kohya_flux_lora_to_diffusers(lora_weights)

    is_xlabs = any("processor" in k for k in lora_weights)
    if is_xlabs:
        lora_weights = _convert_xlabs_flux_lora_to_diffusers(lora_weights)

    check_if_starts_with_transformer = [
        k for k in lora_weights.keys() if k.startswith("transformer.")
    ]
    if len(check_if_starts_with_transformer) > 0:
        lora_weights = convert_diffusers_to_flux_transformer_checkpoint(
            lora_weights, 19, 38, has_guidance=has_guidance, prefix="transformer."
        )
    else:
        lora_weights = convert_from_original_flux_checkpoint(lora_weights)
    logger.info("LoRA weights loaded")
    return lora_weights


@torch.inference_mode()
def load_loras(model: Flux, lora_paths: list[str] | list[Path], lora_scales: list[float], store_clones: bool = False):
    for lora, scale in zip(lora_paths, lora_scales):
        load_lora(model, lora, scale, store_clones) 


@torch.inference_mode()
def load_lora(model: Flux, lora_path: str | Path, lora_scale: float = 1.0, store_clones: bool = False):
    """
    Loads lora weights. 
    
    If store_clones is set, stores backup copy of base weights in vram, and simply overwrites merged lora weights w/backup copy on unmerge.
    """
    t = time.time()
    has_guidance = model.params.guidance_embed

    lora_weights = convert_lora_weights(lora_path, has_guidance)

    apply_lora_to_model_and_optionally_store_clones(model, lora_weights, lora_scale, store_clones)

    logger.success(f"LoRA applied in {time.time() - t:.2}s")

    if hasattr(model, "lora_weights"):
        model.lora_weights.append((lora_weights, lora_scale))
    else:
        model.lora_weights = [(lora_weights, lora_scale)]


@torch.inference_mode()
def unload_loras(model: Flux):
    """
    Unmerges or overwrites lora weights depending on how loras have been stored. 
    """
    t = time.time()
    if not hasattr(model, "lora_weights") or not model.lora_weights:
        return

    if hasattr(model, "clones") and len(model.clones) > 0:
        # restore original weights from backup copy in vram
        restore_base_weights(model)
    else:
        for lora_weights, lora_scale in model.lora_weights[::-1]:
            # subtract lora from merged weights
            apply_lora_to_model_and_optionally_store_clones(model, lora_weights, -lora_scale, False)
    logger.success(f"LoRAs unloaded in {time.time() - t:.2}s")

    model.lora_weights = []


def restore_base_weights(model: Flux):    
    # Get all unique keys that had LoRA weights applied
    expected_keys = set()
    for lora_weights, _ in model.lora_weights:
        keys = set(key.replace(".lora_A.weight", "")
                     .replace(".lora_B.weight", "")
                     .replace(".alpha", "")
                  for key in lora_weights.keys())
        expected_keys.update(keys)
    # Check we have clones for all modules that had LoRA weights
    assert expected_keys == set(model.clones.keys()), \
           f"Mismatch between LoRA keys and weight clones. Missing: {expected_keys - set(model.clones.keys())}"
    
    for key, clone in model.clones.items():
        module = get_module_for_key(key, model)
        if isinstance(module, F8Linear):
            module.float8_data = clone.float8_data
            module.scale = clone.scale
            module.input_scale = clone.input_scale
            module.scale_reciprocal = clone.scale_reciprocal
            module.input_scale_reciprocal = clone.input_scale_reciprocal
        else:
            module.weight.data = clone.weight
    
    logger.info(f"Unloaded {len(model.clones.keys())} layers")

    model.lora_weights = []
    model.clones = {}
    return

@torch.inference_mode()
def apply_lora_to_model_and_optionally_store_clones(model: Flux, lora_weights: dict, lora_scale: float = 1.0, store_clones: bool = False):
    if store_clones and not hasattr(model, "clones"):
        model.clones = {}
    logger.debug("Extracting keys")
    keys_without_ab = [
        key.replace(".lora_A.weight", "")
        .replace(".lora_B.weight", "")
        .replace(".alpha", "")
        for key in lora_weights.keys()
    ]
    logger.debug("Keys extracted")

    keys_without_ab = list(set(keys_without_ab))

    for key in tqdm(keys_without_ab, desc="Applying LoRA", total=len(keys_without_ab)):
        module = get_module_for_key(key, model)

        weight_is_f8 = isinstance(module, F8Linear)
        if weight_is_f8:
            if store_clones and key not in model.clones:
                model.clones[key] = F8LinearClone(module)
            weight_f16 = (
                module.float8_data.clone()
                .float()
                .mul(module.scale_reciprocal)
                .to(module.weight.device)
                .to(torch.bfloat16)
            )
        elif isinstance(module, torch.nn.Linear) or isinstance(module, CublasLinear):
            if store_clones and key not in model.clones:
                model.clones[key] = LinearClone(module)
            weight_f16 = module.weight.clone()
        else:
            raise Exception(f"Trying to load lora on module {key} that is not a linear layer")
        
        lora_sd = get_lora_for_key(key, lora_weights)

        assert weight_f16.dtype == torch.bfloat16, f"{key} is {weight_f16.dtype}, not torch.bfloat16"

        if ".linear1" in key:
            weight_f16 = apply_linear1_lora_weight_to_module(
                weight_f16, lora_sd, lora_scale=lora_scale
            )

        elif "_attn.qkv" in key:
            weight_f16 = apply_attn_qkv_lora_weight_to_module(
                weight_f16, lora_sd, lora_scale=lora_scale
            )

        else:
            weight_f16 = apply_lora_weight_to_module(
                weight_f16, lora_sd, lora_scale=lora_scale
            )

        assert weight_f16.dtype == torch.bfloat16, f"{key} is {weight_f16.dtype} after applying lora, not torch.bfloat16"

        if weight_is_f8:
            module.set_weight_tensor(weight_f16)
        else:
            module.weight.data = weight_f16

    if weight_is_f8:
        logger.info("Loading LoRA in fp8")
    else:
        logger.info("Loading LoRA in bf16")

    return

# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are taken from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
# All credits go to `kohya-ss`.
def _convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_A.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_B.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        alpha = sds_sd.pop(sds_key + ".alpha")
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0
                    )
                i += dims[j]
            if is_sparse:
                logger.info(f"weight is sparse: {sds_key}")

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update({k: down_weight for k in ait_down_keys})

            # up_weight is split to each split
            ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416
        else:
            # down_weight is chunked to each split
            ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})  # noqa: C416

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        remaining_keys = list(sds_sd.keys())
        te_state_dict = {}
        if remaining_keys:
            if not all(k.startswith("lora_te1") for k in remaining_keys):
                raise ValueError(f"Incompatible keys detected: \n\n {', '.join(remaining_keys)}")
            for key in remaining_keys:
                if not key.endswith("lora_down.weight"):
                    continue

                lora_name = key.split(".")[0]
                lora_name_up = f"{lora_name}.lora_up.weight"
                lora_name_alpha = f"{lora_name}.alpha"
                diffusers_name = _convert_text_encoder_lora_key(key, lora_name)

                if lora_name.startswith(("lora_te_", "lora_te1_")):
                    down_weight = sds_sd.pop(key)
                    sd_lora_rank = down_weight.shape[0]
                    te_state_dict[diffusers_name] = down_weight
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = sds_sd.pop(lora_name_up)

                if lora_name_alpha in sds_sd:
                    alpha = sds_sd.pop(lora_name_alpha).item()
                    scale = alpha / sd_lora_rank

                    scale_down = scale
                    scale_up = 1.0
                    while scale_down * 2 < scale_up:
                        scale_down *= 2
                        scale_up /= 2

                    te_state_dict[diffusers_name] *= scale_down
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] *= scale_up

        if len(sds_sd) > 0:
            logger.warning(f"Unsupported keys for ai-toolkit: {sds_sd.keys()}")

        if te_state_dict:
            te_state_dict = {f"text_encoder.{module_name}": params for module_name, params in te_state_dict.items()}

        new_state_dict = {**ait_sd, **te_state_dict}
        return new_state_dict

    return _convert_sd_scripts_to_ai_toolkit(state_dict)


def _convert_text_encoder_lora_key(key, lora_name):
    """
    Converts a text encoder LoRA key to a Diffusers compatible key.
    """
    if lora_name.startswith(("lora_te_", "lora_te1_")):
        key_to_replace = "lora_te_" if lora_name.startswith("lora_te_") else "lora_te1_"
    else:
        key_to_replace = "lora_te2_"

    diffusers_name = key.replace(key_to_replace, "").replace("_", ".")
    diffusers_name = diffusers_name.replace("text.model", "text_model")
    diffusers_name = diffusers_name.replace("self.attn", "self_attn")
    diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
    diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
    diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
    diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
    diffusers_name = diffusers_name.replace("text.projection", "text_projection")

    if "self_attn" in diffusers_name or "text_projection" in diffusers_name:
        pass
    elif "mlp" in diffusers_name:
        # Be aware that this is the new diffusers convention and the rest of the code might
        # not utilize it yet.
        diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
    return diffusers_name

# Adapted from https://gist.github.com/Leommm-byte/6b331a1e9bd53271210b26543a7065d6
# Some utilities were reused from
# https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
def _convert_xlabs_flux_lora_to_diffusers(old_state_dict):
    new_state_dict = {}
    orig_keys = list(old_state_dict.keys())

    def handle_qkv(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        down_weight = sds_sd.pop(sds_key)
        up_weight = sds_sd.pop(sds_key.replace(".down.weight", ".up.weight"))

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_A.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_B.weight" for k in ait_keys]

        # down_weight is copied to each split
        ait_sd.update({k: down_weight for k in ait_down_keys})

        # up_weight is split to each split
        ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416

    for old_key in orig_keys:
        # Handle double_blocks
        if old_key.startswith(("diffusion_model.double_blocks", "double_blocks")):
            block_num = re.search(r"double_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.transformer_blocks.{block_num}"

            if "processor.proj_lora1" in old_key:
                new_key += ".attn.to_out.0"
            elif "processor.proj_lora2" in old_key:
                new_key += ".attn.to_add_out"
            # Handle text latents.
            elif "processor.qkv_lora2" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.add_q_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_k_proj",
                        f"transformer.transformer_blocks.{block_num}.attn.add_v_proj",
                    ],
                )
                # continue
            # Handle image latents.
            elif "processor.qkv_lora1" in old_key and "up" not in old_key:
                handle_qkv(
                    old_state_dict,
                    new_state_dict,
                    old_key,
                    [
                        f"transformer.transformer_blocks.{block_num}.attn.to_q",
                        f"transformer.transformer_blocks.{block_num}.attn.to_k",
                        f"transformer.transformer_blocks.{block_num}.attn.to_v",
                    ],
                )
                # continue

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        # Handle single_blocks
        elif old_key.startswith(("diffusion_model.single_blocks", "single_blocks")):
            block_num = re.search(r"single_blocks\.(\d+)", old_key).group(1)
            new_key = f"transformer.single_transformer_blocks.{block_num}"

            if "proj_lora1" in old_key or "proj_lora2" in old_key:
                new_key += ".proj_out"
            elif "qkv_lora1" in old_key or "qkv_lora2" in old_key:
                new_key += ".norm.linear"

            if "down" in old_key:
                new_key += ".lora_A.weight"
            elif "up" in old_key:
                new_key += ".lora_B.weight"

        else:
            # Handle other potential key patterns here
            new_key = old_key

        # Since we already handle qkv above.
        if "qkv" not in old_key:
            new_state_dict[new_key] = old_state_dict.pop(old_key)

    if len(old_state_dict) > 0:
        raise ValueError(f"`old_state_dict` should be at this point but has: {list(old_state_dict.keys())}.")

    return new_state_dict
