from pathlib import Path
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

        original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_A.weight"] = (
            torch.cat([sample_q_A, sample_k_A, sample_v_A], dim=1)
        )
        original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_B.weight"] = (
            torch.cat([sample_q_B, sample_k_B, sample_v_B], dim=0)
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

        # Q, K, V, mlp
        q_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_A.weight")
        q_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_q.lora_B.weight")
        k_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_A.weight")
        k_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_k.lora_B.weight")
        v_A = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_A.weight")
        v_B = diffusers_state_dict.pop(f"{prefix}{block_prefix}attn.to_v.lora_B.weight")
        mlp_A = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_A.weight"
        )
        mlp_B = diffusers_state_dict.pop(
            f"{prefix}{block_prefix}proj_mlp.lora_B.weight"
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
        if "lora" in k
    }
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
    w_dtype = module_weight.dtype
    device = module_weight.device
    w_orig = module_weight
    w_up = lora_A  # .to(dtype=w_dtype, device=device)
    w_down = lora_B  # .to(dtype=w_dtype, device=device)

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

    # q += lora_scale * torch.mm(q_down, q_up).to(torch.bfloat16)
    # k += lora_scale * torch.mm(k_down, k_up).to(torch.bfloat16)
    # v += lora_scale * torch.mm(v_down, v_up).to(torch.bfloat16)
    # mlp += lora_scale * torch.mm(mlp_down, mlp_up).to(torch.bfloat16)
    q = (q.float() + lora_scale * torch.mm(q_down, q_up)).to(torch.bfloat16)
    k = (k.float() + lora_scale * torch.mm(k_down, k_up)).to(torch.bfloat16)
    v = (v.float() + lora_scale * torch.mm(v_down, v_up)).to(torch.bfloat16)
    mlp = (mlp.float() + lora_scale * torch.mm(mlp_down, mlp_up)).to(torch.bfloat16)

    fused_weight = torch.cat([q, k, v, mlp], dim=0)
    return fused_weight  # .to(dtype=w_dtype, device=device)


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
    w_dtype = module_weight.dtype
    device = module_weight.device
    w_orig = module_weight
    w_up = lora_A  # .to(dtype=w_dtype, device=device)
    w_down = lora_B  # .to(dtype=w_dtype, device=device)

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

    # q += lora_scale * torch.mm(q_down, q_up).to(torch.bfloat16)
    # k += lora_scale * torch.mm(k_down, k_up).to(torch.bfloat16)
    # v += lora_scale * torch.mm(v_down, v_up).to(torch.bfloat16)
    q = (q.float() + lora_scale * torch.mm(q_down, q_up)).to(torch.bfloat16)
    k = (k.float() + lora_scale * torch.mm(k_down, k_up)).to(torch.bfloat16)
    v = (v.float() + lora_scale * torch.mm(v_down, v_up)).to(torch.bfloat16)

    fused_weight = torch.cat([q, k, v], dim=0)
    return fused_weight  # .to(dtype=w_dtype, device=device)


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
    w_up = lora_A  # .to(dtype=w_dtype, device=device)
    w_down = lora_B  # .to(dtype=w_dtype, device=device)

    if alpha != rank:
        w_up = w_up * alpha / rank
    if uneven_rank:
        w_down = w_down.repeat_interleave(int(rank_diff), dim=1)
    #fused_lora = lora_scale * torch.mm(w_down, w_up).to(torch.bfloat16)
    #fused_weight = w_orig + fused_lora
    fused_lora = lora_scale * torch.mm(w_down, w_up)
    fused_weight = (w_orig.float() + fused_lora).to(torch.bfloat16)
    return fused_weight  # .to(dtype=w_dtype, device=device)


@torch.inference_mode()
def load_lora(model: Flux, lora_path: str | Path, lora_scale: float = 1.0):
    t = time.time()
    has_guidance = model.params.guidance_embed
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights = load_file(lora_path, device="cuda")
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

    apply_lora_to_model(model, lora_weights, lora_scale)
    logger.success(f"LoRA applied in {time.time() - t:.2}s")

    if hasattr(model, "lora_weights"):
        model.lora_weights.append((lora_weights, lora_scale))
    else:
        model.lora_weights = [(lora_weights, lora_scale)]


@torch.inference_mode()
def unload_loras(model: Flux):
    t = time.time()
    if not hasattr(model, "lora_weights") or not model.lora_weights:
        return

    for lora_weights, lora_scale in model.lora_weights[::-1]:
        apply_lora_to_model(model, lora_weights, -lora_scale)
    logger.success(f"LoRAs unloaded in {time.time() - t:.2}s")

    model.lora_weights = []


@torch.inference_mode()
def apply_lora_to_model(model: Flux, lora_weights: dict, lora_scale: float = 1.0):
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
        dtype = model.dtype if hasattr(model, "dtype") else torch.bfloat16

        weight_is_f8 = False
        if isinstance(module, F8Linear):
            weight_is_f8 = True
            weight_f16 = (
                module.float8_data.clone()
                # .to(torch.bfloat16)
                .float()
                .mul(module.scale_reciprocal)
                .to(module.weight.device)
                .to(torch.bfloat16)
            )
        elif isinstance(module, torch.nn.Linear):
            weight_f16 = module.weight.clone()  # .detach()
        elif isinstance(module, CublasLinear):
            weight_f16 = module.weight.clone()  # .detach()
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
