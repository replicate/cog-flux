from collections import namedtuple
import os
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from fp8.util import ModelSpec

DISABLE_COMPILE = os.getenv("DISABLE_COMPILE", "0") == "1"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20
torch.set_float32_matmul_precision("high")
import math

from torch import Tensor, nn
from pydantic import BaseModel
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class FluxParams(BaseModel):
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


# attention is always same shape each time it's called per H*W, so compile with fullgraph
# @torch.compile(mode="reduce-overhead", fullgraph=True, disable=DISABLE_COMPILE)
def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    x = x.reshape(*x.shape[:-2], -1)
    return x


# @torch.compile(mode="reduce-overhead", disable=DISABLE_COMPILE)
def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape), xk_out.reshape(*xk.shape)


class EmbedND(nn.Module):
    def __init__(
        self,
        dim: int,
        theta: int,
        axes_dim: list[int],
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.dtype = dtype

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [
                rope(ids[..., i], self.axes_dim[i], self.theta).type(self.dtype)
                for i in range(n_axes)
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, prequantized: bool = False, quantized=False
    ):
        from fp8.float8_quantize import F8Linear

        super().__init__()
        self.in_layer = (
            nn.Linear(in_dim, hidden_dim, bias=True)
            if not prequantized
            else (
                F8Linear(
                    in_features=in_dim,
                    out_features=hidden_dim,
                    bias=True,
                )
                if quantized
                else nn.Linear(in_dim, hidden_dim, bias=True)
            )
        )
        self.silu = nn.SiLU()
        self.out_layer = (
            nn.Linear(hidden_dim, hidden_dim, bias=True)
            if not prequantized
            else (
                F8Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    bias=True,
                )
                if quantized
                else nn.Linear(hidden_dim, hidden_dim, bias=True)
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        #return F.rms_norm(x, self.scale.shape, self.scale, eps=1e-6)
        # todo: this may bork things
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        prequantized: bool = False,
    ):
        super().__init__()
        from fp8.float8_quantize import F8Linear

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = (
            nn.Linear(dim, dim * 3, bias=qkv_bias)
            if not prequantized
            else F8Linear(
                in_features=dim,
                out_features=dim * 3,
                bias=qkv_bias,
            )
        )
        self.norm = QKNorm(head_dim)
        self.proj = (
            nn.Linear(dim, dim)
            if not prequantized
            else F8Linear(
                in_features=dim,
                out_features=dim,
                bias=True,
            )
        )
        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H

    def rearrange_for_norm(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, D = x.shape
        q, k, v = x.reshape(B, L, self.K, self.H, D // self.KH).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = self.rearrange_for_norm(qkv)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


ModulationOut = namedtuple("ModulationOut", ["shift", "scale", "gate"])


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, quantized_modulation: bool = False):
        super().__init__()
        from fp8.float8_quantize import F8Linear

        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = (
            nn.Linear(dim, self.multiplier * dim, bias=True)
            if not quantized_modulation
            else F8Linear(
                in_features=dim,
                out_features=self.multiplier * dim,
                bias=True,
            )
        )
        self.act = nn.SiLU()

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(self.act(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = False,
        dtype: torch.dtype = torch.float16,
        quantized_modulation: bool = False,
        prequantized: bool = False,
    ):
        super().__init__()
        from fp8.float8_quantize import F8Linear

        self.dtype = dtype

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(
            hidden_size, double=True, quantized_modulation=quantized_modulation
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            prequantized=prequantized,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            (
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True)
                if not prequantized
                else F8Linear(
                    in_features=hidden_size,
                    out_features=mlp_hidden_dim,
                    bias=True,
                )
            ),
            nn.GELU(approximate="tanh"),
            (
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
                if not prequantized
                else F8Linear(
                    in_features=mlp_hidden_dim,
                    out_features=hidden_size,
                    bias=True,
                )
            ),
        )

        self.txt_mod = Modulation(
            hidden_size, double=True, quantized_modulation=quantized_modulation
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            prequantized=prequantized,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            (
                nn.Linear(hidden_size, mlp_hidden_dim, bias=True)
                if not prequantized
                else F8Linear(
                    in_features=hidden_size,
                    out_features=mlp_hidden_dim,
                    bias=True,
                )
            ),
            nn.GELU(approximate="tanh"),
            (
                nn.Linear(mlp_hidden_dim, hidden_size, bias=True)
                if not prequantized
                else F8Linear(
                    in_features=mlp_hidden_dim,
                    out_features=hidden_size,
                    bias=True,
                )
            ),
        )
        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H

    def rearrange_for_norm(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, D = x.shape
        q, k, v = x.reshape(B, L, self.K, self.H, D // self.KH).permute(2, 0, 3, 1, 4)
        return q, k, v

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = self.rearrange_for_norm(img_qkv)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = self.rearrange_for_norm(txt_qkv)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]
        # calculate the img bloks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        ).clamp(min=-384 * 2, max=384 * 2)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        ).clamp(min=-384 * 2, max=384 * 2)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
        dtype: torch.dtype = torch.float16,
        quantized_modulation: bool = False,
        prequantized: bool = False,
    ):
        super().__init__()
        from fp8.float8_quantize import F8Linear

        self.dtype = dtype
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = (
            nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
            if not prequantized
            else F8Linear(
                in_features=hidden_size,
                out_features=hidden_size * 3 + self.mlp_hidden_dim,
                bias=True,
            )
        )
        # proj and mlp_out
        self.linear2 = (
            nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
            if not prequantized
            else F8Linear(
                in_features=hidden_size + self.mlp_hidden_dim,
                out_features=hidden_size,
                bias=True,
            )
        )

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(
            hidden_size,
            double=False,
            quantized_modulation=quantized_modulation and prequantized,
        )

        self.K = 3
        self.H = self.num_heads
        self.KH = self.K * self.H

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod = self.modulation(vec)[0]
        pre_norm = self.pre_norm(x)
        x_mod = (1 + mod.scale) * pre_norm + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim],
            dim=-1,
        )
        B, L, D = qkv.shape
        q, k, v = qkv.reshape(B, L, self.K, self.H, D // self.KH).permute(2, 0, 3, 1, 4)
        q, k = self.norm(q, k, v)
        attn = attention(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2)).clamp(
            min=-384 * 4, max=384 * 4
        )
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, config: "ModelSpec", dtype: torch.dtype = torch.float16):
        super().__init__()

        self.dtype = dtype
        self.params = config.params
        self.in_channels = config.params.in_channels
        self.out_channels = self.in_channels
        prequantized_flow = config.prequantized_flow
        quantized_embedders = config.quantize_flow_embedder_layers and prequantized_flow
        quantized_modulation = config.quantize_modulation and prequantized_flow
        from fp8.float8_quantize import F8Linear

        if config.params.hidden_size % config.params.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.params.hidden_size} must be divisible by num_heads {config.params.num_heads}"
            )
        pe_dim = config.params.hidden_size // config.params.num_heads
        if sum(config.params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = config.params.hidden_size
        self.num_heads = config.params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim,
            theta=config.params.theta,
            axes_dim=config.params.axes_dim,
            dtype=self.dtype,
        )
        self.img_in = (
            nn.Linear(self.in_channels, self.hidden_size, bias=True)
            if not prequantized_flow
            else (
                F8Linear(
                    in_features=self.in_channels,
                    out_features=self.hidden_size,
                    bias=True,
                )
                if quantized_embedders
                else nn.Linear(self.in_channels, self.hidden_size, bias=True)
            )
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            prequantized=prequantized_flow,
            quantized=quantized_embedders,
        )
        self.vector_in = MLPEmbedder(
            config.params.vec_in_dim,
            self.hidden_size,
            prequantized=prequantized_flow,
            quantized=quantized_embedders,
        )
        self.guidance_in = (
            MLPEmbedder(
                in_dim=256,
                hidden_dim=self.hidden_size,
                prequantized=prequantized_flow,
                quantized=quantized_embedders,
            )
            if config.params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = (
            nn.Linear(config.params.context_in_dim, self.hidden_size)
            if not quantized_embedders
            else (
                F8Linear(
                    in_features=config.params.context_in_dim,
                    out_features=self.hidden_size,
                    bias=True,
                )
                if quantized_embedders
                else nn.Linear(config.params.context_in_dim, self.hidden_size)
            )
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.params.mlp_ratio,
                    qkv_bias=config.params.qkv_bias,
                    dtype=self.dtype,
                    quantized_modulation=quantized_modulation,
                    prequantized=prequantized_flow,
                )
                for _ in range(config.params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.params.mlp_ratio,
                    dtype=self.dtype,
                    quantized_modulation=quantized_modulation,
                    prequantized=prequantized_flow,
                )
                for _ in range(config.params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        #self.compiling = False

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).type(self.dtype))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(
                timestep_embedding(guidance, 256).type(self.dtype)
            )
        vec = vec + self.vector_in(y)

        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # double stream blocks
        for block in self.double_blocks:
            # if self.compiling: 
            #     torch._dynamo.mark_dynamic(img, 1) 
            #     torch._dynamo.mark_dynamic(pe, 2)
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)

        # single stream blocks
        for block in self.single_blocks:
            # if self.compiling:
            #     torch._dynamo.mark_dynamic(img, 1) 
            #     torch._dynamo.mark_dynamic(pe, 2)
            img = block(img, vec=vec, pe=pe)

        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    @classmethod
    def from_pretrained(
        cls: "Flux", path: str, dtype: torch.dtype = torch.float16
    ) -> "Flux":
        from fp8.util import load_config_from_path
        from safetensors.torch import load_file

        config = load_config_from_path(path)
        with torch.device("meta"):
            klass = cls(config=config, dtype=dtype)
            if not config.prequantized_flow:
                klass.type(dtype)

        ckpt = load_file(config.ckpt_path, device="cpu")
        klass.load_state_dict(ckpt, assign=True)
        return klass.to("cpu")
