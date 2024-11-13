import torch
#from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel



def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # Only enable flash attention backend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    # x = rearrange(x, "B H L D -> B L (H D)")
    x = x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), -1)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    # f64 is problematic
    # https://github.com/pytorch/TensorRT/blob/v2.4.0/py/torch_tensorrt/dynamo/conversion/converter_utils.py#L380
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    # scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    # out = torch.einsum("...n,d->...nd", pos, omega)
    out = pos.unsqueeze(-1) * omega
    # out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)
    out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    # out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    # Reshaping the tensor to (..., n, d, 2, 2)
    out = out.view(*out.shape[:-1], 2, 2)
    return out # .float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
