from loguru import logger
import torch
import torch.nn as nn
from torch.nn import init
import math
from torch.compiler import is_compiling
from torch import __version__
from torch.version import cuda

from fp8.modules.flux_model import Modulation

IS_TORCH_2_4 = __version__ < (2, 4, 9)
LT_TORCH_2_4 = __version__ < (2, 4)
if LT_TORCH_2_4:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError(
            "This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later."
        )
CUDA_VERSION = float(cuda) if cuda else 0
if CUDA_VERSION < 12.4:
    raise RuntimeError(
        f"This version of PyTorch is not supported. Please upgrade to PyTorch 2.4 with CUDA 12.4 or later got torch version {__version__} and CUDA version {cuda}."
    )
try:
    from cublas_ops import CublasLinear
except ImportError:
    CublasLinear = type(None)


class F8Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=torch.float16,
        float8_dtype=torch.float8_e4m3fn,
        float_weight: torch.Tensor = None,
        float_bias: torch.Tensor = None,
        num_scale_trials: int = 12,
        input_float8_dtype=torch.float8_e5m2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.float8_dtype = float8_dtype
        self.input_float8_dtype = input_float8_dtype
        self.input_scale_initialized = False
        self.weight_initialized = False
        self.max_value = torch.finfo(self.float8_dtype).max
        self.input_max_value = torch.finfo(self.input_float8_dtype).max
        factory_kwargs = {"dtype": dtype, "device": device}
        if float_weight is None:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), **factory_kwargs)
            )
        else:
            self.weight = nn.Parameter(
                float_weight, requires_grad=float_weight.requires_grad
            )
        if float_bias is None:
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(out_features, **factory_kwargs),
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(float_bias, requires_grad=float_bias.requires_grad)
        self.num_scale_trials = num_scale_trials
        self.input_amax_trials = torch.zeros(
            num_scale_trials, requires_grad=False, device=device, dtype=torch.float32
        )
        self.trial_index = 0
        self.register_buffer("scale", None)
        self.register_buffer(
            "input_scale",
            None,
        )
        self.register_buffer(
            "float8_data",
            None,
        )
        self.scale_reciprocal = self.register_buffer("scale_reciprocal", None)
        self.input_scale_reciprocal = self.register_buffer(
            "input_scale_reciprocal", None
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,  # noqa
        strict,  # noqa
        missing_keys,  # noqa
        unexpected_keys,  # noqa
        error_msgs,  # noqa
    ):
        sd = {k.replace(prefix, ""): v for k, v in state_dict.items()}
        if "weight" in sd:
            if (
                "float8_data" not in sd
                or sd["float8_data"] is None
                and sd["weight"].shape == (self.out_features, self.in_features)
            ):
                # Initialize as if it's an F8Linear that needs to be quantized
                self._parameters["weight"] = nn.Parameter(
                    sd["weight"], requires_grad=False
                )
                if "bias" in sd:
                    self._parameters["bias"] = nn.Parameter(
                        sd["bias"], requires_grad=False
                    )
                self.quantize_weight()
            elif sd["float8_data"].shape == (
                self.out_features,
                self.in_features,
            ) and sd["weight"] == torch.zeros_like(sd["weight"]):
                w = sd["weight"]
                # Set the init values as if it's already quantized float8_data
                self._buffers["float8_data"] = sd["float8_data"]
                self._parameters["weight"] = nn.Parameter(
                    torch.zeros(
                        1,
                        dtype=w.dtype,
                        device=w.device,
                        requires_grad=False,
                    )
                )
                if "bias" in sd:
                    self._parameters["bias"] = nn.Parameter(
                        sd["bias"], requires_grad=False
                    )
                self.weight_initialized = True

                # Check if scales and reciprocals are initialized
                if all(
                    key in sd
                    for key in [
                        "scale",
                        "input_scale",
                        "scale_reciprocal",
                        "input_scale_reciprocal",
                    ]
                ):
                    self.scale = sd["scale"].float()
                    self.input_scale = sd["input_scale"].float()
                    self.scale_reciprocal = sd["scale_reciprocal"].float()
                    self.input_scale_reciprocal = sd["input_scale_reciprocal"].float()
                    self.input_scale_initialized = True
                    self.trial_index = self.num_scale_trials
                elif "scale" in sd and "scale_reciprocal" in sd:
                    self.scale = sd["scale"].float()
                    self.input_scale = (
                        sd["input_scale"].float() if "input_scale" in sd else None
                    )
                    self.scale_reciprocal = sd["scale_reciprocal"].float()
                    self.input_scale_reciprocal = (
                        sd["input_scale_reciprocal"].float()
                        if "input_scale_reciprocal" in sd
                        else None
                    )
                    self.input_scale_initialized = "input_scale" in sd
                    self.trial_index = (
                        self.num_scale_trials if "input_scale" in sd else 0
                    )
                    self.input_amax_trials = torch.zeros(
                        self.num_scale_trials,
                        requires_grad=False,
                        dtype=torch.float32,
                        device=self.weight.device,
                    )
                    self.input_scale_initialized = False
                    self.trial_index = 0
                else:
                    # If scales are not initialized, reset trials
                    self.input_scale_initialized = False
                    self.trial_index = 0
                    self.input_amax_trials = torch.zeros(
                        self.num_scale_trials, requires_grad=False, dtype=torch.float32
                    )
            else:
                raise RuntimeError(
                    f"Weight tensor not found or has incorrect shape in state dict: {sd.keys()}"
                )
        else:
            raise RuntimeError(
                "Weight tensor not found or has incorrect shape in state dict"
            )

    def quantize_weight(self):
        if self.weight_initialized:
            return
        amax = torch.max(torch.abs(self.weight.data)).float()
        self.scale = self.amax_to_scale(amax, self.max_value)
        self.float8_data = self.to_fp8_saturated(
            self.weight.data, self.scale, self.max_value
        ).to(self.float8_dtype)
        self.scale_reciprocal = self.scale.reciprocal()
        self.weight.data = torch.zeros(
            1, dtype=self.weight.dtype, device=self.weight.device, requires_grad=False
        )
        self.weight_initialized = True

    def set_weight_tensor(self, tensor: torch.Tensor):
        self.weight.data = tensor
        self.weight_initialized = False
        self.quantize_weight()

    def amax_to_scale(self, amax, max_val):
        return (max_val / torch.clamp(amax, min=1e-12)).clamp(max=max_val)

    def to_fp8_saturated(self, x, scale, max_val):
        return (x * scale).clamp(-max_val, max_val)

    def quantize_input(self, x: torch.Tensor):
        if self.input_scale_initialized:
            return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                self.input_float8_dtype
            )
        if self.trial_index < self.num_scale_trials:
            amax = torch.max(torch.abs(x)).float()

            self.input_amax_trials[self.trial_index] = amax
            self.trial_index += 1
            self.input_scale = self.amax_to_scale(
                self.input_amax_trials[: self.trial_index].max(), self.input_max_value
            )
            self.input_scale_reciprocal = self.input_scale.reciprocal()
            return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                self.input_float8_dtype
            )
        self.input_scale = self.amax_to_scale(
            self.input_amax_trials.max(), self.input_max_value
        )
        self.input_scale_reciprocal = self.input_scale.reciprocal()
        self.input_scale_initialized = True
        return self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
            self.input_float8_dtype
        )

    def reset_parameters(self) -> None:
        if self.weight_initialized:
            self.weight = nn.Parameter(
                torch.empty(
                    (self.out_features, self.in_features),
                    **{
                        "dtype": self.weight.dtype,
                        "device": self.weight.device,
                    },
                )
            )
            self.weight_initialized = False
            self.input_scale_initialized = False
            self.trial_index = 0
            self.input_amax_trials.zero_()
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)  # noqa
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        self.quantize_weight()
        self.max_value = torch.finfo(self.float8_dtype).max
        self.input_max_value = torch.finfo(self.input_float8_dtype).max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale_initialized or is_compiling():
            x = self.to_fp8_saturated(x, self.input_scale, self.input_max_value).to(
                self.input_float8_dtype
            )
        else:
            x = self.quantize_input(x)

        prev_dims = x.shape[:-1]
        x = x.view(-1, self.in_features)

        device = x.device
        if x.device.type != 'cpu' and torch.cuda.get_device_capability(x.device) >= (8, 9):
            # float8 matmul, much faster than float16 matmul w/ float32 accumulate on ADA devices!
            out = torch._scaled_mm(  # noqa
                x,
                self.float8_data.T,
                scale_a=self.input_scale_reciprocal,
                scale_b=self.scale_reciprocal,
                bias=self.bias,
                out_dtype=self.weight.dtype,
                use_fast_accum=True,
            )
        else:
            # Plain matrix multiplication for non-ADA devices
            # Assuming x is in float8 and self.float8_data is in float8 as well
            # Convert to float32, perform the multiplication, and then apply scaling and bias if necessary

            # Convert float8 to float32 for the multiplication
            x_float32 = x.to(torch.float32)
            float8_data_float32 = self.float8_data.T.to(torch.float32)

            # Regular matrix multiplication
            out = torch.matmul(x_float32, float8_data_float32)

            # Scale the output accordingly
            out = out * (self.input_scale_reciprocal * self.scale_reciprocal)

            # Add bias if it exists
            if self.bias is not None:
                out += self.bias
                out = out.to(self.weight.dtype)

        if IS_TORCH_2_4:
            out = out[0]
        return out.view(*prev_dims, self.out_features)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        float8_dtype=torch.float8_e4m3fn,
        input_float8_dtype=torch.float8_e5m2,
    ) -> "F8Linear":
        f8_lin = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            float8_dtype=float8_dtype,
            float_weight=linear.weight.data,
            float_bias=(linear.bias.data if linear.bias is not None else None),
            input_float8_dtype=input_float8_dtype,
        )
        f8_lin.quantize_weight()
        return f8_lin


@torch.inference_mode()
def recursive_swap_linears(
    model: nn.Module,
    float8_dtype=torch.float8_e4m3fn,
    input_float8_dtype=torch.float8_e5m2,
    quantize_modulation: bool = True,
    ignore_keys: list[str] = [],
) -> None:
    """
    Recursively swaps all nn.Linear modules in the given model with F8Linear modules.

    This function traverses the model's structure and replaces each nn.Linear
    instance with an F8Linear instance, which uses 8-bit floating point
    quantization for weights. The original linear layer's weights are deleted
    after conversion to save memory.

    Args:
        model (nn.Module): The PyTorch model to modify.

    Note:
        This function modifies the model in-place. After calling this function,
        all linear layers in the model will be using 8-bit quantization.
    """
    for name, child in model.named_children():
        if name in ignore_keys:
            continue
        if isinstance(child, Modulation) and not quantize_modulation:
            continue
        if isinstance(child, nn.Linear) and not isinstance(
            child, (F8Linear, CublasLinear)
        ):
            setattr(
                model,
                name,
                F8Linear.from_linear(
                    child,
                    float8_dtype=float8_dtype,
                    input_float8_dtype=input_float8_dtype,
                ),
            )
            del child
        else:
            recursive_swap_linears(
                child,
                float8_dtype=float8_dtype,
                input_float8_dtype=input_float8_dtype,
                quantize_modulation=quantize_modulation,
                ignore_keys=ignore_keys,
            )


@torch.inference_mode()
def swap_to_cublaslinear(model: nn.Module):
    if not isinstance(CublasLinear, type(torch.nn.Module)):
        return
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and not isinstance(
            child, (F8Linear, CublasLinear)
        ):
            cublas_lin = CublasLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                dtype=child.weight.dtype,
                device=child.weight.device,
            )
            cublas_lin.weight.data = child.weight.clone().detach()
            cublas_lin.bias.data = child.bias.clone().detach()
            setattr(model, name, cublas_lin)
            del child
        else:
            swap_to_cublaslinear(child)


@torch.inference_mode()
def quantize_flow_transformer_and_dispatch_float8(
    flow_model: nn.Module,
    device=torch.device("cuda"),
    float8_dtype=torch.float8_e4m3fn,
    input_float8_dtype=torch.float8_e5m2,
    offload_flow=False,
    swap_linears_with_cublaslinear=False,
    flow_dtype=torch.float16,
    quantize_modulation: bool = True,
    quantize_flow_embedder_layers: bool = True,
) -> nn.Module:
    """
    Quantize the flux flow transformer model (original BFL codebase version) and dispatch to the given device.

    Iteratively pushes each module to device, evals, replaces linear layers with F8Linear except for final_layer, and quantizes.

    Allows for fast dispatch to gpu & quantize without causing OOM on gpus with limited memory.

    After dispatching, if offload_flow is True, offloads the model to cpu.

    if swap_linears_with_cublaslinear is true, and flow_dtype == torch.float16, then swap all linears with cublaslinears for 2x performance boost on consumer GPUs.
    Otherwise will skip the cublaslinear swap.

    For added extra precision, you can set quantize_flow_embedder_layers to False,
    this helps maintain the output quality of the flow transformer moreso than fully quantizing,
    at the expense of ~512MB more VRAM usage.

    For added extra precision, you can set quantize_modulation to False,
    this helps maintain the output quality of the flow transformer moreso than fully quantizing,
    at the expense of ~2GB more VRAM usage, but- has a much higher impact on image quality than the embedder layers.
    """
    for module in flow_model.double_blocks:
        module.to(device)
        module.eval()
        recursive_swap_linears(
            module,
            float8_dtype=float8_dtype,
            input_float8_dtype=input_float8_dtype,
            quantize_modulation=quantize_modulation,
        )
        torch.cuda.empty_cache()
    for module in flow_model.single_blocks:
        module.to(device)
        module.eval()
        recursive_swap_linears(
            module,
            float8_dtype=float8_dtype,
            input_float8_dtype=input_float8_dtype,
            quantize_modulation=quantize_modulation,
        )
        torch.cuda.empty_cache()
    to_gpu_extras = [
        "vector_in",
        "img_in",
        "txt_in",
        "time_in",
        "guidance_in",
        "final_layer",
        "pe_embedder",
    ]
    for module in to_gpu_extras:
        m_extra = getattr(flow_model, module)
        if m_extra is None:
            continue
        m_extra.to(device)
        m_extra.eval()
        if isinstance(m_extra, nn.Linear) and not isinstance(
            m_extra, (F8Linear, CublasLinear)
        ):
            if quantize_flow_embedder_layers:
                setattr(
                    flow_model,
                    module,
                    F8Linear.from_linear(
                        m_extra,
                        float8_dtype=float8_dtype,
                        input_float8_dtype=input_float8_dtype,
                    ),
                )
            del m_extra
        elif module != "final_layer":
            if quantize_flow_embedder_layers:
                recursive_swap_linears(
                    m_extra,
                    float8_dtype=float8_dtype,
                    input_float8_dtype=input_float8_dtype,
                    quantize_modulation=quantize_modulation,
                )
        torch.cuda.empty_cache()
    if (
        swap_linears_with_cublaslinear
        and flow_dtype == torch.float16
        and isinstance(CublasLinear, type(torch.nn.Linear))
    ):
        swap_to_cublaslinear(flow_model)
    elif swap_linears_with_cublaslinear and flow_dtype != torch.float16:
        logger.warning("Skipping cublas linear swap because flow_dtype is not float16")
    if offload_flow:
        flow_model.to("cpu")
        torch.cuda.empty_cache()
    return flow_model
