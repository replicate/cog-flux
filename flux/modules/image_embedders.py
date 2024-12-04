import os
from typing import Protocol

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from safetensors.torch import load_file as load_sft
from torch import nn
from transformers import AutoModelForDepthEstimation, AutoProcessor, SiglipImageProcessor, SiglipVisionModel

# hack to avoid circular imports
def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


class ImageEncoder(Protocol):
    def __call__(self, img: torch.Tensor) -> torch.Tensor: ...


class DepthImageEncoder(ImageEncoder):
    depth_model_name = "LiheYoung/depth-anything-large-hf"

    def __init__(self, device, depth_model_path=depth_model_name):
        self.device = device
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_path).to(device)
        self.processor = AutoProcessor.from_pretrained(depth_model_path)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        hw = img.shape[-2:]

        img = torch.clamp(img, -1.0, 1.0)
        img_byte = ((img + 1.0) * 127.5).byte()

        img = self.processor(img_byte, return_tensors="pt")["pixel_values"]
        depth = self.depth_model(img.to(self.device)).predicted_depth
        depth = repeat(depth, "b h w -> b 3 h w")
        depth = torch.nn.functional.interpolate(depth, hw, mode="bicubic", antialias=True)

        depth = depth / 127.5 - 1.0
        return depth

      
class CannyImageEncoder(ImageEncoder):
    def __init__(
        self,
        device,
        min_t: int = 50,
        max_t: int = 200,
    ):
        self.device = device
        self.min_t = min_t
        self.max_t = max_t

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        assert img.shape[0] == 1, "Only batch size 1 is supported"

        img = rearrange(img[0], "c h w -> h w c")
        img = torch.clamp(img, -1.0, 1.0)
        img_np = ((img + 1.0) * 127.5).numpy().astype(np.uint8)

        # Apply Canny edge detection
        canny = cv2.Canny(img_np, self.min_t, self.max_t)

        # Convert back to torch tensor and reshape
        canny = torch.from_numpy(canny).float() / 127.5 - 1.0
        canny = rearrange(canny, "h w -> 1 1 h w")
        canny = repeat(canny, "b 1 ... -> b 3 ...")
        return canny.to(self.device)


class ReduxImageEncoder(nn.Module):
    siglip_model_name = "google/siglip-so400m-patch14-384"

    def __init__(
        self,
        device,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
        redux_path: str | None = os.getenv("FLUX_REDUX"),
        siglip_path: str | None = siglip_model_name,
        dtype=torch.bfloat16,
    ) -> None:
        assert redux_path is not None, "Redux path must be provided"

        super().__init__()

        self.redux_dim = redux_dim
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        with self.device:
            self.redux_up = nn.Linear(redux_dim, txt_in_features * 3, dtype=dtype)
            self.redux_down = nn.Linear(txt_in_features * 3, txt_in_features, dtype=dtype)

            sd = load_sft(redux_path, device=str(device))
            missing, unexpected = self.load_state_dict(sd, strict=False, assign=True)
            print_load_warning(missing, unexpected)

            self.siglip = SiglipVisionModel.from_pretrained(siglip_path).to(dtype=dtype)
        self.normalize = SiglipImageProcessor.from_pretrained(siglip_path)


    def __call__(self, x: Image.Image) -> torch.Tensor:
        imgs = self.normalize.preprocess(images=[x], do_resize=True, return_tensors="pt", do_convert_rgb=True)

        _encoded_x = self.siglip(**imgs.to(device=self.device, dtype=self.dtype)).last_hidden_state

        projected_x = self.redux_down(nn.functional.silu(self.redux_up(_encoded_x)))

        return projected_x