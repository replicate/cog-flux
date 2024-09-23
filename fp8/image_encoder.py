import io
from PIL import Image
import numpy as np
import torch


class ImageEncoder:
    @torch.inference_mode()
    def encode_pil(self, img: torch.Tensor) -> Image:
        if img.ndim == 2:
            img = (
                img[None]
                .repeat_interleave(3, dim=0)
                .permute(1, 2, 0)
                .contiguous()
                .clamp(0, 255)
                .type(torch.uint8)
            )
        elif img.ndim == 3:
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).contiguous().clamp(0, 255).type(torch.uint8)
            elif img.shape[2] == 3:
                img = img.contiguous().clamp(0, 255).type(torch.uint8)
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise ValueError(f"Unsupported image num dims: {img.ndim}")

        img = img.cpu().numpy().astype(np.uint8)
        return Image.fromarray(img)

    @torch.inference_mode()
    def encode_torch(self, img: torch.Tensor, quality=95):
        im = self.encode_pil(img)
        iob = io.BytesIO()
        im.save(iob, format="JPEG", quality=quality)
        iob.seek(0)
        return iob
