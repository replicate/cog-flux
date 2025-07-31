import asyncio
import base64
import io
import tempfile
from pathlib import Path

import pillow_avif  # noqa: F401
from pi_heif import register_heif_opener
from PIL import Image

from helpers import exception_without_traceback

UNSUPPORTED_JPEG_MODES = ["RGBA", "P"]
MIN_ASPECT_RATIO = 0.4  # 1:2.5
MAX_ASPECT_RATIO = 2.5  # 2.5:1

register_heif_opener()


def resize_image(
    image_path: Path, max_dim: int = 1024, min_dim: int = 300
) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio.

    Args:
        image_path: Path to the image file
        max_dim: Maximum dimension (width or height) in pixels
        min_dim: Minimum dimension (width or height) in pixels

    Returns:
        Resized PIL Image
    """
    img = Image.open(image_path)
    max_dim_current = max(img.size)
    min_dim_current = min(img.size)

    # Scale down if image is too large
    if max_dim_current > max_dim:
        scale = max_dim / max_dim_current
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Scale up if image is too small
    elif min_dim_current < min_dim:
        scale = min_dim / min_dim_current
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return img


def convert_to_supported_jpeg_mode(img: Image.Image) -> Image.Image:
    """
    Convert image to RGB mode if it's in an unsupported JPEG mode.

    Args:
        img: PIL Image to convert

    Returns:
        Converted PIL Image
    """
    if img.mode in UNSUPPORTED_JPEG_MODES:
        img = img.convert("RGB")
    return img


def clear_image_metadata(img: Image.Image) -> Image.Image:
    """
    Remove all metadata from the image.

    Args:
        img: PIL Image to clear metadata from

    Returns:
        PIL Image with cleared metadata
    """
    img.info.clear()
    return img


def save_to_base64(
    image_path: Path,
    img_format: str = "jpeg",
    quality: int = 80,
    max_dim: int = 1024,
    raw: bool = False,
) -> str:
    """
    Save an image to base64 string with optimization.

    Args:
        image_path: Path to the image file
        img_format: Image format (jpeg, png, etc.)
        quality: JPEG quality (1-100)
        max_dim: Maximum dimension in pixels
        raw: If True, return raw base64 without data URI prefix

    Returns:
        Base64 encoded image string
    """
    img = resize_image(image_path, max_dim)
    if img_format.lower() in ["jpg", "jpeg"]:
        img = convert_to_supported_jpeg_mode(img)
        img_format = "JPEG"

    buffer = io.BytesIO()
    img = clear_image_metadata(img)
    img.save(buffer, format=img_format.upper(), quality=quality)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    if raw:
        return img_base64
    return f"data:image/{img_format.lower()};base64,{img_base64}"


def crop_image_to_aspect_ratio(
    img: Image.Image,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
) -> Image.Image:
    """
    Crop the image to fit within the given aspect ratio bounds, centered.

    Args:
        img: PIL Image to crop
        min_aspect_ratio: Minimum allowed aspect ratio (width/height)
        max_aspect_ratio: Maximum allowed aspect ratio (width/height)

    Returns:
        Cropped PIL Image
    """
    width, height = img.size
    current_ar = width / height

    # If already within bounds, return as is
    if min_aspect_ratio <= current_ar <= max_aspect_ratio:
        return img

    # Too wide: crop width
    if current_ar > max_aspect_ratio:
        new_width = int(max_aspect_ratio * height)
        left = (width - new_width) // 2
        right = left + new_width
        box = (left, 0, right, height)
        return img.crop(box)

    # Too tall: crop height
    if current_ar < min_aspect_ratio:
        new_height = int(width / min_aspect_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        box = (0, top, width, bottom)
        return img.crop(box)

    return img


def validate_image_aspect_ratio(
    image_path: Path,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    aspect_ratio_error_message: str = "Image aspect ratio is outside the allowed range. Please use an image with an aspect ratio between 1:2.5 and 2.5:1.",
    crop_to_aspect_ratio: bool = False,
) -> None:
    try:
        img = Image.open(image_path)
        width, height = img.size

        if height == 0:
            exception_without_traceback(
                ValueError(f"Image {image_path} has zero height, which is invalid.")
            )

        current_ar = width / height

        if not (min_aspect_ratio <= current_ar <= max_aspect_ratio):
            if crop_to_aspect_ratio:
                cropped_img = crop_image_to_aspect_ratio(
                    img, min_aspect_ratio, max_aspect_ratio
                )
                # Only save if the image was actually changed
                if cropped_img.size != img.size:
                    print(
                        f"Cropping image to be within aspect ratio range: {image_path} ({current_ar:.2f} -> {cropped_img.size[0] / cropped_img.size[1]:.2f})"
                    )
                    cropped_img.save(image_path)
                return
            exception_without_traceback(
                ValueError(
                    f"Image aspect ratio ({current_ar:.2f}) is outside the allowed range "
                    f"[{min_aspect_ratio:.2f} (1:{1 / min_aspect_ratio:.1f}), {max_aspect_ratio:.2f} ({max_aspect_ratio:.1f}:1)]. "
                    f"{aspect_ratio_error_message}"
                )
            )
    except Exception as e:
        exception_without_traceback(
            ValueError(
                f"Error processing image {image_path} for aspect ratio validation: {e}"
            )
        )


def save_to_file(
    image_path: Path,
    img_format: str = "jpeg",
    quality: int = 80,
    max_dim: int = 1024,
) -> Path:
    """
    Save an optimized image to a temporary file.

    Args:
        image_path: Path to the image file
        img_format: Image format (jpeg, png, etc.)
        quality: JPEG quality (1-100)
        max_dim: Maximum dimension in pixels

    Returns:
        Path to the temporary file
    """
    img = resize_image(image_path, max_dim)
    if img_format.lower() in ["jpg", "jpeg"]:
        img = convert_to_supported_jpeg_mode(img)
        img_format = "JPEG"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=f".{img_format.lower()}"
    ) as tmp:
        img = clear_image_metadata(img)
        img.save(tmp.name, format=img_format.upper(), quality=quality)
        return Path(tmp.name)


async def optimized_base64(
    image_path: Path,
    img_format: str = "jpeg",
    quality: int = 80,
    max_dim: int = 1024,
    raw: bool = False,
) -> str:
    """
    Asynchronously save an image to base64 string with optimization.

    Args:
        image_path: Path to the image file
        img_format: Image format (jpeg, png, etc.)
        quality: JPEG quality (1-100)
        max_dim: Maximum dimension in pixels
        raw: If True, return raw base64 without data URI prefix

    Returns:
        Base64 encoded image string
    """
    return await asyncio.to_thread(
        save_to_base64, image_path, img_format, quality, max_dim, raw
    )


async def optimized_file(
    image_path: Path,
    img_format: str = "jpeg",
    quality: int = 80,
    max_dim: int = 1024,
) -> Path:
    """
    Asynchronously save an optimized image to a temporary file.

    Args:
        image_path: Path to the image file
        img_format: Image format (jpeg, png, etc.)
        quality: JPEG quality (1-100)
        max_dim: Maximum dimension in pixels

    Returns:
        Path to the temporary file
    """
    return await asyncio.to_thread(
        save_to_file, image_path, img_format, quality, max_dim
    )


async def async_validate_image_aspect_ratio(
    image_path: Path,
    min_aspect_ratio: float = MIN_ASPECT_RATIO,
    max_aspect_ratio: float = MAX_ASPECT_RATIO,
    aspect_ratio_error_message: str = "Image aspect ratio is outside the allowed range. Please use an image with an aspect ratio between 1:2.5 and 2.5:1.",
    crop_to_aspect_ratio: bool = False,
) -> None:
    return await asyncio.to_thread(
        validate_image_aspect_ratio,
        image_path,
        min_aspect_ratio,
        max_aspect_ratio,
        aspect_ratio_error_message,
        crop_to_aspect_ratio,
    )
