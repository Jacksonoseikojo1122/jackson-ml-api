"""Base64 image decoding with validation. No TensorFlow here."""

from __future__ import annotations

import base64
import binascii
import io
import re

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_SIZE, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

_DATA_URL_PREFIX = re.compile(r"^data:image/[a-z0-9.+-]+;base64,", re.IGNORECASE)


class ImageDecodeError(ValueError):
    """Raised for client-side image problems; mapped to an HTTP error by the app."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code

    def with_context(self, context: str) -> ImageDecodeError:
        return ImageDecodeError(f"{context}: {self}", self.status_code)


def strip_data_url(value: str) -> str:
    """Remove an optional ``data:image/<type>;base64,`` prefix."""
    value = value.strip()
    match = _DATA_URL_PREFIX.match(value)
    return value[match.end() :] if match else value


def decode_base64(value: str, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    payload = "".join(strip_data_url(value).split())  # tolerate line-wrapped base64
    if not payload:
        raise ImageDecodeError("image payload is empty")
    # Cheap upper-bound check before allocating the decoded buffer.
    if (len(payload) // 4) * 3 > max_bytes + 3:
        raise ImageDecodeError(f"image exceeds the {max_bytes} byte limit", status_code=413)
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageDecodeError("image is not valid base64") from None
    if len(raw) > max_bytes:
        raise ImageDecodeError(f"image exceeds the {max_bytes} byte limit", status_code=413)
    return raw


def load_image(raw: bytes, size: int = IMAGE_SIZE, max_pixels: int = MAX_IMAGE_PIXELS) -> np.ndarray:
    """Decode image bytes to a float32 RGB array of shape (size, size, 3), values 0-255."""
    try:
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise ImageDecodeError(f"image is {width}x{height}; the limit is {max_pixels} pixels")
            rgb = img.convert("RGB").resize((size, size))
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise ImageDecodeError("payload is not a decodable image (expected JPEG, PNG, WebP, ...)") from None
    return np.asarray(rgb, dtype=np.float32)


def decode_image(value: str, size: int = IMAGE_SIZE) -> np.ndarray:
    return load_image(decode_base64(value), size=size)
