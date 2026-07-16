"""Strict single-image ingestion shared by direct and sliced inference."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from ..common.config_io import read_bounded_regular_bytes
from ..common.deps import require

MAX_ENCODED_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
MAX_IMAGE_DIMENSION = 32_768
_STILL_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _validate_dimensions(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
        raise ValueError(f"image dimensions must not exceed {MAX_IMAGE_DIMENSION}")
    if height * width > MAX_IMAGE_PIXELS:
        raise ValueError(f"image exceeds the {MAX_IMAGE_PIXELS}-pixel safety limit")


def prepare_single_image(image: Any) -> Any:
    """Return an owned RGB image and reject URLs, videos, batches, and directories.

    NumPy inputs are interpreted as RGB, matching Pillow and SAHI's public image
    boundary. Backend adapters are responsible for any backend-specific channel
    ordering.
    """
    if type(image) is np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("numpy image must be uint8 with shape (height, width, 3)")
        _validate_dimensions(int(image.shape[0]), int(image.shape[1]))
        return np.ascontiguousarray(image).copy()

    if isinstance(image, (str, Path)):
        raw_path = str(image)
        if any(character in raw_path for character in "\0\r\n") or "://" in raw_path:
            raise ValueError("image must be a local still-image path, not a URL or stream")
        path = Path(image).expanduser().absolute()
        if path.suffix.lower() not in _STILL_IMAGE_SUFFIXES:
            raise ValueError(
                f"image suffix must be one of {sorted(_STILL_IMAGE_SUFFIXES)}; videos are rejected"
            )
        encoded = read_bounded_regular_bytes(path, MAX_ENCODED_IMAGE_BYTES, "input image")
        image_module = require("PIL.Image", "vision")
        image_ops = require("PIL.ImageOps", "vision")
        try:
            decoded = image_module.open(io.BytesIO(encoded))
        except (OSError, ValueError) as exc:
            raise ValueError(f"input image is not a valid bounded still image: {path}") from exc

        with decoded:
            _validate_dimensions(int(decoded.height), int(decoded.width))
            if int(getattr(decoded, "n_frames", 1)) != 1 or bool(
                getattr(decoded, "is_animated", False)
            ):
                raise ValueError("animated or multi-page images are not accepted")
            try:
                loaded = image_ops.exif_transpose(decoded).convert("RGB")
                loaded.load()
                return loaded.copy()
            except (OSError, ValueError) as exc:
                raise ValueError(f"input image is not a valid bounded still image: {path}") from exc

    raise TypeError("image must be one uint8 RGB HWC numpy array or a local still-image path")


__all__ = ["prepare_single_image"]
