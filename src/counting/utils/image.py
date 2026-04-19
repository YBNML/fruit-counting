"""Minimal image I/O helpers. All numpy arrays are RGB uint8 unless stated."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


def read_image_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def ensure_pil(image, *, assume: Literal["rgb", "bgr"] = "rgb") -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image)
        if image.ndim == 3 and image.shape[2] == 3:
            arr = image if assume == "rgb" else np.ascontiguousarray(image[..., ::-1])
            return Image.fromarray(arr)
    raise TypeError(f"Cannot convert to PIL.Image: {type(image)}")


def ensure_np_rgb(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    raise TypeError(f"Cannot convert to np.ndarray: {type(image)}")
