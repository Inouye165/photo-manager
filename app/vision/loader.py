"""Image loading with HEIC/HEIF support.

Centralises Pillow + pillow_heif registration so every consumer gets
consistent orientation and colour-mode handling.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

logger = logging.getLogger(__name__)


def load_pil(path: Path) -> Image.Image:
    """Open *path* as a Pillow RGB image, auto-rotating via EXIF."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_cv2(path: str | Path) -> np.ndarray:
    """Open *path* as a BGR ``numpy`` array (OpenCV convention)."""
    if cv2 is None:
        # Fallback: use PIL and convert to BGR numpy array
        pil = load_pil(Path(path))
        arr = np.asarray(pil)
        return arr[:, :, ::-1].copy()

    ext = str(path).lower()
    if ext.endswith((".heic", ".heif")):
        pil = load_pil(Path(path))
        arr = np.asarray(pil)
        return arr[:, :, ::-1].copy()  # RGB → BGR
    arr = cv2.imread(str(path))
    if arr is None:
        raise ValueError(f"Could not read image from {path}")
    return arr
