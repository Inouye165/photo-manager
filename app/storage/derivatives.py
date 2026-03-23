"""Derivative generation: thumbnails, previews, crops.

Produces browser-safe images from immutable originals without touching the
source tree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None  # allow very large images

# ── Save helpers ─────────────────────────────────────────────────────────

_SAVE_OPTS = {
    "WEBP": {"quality": 84, "method": 6},
    "JPEG": {"quality": 90, "optimize": True},
    "PNG": {},
}


def _save(img: Image.Image, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format=fmt.upper(), **_SAVE_OPTS.get(fmt.upper(), {}))


# ── Public API ───────────────────────────────────────────────────────────


class DerivativeStore:
    """Generate and cache thumbnails / previews / crops in a workspace dir."""

    def __init__(
        self,
        derivatives_dir: Path,
        crops_dir: Path,
        *,
        thumb_size: tuple[int, int] = (384, 384),
        preview_size: tuple[int, int] = (1600, 1600),
        crop_px: int = 224,
        derivative_fmt: str = "WEBP",
        crop_fmt: str = "JPEG",
    ) -> None:
        self.derivatives_dir = derivatives_dir
        self.crops_dir = crops_dir
        self.thumb_size = thumb_size
        self.preview_size = preview_size
        self.crop_px = crop_px
        self.derivative_fmt = derivative_fmt.upper()
        self.crop_fmt = crop_fmt.upper()

        for d in (derivatives_dir / "thumb", derivatives_dir / "preview", crops_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Thumbnails / previews ────────────────────────────────────────────

    def derivative(
        self,
        source_path: Path,
        relative_path: str,
        asset_id: str,
        variant: Literal["thumb", "preview"] = "thumb",
    ) -> Path:
        """Return the path to the requested derivative, creating it if absent."""
        size = self.thumb_size if variant == "thumb" else self.preview_size
        ext = self.derivative_fmt.lower()
        stem = Path(relative_path).stem
        parent = Path(relative_path).parent
        target = (
            self.derivatives_dir
            / variant
            / parent
            / f"{stem}-{asset_id[:12]}-{variant}.{ext}"
        )

        if target.exists():
            return target

        with Image.open(source_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            _save(img, target, self.derivative_fmt)

        return target

    def external_derivative(
        self,
        absolute_path: Path,
        key: str,
        variant: Literal["thumb", "preview"] = "thumb",
    ) -> Path:
        """Create a derivative for files already inside the workspace."""
        ext = self.derivative_fmt.lower()
        target = self.derivatives_dir / variant / "external" / f"{quote(key, safe='/')}.{ext}"

        if target.exists() and target.stat().st_mtime_ns >= absolute_path.stat().st_mtime_ns:
            return target

        size = self.thumb_size if variant == "thumb" else self.preview_size
        with Image.open(absolute_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail(size, Image.Resampling.LANCZOS)
            _save(img, target, self.derivative_fmt)

        return target

    # ── Crops ────────────────────────────────────────────────────────────

    def crop(
        self,
        source_path: Path,
        relative_path: str,
        asset_id: str,
        bbox: tuple[int, int, int, int],
        subject_type: str = "unknown",
        identity_hint: str | None = None,
    ) -> Path:
        """Return a square crop for model input, creating it if absent."""
        ext = self.crop_fmt.lower()
        slug = (identity_hint or subject_type or "crop").replace("/", "-").replace("\\", "-")
        stem = Path(relative_path).stem
        parent = Path(relative_path).parent
        target = (
            self.crops_dir
            / subject_type
            / parent
            / f"{stem}-{slug}-{asset_id[:10]}.{ext}"
        )

        if target.exists():
            return target

        x1, y1, x2, y2 = bbox
        with Image.open(source_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            left, top = max(0, min(x1, w)), max(0, min(y1, h))
            right, bottom = max(left + 1, min(x2, w)), max(top + 1, min(y2, h))
            region = img.crop((left, top, right, bottom))
            fitted = ImageOps.fit(
                region,
                (self.crop_px, self.crop_px),
                method=Image.Resampling.LANCZOS,
            )
            _save(fitted, target, self.crop_fmt)

        return target
