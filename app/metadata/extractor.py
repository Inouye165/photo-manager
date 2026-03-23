"""EXIF / metadata extraction for images.

Extracts timestamps, GPS coordinates, camera info, and basic dimensions
from image EXIF data.  Designed to enrich ``AssetRecord`` after scanning.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS

logger = logging.getLogger(__name__)


class ImageMeta:
    """Parsed metadata for a single image file."""

    __slots__ = (
        "width", "height", "taken_at", "camera_make", "camera_model",
        "gps_lat", "gps_lon", "orientation", "raw",
    )

    def __init__(self) -> None:
        self.width: int = 0
        self.height: int = 0
        self.taken_at: Optional[datetime] = None
        self.camera_make: Optional[str] = None
        self.camera_model: Optional[str] = None
        self.gps_lat: Optional[float] = None
        self.gps_lon: Optional[float] = None
        self.orientation: Optional[int] = None
        self.raw: dict[str, Any] = {}


def extract(path: Path) -> ImageMeta:
    """Return an ``ImageMeta`` populated from EXIF data of *path*."""
    meta = ImageMeta()

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            meta.width, meta.height = img.size
            exif_data = img.getexif()
            if not exif_data:
                return meta

            decoded: dict[str, Any] = {}
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                decoded[tag_name] = value

            meta.raw = decoded
            meta.camera_make = _str_or_none(decoded.get("Make"))
            meta.camera_model = _str_or_none(decoded.get("Model"))
            meta.orientation = decoded.get("Orientation")
            meta.taken_at = _parse_datetime(decoded.get("DateTimeOriginal") or decoded.get("DateTime"))

            gps_info = exif_data.get_ifd(0x8825)
            if gps_info:
                meta.gps_lat, meta.gps_lon = _decode_gps(gps_info)

    except Exception as exc:
        logger.debug("EXIF extraction failed for %s: %s", path, exc)

    return meta


# ── Helpers ──────────────────────────────────────────────────────────────

def _str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _decode_gps(gps_info: dict) -> tuple[Optional[float], Optional[float]]:
    """Convert EXIF GPS IFD into (latitude, longitude) or (None, None)."""
    try:
        decoded: dict[str, Any] = {}
        for key, val in gps_info.items():
            decoded[GPSTAGS.get(key, key)] = val

        lat = _dms_to_decimal(decoded.get("GPSLatitude"), decoded.get("GPSLatitudeRef"))
        lon = _dms_to_decimal(decoded.get("GPSLongitude"), decoded.get("GPSLongitudeRef"))
        return lat, lon
    except Exception:
        return None, None


def _dms_to_decimal(dms: Any, ref: Any) -> Optional[float]:
    if dms is None or ref is None:
        return None
    try:
        d, m, s = [float(x) for x in dms]
        decimal = d + m / 60 + s / 3600
        if str(ref).upper() in ("S", "W"):
            decimal = -decimal
        return decimal
    except (TypeError, ValueError):
        return None
