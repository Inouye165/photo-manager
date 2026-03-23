"""Filesystem scanner – discovers images in configured scan roots.

Walks one or more directory trees looking for supported image files,
hashes them, records dimensions, and upserts into the asset registry.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from app.catalog.registry import AssetRecord, AssetRegistry
from app.storage.paths import ALLOWED_IMAGE_SUFFIXES, compute_sha256

logger = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None


class ScanResult:
    __slots__ = ("discovered", "created", "updated", "unchanged", "errors")

    def __init__(self) -> None:
        self.discovered = 0
        self.created = 0
        self.updated = 0
        self.unchanged = 0
        self.errors: list[str] = []


def scan(
    roots: Sequence[Path],
    registry: AssetRegistry,
    *,
    force_checksums: bool = False,
) -> ScanResult:
    """Walk all *roots* and upsert every image into *registry*.

    Returns a ``ScanResult`` summarising the pass.
    """
    result = ScanResult()

    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            logger.warning("Scan root does not exist or is not a directory: %s", root)
            continue

        for image_path in sorted(root.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
                continue

            result.discovered += 1
            relative = image_path.relative_to(root).as_posix()

            try:
                existing = registry.get(relative)
                stat = image_path.stat()

                needs_refresh = (
                    force_checksums
                    or existing is None
                    or existing.modified_ns != stat.st_mtime_ns
                    or existing.size_bytes != stat.st_size
                )

                if not needs_refresh and existing is not None:
                    result.unchanged += 1
                    continue

                sha = compute_sha256(image_path)
                with Image.open(image_path) as img:
                    img = ImageOps.exif_transpose(img)
                    w, h = img.size

                mime, _ = mimetypes.guess_type(image_path.name)

                rec = AssetRecord(
                    asset_id=sha[:20],
                    relative_path=relative,
                    source_path=str(image_path),
                    sha256=sha,
                    size_bytes=stat.st_size,
                    modified_ns=stat.st_mtime_ns,
                    width=w,
                    height=h,
                    mime_type=mime,
                )
                registry.upsert(rec)

                if existing is None:
                    result.created += 1
                else:
                    result.updated += 1

            except Exception as exc:
                result.errors.append(f"{image_path}: {exc}")
                logger.warning("Scanner error on %s: %s", image_path, exc)

    registry.save()
    return result
