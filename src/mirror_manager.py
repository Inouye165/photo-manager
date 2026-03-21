"""Mirror workspace management for immutable photo archives.

This module establishes a hard separation between a read-only source tree and a
mutable mirror workspace that holds every derived artifact, manifest, and log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Literal, Optional
from urllib.parse import quote

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Image.MAX_IMAGE_PIXELS = None

ALLOWED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".heic",
    ".heif",
    ".webp",
}


class MirrorConfig(BaseModel):
    """Filesystem contract for immutable originals and mutable mirror state."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_originals: Path
    mirror_workspace: Path
    derivative_format: Literal["WEBP", "JPEG", "PNG"] = "WEBP"
    crop_format: Literal["JPEG", "PNG", "WEBP"] = "JPEG"
    thumbnail_size: tuple[int, int] = (384, 384)
    preview_size: tuple[int, int] = (1600, 1600)
    crop_size: int = 224
    jpeg_quality: int = Field(default=90, ge=60, le=100)
    webp_quality: int = Field(default=84, ge=50, le=100)
    manifest_name: str = "source_index.json"

    @field_validator("source_originals", "mirror_workspace", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        return Path(value).expanduser().resolve()

    @model_validator(mode="after")
    def _validate_paths(self) -> "MirrorConfig":
        if self.source_originals == self.mirror_workspace:
            raise ValueError("source_originals and mirror_workspace must be different directories")

        try:
            self.mirror_workspace.relative_to(self.source_originals)
        except ValueError:
            return self

        raise ValueError("mirror_workspace must not live inside source_originals")


class SourceAssetRecord(BaseModel):
    """Canonical record for a file discovered in the immutable source tree."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    asset_id: str
    relative_path: str
    source_path: Path
    sha256: str
    size_bytes: int
    modified_ns: int
    width: int
    height: int
    mime_type: Optional[str] = None


class DerivativeSpec(BaseModel):
    """Instructions for generating a browser-safe derivative."""

    variant: Literal["thumb", "preview"]
    relative_path: str
    width: int
    height: int


class CropRequest(BaseModel):
    """Validated crop request for model-ready assets."""

    relative_path: str
    bbox: tuple[int, int, int, int]
    subject_type: Literal["people", "animals", "unknown"] = "unknown"
    identity_hint: Optional[str] = None

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must be expressed as (x1, y1, x2, y2) with positive area")
        return value


class SyncReport(BaseModel):
    """Operational summary for a mirror synchronization pass."""

    discovered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    manifest_path: Path
    records: list[SourceAssetRecord] = Field(default_factory=list)


class MirrorManager:
    """Build and maintain a mutable mirror around a read-only photo archive."""

    def __init__(self, config: MirrorConfig, logger: Optional[logging.Logger] = None):
        """Create a mirror manager bound to one immutable source tree and workspace."""

        self.config = config
        self.logger = logger or logging.getLogger("photofinder.mirror")
        self.ensure_layout()

    @property
    def manifests_dir(self) -> Path:
        """Directory that stores source manifests and related metadata."""

        return self.config.mirror_workspace / "metadata"

    @property
    def manifest_path(self) -> Path:
        """Manifest file path for the synchronized source index."""

        return self.manifests_dir / self.config.manifest_name

    @property
    def derivatives_dir(self) -> Path:
        """Root directory for browser-safe derivative images."""

        return self.config.mirror_workspace / "derivatives"

    @property
    def crops_dir(self) -> Path:
        """Root directory for normalized model-input crops."""

        return self.config.mirror_workspace / "model_inputs" / str(self.config.crop_size)

    def ensure_layout(self) -> None:
        """Create the mutable workspace layout without touching the source tree."""
        self.config.mirror_workspace.mkdir(parents=True, exist_ok=True)

        for folder in (
            self.manifests_dir,
            self.derivatives_dir / "thumb",
            self.derivatives_dir / "preview",
            self.crops_dir,
            self.config.mirror_workspace / "logs",
            self.config.mirror_workspace / "jobs",
            self.config.mirror_workspace / "vector_index",
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def verify_source_contract(self) -> None:
        """Log when the source directory appears writable.

        The application still guarantees safety by never opening source files in a
        write mode, but this warning highlights when the OS permissions do not yet
        reflect the intended production setup.
        """
        if not self.config.source_originals.exists():
            raise FileNotFoundError(f"source_originals does not exist: {self.config.source_originals}")

        if os.access(self.config.source_originals, os.W_OK):
            self.logger.warning(
                "source_originals appears writable; enforce read-only permissions at the filesystem level: %s",
                self.config.source_originals,
            )

    def sync_source_index(self, force_checksums: bool = False) -> SyncReport:
        """Snapshot the source tree into a manifest stored only in the mirror."""
        self.verify_source_contract()

        existing_records = self._load_manifest()
        discovered_records: list[SourceAssetRecord] = []
        created = 0
        updated = 0
        unchanged = 0

        for source_path in self._iter_source_images():
            relative_path = source_path.relative_to(self.config.source_originals).as_posix()
            stat_result = source_path.stat()
            existing = existing_records.get(relative_path)
            should_refresh = (
                force_checksums
                or existing is None
                or existing.modified_ns != stat_result.st_mtime_ns
                or existing.size_bytes != stat_result.st_size
            )

            if should_refresh:
                record = self._build_source_record(source_path, relative_path)
                if existing is None:
                    created += 1
                else:
                    updated += 1
            else:
                record = existing
                unchanged += 1

            discovered_records.append(record)

        report = SyncReport(
            discovered=len(discovered_records),
            created=created,
            updated=updated,
            unchanged=unchanged,
            manifest_path=self.manifest_path,
            records=discovered_records,
        )
        self._write_manifest(report.records)
        return report

    def materialize_derivative(self, relative_path: str, variant: Literal["thumb", "preview"] = "thumb") -> Path:
        """Create a browser-safe derivative in the mirror workspace."""
        record = self._get_source_record(relative_path)
        size = self.config.thumbnail_size if variant == "thumb" else self.config.preview_size
        artifact_path = self._derivative_target_path(record, variant, self.config.derivative_format.lower())

        if artifact_path.exists():
            return artifact_path

        with Image.open(record.source_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image(image, artifact_path, self.config.derivative_format)

        return artifact_path

    def materialize_external_derivative(
        self,
        absolute_path: Path,
        derivative_key: str,
        variant: Literal["thumb", "preview"] = "thumb",
    ) -> Path:
        """Create a browser-safe derivative for a file that already lives in the mirror workspace."""
        source_path = absolute_path.expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"external asset does not exist: {source_path}")

        size = self.config.thumbnail_size if variant == "thumb" else self.config.preview_size
        extension = self.config.derivative_format.lower()
        artifact_path = self.derivatives_dir / variant / "external" / f"{quote(derivative_key, safe='/')}.{extension}"

        if artifact_path.exists() and artifact_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            return artifact_path

        with Image.open(source_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.thumbnail(size, Image.Resampling.LANCZOS)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image(image, artifact_path, self.config.derivative_format)

        return artifact_path

    def materialize_crop(self, request: CropRequest) -> Path:
        """Create a normalized square crop for model ingestion inside the mirror."""
        record = self._get_source_record(request.relative_path)
        x1, y1, x2, y2 = request.bbox
        artifact_path = self._crop_target_path(record, request)

        if artifact_path.exists():
            return artifact_path

        with Image.open(record.source_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            if image.mode != "RGB":
                image = image.convert("RGB")

            width, height = image.size
            left = max(0, min(x1, width))
            top = max(0, min(y1, height))
            right = max(left + 1, min(x2, width))
            bottom = max(top + 1, min(y2, height))

            crop = image.crop((left, top, right, bottom))
            fitted = ImageOps.fit(
                crop,
                (self.config.crop_size, self.config.crop_size),
                method=Image.Resampling.LANCZOS,
                bleed=0.0,
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_image(fitted, artifact_path, self.config.crop_format)

        return artifact_path

    def _iter_source_images(self) -> Iterable[Path]:
        for path in sorted(self.config.source_originals.rglob("*")):
            if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
                yield path

    def _build_source_record(self, source_path: Path, relative_path: str) -> SourceAssetRecord:
        stat_result = source_path.stat()
        sha256 = self._compute_sha256(source_path)
        with Image.open(source_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            width, height = image.size

        asset_id = sha256[:20]
        mime_type, _ = mimetypes.guess_type(source_path.name)
        return SourceAssetRecord(
            asset_id=asset_id,
            relative_path=relative_path,
            source_path=source_path,
            sha256=sha256,
            size_bytes=stat_result.st_size,
            modified_ns=stat_result.st_mtime_ns,
            width=width,
            height=height,
            mime_type=mime_type,
        )

    def _compute_sha256(self, source_path: Path) -> str:
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _derivative_target_path(self, record: SourceAssetRecord, variant: str, extension: str) -> Path:
        rel_parent = Path(record.relative_path).parent
        file_stem = Path(record.relative_path).stem
        suffix = f"-{record.asset_id[:12]}-{variant}.{extension}"
        return self.derivatives_dir / variant / rel_parent / f"{file_stem}{suffix}"

    def _crop_target_path(self, record: SourceAssetRecord, request: CropRequest) -> Path:
        rel_parent = Path(record.relative_path).parent
        file_stem = Path(record.relative_path).stem
        identity_slug = (request.identity_hint or request.subject_type or "crop").replace("/", "-").replace("\\", "-")
        extension = self.config.crop_format.lower()
        return self.crops_dir / request.subject_type / rel_parent / f"{file_stem}-{identity_slug}-{record.asset_id[:10]}.{extension}"

    def _save_image(self, image: Image.Image, path: Path, image_format: str) -> None:
        if image_format == "WEBP":
            image.save(path, format=image_format, quality=self.config.webp_quality, method=6)
            return

        if image_format == "JPEG":
            image.save(path, format=image_format, quality=self.config.jpeg_quality, optimize=True)
            return

        image.save(path, format=image_format)

    def _load_manifest(self) -> Dict[str, SourceAssetRecord]:
        if not self.manifest_path.exists():
            return {}

        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = payload.get("records", []) if isinstance(payload, dict) else payload
        return {
            record_data["relative_path"]: SourceAssetRecord.model_validate(record_data)
            for record_data in records
        }

    def _write_manifest(self, records: Iterable[SourceAssetRecord]) -> None:
        serializable = {
            "records": [record.model_dump(mode="json") for record in records],
        }

        with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=self.manifests_dir, suffix=".tmp") as handle:
            json.dump(serializable, handle, indent=2)
            temp_path = Path(handle.name)

        temp_path.replace(self.manifest_path)

    def _get_source_record(self, relative_path: str) -> SourceAssetRecord:
        normalized = Path(relative_path).as_posix()
        records = self._load_manifest()
        record = records.get(normalized)
        if record is not None:
            return record

        source_path = (self.config.source_originals / normalized).resolve()
        if not source_path.exists() or self.config.source_originals not in source_path.parents and source_path != self.config.source_originals:
            raise FileNotFoundError(f"source asset not found in source_originals: {relative_path}")

        return self._build_source_record(source_path, normalized)
