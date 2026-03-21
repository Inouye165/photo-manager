"""Upload ingest pipeline for PhotoFinder.

This module is the gatekeeper for the immutable source vault. It de-duplicates
incoming files by SHA-256, writes clean read-only originals, and creates the
working artifacts needed by the existing detection and identity workflows.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Sequence

from PIL import ExifTags, Image
from pydantic import BaseModel, ConfigDict, field_validator

from src.detection_metadata import DetectionMetadata
from src.image_processor import ImageProcessor
from src.mirror_manager import MirrorManager

try:
    from werkzeug.datastructures import FileStorage
except ImportError:  # pragma: no cover
    FileStorage = object


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".webp"}
ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


class IngestConfig(BaseModel):
    """Filesystem configuration for the upload ingest path."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_originals: Path
    processing_root: Path

    @field_validator("source_originals", "processing_root", mode="before")
    @classmethod
    def _coerce_path(cls, value: object) -> Path:
        return Path(value).expanduser().resolve()


class IngestResult(BaseModel):
    """Structured result for a single upload transaction."""

    duplicate: bool = False
    sha256: str
    clean_name: str
    source_path: Path
    source_relative_path: str
    canonical_path: Path
    canonical_relative_path: str
    original_filename: Optional[str] = None
    duplicate_source_filename: Optional[str] = None
    duplicate_source_relative_path: Optional[str] = None
    has_person: bool = False
    has_animal: bool = False
    detection_count: int = 0
    preview_ready: bool = False


class IngestManager:
    """Persist new uploads into the vault and prime the working pipeline."""

    def __init__(
        self,
        config: IngestConfig,
        mirror_manager: MirrorManager,
        image_processor: Optional[ImageProcessor] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.mirror_manager = mirror_manager
        self.image_processor = image_processor or ImageProcessor()
        self.logger = logger or logging.getLogger("photofinder.ingest")
        self._ensure_layout()

    @property
    def library_dir(self) -> Path:
        """Directory that stores canonical working copies of uploaded files."""

        return self.config.processing_root / "_library"

    @property
    def people_dir(self) -> Path:
        """Directory that stores source images containing people."""

        return self.config.processing_root / "people"

    @property
    def animals_dir(self) -> Path:
        """Directory that stores source images containing animals."""

        return self.config.processing_root / "animals"

    @property
    def others_dir(self) -> Path:
        """Directory that stores source images with no tracked subjects."""

        return self.config.processing_root / "others"

    @property
    def debug_dir(self) -> Path:
        """Directory that stores detection debug overlays."""

        return self.config.processing_root / "_debug_boxes"

    @property
    def crops_people_dir(self) -> Path:
        """Directory that stores cropped training examples for people."""

        return self.config.processing_root / "_training_crops" / "people"

    @property
    def crops_animals_dir(self) -> Path:
        """Directory that stores cropped training examples for animals."""

        return self.config.processing_root / "_training_crops" / "animals"

    def _ensure_layout(self) -> None:
        self.config.source_originals.mkdir(parents=True, exist_ok=True)
        self.config.processing_root.mkdir(parents=True, exist_ok=True)
        for folder in (
            self.library_dir,
            self.people_dir,
            self.animals_dir,
            self.others_dir,
            self.debug_dir,
            self.crops_people_dir,
            self.crops_animals_dir,
        ):
            folder.mkdir(parents=True, exist_ok=True)

    def ingest_upload(self, upload: FileStorage) -> IngestResult:
        """Store a new upload in the immutable vault and build working artifacts."""
        if not getattr(upload, "filename", None):
            raise ValueError("Uploaded file is missing a filename")

        original_name = str(upload.filename)
        suffix = Path(original_name).suffix.lower()
        if suffix not in VALID_EXTENSIONS:
            raise ValueError(f"Unsupported upload type: {suffix or 'unknown'}")

        digest, temp_path = self._stream_to_temp(upload)
        try:
            existing_record = self._find_duplicate(digest)
            if existing_record is not None:
                source_path = existing_record.source_path
                source_relative = existing_record.relative_path
                clean_name = source_path.name
                duplicate_source_filename = source_path.name
                duplicate_source_relative_path = source_relative
                duplicate = True
            else:
                clean_name = self._build_clean_name(digest, suffix)
                source_path = self.config.source_originals / clean_name
                source_relative = source_path.relative_to(self.config.source_originals).as_posix()
                shutil.copyfile(temp_path, source_path)
                self._set_read_only(source_path)
                duplicate_source_filename = None
                duplicate_source_relative_path = None
                duplicate = False

            canonical_path = self.library_dir / clean_name
            if not canonical_path.exists():
                shutil.copyfile(source_path, canonical_path)

            image_metadata = {
                **self._extract_image_metadata(source_path),
                "file_hash": digest,
                "original_filename": original_name,
            }
            existing_summary = self._load_existing_processing_summary(canonical_path, digest) if duplicate else None
            if existing_summary is not None:
                has_person, has_animal, detection_count = existing_summary
            else:
                has_person, has_animal, detection_count = self._process_asset(
                    canonical_path,
                    image_metadata=image_metadata,
                )
            self.mirror_manager.sync_source_index()
            self.mirror_manager.materialize_derivative(source_relative, variant="thumb")
            self.mirror_manager.materialize_derivative(source_relative, variant="preview")

            return IngestResult(
                duplicate=duplicate,
                sha256=digest,
                clean_name=clean_name,
                source_path=source_path,
                source_relative_path=source_relative,
                canonical_path=canonical_path,
                canonical_relative_path=canonical_path.relative_to(self.config.processing_root).as_posix(),
                original_filename=original_name,
                duplicate_source_filename=duplicate_source_filename,
                duplicate_source_relative_path=duplicate_source_relative_path,
                has_person=has_person,
                has_animal=has_animal,
                detection_count=detection_count,
                preview_ready=True,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def ingest_uploads(self, uploads: Sequence[FileStorage]) -> list[IngestResult]:
        """Process multiple uploads through the same immutable-ingest pipeline."""

        results: list[IngestResult] = []
        for upload in uploads:
            results.append(self.ingest_upload(upload))
        return results

    def _stream_to_temp(self, upload: FileStorage) -> tuple[str, Path]:
        digest = hashlib.sha256()
        upload.stream.seek(0)
        with NamedTemporaryFile(delete=False, suffix=Path(str(upload.filename)).suffix.lower()) as handle:
            temp_path = Path(handle.name)
            while True:
                chunk = upload.stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
        upload.stream.seek(0)
        return digest.hexdigest(), temp_path

    def _find_duplicate(self, sha256: str):
        report = self.mirror_manager.sync_source_index()
        for record in report.records:
            if record.sha256 == sha256:
                return record
        return None

    def _build_clean_name(self, sha256: str, suffix: str) -> str:
        date_stamp = datetime.now().strftime("%Y%m%d")
        return f"IMG_{date_stamp}_{sha256[:12]}{suffix}"

    def _set_read_only(self, path: Path) -> None:
        current_mode = path.stat().st_mode
        path.chmod(current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        if os.name == "nt":
            path.chmod(stat.S_IREAD)

    def _extract_image_metadata(self, image_path: Path) -> dict[str, str]:
        """Read lightweight EXIF metadata needed by the search and labeling UI."""

        metadata: dict[str, str] = {}

        try:
            with Image.open(image_path) as image:
                exif = image.getexif()
                if not exif:
                    return metadata

                exif_map = {
                    str(ExifTags.TAGS.get(tag_id, tag_id)): value
                    for tag_id, value in exif.items()
                }
                captured_at = (
                    exif_map.get("DateTimeOriginal")
                    or exif_map.get("DateTimeDigitized")
                    or exif_map.get("DateTime")
                )
                if captured_at:
                    normalized = (
                        str(captured_at)
                        .strip()
                        .replace(":", "-", 2)
                        .replace(" ", "T", 1)
                    )
                    metadata["captured_at"] = normalized
        except Exception as exc:  # pragma: no cover - best effort metadata extraction
            self.logger.warning("Could not extract EXIF metadata for %s: %s", image_path, exc)

        return metadata

    def _load_existing_processing_summary(self, canonical_path: Path, digest: str) -> tuple[bool, bool, int] | None:
        """Reuse existing detection metadata for duplicate uploads when it matches the same file hash."""

        metadata_manager = DetectionMetadata(str(self.config.processing_root))
        existing_record = metadata_manager.get_detections(str(canonical_path))
        if not existing_record:
            return None

        record_metadata = existing_record.get("metadata") or {}
        existing_hash = str(record_metadata.get("file_hash") or "").strip()
        if existing_hash and existing_hash != digest:
            return None

        detections = existing_record.get("detections") or []
        if not detections:
            return None

        has_person = any(detection.get("class_name") == "person" for detection in detections)
        has_animal = any(detection.get("class_name") in ANIMAL_CLASSES for detection in detections)
        return has_person, has_animal, len(detections)

    def _process_asset(
        self,
        canonical_path: Path,
        *,
        image_metadata: Optional[dict[str, str]] = None,
    ) -> tuple[bool, bool, int]:
        """Run detection, write derivatives, and persist detection metadata."""

        subjects = self.image_processor.detect_subjects(str(canonical_path))
        has_person = bool(subjects.get("has_person"))
        has_animal = bool(subjects.get("has_animal"))

        if has_person:
            shutil.copyfile(canonical_path, self.people_dir / canonical_path.name)
        if has_animal:
            shutil.copyfile(canonical_path, self.animals_dir / canonical_path.name)
        if not has_person and not has_animal:
            shutil.copyfile(canonical_path, self.others_dir / canonical_path.name)

        debug_path = self.debug_dir / f"{canonical_path.stem}_debug.jpg"
        debug_result = self.image_processor.annotate_detections(str(canonical_path), str(debug_path))
        detailed_detections = self.image_processor.get_detailed_detections(str(canonical_path))

        for index, detection in enumerate(detailed_detections):
            class_name = detection["class_name"]
            if class_name == "person":
                crop_dir = self.crops_people_dir
            elif class_name in ANIMAL_CLASSES:
                crop_dir = self.crops_animals_dir
            else:
                continue

            crop_path = crop_dir / f"{canonical_path.stem}_crop_{index}_{class_name}.jpg"
            if self.image_processor.generate_crop(str(canonical_path), detection["bbox"], str(crop_path)):
                detection["crop_path"] = os.path.relpath(
                    crop_path,
                    self.config.processing_root,
                ).replace("\\", "/")

        metadata_manager = DetectionMetadata(str(self.config.processing_root))
        metadata_manager.add_detections(
            str(canonical_path),
            detailed_detections,
            str(debug_path) if debug_result.get("saved") else None,
            extra_metadata=image_metadata,
        )
        return has_person, has_animal, len(detailed_detections)


def image_bytes_to_upload(filename: str, image: Image.Image) -> FileStorage:
    """Test helper for constructing an upload object from an in-memory PIL image."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return FileStorage(stream=buffer, filename=filename)
