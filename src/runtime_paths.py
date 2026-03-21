"""Runtime path helpers for keeping source originals separate from generated data."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class RuntimeDirectories:
    """Resolved runtime directories for source originals and generated data."""

    source_dir: Path
    data_dir: Path
    legacy_source_mode: bool = False
    migrated_files: tuple[str, ...] = ()
    duplicate_files: tuple[str, ...] = ()
    hardened_files: int = 0


def resolve_runtime_directories(input_dir: str | Path, output_dir: str | Path) -> RuntimeDirectories:
    """Resolve runtime directories and preserve legacy root photos in a separate source vault."""

    requested_source = Path(input_dir).expanduser().resolve()
    data_dir = Path(output_dir).expanduser().resolve()
    legacy_source_mode = requested_source == data_dir
    source_dir = requested_source
    migrated_files: tuple[str, ...] = ()
    duplicate_files: tuple[str, ...] = ()
    hardened_files = 0

    if legacy_source_mode:
        source_dir = data_dir.parent / f"{data_dir.name}__source_originals"
        source_dir.mkdir(parents=True, exist_ok=True)
        migrated, duplicates = bootstrap_legacy_root_images(source_dir, data_dir)
        migrated_files = tuple(migrated)
        duplicate_files = tuple(duplicates)

    hardened_files = harden_source_vault(source_dir)

    return RuntimeDirectories(
        source_dir=source_dir,
        data_dir=data_dir,
        legacy_source_mode=legacy_source_mode,
        migrated_files=migrated_files,
        duplicate_files=duplicate_files,
        hardened_files=hardened_files,
    )


def bootstrap_legacy_root_images(source_dir: Path, data_dir: Path) -> tuple[list[str], list[str]]:
    """Copy legacy root-level images into the immutable source vault using hash-based dedupe."""

    source_dir.mkdir(parents=True, exist_ok=True)
    existing_hashes = build_hash_index(source_dir)
    migrated_files: list[str] = []
    duplicate_files: list[str] = []

    for candidate in sorted(data_dir.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue

        digest = compute_sha256(candidate)
        if digest in existing_hashes:
            duplicate_files.append(candidate.name)
            continue

        target = build_vault_target(source_dir, candidate.name, digest)
        shutil.copy2(candidate, target)
        set_read_only(target)
        existing_hashes[digest] = target
        migrated_files.append(candidate.name)

    return migrated_files, duplicate_files


def build_hash_index(source_dir: Path) -> dict[str, Path]:
    """Build a SHA-256 index for files already present in the source vault."""

    hash_index: dict[str, Path] = {}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            hash_index[compute_sha256(path)] = path
    return hash_index


def iter_source_images(source_dir: Path):
    """Yield all supported image files inside a source vault."""

    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            yield path


def harden_source_vault(source_dir: Path) -> int:
    """Reapply read-only protection to every source-vault image file."""

    hardened_files = 0
    for path in iter_source_images(source_dir):
        if is_path_writable(path):
            set_read_only(path)
            hardened_files += 1
    return hardened_files


def count_writable_source_files(source_dir: Path) -> int:
    """Count image files in the source vault that still appear writable."""

    return sum(1 for path in iter_source_images(source_dir) if is_path_writable(path))


def build_vault_target(source_dir: Path, original_name: str, digest: str) -> Path:
    """Preserve the original filename when possible, otherwise fall back to a hash suffix."""

    candidate = Path(original_name)
    sanitized_name = candidate.name or f"IMG_{digest[:12]}{candidate.suffix.lower()}"
    target = source_dir / sanitized_name
    if not target.exists():
        return target

    return source_dir / f"{target.stem}-{digest[:12]}{target.suffix.lower()}"


def compute_sha256(path: Path) -> str:
    """Return a stable SHA-256 digest for a file on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_path_writable(path: Path) -> bool:
    """Return True when a file still exposes any owner/group/world write bit."""

    return bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def set_read_only(path: Path) -> None:
    """Mark a source-vault file as read-only."""

    current_mode = path.stat().st_mode
    path.chmod(current_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    if os.name == "nt":
        path.chmod(stat.S_IREAD)