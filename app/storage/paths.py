"""Path resolution and vault hardening.

Thin wrapper around the original ``src.runtime_paths`` extracted into the
new package layout.  The heavy lifting (SHA-256, read-only bits, legacy
migration) stays in the original module so nothing breaks while both import
paths coexist during the transition.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Iterator

ALLOWED_IMAGE_SUFFIXES = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tiff", ".tif", ".heic", ".heif", ".webp",
})


# ── Hashing ──────────────────────────────────────────────────────────────

def compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── Image iteration ──────────────────────────────────────────────────────

def iter_images(root: Path) -> Iterator[Path]:
    """Yield every supported image file under *root*, sorted for determinism."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES:
            yield p


# ── Read-only enforcement ────────────────────────────────────────────────

def set_read_only(path: Path) -> None:
    """Strip write bits from *path*. On Windows, also set the read-only attr."""
    mode = path.stat().st_mode
    path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    if os.name == "nt":
        path.chmod(stat.S_IREAD)


def is_writable(path: Path) -> bool:
    return bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def harden_vault(root: Path) -> int:
    """Re-apply read-only protection to every image under *root*."""
    count = 0
    for p in iter_images(root):
        if is_writable(p):
            set_read_only(p)
            count += 1
    return count


# ── Vault target naming ─────────────────────────────────────────────────

def vault_target(vault_dir: Path, original_name: str, digest: str) -> Path:
    """Pick a collision-free filename inside the vault."""
    stem = Path(original_name).stem or f"IMG_{digest[:12]}"
    suffix = Path(original_name).suffix.lower() or ".jpg"
    candidate = vault_dir / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    return vault_dir / f"{stem}-{digest[:12]}{suffix}"
