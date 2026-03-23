"""Immutable source vault for canonical originals.

Encapsulates SHA-256 dedup, read-only enforcement, and ingest into the
content-addressed vault.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from app.storage.paths import (
    ALLOWED_IMAGE_SUFFIXES,
    compute_sha256,
    harden_vault,
    iter_images,
    set_read_only,
    vault_target,
)

logger = logging.getLogger(__name__)


class Vault:
    """Content-addressed, read-only image store."""

    def __init__(self, vault_root: Path) -> None:
        self.root = vault_root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ── Hash index ───────────────────────────────────────────────────────

    def build_hash_index(self) -> dict[str, Path]:
        """Return ``{sha256: Path}`` for every image already in the vault."""
        return {compute_sha256(p): p for p in iter_images(self.root)}

    # ── Ingest ───────────────────────────────────────────────────────────

    def ingest(
        self,
        source: Path,
        *,
        expected_hash: Optional[str] = None,
    ) -> tuple[Path, str, bool]:
        """Copy *source* into the vault.

        Returns ``(vault_path, sha256, is_duplicate)``.
        """
        digest = expected_hash or compute_sha256(source)
        idx = self.build_hash_index()
        if digest in idx:
            return idx[digest], digest, True

        target = vault_target(self.root, source.name, digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        set_read_only(target)
        return target, digest, False

    # ── Maintenance ──────────────────────────────────────────────────────

    def harden(self) -> int:
        """Re-apply read-only protection to all vault files. Returns count."""
        return harden_vault(self.root)

    def contains_hash(self, sha256: str) -> bool:
        return sha256 in self.build_hash_index()
