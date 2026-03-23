"""Typed application settings loaded from environment variables.

Usage::

    from app.config.settings import get_settings
    settings = get_settings()
    print(settings.vault_root)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Single source of truth for every tuneable knob in the system.

    Values come from environment variables prefixed with ``PHOTOINTEL_``,
    a ``.env`` file in the project root, or their coded defaults.
    """

    model_config = {"env_prefix": "PHOTOINTEL_", "env_file": ".env", "extra": "ignore"}

    # ── Storage ──────────────────────────────────────────────────────────
    scan_roots: str = "~/Pictures"
    vault_root: Path = Path("~/.photointel")

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 5000
    workers: int = 4
    max_upload_mb: int = 128
    secret_key: str = "change-me-to-a-random-string"

    # ── Vector DB ────────────────────────────────────────────────────────
    vector_provider: Literal["qdrant_local", "qdrant_remote"] = "qdrant_local"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_scene_collection: str = "scene_embeddings"
    qdrant_region_collection: str = "region_embeddings"

    # ── Models ───────────────────────────────────────────────────────────
    detector_model: str = "yolov8m.pt"
    detector_confidence_person: float = 0.50
    detector_confidence_animal: float = 0.40
    detector_confidence_general: float = 0.25
    embedding_model: str = "openai/clip-vit-large-patch14"
    embedding_dim: int = 768
    embedding_device: Literal["auto", "cuda", "mps", "cpu"] = "auto"

    # ── Feature Flags ────────────────────────────────────────────────────
    enable_scene_embeddings: bool = True
    enable_region_embeddings: bool = True
    enable_exif_gps: bool = True
    enable_upload: bool = True
    enable_active_learning: bool = True

    # ── Background Jobs ──────────────────────────────────────────────────
    scan_on_startup: bool = False
    index_batch_size: int = 32
    background_workers: int = 2

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived Helpers ──────────────────────────────────────────────────

    @field_validator("vault_root", mode="before")
    @classmethod
    def _expand_vault(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    @property
    def resolved_scan_roots(self) -> list[Path]:
        """Parse the comma-separated scan-roots string into resolved Paths."""
        return [
            Path(p.strip()).expanduser().resolve()
            for p in self.scan_roots.split(",")
            if p.strip()
        ]

    @property
    def derivatives_dir(self) -> Path:
        return self.vault_root / "derivatives"

    @property
    def vector_db_dir(self) -> Path:
        return self.vault_root / "vector_db"

    @property
    def metadata_dir(self) -> Path:
        return self.vault_root / "metadata"

    @property
    def crops_dir(self) -> Path:
        return self.vault_root / "model_inputs"

    @property
    def logs_dir(self) -> Path:
        return self.vault_root / "logs"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
