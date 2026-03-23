"""Tests for app.config.settings."""

import os
from unittest.mock import patch

from app.config.settings import Settings


def test_defaults():
    """Settings loads with sensible defaults."""
    with patch.dict(os.environ, {}, clear=False):
        s = Settings()
    assert s.port == 5000
    assert s.embedding_dim == 768
    assert s.enable_scene_embeddings is True


def test_scan_roots_parsing():
    s = Settings(scan_roots="~/Pictures, /mnt/photos")
    roots = s.resolved_scan_roots
    assert len(roots) == 2


def test_vault_root_expansion():
    s = Settings(vault_root="~/.photointel")
    assert s.vault_root.is_absolute()
    assert "~" not in str(s.vault_root)
