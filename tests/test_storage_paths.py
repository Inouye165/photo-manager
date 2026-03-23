"""Tests for app.storage.paths."""

import tempfile
from pathlib import Path

from app.storage.paths import compute_sha256, iter_images, vault_target


def test_compute_sha256(tmp_path):
    f = tmp_path / "test.jpg"
    f.write_bytes(b"hello world")
    digest = compute_sha256(f)
    assert len(digest) == 64
    assert digest == compute_sha256(f)  # deterministic


def test_iter_images(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    found = list(iter_images(tmp_path))
    assert len(found) == 2


def test_vault_target_no_collision(tmp_path):
    target = vault_target(tmp_path, "photo.jpg", "abc123")
    assert target.name == "photo.jpg"


def test_vault_target_collision(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"x")
    target = vault_target(tmp_path, "photo.jpg", "abc123")
    assert "abc123" in target.name
