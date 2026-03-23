"""Tests for app.storage.vault."""

from pathlib import Path
from app.storage.vault import Vault


def test_ingest_and_dedup(tmp_path):
    vault = Vault(tmp_path / "vault")

    # Create a fake image file
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"fake-image-content")

    path1, sha1, dup1 = vault.ingest(src)
    assert not dup1
    assert path1.exists()

    # Second ingest should detect duplicate
    path2, sha2, dup2 = vault.ingest(src)
    assert dup2
    assert sha1 == sha2


def test_harden(tmp_path):
    vault = Vault(tmp_path / "vault")
    src = tmp_path / "photo.jpg"
    src.write_bytes(b"data")
    vault.ingest(src)
    count = vault.harden()
    # All files should already be read-only from ingest
    assert count == 0
