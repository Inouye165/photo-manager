"""Tests for app.catalog.registry."""

from pathlib import Path
from app.catalog.registry import AssetRecord, AssetRegistry


def _make_record(rel: str = "photo.jpg", sha: str = "abc123") -> AssetRecord:
    return AssetRecord(
        asset_id=sha[:20],
        relative_path=rel,
        source_path="/tmp/photo.jpg",
        sha256=sha,
        size_bytes=1024,
        modified_ns=0,
        width=100,
        height=100,
    )


def test_upsert_and_get(tmp_path):
    reg = AssetRegistry(tmp_path / "registry.json")
    rec = _make_record()
    reg.upsert(rec)
    assert "photo.jpg" in reg
    assert reg.get("photo.jpg") is not None
    assert len(reg) == 1


def test_save_and_reload(tmp_path):
    path = tmp_path / "registry.json"
    reg = AssetRegistry(path)
    reg.upsert(_make_record("a.jpg", "aaa"))
    reg.upsert(_make_record("b.jpg", "bbb"))
    reg.save()

    reg2 = AssetRegistry(path)
    assert len(reg2) == 2
    assert reg2.get("a.jpg").sha256 == "aaa"


def test_remove(tmp_path):
    reg = AssetRegistry(tmp_path / "r.json")
    reg.upsert(_make_record())
    reg.remove("photo.jpg")
    assert len(reg) == 0
