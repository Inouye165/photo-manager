"""Tests for the Flask API endpoints (app.api)."""

import json
from app.api.factory import create_app
from app.config.settings import Settings


def _test_client(tmp_path):
    """Create a test client with an isolated workspace."""
    settings = Settings(
        vault_root=tmp_path / "vault",
        scan_roots=str(tmp_path / "photos"),
        vector_provider="qdrant_local",  # will fallback to in-memory if qdrant absent
        scan_on_startup=False,
        enable_upload=True,
        secret_key="test-secret",
    )
    (tmp_path / "photos").mkdir()
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def test_list_assets_empty(tmp_path):
    client = _test_client(tmp_path)
    resp = client.get("/api/assets")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["total"] == 0


def test_text_search_requires_query(tmp_path):
    client = _test_client(tmp_path)
    resp = client.post("/api/search/text", json={})
    assert resp.status_code == 400


def test_text_search_empty(tmp_path):
    client = _test_client(tmp_path)
    resp = client.post("/api/search/text", json={"query": "sunset"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["results"] == []


def test_label_crud(tmp_path):
    client = _test_client(tmp_path)

    # Add
    resp = client.post("/api/identity/labels", json={
        "image_relative_path": "photo.jpg",
        "assigned_label": "Alice",
        "detected_class": "person",
        "bbox": [10, 20, 100, 200],
        "status": "confirmed",
    })
    assert resp.status_code == 201
    lid = resp.get_json()["id"]

    # Get
    resp = client.get(f"/api/identity/labels/{lid}")
    assert resp.status_code == 200
    assert resp.get_json()["assigned_label"] == "Alice"

    # Update
    resp = client.patch(f"/api/identity/labels/{lid}", json={"status": "rejected"})
    assert resp.status_code == 200

    # Delete
    resp = client.delete(f"/api/identity/labels/{lid}")
    assert resp.status_code == 200


def test_label_stats(tmp_path):
    client = _test_client(tmp_path)
    resp = client.get("/api/identity/stats")
    assert resp.status_code == 200
    assert "total" in resp.get_json()


def test_scan_endpoint(tmp_path):
    client = _test_client(tmp_path)
    resp = client.post("/api/assets/scan")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["discovered"] == 0
