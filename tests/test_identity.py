"""Tests for app.identity.manager."""

from app.identity.manager import IdentityManager, LabelRecord


def test_add_and_get(tmp_path):
    mgr = IdentityManager(tmp_path / "labels.json")
    rec = LabelRecord(
        image_relative_path="photo.jpg",
        detected_class="person",
        assigned_label="Alice",
        bbox=[10, 20, 100, 200],
        status="confirmed",
    )
    lid = mgr.add(rec)
    assert mgr.get(lid) is not None
    assert len(mgr) == 1


def test_update_and_delete(tmp_path):
    mgr = IdentityManager(tmp_path / "labels.json")
    rec = LabelRecord(image_relative_path="a.jpg", assigned_label="Bob", status="pending")
    lid = mgr.add(rec)
    mgr.update_label(lid, status="confirmed")
    assert mgr.get(lid).status == "confirmed"
    mgr.delete(lid)
    assert len(mgr) == 0


def test_unique_labels(tmp_path):
    mgr = IdentityManager(tmp_path / "labels.json")
    for name in ("Alice", "Bob", "Alice"):
        mgr.add(LabelRecord(
            image_relative_path="x.jpg", assigned_label=name, status="confirmed",
        ))
    assert mgr.unique_labels() == ["Alice", "Bob"]


def test_persistence(tmp_path):
    path = tmp_path / "labels.json"
    mgr1 = IdentityManager(path)
    mgr1.add(LabelRecord(image_relative_path="x.jpg", assigned_label="Ron", status="confirmed"))
    mgr2 = IdentityManager(path)
    assert len(mgr2) == 1
    assert mgr2.all()[0].assigned_label == "Ron"
