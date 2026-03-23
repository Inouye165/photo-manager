"""Tests for app.search.vector_store (in-memory index)."""

import numpy as np

from app.search.vector_store import InMemoryIndex, VectorRecord


def _random_record(label: str = "Alice", dim: int = 768) -> VectorRecord:
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return VectorRecord(
        relative_path="photo.jpg",
        identity_label=label,
        subject_type="person",
        vector=vec.tolist(),
    )


def test_upsert_and_query():
    idx = InMemoryIndex()
    rec = _random_record("Alice")
    idx.upsert(rec)

    hits = idx.query(rec.vector, top_k=5)
    assert len(hits) == 1
    assert hits[0].identity_label == "Alice"
    assert hits[0].score > 0.99  # self-similarity


def test_empty_index():
    idx = InMemoryIndex()
    vec = np.random.randn(768).tolist()
    hits = idx.query(vec, top_k=5)
    assert len(hits) == 0


def test_confirmed_filter():
    idx = InMemoryIndex()
    rec = _random_record("Bob")
    rec.record_status = "pending"
    idx.upsert(rec)

    # confirmed_only should exclude pending
    hits = idx.query(rec.vector, top_k=5, confirmed_only=True)
    assert len(hits) == 0

    # without filter should include
    hits = idx.query(rec.vector, top_k=5, confirmed_only=False)
    assert len(hits) == 1
