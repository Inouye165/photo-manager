"""Vector store abstraction – wraps Qdrant for both scene and region collections.

Supports dual-collection layout (scene_embeddings + region_embeddings) as
configured in settings, with an in-memory fallback for dev/testing.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        HnswConfigDiff,
        MatchValue,
        OptimizersConfigDiff,
        PointStruct,
        SearchParams,
        VectorParams,
    )
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False


# ── Data classes ─────────────────────────────────────────────────────────

class VectorRecord(BaseModel):
    """One stored embedding + payload."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relative_path: str
    asset_id: Optional[str] = None
    identity_label: str = ""
    subject_type: str = "unknown"
    class_name: Optional[str] = None
    record_status: Literal["pending", "confirmed", "rejected"] = "confirmed"
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    record_id: str
    identity_label: str
    subject_type: str
    class_name: Optional[str] = None
    score: float
    relative_path: str
    record_status: str = "confirmed"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── In-memory fallback ───────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class InMemoryIndex:
    """Cosine-similarity index for dev / testing."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def upsert(self, rec: VectorRecord) -> None:
        self._records[rec.record_id] = rec

    def query(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        subject_type: Optional[str] = None,
        class_name: Optional[str] = None,
        confirmed_only: bool = True,
    ) -> list[SearchHit]:
        q = _norm(np.asarray(vector, dtype=np.float32))
        hits: list[SearchHit] = []
        for r in self._records.values():
            if confirmed_only and r.record_status != "confirmed":
                continue
            if subject_type and r.subject_type not in (subject_type, "unknown"):
                continue
            if class_name and r.class_name != class_name:
                continue
            c = _norm(np.asarray(r.vector, dtype=np.float32))
            score = float(np.dot(q, c))
            hits.append(SearchHit(
                record_id=r.record_id,
                identity_label=r.identity_label,
                subject_type=r.subject_type,
                class_name=r.class_name,
                score=max(-1.0, min(1.0, score)),
                relative_path=r.relative_path,
                record_status=r.record_status,
                metadata=r.metadata,
            ))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def close(self) -> None:
        pass


# ── Qdrant-backed index ─────────────────────────────────────────────────

class QdrantIndex:
    """Production vector index backed by Qdrant (local or remote)."""

    def __init__(
        self,
        *,
        collection_name: str,
        dim: int,
        location: Optional[Path] = None,
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        if not _HAS_QDRANT:
            raise RuntimeError("qdrant-client is not installed")

        self.collection_name = collection_name

        if location is not None:
            location.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(location), force_disable_check_same_thread=True)
        else:
            self.client = QdrantClient(host=host, port=port)

        self._search_params = SearchParams(hnsw_ef=256, exact=False, indexed_only=False)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                on_disk_payload=True,
                hnsw_config=HnswConfigDiff(m=32, ef_construct=256, on_disk=True),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
            )

    def upsert(self, rec: VectorRecord) -> None:
        payload = {
            "record_id": rec.record_id,
            "identity_label": rec.identity_label,
            "relative_path": rec.relative_path,
            "subject_type": rec.subject_type,
            "class_name": rec.class_name,
            "record_status": rec.record_status,
            "asset_id": rec.asset_id,
            "metadata": rec.metadata,
        }
        pid = uuid.uuid5(uuid.NAMESPACE_URL, rec.record_id).hex
        point = PointStruct(id=pid, vector=rec.vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)

    def query(
        self,
        vector: Sequence[float],
        top_k: int = 10,
        subject_type: Optional[str] = None,
        class_name: Optional[str] = None,
        confirmed_only: bool = True,
    ) -> list[SearchHit]:
        conditions = []
        if subject_type:
            conditions.append(FieldCondition(key="subject_type", match=MatchValue(value=subject_type)))
        if class_name:
            conditions.append(FieldCondition(key="class_name", match=MatchValue(value=class_name)))
        if confirmed_only:
            conditions.append(FieldCondition(key="record_status", match=MatchValue(value="confirmed")))

        qf = Filter(must=conditions) if conditions else None
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=top_k,
            query_filter=qf,
            search_params=self._search_params,
            with_payload=True,
        ).points

        hits: list[SearchHit] = []
        for pt in resp:
            p = pt.payload or {}
            hits.append(SearchHit(
                record_id=str(p.get("record_id", pt.id)),
                identity_label=str(p.get("identity_label", "")),
                subject_type=str(p.get("subject_type", "unknown")),
                class_name=p.get("class_name"),
                score=float(pt.score),
                relative_path=str(p.get("relative_path", "")),
                record_status=str(p.get("record_status", "confirmed")),
                metadata=dict(p.get("metadata", {})),
            ))
        return hits

    def close(self) -> None:
        self.client.close()


# ── Factory ──────────────────────────────────────────────────────────────

def build_index(
    *,
    collection_name: str,
    dim: int,
    provider: str = "qdrant_local",
    location: Optional[Path] = None,
    host: str = "localhost",
    port: int = 6333,
) -> InMemoryIndex | QdrantIndex:
    """Create the right vector index from settings."""
    if provider == "qdrant_local":
        try:
            return QdrantIndex(collection_name=collection_name, dim=dim, location=location)
        except RuntimeError:
            logger.warning("Qdrant unavailable, falling back to in-memory index")
            return InMemoryIndex()
    if provider == "qdrant_remote":
        try:
            return QdrantIndex(collection_name=collection_name, dim=dim, host=host, port=port)
        except RuntimeError:
            logger.warning("Qdrant unavailable, falling back to in-memory index")
            return InMemoryIndex()
    return InMemoryIndex()
