"""Semantic search engine – unified query interface over dual vector indices.

Supports:
- Natural-language text search (CLIP text → scene + region index)
- Similar-image search (CLIP image → scene + region index)
- Filtered retrieval by identity, subject type, class name
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from app.search.vector_store import InMemoryIndex, QdrantIndex, SearchHit, VectorRecord, build_index
from app.vision.embedder import Embedder

logger = logging.getLogger(__name__)


class SearchEngine:
    """High-level search facade over scene + region vector indices."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        scene_index: InMemoryIndex | QdrantIndex,
        region_index: InMemoryIndex | QdrantIndex,
    ) -> None:
        self.embedder = embedder
        self.scene_index = scene_index
        self.region_index = region_index

    # ── Text search ──────────────────────────────────────────────────────

    def text_search(
        self,
        query: str,
        *,
        top_k: int = 20,
        subject_type: Optional[str] = None,
        class_name: Optional[str] = None,
        scope: str = "both",
    ) -> list[SearchHit]:
        """Search by natural language across scene and/or region indices."""
        vec = self.embedder.embed_text(query)
        return self._query(vec, top_k=top_k, subject_type=subject_type,
                           class_name=class_name, scope=scope)

    # ── Image similarity ─────────────────────────────────────────────────

    def image_search(
        self,
        image_path: Path,
        *,
        top_k: int = 20,
        subject_type: Optional[str] = None,
        class_name: Optional[str] = None,
        scope: str = "both",
    ) -> list[SearchHit]:
        """Find similar images by embedding the query image."""
        vec = self.embedder.embed_image(image_path)
        return self._query(vec, top_k=top_k, subject_type=subject_type,
                           class_name=class_name, scope=scope)

    # ── Upsert helpers ───────────────────────────────────────────────────

    def upsert_scene(self, rec: VectorRecord) -> None:
        self.scene_index.upsert(rec)

    def upsert_region(self, rec: VectorRecord) -> None:
        self.region_index.upsert(rec)

    # ── Private ──────────────────────────────────────────────────────────

    def _query(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        subject_type: Optional[str],
        class_name: Optional[str],
        scope: str,
    ) -> list[SearchHit]:
        results: list[SearchHit] = []
        vec_list = vector.tolist()

        if scope in ("both", "scene"):
            results.extend(self.scene_index.query(
                vec_list, top_k=top_k,
                subject_type=subject_type, class_name=class_name,
            ))

        if scope in ("both", "region"):
            results.extend(self.region_index.query(
                vec_list, top_k=top_k,
                subject_type=subject_type, class_name=class_name,
            ))

        # Deduplicate by record_id, keep highest score
        by_id: dict[str, SearchHit] = {}
        for hit in results:
            existing = by_id.get(hit.record_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.record_id] = hit

        merged = sorted(by_id.values(), key=lambda h: h.score, reverse=True)
        return merged[:top_k]
