"""GPU-aware active learning blueprint for PhotoFinder.

The current repo still uses lightweight heuristics. This module adds the next
layer: a CLIP-style embedding backbone, an active-learning verification queue,
and a vector index that can update immediately when new labels are confirmed.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Optional, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import torch
except ImportError:  # pragma: no cover - optional at runtime
    torch = None

try:
    from transformers import AutoProcessor, CLIPModel
except ImportError:  # pragma: no cover - optional at runtime
    AutoProcessor = None
    CLIPModel = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:  # pragma: no cover - optional at runtime
    QdrantClient = None
    Distance = None
    FieldCondition = None
    Filter = None
    MatchValue = None
    PointStruct = None
    VectorParams = None


class BackboneConfig(BaseModel):
    """Embedding backbone configuration for few-shot image intelligence."""

    model_name: str = "openai/clip-vit-large-patch14"
    processor_name: Optional[str] = None
    batch_size: int = Field(default=12, ge=1)
    normalize_embeddings: bool = True
    use_half_precision: bool = True
    expected_embedding_dim: int = Field(default=768, ge=128)


class VectorStoreConfig(BaseModel):
    """Vector index configuration.

    Defaulting to in_memory keeps this repo import-safe. Switching to qdrant is a
    one-line config change once the service is available.
    """

    provider: Literal["in_memory", "qdrant"] = "in_memory"
    collection_name: str = "photofinder_embeddings"
    location: Optional[Path] = None
    host: Optional[str] = None
    port: Optional[int] = Field(default=6333, ge=1)
    distance: Literal["cosine"] = "cosine"

    @field_validator("location", mode="before")
    @classmethod
    def _coerce_location(cls, value: object) -> Optional[Path]:
        if value in (None, ""):
            return None
        return Path(value).expanduser().resolve()


class ActiveLearningConfig(BaseModel):
    """Decision thresholds for the identity verification loop."""

    auto_accept_threshold: float = Field(default=0.97, ge=0.0, le=1.0)
    verify_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    reject_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=50)
    background_workers: int = Field(default=1, ge=1, le=8)
    schedule_fine_tune: bool = True


class IntelligenceConfig(BaseModel):
    """Top-level production config for active learning and embeddings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace_root: Path
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    device_preference: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    enable_lora_fine_tune: bool = False

    @field_validator("workspace_root", mode="before")
    @classmethod
    def _coerce_workspace_root(cls, value: object) -> Path:
        return Path(value).expanduser().resolve()


class DeviceSummary(BaseModel):
    """Resolved accelerator details for the current runtime."""

    device: str
    accelerator: Literal["cuda", "mps", "cpu"]
    half_precision: bool


class EmbeddingRecord(BaseModel):
    """Stored vector plus metadata needed for active learning."""

    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    relative_path: str
    subject_type: str = "unknown"
    class_name: Optional[str] = None
    identity_label: str
    vector: list[float]
    source_asset_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    human_verified: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NegativeEmbeddingRecord(BaseModel):
    """Stored negative feedback for a label that should be penalized in future matches."""

    record_id: str
    negative_label: str
    relative_path: str
    subject_type: str = "unknown"
    class_name: Optional[str] = None
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SearchHit(BaseModel):
    """Search result returned from the vector index."""

    record_id: str
    identity_label: str
    subject_type: str
    class_name: Optional[str] = None
    score: float
    relative_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationTask(BaseModel):
    """Work item shown in the React identity verification flow."""

    relative_path: str
    subject_type: str = "unknown"
    class_name: Optional[str] = None
    status: Literal["auto_accept", "needs_confirmation", "new_identity"]
    proposed_label: Optional[str] = None
    confidence: float = 0.0
    hits: list[SearchHit] = Field(default_factory=list)
    next_action: Literal["auto_confirm", "queue_yes_no", "collect_label"]
    candidate_image_url: Optional[str] = None
    candidate_full_url: Optional[str] = None
    known_gallery: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationDecision(BaseModel):
    """Human decision fed back into the learning loop."""

    relative_path: str
    subject_type: str = "unknown"
    accepted: bool
    confirmed_label: Optional[str] = None
    schedule_fine_tune: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorIndex(Protocol):
    """Storage abstraction so the ranking engine is not tied to one backend."""

    def upsert(self, record: EmbeddingRecord) -> None:
        """Insert or replace one positive embedding record."""

        raise NotImplementedError

    def query(
        self,
        vector: Sequence[float],
        subject_type: Optional[str],
        top_k: int,
        class_name: Optional[str] = None,
    ) -> list[SearchHit]:
        """Return the highest scoring matches for the provided query vector."""

        raise NotImplementedError


class InMemoryVectorIndex:
    """Simple cosine index used as a safe default and unit-test target."""

    def __init__(self) -> None:
        """Initialize the in-memory positive embedding store."""

        self._records: dict[str, EmbeddingRecord] = {}

    def upsert(self, record: EmbeddingRecord) -> None:
        """Insert or replace one positive embedding record."""

        self._records[record.record_id] = record

    def query(
        self,
        vector: Sequence[float],
        subject_type: Optional[str],
        top_k: int,
        class_name: Optional[str] = None,
    ) -> list[SearchHit]:
        """Return the top cosine matches from the in-memory positive index."""

        query_vector = _normalize_vector(np.asarray(vector, dtype=np.float32))
        hits: list[SearchHit] = []

        for record in self._records.values():
            if (
                subject_type
                and record.subject_type != subject_type
                and record.subject_type != "unknown"
            ):
                continue

            if class_name and record.class_name != class_name:
                continue

            candidate_vector = _normalize_vector(np.asarray(record.vector, dtype=np.float32))
            score = float(np.dot(query_vector, candidate_vector))
            hits.append(
                SearchHit(
                    record_id=record.record_id,
                    identity_label=record.identity_label,
                    subject_type=record.subject_type,
                    class_name=record.class_name,
                    score=max(-1.0, min(1.0, score)),
                    relative_path=record.relative_path,
                    metadata=record.metadata,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class InMemoryNegativeIndex:
    """Negative-memory store keyed by rejected identity label."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, NegativeEmbeddingRecord]] = {}

    def upsert(self, record: NegativeEmbeddingRecord) -> None:
        """Store or replace one negative example for a label."""

        self._records.setdefault(record.negative_label, {})[record.record_id] = record

    def remove(self, negative_label: str, record_id: str) -> None:
        """Delete one stored negative example for a label."""

        label_records = self._records.get(negative_label, {})
        label_records.pop(record_id, None)
        if not label_records and negative_label in self._records:
            del self._records[negative_label]

    def penalty(
        self,
        negative_label: str,
        vector: Sequence[float],
        class_name: Optional[str] = None,
    ) -> float:
        """Return the strongest similarity penalty for the rejected label."""

        label_records = self._records.get(negative_label, {})
        if not label_records:
            return 0.0

        query_vector = _normalize_vector(np.asarray(vector, dtype=np.float32))
        penalties: list[float] = []

        for record in label_records.values():
            if class_name and record.class_name and record.class_name != class_name:
                continue
            candidate_vector = _normalize_vector(np.asarray(record.vector, dtype=np.float32))
            penalties.append(max(0.0, float(np.dot(query_vector, candidate_vector))))

        return max(penalties) if penalties else 0.0

    def export_records(self) -> list[NegativeEmbeddingRecord]:
        """Flatten the in-memory negative index for persistence."""

        records: list[NegativeEmbeddingRecord] = []
        for label_records in self._records.values():
            records.extend(label_records.values())
        return records

    def load_records(self, records: Sequence[NegativeEmbeddingRecord]) -> None:
        """Replace the in-memory negative index from persisted records."""

        self._records = {}
        for record in records:
            self.upsert(record)


class QdrantVectorIndex:
    """Optional Qdrant-backed index for production-scale search."""

    def __init__(self, config: IntelligenceConfig):
        """Connect to Qdrant and create the collection when required."""

        if (
            QdrantClient is None
            or VectorParams is None
            or PointStruct is None
            or Distance is None
        ):
            raise RuntimeError(
                "qdrant-client is not installed; use provider='in_memory' "
                "or install qdrant-client"
            )

        store_config = config.vector_store
        if store_config.location is not None:
            self.client = QdrantClient(path=str(store_config.location))
        else:
            self.client = QdrantClient(host=store_config.host, port=store_config.port)

        self.collection_name = store_config.collection_name
        collections = {item.name for item in self.client.get_collections().collections}
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=config.backbone.expected_embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, record: EmbeddingRecord) -> None:
        """Insert or replace one positive embedding record in Qdrant."""

        payload = {
            "identity_label": record.identity_label,
            "relative_path": record.relative_path,
            "subject_type": record.subject_type,
            "class_name": record.class_name,
            "metadata": record.metadata,
        }
        point = PointStruct(id=record.record_id, vector=record.vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)

    def query(
        self,
        vector: Sequence[float],
        subject_type: Optional[str],
        top_k: int,
        class_name: Optional[str] = None,
    ) -> list[SearchHit]:
        """Return the top vector matches from the configured Qdrant collection."""

        if Filter is None or FieldCondition is None or MatchValue is None:
            return []

        filter_rules = []
        if subject_type:
            filter_rules.append(
                FieldCondition(
                    key="subject_type",
                    match=MatchValue(value=subject_type),
                )
            )
        if class_name:
            filter_rules.append(
                FieldCondition(
                    key="class_name",
                    match=MatchValue(value=class_name),
                )
            )

        query_filter = Filter(must=filter_rules) if filter_rules else None
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=list(vector),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[SearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                SearchHit(
                    record_id=str(point.id),
                    identity_label=str(payload.get("identity_label", "")),
                    subject_type=str(payload.get("subject_type", subject_type)),
                    class_name=str(payload.get("class_name", "")) or None,
                    score=float(point.score),
                    relative_path=str(payload.get("relative_path", "")),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return results


class IntelligenceCore:
    """Active-learning engine with immediate vector updates and optional fine-tuning."""

    def __init__(self, config: IntelligenceConfig, logger: Optional[logging.Logger] = None):
        """Create the intelligence engine and initialize runtime-backed stores."""

        self.config = config
        self.logger = logger or logging.getLogger("photofinder.intelligence")
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_root / "logs").mkdir(parents=True, exist_ok=True)
        self.device_summary = self._resolve_device()
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.active_learning.background_workers
        )
        self._processor = None
        self._backbone_model = None
        self.vector_index = self._build_vector_index()
        self.negative_index = InMemoryNegativeIndex()
        self._load_negative_index()

    @property
    def negative_index_path(self) -> Path:
        """Path used to persist rejected-label examples across restarts."""

        return self.config.workspace_root / "negative_samples.json"

    def _resolve_device(self) -> DeviceSummary:
        """Resolve the preferred accelerator based on runtime support and config."""

        if torch is None:
            return DeviceSummary(device="cpu", accelerator="cpu", half_precision=False)

        preference = self.config.device_preference
        cuda_available = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if preference in {"auto", "cuda"} and cuda_available:
            return DeviceSummary(
                device="cuda",
                accelerator="cuda",
                half_precision=self.config.backbone.use_half_precision,
            )

        if preference in {"auto", "mps"} and mps_available:
            return DeviceSummary(device="mps", accelerator="mps", half_precision=False)

        return DeviceSummary(device="cpu", accelerator="cpu", half_precision=False)

    def _build_vector_index(self) -> VectorIndex:
        """Create the configured positive vector index, with safe fallback to memory."""

        if self.config.vector_store.provider == "qdrant":
            try:
                return QdrantVectorIndex(self.config)
            except RuntimeError as exc:
                self.logger.warning("Falling back to in-memory vector index: %s", exc)

        return InMemoryVectorIndex()

    def reset_vector_index(self) -> None:
        """Recreate the live vector index from configuration."""
        self.vector_index = self._build_vector_index()

    def _load_negative_index(self) -> None:
        """Restore persisted negative samples when the workspace already has them."""

        if not self.negative_index_path.exists():
            return

        try:
            raw_records = json.loads(self.negative_index_path.read_text(encoding="utf-8"))
            records = [
                NegativeEmbeddingRecord.model_validate(record)
                for record in raw_records
            ]
            self.negative_index.load_records(records)
        except Exception as exc:  # pragma: no cover - defensive runtime load
            self.logger.warning("Could not load negative samples: %s", exc)

    def _save_negative_index(self) -> None:
        """Persist the negative-memory index to disk."""

        serialized = [
            record.model_dump(mode="json")
            for record in self.negative_index.export_records()
        ]
        self.negative_index_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    def load_backbone(self) -> None:
        """Load the embedding backbone lazily so API startup stays responsive."""
        if self._backbone_model is not None and self._processor is not None:
            return

        if CLIPModel is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed. Install torch + transformers "
                "to enable GPU embeddings."
            )

        processor_name = (
            self.config.backbone.processor_name
            or self.config.backbone.model_name
        )
        self._processor = AutoProcessor.from_pretrained(processor_name)
        self._backbone_model = CLIPModel.from_pretrained(self.config.backbone.model_name)
        self._backbone_model.eval()

        if torch is not None:
            self._backbone_model.to(self.device_summary.device)
            if self.device_summary.accelerator == "cuda" and self.device_summary.half_precision:
                self._backbone_model = self._backbone_model.half()

    def embed_image(self, image_path: Path) -> np.ndarray:
        """Compute a CLIP-style embedding and place tensors on GPU when available."""
        if CLIPModel is None or AutoProcessor is None or torch is None:
            return self._fallback_embed_image(image_path)

        try:
            self.load_backbone()
            if self._processor is None or self._backbone_model is None:
                return self._fallback_embed_image(image_path)

            with Image.open(image_path) as opened_image:
                image = ImageOps.exif_transpose(opened_image)
                if image.mode != "RGB":
                    image = image.convert("RGB")

            inputs = self._processor(images=image, return_tensors="pt")
            for key, value in inputs.items():
                inputs[key] = value.to(self.device_summary.device)

            autocast_enabled = (
                self.device_summary.accelerator == "cuda"
                and self.device_summary.half_precision
            )
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._backbone_model.get_image_features(**inputs)
                else:
                    features = self._backbone_model.get_image_features(**inputs)

            if hasattr(features, "image_embeds"):
                features = features.image_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output

            embedding = features.detach().float().cpu().numpy()[0]
            return _normalize_vector(embedding)
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self.logger.warning(
                "Falling back to deterministic image embedding for %s: %s",
                image_path,
                exc,
            )
            return self._fallback_embed_image(image_path)

    def embed_text(self, query: str) -> np.ndarray:
        """Compute a CLIP text embedding for semantic search queries."""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        if CLIPModel is None or AutoProcessor is None or torch is None:
            return self._fallback_embed_text(normalized_query)

        try:
            self.load_backbone()
            if self._processor is None or self._backbone_model is None:
                return self._fallback_embed_text(normalized_query)

            inputs = self._processor(
                text=normalized_query,
                padding=True,
                return_tensors="pt",
            )
            for key, value in inputs.items():
                inputs[key] = value.to(self.device_summary.device)

            autocast_enabled = (
                self.device_summary.accelerator == "cuda"
                and self.device_summary.half_precision
            )
            with torch.inference_mode():
                if autocast_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._backbone_model.get_text_features(**inputs)
                else:
                    features = self._backbone_model.get_text_features(**inputs)

            if hasattr(features, "text_embeds"):
                features = features.text_embeds
            elif hasattr(features, "pooler_output"):
                features = features.pooler_output

            embedding = features.detach().float().cpu().numpy()[0]
            return _normalize_vector(embedding)
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self.logger.warning(
                "Falling back to deterministic text embedding for %s: %s",
                normalized_query,
                exc,
            )
            return self._fallback_embed_text(normalized_query)

    def _fallback_embed_image(self, image_path: Path) -> np.ndarray:
        """Return a deterministic embedding when the ML stack is unavailable.

        This keeps the active-learning loop operational in lightweight dev setups
        while still preferring the GPU backbone whenever torch+transformers exist.
        """
        with Image.open(image_path) as opened_image:
            image = ImageOps.exif_transpose(opened_image)
            if image.mode != "RGB":
                image = image.convert("RGB")
            resized = image.resize((32, 32), Image.Resampling.BICUBIC)

        pixel_array = np.asarray(resized, dtype=np.float32) / 255.0
        channel_means = pixel_array.mean(axis=(0, 1))
        channel_stds = pixel_array.std(axis=(0, 1))
        flattened = pixel_array.reshape(-1)
        feature_vector = np.concatenate((flattened, channel_means, channel_stds))
        return _normalize_vector(feature_vector)

    def _fallback_embed_text(self, query: str) -> np.ndarray:
        """Return a deterministic text embedding with the image fallback dimension."""

        target_dim = 32 * 32 * 3 + 6
        vector = np.zeros(target_dim, dtype=np.float32)

        for token in query.lower().split():
            token_bytes = token.encode("utf-8", errors="ignore")
            for index, byte_value in enumerate(token_bytes):
                vector[(byte_value + index * 31) % target_dim] += (byte_value / 255.0)

        if not vector.any():
            vector[0] = 1.0
        return _normalize_vector(vector)

    def learn_from_label(
        self,
        *,
        relative_path: str,
        image_path: Optional[Path],
        identity_label: str,
        subject_type: str = "unknown",
        embedding: Optional[Sequence[float]] = None,
        source_asset_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        human_verified: bool = True,
        schedule_fine_tune: Optional[bool] = None,
        class_name: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> EmbeddingRecord:
        """Immediately update the live vector index after a confirmed label."""
        vector = (
            list(_normalize_vector(np.asarray(embedding, dtype=np.float32)))
            if embedding is not None
            else None
        )
        if vector is None:
            if image_path is None:
                raise ValueError("image_path is required when embedding is not provided")
            vector = list(self.embed_image(image_path))

        record = EmbeddingRecord(
            record_id=record_id or str(uuid.uuid4()),
            relative_path=relative_path,
            subject_type=subject_type,
            class_name=class_name,
            identity_label=identity_label,
            vector=vector,
            source_asset_id=source_asset_id,
            metadata=metadata or {},
            human_verified=human_verified,
        )
        self.vector_index.upsert(record)
        self._log_event(
            "vector_upsert",
            {"record_id": record.record_id, "identity_label": identity_label},
        )

        should_schedule = (
            self.config.active_learning.schedule_fine_tune
            if schedule_fine_tune is None
            else schedule_fine_tune
        )
        if human_verified and should_schedule:
            self.schedule_background_adaptation(record)

        return record

    def add_negative_sample(
        self,
        *,
        record_id: str,
        relative_path: str,
        negative_label: str,
        image_path: Optional[Path],
        subject_type: str = "unknown",
        class_name: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> NegativeEmbeddingRecord:
        """Store a negative example so future ranking can penalize the bad label."""
        vector = (
            list(_normalize_vector(np.asarray(embedding, dtype=np.float32)))
            if embedding is not None
            else None
        )
        if vector is None:
            if image_path is None:
                raise ValueError("image_path is required when embedding is not provided")
            vector = list(self.embed_image(image_path))

        record = NegativeEmbeddingRecord(
            record_id=record_id,
            negative_label=negative_label,
            relative_path=relative_path,
            subject_type=subject_type,
            class_name=class_name,
            vector=vector,
            metadata=metadata or {},
        )
        self.negative_index.upsert(record)
        self._save_negative_index()
        self._log_event(
            "negative_upsert",
            {"record_id": record_id, "negative_label": negative_label},
        )
        return record

    def remove_negative_sample(self, negative_label: str, record_id: str) -> None:
        """Drop a prior negative sample when the user later confirms the label."""
        self.negative_index.remove(negative_label, record_id)
        self._save_negative_index()

    def propose_identity(
        self,
        *,
        relative_path: str,
        subject_type: str = "unknown",
        image_path: Optional[Path] = None,
        embedding: Optional[Sequence[float]] = None,
        class_name: Optional[str] = None,
    ) -> VerificationTask:
        """Rank candidates and decide whether the UI should auto-accept or ask Yes/No."""
        vector = (
            list(_normalize_vector(np.asarray(embedding, dtype=np.float32)))
            if embedding is not None
            else None
        )
        if vector is None:
            if image_path is None:
                raise ValueError("image_path is required when embedding is not provided")
            vector = list(self.embed_image(image_path))

        candidate_limit = min(
            max(
                self.config.active_learning.top_k * 4,
                self.config.active_learning.top_k,
            ),
            100,
        )
        hits = self.vector_index.query(
            vector=vector,
            subject_type=subject_type,
            top_k=candidate_limit,
            class_name=class_name,
        )
        hits = self._apply_negative_penalties(vector, hits, class_name=class_name)
        return self._build_verification_task(
            relative_path=relative_path,
            subject_type=subject_type,
            class_name=class_name,
            hits=hits[: self.config.active_learning.top_k],
        )

    def apply_verification_decision(
        self,
        decision: VerificationDecision,
        *,
        image_path: Optional[Path],
        embedding: Optional[Sequence[float]] = None,
    ) -> Optional[EmbeddingRecord]:
        """Feed human feedback into the index and optionally queue adaptation work."""
        if not decision.accepted or not decision.confirmed_label:
            self._log_event(
                "verification_rejected",
                {
                    "relative_path": decision.relative_path,
                    "subject_type": decision.subject_type,
                },
            )
            return None

        return self.learn_from_label(
            relative_path=decision.relative_path,
            image_path=image_path,
            identity_label=decision.confirmed_label,
            subject_type=decision.subject_type,
            embedding=embedding,
            metadata=decision.metadata,
            human_verified=True,
            schedule_fine_tune=decision.schedule_fine_tune,
            class_name=(
                str((decision.metadata or {}).get("detected_class") or "")
                .strip()
                .lower()
                or None
            ),
        )

    def semantic_search(
        self,
        *,
        query: str,
        top_k: Optional[int] = None,
        subject_type: Optional[str] = None,
        class_name: Optional[str] = None,
    ) -> list[SearchHit]:
        """Search the vector index using a text query."""
        vector = self.embed_text(query)
        effective_top_k = top_k or self.config.active_learning.top_k
        candidate_limit = min(max(effective_top_k * 4, effective_top_k), 100)
        hits = self.vector_index.query(
            vector=vector,
            subject_type=subject_type,
            top_k=candidate_limit,
            class_name=class_name,
        )
        hits = self._apply_negative_penalties(vector, hits, class_name=class_name)
        return hits[: (top_k or self.config.active_learning.top_k)]

    @staticmethod
    def score_to_confidence(score: float) -> float:
        """Convert a raw cosine score into the UI confidence scale."""

        return _score_to_confidence(score)

    def _apply_negative_penalties(
        self,
        query_vector: Sequence[float],
        hits: Sequence[SearchHit],
        *,
        class_name: Optional[str] = None,
    ) -> list[SearchHit]:
        adjusted_hits: list[SearchHit] = []
        for hit in hits:
            penalty = self.negative_index.penalty(
                hit.identity_label,
                query_vector,
                class_name=class_name or hit.class_name,
            )
            adjusted_score = max(-1.0, min(1.0, hit.score - penalty))
            adjusted_hits.append(
                SearchHit(
                    record_id=hit.record_id,
                    identity_label=hit.identity_label,
                    subject_type=hit.subject_type,
                    class_name=hit.class_name,
                    score=adjusted_score,
                    relative_path=hit.relative_path,
                    metadata={**hit.metadata, "negative_penalty": penalty},
                )
            )

        adjusted_hits.sort(key=lambda item: item.score, reverse=True)
        return adjusted_hits

    def schedule_background_adaptation(self, record: EmbeddingRecord) -> Future:
        """Schedule LoRA or adapter fine-tuning work without blocking the API."""
        self._log_event(
            "adaptation_scheduled",
            {"record_id": record.record_id, "identity_label": record.identity_label},
        )
        return self.executor.submit(self._run_background_adaptation, record)

    def _run_background_adaptation(self, record: EmbeddingRecord) -> None:
        """Run or skip deferred adaptation work for a confirmed embedding."""

        if not self.config.enable_lora_fine_tune:
            self._log_event(
                "adaptation_skipped",
                {
                    "record_id": record.record_id,
                    "reason": "enable_lora_fine_tune is false",
                },
            )
            return

        # Blueprint only: production implementation would assemble a short-horizon
        # fine-tuning set for the confirmed identity and update an adapter checkpoint.
        self._log_event(
            "adaptation_blueprint",
            {
                "record_id": record.record_id,
                "identity_label": record.identity_label,
                    "message": (
                        "Hook PEFT/LoRA training here using the mirror "
                        "workspace dataset snapshot."
                    ),
            },
        )

    def _build_verification_task(
        self,
        *,
        relative_path: str,
        subject_type: str,
        class_name: Optional[str],
        hits: list[SearchHit],
    ) -> VerificationTask:
        if not hits:
            return VerificationTask(
                relative_path=relative_path,
                subject_type=subject_type,
                class_name=class_name,
                status="new_identity",
                confidence=0.0,
                hits=[],
                next_action="collect_label",
            )

        best = hits[0]
        confidence = _score_to_confidence(best.score)

        if confidence >= self.config.active_learning.auto_accept_threshold:
            status = "auto_accept"
            next_action = "auto_confirm"
        elif confidence >= self.config.active_learning.verify_threshold:
            status = "needs_confirmation"
            next_action = "queue_yes_no"
        elif confidence < self.config.active_learning.reject_threshold:
            status = "new_identity"
            next_action = "collect_label"
        else:
            status = "needs_confirmation"
            next_action = "queue_yes_no"

        return VerificationTask(
            relative_path=relative_path,
            subject_type=subject_type,
            class_name=class_name,
            status=status,
            proposed_label=best.identity_label,
            confidence=confidence,
            hits=hits,
            next_action=next_action,
        )

    def _log_event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Append one structured training event to the workspace log."""

        log_path = self.config.workspace_root / "logs" / "training_tasks.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            "payload": payload,
        }

        with NamedTemporaryFile(
            "w",
            delete=False,
            encoding="utf-8",
            dir=log_path.parent,
            suffix=".tmp",
        ) as handle:
            if log_path.exists():
                handle.write(log_path.read_text(encoding="utf-8"))
            handle.write(json.dumps(entry) + "\n")
            temp_path = Path(handle.name)

        temp_path.replace(log_path)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a float32 vector with unit norm when possible."""

    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _score_to_confidence(score: float) -> float:
    """Map cosine similarity into the UI confidence scale."""

    bounded = max(0.0, min(1.0, score))
    if bounded < 0.35:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-24.0 * (bounded - 0.78))))
