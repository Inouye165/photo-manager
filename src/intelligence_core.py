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
    from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
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
    subject_type: Literal["people", "animals", "unknown"] = "unknown"
    identity_label: str
    vector: list[float]
    source_asset_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    human_verified: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SearchHit(BaseModel):
    """Search result returned from the vector index."""

    record_id: str
    identity_label: str
    subject_type: str
    score: float
    relative_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationTask(BaseModel):
    """Work item shown in the React identity verification flow."""

    relative_path: str
    subject_type: Literal["people", "animals", "unknown"] = "unknown"
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
    subject_type: Literal["people", "animals", "unknown"] = "unknown"
    accepted: bool
    confirmed_label: Optional[str] = None
    schedule_fine_tune: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorIndex(Protocol):
    """Storage abstraction so the ranking engine is not tied to one backend."""

    def upsert(self, record: EmbeddingRecord) -> None:
        ...

    def query(self, vector: Sequence[float], subject_type: str, top_k: int) -> list[SearchHit]:
        ...


class InMemoryVectorIndex:
    """Simple cosine index used as a safe default and unit-test target."""

    def __init__(self) -> None:
        self._records: dict[str, EmbeddingRecord] = {}

    def upsert(self, record: EmbeddingRecord) -> None:
        self._records[record.record_id] = record

    def query(self, vector: Sequence[float], subject_type: str, top_k: int) -> list[SearchHit]:
        query_vector = _normalize_vector(np.asarray(vector, dtype=np.float32))
        hits: list[SearchHit] = []

        for record in self._records.values():
            if record.subject_type != subject_type and record.subject_type != "unknown":
                continue

            candidate_vector = _normalize_vector(np.asarray(record.vector, dtype=np.float32))
            score = float(np.dot(query_vector, candidate_vector))
            hits.append(
                SearchHit(
                    record_id=record.record_id,
                    identity_label=record.identity_label,
                    subject_type=record.subject_type,
                    score=max(-1.0, min(1.0, score)),
                    relative_path=record.relative_path,
                    metadata=record.metadata,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]


class QdrantVectorIndex:
    """Optional Qdrant-backed index for production-scale search."""

    def __init__(self, config: IntelligenceConfig):
        if QdrantClient is None or VectorParams is None or PointStruct is None or Distance is None:
            raise RuntimeError("qdrant-client is not installed; use provider='in_memory' or install qdrant-client")

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
                vectors_config=VectorParams(size=config.backbone.expected_embedding_dim, distance=Distance.COSINE),
            )

    def upsert(self, record: EmbeddingRecord) -> None:
        payload = {
            "identity_label": record.identity_label,
            "relative_path": record.relative_path,
            "subject_type": record.subject_type,
            "metadata": record.metadata,
        }
        point = PointStruct(id=record.record_id, vector=record.vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point], wait=True)

    def query(self, vector: Sequence[float], subject_type: str, top_k: int) -> list[SearchHit]:
        if Filter is None or FieldCondition is None or MatchValue is None:
            return []

        query_filter = Filter(
            must=[FieldCondition(key="subject_type", match=MatchValue(value=subject_type))]
        )
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
                    score=float(point.score),
                    relative_path=str(payload.get("relative_path", "")),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return results


class IntelligenceCore:
    """Active-learning engine with immediate vector updates and optional fine-tuning."""

    def __init__(self, config: IntelligenceConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("photofinder.intelligence")
        self.config.workspace_root.mkdir(parents=True, exist_ok=True)
        (self.config.workspace_root / "logs").mkdir(parents=True, exist_ok=True)
        self.device_summary = self._resolve_device()
        self.executor = ThreadPoolExecutor(max_workers=self.config.active_learning.background_workers)
        self._processor = None
        self._backbone_model = None
        self.vector_index = self._build_vector_index()

    def _resolve_device(self) -> DeviceSummary:
        if torch is None:
            return DeviceSummary(device="cpu", accelerator="cpu", half_precision=False)

        preference = self.config.device_preference
        cuda_available = torch.cuda.is_available()
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend and mps_backend.is_available())

        if preference in {"auto", "cuda"} and cuda_available:
            return DeviceSummary(device="cuda", accelerator="cuda", half_precision=self.config.backbone.use_half_precision)

        if preference in {"auto", "mps"} and mps_available:
            return DeviceSummary(device="mps", accelerator="mps", half_precision=False)

        return DeviceSummary(device="cpu", accelerator="cpu", half_precision=False)

    def _build_vector_index(self) -> VectorIndex:
        if self.config.vector_store.provider == "qdrant":
            try:
                return QdrantVectorIndex(self.config)
            except RuntimeError as exc:
                self.logger.warning("Falling back to in-memory vector index: %s", exc)

        return InMemoryVectorIndex()

    def reset_vector_index(self) -> None:
        """Recreate the live vector index from configuration."""
        self.vector_index = self._build_vector_index()

    def load_backbone(self) -> None:
        """Load the embedding backbone lazily so API startup stays responsive."""
        if self._backbone_model is not None and self._processor is not None:
            return

        if CLIPModel is None or AutoProcessor is None:
            raise RuntimeError(
                "transformers is not installed. Install torch + transformers to enable GPU embeddings."
            )

        processor_name = self.config.backbone.processor_name or self.config.backbone.model_name
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

            autocast_enabled = self.device_summary.accelerator == "cuda" and self.device_summary.half_precision
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
            self.logger.warning("Falling back to deterministic image embedding for %s: %s", image_path, exc)
            return self._fallback_embed_image(image_path)

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

    def learn_from_label(
        self,
        *,
        relative_path: str,
        image_path: Optional[Path],
        identity_label: str,
        subject_type: Literal["people", "animals", "unknown"] = "unknown",
        embedding: Optional[Sequence[float]] = None,
        source_asset_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        human_verified: bool = True,
        schedule_fine_tune: Optional[bool] = None,
    ) -> EmbeddingRecord:
        """Immediately update the live vector index after a confirmed label."""
        vector = list(_normalize_vector(np.asarray(embedding, dtype=np.float32))) if embedding is not None else None
        if vector is None:
            if image_path is None:
                raise ValueError("image_path is required when embedding is not provided")
            vector = list(self.embed_image(image_path))

        record = EmbeddingRecord(
            relative_path=relative_path,
            subject_type=subject_type,
            identity_label=identity_label,
            vector=vector,
            source_asset_id=source_asset_id,
            metadata=metadata or {},
            human_verified=human_verified,
        )
        self.vector_index.upsert(record)
        self._log_event("vector_upsert", {"record_id": record.record_id, "identity_label": identity_label})

        should_schedule = self.config.active_learning.schedule_fine_tune if schedule_fine_tune is None else schedule_fine_tune
        if human_verified and should_schedule:
            self.schedule_background_adaptation(record)

        return record

    def propose_identity(
        self,
        *,
        relative_path: str,
        subject_type: Literal["people", "animals", "unknown"] = "unknown",
        image_path: Optional[Path] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> VerificationTask:
        """Rank candidates and decide whether the UI should auto-accept or ask Yes/No."""
        vector = list(_normalize_vector(np.asarray(embedding, dtype=np.float32))) if embedding is not None else None
        if vector is None:
            if image_path is None:
                raise ValueError("image_path is required when embedding is not provided")
            vector = list(self.embed_image(image_path))

        hits = self.vector_index.query(vector=vector, subject_type=subject_type, top_k=self.config.active_learning.top_k)
        return self._build_verification_task(relative_path=relative_path, subject_type=subject_type, hits=hits)

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
                {"relative_path": decision.relative_path, "subject_type": decision.subject_type},
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
        )

    def schedule_background_adaptation(self, record: EmbeddingRecord) -> Future:
        """Schedule LoRA or adapter fine-tuning work without blocking the API."""
        self._log_event(
            "adaptation_scheduled",
            {"record_id": record.record_id, "identity_label": record.identity_label},
        )
        return self.executor.submit(self._run_background_adaptation, record)

    def _run_background_adaptation(self, record: EmbeddingRecord) -> None:
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
                "message": "Hook PEFT/LoRA training here using the mirror workspace dataset snapshot.",
            },
        )

    def _build_verification_task(
        self,
        *,
        relative_path: str,
        subject_type: Literal["people", "animals", "unknown"],
        hits: list[SearchHit],
    ) -> VerificationTask:
        if not hits:
            return VerificationTask(
                relative_path=relative_path,
                subject_type=subject_type,
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
            status=status,
            proposed_label=best.identity_label,
            confidence=confidence,
            hits=hits,
            next_action=next_action,
        )

    def _log_event(self, event_name: str, payload: dict[str, Any]) -> None:
        log_path = self.config.workspace_root / "logs" / "training_tasks.jsonl"
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            "payload": payload,
        }

        with NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=log_path.parent, suffix=".tmp") as handle:
            if log_path.exists():
                handle.write(log_path.read_text(encoding="utf-8"))
            handle.write(json.dumps(entry) + "\n")
            temp_path = Path(handle.name)

        temp_path.replace(log_path)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if math.isclose(norm, 0.0):
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _score_to_confidence(score: float) -> float:
    bounded = max(-1.0, min(1.0, score))
    return float((bounded + 1.0) / 2.0)