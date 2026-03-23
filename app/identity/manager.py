"""Identity management – naming, status, persistence.

Wraps the label lifecycle (add / confirm / reject) with JSON persistence.
This is the new-architecture equivalent of ``src.label_manager``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LabelRecord(BaseModel):
    """One human-assigned identity label for a detected region."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    image_relative_path: str
    detection_index: int = 0
    detected_class: str = ""
    assigned_label: str = ""
    bbox: list[int] = Field(default_factory=list)
    crop_relative_path: Optional[str] = None
    status: Literal["confirmed", "rejected", "pending"] = "pending"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class IdentityManager:
    """JSON-backed label store with CRUD, filtering, and export."""

    def __init__(self, labels_path: Path) -> None:
        self._path = labels_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, LabelRecord] = {}
        self._load()

    # ── Queries ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._records)

    def get(self, label_id: str) -> Optional[LabelRecord]:
        return self._records.get(label_id)

    def all(
        self,
        *,
        status: Optional[str] = None,
        detected_class: Optional[str] = None,
        assigned_label: Optional[str] = None,
    ) -> List[LabelRecord]:
        out: list[LabelRecord] = []
        for r in self._records.values():
            if status and r.status != status:
                continue
            if detected_class and r.detected_class != detected_class:
                continue
            if assigned_label and r.assigned_label != assigned_label:
                continue
            out.append(r)
        out.sort(key=lambda r: r.timestamp, reverse=True)
        return out

    def for_image(self, relative_path: str) -> List[LabelRecord]:
        return [r for r in self._records.values() if r.image_relative_path == relative_path]

    def unique_labels(self) -> List[str]:
        return sorted({
            r.assigned_label
            for r in self._records.values()
            if r.status == "confirmed" and r.assigned_label
        })

    def statistics(self) -> dict:
        stats: dict = {
            "total": len(self._records),
            "confirmed": 0, "rejected": 0, "pending": 0,
            "people": 0, "animals": 0, "unique_labels": set(),
        }
        for r in self._records.values():
            stats[r.status] = stats.get(r.status, 0) + 1
            if r.detected_class == "person":
                stats["people"] += 1
            else:
                stats["animals"] += 1
            if r.status == "confirmed" and r.assigned_label:
                stats["unique_labels"].add(r.assigned_label)
        stats["unique_labels"] = sorted(stats["unique_labels"])
        return stats

    # ── Mutations ────────────────────────────────────────────────────────

    def add(self, record: LabelRecord) -> str:
        self._records[record.id] = record
        self._save()
        return record.id

    def update_status(self, label_id: str, status: str) -> bool:
        rec = self._records.get(label_id)
        if rec is None:
            return False
        rec.status = status
        rec.timestamp = datetime.now().isoformat()
        self._save()
        return True

    def update_label(self, label_id: str, *, assigned_label: Optional[str] = None, status: Optional[str] = None) -> bool:
        rec = self._records.get(label_id)
        if rec is None:
            return False
        if assigned_label is not None:
            rec.assigned_label = assigned_label
        if status is not None:
            rec.status = status
        rec.timestamp = datetime.now().isoformat()
        self._save()
        return True

    def delete(self, label_id: str) -> bool:
        if label_id not in self._records:
            return False
        del self._records[label_id]
        self._save()
        return True

    # ── Export ────────────────────────────────────────────────────────────

    def export_for_training(self, status: str = "confirmed") -> Dict[str, list]:
        grouped: Dict[str, list] = {}
        for r in self.all(status=status):
            grouped.setdefault(r.assigned_label, []).append({
                "image_path": r.image_relative_path,
                "crop_path": r.crop_relative_path,
                "bbox": r.bbox,
                "detected_class": r.detected_class,
                "label_id": r.id,
            })
        return grouped

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            blob = json.loads(self._path.read_text(encoding="utf-8"))
            raw = blob.get("labels", {}) if isinstance(blob, dict) else {}
            for lid, data in raw.items():
                # Bridge old field names
                if "image_path" in data and "image_relative_path" not in data:
                    data["image_relative_path"] = data.pop("image_path")
                if "crop_path" in data and "crop_relative_path" not in data:
                    data["crop_relative_path"] = data.pop("crop_path")
                data.setdefault("id", lid)
                self._records[lid] = LabelRecord.model_validate(data)
        except Exception as exc:
            logger.warning("Could not load labels from %s: %s", self._path, exc)

    def _save(self) -> None:
        payload = {
            "labels": {lid: r.model_dump(mode="json") for lid, r in self._records.items()},
            "last_updated": datetime.now().isoformat(),
        }
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
