"""YOLOv8 object detection wrapper.

Produces structured detection dicts (class, confidence, bbox) from images.
The model is loaded lazily so import stays fast.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.vision.loader import load_cv2

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

ANIMAL_CLASSES = frozenset({
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
})


class Detector:
    """Lazy-loaded YOLOv8 detector."""

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        *,
        conf_person: float = 0.50,
        conf_animal: float = 0.40,
        conf_general: float = 0.25,
    ) -> None:
        self._model_path = model_path
        self.conf_person = conf_person
        self.conf_animal = conf_animal
        self.conf_general = conf_general
        self._model: Any = None

    @property
    def available(self) -> bool:
        return _HAS_YOLO

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not _HAS_YOLO:
            raise RuntimeError("ultralytics is not installed")
        self._model = YOLO(self._model_path)

    # ── Quick presence check ─────────────────────────────────────────────

    def detect_subjects(self, image_path: str | Path) -> dict[str, bool]:
        """Return ``{has_person, has_animal}`` booleans."""
        result = {"has_person": False, "has_animal": False}
        if not _HAS_YOLO:
            return result

        try:
            self._ensure_model()
            img = load_cv2(image_path)
            preds = self._model(img, verbose=False, conf=self.conf_general)
            for r in preds:
                if not r.boxes:
                    continue
                for cls, conf in zip(r.boxes.cls, r.boxes.conf):
                    name = self._model.names[int(cls.item())]
                    c = conf.item()
                    if name == "person" and c >= self.conf_person:
                        result["has_person"] = True
                    elif name in ANIMAL_CLASSES and c >= self.conf_animal:
                        result["has_animal"] = True
        except Exception as exc:
            logger.error("Detection error on %s: %s", image_path, exc)

        return result

    # ── Detailed detections with bboxes ──────────────────────────────────

    def detect(self, image_path: str | Path) -> list[dict]:
        """Return a list of detection dicts with class, conf, bbox."""
        if not _HAS_YOLO:
            return []

        try:
            self._ensure_model()
            img = load_cv2(image_path)
            preds = self._model(img, verbose=False, conf=self.conf_general)
            detections: list[dict] = []

            for r in preds:
                if not r.boxes:
                    continue
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    cid = int(cls.item())
                    name = self._model.names[cid]
                    c = conf.item()
                    if name != "person" and name not in ANIMAL_CLASSES:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in box[:4])
                    detections.append({
                        "class_name": name,
                        "class_id": cid,
                        "confidence": c,
                        "bbox": [x1, y1, x2, y2],
                        "bbox_area": (x2 - x1) * (y2 - y1),
                    })
            return detections
        except Exception as exc:
            logger.error("Detection error on %s: %s", image_path, exc)
            return []
