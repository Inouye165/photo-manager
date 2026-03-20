"""Lightweight identity suggestion engine for PhotoFinder.

Builds simple visual signatures from labeled crops and proposes likely identities for
pending detections. This is intentionally lightweight so the project can improve
incrementally without requiring a heavy training stack first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from src.label_manager import LabelManager


pillow_heif.register_heif_opener()


@dataclass
class IdentityPrototype:
    name: str
    category: str
    sample_count: int
    centroid: np.ndarray
    exemplars: tuple[np.ndarray, ...]
    observed_classes: tuple[str, ...]


class IdentityEngine:
    """Suggest likely identities using lightweight visual similarity."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        suffix = image_path.suffix.lower()

        if suffix in {".heic", ".heif"}:
            try:
                pil_image = Image.open(image_path)
                pil_image = ImageOps.exif_transpose(pil_image)
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        image = cv2.imread(str(image_path))
        return image

    def _get_detection_class(self, detection: Dict) -> str:
        detected_class = detection.get("detected_class") or detection.get("class_name") or ""
        return str(detected_class).strip().lower()

    def _extract_signature(self, image_path: Path) -> Optional[np.ndarray]:
        image = self._load_image(image_path)
        if image is None or image.size == 0:
            return None

        resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 6, 6], [0, 180, 0, 256, 0, 256])
        hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0

        edges = cv2.Canny(gray, 40, 120)
        edge_small = cv2.resize(edges, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).flatten() / 255.0

        signature = np.concatenate([hsv_hist, gray_small, edge_small]).astype(np.float32)
        norm = np.linalg.norm(signature)
        if norm == 0:
            return None
        return signature / norm

    def _category_for_detection(self, detected_class: str) -> str:
        return "people" if detected_class == "person" else "animals"

    def build_index(self, label_manager: LabelManager) -> List[IdentityPrototype]:
        grouped_signatures: Dict[tuple[str, str], List[np.ndarray]] = {}
        grouped_classes: Dict[tuple[str, str], set[str]] = {}

        for label_record in label_manager.get_all_labels(status="confirmed"):
            assigned_label = label_record.get("assigned_label", "").strip()
            if not assigned_label:
                continue

            detected_class = str(label_record.get("detected_class", "")).strip().lower()
            category = self._category_for_detection(detected_class)
            crop_path = label_manager.resolve_stored_path(label_record.get("crop_path"))
            image_path = label_manager.resolve_stored_path(label_record.get("image_path"))
            source_path = crop_path if crop_path and crop_path.exists() else image_path

            if not source_path or not source_path.exists():
                continue

            signature = self._extract_signature(source_path)
            if signature is None:
                continue

            prototype_key = (assigned_label, category)
            grouped_signatures.setdefault(prototype_key, []).append(signature)
            grouped_classes.setdefault(prototype_key, set()).add(detected_class)

        prototypes: List[IdentityPrototype] = []
        for (assigned_label, category), signatures in grouped_signatures.items():
            centroid = np.mean(np.stack(signatures), axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue

            prototypes.append(
                IdentityPrototype(
                    name=assigned_label,
                    category=category,
                    sample_count=len(signatures),
                    centroid=centroid / norm,
                    exemplars=tuple(signatures[: min(6, len(signatures))]),
                    observed_classes=tuple(sorted(grouped_classes.get((assigned_label, category), set()))),
                )
            )

        return prototypes

    def suggest_for_detection(
        self,
        label_manager: LabelManager,
        detection: Dict,
        min_samples: int = 2,
    ) -> Optional[Dict[str, object]]:
        category = self._category_for_detection(self._get_detection_class(detection))
        prototypes = self.build_index(label_manager)
        return self.suggest_from_prototypes(detection, prototypes, min_samples=min_samples, category=category)

    def suggest_from_prototypes(
        self,
        detection: Dict,
        prototypes: List[IdentityPrototype],
        min_samples: int = 2,
        category: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        detected_class = self._get_detection_class(detection)
        resolved_category = category or self._category_for_detection(detected_class)
        required_samples = min_samples if resolved_category == "people" else max(1, min_samples - 1)
        filtered_prototypes = [
            prototype
            for prototype in prototypes
            if prototype.category == resolved_category
            and prototype.sample_count >= required_samples
            and (resolved_category == "people" or detected_class in prototype.observed_classes)
        ]

        if not filtered_prototypes:
            return None

        crop_path = detection.get("crop_path")
        image_path = detection.get("image_path")
        source_path = None
        if crop_path:
            source_path = self.output_dir / str(crop_path).replace("\\", "/")
        if (source_path is None or not source_path.exists()) and image_path:
            source_path = self.output_dir / str(image_path).replace("\\", "/")

        if source_path is None or not source_path.exists():
            return None

        signature = self._extract_signature(source_path)
        if signature is None:
            return None

        scored: List[Dict[str, object]] = []
        for prototype in filtered_prototypes:
            centroid_score = float(np.dot(signature, prototype.centroid))
            exemplar_scores = sorted((float(np.dot(signature, exemplar)) for exemplar in prototype.exemplars), reverse=True)

            if resolved_category == "people" and exemplar_scores:
                exemplar_slice = exemplar_scores[: min(2, len(exemplar_scores))]
                exemplar_score = sum(exemplar_slice) / len(exemplar_slice)
                similarity = (centroid_score * 0.58) + (exemplar_score * 0.42)
            else:
                similarity = centroid_score

            confidence = max(0.0, min(0.999, (similarity + 1.0) / 2.0))
            scored.append(
                {
                    "name": prototype.name,
                    "score": similarity,
                    "centroid_score": centroid_score,
                    "exemplar_score": exemplar_scores[0] if exemplar_scores else centroid_score,
                    "confidence": confidence,
                    "sample_count": prototype.sample_count,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0]
        min_score = 0.8 if resolved_category == "people" else 0.86

        if best["score"] < min_score:
            return None

        runner_up_gap = best["score"] - scored[1]["score"] if len(scored) > 1 else best["score"]
        min_gap = 0.01 if resolved_category == "people" else 0.03

        if runner_up_gap < min_gap:
            return None

        return {
            "label": best["name"],
            "confidence": int(best["confidence"] * 100),
            "score": round(best["score"], 3),
            "sample_count": best["sample_count"],
            "runner_up_gap": round(runner_up_gap, 3),
            "alternatives": scored[1:3],
        }