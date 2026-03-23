"""Background job definitions: scan, index, rebuild.

Jobs can be triggered via API endpoints or run on startup.
They operate on the same service instances attached to the Flask app.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)


def run_full_pipeline(app: Flask) -> dict:
    """Execute scan → detect → scene-embed → region-embed in sequence.

    Returns a summary dict suitable for JSON serialization.
    """
    with app.app_context():
        from flask import current_app

        cfg = current_app.config["SETTINGS"]
        registry = current_app.config["REGISTRY"]
        detector = current_app.config["DETECTOR"]
        embedder = current_app.config["EMBEDDER"]
        derivatives = current_app.config["DERIVATIVES"]
        identity_mgr = current_app.config["IDENTITY"]
        search_engine = current_app.config["SEARCH_ENGINE"]

        # 1. Scan
        from app.catalog.scanner import scan
        scan_result = scan(cfg.resolved_scan_roots, registry)
        logger.info(
            "Scan: %d discovered, %d new, %d updated",
            scan_result.discovered, scan_result.created, scan_result.updated,
        )

        # 2. Detect subjects on new/un-detected assets
        detect_count = 0
        for rec in registry.all():
            if rec.detection_count > 0:
                continue
            try:
                dets = detector.detect(rec.source_path)
                rec.detection_count = len(dets)
                registry.upsert(rec)
                detect_count += 1
            except Exception as exc:
                logger.warning("Detection failed for %s: %s", rec.relative_path, exc)
        registry.save()

        # 3. Scene embeddings
        scene_count = 0
        if cfg.enable_scene_embeddings:
            from app.search.vector_store import VectorRecord
            for rec in registry.all():
                if rec.scene_indexed:
                    continue
                try:
                    vec = embedder.embed_image(Path(rec.source_path))
                    vr = VectorRecord(
                        record_id=f"scene-{rec.asset_id}",
                        relative_path=rec.relative_path,
                        asset_id=rec.asset_id,
                        vector=vec.tolist(),
                        metadata={"type": "scene"},
                    )
                    search_engine.upsert_scene(vr)
                    rec.scene_indexed = True
                    registry.upsert(rec)
                    scene_count += 1
                except Exception as exc:
                    logger.warning("Scene embed failed for %s: %s", rec.relative_path, exc)
            registry.save()

        # 4. Region embeddings for confirmed labels
        region_count = 0
        if cfg.enable_region_embeddings:
            from app.search.vector_store import VectorRecord
            for label in identity_mgr.all(status="confirmed"):
                rec = registry.get(label.image_relative_path)
                if rec is None or not label.bbox or len(label.bbox) < 4:
                    continue
                try:
                    crop_path = derivatives.crop(
                        Path(rec.source_path), rec.relative_path, rec.asset_id,
                        tuple(label.bbox[:4]),
                        subject_type=label.detected_class or "unknown",
                        identity_hint=label.assigned_label,
                    )
                    vec = embedder.embed_image(crop_path)
                    vr = VectorRecord(
                        record_id=f"region-{label.id}",
                        relative_path=rec.relative_path,
                        asset_id=rec.asset_id,
                        identity_label=label.assigned_label,
                        subject_type=label.detected_class or "unknown",
                        class_name=label.detected_class,
                        vector=vec.tolist(),
                        metadata={"type": "region", "label_id": label.id},
                    )
                    search_engine.upsert_region(vr)
                    region_count += 1
                except Exception as exc:
                    logger.warning("Region embed failed for label %s: %s", label.id, exc)

        return {
            "scan": {
                "discovered": scan_result.discovered,
                "created": scan_result.created,
                "updated": scan_result.updated,
            },
            "detections": detect_count,
            "scene_embeddings": scene_count,
            "region_embeddings": region_count,
        }
