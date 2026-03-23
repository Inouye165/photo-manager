"""Search routes: text search, image similarity, index management."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

bp = Blueprint("search", __name__, url_prefix="/api/search")
logger = logging.getLogger(__name__)


def _engine():
    return current_app.config["SEARCH_ENGINE"]

def _embedder():
    return current_app.config["EMBEDDER"]

def _registry():
    return current_app.config["REGISTRY"]

def _cfg():
    return current_app.config["SETTINGS"]


# ── Text search ──────────────────────────────────────────────────────────

@bp.route("/text", methods=["POST"])
def text_search():
    """Natural-language search across scene + region embeddings."""
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query required"}), 400

    top_k = min(body.get("top_k", 20), 100)
    scope = body.get("scope", "both")
    subject_type = body.get("subject_type")

    hits = _engine().text_search(
        query, top_k=top_k, subject_type=subject_type, scope=scope,
    )
    return jsonify({
        "query": query,
        "results": [h.model_dump(mode="json") for h in hits],
    })


# ── Image similarity ─────────────────────────────────────────────────────

@bp.route("/similar/<path:relative_path>", methods=["GET"])
def similar(relative_path: str):
    """Find images similar to an existing asset."""
    rec = _registry().get(relative_path)
    if rec is None:
        return jsonify({"error": "not found"}), 404

    top_k = min(request.args.get("top_k", 20, type=int), 100)
    hits = _engine().image_search(Path(rec.source_path), top_k=top_k)
    return jsonify({
        "reference": relative_path,
        "results": [h.model_dump(mode="json") for h in hits],
    })


# ── Index management ─────────────────────────────────────────────────────

@bp.route("/index/scene", methods=["POST"])
def index_scenes():
    """Embed and index whole images (scene embeddings) for all un-indexed assets."""
    cfg = _cfg()
    if not cfg.enable_scene_embeddings:
        return jsonify({"error": "scene embeddings disabled"}), 403

    from app.search.vector_store import VectorRecord

    count = 0
    for rec in _registry().all():
        if rec.scene_indexed:
            continue
        try:
            vec = _embedder().embed_image(Path(rec.source_path))
            vr = VectorRecord(
                record_id=f"scene-{rec.asset_id}",
                relative_path=rec.relative_path,
                asset_id=rec.asset_id,
                vector=vec.tolist(),
                metadata={"type": "scene", "width": rec.width, "height": rec.height},
            )
            _engine().upsert_scene(vr)
            rec.scene_indexed = True
            _registry().upsert(rec)
            count += 1
        except Exception as exc:
            logger.warning("Scene index error for %s: %s", rec.relative_path, exc)

    _registry().save()
    return jsonify({"indexed": count})


@bp.route("/index/regions", methods=["POST"])
def index_regions():
    """Embed and index detection crops (region embeddings)."""
    cfg = _cfg()
    if not cfg.enable_region_embeddings:
        return jsonify({"error": "region embeddings disabled"}), 403

    from app.search.vector_store import VectorRecord

    identity_mgr = current_app.config["IDENTITY"]
    derivatives = current_app.config["DERIVATIVES"]
    count = 0

    for label in identity_mgr.all(status="confirmed"):
        rec = _registry().get(label.image_relative_path)
        if rec is None or not label.bbox or len(label.bbox) < 4:
            continue
        try:
            crop_path = derivatives.crop(
                Path(rec.source_path),
                rec.relative_path,
                rec.asset_id,
                tuple(label.bbox[:4]),
                subject_type=label.detected_class or "unknown",
                identity_hint=label.assigned_label,
            )
            vec = _embedder().embed_image(crop_path)
            vr = VectorRecord(
                record_id=f"region-{label.id}",
                relative_path=rec.relative_path,
                asset_id=rec.asset_id,
                identity_label=label.assigned_label,
                subject_type=label.detected_class or "unknown",
                class_name=label.detected_class,
                vector=vec.tolist(),
                metadata={"type": "region", "label_id": label.id, "bbox": label.bbox},
            )
            _engine().upsert_region(vr)
            count += 1
        except Exception as exc:
            logger.warning("Region index error for label %s: %s", label.id, exc)

    return jsonify({"indexed": count})
