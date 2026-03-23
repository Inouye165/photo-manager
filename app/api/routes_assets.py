"""Asset-management routes: scan, list, upload, derivatives."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

bp = Blueprint("assets", __name__, url_prefix="/api/assets")
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────

def _cfg():
    return current_app.config["SETTINGS"]

def _registry():
    return current_app.config["REGISTRY"]

def _vault():
    return current_app.config["VAULT"]

def _derivatives():
    return current_app.config["DERIVATIVES"]

def _detector():
    return current_app.config["DETECTOR"]


# ── Scan ─────────────────────────────────────────────────────────────────

@bp.route("/scan", methods=["POST"])
def scan_roots():
    """Trigger a filesystem scan of configured roots."""
    from app.catalog.scanner import scan

    cfg = _cfg()
    result = scan(cfg.resolved_scan_roots, _registry())
    return jsonify({
        "discovered": result.discovered,
        "created": result.created,
        "updated": result.updated,
        "unchanged": result.unchanged,
        "errors": result.errors[:50],
    })


# ── List ─────────────────────────────────────────────────────────────────

@bp.route("", methods=["GET"])
def list_assets():
    """Return summary of all known assets."""
    page = request.args.get("page", 1, type=int)
    per_page = min(request.args.get("per_page", 100, type=int), 500)
    records = list(_registry().all())
    start = (page - 1) * per_page
    subset = records[start : start + per_page]
    return jsonify({
        "total": len(records),
        "page": page,
        "per_page": per_page,
        "assets": [r.model_dump(mode="json") for r in subset],
    })


@bp.route("/<path:relative_path>", methods=["GET"])
def get_asset(relative_path: str):
    """Return metadata for a single asset."""
    rec = _registry().get(relative_path)
    if rec is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(rec.model_dump(mode="json"))


# ── Thumbnails / previews ────────────────────────────────────────────────

@bp.route("/<path:relative_path>/thumb", methods=["GET"])
def asset_thumb(relative_path: str):
    rec = _registry().get(relative_path)
    if rec is None:
        return jsonify({"error": "not found"}), 404
    img = _derivatives().derivative(
        Path(rec.source_path), relative_path, rec.asset_id, variant="thumb",
    )
    return send_file(img)


@bp.route("/<path:relative_path>/preview", methods=["GET"])
def asset_preview(relative_path: str):
    rec = _registry().get(relative_path)
    if rec is None:
        return jsonify({"error": "not found"}), 404
    img = _derivatives().derivative(
        Path(rec.source_path), relative_path, rec.asset_id, variant="preview",
    )
    return send_file(img)


# ── Upload ───────────────────────────────────────────────────────────────

@bp.route("/upload", methods=["POST"])
def upload():
    """Accept one uploaded image, dedup, vault, detect, and index."""
    cfg = _cfg()
    if not cfg.enable_upload:
        return jsonify({"error": "uploads disabled"}), 403

    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "empty filename"}), 400

    import tempfile
    from app.storage.paths import ALLOWED_IMAGE_SUFFIXES, compute_sha256

    suffix = Path(uploaded.filename).suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return jsonify({"error": f"unsupported format: {suffix}"}), 400

    # Save to temp, then ingest into vault
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.save(tmp)
        tmp_path = Path(tmp.name)

    try:
        vault_path, sha, is_dup = _vault().ingest(tmp_path)
        if is_dup:
            return jsonify({"duplicate": True, "sha256": sha}), 200

        # Register the new asset
        from app.catalog.registry import AssetRecord
        from PIL import Image, ImageOps
        import mimetypes

        stat = vault_path.stat()
        with Image.open(vault_path) as img:
            img = ImageOps.exif_transpose(img)
            w, h = img.size

        mime, _ = mimetypes.guess_type(vault_path.name)
        rec = AssetRecord(
            asset_id=sha[:20],
            relative_path=vault_path.name,
            source_path=str(vault_path),
            sha256=sha,
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            width=w,
            height=h,
            mime_type=mime,
        )
        _registry().upsert(rec)
        _registry().save()

        # Run detection
        detections = _detector().detect(vault_path)

        return jsonify({
            "duplicate": False,
            "sha256": sha,
            "asset_id": rec.asset_id,
            "relative_path": rec.relative_path,
            "detection_count": len(detections),
            "detections": detections,
        }), 201
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Detect ───────────────────────────────────────────────────────────────

@bp.route("/<path:relative_path>/detect", methods=["POST"])
def detect(relative_path: str):
    """Run YOLO detection on an existing asset."""
    rec = _registry().get(relative_path)
    if rec is None:
        return jsonify({"error": "not found"}), 404
    detections = _detector().detect(rec.source_path)
    rec.detection_count = len(detections)
    _registry().upsert(rec)
    _registry().save()
    return jsonify({"detections": detections, "count": len(detections)})
