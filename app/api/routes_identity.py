"""Identity / label routes: add, update, review, export."""

from __future__ import annotations

import logging

from flask import Blueprint, current_app, jsonify, request

from app.identity.manager import LabelRecord

bp = Blueprint("identity", __name__, url_prefix="/api/identity")
logger = logging.getLogger(__name__)


def _mgr():
    return current_app.config["IDENTITY"]


# ── CRUD ─────────────────────────────────────────────────────────────────

@bp.route("/labels", methods=["GET"])
def list_labels():
    status = request.args.get("status")
    detected_class = request.args.get("detected_class")
    assigned_label = request.args.get("assigned_label")
    labels = _mgr().all(
        status=status, detected_class=detected_class,
        assigned_label=assigned_label,
    )
    return jsonify([r.model_dump(mode="json") for r in labels])


@bp.route("/labels/<label_id>", methods=["GET"])
def get_label(label_id: str):
    rec = _mgr().get(label_id)
    if rec is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(rec.model_dump(mode="json"))


@bp.route("/labels", methods=["POST"])
def add_label():
    body = request.get_json(silent=True) or {}
    rec = LabelRecord(
        image_relative_path=body.get("image_relative_path", ""),
        detection_index=body.get("detection_index", 0),
        detected_class=body.get("detected_class", ""),
        assigned_label=body.get("assigned_label", ""),
        bbox=body.get("bbox", []),
        crop_relative_path=body.get("crop_relative_path"),
        status=body.get("status", "pending"),
    )
    lid = _mgr().add(rec)
    return jsonify({"id": lid}), 201


@bp.route("/labels/<label_id>", methods=["PATCH"])
def update_label(label_id: str):
    body = request.get_json(silent=True) or {}
    ok = _mgr().update_label(
        label_id,
        assigned_label=body.get("assigned_label"),
        status=body.get("status"),
    )
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"updated": True})


@bp.route("/labels/<label_id>", methods=["DELETE"])
def delete_label(label_id: str):
    ok = _mgr().delete(label_id)
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": True})


# ── Bulk / stats ─────────────────────────────────────────────────────────

@bp.route("/stats", methods=["GET"])
def label_stats():
    return jsonify(_mgr().statistics())


@bp.route("/unique", methods=["GET"])
def unique_labels():
    return jsonify(_mgr().unique_labels())


@bp.route("/export", methods=["GET"])
def export_training():
    return jsonify(_mgr().export_for_training())
