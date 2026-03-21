"""Migrate confirmed PhotoFinder labels into the persistent local Qdrant store.

Usage:
    python scripts/migrate_labels_to_qdrant.py <source_originals> <working_dir> [--mirror-workspace <path>]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def load_runtime_dependencies():
    """Load project modules after ensuring the repository root is importable."""

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.detection_metadata import DetectionMetadata
    from src.intelligence_core import IntelligenceConfig, IntelligenceCore
    from src.label_manager import LabelManager
    from src.mirror_manager import MirrorConfig, MirrorManager
    from web_ui import (
        build_detection_record_id,
        get_subject_group,
        get_subject_lane,
        normalize_relative_path,
        resolve_detection_asset_urls,
    )

    return {
        "DetectionMetadata": DetectionMetadata,
        "IntelligenceConfig": IntelligenceConfig,
        "IntelligenceCore": IntelligenceCore,
        "LabelManager": LabelManager,
        "MirrorConfig": MirrorConfig,
        "MirrorManager": MirrorManager,
        "build_detection_record_id": build_detection_record_id,
        "get_subject_group": get_subject_group,
        "get_subject_lane": get_subject_lane,
        "normalize_relative_path": normalize_relative_path,
        "resolve_detection_asset_urls": resolve_detection_asset_urls,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Qdrant migration helper."""

    parser = argparse.ArgumentParser(description="Migrate existing confirmed labels into local Qdrant storage.")
    parser.add_argument("source_originals", help="Path to the immutable source archive")
    parser.add_argument("working_dir", help="Path to the PhotoFinder working directory")
    parser.add_argument(
        "--mirror-workspace",
        help="Optional explicit mirror workspace path. Defaults next to the working directory.",
    )
    parser.add_argument(
        "--collection-name",
        default="photo_features",
        help="Qdrant collection name to populate.",
    )
    return parser.parse_args()


def main() -> int:
    """Rebuild the persistent local Qdrant collection from confirmed labels."""

    args = parse_args()
    deps = load_runtime_dependencies()
    source_dir = Path(args.source_originals).expanduser().resolve()
    data_dir = Path(args.working_dir).expanduser().resolve()
    mirror_workspace = (
        Path(args.mirror_workspace).expanduser().resolve()
        if args.mirror_workspace
        else data_dir.parent / f"{data_dir.name}__mirror_workspace"
    )

    mirror_manager = deps["MirrorManager"](
        deps["MirrorConfig"](source_originals=source_dir, mirror_workspace=mirror_workspace)
    )
    mirror_manager.sync_source_index()

    intelligence_core = deps["IntelligenceCore"](
        deps["IntelligenceConfig"](
            workspace_root=mirror_workspace / "active_learning",
            vector_store={
                "provider": "qdrant",
                "location": mirror_workspace / "vector_db",
                "collection_name": args.collection_name,
            },
            active_learning={"schedule_fine_tune": False},
        )
    )

    label_manager = deps["LabelManager"](str(data_dir))
    metadata_manager = deps["DetectionMetadata"](str(data_dir))

    migrated = 0
    skipped = 0
    for label_record in label_manager.get_all_labels(status="confirmed"):
        assigned_label = str(label_record.get("assigned_label") or "").strip()
        if not assigned_label:
            skipped += 1
            continue

        image_path = deps["normalize_relative_path"](label_record.get("image_path"))
        if not image_path:
            skipped += 1
            continue

        detection_record = metadata_manager.get_detections(os.path.join(str(data_dir), image_path))
        image_metadata = dict(detection_record.get("metadata") or {})
        asset_payload = deps["resolve_detection_asset_urls"](
            mirror_manager=mirror_manager,
            source_dir=source_dir,
            data_dir=data_dir,
            browser_workspace=mirror_workspace,
            image_path=image_path,
            crop_path=label_record.get("crop_path"),
            bbox=label_record.get("bbox"),
            subject_type=deps["get_subject_lane"](str(label_record.get("detected_class") or "")),
            identity_hint=assigned_label,
        )
        embedding_path = asset_payload.get("embedding_path")
        if not embedding_path:
            skipped += 1
            continue

        detection_index = int(label_record.get("detection_index", 0))
        intelligence_core.learn_from_label(
            record_id=deps["build_detection_record_id"](image_path, detection_index),
            relative_path=deps["build_detection_record_id"](image_path, detection_index),
            image_path=Path(embedding_path),
            identity_label=assigned_label,
            subject_type=deps["get_subject_group"](str(label_record.get("detected_class") or "")),
            class_name=str(label_record.get("detected_class") or "").strip().lower() or None,
            source_asset_id=str(image_metadata.get("file_hash") or "").strip() or None,
            metadata={
                "preview_url": asset_payload.get("candidate_image_url"),
                "full_url": asset_payload.get("candidate_full_url"),
                "image_path": image_path,
                "crop_path": deps["normalize_relative_path"](label_record.get("crop_path")),
                "label_id": label_record.get("id"),
                "captured_at": image_metadata.get("captured_at"),
                "file_hash": image_metadata.get("file_hash"),
                "confidence": label_record.get("confidence") or image_metadata.get("confidence"),
            },
            human_verified=False,
            schedule_fine_tune=False,
            record_status="confirmed",
        )
        migrated += 1

    print(
        f"Qdrant migration complete: migrated {migrated} confirmed label"
        f"{'s' if migrated != 1 else ''}, skipped {skipped}."
    )
    print(f"Collection: {args.collection_name}")
    print(f"Vector DB path: {mirror_workspace / 'vector_db'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
