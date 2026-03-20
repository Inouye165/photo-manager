#!/usr/bin/env python3
"""
Web UI for PhotoFinder - shows thumbnails organized by detection categories.
Security-focused implementation with proper input validation and file access controls.
"""

import os
import sys
import logging
from pathlib import Path
from urllib.parse import quote
from typing import Dict, List, Optional

from flask import Flask, abort, render_template, send_file, send_from_directory, request, jsonify
import pillow_heif
from PIL import Image, ImageOps

from src.detection_metadata import DetectionMetadata
from src.ingest_manager import IngestConfig, IngestManager
from src.intelligence_core import IntelligenceConfig, IntelligenceCore, VerificationDecision
from src.identity_engine import IdentityEngine
from src.label_manager import LabelManager
from src.mirror_manager import CropRequest, MirrorConfig, MirrorManager
from src.image_processor import ImageProcessor


def safe_join(directory: str, filename: str) -> str:
    """Safe path joining to prevent directory traversal"""
    # Normalize the paths
    directory = os.path.normpath(os.path.abspath(directory))
    filename = os.path.normpath(filename)

    # Join and normalize
    joined = os.path.normpath(os.path.join(directory, filename))

    # Ensure the result is within the directory
    if not joined.startswith(directory):
        raise ValueError("Attempted directory traversal")

    return joined


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024  # 128MB max file size

pillow_heif.register_heif_opener()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"

# Security: Define allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/heic", "image/heif"}
LAB_PAYLOAD_CACHE: Dict[str, object] = {"key": None, "payload": None}
IDENTITY_TASK_CACHE: Dict[str, object] = {"key": None, "payload": None}
SERVICE_BUNDLE_CACHE: Dict[str, object] = {"key": None, "bundle": None}
WEB_VARIANTS = {
    "thumb": {"max_size": (280, 280), "quality": 60},
    "full": {"max_size": (1600, 1600), "quality": 82},
}
LOGGER = logging.getLogger("photofinder.web")


def is_safe_path(base_path: str, target_path: str) -> bool:
    """Security check: ensure target path is within base path"""
    try:
        base = Path(base_path).resolve()
        target = Path(target_path).resolve()
        return str(target).startswith(str(base))
    except (OSError, ValueError):
        return False


def is_allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def convert_heic_to_jpeg(heic_path: str) -> bytes:
    """Convert HEIC file to JPEG bytes for web display"""
    try:
        pillow_heif.register_heif_opener()
        
        # Open HEIC and convert to RGB
        img = Image.open(heic_path)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        
        # Save to JPEG bytes
        from io import BytesIO
        jpeg_bytes = BytesIO()
        img.save(jpeg_bytes, format='JPEG', quality=85)
        jpeg_bytes.seek(0)
        return jpeg_bytes.getvalue()
    except Exception as e:
        print(f"Error converting HEIC to JPEG: {e}")
        return None


def load_image_for_web(image_path: str) -> Image.Image | None:
    """Load and normalize an image for cached web delivery."""
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image for web cache {image_path}: {e}")
        return None


def get_cache_file_path(base_dir: str, filename: str, variant: str) -> Path:
    """Build a cache path for the derived browser-friendly asset."""
    relative_path = Path(filename.replace("\\", "/"))
    suffix_tag = relative_path.suffix.lower().replace(".", "_") or "_img"
    variant_config = WEB_VARIANTS[variant]
    size_tag = f"{variant_config['max_size'][0]}x{variant_config['max_size'][1]}"
    quality_tag = f"q{variant_config['quality']}"
    cache_name = f"{relative_path.stem}{suffix_tag}-{size_tag}-{quality_tag}.webp"
    return Path(base_dir) / "_web_cache" / variant / relative_path.parent / cache_name


def get_cached_web_image(base_dir: str, filename: str, variant: str) -> Path | None:
    """Return a cached WebP derivative for browser use."""
    if variant not in WEB_VARIANTS:
        return None

    source_path = safe_join(base_dir, filename)
    if not os.path.exists(source_path) or not os.path.isfile(source_path):
        return None

    cache_path = get_cache_file_path(base_dir, filename, variant)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    source_mtime = os.path.getmtime(source_path)
    if cache_path.exists() and cache_path.stat().st_mtime >= source_mtime:
        return cache_path

    image = load_image_for_web(source_path)
    if image is None:
        return None

    derived = image.copy()
    derived.thumbnail(WEB_VARIANTS[variant]["max_size"], Image.Resampling.LANCZOS)
    derived.save(cache_path, format="WEBP", quality=WEB_VARIANTS[variant]["quality"], method=6)
    return cache_path


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f} {size_names[i]}"


def to_web_path(path: str) -> str:
    """Normalize filesystem-relative paths for use in URLs."""
    return path.replace("\\", "/") if path else path


def frontend_build_exists() -> bool:
    """Return True when a production React build is available."""
    return (FRONTEND_DIST_DIR / "index.html").exists()


def invalidate_runtime_caches() -> None:
    """Clear API payload caches after mutations."""
    LAB_PAYLOAD_CACHE["key"] = None
    LAB_PAYLOAD_CACHE["payload"] = None
    IDENTITY_TASK_CACHE["key"] = None
    IDENTITY_TASK_CACHE["payload"] = None


def get_runtime_bundle() -> Dict[str, object]:
    """Resolve the live source/mirror/intelligence services for this process."""
    if len(sys.argv) < 3:
        raise RuntimeError("PhotoFinder requires source_originals and mirror metadata directories")

    source_dir = Path(sys.argv[1]).resolve()
    data_dir = Path(sys.argv[2]).resolve()
    browser_workspace = data_dir if source_dir != data_dir else data_dir.parent / f"{data_dir.name}__mirror_workspace"
    cache_key = (str(source_dir), str(data_dir), str(browser_workspace))

    if SERVICE_BUNDLE_CACHE["key"] == cache_key and SERVICE_BUNDLE_CACHE["bundle"] is not None:
        return SERVICE_BUNDLE_CACHE["bundle"]

    mirror_manager = MirrorManager(
        MirrorConfig(source_originals=source_dir, mirror_workspace=browser_workspace)
    )
    mirror_manager.sync_source_index()

    processing_root = data_dir
    image_processor = ImageProcessor()
    ingest_manager = IngestManager(
        IngestConfig(source_originals=source_dir, processing_root=processing_root),
        mirror_manager,
        image_processor=image_processor,
    )

    intelligence_core = IntelligenceCore(
        IntelligenceConfig(workspace_root=browser_workspace / "active_learning")
    )

    bundle = {
        "source_dir": source_dir,
        "data_dir": data_dir,
        "browser_workspace": browser_workspace,
        "processing_root": processing_root,
        "mirror_manager": mirror_manager,
        "image_processor": image_processor,
        "ingest_manager": ingest_manager,
        "intelligence_core": intelligence_core,
    }
    SERVICE_BUNDLE_CACHE["key"] = cache_key
    SERVICE_BUNDLE_CACHE["bundle"] = bundle
    return bundle


def get_identity_cache_key(data_dir: Path) -> tuple[str, int | None, int | None]:
    """Cache key for the verification queue payload."""
    detections_file = data_dir / "_detections.json"
    labels_file = data_dir / "_labels.json"
    return (
        str(data_dir),
        detections_file.stat().st_mtime_ns if detections_file.exists() else None,
        labels_file.stat().st_mtime_ns if labels_file.exists() else None,
    )


def normalize_relative_path(path_value: Optional[str]) -> Optional[str]:
    """Normalize stored paths into URL-friendly separators."""
    if not path_value:
        return None
    return Path(str(path_value).replace("\\", "/")).as_posix()


def resolve_existing_asset(base_dir: Path, relative_path: Optional[str]) -> Optional[Path]:
    """Resolve an existing asset when metadata may contain mixed relative forms."""
    normalized = normalize_relative_path(relative_path)
    if not normalized:
        return None

    candidate = (base_dir / normalized).resolve()
    if candidate.exists() and candidate.is_file():
        return candidate

    trimmed = normalized
    while trimmed.startswith("../"):
        trimmed = trimmed[3:]
        candidate = (base_dir / trimmed).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def resolve_source_asset(source_dir: Path, data_dir: Path, relative_path: Optional[str]) -> tuple[Optional[str], Optional[Path]]:
    """Resolve a detection image back to the immutable source archive when possible."""
    normalized = normalize_relative_path(relative_path)
    if not normalized:
        return None, None

    source_candidate = (source_dir / normalized).resolve()
    if source_candidate.exists() and source_candidate.is_file():
        return normalized, source_candidate

    data_candidate = resolve_existing_asset(data_dir, normalized)
    if data_candidate is not None and is_safe_path(str(source_dir), str(data_candidate)):
        return data_candidate.relative_to(source_dir).as_posix(), data_candidate

    return None, None


def to_variant_name(variant: str) -> str:
    """Map web query variants into MirrorManager variants."""
    return "thumb" if variant == "thumb" else "preview"


def build_mirror_url(browser_workspace: Path, asset_path: Path) -> str:
    """Return a route URL for a generated mirror asset."""
    relative = asset_path.resolve().relative_to(browser_workspace.resolve()).as_posix()
    return f"/mirror/{quote(relative, safe='/')}"


def build_source_preview_url(mirror_manager: MirrorManager, browser_workspace: Path, relative_path: str, variant: str) -> str:
    """Materialize and reference a derivative for an immutable source asset."""
    derivative_path = mirror_manager.materialize_derivative(relative_path, variant=to_variant_name(variant))
    return build_mirror_url(browser_workspace, derivative_path)


def build_external_preview_url(mirror_manager: MirrorManager, browser_workspace: Path, asset_path: Path, cache_key: str, variant: str) -> str:
    """Materialize and reference a derivative for an asset already living in the mirror/data tree."""
    derivative_path = mirror_manager.materialize_external_derivative(asset_path, cache_key, variant=to_variant_name(variant))
    return build_mirror_url(browser_workspace, derivative_path)


def resolve_detection_asset_urls(
    *,
    mirror_manager: MirrorManager,
    source_dir: Path,
    data_dir: Path,
    browser_workspace: Path,
    image_path: str,
    crop_path: Optional[str],
    bbox: Optional[list[int]] = None,
    subject_type: str = "unknown",
    identity_hint: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Build mirror-backed candidate URLs and an embedding-ready image path."""
    source_relative, source_absolute = resolve_source_asset(source_dir, data_dir, image_path)
    crop_absolute = resolve_existing_asset(data_dir, crop_path)

    if source_relative and bbox:
        crop_artifact = mirror_manager.materialize_crop(
            CropRequest(
                relative_path=source_relative,
                bbox=tuple(int(value) for value in bbox),
                subject_type=subject_type if subject_type in {"people", "animals", "unknown"} else "unknown",
                identity_hint=identity_hint,
            )
        )
        crop_absolute = crop_artifact

    candidate_image_url = None
    candidate_full_url = None
    embedding_path = crop_absolute or source_absolute

    if crop_absolute is not None:
        candidate_image_url = build_external_preview_url(
            mirror_manager,
            browser_workspace,
            crop_absolute,
            f"candidate/{normalize_relative_path(image_path) or 'image'}",
            "thumb",
        )
        candidate_full_url = build_external_preview_url(
            mirror_manager,
            browser_workspace,
            crop_absolute,
            f"candidate/{normalize_relative_path(image_path) or 'image'}",
            "full",
        )
    elif source_relative:
        candidate_image_url = build_source_preview_url(mirror_manager, browser_workspace, source_relative, "thumb")
        candidate_full_url = build_source_preview_url(mirror_manager, browser_workspace, source_relative, "full")

    return {
        "candidate_image_url": candidate_image_url,
        "candidate_full_url": candidate_full_url,
        "embedding_path": str(embedding_path) if embedding_path else None,
        "source_relative_path": source_relative,
    }


def get_lab_cache_key(output_dir: str) -> tuple[str, int | None, int | None]:
    """Create a lightweight cache key for the lab payload."""
    output_path = Path(output_dir).resolve()
    detections_file = output_path / "_detections.json"
    labels_file = output_path / "_labels.json"

    detections_mtime = detections_file.stat().st_mtime_ns if detections_file.exists() else None
    labels_mtime = labels_file.stat().st_mtime_ns if labels_file.exists() else None

    return (str(output_path), detections_mtime, labels_mtime)


def serve_frontend_shell():
    """Serve the built React application shell."""
    index_file = FRONTEND_DIST_DIR / "index.html"
    if not index_file.exists():
        abort(404)
    return send_file(index_file)


def get_image_files(directory: str) -> List[Dict[str, str]]:
    """Get list of image files in directory with metadata"""
    if not os.path.exists(directory):
        return []

    files = []
    try:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and is_allowed_file(filename) and is_safe_path(directory, filepath):

                # Get file size for display
                try:
                    size = os.path.getsize(filepath)
                    size_str = f"{size / 1024:.1f} KB"
                except OSError:
                    size_str = "Unknown"

                files.append({"filename": filename, "path": filepath, "size": size_str})
    except (OSError, PermissionError) as e:
        print(f"Error accessing directory {directory}: {e}")

    # Sort by filename
    files.sort(key=lambda x: x["filename"].lower())
    return files


def organize_images_by_category(base_dir: str) -> Dict[str, List[Dict[str, str]]]:
    """Organize images into categories based on directory structure"""
    categories = {"people": [], "animals": [], "others": [], "both": [], "none": [], "working_dir": []}

    # Security: ensure base_dir exists and is accessible
    if not os.path.exists(base_dir):
        return categories

    # Get files from each category directory
    people_dir = os.path.join(base_dir, "people")
    animals_dir = os.path.join(base_dir, "animals")
    others_dir = os.path.join(base_dir, "others")

    people_files = {f["filename"] for f in get_image_files(people_dir)}
    animals_files = {f["filename"] for f in get_image_files(animals_dir)}
    others_files = {f["filename"] for f in get_image_files(others_dir)}

    # Get all files in base directory and subdirectories (excluding _debug_boxes)
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        # Skip _debug_boxes directory
        dirs[:] = [d for d in dirs if d != "_debug_boxes"]
        for file in files:
            if is_allowed_file(file):
                file_path = os.path.join(root, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    all_files.append({
                        "filename": file,
                        "path": file_path,
                        "size": format_size(size)
                    })

    # Sort by filename
    all_files.sort(key=lambda x: x["filename"].lower())

    # Get working directory files (files in root of working_dir, not in subfolders)
    working_files = []
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        if os.path.isfile(file_path) and is_allowed_file(file):
            size = os.path.getsize(file_path)
            working_files.append({
                "filename": file,
                "path": file_path,
                "size": format_size(size)
            })
    working_files.sort(key=lambda x: x["filename"].lower())

    # Categorize files (only from subdirectories, not root working_dir files)
    working_file_names = {f["filename"] for f in working_files}
    
    for file_info in all_files:
        filename = file_info["filename"]
        # Skip files that are in the root working directory
        if filename in working_file_names:
            continue
            
        in_people = filename in people_files
        in_animals = filename in animals_files
        in_others = filename in others_files

        if in_people and in_animals:
            categories["both"].append(file_info)
        elif in_people:
            categories["people"].append(file_info)
        elif in_animals:
            categories["animals"].append(file_info)
        elif in_others:
            categories["others"].append(file_info)
        else:
            categories["none"].append(file_info)

    categories["working_dir"] = working_files

    return categories


def get_identity_stage(sample_count: int) -> Dict[str, object]:
    """Translate label counts into a more human progress story."""
    if sample_count >= 12:
        return {
            "title": "Dialed In",
            "summary": "This identity has enough examples to feel stable.",
            "tone": "dialed",
            "target": None,
            "progress": 100,
        }
    if sample_count >= 6:
        return {
            "title": "Locking In",
            "summary": "The model should start making consistent suggestions soon.",
            "tone": "locking",
            "target": 12,
            "progress": int((sample_count / 12) * 100),
        }
    if sample_count >= 3:
        return {
            "title": "Warming Up",
            "summary": "There is enough signal to start recognizing patterns.",
            "tone": "warming",
            "target": 6,
            "progress": int((sample_count / 6) * 100),
        }

    return {
        "title": "Spark",
        "summary": "The identity exists, but it still needs more examples.",
        "tone": "spark",
        "target": 3,
        "progress": int((sample_count / 3) * 100) if sample_count else 0,
    }


def build_identity_collections(label_manager: LabelManager) -> List[Dict[str, object]]:
    """Build a named-identity view from confirmed labels."""
    training_data = label_manager.export_for_training(status="confirmed")
    collections = []

    for assigned_label, examples in sorted(training_data.items(), key=lambda item: (-len(item[1]), item[0].lower())):
        people_count = sum(1 for example in examples if example.get("detected_class") == "person")
        animal_count = sum(1 for example in examples if example.get("detected_class") != "person")
        sample_count = len(examples)
        stage = get_identity_stage(sample_count)

        cover_path = None
        for example in examples:
            cover_path = example.get("crop_path") or example.get("image_path")
            if cover_path:
                cover_path = to_web_path(cover_path)
                break

        collections.append(
            {
                "name": assigned_label,
                "sample_count": sample_count,
                "people_count": people_count,
                "animal_count": animal_count,
                "cover_path": cover_path,
                "stage": stage,
                "next_target": stage["target"],
                "missing_to_next": max(stage["target"] - sample_count, 0) if stage["target"] else 0,
            }
        )

    return collections


def build_lab_insights(detections: List[Dict[str, object]], identity_collections: List[Dict[str, object]]) -> Dict[str, object]:
    """Create product guidance for the current labeling session."""
    pending_detections = [detection for detection in detections if detection["status"] == "pending"]
    focus_detection = pending_detections[0] if pending_detections else None
    ready_identities = sum(1 for identity in identity_collections if identity["sample_count"] >= 3)
    strongest_identity = identity_collections[0] if identity_collections else None
    likely_matches = sum(1 for detection in pending_detections if detection.get("review_bucket") == "likely")

    if strongest_identity:
        momentum = f"{strongest_identity['name']} leads with {strongest_identity['sample_count']} labeled samples."
    else:
        momentum = "Start by locking in a few obvious faces or pets to create your first identity spark."

    if likely_matches:
        action_prompt = f"{likely_matches} likely matches are ready for one-click confirmation."
    elif pending_detections:
        action_prompt = f"{len(pending_detections)} detections are still waiting for a name."
    else:
        action_prompt = "Everything visible is labeled. The next move is building identity strength with more varied examples."

    suggestion_count = sum(1 for detection in pending_detections if detection.get("suggestion"))

    return {
        "focus_detection": focus_detection,
        "ready_identities": ready_identities,
        "momentum": momentum,
        "action_prompt": action_prompt,
        "suggestion_count": suggestion_count,
    }


def serialize_gallery_items(items: List[Dict[str, str]], base_dir: str, route_prefix: str, limit: int = 6) -> List[Dict[str, str]]:
    """Prepare gallery items for JSON responses."""
    serialized = []

    for item in items[:limit]:
        relative_path = to_web_path(os.path.relpath(item["path"], base_dir))
        serialized.append(
            {
                "filename": item["filename"],
                "size": item["size"],
                "url": f"{route_prefix}/{relative_path}?variant=thumb",
            }
        )

    return serialized


def get_subject_group(detected_class: str) -> str:
    """Map concrete detector classes into the product's two major lanes."""
    return "people" if detected_class == "person" else "animals"


def get_review_bucket(detection: Dict[str, object]) -> str:
    """Assign queue buckets that drive the lab workflow."""
    status = detection["status"]
    if status == "confirmed":
        return "confirmed"
    if status == "rejected":
        return "rejected"

    suggestion = detection.get("suggestion") or {}
    if suggestion:
        if suggestion.get("confidence", 0) >= 90 and suggestion.get("runner_up_gap", 0) >= 0.015:
            return "likely"
        return "review"

    if detection.get("confidence", 0) >= 0.6 or detection.get("bbox_area", 0) >= 250000:
        return "queue"

    return "backlog"


def build_queue_summary(detections: List[Dict[str, object]]) -> Dict[str, int]:
    """Build bucket counts for queue filters and review actions."""
    summary = {
        "all": len(detections),
        "likely": 0,
        "review": 0,
        "queue": 0,
        "backlog": 0,
        "done": 0,
        "people": 0,
        "animals": 0,
        "suggested": 0,
    }

    for detection in detections:
        subject_group = detection.get("subject_group", "animals")
        summary[subject_group] += 1

        if detection["status"] != "pending":
            summary["done"] += 1
            continue

        bucket = detection.get("review_bucket", "backlog")
        if bucket in summary:
            summary[bucket] += 1
        if detection.get("suggestion"):
            summary["suggested"] += 1

    return summary


def bootstrap_intelligence_index(
    *,
    intelligence_core: IntelligenceCore,
    label_manager: LabelManager,
    mirror_manager: MirrorManager,
    source_dir: Path,
    data_dir: Path,
    browser_workspace: Path,
) -> None:
    """Rebuild the live vector index from confirmed labels."""
    intelligence_core.reset_vector_index()

    for label_record in label_manager.get_all_labels(status="confirmed"):
        assigned_label = str(label_record.get("assigned_label") or "").strip()
        if not assigned_label:
            continue

        subject_type = get_subject_group(str(label_record.get("detected_class") or ""))
        asset_payload = resolve_detection_asset_urls(
            mirror_manager=mirror_manager,
            source_dir=source_dir,
            data_dir=data_dir,
            browser_workspace=browser_workspace,
            image_path=str(label_record.get("image_path") or ""),
            crop_path=label_record.get("crop_path"),
            bbox=label_record.get("bbox"),
            subject_type=subject_type,
            identity_hint=assigned_label,
        )
        embedding_path = asset_payload.get("embedding_path")
        if not embedding_path:
            continue

        intelligence_core.learn_from_label(
            relative_path=normalize_relative_path(label_record.get("image_path")) or assigned_label,
            image_path=Path(embedding_path),
            identity_label=assigned_label,
            subject_type=subject_type,
            metadata={
                "preview_url": asset_payload.get("candidate_image_url"),
                "full_url": asset_payload.get("candidate_full_url"),
                "image_path": normalize_relative_path(label_record.get("image_path")),
                "crop_path": normalize_relative_path(label_record.get("crop_path")),
                "label_id": label_record.get("id"),
            },
            human_verified=False,
            schedule_fine_tune=False,
        )


def build_identity_tasks(limit: int = 18) -> List[Dict[str, object]]:
    """Build verification tasks from pending detections using the active-learning core."""
    bundle = get_runtime_bundle()
    source_dir = bundle["source_dir"]
    data_dir = bundle["data_dir"]
    browser_workspace = bundle["browser_workspace"]
    mirror_manager = bundle["mirror_manager"]
    intelligence_core = bundle["intelligence_core"]

    cache_key = get_identity_cache_key(data_dir)
    if IDENTITY_TASK_CACHE["key"] == cache_key and IDENTITY_TASK_CACHE["payload"] is not None:
        return IDENTITY_TASK_CACHE["payload"]

    metadata_manager = DetectionMetadata(str(data_dir))
    label_manager = LabelManager(str(data_dir))
    bootstrap_intelligence_index(
        intelligence_core=intelligence_core,
        label_manager=label_manager,
        mirror_manager=mirror_manager,
        source_dir=source_dir,
        data_dir=data_dir,
        browser_workspace=browser_workspace,
    )

    tasks = []
    detection_records = metadata_manager.get_images_with_detections()

    for record in detection_records:
        image_rel_path = to_web_path(record["image_path"])
        existing_labels = label_manager.get_labels_for_image(os.path.join(str(data_dir), image_rel_path))
        existing_labels_by_index = {label["detection_index"]: label for label in existing_labels}

        for detection_index, detection in enumerate(record["detections"]):
            if detection_index in existing_labels_by_index:
                continue

            detected_class = str(detection.get("class_name") or "")
            subject_type = get_subject_group(detected_class)
            asset_payload = resolve_detection_asset_urls(
                mirror_manager=mirror_manager,
                source_dir=source_dir,
                data_dir=data_dir,
                browser_workspace=browser_workspace,
                image_path=image_rel_path,
                crop_path=detection.get("crop_path"),
                bbox=detection.get("bbox"),
                subject_type=subject_type,
                identity_hint=detected_class,
            )
            embedding_path = asset_payload.get("embedding_path")
            if not embedding_path:
                continue

            task = intelligence_core.propose_identity(
                relative_path=f"{image_rel_path}#{detection_index}",
                subject_type=subject_type,
                image_path=Path(embedding_path),
            )

            if task.status == "new_identity" or not task.proposed_label:
                continue

            task.candidate_image_url = asset_payload.get("candidate_image_url")
            task.candidate_full_url = asset_payload.get("candidate_full_url")
            task.metadata = {
                "image_path": image_rel_path,
                "detection_index": detection_index,
                "bbox": detection.get("bbox"),
                "crop_path": normalize_relative_path(detection.get("crop_path")),
                "detected_class": detected_class,
                "confidence": detection.get("confidence"),
                "subject_type": subject_type,
                "source_relative_path": asset_payload.get("source_relative_path") or image_rel_path,
            }

            gallery_items = []
            for gallery_label in label_manager.get_all_labels(status="confirmed", assigned_label=task.proposed_label)[:4]:
                gallery_assets = resolve_detection_asset_urls(
                    mirror_manager=mirror_manager,
                    source_dir=source_dir,
                    data_dir=data_dir,
                    browser_workspace=browser_workspace,
                    image_path=str(gallery_label.get("image_path") or ""),
                    crop_path=gallery_label.get("crop_path"),
                    bbox=gallery_label.get("bbox"),
                    subject_type=get_subject_group(str(gallery_label.get("detected_class") or "")),
                    identity_hint=task.proposed_label,
                )
                if gallery_assets.get("candidate_image_url"):
                    gallery_items.append(
                        {
                            "imageUrl": gallery_assets.get("candidate_image_url"),
                            "fullUrl": gallery_assets.get("candidate_full_url"),
                            "label": task.proposed_label,
                        }
                    )

            task.known_gallery = gallery_items
            tasks.append(task.model_dump(mode="json"))

            if len(tasks) >= limit:
                break

        if len(tasks) >= limit:
            break

    tasks.sort(key=lambda item: (0 if item["status"] == "needs_confirmation" else 1, item["confidence"]))
    IDENTITY_TASK_CACHE["key"] = cache_key
    IDENTITY_TASK_CACHE["payload"] = tasks
    return tasks


def build_upload_response(result) -> Dict[str, object]:
    """Create a stable JSON payload for the upload UI."""
    queue_preview = build_identity_tasks(limit=18)
    matching_tasks = [task for task in queue_preview if task.get("metadata", {}).get("image_path") == result.canonical_relative_path]
    preview_url = f"/working_dir/{quote(result.source_relative_path, safe='/')}?variant=thumb"

    return {
        "success": True,
        "duplicate": result.duplicate,
        "clean_name": result.clean_name,
        "source_relative_path": result.source_relative_path,
        "canonical_relative_path": result.canonical_relative_path,
        "preview_url": preview_url,
        "detection_count": result.detection_count,
        "has_person": result.has_person,
        "has_animal": result.has_animal,
        "verification_queue_count": len(queue_preview),
        "uploaded_task_count": len(matching_tasks),
        "identity_lab_url": "/identity-lab",
    }


def persist_identity_decision(decision: VerificationDecision) -> Dict[str, object]:
    """Persist a verification decision and update the live intelligence core."""
    bundle = get_runtime_bundle()
    source_dir = bundle["source_dir"]
    data_dir = bundle["data_dir"]
    browser_workspace = bundle["browser_workspace"]
    mirror_manager = bundle["mirror_manager"]
    intelligence_core = bundle["intelligence_core"]

    metadata = decision.metadata or {}
    image_path = normalize_relative_path(metadata.get("image_path"))
    if not image_path:
        raise ValueError("VerificationDecision metadata must include image_path")

    detection_index = int(metadata.get("detection_index", -1))
    if detection_index < 0:
        raise ValueError("VerificationDecision metadata must include detection_index")

    metadata_manager = DetectionMetadata(str(data_dir))
    label_manager = LabelManager(str(data_dir))
    detection_record = metadata_manager.get_detections(os.path.join(str(data_dir), image_path))
    if not detection_record or not detection_record.get("detections"):
        raise FileNotFoundError("No detection data found for the verification task")

    detections = detection_record["detections"]
    if detection_index >= len(detections):
        raise ValueError("Invalid detection index for verification task")

    detection = detections[detection_index]
    existing_labels = label_manager.get_labels_for_image(os.path.join(str(data_dir), image_path))
    existing_label = next((label for label in existing_labels if label["detection_index"] == detection_index), None)

    asset_payload = resolve_detection_asset_urls(
        mirror_manager=mirror_manager,
        source_dir=source_dir,
        data_dir=data_dir,
        browser_workspace=browser_workspace,
        image_path=image_path,
        crop_path=detection.get("crop_path"),
        bbox=detection.get("bbox"),
        subject_type=decision.subject_type,
        identity_hint=decision.confirmed_label,
    )
    embedding_path = Path(asset_payload["embedding_path"]) if asset_payload.get("embedding_path") else None

    if decision.accepted and decision.confirmed_label:
        if existing_label:
            label_manager.update_label(existing_label["id"], assigned_label=decision.confirmed_label, status="confirmed")
        else:
            label_manager.add_label(
                image_path=os.path.join(str(data_dir), image_path),
                detection_index=detection_index,
                assigned_label=decision.confirmed_label,
                detected_class=detection.get("class_name", ""),
                bbox=detection.get("bbox", []),
                crop_path=detection.get("crop_path"),
                status="confirmed",
            )

        intelligence_core.learn_from_label(
            relative_path=decision.relative_path,
            image_path=embedding_path,
            identity_label=decision.confirmed_label,
            subject_type=decision.subject_type,
            metadata={
                "preview_url": asset_payload.get("candidate_image_url"),
                "full_url": asset_payload.get("candidate_full_url"),
                "image_path": image_path,
                "crop_path": normalize_relative_path(detection.get("crop_path")),
            },
            human_verified=True,
            schedule_fine_tune=decision.schedule_fine_tune,
        )
        outcome = "confirmed"
    else:
        if existing_label:
            label_manager.update_label_status(existing_label["id"], "rejected")
        else:
            label_manager.add_label(
                image_path=os.path.join(str(data_dir), image_path),
                detection_index=detection_index,
                assigned_label="",
                detected_class=detection.get("class_name", ""),
                bbox=detection.get("bbox", []),
                crop_path=detection.get("crop_path"),
                status="rejected",
            )
        intelligence_core.apply_verification_decision(decision, image_path=embedding_path)
        outcome = "rejected"

    export_dir = label_manager.rebuild_named_exports()
    invalidate_runtime_caches()
    return {"success": True, "outcome": outcome, "export_dir": export_dir}


def build_dashboard_payload(input_dir: str, output_dir: str) -> Dict[str, object]:
    """Create a frontend-friendly snapshot of atlas data."""
    categories = organize_images_by_category(output_dir)
    label_manager = LabelManager(output_dir)
    identity_collections = build_identity_collections(label_manager)

    lane_payload = {
        "working_dir": {
            "title": "Original Photos",
            "count": len(categories["working_dir"]),
            "items": serialize_gallery_items(categories["working_dir"], input_dir, "/working_dir"),
        },
        "people": {
            "title": "People Signals",
            "count": len(categories["people"]) + len(categories["both"]),
            "items": serialize_gallery_items(categories["people"], output_dir, "/image/people"),
        },
        "animals": {
            "title": "Animal Signals",
            "count": len(categories["animals"]) + len(categories["both"]),
            "items": serialize_gallery_items(categories["animals"], output_dir, "/image/animals"),
        },
        "mixed": {
            "title": "Mixed Scenes",
            "count": len(categories["both"]),
            "items": serialize_gallery_items(categories["both"], output_dir, "/image"),
        },
    }

    collections_payload = [
        {
            "name": identity["name"],
            "sampleCount": identity["sample_count"],
            "peopleCount": identity["people_count"],
            "animalCount": identity["animal_count"],
            "stage": identity["stage"],
            "missingToNext": identity["missing_to_next"],
            "coverUrl": f"/image/{identity['cover_path']}?variant=thumb" if identity.get("cover_path") else None,
        }
        for identity in identity_collections
    ]

    return {
        "hero": {
            "title": "Identity Atlas",
            "summary": "A React-powered front door for growing named people and pet collections from your photo stream.",
            "momentum": collections_payload[0]["name"] if collections_payload else None,
        },
        "stats": {
            "originalCount": len(categories["working_dir"]),
            "processedCount": len(categories["people"]) + len(categories["animals"]) + len(categories["both"]) + len(categories["none"]),
            "identityCount": len(collections_payload),
            "peopleSignalCount": lane_payload["people"]["count"],
            "animalSignalCount": lane_payload["animals"]["count"],
        },
        "identityCollections": collections_payload,
        "lanes": lane_payload,
    }


def build_lab_payload(output_dir: str) -> Dict[str, object]:
    """Create a frontend-friendly snapshot of label-lab data."""
    cache_key = get_lab_cache_key(output_dir)
    if LAB_PAYLOAD_CACHE["key"] == cache_key and LAB_PAYLOAD_CACHE["payload"] is not None:
        return LAB_PAYLOAD_CACHE["payload"]

    metadata_manager = DetectionMetadata(output_dir)
    label_manager = LabelManager(output_dir)
    identity_engine = IdentityEngine(output_dir)
    prototypes = identity_engine.build_index(label_manager)

    detections_with_labels = []
    detection_records = metadata_manager.get_images_with_detections()

    for record in detection_records:
        image_rel_path = record["image_path"]
        detections = record["detections"]

        existing_labels = label_manager.get_labels_for_image(os.path.join(output_dir, image_rel_path))
        existing_labels_by_index = {label["detection_index"]: label for label in existing_labels}

        for index, detection in enumerate(detections):
            existing_label = existing_labels_by_index.get(index)
            detection_info = {
                "image_path": to_web_path(image_rel_path),
                "detection_index": index,
                "detected_class": detection["class_name"],
                "subject_group": get_subject_group(detection["class_name"]),
                "confidence": detection["confidence"],
                "bbox": detection["bbox"],
                "bbox_area": detection.get("bbox_area", 0),
                "crop_path": to_web_path(detection.get("crop_path")),
                "status": existing_label["status"] if existing_label else "pending",
                "assigned_label": existing_label["assigned_label"] if existing_label else None,
                "label_id": existing_label["id"] if existing_label else None,
                "suggestion": None,
            }

            if not existing_label:
                detection_info["suggestion"] = identity_engine.suggest_from_prototypes(detection_info, prototypes)

            detection_info["review_bucket"] = get_review_bucket(detection_info)

            detections_with_labels.append(detection_info)

    detections_with_labels.sort(
        key=lambda detection: (
            0 if detection["status"] == "pending" else 1,
            {"likely": 0, "review": 1, "queue": 2, "backlog": 3, "confirmed": 4, "rejected": 5}.get(detection.get("review_bucket"), 6),
            -detection["confidence"],
            -detection.get("bbox_area", 0),
        )
    )

    identity_collections = build_identity_collections(label_manager)
    lab_insights = build_lab_insights(detections_with_labels, identity_collections)

    stats = label_manager.get_label_statistics()
    stats["total_detections"] = len(detections_with_labels)
    stats["pending_labels"] = sum(1 for detection in detections_with_labels if detection["status"] == "pending")
    stats["identity_collections"] = len(identity_collections)
    stats["ready_identities"] = lab_insights["ready_identities"]
    queue_summary = build_queue_summary(detections_with_labels)

    payload = {
        "detections": detections_with_labels,
        "stats": stats,
        "quick_labels": stats["unique_assigned_labels"],
        "identity_collections": identity_collections,
        "lab_insights": lab_insights,
        "queue_summary": queue_summary,
    }

    LAB_PAYLOAD_CACHE["key"] = cache_key
    LAB_PAYLOAD_CACHE["payload"] = payload
    return payload


@app.route("/")
def index():
    """Main page - show image categories"""
    if len(sys.argv) < 3:
        return (
            "Error: PhotoFinder web UI requires input and output directories.<br>"
            "Usage: python web_ui.py <input_directory> <output_directory>",
            400,
        )

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Security: validate paths
    if not is_safe_path(os.getcwd(), input_dir) or not is_safe_path(os.getcwd(), output_dir):
        return "Error: Invalid directory paths", 400

    if frontend_build_exists():
        return serve_frontend_shell()

    categories = organize_images_by_category(output_dir)
    label_manager = LabelManager(output_dir)
    identity_collections = build_identity_collections(label_manager)

    return render_template(
        "index.html",
        input_dir=input_dir,
        output_dir=output_dir,
        categories=categories,
        identity_collections=identity_collections,
    )


@app.route("/api/dashboard")
def dashboard_api():
    """JSON payload for the React frontend dashboard."""
    if len(sys.argv) < 3:
        return jsonify({"error": "Missing directory arguments"}), 400

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not is_safe_path(os.getcwd(), input_dir) or not is_safe_path(os.getcwd(), output_dir):
        return jsonify({"error": "Invalid directory paths"}), 400

    return jsonify(build_dashboard_payload(input_dir, output_dir))


@app.route("/lab")
def react_lab():
    """Serve the React app at the lab route when available."""
    if frontend_build_exists():
        return serve_frontend_shell()
    return label_page()


@app.route("/identity-lab")
def react_identity_lab():
    """Serve the React app at the identity verification route when available."""
    if frontend_build_exists():
        return serve_frontend_shell()
    return label_page()


@app.route("/api/lab")
def lab_api():
    """JSON payload for the React frontend lab view."""
    if len(sys.argv) < 3:
        return jsonify({"error": "Missing directory arguments"}), 400

    output_dir = sys.argv[2]

    if not is_safe_path(os.getcwd(), output_dir):
        return jsonify({"error": "Invalid directory paths"}), 400

    return jsonify(build_lab_payload(output_dir))


@app.route("/api/identity/tasks")
def identity_tasks_api():
    """JSON payload for the active-learning verification queue."""
    try:
        bundle = get_runtime_bundle()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    if not is_safe_path(str(BASE_DIR), str(bundle["data_dir"])):
        return jsonify({"error": "Invalid directory paths"}), 400

    return jsonify(build_identity_tasks())


@app.route("/api/identity/verify", methods=["POST"])
def identity_verify_api():
    """Accept a verification decision and update labels plus embeddings immediately."""
    try:
        decision = VerificationDecision.model_validate(request.get_json() or {})
        return jsonify(persist_identity_decision(decision))
    except Exception as exc:
        LOGGER.exception("Identity verification failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ingest/upload", methods=["POST"])
def ingest_upload_api():
    """Handle vault-safe file uploads and immediately mirror them into the app workflow."""
    try:
        bundle = get_runtime_bundle()
        upload = request.files.get("file")
        if upload is None:
            return jsonify({"error": "Missing uploaded file under field name 'file'"}), 400

        result = bundle["ingest_manager"].ingest_upload(upload)
        invalidate_runtime_caches()
        return jsonify(build_upload_response(result))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        LOGGER.exception("Upload ingest failed")
        return jsonify({"error": str(exc)}), 500


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    """Serve built frontend assets."""
    if not frontend_build_exists():
        abort(404)
    return send_from_directory(FRONTEND_DIST_DIR / "assets", filename, max_age=3600)


@app.route("/favicon.svg")
def frontend_favicon():
    """Serve frontend favicon when present."""
    if not frontend_build_exists():
        abort(404)
    return send_from_directory(FRONTEND_DIST_DIR, "favicon.svg", max_age=3600)


@app.route("/mirror/<path:filename>")
def serve_mirror_asset(filename):
    """Serve generated assets that already live inside the mirror workspace."""
    bundle = get_runtime_bundle()
    browser_workspace = bundle["browser_workspace"]

    try:
        safe_path = safe_join(str(browser_workspace), filename)
        if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
            abort(404)
        return send_file(safe_path, conditional=True, max_age=3600)
    except Exception:
        abort(500)


@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve mirror-derived previews for assets that live in the mirror/data tree."""
    bundle = get_runtime_bundle()
    data_dir = bundle["data_dir"]
    browser_workspace = bundle["browser_workspace"]
    mirror_manager = bundle["mirror_manager"]
    variant = request.args.get("variant", "thumb")

    try:
        safe_path = safe_join(str(data_dir), filename)
        if os.path.exists(safe_path) and os.path.isfile(safe_path):
            derivative_path = mirror_manager.materialize_external_derivative(
                Path(safe_path),
                f"image/{to_web_path(filename)}",
                variant=to_variant_name(variant),
            )
        else:
            source_relative, source_path = resolve_source_asset(bundle["source_dir"], data_dir, filename)
            if not source_relative or source_path is None:
                abort(404)
            derivative_path = mirror_manager.materialize_derivative(source_relative, variant=to_variant_name(variant))

        return send_file(derivative_path, mimetype="image/webp", conditional=True, max_age=3600)
    except Exception:
        abort(500)


@app.route("/working_dir/<path:filename>")
def serve_working_dir(filename):
    """Serve mirror-derived previews for immutable source originals."""
    bundle = get_runtime_bundle()
    mirror_manager = bundle["mirror_manager"]
    variant = request.args.get("variant", "thumb")

    try:
        derivative_path = mirror_manager.materialize_derivative(filename, variant=to_variant_name(variant))
        return send_file(derivative_path, mimetype="image/webp", conditional=True, max_age=3600)
    except Exception:
        abort(500)


@app.route("/label")
def label_page():
    """Labeling page for reviewing and assigning labels to detections"""
    if len(sys.argv) < 3:
        return "Error: PhotoFinder web UI requires input and output directories.", 400

    output_dir = sys.argv[2]

    # Security: validate path
    if not is_safe_path(os.getcwd(), output_dir):
        return "Error: Invalid directory paths", 400

    if frontend_build_exists():
        return serve_frontend_shell()

    lab_payload = build_lab_payload(output_dir)

    return render_template(
        "label.html",
        detections=lab_payload["detections"],
        stats=lab_payload["stats"],
        quick_labels=lab_payload["quick_labels"],
        identity_collections=lab_payload["identity_collections"],
        lab_insights=lab_payload["lab_insights"],
    )


@app.route("/api/label", methods=["POST"])
def save_label():
    """API endpoint for saving/rejecting labels"""
    if len(sys.argv) < 3:
        return jsonify({"error": "Missing directory arguments"}), 400

    output_dir = sys.argv[2]

    # Security: validate path
    if not is_safe_path(os.getcwd(), output_dir):
        return jsonify({"error": "Invalid directory paths"}), 400

    try:
        data = request.get_json()
        image_path = data.get("image_path")
        detection_index = int(data.get("detection_index"))
        assigned_label = data.get("assigned_label")
        status = data.get("status", "confirmed")

        if not image_path:
            return jsonify({"error": "Missing image_path"}), 400

        # Initialize managers
        metadata_manager = DetectionMetadata(output_dir)
        label_manager = LabelManager(output_dir)

        # Get detection metadata
        detection_record = metadata_manager.get_detections(os.path.join(output_dir, image_path))
        if not detection_record or not detection_record.get("detections"):
            return jsonify({"error": "No detection data found"}), 404

        detections = detection_record["detections"]
        if detection_index >= len(detections):
            return jsonify({"error": "Invalid detection index"}), 400

        detection = detections[detection_index]
        
        # Check if label already exists
        existing_labels = label_manager.get_labels_for_image(os.path.join(output_dir, image_path))
        existing_label = None
        
        for label in existing_labels:
            if label["detection_index"] == detection_index:
                existing_label = label
                break

        if existing_label:
            # Update existing label
            if status == "pending":
                label_manager.delete_label(existing_label["id"])
            elif status == "rejected":
                label_manager.update_label_status(existing_label["id"], "rejected")
            elif assigned_label:
                # Update with new label
                label_manager.update_label(existing_label["id"], assigned_label=assigned_label, status="confirmed")
        else:
            # Create new label
            if status == "pending":
                return jsonify({"success": True, "export_dir": label_manager.rebuild_named_exports()})
            if status != "rejected" and assigned_label:
                label_manager.add_label(
                    image_path=os.path.join(output_dir, image_path),
                    detection_index=detection_index,
                    assigned_label=assigned_label,
                    detected_class=detection["class_name"],
                    bbox=detection["bbox"],
                    crop_path=detection.get("crop_path"),
                    status="confirmed"
                )
            elif status == "rejected":
                label_manager.add_label(
                    image_path=os.path.join(output_dir, image_path),
                    detection_index=detection_index,
                    assigned_label="",
                    detected_class=detection["class_name"],
                    bbox=detection["bbox"],
                    crop_path=detection.get("crop_path"),
                    status="rejected"
                )

        export_dir = label_manager.rebuild_named_exports()

        if status == "confirmed" and assigned_label:
            bundle = get_runtime_bundle()
            subject_type = get_subject_group(detection.get("class_name", ""))
            asset_payload = resolve_detection_asset_urls(
                mirror_manager=bundle["mirror_manager"],
                source_dir=bundle["source_dir"],
                data_dir=bundle["data_dir"],
                browser_workspace=bundle["browser_workspace"],
                image_path=image_path,
                crop_path=detection.get("crop_path"),
                bbox=detection.get("bbox"),
                subject_type=subject_type,
                identity_hint=assigned_label,
            )
            embedding_path = Path(asset_payload["embedding_path"]) if asset_payload.get("embedding_path") else None
            if embedding_path is not None:
                bundle["intelligence_core"].learn_from_label(
                    relative_path=f"{image_path}#{detection_index}",
                    image_path=embedding_path,
                    identity_label=assigned_label,
                    subject_type=subject_type,
                    metadata={
                        "preview_url": asset_payload.get("candidate_image_url"),
                        "full_url": asset_payload.get("candidate_full_url"),
                        "image_path": image_path,
                    },
                    human_verified=True,
                    schedule_fine_tune=False,
                )

        invalidate_runtime_caches()

        return jsonify({"success": True, "export_dir": export_dir})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return "File not found", 404


@app.errorhandler(403)
def forbidden(error):
    return "Access forbidden", 403


@app.errorhandler(500)
def internal_error(error):
    return "Internal server error", 500


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python web_ui.py <input_directory> <output_directory>")
        print("Example: python web_ui.py ./working_dir ./sorted_output")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)

    print("Starting PhotoFinder Web UI...")
    print(f"Working directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print("Open http://localhost:5000 in your browser")

    # Run in development mode with security considerations
    app.run(host="127.0.0.1", port=5000, debug=False)
