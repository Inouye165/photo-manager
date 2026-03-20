#!/usr/bin/env python3
"""
Web UI for PhotoFinder - shows thumbnails organized by detection categories.
Security-focused implementation with proper input validation and file access controls.
"""

import mimetypes
import os
import sys
from pathlib import Path
from typing import Dict, List

from flask import Flask, abort, render_template, send_file, request, jsonify

from src.detection_metadata import DetectionMetadata
from src.identity_engine import IdentityEngine
from src.label_manager import LabelManager


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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Security: Define allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff", "image/heic", "image/heif"}


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
        from PIL import Image
        import pillow_heif
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

    if strongest_identity:
        momentum = f"{strongest_identity['name']} leads with {strongest_identity['sample_count']} labeled samples."
    else:
        momentum = "Start by locking in a few obvious faces or pets to create your first identity spark."

    if pending_detections:
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
                "url": f"{route_prefix}/{relative_path}",
            }
        )

    return serialized


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
            "coverUrl": f"/image/{identity['cover_path']}" if identity.get("cover_path") else None,
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


@app.route("/image/<path:filename>")
def serve_image(filename):
    """Serve images securely with proper validation"""
    if len(sys.argv) < 3:
        abort(400)

    output_dir = sys.argv[2]

    # Security: construct safe file path
    try:
        safe_path = safe_join(output_dir, filename)
        if not safe_path or not is_safe_path(output_dir, safe_path):
            abort(403)

        if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
            abort(404)

        # Security: validate file type
        if not is_allowed_file(safe_path):
            abort(403)

        # Security: validate MIME type
        mime_type, _ = mimetypes.guess_type(safe_path)
        # Handle HEIC files explicitly since mimetypes.guess_type doesn't recognize them
        if safe_path.lower().endswith(('.heic', '.heif')):
            mime_type = 'image/heic'
            # Convert HEIC to JPEG for web display
            jpeg_bytes = convert_heic_to_jpeg(safe_path)
            if jpeg_bytes:
                from flask import Response
                return Response(jpeg_bytes, mimetype='image/jpeg')
        if mime_type not in ALLOWED_MIME_TYPES:
            abort(403)

        return send_file(safe_path)

    except Exception:
        abort(500)


@app.route("/working_dir/<path:filename>")
def serve_working_dir(filename):
    """Serve images from working directory"""
    if len(sys.argv) < 3:
        abort(400)

    input_dir = sys.argv[1]

    # Security: construct safe file path
    try:
        safe_path = safe_join(input_dir, filename)
        if not safe_path or not is_safe_path(input_dir, safe_path):
            abort(403)

        if not os.path.exists(safe_path) or not os.path.isfile(safe_path):
            abort(404)

        # Security: validate file type
        if not is_allowed_file(safe_path):
            abort(403)

        # Security: validate MIME type
        mime_type, _ = mimetypes.guess_type(safe_path)
        # Handle HEIC files explicitly since mimetypes.guess_type doesn't recognize them
        if safe_path.lower().endswith(('.heic', '.heif')):
            mime_type = 'image/heic'
            # Convert HEIC to JPEG for web display
            jpeg_bytes = convert_heic_to_jpeg(safe_path)
            if jpeg_bytes:
                from flask import Response
                return Response(jpeg_bytes, mimetype='image/jpeg')
        if mime_type not in ALLOWED_MIME_TYPES:
            abort(403)

        return send_file(safe_path)

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

    # Initialize managers
    metadata_manager = DetectionMetadata(output_dir)
    label_manager = LabelManager(output_dir)
    identity_engine = IdentityEngine(output_dir)
    prototypes = identity_engine.build_index(label_manager)

    # Get all detections with labels
    detections_with_labels = []
    
    # Detection metadata already contains only labelable subjects from the processing pipeline.
    detection_records = metadata_manager.get_images_with_detections()
    
    for record in detection_records:
        image_rel_path = record["image_path"]
        detections = record["detections"]
        
        # Get existing labels for this image
        existing_labels = label_manager.get_labels_for_image(os.path.join(output_dir, image_rel_path))
        existing_labels_by_index = {label["detection_index"]: label for label in existing_labels}
        
        for i, detection in enumerate(detections):
            # Check if this detection already has a label
            existing_label = existing_labels_by_index.get(i)
            
            detection_info = {
                "image_path": to_web_path(image_rel_path),
                "detection_index": i,
                "detected_class": detection["class_name"],
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
                suggestion = identity_engine.suggest_from_prototypes(detection_info, prototypes)
                detection_info["suggestion"] = suggestion
            
            detections_with_labels.append(detection_info)
    
    detections_with_labels.sort(
        key=lambda detection: (
            0 if detection["status"] == "pending" else 1,
            0 if detection.get("suggestion") else 1,
            -detection["confidence"],
            -detection.get("bbox_area", 0),
        )
    )

    identity_collections = build_identity_collections(label_manager)
    lab_insights = build_lab_insights(detections_with_labels, identity_collections)

    # Get statistics
    stats = label_manager.get_label_statistics()
    stats["total_detections"] = len(detections_with_labels)
    stats["pending_labels"] = sum(1 for d in detections_with_labels if d["status"] == "pending")
    stats["identity_collections"] = len(identity_collections)
    stats["ready_identities"] = lab_insights["ready_identities"]
    
    return render_template(
        "label.html",
        detections=detections_with_labels,
        stats=stats,
        quick_labels=stats["unique_assigned_labels"],
        identity_collections=identity_collections,
        lab_insights=lab_insights,
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
            if status == "rejected":
                label_manager.update_label_status(existing_label["id"], "rejected")
            elif assigned_label:
                # Update with new label
                label_manager.update_label(existing_label["id"], assigned_label=assigned_label, status="confirmed")
        else:
            # Create new label
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
