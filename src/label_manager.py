"""
Label management system for PhotoFinder labeling workflow.
Stores and retrieves assigned labels for detected people and animals.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class LabelManager:
    """Manages label storage and retrieval for detected objects"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.labels_file = self.output_dir / "_labels.json"
        self.labels_file.parent.mkdir(parents=True, exist_ok=True)
        self._labels = self._load_labels()
    
    def _load_labels(self) -> Dict:
        """Load existing labels from file"""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                    self._normalize_loaded_labels(labels)
                    return labels
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load labels from {self.labels_file}")
        
        return {"labels": {}, "last_updated": None}

    def _normalize_relative_path(self, path_value: Optional[str]) -> Optional[str]:
        """Store paths relative to output_dir using URL-friendly separators."""
        if not path_value:
            return None

        normalized = str(path_value).replace("\\", "/")
        path_obj = Path(normalized)

        if path_obj.is_absolute():
            try:
                return path_obj.resolve().relative_to(self.output_dir.resolve()).as_posix()
            except ValueError:
                return os.path.relpath(path_obj, self.output_dir).replace("\\", "/")

        trimmed = normalized
        while trimmed.startswith("../"):
            candidate = self.output_dir / trimmed[3:]
            if candidate.exists():
                trimmed = trimmed[3:]
                break
            trimmed = trimmed[3:]

        return Path(trimmed).as_posix()

    def _normalize_loaded_labels(self, labels: Dict) -> None:
        """Repair older stored paths so exports and training keep working."""
        changed = False

        for label_record in labels.get("labels", {}).values():
            for field in ("image_path", "crop_path"):
                existing_value = label_record.get(field)
                normalized_value = self._normalize_relative_path(existing_value)
                if existing_value != normalized_value:
                    label_record[field] = normalized_value
                    changed = True

        if changed:
            labels["last_updated"] = datetime.now().isoformat()
            try:
                with open(self.labels_file, 'w', encoding='utf-8') as f:
                    json.dump(labels, f, indent=2)
            except IOError as e:
                print(f"Error normalizing labels: {e}")

    def _resolve_stored_path(self, path_value: Optional[str]) -> Optional[Path]:
        """Resolve a stored relative path back to an absolute path inside output_dir."""
        normalized = self._normalize_relative_path(path_value)
        if not normalized:
            return None
        return self.output_dir / normalized

    def _get_export_category(self, detected_class: str) -> str:
        return "people" if detected_class == "person" else "animals"

    def _sanitize_label_name(self, assigned_label: str) -> str:
        sanitized = "".join(
            character for character in assigned_label.strip() if character not in '\\/:*?"<>|'
        ).strip()
        return sanitized or "unlabeled"
    
    def _save_labels(self):
        """Save labels to file"""
        self._labels["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                json.dump(self._labels, f, indent=2)
        except IOError as e:
            print(f"Error saving labels: {e}")
    
    def add_label(self, image_path: str, detection_index: int, assigned_label: str, 
                  detected_class: str, bbox: List[int], crop_path: Optional[str] = None,
                  status: str = "confirmed") -> str:
        """
        Add a label for a specific detection.
        
        Args:
            image_path: Path to source image
            detection_index: Index of detection in the image
            assigned_label: Human-assigned label (e.g., "Ron", "Dobby")
            detected_class: Detected class (e.g., "person", "dog")
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            crop_path: Path to cropped image if available
            status: Label status ("confirmed", "rejected", "pending")
            
        Returns:
            Unique label ID
        """
        label_id = str(uuid.uuid4())
        
        # Use relative path from output_dir for portability
        rel_image_path = self._normalize_relative_path(image_path)
        rel_crop_path = self._normalize_relative_path(crop_path)
        
        label_record = {
            "id": label_id,
            "image_path": rel_image_path,
            "detection_index": detection_index,
            "detected_class": detected_class,
            "assigned_label": assigned_label,
            "bbox": bbox,
            "crop_path": rel_crop_path,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        self._labels["labels"][label_id] = label_record
        self._save_labels()
        
        return label_id

    def rebuild_named_exports(self) -> str:
        """Export confirmed labels into name-based folders for browsing and training."""
        export_root = self.output_dir / "_sorted_by_name"

        if export_root.exists():
            shutil.rmtree(export_root)

        for label_record in self.get_all_labels(status="confirmed"):
            assigned_label = label_record.get("assigned_label", "").strip()
            if not assigned_label:
                continue

            category = self._get_export_category(label_record.get("detected_class", ""))
            safe_label = self._sanitize_label_name(assigned_label)
            label_root = export_root / category / safe_label
            images_dir = label_root / "images"
            crops_dir = label_root / "crops"
            images_dir.mkdir(parents=True, exist_ok=True)
            crops_dir.mkdir(parents=True, exist_ok=True)

            image_source = self._resolve_stored_path(label_record.get("image_path"))
            crop_source = self._resolve_stored_path(label_record.get("crop_path"))
            detection_index = label_record.get("detection_index", 0)

            if image_source and image_source.exists():
                image_target = images_dir / f"{image_source.stem}_det_{detection_index}{image_source.suffix}"
                shutil.copy2(image_source, image_target)

            if crop_source and crop_source.exists():
                crop_target = crops_dir / f"{crop_source.stem}_label_{safe_label}{crop_source.suffix}"
                shutil.copy2(crop_source, crop_target)

        return str(export_root)
    
    def get_label(self, label_id: str) -> Optional[Dict]:
        """Get a specific label by ID"""
        return self._labels["labels"].get(label_id)
    
    def get_all_labels(self, status: Optional[str] = None, 
                      detected_class: Optional[str] = None,
                      assigned_label: Optional[str] = None) -> List[Dict]:
        """
        Get all labels, optionally filtered by status, detected class, or assigned label.
        """
        labels = []
        
        for label_record in self._labels["labels"].values():
            # Apply filters
            if status and label_record.get("status") != status:
                continue
            if detected_class and label_record.get("detected_class") != detected_class:
                continue
            if assigned_label and label_record.get("assigned_label") != assigned_label:
                continue
            
            labels.append(label_record)
        
        # Sort by timestamp (newest first)
        labels.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return labels
    
    def get_labels_for_image(self, image_path: str) -> List[Dict]:
        """Get all labels for a specific image"""
        rel_image_path = self._normalize_relative_path(image_path)
        return [
            label for label in self._labels["labels"].values()
            if label.get("image_path") == rel_image_path
        ]
    
    def get_unique_assigned_labels(self) -> List[str]:
        """Get list of all unique assigned labels"""
        labels = set()
        for label_record in self._labels["labels"].values():
            if label_record.get("status") == "confirmed":
                labels.add(label_record.get("assigned_label", ""))
        return sorted(list(labels))
    
    def update_label(self, label_id: str, assigned_label: str = None, status: str = None) -> bool:
        """Update label fields"""
        if label_id in self._labels["labels"]:
            if assigned_label is not None:
                self._labels["labels"][label_id]["assigned_label"] = assigned_label
            if status is not None:
                self._labels["labels"][label_id]["status"] = status
            self._labels["labels"][label_id]["timestamp"] = datetime.now().isoformat()
            self._save_labels()
            return True
        return False

    def update_label_status(self, label_id: str, status: str) -> bool:
        """Update the status of a label"""
        if label_id in self._labels["labels"]:
            self._labels["labels"][label_id]["status"] = status
            self._labels["labels"][label_id]["timestamp"] = datetime.now().isoformat()
            self._save_labels()
            return True
        return False
    
    def delete_label(self, label_id: str) -> bool:
        """Delete a label"""
        if label_id in self._labels["labels"]:
            del self._labels["labels"][label_id]
            self._save_labels()
            return True
        return False
    
    def get_label_statistics(self) -> Dict:
        """Get statistics about labels"""
        stats = {
            "total_labels": len(self._labels["labels"]),
            "confirmed_labels": 0,
            "rejected_labels": 0,
            "pending_labels": 0,
            "people_labels": 0,
            "animal_labels": 0,
            "unique_assigned_labels": set()
        }
        
        for label_record in self._labels["labels"].values():
            status = label_record.get("status", "pending")
            detected_class = label_record.get("detected_class", "")
            assigned_label = label_record.get("assigned_label", "")
            
            if status == "confirmed":
                stats["confirmed_labels"] += 1
                if assigned_label:
                    stats["unique_assigned_labels"].add(assigned_label)
            elif status == "rejected":
                stats["rejected_labels"] += 1
            else:
                stats["pending_labels"] += 1
            
            if detected_class == "person":
                stats["people_labels"] += 1
            else:
                stats["animal_labels"] += 1
        
        stats["unique_assigned_labels"] = sorted(list(stats["unique_assigned_labels"]))
        return stats
    
    def export_for_training(self, status: str = "confirmed") -> Dict:
        """
        Export labels in a format suitable for training.
        Groups labels by assigned_label.
        """
        confirmed_labels = self.get_all_labels(status=status)
        
        training_data = {}
        for label_record in confirmed_labels:
            assigned_label = label_record.get("assigned_label", "")
            if assigned_label not in training_data:
                training_data[assigned_label] = []
            
            training_data[assigned_label].append({
                "image_path": label_record.get("image_path"),
                "crop_path": label_record.get("crop_path"),
                "bbox": label_record.get("bbox"),
                "detected_class": label_record.get("detected_class"),
                "label_id": label_record.get("id")
            })
        
        return training_data
