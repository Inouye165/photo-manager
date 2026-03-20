"""
Detection metadata persistence for PhotoFinder labeling workflow.
Stores detection results with bounding boxes, confidence, and class information.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DetectionMetadata:
    """Manages detection metadata storage and retrieval"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metadata_file = self.output_dir / "_detections.json"
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                print(f"Warning: Could not load metadata from {self.metadata_file}")
        
        return {"images": {}, "last_updated": None}
    
    def _save_metadata(self):
        """Save metadata to file"""
        self._metadata["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except IOError as e:
            print(f"Error saving metadata: {e}")
    
    def add_detections(self, image_path: str, detections: List[Dict], debug_image_path: Optional[str] = None):
        """Add detection results for an image"""
        # Use relative path from output_dir for portability
        rel_path = os.path.relpath(image_path, self.output_dir)
        
        detection_record = {
            "image_path": rel_path,
            "absolute_path": image_path,
            "detections": detections,
            "debug_image_path": os.path.relpath(debug_image_path, self.output_dir) if debug_image_path else None,
            "timestamp": datetime.now().isoformat(),
            "processed": False
        }
        
        self._metadata["images"][rel_path] = detection_record
        self._save_metadata()
    
    def get_detections(self, image_path: Optional[str] = None) -> Dict:
        """Get detection metadata for specific image or all images"""
        if image_path:
            rel_path = os.path.relpath(image_path, self.output_dir)
            return self._metadata["images"].get(rel_path, {})
        return self._metadata["images"]
    
    def get_images_with_detections(self, classes: Optional[List[str]] = None) -> List[Dict]:
        """Get all images that have detections, optionally filtered by class"""
        results = []
        
        for image_path, record in self._metadata["images"].items():
            detections = record.get("detections", [])
            
            if not detections:
                continue
                
            # Filter by class if specified
            if classes:
                has_target_class = any(det.get("class_name") in classes for det in detections)
                if not has_target_class:
                    continue
            
            results.append(record)
        
        return results
    
    def get_unprocessed_detections(self, classes: Optional[List[str]] = None) -> List[Dict]:
        """Get unprocessed detections, optionally filtered by class"""
        results = []
        
        for image_path, record in self._metadata["images"].items():
            if record.get("processed", False):
                continue
                
            detections = record.get("detections", [])
            
            if not detections:
                continue
                
            # Filter by class if specified
            if classes:
                has_target_class = any(det.get("class_name") in classes for det in detections)
                if not has_target_class:
                    continue
            
            results.append(record)
        
        return results
    
    def mark_processed(self, image_path: str):
        """Mark an image as processed"""
        rel_path = os.path.relpath(image_path, self.output_dir)
        if rel_path in self._metadata["images"]:
            self._metadata["images"][rel_path]["processed"] = True
            self._save_metadata()
