import logging
import os
from typing import Tuple, List, Dict, Optional

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

try:
    import clip
    import torch

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    clip = None
    torch = None

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

pillow_heif.register_heif_opener()

COARSE_ANIMAL_CLASSES = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
REFINED_ANIMAL_CLASSES = {"bison", "bison calf", "buffalo", "ox", "yak", "goat", "deer", "elk", "moose", "calf"}
ALL_ANIMAL_CLASSES = COARSE_ANIMAL_CLASSES | REFINED_ANIMAL_CLASSES
UNGULATE_RELABEL_SOURCE_CLASSES = {"sheep", "cow", "horse"}
ANIMAL_LABEL_CANONICAL_MAP = {"buffalo": "bison"}
UNGULATE_RELABEL_CANDIDATES = (
    "bison",
    "bison calf",
    "buffalo",
    "cow",
    "calf",
    "sheep",
    "goat",
    "ox",
    "yak",
    "deer",
    "elk",
    "moose",
    "horse",
)


class ImageProcessor:

    def __init__(self, laplacian_threshold: float = 100.0, entropy_threshold: float = 5.0):
        self.laplacian_threshold = laplacian_threshold
        self.entropy_threshold = entropy_threshold
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_text_features = None
        self.clip_device = "cpu"
        self._clip_attempted = False
        if HAS_YOLO:
            try:
                # Load YOLOv8 model for object detection
                self.yolo_model = YOLO("yolov8m.pt")  # Using medium model for better accuracy
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")

    def _ensure_clip_model(self) -> bool:
        """Load CLIP lazily for open-vocabulary animal relabeling."""
        if self.clip_model is not None and self.clip_preprocess is not None and self.clip_text_features is not None:
            return True
        if self._clip_attempted or not HAS_CLIP:
            return False

        self._clip_attempted = True
        try:
            self.clip_device = "cuda" if torch and torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.clip_device)
            prompts = [f"a photo of a {label}" for label in UNGULATE_RELABEL_CANDIDATES]
            tokenized_prompts = clip.tokenize(prompts).to(self.clip_device)
            with torch.no_grad():
                self.clip_text_features = self.clip_model.encode_text(tokenized_prompts)
                self.clip_text_features /= self.clip_text_features.norm(dim=-1, keepdim=True)
            return True
        except Exception as e:
            logging.warning(f"Failed to load CLIP animal relabeler: {e}")
            self.clip_model = None
            self.clip_preprocess = None
            self.clip_text_features = None
            return False

    def _crop_for_relabel(self, image: np.ndarray, bbox: List[int], padding_ratio: float = 0.08) -> Optional[Image.Image]:
        """Extract a lightly padded PIL crop for zero-shot animal relabeling."""
        x1, y1, x2, y2 = bbox
        img_h, img_w = image.shape[:2]
        pad_x = int((x2 - x1) * padding_ratio)
        pad_y = int((y2 - y1) * padding_ratio)
        left = max(0, x1 - pad_x)
        top = max(0, y1 - pad_y)
        right = min(img_w, x2 + pad_x)
        bottom = min(img_h, y2 + pad_y)

        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def _refine_animal_label(self, image: np.ndarray, bbox: List[int], coarse_label: str) -> str:
        """Relabel sheep/cow/horse-like detections using CLIP zero-shot classification."""
        normalized_label = str(coarse_label or "").strip().lower()
        if normalized_label not in UNGULATE_RELABEL_SOURCE_CLASSES:
            return normalized_label
        if not self._ensure_clip_model():
            return normalized_label

        crop_image = self._crop_for_relabel(image, bbox)
        if crop_image is None:
            return normalized_label

        try:
            image_tensor = self.clip_preprocess(crop_image).unsqueeze(0).to(self.clip_device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                probabilities = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)[0]

            best_index = int(probabilities.argmax().item())
            best_label = UNGULATE_RELABEL_CANDIDATES[best_index]
            best_confidence = float(probabilities[best_index].item())
            if best_confidence >= 0.45:
                return ANIMAL_LABEL_CANONICAL_MAP.get(best_label, best_label)
        except Exception as e:
            logging.warning(f"Animal relabel failed for {coarse_label}: {e}")

        return normalized_label

    def calculate_laplacian_variance(self, image: np.ndarray) -> float:
        """
        Calculates the variance of the Laplacian.
        Often used to detect blur, but also helps identify sharp synthetic lines vs natural textures.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_color_entropy(self, image: np.ndarray) -> float:
        """
        Calculates the Shannon entropy of the image.
        Photos tend to have higher entropy than simple digital graphics due to noise and texture.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()

        # Calculate entropy, avoiding log(0)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        return entropy

    def _load_image(self, image_path: str) -> np.ndarray:
        ext = image_path.lower()
        if ext.endswith(".heic") or ext.endswith(".heif"):
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            image = np.array(pil_img)
            # OpenCV uses BGR natively, so swap RGB from PIL to BGR
            if len(image.shape) == 3:
                image = image[:, :, ::-1].copy()
        else:
            image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        return image

    def is_true_photo(self, image_path: str) -> Tuple[bool, dict]:
        """
        Determines whether the given image is a true photo or a digital graphic.
        Returns a tuple of (is_photo, metrics).
        """
        image = self._load_image(image_path)

        laplacian_var = self.calculate_laplacian_variance(image)
        entropy = self.calculate_color_entropy(image)

        # Heuristic: True photos usually have high entropy (> 6.0).
        # Screenshots/graphics often have lower entropy due to flat colors,
        # but extremely high laplacian variance due to sharp digital text/lines.
        is_photo = entropy > 6.0

        metrics = {"laplacian_variance": laplacian_var, "color_entropy": entropy}
        return is_photo, metrics

    def detect_subjects(self, image_path: str) -> dict:
        """
        Detects if the image contains people or animals using YOLO object detection.
        Returns a dict: {"has_person": bool, "has_animal": bool}
        """
        results_dict = {"has_person": False, "has_animal": False}

        if not self.yolo_model:
            logging.warning("YOLO model not loaded. Skipping subject detection.")
            return results_dict

        try:
            # We use _load_image to correctly handle HEIC images and convert to BGR for YOLO
            image = self._load_image(image_path)

            # YOLO predicts on BGR numpy arrays (cv2 format) natively without issues
            results = self.yolo_model(image, verbose=False, conf=0.25)

            for r in results:
                if r.boxes:
                    for cls, conf in zip(r.boxes.cls, r.boxes.conf):
                        class_id = int(cls.item())
                        class_name = self.yolo_model.names[class_id]
                        confidence = conf.item()

                        if class_name == "person" and confidence > 0.5:
                            results_dict["has_person"] = True
                        elif class_name in COARSE_ANIMAL_CLASSES and confidence > 0.4:
                            results_dict["has_animal"] = True
            return results_dict
        except Exception as e:
            logging.error(f"Error during YOLO inference on {image_path}: {e}")
            return results_dict

    def get_detailed_detections(self, image_path: str, confidence_threshold: float = 0.25) -> List[Dict]:
        """
        Get detailed detection information including bounding boxes, confidence, and class names.
        Returns a list of detection dictionaries.
        """
        if not self.yolo_model:
            logging.warning("YOLO model not loaded. Skipping detailed detection.")
            return []

        try:
            image = self._load_image(image_path)
            results = self.yolo_model(image, verbose=False, conf=confidence_threshold)
            
            detections = []
            
            for r in results:
                if r.boxes:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        class_id = int(cls.item())
                        class_name = self.yolo_model.names[class_id]
                        confidence = conf.item()
                        
                        x1, y1, x2, y2 = map(int, box[:4])
                        
                        # Only include people and animals for labeling workflow
                        if class_name == "person" or class_name in COARSE_ANIMAL_CLASSES:
                            refined_class_name = (
                                self._refine_animal_label(image, [x1, y1, x2, y2], class_name)
                                if class_name in COARSE_ANIMAL_CLASSES
                                else class_name
                            )
                            detection = {
                                "class_name": refined_class_name,
                                "class_id": class_id,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                                "bbox_area": (x2 - x1) * (y2 - y1)
                            }
                            detections.append(detection)
            
            return detections
        except Exception as e:
            logging.error(f"Error during detailed detection on {image_path}: {e}")
            return []

    def generate_crop(self, image_path: str, bbox: List[int], output_path: str) -> bool:
        """
        Generate a crop from the image using the provided bounding box.
        bbox should be [x1, y1, x2, y2]
        """
        try:
            image = self._load_image(image_path)
            x1, y1, x2, y2 = bbox
            
            # Generous padding so crops include the full head/face for identification
            img_h, img_w = image.shape[:2]
            box_w = x2 - x1
            box_h = y2 - y1
            pad_x = int(box_w * 0.5)
            pad_top = int(box_h * 0.75)   # extra room above for faces
            pad_bottom = int(box_h * 0.3)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_top)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_bottom)
            
            # Crop the image
            crop = image[y1:y2, x1:x2]
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save crop as JPEG for consistency
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(crop_rgb)
            pil_image.save(output_path, 'JPEG', quality=85)
            
            return True
        except Exception as e:
            logging.error(f"Error generating crop for {image_path}: {e}")
            return False

    def annotate_detections(self, image_path: str, output_path: str, confidence_threshold: float = 0.10) -> dict:
        """Draws bounding boxes around detected objects for debugging.
        Blue for people, Red for animals, Black for others.
        Saves the annotated image to output_path.
        """
        results_dict = {
            "saved": False,
            "output_path": output_path,
            "person_count": 0,
            "animal_count": 0,
            "other_count": 0,
        }

        if not self.yolo_model:
            logging.warning("YOLO model not loaded. Skipping annotation.")
            return results_dict

        try:
            image = self._load_image(image_path)
            annotated_image = image.copy()

            results = self.yolo_model(image, verbose=False, conf=confidence_threshold)
            has_detections = False

            for r in results:
                if r.boxes:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        has_detections = True
                        class_id = int(cls.item())
                        class_name = self.yolo_model.names[class_id]
                        confidence = conf.item()

                        x1, y1, x2, y2 = map(int, box[:4])
                        display_class_name = (
                            self._refine_animal_label(image, [x1, y1, x2, y2], class_name)
                            if class_name in COARSE_ANIMAL_CLASSES
                            else class_name
                        )

                        if class_name == "person":
                            color = (255, 0, 0)  # Blue in BGR
                            results_dict["person_count"] += 1
                        elif class_name in COARSE_ANIMAL_CLASSES:
                            color = (0, 0, 255)  # Red in BGR
                            results_dict["animal_count"] += 1
                        else:
                            color = (0, 0, 0)  # Black in BGR
                            results_dict["other_count"] += 1
                        # Draw rectangle
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                        # Draw label
                        label = f"{display_class_name} {confidence:.2f}"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                        cv2.putText(
                            annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                        )

            if has_detections:
                import os

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, annotated_image)
                results_dict["saved"] = True

            return results_dict
        except Exception as e:
            logging.error(f"Error during YOLO annotation on {image_path}: {e}")
            return results_dict
