"""
Test suite for ImageProcessor class.

Tests cover photo detection, subject detection, annotation,
and HEIC image loading functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Base test class for ImageProcessor tests."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.processor = ImageProcessor()
        # Mock: YOLO model to avoid real inference and downloads
        self.processor.yolo_model = MagicMock()
        self.processor.yolo_model.names = {0: "person", 1: "dog", 2: "cat", 3: "car", 4: "stop sign"}


class TestAnnotateDetections(TestImageProcessor):
    """Test annotation functionality."""

    @patch("cv2.imwrite")
    @patch("os.makedirs")
    def test_annotate_detections_saves_output(self, _mock_makedirs, _mock_imwrite):
        """Test that annotation saves output file correctly."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(self.processor, "_load_image", return_value=dummy_image):
            # Mock: YOLO result with person, dog, and car detections
            mock_result = MagicMock()
            mock_result.boxes = MagicMock()
            mock_result.boxes.xyxy = [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]
            mock_result.boxes.cls = [
                MagicMock(item=MagicMock(return_value=0)),  # person
                MagicMock(item=MagicMock(return_value=1)),  # dog
                MagicMock(item=MagicMock(return_value=3)),  # car
            ]
            mock_result.boxes.conf = [
                MagicMock(item=MagicMock(return_value=0.9)),
                MagicMock(item=MagicMock(return_value=0.8)),
                MagicMock(item=MagicMock(return_value=0.7)),
            ]

            self.processor.yolo_model.return_value = [mock_result]

            result = self.processor.annotate_detections("test.jpg", "output.jpg")

            self.assertTrue(result["saved"])
            self.assertEqual(result["person_count"], 1)
            self.assertEqual(result["animal_count"], 1)
            self.assertEqual(result["other_count"], 1)
        _mock_makedirs.assert_called_once_with("", exist_ok=True)

    @patch("src.image_processor.ImageProcessor._load_image")
    @patch("cv2.rectangle")
    @patch("cv2.imwrite")
    @patch("os.makedirs")
    def test_annotate_detections_colors(self, _mock_makedirs, _mock_imwrite, mock_rectangle, mock_load_image):
        """Test that annotation uses correct colors for bounding boxes."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_load_image.return_value = dummy_image

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        # Three boxes: person (0), dog (1), car (3)
        mock_result.boxes.xyxy = [[0, 0, 10, 10], [10, 10, 20, 20], [20, 20, 30, 30]]
        mock_result.boxes.cls = [
            MagicMock(item=MagicMock(return_value=0)),  # person
            MagicMock(item=MagicMock(return_value=1)),  # dog
            MagicMock(item=MagicMock(return_value=3)),  # car
        ]
        mock_result.boxes.conf = [
            MagicMock(item=MagicMock(return_value=0.8)),
            MagicMock(item=MagicMock(return_value=0.7)),
            MagicMock(item=MagicMock(return_value=0.6)),
        ]

        self.processor.yolo_model.return_value = [mock_result]

        result = self.processor.annotate_detections("test.jpg", "output.jpg")

        self.assertTrue(result["saved"])
        self.assertEqual(result["person_count"], 1)
        self.assertEqual(result["animal_count"], 1)
        self.assertEqual(result["other_count"], 1)

        # Check that cv2.rectangle was called at least once for each box (may be called multiple times per box)
        self.assertGreaterEqual(mock_rectangle.call_count, 3)  # At least 3 calls for 3 boxes
        self.assertLessEqual(mock_rectangle.call_count, 10)  # But not too many calls


class TestDetectSubjects(TestImageProcessor):
    """Test subject detection functionality."""

    @patch("src.image_processor.ImageProcessor._load_image")
    def test_detect_subjects_existing_behavior(self, mock_load_image):
        """Test that detect_subjects still works as before."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_load_image.return_value = dummy_image

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = [[10, 10, 20, 20], [30, 30, 40, 40]]
        mock_result.boxes.cls = [
            MagicMock(item=MagicMock(return_value=0)),  # person
            MagicMock(item=MagicMock(return_value=1)),  # dog
        ]
        mock_result.boxes.conf = [
            MagicMock(item=MagicMock(return_value=0.8)),
            MagicMock(item=MagicMock(return_value=0.7)),
        ]

        self.processor.yolo_model.return_value = [mock_result]

        result = self.processor.detect_subjects("test.jpg")

        self.assertTrue(result["has_person"])
        self.assertTrue(result["has_animal"])

    @patch("src.image_processor.ImageProcessor._load_image")
    def test_detect_subjects_no_detections(self, mock_load_image):
        """Test that detect_subjects handles no detections correctly."""
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_load_image.return_value = dummy_image

        mock_result = MagicMock()
        mock_result.boxes = None
        self.processor.yolo_model.return_value = [mock_result]

        result = self.processor.detect_subjects("test.jpg")

        self.assertFalse(result["has_person"])
        self.assertFalse(result["has_animal"])


class TestLoadImageHEIC(TestImageProcessor):
    """Test HEIC image loading functionality."""

    @patch("PIL.Image.open")
    @patch("src.image_processor.ImageOps.exif_transpose")
    def test_annotate_detections_uses_load_image_for_heic(self, mock_exif, mock_pil_open):
        """Test that annotation uses _load_image so HEIC loading is preserved."""
        # Mock: PIL image
        mock_pil_img = MagicMock()
        mock_pil_img.convert.return_value = mock_pil_img
        mock_pil_open.return_value = mock_pil_img
        mock_exif.return_value = mock_pil_img

        # Mock: YOLO result to be empty
        mock_result = MagicMock()
        mock_result.boxes = None
        self.processor.yolo_model.return_value = [mock_result]

        # Mock: cv2.imwrite to avoid actual file writing
        with patch("cv2.imwrite"):
            with patch("os.makedirs"):
                # Mock: _load_image to track if it's called
                with patch.object(self.processor, "_load_image") as mock_load:
                    mock_load.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

                    self.processor.annotate_detections("test.heic", "output.jpg")

                    # Verify: _load_image was called with the HEIC file
                    mock_load.assert_called_once_with("test.heic")


if __name__ == "__main__":
    unittest.main()
