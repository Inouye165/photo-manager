"""Tests for semantic-search helpers exposed by the Flask UI layer."""

import tempfile
import unittest

from src.label_manager import LabelManager
from web_ui import infer_semantic_search_class_name


class TestSemanticSearchClassInference(unittest.TestCase):
    """Verify label text resolves to a strict detected-class filter when possible."""

    def test_exact_label_name_resolves_to_single_detected_class(self):
        """One known name with one class should infer that class for search."""

        with tempfile.TemporaryDirectory() as temp_dir:
            label_manager = LabelManager(temp_dir)
            label_manager.add_label(
                image_path="working_dir/dobby1.jpg",
                detection_index=0,
                assigned_label="Dobby",
                detected_class="dog",
                bbox=[0, 0, 10, 10],
                crop_path="_training_crops/animals/dobby1.jpg",
                status="confirmed",
            )
            label_manager.add_label(
                image_path="working_dir/dobby2.jpg",
                detection_index=0,
                assigned_label="Dobby",
                detected_class="dog",
                bbox=[0, 0, 10, 10],
                crop_path="_training_crops/animals/dobby2.jpg",
                status="confirmed",
            )

            self.assertEqual(infer_semantic_search_class_name("Dobby", label_manager), "dog")
