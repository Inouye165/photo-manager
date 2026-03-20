import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.ingest_manager import IngestConfig, IngestManager, image_bytes_to_upload
from src.mirror_manager import MirrorConfig, MirrorManager


class StubImageProcessor:
    def detect_subjects(self, _image_path):
        return {"has_person": True, "has_animal": False}

    def annotate_detections(self, _image_path, output_path, _confidence_threshold=0.10):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b'debug')
        return {"saved": True}

    def get_detailed_detections(self, _image_path, _confidence_threshold=0.25):
        return [{"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "bbox_area": 100}]

    def generate_crop(self, _image_path, _bbox, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 32), color=(100, 50, 25)).save(output_path)
        return True


class TestIngestManager(unittest.TestCase):
    def test_ingest_upload_sets_read_only_and_detects_duplicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / 'source_originals'
            processing_root = root / 'processing'
            mirror = MirrorManager(MirrorConfig(source_originals=source_dir, mirror_workspace=root / 'mirror_workspace'))
            manager = IngestManager(
                IngestConfig(source_originals=source_dir, processing_root=processing_root),
                mirror,
                image_processor=StubImageProcessor(),
            )

            upload = image_bytes_to_upload('test.jpg', Image.new('RGB', (64, 64), color=(255, 0, 0)))
            result_one = manager.ingest_upload(upload)

            self.assertFalse(result_one.duplicate)
            self.assertTrue(result_one.source_path.exists())
            self.assertTrue(result_one.canonical_path.exists())
            self.assertEqual(result_one.detection_count, 1)
            self.assertFalse(bool(result_one.source_path.stat().st_mode & 0o200))

            upload_two = image_bytes_to_upload('copy.jpg', Image.new('RGB', (64, 64), color=(255, 0, 0)))
            result_two = manager.ingest_upload(upload_two)

            self.assertTrue(result_two.duplicate)
            self.assertEqual(result_one.sha256, result_two.sha256)
            self.assertEqual(result_one.source_path, result_two.source_path)