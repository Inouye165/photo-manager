"""Regression tests for the immutable ingest pipeline."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.ingest_manager import IngestConfig, IngestManager, image_bytes_to_upload
from src.mirror_manager import MirrorConfig, MirrorManager


class StubImageProcessor:
    """Minimal detector stub used to exercise ingest behavior in isolation."""

    def detect_subjects(self, _image_path):
        """Return a stable detection summary for the uploaded image."""

        return {"has_person": True, "has_animal": False}

    def annotate_detections(self, _image_path, output_path, _confidence_threshold=0.10):
        """Write a fake debug artifact and report success."""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b'debug')
        return {"saved": True}

    def get_detailed_detections(self, _image_path, _confidence_threshold=0.25):
        """Return one deterministic person detection for test assertions."""

        return [{"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10], "bbox_area": 100}]

    def generate_crop(self, _image_path, _bbox, output_path):
        """Write a deterministic crop image for downstream ingest checks."""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (32, 32), color=(100, 50, 25)).save(output_path)
        return True


class CountingImageProcessor(StubImageProcessor):
    """Track how many times the ingest pipeline reprocesses the same asset."""

    def __init__(self):
        self.detect_calls = 0

    def detect_subjects(self, image_path):
        self.detect_calls += 1
        return super().detect_subjects(image_path)


class TestIngestManager(unittest.TestCase):
    """Verify ingest persistence, duplicate detection, and file protections."""

    def test_ingest_upload_sets_read_only_and_detects_duplicate(self):
        """The second upload of identical bytes should reuse the first immutable source file."""

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
            self.assertEqual(result_two.duplicate_source_filename, result_one.clean_name)
            self.assertEqual(result_two.duplicate_source_relative_path, result_one.source_relative_path)

    def test_ingest_uploads_processes_multiple_files(self):
        """Bulk ingest should run each selected file through the same single-file pipeline."""

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

            uploads = [
                image_bytes_to_upload('alpha.jpg', Image.new('RGB', (64, 64), color=(255, 0, 0))),
                image_bytes_to_upload('beta.jpg', Image.new('RGB', (64, 64), color=(0, 255, 0))),
            ]

            results = manager.ingest_uploads(uploads)

            self.assertEqual(len(results), 2)
            self.assertEqual(sum(result.detection_count for result in results), 2)
            self.assertTrue(all(result.source_path.exists() for result in results))
            self.assertTrue(all(result.canonical_path.exists() for result in results))

    def test_duplicate_upload_reuses_existing_processing_artifacts(self):
        """Duplicate bytes should be recognized by hash and should not rerun detection."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / 'source_originals'
            processing_root = root / 'processing'
            image_processor = CountingImageProcessor()
            mirror = MirrorManager(MirrorConfig(source_originals=source_dir, mirror_workspace=root / 'mirror_workspace'))
            manager = IngestManager(
                IngestConfig(source_originals=source_dir, processing_root=processing_root),
                mirror,
                image_processor=image_processor,
            )

            first_upload = image_bytes_to_upload('alpha.jpg', Image.new('RGB', (64, 64), color=(10, 20, 30)))
            second_upload = image_bytes_to_upload('renamed-copy.jpg', Image.new('RGB', (64, 64), color=(10, 20, 30)))

            result_one = manager.ingest_upload(first_upload)
            result_two = manager.ingest_upload(second_upload)

            self.assertFalse(result_one.duplicate)
            self.assertTrue(result_two.duplicate)
            self.assertEqual(result_one.sha256, result_two.sha256)
            self.assertEqual(result_two.duplicate_source_filename, result_one.clean_name)
            self.assertEqual(image_processor.detect_calls, 1)
