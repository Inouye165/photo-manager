import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.mirror_manager import CropRequest, MirrorConfig, MirrorManager


class TestMirrorManager(unittest.TestCase):
    def test_sync_and_derivatives_stay_in_mirror(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source_originals"
            mirror_dir = root / "mirror_workspace"
            source_dir.mkdir()

            image_path = source_dir / "family" / "photo1.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (640, 480), color=(120, 50, 30)).save(image_path)

            manager = MirrorManager(MirrorConfig(source_originals=source_dir, mirror_workspace=mirror_dir))
            report = manager.sync_source_index()

            self.assertEqual(report.discovered, 1)
            self.assertTrue(report.manifest_path.exists())

            thumb_path = manager.materialize_derivative("family/photo1.jpg", variant="thumb")
            crop_path = manager.materialize_crop(
                CropRequest(relative_path="family/photo1.jpg", bbox=(10, 20, 210, 220), subject_type="people")
            )

            self.assertTrue(thumb_path.exists())
            self.assertTrue(crop_path.exists())
            self.assertTrue(str(thumb_path).startswith(str(mirror_dir)))
            self.assertTrue(str(crop_path).startswith(str(mirror_dir)))
            self.assertEqual(len(list(source_dir.rglob("*"))), 2)

    def test_source_and_mirror_must_be_different(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                MirrorConfig(source_originals=root, mirror_workspace=root)