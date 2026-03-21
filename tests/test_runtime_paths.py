"""Tests for runtime path normalization and legacy source-vault bootstrapping."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.runtime_paths import resolve_runtime_directories


class TestRuntimePaths(unittest.TestCase):
    """Verify legacy single-folder installs retain prior photos in a separate source vault."""

    def test_same_input_output_bootstraps_separate_source_vault(self):
        """A legacy single-folder startup should preserve root photos in a dedicated source vault."""

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / 'working_dir'
            working_dir.mkdir(parents=True, exist_ok=True)

            Image.new('RGB', (24, 24), color=(255, 0, 0)).save(working_dir / 'alpha.jpg')
            Image.new('RGB', (24, 24), color=(0, 255, 0)).save(working_dir / 'beta.jpg')

            runtime_dirs = resolve_runtime_directories(str(working_dir), str(working_dir))

            self.assertTrue(runtime_dirs.legacy_source_mode)
            self.assertNotEqual(runtime_dirs.source_dir, runtime_dirs.data_dir)
            self.assertTrue((runtime_dirs.source_dir / 'alpha.jpg').exists())
            self.assertTrue((runtime_dirs.source_dir / 'beta.jpg').exists())
            self.assertEqual(len(runtime_dirs.migrated_files), 2)
            self.assertEqual(runtime_dirs.hardened_files, 0)
            self.assertFalse(bool((runtime_dirs.source_dir / 'alpha.jpg').stat().st_mode & 0o200))
            self.assertFalse(bool((runtime_dirs.source_dir / 'beta.jpg').stat().st_mode & 0o200))

    def test_bootstrap_uses_file_hash_instead_of_filename(self):
        """Duplicate bytes with different names should only populate the source vault once."""

        with tempfile.TemporaryDirectory() as temp_dir:
            working_dir = Path(temp_dir) / 'working_dir'
            working_dir.mkdir(parents=True, exist_ok=True)

            duplicate = Image.new('RGB', (24, 24), color=(123, 45, 67))
            duplicate.save(working_dir / 'first.jpg')
            duplicate.save(working_dir / 'second.jpg')

            runtime_dirs = resolve_runtime_directories(str(working_dir), str(working_dir))
            source_images = list(runtime_dirs.source_dir.glob('*.jpg'))

            self.assertEqual(len(source_images), 1)
            self.assertEqual(set(runtime_dirs.duplicate_files), {'second.jpg'})

    def test_existing_source_file_is_rehardened_on_resolve(self):
        """Startup should relock mutable source originals instead of only warning about them."""

        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / 'source_originals'
            output_dir = Path(temp_dir) / 'output'
            source_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            image_path = source_dir / 'mutable.jpg'
            Image.new('RGB', (24, 24), color=(0, 0, 255)).save(image_path)
            image_path.chmod(0o644)

            runtime_dirs = resolve_runtime_directories(str(source_dir), str(output_dir))

            self.assertEqual(runtime_dirs.hardened_files, 1)
            self.assertFalse(bool(image_path.stat().st_mode & 0o200))


if __name__ == '__main__':
    unittest.main()