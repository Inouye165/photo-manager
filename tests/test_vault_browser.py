"""Tests for the immutable source-vault browser payload."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.detection_metadata import DetectionMetadata
from src.runtime_paths import compute_sha256
from web_ui import build_vault_browser_payload


class TestVaultBrowserPayload(unittest.TestCase):
    """Verify the vault browser exposes hashes, metadata, and debug-image links."""

    def test_browser_payload_includes_hash_metadata_and_debug_preview(self):
        """A source original should surface with its hash and linked detection artifacts."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / 'source_originals'
            output_dir = root / 'output'
            source_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / '_library').mkdir(parents=True, exist_ok=True)
            (output_dir / '_debug_boxes').mkdir(parents=True, exist_ok=True)

            source_image = source_dir / 'alpha.jpg'
            canonical_image = output_dir / '_library' / 'alpha.jpg'
            debug_image = output_dir / '_debug_boxes' / 'alpha_debug.jpg'

            Image.new('RGB', (40, 30), color=(10, 20, 30)).save(source_image)
            Image.new('RGB', (40, 30), color=(10, 20, 30)).save(canonical_image)
            Image.new('RGB', (40, 30), color=(255, 0, 0)).save(debug_image)

            file_hash = compute_sha256(source_image)
            DetectionMetadata(str(output_dir)).add_detections(
                str(canonical_image),
                [{"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]}],
                str(debug_image),
                extra_metadata={
                    'file_hash': file_hash,
                    'captured_at': '2026-03-20T10:11:12',
                    'original_filename': 'IMG_0001.JPG',
                },
            )

            payload = build_vault_browser_payload(source_dir, output_dir)

            self.assertEqual(payload['sourceCount'], 1)
            self.assertEqual(payload['uniqueHashCount'], 1)
            self.assertEqual(payload['duplicateHashGroups'], 0)
            self.assertEqual(payload['folders'][0]['path'], 'root')

            item = payload['folders'][0]['items'][0]
            self.assertEqual(item['filename'], 'alpha.jpg')
            self.assertEqual(item['sha256'], file_hash)
            self.assertEqual(item['capturedAt'], '2026-03-20T10:11:12')
            self.assertEqual(item['originalFilename'], 'IMG_0001.JPG')
            self.assertEqual(item['detectionCount'], 1)
            self.assertEqual(item['detectedClasses'], ['person'])
            self.assertTrue(item['previewUrl'].startswith('/working_dir/alpha.jpg'))
            self.assertTrue(item['debugUrl'].startswith('/image/_debug_boxes/alpha_debug.jpg'))


if __name__ == '__main__':
    unittest.main()