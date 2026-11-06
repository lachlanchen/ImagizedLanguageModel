"""Pixel preservation and persistent layout contracts for the auxiliary reader."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.audit_native_neural_transcripts_v80 import view


class PixelLayoutTests(unittest.TestCase):
    def test_persisted_layout_reconstructs_every_source_pixel(self):
        rng = np.random.default_rng(20268041)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for width in (1, 64, 1024, 1025, 8192):
                with self.subTest(width=width):
                    pixels = np.where(rng.random((32, width)) < .15, 0, 255).astype(np.uint8)
                    # Exercise both a white break and a forced break through ink.
                    if width > 1024:
                        pixels[:, 900] = 255
                    key, layout = view(pixels, folder)
                    persisted = json.loads(json.dumps(layout))
                    self.assertEqual(persisted, layout)
                    actual = np.asarray(Image.open(folder / (key + ".png")))
                    page = actual[16:-16:4, 16:-16:4]
                    rebuilt = np.concatenate([
                        page[i*36:i*36+32, :end-start]
                        for i, (start, end) in enumerate(persisted["spans"])
                    ], axis=1)
                    np.testing.assert_array_equal(rebuilt, pixels)
                    self.assertLessEqual(actual.shape[1], 4128)
                    self.assertTrue(np.all(actual[:16] == 255))
                    self.assertTrue(np.all(actual[-16:] == 255))
                    self.assertEqual(view(pixels, folder), (key, layout))

    def test_invalid_physical_pixels_are_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for pixels in (np.zeros((31, 64), np.uint8),
                           np.zeros((32, 0), np.uint8),
                           np.ones((32, 64), np.uint8),
                           np.zeros((32, 64), np.float32)):
                with self.assertRaises(ValueError):
                    view(pixels, folder)
            self.assertEqual(list(folder.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
