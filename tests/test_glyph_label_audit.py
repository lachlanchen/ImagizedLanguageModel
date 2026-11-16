"""Coverage and nonidentifiability controls, not model-accuracy fixtures."""
import unittest

import numpy as np

from ilm.training_only.glyph_label_audit import add_pairs, inventory_summary, verify_record


class FixtureRenderer:
    def tile(self, char):
        return bytes([0 if char.isspace() or char == '\u200b' else 1]) + bytes(127)


class GlyphLabelAuditTest(unittest.TestCase):
    def test_complete_pixels_and_boundaries_checked(self):
        renderer = FixtureRenderer()
        packed = np.frombuffer(renderer.tile('中') + renderer.tile(' '), dtype=np.uint8).reshape(2, 128)
        self.assertEqual(verify_record(packed, '中 ', 1, 1, renderer), 0)
        changed = packed.copy(); changed[0, 3] = 128
        self.assertEqual(verify_record(changed, '中 ', 1, 1, renderer), 1)
        with self.assertRaises(ValueError): verify_record(packed, '中', 1, 1, renderer)
        with self.assertRaises(ValueError): verify_record(packed, '中 ', 1, 0, renderer)

    def test_aliases_never_overwritten_and_all_occurrences_counted(self):
        renderer = FixtureRenderer(); text = 'AAΑ\n '
        packed = np.frombuffer(b''.join(renderer.tile(c) for c in text), dtype=np.uint8).reshape(-1, 128)
        bank = {}; add_pairs(bank, packed, text); report = inventory_summary(bank)
        self.assertEqual(report['glyph_occurrences'], 5)
        self.assertEqual(report['physical_masks'], 2)
        self.assertFalse(report['context_free_nonwhite_labels_unique'])
        alias, = report['nonwhite_aliases']
        self.assertEqual({v['codepoint']:v['occurrences'] for v in alias['labels']}, {ord('A'):2, ord('Α'):1})
        self.assertTrue(report['physically_white_labels_only_whitespace'])

    def test_invisible_nonspace_loss_is_separate(self):
        renderer = FixtureRenderer(); text = '\u200b '
        packed = np.frombuffer(b''.join(renderer.tile(c) for c in text), dtype=np.uint8).reshape(-1,128)
        bank = {}; add_pairs(bank, packed, text); report = inventory_summary(bank)
        self.assertTrue(report['context_free_nonwhite_labels_unique'])
        self.assertFalse(report['physically_white_labels_only_whitespace'])
        self.assertEqual(report['white_nonspace_labels'][0]['codepoint'], 0x200b)


if __name__ == '__main__': unittest.main()
