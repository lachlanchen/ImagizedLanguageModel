"""Geometric/sequence contracts; these do not qualify a language model."""
import unittest
import numpy as np

from ilm.training_only.visual_content_verifier import SelectiveGlyphVerifier, packed_images


def cells(bits):
    x = np.zeros((len(bits), 1, 32, 32), dtype=np.uint8)
    for i, active in enumerate(bits):
        x[i].reshape(-1)[active] = 1
    return x


class SelectiveVerifierTests(unittest.TestCase):
    def setUp(self):
        self.clean = cells([[], list(range(8)), list(range(16, 24))])
        self.v = SelectiveGlyphVerifier(packed_images(self.clean))

    def test_exact_clean_membership_including_space(self):
        result = self.v.decode(self.clean)
        self.assertTrue(result['accepted'].all())
        np.testing.assert_array_equal(result['identities'], [0, 1, 2])
        self.assertTrue(np.all(self.v.minimum_distance == 8))

    def test_exhaustive_disjoint_balls_on_actual_binary_cells(self):
        # Enumerate every possible 8-bit corruption of the first source form.
        x = self.clean[1:2].repeat(256, axis=0)
        for mask in range(256):
            for bit in range(8):
                if (mask >> bit) & 1:
                    x[mask].reshape(-1)[bit] ^= 1
        result = self.v.decode(x)
        for mask, accepted, identity in zip(range(256), result['accepted'], result['identities']):
            errors = mask.bit_count()
            if errors <= 3:
                self.assertTrue(accepted); self.assertEqual(identity, 1)
            elif errors == 4:
                self.assertFalse(accepted); self.assertEqual(identity, -1)
            else:
                # This lies in the blank prototype's ball; bounded corruption
                # of source 1 no longer holds. Geometry alone is not semantics.
                self.assertTrue(accepted); self.assertEqual(identity, 0)

    def test_duplicates_order_and_spaces_are_not_ignored(self):
        reference = self.clean[[1, 0, 2]]
        self.assertEqual(self.v.score(reference, reference, stopped=True)['reward'], 1.)
        for wrong in [reference[::-1], reference[[0, 2]], reference[[0, 1, 2, 2]]]:
            score = self.v.score(wrong, reference, stopped=True)
            self.assertFalse(score['complete_content']); self.assertLess(score['reward'], .5)

    def test_no_reference_driven_identity(self):
        ink = self.clean[[1]]
        before = self.v.decode(ink)['identities'].copy()
        self.assertEqual(self.v.score(ink, self.clean[[2]], stopped=True)['reward'], 0.)
        np.testing.assert_array_equal(before, self.v.decode(ink)['identities'])

    def test_unseen_reference_abstains_without_inventory_extension(self):
        before = self.v.prototypes.copy()
        score = self.v.score(self.clean[[1]], cells([[40, 41]]), stopped=True)
        self.assertFalse(score['reference_covered']); self.assertEqual(score['reward'], 0.)
        np.testing.assert_array_equal(before, self.v.prototypes)

    def test_blank_and_cap_do_not_reward(self):
        ref = self.clean[[1]]
        self.assertEqual(self.v.score(ref, ref, stopped=False)['reward'], 0.)
        self.assertEqual(self.v.score(self.clean[[0]], ref, stopped=True)['reward'], 0.)

    def test_duplicates_and_invalid_pixels_rejected(self):
        with self.assertRaises(ValueError):
            SelectiveGlyphVerifier(packed_images(self.clean[[0, 0]]))
        with self.assertRaises(ValueError):
            self.v.decode(self.clean.astype(float) + .1)


if __name__ == '__main__':
    unittest.main()
