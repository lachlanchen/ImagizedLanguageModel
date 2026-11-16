"""Regression for the observed unordered Ollama parameter metadata."""
import unittest

from scripts.calibrate_isolated_glyph_judge_gpu0 import canonical_parameters


class TeacherParameterIdentityTest(unittest.TestCase):
    def test_field_order_is_not_a_model_change_but_values_are(self):
        first = 'presence_penalty               1.5\ntemperature                    1\ntop_k                          20\ntop_p                          0.95'
        second = 'temperature                    1\ntop_k                          20\ntop_p                          0.95\npresence_penalty               1.5'
        self.assertEqual(canonical_parameters(first), canonical_parameters(second))
        self.assertNotEqual(canonical_parameters(first), canonical_parameters(second.replace('1.5', '1.6')))


if __name__ == '__main__':
    unittest.main()
