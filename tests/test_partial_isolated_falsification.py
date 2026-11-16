"""Missing observations cannot become failures, successes or repaired readings."""
import unittest
from scripts.falsify_partial_isolated_judge import reference_failure


class OneSidedFailureTest(unittest.TestCase):
    def test_missing_stays_unknown(self):
        self.assertEqual(reference_failure(None, '言'), 'unknown')

    def test_observed_abstention_or_wrong_reading_is_a_coverage_violation(self):
        self.assertEqual(reference_failure(dict(accepted=False, text=None), '言'), 'observed_abstention')
        self.assertEqual(reference_failure(dict(accepted=True, text='中'), '言'), 'observed_wrong_character')
        self.assertIsNone(reference_failure(dict(accepted=True, text='言'), '言'))


if __name__ == '__main__':
    unittest.main()
