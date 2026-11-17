import unittest

import torch

from ilm.visual_lm.residual_motor_metrics import residual_motor_metrics


class ResidualMotorMetricsTest(unittest.TestCase):
    def test_generated_work_is_not_remaining_error_complexity(self):
        target = torch.zeros((2, 1, 32, 32), dtype=torch.uint8)
        target[1, 0, 4, 5] = 1
        partition = dict(held=torch.tensor([False, True]), group=torch.arange(2), group_mask_sha256=['a', 'b'])
        result = residual_motor_metrics(target, target, partition, torch.ones(2, dtype=torch.long), torch.tensor([1, 17]))
        self.assertEqual(result['all']['pixels']['f1'], 1.)
        self.assertEqual(result['all']['mean_remaining_error_tree_events'], 1.)
        self.assertEqual(result['all']['mean_plan_and_generated_events'], 41.)
        self.assertEqual(result['train']['mean_plan_and_generated_events'], 33.)
        self.assertEqual(result['held']['mean_plan_and_generated_events'], 49.)
        self.assertNotIn('mean_correction_tree_events', result['all'])


if __name__ == '__main__':
    unittest.main()
