import unittest

import torch

from ilm.visual_lm.residual_motor_data import decode_motor_patterns


class ResidualMotorDataTest(unittest.TestCase):
    def test_every_serialized_byte_value_in_every_plan_position(self):
        patterns = torch.arange(256).to(torch.uint8)[:, None].expand(256, 32).contiguous()
        actual = decode_motor_patterns(patterns)
        expected = ((torch.arange(256)[:, None] >> torch.arange(8)) & 1).to(torch.uint8).repeat(1, 32)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.shape, (256, 256))


if __name__ == '__main__':
    unittest.main()
