import copy
import unittest

import torch

from ilm.visual_lm.binary_plan_glyph_density import xor_correction
from ilm.visual_lm.residual_plan_motor import ResidualPlanMotor
from ilm.visual_lm.spatial_plan_codec import SpatialPlanCodec


class ResidualPlanMotorTest(unittest.TestCase):
    def test_initial_actual_writing_equals_base_without_target_encoder(self):
        torch.manual_seed(202611211)
        codec = SpatialPlanCodec(16, channels=(8, 8, 8, 8))
        motor = ResidualPlanMotor(codec, width=16, layers=1, heads=4).eval()
        bits = torch.randint(0, 2, (5, 256), dtype=torch.uint8)
        expected = codec.decode_plan(bits)
        def forbidden(_):
            raise RuntimeError('Future target entered generation')
        codec.encode_plan = forbidden
        actual, trace = motor.generate(bits)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(trace['residual'].any())
        self.assertEqual(len(trace['correction_trace']), 1)

    def test_pixel_evidence_is_present_and_likelihood_is_image_normalized(self):
        torch.manual_seed(202611212)
        motor = ResidualPlanMotor(SpatialPlanCodec(16, channels=(8, 8, 8, 8)), width=16, layers=1, heads=4)
        bits = torch.randint(0, 2, (2, 256), dtype=torch.uint8)
        base = motor.codec.decode_plan(bits)
        altered = base.clone()
        altered[:, :, 1, 2] ^= 1
        first, second = motor.features(bits, base), motor.features(bits, altered)
        self.assertEqual(int((first != second).sum()), 2)
        # The empty correction needs exactly the root decision: log sigmoid(2).
        logp = motor.cached_image_log_prob(bits, base, base)
        torch.testing.assert_close(logp, torch.full((2,), -torch.nn.functional.softplus(torch.tensor(-2.)).item()), atol=1e-6, rtol=0)
        other = motor.cached_image_log_prob(bits, base, altered)
        self.assertTrue(torch.isfinite(other).all())
        self.assertTrue((other < logp).all())

    def test_actual_refinement_update_resets_and_preserves_fixed_codec(self):
        torch.manual_seed(202611213)
        codec = SpatialPlanCodec(16, channels=(8, 8, 8, 8))
        motor = ResidualPlanMotor(codec, width=16, layers=1, heads=4).train()
        frozen = copy.deepcopy(codec.state_dict())
        initial = copy.deepcopy(motor.refinement.state_dict())
        target = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8)
        target[0] = 0
        bits, base = motor.posterior_plan(target)
        loss = -motor.cached_image_log_prob(bits, base, target).mean() / 1024
        loss.backward()
        self.assertTrue(all(p.grad is None for p in codec.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in motor.refinement.parameters()))
        for module in [motor.refinement.condition[1], motor.refinement.root_output, motor.refinement.mask_output]:
            self.assertGreater(float(module.weight.grad.norm()), 0)
        torch.optim.AdamW(motor.refinement.parameters(), lr=.0002).step()
        self.assertTrue(all(torch.equal(v, frozen[n]) for n, v in codec.state_dict().items()))
        self.assertFalse(torch.equal(initial['mask_output.weight'], motor.refinement.state_dict()['mask_output.weight']))
        motor.refinement.load_state_dict(initial)
        self.assertTrue(all(torch.equal(v, initial[n]) for n, v in motor.refinement.state_dict().items()))
        self.assertFalse(codec.training)


if __name__ == '__main__':
    unittest.main()
