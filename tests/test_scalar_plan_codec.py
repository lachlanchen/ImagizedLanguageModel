import copy
import math
import unittest

import torch

from ilm.visual_lm.binary_plan_codec_learning import reconstruction_terms
from ilm.visual_lm.binary_plan_glyph_density import BinaryPlanGlyphDensity, pack_plan, unpack_plan, xor_correction
from ilm.visual_lm.scalar_plan_codec import ScalarPlanCodec, scalar_codec_forward
from ilm.visual_lm.spatial_plan_codec import SpatialPlanCodec


class ScalarPlanCodecTest(unittest.TestCase):
    def test_published_bound_levels_and_actual_straight_through_gradient(self):
        model = ScalarPlanCodec(channels=(8, 8, 8, 8))
        raw = torch.tensor([-100., -1., 0., 1., 100.], requires_grad=True)
        bounded = model.bound(raw)
        half = 3 * .999 / 2
        expected = (raw + math.tan(.5 / half)).tanh() * half - .5
        torch.testing.assert_close(bounded, expected, atol=0, rtol=0)
        self.assertEqual(bounded.round().tolist(), [-2., -1., 0., 1., 1.])
        ste = bounded.round().detach() + (bounded - bounded.detach())
        self.assertTrue(torch.equal(ste, bounded.round()))
        ste.sum().backward()
        derivative = half * (1 - (raw.detach() + math.tan(.5 / half)).tanh().square())
        torch.testing.assert_close(raw.grad, derivative, atol=1e-7, rtol=1e-6)

    def test_all_scalar_levels_and_every_bit_coordinate_roundtrip(self):
        model = ScalarPlanCodec(channels=(8, 8, 8, 8))
        bits = torch.eye(256, dtype=torch.uint8)
        values = model.bits_to_values(bits)
        self.assertTrue(torch.equal(bits, model.indices_to_bits((2 * values + 2).long())))
        self.assertTrue(torch.equal(bits, unpack_plan(pack_plan(bits))))
        indices = torch.arange(4).repeat(128, 1).T
        values = model.bits_to_values(model.indices_to_bits(indices))
        torch.testing.assert_close(values[:, 0], torch.tensor([-1., -.5, 0., .5]), atol=0, rtol=0)
        self.assertEqual(model.code_bits, 256)
        self.assertEqual(model.scalar_count, 128)

    def test_common_initial_cnn_real_updates_and_exact_reset(self):
        torch.manual_seed(202611195)
        state = torch.get_rng_state()
        old = SpatialPlanCodec(16)
        torch.set_rng_state(state)
        model = ScalarPlanCodec()
        shared = [n for n in old.state_dict() if n.startswith('decoder.') or (n.startswith('encoder.') and int(n.split('.')[1]) < 8)]
        self.assertTrue(all(torch.equal(old.state_dict()[n], model.state_dict()[n]) for n in shared))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1768393)
        images = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8)
        images[0] = 0
        for mode in ['continuous', 'binary']:
            model = ScalarPlanCodec(channels=(8, 8, 8, 8))
            before = copy.deepcopy(model.state_dict())
            out = scalar_codec_forward(model, images, mode)
            if mode == 'binary':
                self.assertTrue(torch.isin(out['plan_values'], torch.tensor([-1., -.5, 0., .5])).all())
                torch.testing.assert_close(out['image_logits'], model.decode_logits(out['plan_bits']), atol=0, rtol=0)
            reconstruction_terms(out['image_logits'], images)['loss'].mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertGreater(float(model.plan_projection.weight.grad.norm()), 0)
            torch.optim.AdamW(model.parameters(), lr=.0002).step()
            self.assertFalse(torch.equal(before['expand.weight'], model.expand.weight))
            model.load_state_dict(before)
            self.assertTrue(all(torch.equal(v, before[n]) for n, v in model.state_dict().items()))

    def test_joint_writer_generation_has_no_target_encoder_or_glyph_table(self):
        torch.manual_seed(202611205)
        codec = ScalarPlanCodec(channels=(8, 8, 8, 8))
        model = BinaryPlanGlyphDensity(codec, 7, width=16, layers=1, heads=4).eval()
        states = torch.randn(2, 7, requires_grad=True)
        images = torch.randint(0, 2, (2, 1, 32, 32), dtype=torch.uint8)
        score = model.training_log_bound(states, images)
        (-score['log_bound'].mean() / 1024).backward()
        self.assertGreater(float(states.grad.norm()), 0)
        def forbidden(_):
            raise RuntimeError('Target entered inference')
        codec.encode_plan = forbidden
        generated, trace = model.generate(states.detach())
        self.assertEqual(trace['plan_patterns'].shape, (2, 32))
        self.assertTrue(torch.equal(generated, xor_correction(trace['base'], trace['residual'])))


if __name__ == '__main__':
    unittest.main()
