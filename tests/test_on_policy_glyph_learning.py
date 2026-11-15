"""Implementation controls, not a trained language experiment."""
import itertools
import unittest

import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs, unpack_glyphs
from ilm.visual_lm.on_policy_glyph_learning import (
    SampledAnswer, backward_answer_group, image_answer_reward, physical_log_probability,
    rloo_loss, sample_physical_patterns, stopping_log_probability,
)


def image(*positions):
    value = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    for r, c in positions:
        value[0, 0, r, c] = 1
    return value


class AnswerRewardTests(unittest.TestCase):
    def setUp(self):
        self.a = image((3, 4), (3, 5), (4, 4))
        self.b = image((22, 10), (22, 11))
        self.c = image((15, 24), (16, 24))
        self.ref = torch.cat([self.a, self.b, self.c])

    def reward(self, answer, stopped=True):
        return image_answer_reward(answer, self.ref, stopped=stopped)

    def test_exact_and_capped(self):
        self.assertEqual(self.reward(self.ref), 1.)
        self.assertEqual(self.reward(self.ref, False), 0.)

    def test_white_and_empty_cannot_win(self):
        self.assertEqual(self.reward(torch.zeros_like(self.ref)), 0.)
        self.assertEqual(self.reward(self.ref[:0]), 0.)

    def test_order_insertion_deletion_and_repetition(self):
        for wrong in [self.ref.flip(0), self.ref[:2],
                      torch.cat([self.ref, self.c]), self.a.repeat(3, 1, 1, 1)]:
            self.assertLess(self.reward(wrong), .5)

    def test_a_wrong_similar_glyph_is_not_exact(self):
        changed = self.ref.clone(); changed[0, 0, 4, 5] = 1
        self.assertGreater(self.reward(changed), 0.)
        self.assertLess(self.reward(changed), .5)

    def test_real_whitespace_preserved(self):
        reference = torch.cat([self.a, torch.zeros_like(self.a), self.b])
        self.assertEqual(image_answer_reward(reference, reference, stopped=True), 1.)
        self.assertLess(image_answer_reward(torch.cat([self.a, self.b]), reference, stopped=True), .5)

    def test_question_ownership_changes_reward(self):
        self.assertGreater(image_answer_reward(self.a, self.a, stopped=True),
                           image_answer_reward(self.a, self.b, stopped=True))

    def test_reject_invalid_references_and_nonbinary_pixels(self):
        with self.assertRaises(ValueError):
            image_answer_reward(self.a, torch.zeros_like(self.a), stopped=True)
        with self.assertRaises(ValueError):
            image_answer_reward(self.a.float() + .1, self.a, stopped=True)


class SequenceProbabilityTests(unittest.TestCase):
    def test_failed_default_reward_refuses_before_actor_or_sampling(self):
        with self.assertRaisesRegex(RuntimeError, 'No qualified default'):
            backward_answer_group(None, None, None, generator=None)

    def test_invalid_explicit_reward_refuses_before_actor_or_sampling(self):
        with self.assertRaises(TypeError):
            backward_answer_group(None, None, None, generator=None, reward_function=5)

    def test_stopped_likelihood_and_gradient(self):
        z = torch.tensor([-.4, .2, .7], dtype=torch.float64, requires_grad=True)
        score = stopping_log_probability(z, 2, stopped=True)
        expected = F.logsigmoid(-z[0]) + F.logsigmoid(-z[1]) + F.logsigmoid(z[2])
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(torch.autograd.grad(score, z)[0],
                                   torch.tensor([0., 0., 1.], dtype=z.dtype) - z.sigmoid())

    def test_cap_does_not_add_false_terminal_action(self):
        z = torch.tensor([-.4, .2, .7], requires_grad=True)
        score = stopping_log_probability(z, 2, stopped=False)
        torch.testing.assert_close(score, F.logsigmoid(-z[:2]).sum())
        self.assertEqual(float(torch.autograd.grad(score, z)[0][-1]), 0.)

    def test_empty_stop_is_one_real_action(self):
        z = torch.tensor([.4], requires_grad=True)
        torch.testing.assert_close(stopping_log_probability(z, 0, stopped=True), F.logsigmoid(z[0]))

    def test_sample_event_validation(self):
        empty = torch.empty(0, 1, 32, 32, dtype=torch.uint8)
        SampledAnswer(empty, True, 32, -1.)
        with self.assertRaises(ValueError):
            SampledAnswer(empty, False, 32, -1.)
        with self.assertRaises(ValueError):
            SampledAnswer(torch.zeros(32, 1, 32, 32), True, 32, -1.)

    def test_rloo_exact_expected_gradient(self):
        # Enumerate ALL pairs, not a noisy statistical gradient-sign test.
        # This verifies the estimator, not any learned language capability.
        logits = torch.tensor([-.7, .3, .8], dtype=torch.float64, requires_grad=True)
        rewards = torch.tensor([0., .2, 1.], dtype=torch.float64)
        log_probs = logits.log_softmax(-1); probabilities = log_probs.exp()
        losses = []
        for a, b in itertools.product(range(3), repeat=2):
            weight = (probabilities[a] * probabilities[b]).detach()
            losses.append(weight * rloo_loss(log_probs[[a, b]], rewards[[a, b]]))
        estimated = torch.autograd.grad(torch.stack(losses).sum(), logits, retain_graph=True)[0]
        exact = torch.autograd.grad(-(probabilities * rewards).sum(), logits)[0]
        torch.testing.assert_close(estimated, exact, atol=1e-14, rtol=1e-14)

    def test_equal_rewards_give_no_policy_signal(self):
        p = torch.tensor([-3., -5., -8.], requires_grad=True)
        rloo_loss(p, [1., 1., 1.]).backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_rewards_are_detached(self):
        p = torch.tensor([-3., -5.], requires_grad=True)
        r = torch.tensor([1., .2], requires_grad=True)
        rloo_loss(p, r).backward()
        self.assertIsNone(r.grad)
        self.assertLess(float(p.grad[0]), 0.)


class ActualDensitySamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(20261116)
        cls.density = GlyphJointDensity(torch.zeros(4096), torch.ones(4096)).eval()
        cls.states = torch.randn(2, 4096)

    def test_sample_dense_replay_and_roundtrip(self):
        patterns, sampled = sample_physical_patterns(self.density, self.states,
            generator=torch.Generator().manual_seed(654))
        self.assertEqual(tuple(patterns.shape), (2, 128))
        self.assertTrue(torch.equal(patterns, pack_glyphs(unpack_glyphs(patterns))))
        states = self.states.detach().clone().requires_grad_()
        replay = physical_log_probability(self.density, states, patterns)
        torch.testing.assert_close(sampled, replay, atol=2e-4, rtol=1e-6)
        self.density.zero_grad(set_to_none=True)
        replay.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.density.parameters()))
        self.assertTrue(torch.isfinite(states.grad).all() and bool(states.grad.abs().sum() > 0))

    def test_random_generator_replay(self):
        a, pa = sample_physical_patterns(self.density, self.states,
            generator=torch.Generator().manual_seed(42))
        b, pb = sample_physical_patterns(self.density, self.states,
            generator=torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a, b) and torch.equal(pa, pb))

    def test_eval_mode_is_required(self):
        self.density.train()
        try:
            with self.assertRaises(ValueError):
                sample_physical_patterns(self.density, self.states,
                    generator=torch.Generator().manual_seed(42))
        finally:
            self.density.eval()


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
