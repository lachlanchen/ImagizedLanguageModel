"""Independent full-autograd and whitespace controls for a proposed objective."""
import copy

import pytest
import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs
from ilm.visual_lm.global_glyph_ranking import backward_global_glyph_ranking


@pytest.mark.parametrize('strength,all_blank', [(1., False), (0., False), (1., True)])
def test_exact_context_and_writer_gradients_against_unsplit_reference(strength, all_blank):
    torch.set_num_threads(2);torch.manual_seed(421)
    context = torch.nn.Linear(9, 4096); writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    other_context, other_writer = copy.deepcopy(context), copy.deepcopy(writer)
    inputs = torch.randn(6, 9);ink = torch.randint(0, 2, (5, 1, 32, 32), dtype=torch.uint8)
    ink[1] = 0
    if all_blank:ink.zero_()
    h = other_context(inputs);target = pack_glyphs(ink)
    source = other_writer(h[:-1], target).log_softmax(-1).gather(-1, target[..., None]).squeeze(-1).sum(1)
    blank = other_writer(h[:-1], torch.zeros_like(target)).log_softmax(-1)[..., 0].sum(1)
    active = ink.reshape(5, -1).sum(1) > 0
    image = -source.mean() / 128
    rank = (F.softplus(1. + blank - source) * active).mean() / 128
    s = stop(h).flatten(); eos = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
    reference = image + strength * rank + .1 * eos;reference.backward()
    actual = backward_global_glyph_ranking(context(inputs), writer, stop, ink,
        ranking_strength=strength, microbatch_glyphs=2)
    assert abs(actual['loss'] - float(reference.detach())) < 3e-6
    assert actual['image_microbatches'] == 3 and actual['nonempty_targets'] == int(active.sum())
    if all_blank or strength == 0:assert actual['ranking_loss'] == 0
    for module, expected in [(context, other_context), (writer, other_writer)]:
        for (name, p), (other_name, q) in zip(module.named_parameters(), expected.named_parameters()):
            assert name == other_name and p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad, q.grad, atol=4e-7, rtol=5e-4), name
    assert all(p.grad is None for p in stop.parameters())


def test_invalid_state_and_hyperparameters_fail_before_backward():
    density = GlyphJointDensity(torch.zeros(4096), torch.ones(4096)); stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    ink = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    for kwargs in [dict(ranking_strength=-1), dict(margin_nats=float('nan')), dict(microbatch_glyphs=0)]:
        with pytest.raises(ValueError):
            backward_global_glyph_ranking(torch.zeros(2, 4096, requires_grad=True), density, stop, ink, **kwargs)
    with pytest.raises(ValueError, match='differentiable'):
        backward_global_glyph_ranking(torch.zeros(2, 4096), density, stop, ink)
