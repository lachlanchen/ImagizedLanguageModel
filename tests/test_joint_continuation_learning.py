"""Independent full-autograd checks of the actual joint writer's state VJP."""
import copy

import pytest
import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs
from ilm.visual_lm.joint_continuation_learning import backward_complete_image_likelihood


def test_microbatched_state_vjp_matches_unsplit_autograd_for_every_parameter():
    torch.set_num_threads(2); torch.manual_seed(143)
    context = torch.nn.Linear(12, 4096)
    writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    reference_context, reference_writer = copy.deepcopy(context), copy.deepcopy(writer)
    inputs = torch.randn(6, 12)
    ink = torch.randint(0, 2, (5, 1, 32, 32), dtype=torch.uint8)
    # Direct, unsplit differentiation is the independent numerical reference.
    h = reference_context(inputs); targets = pack_glyphs(ink)
    nll = -reference_writer(h[:-1], targets).log_softmax(-1).gather(-1, targets[..., None]).mean()
    s = stop(h).flatten()
    eos = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
    expected = nll + .1 * eos; expected.backward()
    result = backward_complete_image_likelihood(context(inputs), writer, stop, ink, microbatch_glyphs=2)
    assert result['image_microbatches'] == 3
    assert abs(result['loss'] - float(expected.detach())) < 2e-6
    assert result['state_gradient_norm'] > 0
    for actual, reference in [(context, reference_context), (writer, reference_writer)]:
        for (name, parameter), (other_name, other) in zip(actual.named_parameters(), reference.named_parameters()):
            assert name == other_name and parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.allclose(parameter.grad, other.grad, atol=2e-7, rtol=3e-4), name
    assert all(p.grad is None for p in stop.parameters())


def test_rejects_an_accidentally_frozen_state_or_unfrozen_stop():
    torch.set_num_threads(2)
    writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    ink = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match='differentiable'):
        backward_complete_image_likelihood(torch.zeros(2, 4096), writer, stop, ink)
    stop.requires_grad_(True)
    with pytest.raises(ValueError, match='stop-head'):
        backward_complete_image_likelihood(torch.zeros(2, 4096, requires_grad=True), writer, stop, ink)
