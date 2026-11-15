"""Independent full-autograd gradients and enumerable blank-bound checks."""
import copy
import itertools
import os
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs
from ilm.visual_lm.global_glyph_ranking import backward_global_glyph_ranking
from ilm.visual_lm.white_target_ranking import backward_white_target_ranking, blank_prefix_bound


@pytest.mark.parametrize('kind,white_strength', [('mixed', 1.), ('white', 1.), ('ink', 1.), ('mixed', 0.)])
def test_all_context_and_real_writer_gradients_match_unsplit(kind, white_strength):
    torch.set_num_threads(2); torch.manual_seed(515)
    context = torch.nn.Linear(9, 4096); writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    other_context, other_writer = copy.deepcopy(context), copy.deepcopy(writer)
    inputs = torch.randn(6, 9); ink = torch.randint(0, 2, (5, 1, 32, 32), dtype=torch.uint8)
    if kind == 'mixed': ink[1] = 0
    if kind == 'white': ink.zero_()
    h = other_context(inputs); target = pack_glyphs(ink)
    lp = other_writer(h[:-1], target).float().log_softmax(-1)
    source = lp.gather(-1, target[..., None]).squeeze(-1).sum(1)
    zero_lp = other_writer(h[:-1], torch.zeros_like(target)).float().log_softmax(-1)
    blank = zero_lp[..., 0].sum(1); active = (target != 0).any(1)
    image = -source.mean() / 128
    rank = (F.softplus(1 + blank - source) * active).mean() / 128
    # Independent direct formula; do not use the production bound helper.
    candidates = [zero_lp[:, :j, 0].sum(1) + zero_lp[:, j, 1:].max(1).values for j in range(128)]
    upper = torch.stack(candidates, 1).max(1).values
    white = (F.softplus(1 + upper - blank) * ~active).mean() / 128
    s = stop(h).flatten(); eos = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
    reference = image + rank + white_strength * white + .1 * eos
    reference.backward()
    result = backward_white_target_ranking(context(inputs), writer, stop, ink,
        white_strength=white_strength, microbatch_glyphs=2)
    assert abs(result['loss'] - float(reference.detach())) < 3e-6
    assert result['white_targets'] == int((~active).sum()) and result['image_microbatches'] == 3
    assert (result['white_ranking_loss'] > 0) == (bool((~active).any()) and bool(white_strength))
    for module, expected in [(context, other_context), (writer, other_writer)]:
        for (name, p), (other_name, q) in zip(module.named_parameters(), expected.named_parameters()):
            assert name == other_name and p.grad is not None and torch.isfinite(p.grad).all()
            assert torch.allclose(p.grad, q.grad, atol=5e-7, rtol=7e-4), name
    assert all(p.grad is None for p in stop.parameters())


def test_zero_added_strength_reproduces_existing_ranking_gradients():
    torch.set_num_threads(2); torch.manual_seed(517)
    context = torch.nn.Linear(9, 4096); writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    c2, w2 = copy.deepcopy(context), copy.deepcopy(writer)
    stop = torch.nn.Linear(4096, 1).requires_grad_(False); inputs = torch.randn(6, 9)
    ink = torch.randint(0, 2, (5, 1, 32, 32), dtype=torch.uint8); ink[2] = 0
    old = backward_global_glyph_ranking(c2(inputs), w2, stop, ink, microbatch_glyphs=2)
    new = backward_white_target_ranking(context(inputs), writer, stop, ink, white_strength=0., microbatch_glyphs=2)
    assert old['loss'] == new['loss'] and new['white_ranking_loss'] == 0
    for actual, expected in [(context, c2), (writer, w2)]:
        assert all(torch.equal(a.grad, b.grad) for a, b in zip(actual.parameters(), expected.parameters()))


@pytest.mark.parametrize('uniform', [False, True])
def test_prefix_bound_dominates_every_enumerated_nonempty_path_with_ties(uniform):
    torch.manual_seed(519); length, alphabet = 4, 3; tables = {}
    for j in range(length):
        for prefix in itertools.product(range(alphabet), repeat=j):
            logits = torch.zeros(alphabet, dtype=torch.float64) if uniform else torch.randn(alphabet, dtype=torch.float64)
            tables[prefix] = logits.log_softmax(0)
    zero_lp = torch.stack([tables[(0,) * j] for j in range(length)])[None].requires_grad_(True)
    blank, upper = blank_prefix_bound(zero_lp)
    paths = []
    for path in itertools.product(range(alphabet), repeat=length):
        score = sum(tables[path[:j]][path[j]] for j in range(length))
        if any(path): paths.append(score)
        else: assert torch.allclose(blank[0], score, atol=1e-12, rtol=0)
    assert float(upper.detach()) + 1e-12 >= max(float(p) for p in paths)
    (upper - blank).sum().backward(); assert torch.isfinite(zero_lp.grad).all()
    if uniform: assert float(upper.detach()) > float(blank.detach())


def test_invalid_bound_inputs_and_training_arguments_reject():
    for value in [torch.zeros(1, 2, 3), torch.full((1, 2, 3), float('nan')), torch.ones(1, 2)]:
        with pytest.raises(ValueError): blank_prefix_bound(value)
    stop = torch.nn.Linear(4096, 1).requires_grad_(False)
    writer = GlyphJointDensity(torch.zeros(4096), torch.ones(4096))
    ink = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    with pytest.raises(ValueError, match='nonnegative'):
        backward_white_target_ranking(torch.zeros(2, 4096, requires_grad=True), writer, stop, ink, white_strength=-1)
    with pytest.raises(ValueError, match='stop-head'):
        backward_white_target_ranking(torch.zeros(2, 4096, requires_grad=True), writer, stop.requires_grad_(True), ink)


def test_missing_preflight_binding_rejects_before_neural_import_or_output(tmp_path):
    program = '''
import sys
from pathlib import Path
from scripts import preflight_white_target_ranking as p
p.PUBLIC=Path(sys.argv[1])/'absent.json'; p.RUN=Path(sys.argv[1])/'absent'
try:p.validate()
except FileNotFoundError:pass
else:raise AssertionError('Missing preflight binding admitted')
assert 'torch' not in sys.modules and not p.RUN.exists()
'''
    subprocess.run([sys.executable, '-c', program, str(tmp_path)], check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
