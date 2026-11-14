"""Whole-image anti-blank ranking with exact bounded-memory context gradients.

This candidate objective changes training only. It preserves the complete
normalized physical density, true whitespace likelihood and inference decoder.
Its implementation and gradient checks establish no language result.
"""
import math

import torch
from torch.nn import functional as F

from .glyph_joint_density import pack_glyphs


def backward_global_glyph_ranking(states, density, stop, ink, *, ranking_strength=1.,
                                  margin_nats=1., microbatch_glyphs=64, guard=lambda: None):
    """Backpropagate one complete source, leaving the optimizer to the caller.

    For a nonempty target, add softplus(margin + logq(blank) - logq(target)).
    Divide complete-glyph log probabilities and the ranking term by128 so their
    scale matches the original per-mask likelihood. Average over all target
    positions, retaining the original equal-source weighting. Empty targets
    receive their original likelihood and no anti-blank ranking penalty.
    """
    if states.ndim != 2 or states.shape[1] != 4096 or len(states) != len(ink) + 1:
        raise ValueError('Complete [T+1,4096] predictor states and T image targets required')
    if not states.requires_grad or not len(ink) or not bool(torch.isfinite(states).all()):
        raise ValueError('Finite differentiable continuation states required')
    if type(microbatch_glyphs) is not int or microbatch_glyphs < 1:
        raise ValueError('Positive image microbatch size required')
    if not math.isfinite(ranking_strength) or ranking_strength < 0 or not math.isfinite(margin_nats) or margin_nats < 0:
        raise ValueError('Finite nonnegative ranking strength and margin required')
    if any(p.requires_grad for p in stop.parameters()):
        raise ValueError('Preserve the stop-head parameters')
    patterns = pack_glyphs(ink).to(states.device)
    proxy = states.detach().float().requires_grad_(True)
    count = len(patterns); image_value = 0.; ranking_value = 0.; nonempty_count = 0
    for start in range(0, count, microbatch_glyphs):
        guard(); end = min(start + microbatch_glyphs, count); target = patterns[start:end]
        source_lp = density(proxy[start:end], target).float().log_softmax(-1)
        source_logp = source_lp.gather(-1, target[..., None]).squeeze(-1).sum(1)
        image_loss = -source_logp.sum() / (128 * count)
        nonempty = (target != 0).any(1); nonempty_count += int(nonempty.sum())
        ranking_loss = source_logp.new_zeros(())
        if ranking_strength and bool(nonempty.any()):
            zero = torch.zeros_like(target)
            blank_logp = density(proxy[start:end], zero).float().log_softmax(-1)[..., 0].sum(1)
            ranking_loss = (F.softplus(margin_nats + blank_logp - source_logp) * nonempty).sum() / (128 * count)
        loss = image_loss + ranking_strength * ranking_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite complete-image/ranking loss')
        loss.backward(); image_value += float(image_loss.detach()); ranking_value += float(ranking_loss.detach())
        del source_lp, source_logp, image_loss, ranking_loss, loss
    s = stop(proxy).flatten()
    if s.shape != (count + 1,):
        raise ValueError('One stop prediction per image boundary required')
    stop_loss = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
    if not bool(torch.isfinite(stop_loss)):
        raise ValueError('Nonfinite stop loss')
    (.1 * stop_loss).backward()
    gradient = proxy.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Finite complete context gradient required')
    guard(); states.backward(gradient.to(states.dtype)); guard()
    return dict(image_loss=image_value, ranking_loss=ranking_value, stop_loss=float(stop_loss.detach()),
        loss=image_value + ranking_strength * ranking_value + .1 * float(stop_loss.detach()),
        state_gradient_norm=float(gradient.norm()), target_glyphs=count, nonempty_targets=nonempty_count,
        image_microbatches=(count + microbatch_glyphs - 1) // microbatch_glyphs)
