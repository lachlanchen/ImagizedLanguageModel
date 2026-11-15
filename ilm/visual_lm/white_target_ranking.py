"""Whole-image ranking for both nonempty and truly blank training targets.

The physical density and inference decoder are unchanged. This objective is a
bounded learning intervention, not evidence of useful visual language.
"""
import math

import torch
from torch.nn import functional as F

from .glyph_joint_density import pack_glyphs


def blank_prefix_bound(log_probabilities):
    """Return blank log probability and an upper bound for every nonblank image.

    The input scores each mask after an all-zero prefix. Every nonblank path
    first diverges at some mask; its remaining suffix has log probability <=0.
    Preserve autograd through the selected maximum (one subgradient at ties).
    """
    x = log_probabilities
    if x.ndim != 3 or not len(x) or x.shape[1] < 1 or x.shape[2] < 2:
        raise ValueError('Nonempty [batch, positions, alphabet] log probabilities required')
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()) or bool((x > 1e-7).any()):
        raise ValueError('Finite normalized log probabilities required')
    if float(x.logsumexp(-1).abs().max().detach()) > 1e-5:
        raise ValueError('Normalized log probabilities required')
    zero = x[..., 0]
    previous = torch.cat((torch.zeros_like(zero[:, :1]), zero.cumsum(1)[:, :-1]), 1)
    upper = (previous + x[..., 1:].max(-1).values).max(-1).values
    return zero.sum(1), upper


def backward_white_target_ranking(states, density, stop, ink, *, ranking_strength=1.,
                                  white_strength=1., margin_nats=1.,
                                  microbatch_glyphs=64, guard=lambda: None):
    """Accumulate exact head/state gradients for one complete source.

    Reuse source-path logits at white targets: they already have zero history.
    Keep the existing nonempty source-over-blank term and equal-source average.
    Accumulate on a detached state proxy, then differentiate the actual context
    once. The caller owns zeroing gradients, clipping, optimizer and reset.
    """
    if states.ndim != 2 or states.shape[1] != 4096 or len(states) != len(ink) + 1:
        raise ValueError('Complete [T+1,4096] predictor states and T image targets required')
    if not states.requires_grad or not len(ink) or not bool(torch.isfinite(states).all()):
        raise ValueError('Finite differentiable continuation states required')
    if type(microbatch_glyphs) is not int or microbatch_glyphs < 1:
        raise ValueError('Positive image microbatch size required')
    if any(not math.isfinite(v) or v < 0 for v in [ranking_strength, white_strength, margin_nats]):
        raise ValueError('Finite nonnegative ranking strengths and margin required')
    if any(p.requires_grad for p in stop.parameters()):
        raise ValueError('Preserve the stop-head parameters')
    patterns = pack_glyphs(ink).to(states.device)
    proxy = states.detach().float().requires_grad_(True)
    count = len(patterns); image_value = rank_value = white_value = 0.; white_count = 0
    for start in range(0, count, microbatch_glyphs):
        guard(); end = min(start + microbatch_glyphs, count); target = patterns[start:end]
        source_lp = density(proxy[start:end], target).float().log_softmax(-1)
        source_logp = source_lp.gather(-1, target[..., None]).squeeze(-1).sum(1)
        image_loss = -source_logp.sum() / (128 * count)
        nonempty = (target != 0).any(1); white = ~nonempty; white_count += int(white.sum())
        rank_loss = source_logp.new_zeros(()); white_loss = source_logp.new_zeros(())
        if ranking_strength and bool(nonempty.any()):
            zero = torch.zeros_like(target)
            blank = density(proxy[start:end], zero).float().log_softmax(-1)[..., 0].sum(1)
            rank_loss = (F.softplus(margin_nats + blank - source_logp) * nonempty).sum() / (128 * count)
        if white_strength and bool(white.any()):
            blank, upper = blank_prefix_bound(source_lp[white])
            white_loss = F.softplus(margin_nats + upper - blank).sum() / (128 * count)
        loss = image_loss + ranking_strength * rank_loss + white_strength * white_loss
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite complete-image or ranking loss')
        loss.backward()
        image_value += float(image_loss.detach()); rank_value += float(rank_loss.detach()); white_value += float(white_loss.detach())
        del source_lp, source_logp, image_loss, rank_loss, white_loss, loss
    s = stop(proxy).flatten()
    if s.shape != (count + 1,):
        raise ValueError('One stop prediction per image boundary required')
    stop_loss = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
    if not bool(torch.isfinite(stop_loss)):
        raise ValueError('Nonfinite stop loss')
    (.1 * stop_loss).backward(); gradient = proxy.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Finite complete context gradient required')
    guard(); states.backward(gradient.to(states.dtype)); guard()
    return dict(image_loss=image_value, ranking_loss=rank_value, white_ranking_loss=white_value,
        stop_loss=float(stop_loss.detach()),
        loss=image_value + ranking_strength * rank_value + white_strength * white_value + .1 * float(stop_loss.detach()),
        state_gradient_norm=float(gradient.norm()), target_glyphs=count,
        white_targets=white_count, nonempty_targets=count-white_count,
        image_microbatches=(count + microbatch_glyphs - 1) // microbatch_glyphs)
