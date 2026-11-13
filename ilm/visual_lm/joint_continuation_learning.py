"""Exact state-gradient accumulation for complete joint-image supervision.

The continuation graph is differentiated once, after small physical-writer
microbatches accumulate its state gradient. This is an ordinary chain-rule
factorization, not a separate teacher or a learned gradient approximation.
"""
import torch
from torch.nn import functional as F

from .glyph_joint_density import pack_glyphs


def backward_complete_image_likelihood(states, density, stop, ink, *,
                                       microbatch_glyphs=64, guard=lambda: None):
    """Accumulate writer and actual continuation gradients without an update.

    ``states[t]`` must precede target image ``ink[t]``; the caller owns that
    causal construction. The terminal state has stop supervision only. The
    caller zeroes gradients, clips them and performs one optimizer update per
    complete source. Feedback, native body and stop parameters stay frozen.
    """
    if states.ndim != 2 or states.shape[1] != 4096 or len(states) != len(ink) + 1:
        raise ValueError('Complete [T+1,4096] predictor states and T image targets required')
    if not states.requires_grad or not len(ink) or not bool(torch.isfinite(states).all()):
        raise ValueError('Finite differentiable continuation states required')
    if type(microbatch_glyphs) is not int or microbatch_glyphs < 1:
        raise ValueError('Positive image microbatch size required')
    if any(parameter.requires_grad for parameter in stop.parameters()):
        raise ValueError('This comparison preserves the stop-head parameters')
    patterns = pack_glyphs(ink).to(states.device)
    proxy = states.detach().float().requires_grad_(True)
    count = len(patterns); image_loss = 0.
    for start in range(0, count, microbatch_glyphs):
        guard(); end = min(start + microbatch_glyphs, count)
        target = patterns[start:end]
        logits = density(proxy[start:end], target).float()
        loss = -logits.log_softmax(-1).gather(-1, target[..., None]).sum() / (128 * count)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite complete-image likelihood')
        loss.backward(); image_loss += float(loss.detach())
        del logits, loss
    stop_logits = stop(proxy).flatten()
    if stop_logits.shape != (count + 1,):
        raise ValueError('One stop prediction per image boundary required')
    stop_loss = .5 * (F.softplus(stop_logits[:-1]).mean() + F.softplus(-stop_logits[-1]))
    if not bool(torch.isfinite(stop_loss)):
        raise ValueError('Nonfinite stop loss')
    (.1 * stop_loss).backward()
    gradient = proxy.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Finite complete continuation gradient required')
    guard()
    states.backward(gradient.to(states.dtype))
    guard()
    return dict(image_loss=image_loss, stop_loss=float(stop_loss.detach()),
                loss=image_loss + .1 * float(stop_loss.detach()),
                state_gradient_norm=float(gradient.norm()),
                target_glyphs=count, image_microbatches=(count + microbatch_glyphs - 1) // microbatch_glyphs)
