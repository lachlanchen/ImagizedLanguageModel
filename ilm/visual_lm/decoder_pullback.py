"""State-only decoder geometry for continuous clean-code regression.

The metric point is the observed noisy code, detached from optimization.
Only the residual carries gradients. No hidden target, predicted endpoint,
character inventory or semantic teacher determines the metric.
"""
import math

import torch
from torch.nn import functional as F

from .continuous_glyph_codec import ContinuousGlyphCodec


def normalized_decoder_logits(codec, value):
    """The sampler's per-tile normalization followed by the frozen decoder."""
    tiles = F.layer_norm(value.reshape(-1, 768), (768,), eps=codec.config.layer_norm_epsilon)
    # Direct decoder access preserves dtype for independent FP64 checks.
    # In the actual FP32 loss this equals codec.decode numerically.
    return codec.decoder(tiles).reshape(-1, 2048)


def decoder_pullback_error(codec, noisy, residual, *, strength=1e-3):
    """Return per-block code MSE plus strength * mean((J_R(noisy) residual)^2).

    R is the normalized decoder-logit map. The positive code-MSE term makes
    the state-only quadratic metric positive definite, including normalization
    null directions. This weights the original regression; it does not change
    the flow path, guarantee convergence, or qualify generated writing.
    """
    if not isinstance(codec, ContinuousGlyphCodec):
        raise TypeError("the actual continuous writing codec is required")
    if (noisy.ndim != 2 or noisy.shape[1] != 1536 or residual.shape != noisy.shape
            or noisy.dtype != torch.float32 or residual.dtype != torch.float32
            or noisy.device != residual.device or not len(noisy)):
        raise ValueError("nonempty FP32 noisy codes and residuals [N,1536] required")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("finite nonnegative metric strength required")
    parameters = tuple(codec.parameters())
    if any(p.requires_grad for p in parameters) or any(p.dtype != torch.float32 or p.device != noisy.device for p in parameters):
        raise ValueError("a frozen FP32 codec on the same device is required")
    if not bool(torch.isfinite(noisy).all() and torch.isfinite(residual).all()):
        raise ValueError("finite image coordinates required")
    with torch.autocast(device_type=noisy.device.type, enabled=False):
        code = residual.square().mean(-1)
        if strength == 0:
            return code, dict(code=code, decoder_tangent=torch.zeros_like(code))
        _, tangent = torch.func.jvp(lambda x:normalized_decoder_logits(codec, x),
            (noisy.detach(),), (residual,))
        physical = tangent.square().mean(-1)
    return code+strength*physical, dict(code=code, decoder_tangent=physical)
