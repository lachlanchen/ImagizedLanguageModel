"""Normalized conditional likelihood of a complete physical 32-by-32 glyph.

The 256 output values enumerate all possible eight-pixel masks, not characters.
This module is a writer component; its existence establishes no language result.
"""
import torch
from torch import nn

from .conditional_pixel_density import ConditionalPixelDensity, PixelDensityConfig, pattern_bits


def pack_glyphs(ink):
    if ink.ndim != 4 or ink.shape[1:] != (1, 32, 32):
        raise ValueError('Complete [N,1,32,32] glyph masks required')
    if not bool(((ink == 0) | (ink == 1)).all()):
        raise ValueError('Binary physical pixels required')
    pieces = ink.reshape(-1, 16, 2, 8, 4).permute(0, 1, 3, 2, 4).reshape(-1, 128, 8)
    return (pieces.long() * (1 << torch.arange(8, device=ink.device))).sum(-1)


def unpack_glyphs(patterns):
    if patterns.ndim != 2 or patterns.shape[1] != 128:
        raise ValueError('128 physical patterns required')
    if not bool(((patterns >= 0) & (patterns < 256) & (patterns == patterns.long())).all()):
        raise ValueError('Eight-bit patterns required')
    return pattern_bits(patterns).reshape(-1, 16, 8, 2, 4).permute(0, 1, 3, 2, 4).reshape(-1, 1, 32, 32).to(torch.uint8)


class GlyphJointDensity(ConditionalPixelDensity):
    def __init__(self, mean, scale):
        super().__init__(mean, scale, PixelDensityConfig(condition_width=4096, positions=128))

    def image_nll(self, states, ink):
        """One mean pattern NLL per glyph; caller controls sequence weighting."""
        if states.shape != (len(ink), 4096) or not len(ink) or not bool(torch.isfinite(states).all()):
            raise ValueError('One finite contextual state per target image required')
        patterns = pack_glyphs(ink)
        logits = self(states, patterns)
        return -logits.float().log_softmax(-1).gather(-1, patterns[..., None]).squeeze(-1).mean(1)

    @torch.no_grad()
    def write(self, states):
        if states.ndim != 2 or states.shape[1] != 4096 or not len(states) or not bool(torch.isfinite(states).all()):
            raise ValueError('Nonempty finite contextual states required')
        return unpack_glyphs(self.generate(states))


class JointGlyphBridge(nn.Module):
    """Use the new writer; retain the exact frozen image feedback and stop path."""
    def __init__(self, base, density):
        super().__init__()
        self.encoder = base.encoder.requires_grad_(False).eval()
        self.stop = base.stop.requires_grad_(False).eval()
        self.register_buffer('semantic_mean', base.semantic_mean.detach().clone())
        self.register_buffer('semantic_scale', base.semantic_scale.detach().clone())
        self.density = density

    def feedback_embedding(self, images):
        return self.encoder(images.float()).float() * self.semantic_scale + self.semantic_mean

    @torch.no_grad()
    def predict(self, state):
        ink = self.density.write(state)
        # The continuation actor thresholds these exact emitted physical bits.
        logits = (ink.float() * 2 - 1) * 20
        return logits, state, self.stop(state.float()).flatten()
