"""A normalized physical-image likelihood, without a lexical output inventory.

The conditioning states are supplied by a separate image reader. This module
alone is a writing head, not a language model or an image-question interface.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def pack_pixels(ink):
    """Losslessly pack Bx32x64 black-pixel masks into 256 eight-bit patterns."""
    if ink.ndim != 3 or ink.shape[1:] != (32, 64):
        raise ValueError("physical blocks must have shape Bx32x64")
    if not bool(((ink == 0) | (ink == 1)).all()):
        raise ValueError("binary foreground masks required")
    pieces = ink.reshape(-1, 16, 2, 16, 4).permute(0, 1, 3, 2, 4).reshape(-1, 256, 8)
    return (pieces.long() * (1 << torch.arange(8, device=ink.device))).sum(-1)


def pattern_bits(patterns):
    return ((patterns[..., None].long() >> torch.arange(8, device=patterns.device)) & 1).float()


def unpack_pixels(patterns):
    """The inverse packing operation contains no font, character or glyph table."""
    if patterns.ndim != 2 or patterns.shape[1] != 256:
        raise ValueError("256 physical patterns required")
    if not bool(((patterns >= 0) & (patterns < 256)).all()):
        raise ValueError("eight-bit patterns required")
    return pattern_bits(patterns).reshape(-1, 16, 16, 2, 4).permute(0, 1, 3, 2, 4).reshape(-1, 32, 64).bool()


@dataclass(frozen=True)
class PixelDensityConfig:
    condition_width: int = 8192
    width: int = 192
    layers: int = 4
    heads: int = 6
    feedforward_width: int = 768
    positions: int = 256


class PixelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3)
        self.project = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(nn.Linear(config.width, config.feedforward_width),
            nn.GELU(), nn.Linear(config.feedforward_width, config.width))

    def forward(self, x, *, cache=None, incremental=False):
        b, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads,
            width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if incremental:
            if length != 1:
                raise ValueError("incremental calls accept one new physical position")
            if cache is not None:
                k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
            # A single last-position query may see the entire cached prefix.
            # Rectangular is_causal=True would incorrectly mask that prefix.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            if cache is not None:
                raise ValueError("full-prefix calls do not accept a cache")
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.project(y.transpose(1, 2).reshape(b, length, width))
        x = x + self.feedforward(self.norm2(x))
        return x, (k, v) if incremental else None


class ConditionalPixelDensity(nn.Module):
    def __init__(self, mean, scale, config=PixelDensityConfig()):
        super().__init__()
        if config.width % config.heads or mean.shape != (config.condition_width,) or scale.shape != mean.shape:
            raise ValueError("conditioning/configuration shapes differ")
        if not bool(torch.isfinite(mean).all() & torch.isfinite(scale).all() & (scale > 0).all()):
            raise ValueError("finite training-only normalization required")
        self.config = config
        self.register_buffer("condition_mean", mean.detach().float().clone())
        self.register_buffer("condition_scale", scale.detach().float().clone())
        self.condition = nn.Linear(config.condition_width, config.width)
        self.bits = nn.Linear(8, config.width)
        self.start = nn.Parameter(torch.zeros(config.width))
        self.positions = nn.Parameter(torch.empty(config.positions, config.width))
        nn.init.normal_(self.positions, std=.02)
        self.blocks = nn.ModuleList(PixelBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 256)

    def configuration(self):
        return asdict(self.config)

    def condition_features(self, states):
        states = states.reshape(len(states), self.config.condition_width).float()
        return self.condition((states - self.condition_mean) / self.condition_scale)

    def forward(self, states, targets):
        """Score every target piece; target j only enters inputs at position j+1."""
        if targets.ndim != 2 or targets.shape[1] > self.config.positions or targets.shape[1] < 1:
            raise ValueError("a nonempty physical target prefix is required")
        previous = self.bits(pattern_bits(targets[:, :-1]) * 2 - 1)
        x = torch.cat((self.start.expand(len(targets), 1, -1), previous), dim=1)
        x = x + self.positions[:targets.shape[1]] + self.condition_features(states)[:, None]
        for block in self.blocks:
            x, _ = block(x)
        return self.output(self.final_norm(x))

    def step(self, condition, previous, position, caches):
        if not 0 <= position < self.config.positions:
            raise ValueError("physical position out of bounds")
        if position == 0:
            if previous is not None or caches is not None:
                raise ValueError("the first step has no preceding pattern or cache")
            x = self.start.expand(len(condition), 1, -1)
        else:
            if previous is None or caches is None:
                raise ValueError("later steps require the emitted prefix")
            x = self.bits(pattern_bits(previous) * 2 - 1)[:, None]
        x = x + self.positions[position] + condition[:, None]
        updated = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, cache=None if caches is None else caches[i], incremental=True)
            updated.append(cache)
        return self.output(self.final_norm(x[:, 0])), updated

    @torch.no_grad()
    def generate(self, states):
        """Autonomous greedy physical blocks; accepts no answer or target length."""
        condition, caches, previous, pieces = self.condition_features(states), None, None, []
        for position in range(self.config.positions):
            logits, caches = self.step(condition, previous, position, caches)
            previous = logits.argmax(-1)
            pieces.append(previous)
        return torch.stack(pieces, dim=1)
