"""Causal image history and a normalized density over continuous writing codes.

This research candidate has no measured language quality. Code density is not
pixel likelihood: the frozen encoder/decoder is not an invertible change of
variables. A generated code must be drawn and reread before the next glyph.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .conditional_pixel_density import PixelBlock, PixelDensityConfig
from .continuous_glyph_codec import ContinuousGlyphCodec
from .visual_cell_stream import CausalVisualBlock, RMSNorm, VisualCellStreamConfig


@dataclass(frozen=True)
class ContinuousDensityConfig:
    latent_width: int = 768
    group_width: int = 16
    components: int = 8
    local_width: int = 128
    local_layers: int = 2
    local_heads: int = 4
    minimum_scale: float = .02

    def __post_init__(self):
        if (self.latent_width < 1 or self.group_width < 1
                or self.latent_width % self.group_width or self.components < 1
                or self.local_heads < 1 or self.local_width < 1
                or self.local_width % self.local_heads or self.local_layers < 1
                or not 0 < self.minimum_scale < 1):
            raise ValueError('Invalid continuous writing-density dimensions')


def mixture_log_prob(target, logits, means, scales):
    """One mixture component is shared by all coordinates of a visual group."""
    if (means.shape != scales.shape or means.shape[:-1] != logits.shape
            or means.shape[:-2] + means.shape[-1:] != target.shape):
        raise ValueError('Continuous group and mixture shapes differ')
    if not bool(torch.isfinite(scales).all() & (scales > 0).all()):
        raise ValueError('Positive finite Gaussian scales required')
    residual = (target[..., None, :] - means) / scales
    component = (-.5 * residual.square() - scales.log() - .5 * math.log(2 * math.pi)).sum(-1)
    return (logits.log_softmax(-1) + component).logsumexp(-1)


class ContinuousWritingDensity(nn.Module):
    """Autoregressive groups; target group j enters only the input for j+1."""
    def __init__(self, condition_width, config=ContinuousDensityConfig()):
        super().__init__()
        self.config = config
        self.groups = config.latent_width // config.group_width
        local = PixelDensityConfig(condition_width=condition_width, width=config.local_width,
            layers=config.local_layers, heads=config.local_heads,
            feedforward_width=4 * config.local_width, positions=self.groups)
        self.condition = nn.Linear(condition_width, config.local_width)
        self.previous = nn.Linear(config.group_width, config.local_width)
        self.start = nn.Parameter(torch.zeros(config.local_width))
        self.positions = nn.Parameter(torch.empty(self.groups, config.local_width))
        nn.init.normal_(self.positions, std=.02)
        self.blocks = nn.ModuleList(PixelBlock(local) for _ in range(config.local_layers))
        self.norm = nn.LayerNorm(config.local_width)
        self.output = nn.Linear(config.local_width, config.components * (1 + 2 * config.group_width))
        # Start near a unit normal, not a numerically singular point density.
        with torch.no_grad():
            self.output.weight.mul_(.02)
            self.output.bias.zero_()
            self.output.bias.view(config.components, -1)[:, 1 + config.group_width:] = math.log(math.expm1(1 - config.minimum_scale))

    def parameters_at(self, state):
        c = self.config
        raw = self.output(self.norm(state)).float().reshape(*state.shape[:-1], c.components, -1)
        return raw[..., 0], raw[..., 1:1 + c.group_width], F.softplus(raw[..., 1 + c.group_width:]) + c.minimum_scale

    def forward(self, condition, target):
        if target.ndim != 2 or target.shape != (len(condition), self.config.latent_width):
            raise ValueError('Complete continuous target codes required')
        groups = target.reshape(len(target), self.groups, self.config.group_width)
        previous = self.previous(groups[:, :-1])
        state = torch.cat((self.start.expand(len(target), 1, -1), previous), dim=1)
        state = state + self.positions + self.condition(condition)[:, None]
        for block in self.blocks:
            state, _ = block(state)
        return self.parameters_at(state)

    def log_prob(self, condition, target):
        parameters = self(condition, target)
        return mixture_log_prob(target.reshape(len(target), self.groups, -1), *parameters).sum(-1)

    def step(self, features, previous, position, caches):
        if not 0 <= position < self.groups:
            raise ValueError('Visual group index out of range')
        if position == 0:
            if previous is not None or caches is not None:
                raise ValueError('First visual group has no emitted prefix')
            state = self.start.expand(len(features), 1, -1)
        else:
            if previous is None or caches is None:
                raise ValueError('Later groups require their actual emitted prefix')
            state = self.previous(previous)[:, None]
        state = state + self.positions[position] + features[:, None]
        updated = []
        for i, block in enumerate(self.blocks):
            state, cache = block(state, cache=None if caches is None else caches[i], incremental=True)
            updated.append(cache)
        return self.parameters_at(state[:, 0]), updated

    @torch.no_grad()
    def generate(self, condition, *, stochastic=False, generator=None):
        features = self.condition(condition)
        previous, caches, emitted = None, None, []
        for position in range(self.groups):
            (logits, means, scales), caches = self.step(features, previous, position, caches)
            choice = (torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
                      if stochastic else logits.argmax(-1))
            rows = torch.arange(len(condition), device=condition.device)
            previous = means[rows, choice]
            if stochastic:
                previous = previous + scales[rows, choice] * torch.randn(previous.shape,
                    device=previous.device, dtype=previous.dtype, generator=generator)
            emitted.append(previous)
        # Greedy component mean is an explicit heuristic, not joint MAP.
        return torch.cat(emitted, -1)


class ContinuousImageLanguage(nn.Module):
    """Native image-only causal reader, writing density and learned stopping."""
    def __init__(self, codec, *, components=8, model_dim=384, layers=8, heads=6,
                 maximum_cells=2048, density_config=None):
        super().__init__()
        if not isinstance(codec, ContinuousGlyphCodec):
            raise TypeError('A real continuous neural writing codec is required')
        self.codec = codec.requires_grad_(False).eval()
        self.maximum_cells = maximum_cells
        config = VisualCellStreamConfig(maximum_cells=maximum_cells, model_dim=model_dim,
            layers=layers, heads=heads, dropout=0.)
        self.input_projection = nn.Linear(768, model_dim, bias=False)
        self.blocks = nn.ModuleList(CausalVisualBlock(config) for _ in range(layers))
        self.norm = RMSNorm(model_dim)
        self.stop = nn.Linear(model_dim, 1)
        self.writer = ContinuousWritingDensity(model_dim, density_config or ContinuousDensityConfig(components=components))

    def train(self, mode=True):
        super().train(mode)
        self.codec.eval()
        return self

    @torch.no_grad()
    def encode_ink(self, ink):
        if ink.ndim != 4 or ink.shape[1:] != (1, 32, 32) or not bool(((ink == 0) | (ink == 1)).all()):
            raise ValueError('Complete binary foreground images required')
        # V34 was trained on white-positive pixels; the corpus is ink-positive.
        return self.codec.encode(1 - ink.float())

    def history_codes(self, codes):
        if codes.ndim != 3 or codes.shape[-1] != 768 or not 1 <= codes.shape[1] <= self.maximum_cells:
            raise ValueError('Nonempty causal visual code history required')
        state = self.input_projection(codes)
        for block in self.blocks:
            state = block(state)
        return self.norm(state)

    def history(self, ink):
        return self.history_codes(self.encode_ink(ink)[None])

    def training_terms(self, ink, prompt_glyphs):
        if not 0 <= prompt_glyphs < len(ink) <= self.maximum_cells:
            raise ValueError('Complete prompt and answer/continuation required')
        # Book pretraining marks no instruction prefix. Its first image is the
        # causal anchor, exactly as in the original continuation schedule.
        prompt_glyphs = max(1, prompt_glyphs)
        codes = self.encode_ink(ink)
        hidden = self.history_codes(codes[None])[0]
        first = prompt_glyphs - 1
        logp = self.writer.log_prob(hidden[first:-1], codes[prompt_glyphs:])
        stop_logits = self.stop(hidden[first:]).squeeze(-1)
        targets = torch.zeros_like(stop_logits)
        targets[-1] = 1
        return dict(code_nll=-logp.mean() / 768,
                    stop_bce=F.binary_cross_entropy_with_logits(stop_logits, targets),
                    target_positions=len(logp), stop_positions=len(stop_logits))

    @torch.no_grad()
    def draw(self, states, *, stochastic=False, generator=None):
        codes = self.writer.generate(states, stochastic=stochastic, generator=generator)
        return (self.codec.decode(codes) < 0).to(torch.uint8), codes

    @torch.no_grad()
    def generate(self, prompt, *, maximum_new=128, stochastic=False, generator=None):
        if self.training:
            raise ValueError('Set eval mode for standalone generation')
        if maximum_new < 1:
            raise ValueError('Positive finite output cap required')
        # Re-encode the actual emitted pixels, never an unrendered predicted code.
        history, emitted, stopped = prompt, [], False
        for _ in range(maximum_new):
            states = self.history(history[-self.maximum_cells:])[0, -1:]
            if bool(self.stop(states).squeeze() >= 0):
                stopped = True
                break
            ink, _ = self.draw(states, stochastic=stochastic, generator=generator)
            emitted.append(ink)
            history = torch.cat((history, ink))
        return (torch.cat(emitted) if emitted else prompt[:0]), dict(stopped=stopped, capped=not stopped,
            actual_emitted_pixel_feedback=True, glyphs=len(emitted))
