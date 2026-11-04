from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .hierarchical_raster_language import CausalRasterReader
from .visual_cell_stream import CausalVisualBlock, VisualCellStreamConfig

V54_ARCHITECTURE = "exact-raster-density-v54"


@dataclass(frozen=True)
class ExactRasterDensityConfig:
    width: int = 192
    layers: int = 4
    heads: int = 6
    pixel_hidden: int = 256
    plan_logit_scale: float = 6.0


class MaskedLinear(nn.Linear):
    def __init__(self, inputs: int, outputs: int, mask: torch.Tensor):
        super().__init__(inputs, outputs)
        if mask.shape != (outputs, inputs):
            raise ValueError("raster mask dimensions differ from weights")
        self.register_buffer("mask", mask.float(), persistent=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.linear(values, self.weight * self.mask, self.bias)


class ConditionalRowMADE(nn.Module):
    """Exact Bernoulli factorization inside one 32-pixel image row."""

    def __init__(self, width: int, hidden: int):
        super().__init__()
        pixel_degree = torch.arange(32)
        hidden_degree = torch.arange(hidden).remainder(31)
        self.first = MaskedLinear(
            32, hidden, hidden_degree[:, None] >= pixel_degree[None]
        )
        self.second = MaskedLinear(
            hidden, hidden, hidden_degree[:, None] >= hidden_degree[None]
        )
        self.output = MaskedLinear(
            hidden, 32, pixel_degree[:, None] > hidden_degree[None]
        )
        self.condition1 = nn.Linear(width, hidden, bias=False)
        self.condition2 = nn.Linear(width, hidden, bias=False)
        self.condition_out = nn.Linear(width, 32, bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.condition_out.weight)

    def forward(self, pixels: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        first = F.silu(self.first(pixels.mul(2).sub(1)) + self.condition1(condition))
        second = F.silu(self.second(first) + self.condition2(condition))
        return self.output(second) + self.condition_out(condition)


class ExactRasterDensity(nn.Module):
    """Normalized next-image probability, with no glyph inventory or codebook."""

    def __init__(self, reader: CausalRasterReader, config: ExactRasterDensityConfig):
        super().__init__()
        self.reader = reader.requires_grad_(False).eval()
        self.config = config
        self.condition = nn.Sequential(
            nn.LayerNorm(1408),
            nn.Linear(1408, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.previous_row = nn.Linear(32, config.width, bias=False)
        self.plan_row = nn.Linear(32, config.width, bias=False)
        self.row_position = nn.Parameter(torch.randn(32, config.width) * 0.02)
        row_config = VisualCellStreamConfig(
            model_dim=config.width,
            layers=config.layers,
            heads=config.heads,
            mlp_ratio=3.0,
            dropout=0.0,
        )
        self.blocks = nn.ModuleList(
            [CausalVisualBlock(row_config) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.width)
        self.pixel_density = ConditionalRowMADE(config.width, config.pixel_hidden)

    def train(self, mode=True):
        super().train(mode)
        self.reader.eval()
        return self

    @torch.no_grad()
    def read(self, context: torch.Tensor):
        states = self.reader.language(context)
        return states["hidden_states"][:, -1].float(), states["anchor_fields"][
            :, -1
        ].float()

    def plan(self, fields: torch.Tensor):
        if fields.ndim != 2 or fields.shape[1] != 1024:
            raise ValueError("image fields must be [B,1024]")
        return self.reader.field.signed_spatial(fields).mul(32).clamp(-2, 2)[:, 0]

    def row_states(self, rows, hidden, fields):
        if rows.ndim != 3 or rows.shape[-1] != 32 or not 1 <= rows.shape[1] <= 32:
            raise ValueError("row history must be [B,R<=32,32]")
        plan = self.plan(fields)[:, : rows.shape[1]]
        shifted = F.pad(rows[:, :-1].mul(2).sub(1), (0, 0, 1, 0))
        condition = self.condition(torch.cat((hidden.float(), fields.float()), dim=-1))
        states = self.previous_row(shifted) + self.plan_row(plan)
        states = states + self.row_position[None, : len(plan[0])] + condition[:, None]
        for block in self.blocks:
            states = block(states)
        return self.norm(states), plan

    def logits(self, images, hidden, fields):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 32, 32):
            raise ValueError("raster density images must be [B,1,32,32]")
        states, plan = self.row_states(images[:, 0], hidden, fields)
        residual = self.pixel_density(images[:, 0], states)
        return (self.config.plan_logit_scale * plan + residual).float()[:, None]

    def image_nll(self, images, hidden, fields):
        if not bool(((images == 0) | (images == 1)).all()):
            raise ValueError("exact Bernoulli image likelihood requires binary targets")
        logits = self.logits(images, hidden, fields)
        return (
            F.binary_cross_entropy_with_logits(logits, images.float(), reduction="none")
            .flatten(1)
            .sum(1)
        )

    @torch.no_grad()
    def sample_state(self, hidden, fields, *, generator, greedy=False):
        batch = len(hidden)
        rows = torch.zeros(batch, 32, 32, device=hidden.device)
        for row in range(32):
            states, plan = self.row_states(rows[:, : row + 1], hidden, fields)
            condition = states[:, -1]
            bias = self.config.plan_logit_scale * plan[:, -1]
            for column in range(32):
                logit = bias + self.pixel_density(rows[:, row], condition)
                probability = logit[:, column].float().sigmoid()
                if greedy:
                    rows[:, row, column] = (probability >= 0.5).float()
                else:
                    rows[:, row, column] = (
                        torch.rand(batch, device=rows.device, generator=generator)
                        < probability
                    ).float()
        return rows[:, None]

    @torch.no_grad()
    def sample_next(self, context, *, generator, greedy=False, conditioned=True):
        hidden, fields = self.read(context)
        if not conditioned:
            hidden, fields = torch.zeros_like(hidden), torch.zeros_like(fields)
        return self.sample_state(hidden, fields, generator=generator, greedy=greedy)

    def boundary_receipt(self):
        return {
            "architecture": V54_ARCHITECTURE,
            "config": asdict(self.config),
            "density_parameters": sum(
                p.numel()
                for name, p in self.named_parameters()
                if not name.startswith("reader.")
            ),
            "runtime_parameters": sum(p.numel() for p in self.parameters()),
            "text_tokenizer": False,
            "symbol_ids": False,
            "glyph_inventory": False,
            "codebook": False,
            "ocr": False,
            "external_runtime_model": False,
            "probability": "normalized product of 1024 causal Bernoulli image measurements",
            "nll_units": "nats per image; divide by 1024*log(2) for bits per pixel",
            "binary_sensor_space_size_log2": 1024,
            "reader_frozen": not any(p.requires_grad for p in self.reader.parameters()),
        }


def nats_to_bits_per_pixel(nll):
    return nll / (1024 * math.log(2))
