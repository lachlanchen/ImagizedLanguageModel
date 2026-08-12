from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InkStreamConfig:
    ribbon_height: int = 48
    strip_width: int = 8
    maximum_strips: int = 256
    model_dim: int = 256
    layers: int = 8
    heads: int = 8
    mlp_ratio: float = 3.0
    dropout: float = 0.0
    local_motor_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.ribbon_height < 24 or self.strip_width < 2:
            raise ValueError("ink strips are too small to preserve writing")
        if self.maximum_strips < 16:
            raise ValueError("maximum_strips must be at least 16")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")

    @property
    def strip_pixels(self) -> int:
        return self.ribbon_height * self.strip_width


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.epsilon).to(x.dtype)
        return normalized * self.weight


def apply_rotary(x: torch.Tensor) -> torch.Tensor:
    """Apply parameter-free 1D rotary position to [B, heads, T, D]."""

    length = x.shape[-2]
    dimension = x.shape[-1]
    rotary_dimension = dimension - dimension % 2
    if rotary_dimension == 0:
        return x
    positions = torch.arange(length, device=x.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(0, rotary_dimension, 2, device=x.device, dtype=torch.float32)
        / rotary_dimension
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos().to(x.dtype)[None, None]
    sine = angles.sin().to(x.dtype)[None, None]
    rotary = x[..., :rotary_dimension]
    first = rotary[..., 0::2]
    second = rotary[..., 1::2]
    rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1)
    rotated = rotated.flatten(-2)
    return torch.cat((rotated, x[..., rotary_dimension:]), dim=-1)


class CausalInkAttention(nn.Module):
    def __init__(self, config: InkStreamConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.model_dim // config.heads
        self.qkv = nn.Linear(config.model_dim, config.model_dim * 3, bias=False)
        self.output = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.dropout = float(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = apply_rotary(query.transpose(1, 2))
        key = apply_rotary(key.transpose(1, 2))
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return self.output(attended)


class InkStreamBlock(nn.Module):
    def __init__(self, config: InkStreamConfig):
        super().__init__()
        hidden = int(config.model_dim * config.mlp_ratio)
        self.attention_norm = RMSNorm(config.model_dim)
        self.attention = CausalInkAttention(config)
        self.mlp_norm = RMSNorm(config.model_dim)
        self.gate = nn.Linear(config.model_dim, hidden, bias=False)
        self.value = nn.Linear(config.model_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, config.model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        normalized = self.mlp_norm(x)
        x = x + self.down(F.silu(self.gate(normalized)) * self.value(normalized))
        return x


class InkStreamLM(nn.Module):
    """Causal language dynamics over continuous columns of visible ink.

    The model accepts floating image strips shaped [B, T, 1, H, W]. It has no
    vocabulary, embedding lookup, Unicode IDs, or classification head.
    """

    def __init__(self, config: InkStreamConfig):
        super().__init__()
        self.config = config
        pixels = config.strip_pixels
        self.retina = nn.Sequential(
            nn.LayerNorm(pixels),
            nn.Linear(pixels, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.blocks = nn.ModuleList([InkStreamBlock(config) for _ in range(config.layers)])
        self.output_norm = RMSNorm(config.model_dim)
        self.motor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, pixels),
        )
        self.local_motor = nn.Linear(pixels, pixels, bias=False)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(2 * max(1, self.config.layers))
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def forward(self, strips: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(strips):
            raise TypeError("InkStream accepts floating image strips only")
        expected = (1, self.config.ribbon_height, self.config.strip_width)
        if strips.ndim != 5 or tuple(strips.shape[2:]) != expected:
            raise ValueError(f"expected [batch, time, {expected}] visual strips")
        if strips.shape[1] > self.config.maximum_strips:
            raise ValueError("visual sequence exceeds configured maximum")
        flat = strips.flatten(2)
        state = self.retina(flat)
        for block in self.blocks:
            state = block(state)
        state = self.output_norm(state)
        return self.motor(state) + self.config.local_motor_gain * self.local_motor(flat)

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        *,
        maximum_new_strips: int,
        threshold: float = 0.5,
        temperature: float = 1.0,
        stochastic: bool = False,
        feedback_mode: str = "soft",
    ) -> torch.Tensor:
        if prefix.ndim != 5 or prefix.shape[0] != 1:
            raise ValueError("generation currently requires one visual stream")
        sequence = prefix
        for _ in range(maximum_new_strips):
            context = sequence[:, -self.config.maximum_strips :]
            logits = self(context)[:, -1] / max(temperature, 1e-4)
            probability = logits.sigmoid()
            if stochastic:
                next_flat = torch.bernoulli(probability)
            elif feedback_mode == "soft":
                next_flat = probability
            elif feedback_mode == "hard":
                next_flat = (probability >= threshold).to(probability.dtype)
            else:
                raise ValueError("feedback_mode must be soft or hard")
            next_strip = next_flat.reshape(
                1,
                1,
                1,
                self.config.ribbon_height,
                self.config.strip_width,
            )
            sequence = torch.cat((sequence, next_strip), dim=1)
        return sequence


def ink_stream_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sequence_weight: torch.Tensor,
    *,
    ink_weight: float = 4.0,
    edge_weight: float = 0.15,
    density_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_flat = target.flatten(2)
    pixel_weight = 1.0 + target_flat * ink_weight
    valid_weight = sequence_weight[..., None]
    bce = F.binary_cross_entropy_with_logits(logits.float(), target_flat.float(), reduction="none")
    reconstruction = (bce * pixel_weight * valid_weight).sum() / (
        (pixel_weight * valid_weight).sum().clamp_min(1.0)
    )
    prediction = logits.sigmoid().reshape_as(target)
    pred_dx = prediction[..., 1:] - prediction[..., :-1]
    target_dx = target[..., 1:] - target[..., :-1]
    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    strip_weight = sequence_weight[:, :, None, None, None]
    edge = (
        ((pred_dx - target_dx).abs() * strip_weight).sum() / strip_weight.sum().clamp_min(1.0) / pred_dx[0, 0].numel()
        + ((pred_dy - target_dy).abs() * strip_weight).sum() / strip_weight.sum().clamp_min(1.0) / pred_dy[0, 0].numel()
    )
    prediction_density = prediction.mean(dim=(-3, -2, -1))
    target_density = target.mean(dim=(-3, -2, -1))
    density = (((prediction_density - target_density).square()) * sequence_weight).sum() / sequence_weight.sum().clamp_min(1.0)
    loss = reconstruction + edge_weight * edge + density_weight * density
    with torch.no_grad():
        hard = prediction >= 0.5
        truth = target >= 0.5
        valid = sequence_weight[:, :, None, None, None] > 0
        true_positive = (hard & truth & valid).sum()
        precision = true_positive.float() / ((hard & valid).sum().float().clamp_min(1.0))
        recall = true_positive.float() / ((truth & valid).sum().float().clamp_min(1.0))
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return loss, {
        "bce": reconstruction.detach(),
        "edge": edge.detach(),
        "density": density.detach(),
        "predicted_ink_density": (
            (prediction_density * sequence_weight).sum() / sequence_weight.sum().clamp_min(1.0)
        ).detach(),
        "target_ink_density": (
            (target_density * sequence_weight).sum() / sequence_weight.sum().clamp_min(1.0)
        ).detach(),
        "ink_precision": precision.detach(),
        "ink_recall": recall.detach(),
        "ink_f1": f1.detach(),
    }


def ink_stream_config_payload(config: InkStreamConfig) -> dict[str, Any]:
    return asdict(config)


def ink_stream_config_from_payload(payload: dict[str, Any]) -> InkStreamConfig:
    return InkStreamConfig(**payload)
