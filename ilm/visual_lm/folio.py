from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FolioRetinaConfig:
    image_height: int = 192
    image_width: int = 768
    model_dim: int = 256
    layers: int = 8
    heads: int = 8
    mlp_ratio: float = 3.0
    output_dim: int = 1024
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("folio dimensions must be divisible by 16")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.layers < 2:
            raise ValueError("the folio retina requires at least two axial blocks")

    @property
    def grid_height(self) -> int:
        return self.image_height // 16

    @property
    def grid_width(self) -> int:
        return self.image_width // 16


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.epsilon)
        return x * scale.to(x.dtype) * self.weight


class ConvNeXtInkBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.expand = nn.Conv2d(channels, channels * 3, 1)
        self.contract = nn.Conv2d(channels * 3, channels, 1)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.contract(F.silu(self.expand(x)))
        return residual + self.scale * x


class InkPatchStem(nn.Module):
    """Preserve thin strokes while reducing a page to a stride-16 field."""

    def __init__(self, dimension: int):
        super().__init__()
        widths = (32, 64, 128, dimension)
        layers: list[nn.Module] = []
        incoming = 1
        for width in widths:
            layers.extend(
                (
                    nn.Conv2d(incoming, width, 3, stride=2, padding=1),
                    nn.GroupNorm(1, width),
                    nn.SiLU(),
                    ConvNeXtInkBlock(width),
                )
            )
            incoming = width
        self.layers = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def _rotary_1d(x: torch.Tensor) -> torch.Tensor:
    """Apply parameter-free position to [batch, heads, length, head_dim]."""

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
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(x.dtype)[None, None]
    sine = angles.sin().to(x.dtype)[None, None]
    rotated = x[..., :rotary_dimension]
    first, second = rotated[..., 0::2], rotated[..., 1::2]
    rotated = torch.stack((first * cosine - second * sine, first * sine + second * cosine), dim=-1)
    return torch.cat((rotated.flatten(-2), x[..., rotary_dimension:]), dim=-1)


class AxialAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dimension // heads
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = _rotary_1d(query.transpose(1, 2))
        key = _rotary_1d(key.transpose(1, 2))
        value = value.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.output(attended.transpose(1, 2).reshape(batch, length, dimension))


class AxialFolioBlock(nn.Module):
    """Read along writing lines, then exchange evidence between lines."""

    def __init__(self, config: FolioRetinaConfig):
        super().__init__()
        dimension = config.model_dim
        hidden = int(dimension * config.mlp_ratio)
        self.row_norm = RMSNorm(dimension)
        self.row_attention = AxialAttention(dimension, config.heads, config.dropout)
        self.column_norm = RMSNorm(dimension)
        self.column_attention = AxialAttention(dimension, config.heads, config.dropout)
        self.mlp_norm = RMSNorm(dimension)
        self.gate = nn.Linear(dimension, hidden, bias=False)
        self.value = nn.Linear(dimension, hidden, bias=False)
        self.down = nn.Linear(hidden, dimension, bias=False)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        batch, rows, columns, dimension = field.shape
        row_input = self.row_norm(field).reshape(batch * rows, columns, dimension)
        field = field + self.row_attention(row_input).reshape(batch, rows, columns, dimension)
        column_input = self.column_norm(field).transpose(1, 2).reshape(batch * columns, rows, dimension)
        column_update = self.column_attention(column_input).reshape(batch, columns, rows, dimension)
        field = field + column_update.transpose(1, 2)
        normalized = self.mlp_norm(field)
        return field + self.down(F.silu(self.gate(normalized)) * self.value(normalized))


class InkAwarePool(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dimension) * 0.02)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Sequential(
            RMSNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, field: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        batch, rows, columns, dimension = field.shape
        flat = field.reshape(batch, rows * columns, dimension)
        scores = torch.einsum("bnd,d->bn", self.key(flat), self.query) / math.sqrt(dimension)
        prior = occupancy.reshape(batch, rows * columns).clamp_min(1e-4).log()
        attention = (scores + 0.25 * prior).softmax(dim=-1)
        pooled = torch.einsum("bn,bnd->bd", attention, self.value(flat))
        return pooled + self.output(pooled)


class FolioRetina(nn.Module):
    """Map a writing image to a continuous semantic field without OCR or IDs."""

    def __init__(self, config: FolioRetinaConfig):
        super().__init__()
        self.config = config
        self.stem = InkPatchStem(config.model_dim)
        self.row_position = nn.Parameter(torch.randn(config.grid_height, config.model_dim) * 0.01)
        self.column_position = nn.Parameter(torch.randn(config.grid_width, config.model_dim) * 0.01)
        self.blocks = nn.ModuleList([AxialFolioBlock(config) for _ in range(config.layers)])
        self.pool = InkAwarePool(config.model_dim)
        self.projection = nn.Sequential(
            RMSNorm(config.model_dim),
            nn.Linear(config.model_dim, config.model_dim * 2),
            nn.SiLU(),
            nn.Linear(config.model_dim * 2, config.output_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(3 * max(1, self.config.layers))
        for block in self.blocks:
            block.row_attention.output.weight.data.mul_(residual_scale)
            block.column_attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(images):
            raise TypeError("FolioRetina accepts continuous image tensors only")
        expected = (1, self.config.image_height, self.config.image_width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {expected}] writing images")
        images = images.clamp(0, 1)
        occupancy = F.adaptive_avg_pool2d(images, (self.config.grid_height, self.config.grid_width))
        field = self.stem(images).permute(0, 2, 3, 1)
        field = field + self.row_position[None, :, None] + self.column_position[None, None, :]
        for block in self.blocks:
            field = block(field)
        pooled = self.pool(field, occupancy[:, 0])
        return F.normalize(self.projection(pooled).float(), dim=-1)

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def folio_distillation_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    teacher: torch.Tensor,
    *,
    contrastive_scale: torch.Tensor,
    view_weight: float = 0.20,
    relational_weight: float = 0.50,
    contrastive_weight: float = 0.20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if first.shape != second.shape or first.shape != teacher.shape:
        raise ValueError("student views and teacher fields must have matching shapes")
    teacher = F.normalize(teacher.float(), dim=-1)
    first = F.normalize(first.float(), dim=-1)
    second = F.normalize(second.float(), dim=-1)
    cosine = 1.0 - 0.5 * ((first * teacher).sum(dim=-1) + (second * teacher).sum(dim=-1))
    distillation = cosine.mean()
    view = (1.0 - (first * second).sum(dim=-1)).mean()
    teacher_relations = teacher @ teacher.transpose(0, 1)
    student_relations = 0.5 * (first @ first.transpose(0, 1) + second @ second.transpose(0, 1))
    relational = F.smooth_l1_loss(student_relations, teacher_relations)
    labels = torch.arange(first.shape[0], device=first.device)
    logits_first = contrastive_scale * first @ teacher.transpose(0, 1)
    logits_second = contrastive_scale * second @ teacher.transpose(0, 1)
    contrastive = 0.25 * (
        F.cross_entropy(logits_first, labels)
        + F.cross_entropy(logits_first.transpose(0, 1), labels)
        + F.cross_entropy(logits_second, labels)
        + F.cross_entropy(logits_second.transpose(0, 1), labels)
    )
    loss = (
        distillation
        + view_weight * view
        + relational_weight * relational
        + contrastive_weight * contrastive
    )
    with torch.no_grad():
        retrieval = (logits_first.argmax(dim=1) == labels).float().mean()
        teacher_similarity = (first * teacher).sum(dim=-1).mean()
    return loss, {
        "distillation": distillation.detach(),
        "view": view.detach(),
        "relational": relational.detach(),
        "contrastive": contrastive.detach(),
        "batch_retrieval_accuracy": retrieval.detach(),
        "teacher_cosine": teacher_similarity.detach(),
    }


def folio_config_payload(config: FolioRetinaConfig) -> dict[str, Any]:
    return asdict(config)


def folio_config_from_payload(payload: dict[str, Any]) -> FolioRetinaConfig:
    return FolioRetinaConfig(**payload)
