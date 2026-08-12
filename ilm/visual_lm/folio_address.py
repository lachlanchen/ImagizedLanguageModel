from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .folio import AxialFolioBlock, ConvNeXtInkBlock, RMSNorm


@dataclass(frozen=True)
class FolioAddressConfig:
    image_height: int = 192
    image_width: int = 768
    patch_size: int = 8
    model_dim: int = 192
    layers: int = 6
    heads: int = 6
    mlp_ratio: float = 3.0
    carriers: int = 256
    glance_slots: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("folio dimensions must be divisible by patch_size")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.layers < 2:
            raise ValueError("the interference retina requires at least two axial blocks")
        if self.carriers < 16 or self.glance_slots < 1:
            raise ValueError("the interference address is underspecified")

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size

    @property
    def output_dim(self) -> int:
        return self.carriers * 2


class LosslessInkStem(nn.Module):
    """Rearrange every input pixel into channels before learned projection."""

    def __init__(self, patch_size: int, dimension: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.project = nn.Sequential(
            nn.Conv2d(patch_size * patch_size, dimension, 1),
            nn.GroupNorm(1, dimension),
            nn.SiLU(),
        )
        self.local = nn.Sequential(ConvNeXtInkBlock(dimension), ConvNeXtInkBlock(dimension))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.local(self.project(self.unshuffle(images)))


class MultiGlancePool(nn.Module):
    """Pool several independent visual glances instead of one global bottleneck."""

    def __init__(self, dimension: int, slots: int):
        super().__init__()
        self.slots = int(slots)
        self.queries = nn.Parameter(torch.randn(slots, dimension) * 0.02)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        fused_dimension = slots * dimension
        self.fuse = nn.Sequential(
            RMSNorm(fused_dimension),
            nn.Linear(fused_dimension, dimension * 3),
            nn.SiLU(),
            nn.Linear(dimension * 3, dimension * 2),
        )

    def forward(self, field: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        batch, rows, columns, dimension = field.shape
        flat = field.reshape(batch, rows * columns, dimension)
        scores = torch.einsum("bnd,sd->bsn", self.key(flat), self.queries) / math.sqrt(dimension)
        ink_prior = occupancy.reshape(batch, rows * columns).clamp_min(1e-4).log()
        attention = (scores + 0.20 * ink_prior[:, None]).softmax(dim=-1)
        glances = torch.einsum("bsn,bnd->bsd", attention, self.value(flat))
        return self.fuse(glances.reshape(batch, self.slots * dimension))


class FolioAddressRetina(nn.Module):
    """Map continuous writing pixels to a continuous error-correcting phase address."""

    def __init__(self, config: FolioAddressConfig):
        super().__init__()
        self.config = config
        self.stem = LosslessInkStem(config.patch_size, config.model_dim)
        self.row_position = nn.Parameter(torch.randn(config.grid_height, config.model_dim) * 0.01)
        self.column_position = nn.Parameter(torch.randn(config.grid_width, config.model_dim) * 0.01)
        self.blocks = nn.ModuleList([AxialFolioBlock(config) for _ in range(config.layers)])
        self.pool = MultiGlancePool(config.model_dim, config.glance_slots)
        self.address = nn.Sequential(
            RMSNorm(config.model_dim * 2),
            nn.Linear(config.model_dim * 2, config.model_dim * 3),
            nn.SiLU(),
            nn.Linear(config.model_dim * 3, config.output_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.05)))
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
            raise TypeError("FolioAddressRetina accepts continuous image tensors only")
        expected = (1, self.config.image_height, self.config.image_width)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"expected [batch, {expected}] writing images")
        images = images.clamp(0, 1)
        occupancy = F.avg_pool2d(images, self.config.patch_size, self.config.patch_size)
        field = self.stem(images).permute(0, 2, 3, 1)
        field = field + self.row_position[None, :, None] + self.column_position[None, None, :]
        for block in self.blocks:
            field = block(field)
        raw = self.address(self.pool(field, occupancy[:, 0]))
        phase_pairs = F.normalize(raw.float().reshape(images.shape[0], self.config.carriers, 2), dim=-1)
        return phase_pairs.flatten(1) / math.sqrt(self.config.carriers)

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def make_interference_transform(
    input_dimension: int,
    carriers: int,
    *,
    seed: int,
    frequency_scale: float = 2.0,
) -> dict[str, Any]:
    """Create random Fourier carriers whose dot product preserves semantic distance."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(carriers, input_dimension, generator=generator)
    directions = F.normalize(directions, dim=-1) * (math.sqrt(input_dimension) * frequency_scale)
    bias = torch.rand(carriers, generator=generator) * (2.0 * math.pi)
    return {
        "architecture": "semantic-interference-transform-v1",
        "input_dimension": int(input_dimension),
        "carriers": int(carriers),
        "seed": int(seed),
        "frequency_scale": float(frequency_scale),
        "directions": directions,
        "bias": bias,
    }


def interference_addresses(fields: torch.Tensor, transform: dict[str, Any]) -> torch.Tensor:
    if transform.get("architecture") != "semantic-interference-transform-v1":
        raise ValueError("unsupported semantic interference transform")
    fields = F.normalize(fields.float(), dim=-1)
    directions = transform["directions"].to(fields.device, fields.dtype)
    bias = transform["bias"].to(fields.device, fields.dtype)
    if fields.shape[-1] != directions.shape[-1]:
        raise ValueError("semantic field and interference transform dimensions differ")
    phase = fields @ directions.transpose(0, 1) + bias
    pairs = torch.stack((phase.cos(), phase.sin()), dim=-1)
    return pairs.flatten(-2) / math.sqrt(directions.shape[0])


def folio_address_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    target: torch.Tensor,
    target_bank: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    contrastive_scale: torch.Tensor,
    alignment_weight: float = 0.50,
    view_weight: float = 0.10,
    relational_weight: float = 0.10,
    contrastive_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if first.shape != second.shape or first.shape != target.shape:
        raise ValueError("student views and target addresses must have matching shapes")
    first = F.normalize(first.float(), dim=-1)
    second = F.normalize(second.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    target_bank = F.normalize(target_bank.float(), dim=-1)
    alignment = (1.0 - 0.5 * ((first * target).sum(dim=-1) + (second * target).sum(dim=-1))).mean()
    view = (1.0 - (first * second).sum(dim=-1)).mean()
    target_relations = target @ target.transpose(0, 1)
    student_relations = 0.5 * (first @ first.transpose(0, 1) + second @ second.transpose(0, 1))
    relational = F.smooth_l1_loss(student_relations, target_relations)
    logits_first = contrastive_scale * first @ target_bank.transpose(0, 1)
    logits_second = contrastive_scale * second @ target_bank.transpose(0, 1)
    contrastive = 0.5 * (
        F.cross_entropy(logits_first, target_indices) + F.cross_entropy(logits_second, target_indices)
    )
    loss = (
        alignment_weight * alignment
        + view_weight * view
        + relational_weight * relational
        + contrastive_weight * contrastive
    )
    with torch.no_grad():
        prediction = logits_first.argmax(dim=1)
        retrieval = (prediction == target_indices).float().mean()
        positive = logits_first.gather(1, target_indices[:, None]).squeeze(1)
        masked = logits_first.clone()
        masked.scatter_(1, target_indices[:, None], float("-inf"))
        margin = (positive - masked.max(dim=1).values).mean() / contrastive_scale
        target_similarity = (first * target).sum(dim=-1).mean()
    return loss, {
        "alignment": alignment.detach(),
        "view": view.detach(),
        "relational": relational.detach(),
        "full_corpus_contrastive": contrastive.detach(),
        "full_corpus_retrieval_accuracy": retrieval.detach(),
        "target_cosine": target_similarity.detach(),
        "target_margin": margin.detach(),
    }


def folio_address_config_payload(config: FolioAddressConfig) -> dict[str, Any]:
    return asdict(config)


def folio_address_config_from_payload(payload: dict[str, Any]) -> FolioAddressConfig:
    return FolioAddressConfig(**payload)
