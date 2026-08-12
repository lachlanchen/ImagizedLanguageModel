from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


@dataclass(frozen=True)
class RetinalFieldConfig:
    image_size: int = 384
    peripheral_size: int = 128
    fovea_size: int = 96
    fovea_extent: int = 112
    fovea_count: int = 8
    saliency_grid: int = 12
    base_channels: int = 32
    field_dim: int = 256
    embedding_dim: int = 256
    read_steps: int = 4
    inhibition: float = 1.5

    def __post_init__(self) -> None:
        if self.image_size < 128:
            raise ValueError("image_size must be at least 128")
        if self.fovea_count < 1:
            raise ValueError("fovea_count must be positive")
        if self.saliency_grid * self.saliency_grid < self.fovea_count:
            raise ValueError("saliency grid has fewer cells than foveae")
        if self.field_dim % 8:
            raise ValueError("field_dim must be divisible by 8")


@dataclass
class RetinalReadout:
    field: torch.Tensor
    coordinates: torch.Tensor
    saliency: torch.Tensor
    attention: torch.Tensor


class ResidualRetinaBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return residual + x


class RetinaTower(nn.Module):
    """A compact shared cortex for peripheral views and full-detail foveae."""

    def __init__(self, base_channels: int, output_dim: int):
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 4, base_channels * 6)
        layers: list[nn.Module] = [nn.Conv2d(3, widths[0], 5, stride=2, padding=2)]
        for index, width in enumerate(widths):
            if index:
                layers.append(nn.Conv2d(widths[index - 1], width, 3, stride=2, padding=1))
            layers.extend((ResidualRetinaBlock(width), ResidualRetinaBlock(width)))
        self.features = nn.Sequential(*layers)
        self.projection = nn.Sequential(
            nn.LayerNorm(widths[-1]),
            nn.Linear(widths[-1], output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(-2, -1))
        return self.projection(x)


def _assert_image_tensor(images: torch.Tensor) -> None:
    if not torch.is_floating_point(images):
        raise TypeError("the visual model accepts floating-point image tensors only")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("expected image tensor shaped [batch, 3, height, width]")


def ink_energy(images: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Return image-derived ink/edge energy on a coarse retinal grid."""

    _assert_image_tensor(images)
    rgb = images.float().clamp(-1, 1).add(1).mul(0.5)
    luminance = (
        rgb[:, 0:1] * 0.299
        + rgb[:, 1:2] * 0.587
        + rgb[:, 2:3] * 0.114
    )
    darkness = (1.0 - luminance).clamp_min(0.0)
    dx = F.pad(luminance[:, :, :, 1:] - luminance[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(luminance[:, :, 1:, :] - luminance[:, :, :-1, :], (0, 0, 0, 1))
    edges = dx.abs() + dy.abs()
    energy = darkness * 0.62 + edges * 0.38
    return F.adaptive_avg_pool2d(energy, (grid_size, grid_size)).squeeze(1)


def select_foveae(
    images: torch.Tensor,
    *,
    count: int,
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose non-maximal ink locations without labels or OCR."""

    energy = ink_energy(images, grid_size)
    maxima = F.max_pool2d(energy.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    local = energy >= maxima - 1e-7
    ranked = energy + local.to(energy.dtype) * (energy.detach().amax(dim=(1, 2), keepdim=True) + 1e-4)
    _, flat_indices = ranked.flatten(1).topk(count, dim=1)
    raw_values = energy.flatten(1).gather(1, flat_indices)
    rows = torch.div(flat_indices, grid_size, rounding_mode="floor")
    columns = flat_indices.remainder(grid_size)
    x = (columns.to(images.dtype) + 0.5) * (2.0 / grid_size) - 1.0
    y = (rows.to(images.dtype) + 0.5) * (2.0 / grid_size) - 1.0
    coordinates = torch.stack((x, y), dim=-1)
    saliency = raw_values / raw_values.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return coordinates, saliency


def extract_foveal_crops(
    images: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    extent: int,
    output_size: int,
) -> torch.Tensor:
    """Sample full-detail image crops at normalized retinal coordinates."""

    batch, _, height, width = images.shape
    count = coordinates.shape[1]
    cx = (coordinates[..., 0] + 1.0) * 0.5 * width
    cy = (coordinates[..., 1] + 1.0) * 0.5 * height
    half = float(extent) * 0.5
    batch_index = torch.arange(batch, device=images.device, dtype=images.dtype)[:, None].expand(-1, count)
    boxes = torch.stack(
        (
            batch_index,
            (cx - half).clamp(0, width - 1),
            (cy - half).clamp(0, height - 1),
            (cx + half).clamp(1, width),
            (cy + half).clamp(1, height),
        ),
        dim=-1,
    ).reshape(-1, 5)
    crops = roi_align(images, boxes, output_size=(output_size, output_size), aligned=True)
    return crops.reshape(batch, count, 3, output_size, output_size)


class RetinalFieldEncoder(nn.Module):
    """Read a page through peripheral context and recurrent visual foveation."""

    def __init__(self, config: RetinalFieldConfig):
        super().__init__()
        self.config = config
        dimension = config.field_dim
        self.tower = RetinaTower(config.base_channels, dimension)
        self.coordinate = nn.Sequential(
            nn.Linear(6, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )
        self.initial_state = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, dimension))
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.update = nn.GRUCell(dimension, dimension)
        self.refine = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, dimension),
        )
        self.output_norm = nn.LayerNorm(dimension)

    @staticmethod
    def coordinate_features(coordinates: torch.Tensor) -> torch.Tensor:
        x, y = coordinates.unbind(dim=-1)
        return torch.stack(
            (x, y, torch.sin(math.pi * x), torch.cos(math.pi * x), torch.sin(math.pi * y), torch.cos(math.pi * y)),
            dim=-1,
        )

    def forward(self, images: torch.Tensor, *, return_details: bool = False) -> torch.Tensor | RetinalReadout:
        _assert_image_tensor(images)
        cfg = self.config
        coordinates, saliency = select_foveae(
            images,
            count=cfg.fovea_count,
            grid_size=cfg.saliency_grid,
        )
        crops = extract_foveal_crops(
            images,
            coordinates,
            extent=cfg.fovea_extent,
            output_size=cfg.fovea_size,
        )
        peripheral = F.interpolate(
            images,
            size=(cfg.peripheral_size, cfg.peripheral_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        batch, fovea_count = crops.shape[:2]
        global_feature = self.tower(peripheral)
        local_features = self.tower(crops.flatten(0, 1)).reshape(batch, fovea_count, -1)
        local_features = local_features + self.coordinate(self.coordinate_features(coordinates))

        state = self.initial_state(global_feature)
        coverage = torch.zeros_like(saliency)
        attention_history: list[torch.Tensor] = []
        keys = self.key(local_features)
        values = self.value(local_features)
        saliency_prior = saliency.clamp_min(1e-6).log()
        for _ in range(cfg.read_steps):
            query = self.query(state)[:, None, :]
            scores = (query * keys).sum(dim=-1) / math.sqrt(cfg.field_dim)
            scores = scores + 0.35 * saliency_prior - cfg.inhibition * coverage
            attention = scores.softmax(dim=1)
            read = (attention[..., None] * values).sum(dim=1)
            state = self.update(read, state)
            state = state + self.refine(state)
            coverage = coverage + attention.detach()
            attention_history.append(attention)
        field = self.output_norm(state)
        if not return_details:
            return field
        return RetinalReadout(
            field=field,
            coordinates=coordinates,
            saliency=saliency,
            attention=torch.stack(attention_history, dim=1),
        )


class VisualAssociativeReader(nn.Module):
    """Image-only query/answer encoder for provenance-bearing visual memory."""

    def __init__(self, config: RetinalFieldConfig):
        super().__init__()
        self.config = config
        self.retina = RetinalFieldEncoder(config)
        self.query_head = nn.Sequential(
            nn.Linear(config.field_dim, config.field_dim),
            nn.SiLU(),
            nn.Linear(config.field_dim, config.embedding_dim),
        )
        self.answer_head = nn.Sequential(
            nn.Linear(config.field_dim, config.field_dim),
            nn.SiLU(),
            nn.Linear(config.field_dim, config.embedding_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_query(self, images: torch.Tensor) -> torch.Tensor:
        field = self.retina(images)
        return F.normalize(self.project_query(field), dim=-1)

    def encode_answer(self, images: torch.Tensor) -> torch.Tensor:
        field = self.retina(images)
        return F.normalize(self.project_answer(field), dim=-1)

    def project_query(self, field: torch.Tensor) -> torch.Tensor:
        return self.query_head(field)

    def project_answer(self, field: torch.Tensor) -> torch.Tensor:
        return self.answer_head(field)

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def symmetric_info_nce(
    left: torch.Tensor,
    right: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if left.shape != right.shape:
        raise ValueError("paired contrastive fields must have matching shapes")
    logits = scale * left @ right.transpose(0, 1)
    labels = torch.arange(left.shape[0], device=left.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
    accuracy = (logits.argmax(dim=1) == labels).float().mean()
    return loss, accuracy


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    if matrix.shape != (size, size):
        raise ValueError("covariance matrix must be square")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_field_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    target_std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Keep continuous fields informative without requiring negative labels."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg fields must be paired [batch, dimensions] tensors")
    first = first.float()
    second = second.float()
    invariance = F.mse_loss(first, second)
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    first_std = torch.sqrt(first_centered.var(dim=0, unbiased=False) + 1e-4)
    second_std = torch.sqrt(second_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(target_std - first_std).mean()
        + F.relu(target_std - second_std).mean()
    )
    denominator = max(1, first.shape[0] - 1)
    first_covariance = first_centered.transpose(0, 1) @ first_centered / denominator
    second_covariance = second_centered.transpose(0, 1) @ second_centered / denominator
    dimensions = first.shape[1]
    covariance = (
        _off_diagonal(first_covariance).square().sum()
        + _off_diagonal(second_covariance).square().sum()
    ) / max(1, 2 * dimensions)
    loss = (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
    return loss, {
        "invariance": invariance.detach(),
        "variance": variance.detach(),
        "covariance": covariance.detach(),
        "field_std": (0.5 * (first_std.mean() + second_std.mean())).detach(),
    }


@dataclass(frozen=True)
class VisualMemoryHit:
    score: float
    key_index: int
    entry_index: int
    metadata: dict[str, Any]


class VisualEpisodeMemory:
    """Exact image values addressed by continuous image-derived keys."""

    INDEX_NAME = "visual_memory.pt"
    MANIFEST_NAME = "memory_manifest.jsonl"

    def __init__(
        self,
        keys: torch.Tensor,
        entry_indices: torch.Tensor,
        entries: Sequence[dict[str, Any]],
        *,
        root: Path,
    ):
        if keys.ndim != 2 or not torch.is_floating_point(keys):
            raise ValueError("memory keys must be a floating [keys, dimensions] tensor")
        if entry_indices.shape != (keys.shape[0],):
            raise ValueError("entry_indices must identify every key")
        self.keys = F.normalize(keys.float().cpu(), dim=-1)
        self.entry_indices = entry_indices.long().cpu()
        self.entries = list(entries)
        self.root = Path(root)

    @classmethod
    def load(cls, root: str | Path) -> "VisualEpisodeMemory":
        root = Path(root)
        payload = torch.load(root / cls.INDEX_NAME, map_location="cpu", weights_only=True)
        entries = [
            json.loads(line)
            for line in (root / cls.MANIFEST_NAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(payload["keys"], payload["entry_indices"], entries, root=root)

    def search(self, query: torch.Tensor, *, top_k: int = 5) -> list[list[VisualMemoryHit]]:
        if query.ndim != 2 or query.shape[1] != self.keys.shape[1]:
            raise ValueError("query embedding has the wrong shape")
        scores = F.normalize(query.float().cpu(), dim=-1) @ self.keys.transpose(0, 1)
        key_count = min(self.keys.shape[0], max(top_k * 4, top_k))
        values, indices = scores.topk(key_count, dim=1)
        batches: list[list[VisualMemoryHit]] = []
        for row_values, row_indices in zip(values, indices):
            seen: set[int] = set()
            hits: list[VisualMemoryHit] = []
            for score, key_index_tensor in zip(row_values.tolist(), row_indices.tolist()):
                entry_index = int(self.entry_indices[key_index_tensor])
                if entry_index in seen:
                    continue
                seen.add(entry_index)
                hits.append(
                    VisualMemoryHit(
                        score=float(score),
                        key_index=int(key_index_tensor),
                        entry_index=entry_index,
                        metadata=self.entries[entry_index],
                    )
                )
                if len(hits) >= top_k:
                    break
            batches.append(hits)
        return batches

    def answer_path(self, hit: VisualMemoryHit) -> Path:
        path = Path(str(hit.metadata["answer_image"]))
        return path if path.is_absolute() else self.root / path


def retinal_config_from_payload(payload: dict[str, Any]) -> RetinalFieldConfig:
    return RetinalFieldConfig(**payload)


def retinal_config_payload(config: RetinalFieldConfig) -> dict[str, Any]:
    return asdict(config)
