from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .folio import AxialFolioBlock, ConvNeXtInkBlock, RMSNorm


@dataclass(frozen=True)
class InkJEPAConfig:
    image_height: int = 192
    image_width: int = 768
    patch_size: int = 8
    model_dim: int = 128
    representation_dim: int = 128
    encoder_layers: int = 5
    predictor_dim: int = 128
    predictor_layers: int = 2
    heads: int = 4
    mlp_ratio: float = 3.0
    pool_slots: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError("retinal dimensions must be divisible by patch_size")
        if self.model_dim % self.heads or self.predictor_dim % self.heads:
            raise ValueError("encoder and predictor dimensions must be divisible by heads")
        if self.encoder_layers < 2 or self.predictor_layers < 1:
            raise ValueError("InkJEPA needs at least two encoder and one predictor layers")
        if self.representation_dim < 32 or self.pool_slots < 1:
            raise ValueError("InkJEPA representation is underspecified")

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size


@dataclass(frozen=True)
class _AxialConfig:
    model_dim: int
    heads: int
    mlp_ratio: float
    dropout: float


class LosslessRetinalStem(nn.Module):
    """Move every pixel into channels before any learned spatial reduction."""

    def __init__(self, patch_size: int, dimension: int):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(patch_size)
        self.project = nn.Sequential(
            nn.Conv2d(patch_size * patch_size, dimension, 1),
            nn.GroupNorm(1, dimension),
            nn.SiLU(),
        )
        self.local = nn.Sequential(ConvNeXtInkBlock(dimension), ConvNeXtInkBlock(dimension))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.local(self.project(self.unshuffle(image)))


class RetinalPagePool(nn.Module):
    """Integrate several image-derived glances into one continuous page state."""

    def __init__(self, dimension: int, output_dimension: int, slots: int):
        super().__init__()
        self.slots = int(slots)
        self.queries = nn.Parameter(torch.randn(slots, dimension) * 0.02)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Sequential(
            RMSNorm(slots * dimension),
            nn.Linear(slots * dimension, dimension * 2),
            nn.SiLU(),
            nn.Linear(dimension * 2, output_dimension),
        )

    def forward(self, field: torch.Tensor, occupancy: torch.Tensor) -> torch.Tensor:
        batch, rows, columns, dimension = field.shape
        flat = field.reshape(batch, rows * columns, dimension)
        scores = torch.einsum("bnd,sd->bsn", self.key(flat), self.queries) / math.sqrt(dimension)
        ink_prior = occupancy.reshape(batch, rows * columns).clamp_min(1e-3).log()
        attention = (scores + 0.12 * ink_prior[:, None]).softmax(dim=-1)
        glances = torch.einsum("bsn,bnd->bsd", attention, self.value(flat))
        return self.output(glances.flatten(1))


class RetinalFieldEncoder(nn.Module):
    """Continuous image encoder used by both online and EMA target branches."""

    def __init__(self, config: InkJEPAConfig):
        super().__init__()
        self.config = config
        self.stem = LosslessRetinalStem(config.patch_size, config.model_dim)
        self.hidden_mark = nn.Parameter(torch.randn(config.model_dim) * 0.02)
        self.row_position = nn.Parameter(torch.randn(config.grid_height, config.model_dim) * 0.01)
        self.column_position = nn.Parameter(torch.randn(config.grid_width, config.model_dim) * 0.01)
        axial = _AxialConfig(config.model_dim, config.heads, config.mlp_ratio, config.dropout)
        self.blocks = nn.ModuleList([AxialFolioBlock(axial) for _ in range(config.encoder_layers)])
        self.norm = RMSNorm(config.model_dim)
        self.local_projection = nn.Linear(config.model_dim, config.representation_dim, bias=False)
        self.pool = RetinalPagePool(
            config.model_dim,
            config.representation_dim,
            config.pool_slots,
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = 1.0 / math.sqrt(3 * max(1, self.config.encoder_layers))
        for block in self.blocks:
            block.row_attention.output.weight.data.mul_(residual_scale)
            block.column_attention.output.weight.data.mul_(residual_scale)
            block.down.weight.data.mul_(residual_scale)

    def forward(
        self,
        image: torch.Tensor,
        *,
        hidden_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expected = (1, self.config.image_height, self.config.image_width)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"expected continuous ink image [batch, {expected}]")
        if not torch.is_floating_point(image):
            raise TypeError("RetinalFieldEncoder accepts continuous image tensors only")
        image = image.clamp(0, 1)
        occupancy = F.avg_pool2d(image, self.config.patch_size, self.config.patch_size)[:, 0]
        field = self.stem(image).permute(0, 2, 3, 1)
        field = field + self.row_position[None, :, None] + self.column_position[None, None, :]
        if hidden_mask is not None:
            if hidden_mask.shape != field.shape[:3]:
                raise ValueError("hidden_mask must match the retinal field grid")
            field = field + hidden_mask.to(field.dtype)[..., None] * self.hidden_mark
        for block in self.blocks:
            field = block(field)
        normalized = self.norm(field)
        return {
            "field": field,
            "local": self.local_projection(normalized),
            "page": self.pool(normalized, occupancy),
            "occupancy": occupancy,
        }


class RetinalFieldPredictor(nn.Module):
    def __init__(self, config: InkJEPAConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.model_dim, config.predictor_dim, bias=False)
        self.hidden_query = nn.Parameter(torch.randn(config.predictor_dim) * 0.02)
        axial = _AxialConfig(config.predictor_dim, config.heads, config.mlp_ratio, config.dropout)
        self.blocks = nn.ModuleList([AxialFolioBlock(axial) for _ in range(config.predictor_layers)])
        self.norm = RMSNorm(config.predictor_dim)
        self.local = nn.Linear(config.predictor_dim, config.representation_dim, bias=False)
        self.ink = nn.Linear(config.predictor_dim, config.patch_size * config.patch_size)

    def forward(self, field: torch.Tensor, hidden_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.input(field)
        prediction = prediction + hidden_mask.to(prediction.dtype)[..., None] * self.hidden_query
        for block in self.blocks:
            prediction = block(prediction)
        prediction = self.norm(prediction)
        return {"local": self.local(prediction), "ink_logits": self.ink(prediction)}


class InkJEPA(nn.Module):
    """Image-only joint-embedding predictive model for written-language fields."""

    def __init__(self, config: InkJEPAConfig):
        super().__init__()
        self.config = config
        self.online_encoder = RetinalFieldEncoder(config)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.predictor = RetinalFieldPredictor(config)
        self.page_predictor = nn.Sequential(
            RMSNorm(config.representation_dim),
            nn.Linear(config.representation_dim, config.representation_dim * 2),
            nn.SiLU(),
            nn.Linear(config.representation_dim * 2, config.representation_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.08)))

    def train(self, mode: bool = True) -> "InkJEPA":
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(
        self,
        context_image: torch.Tensor,
        target_image: torch.Tensor,
        hidden_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        online = self.online_encoder(context_image, hidden_mask=hidden_mask)
        predicted = self.predictor(online["field"], hidden_mask)
        with torch.no_grad():
            target = self.target_encoder(target_image)
        return {
            "predicted_local": predicted["local"],
            "predicted_ink": predicted["ink_logits"],
            "predicted_page": self.page_predictor(online["page"]),
            "online_local": online["local"],
            "target_local": target["local"],
            "target_page": target["page"],
        }

    @property
    def contrastive_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("EMA momentum must be in [0, 1]")
        online_parameters = dict(self.online_encoder.named_parameters())
        for name, target in self.target_encoder.named_parameters():
            target.lerp_(online_parameters[name], 1.0 - momentum)
        online_buffers = dict(self.online_encoder.named_buffers())
        for name, target in self.target_encoder.named_buffers():
            target.copy_(online_buffers[name])


def sample_predictive_masks(
    batch_size: int,
    grid_height: int,
    grid_width: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix semantic block, future-line, and line-suffix occlusions."""

    masks = torch.zeros(batch_size, grid_height, grid_width, dtype=torch.bool, device=device)
    modes = torch.randint(0, 3, (batch_size,), device=device, generator=generator)

    def randint(low: int, high: int) -> int:
        return int(torch.randint(low, max(low + 1, high), (), device=device, generator=generator))

    for batch_index, mode_tensor in enumerate(modes):
        mode = int(mode_tensor)
        if mode == 0:
            target_fraction = 0.38 + 0.16 * float(torch.rand((), device=device, generator=generator))
            attempts = 0
            while float(masks[batch_index].float().mean()) < target_fraction and attempts < 8:
                height = randint(max(2, grid_height // 5), max(3, grid_height // 2 + 1))
                width = randint(max(3, grid_width // 8), max(4, grid_width // 3 + 1))
                top = randint(0, grid_height - height + 1)
                left = randint(0, grid_width - width + 1)
                masks[batch_index, top : top + height, left : left + width] = True
                attempts += 1
        elif mode == 1:
            cut = randint(max(2, grid_height * 2 // 5), max(3, grid_height * 3 // 4))
            masks[batch_index, cut:] = True
        else:
            cut = randint(max(3, grid_width * 2 // 5), max(4, grid_width * 3 // 4))
            masks[batch_index, :, cut:] = True
    return masks, modes


def hide_retinal_regions(image: torch.Tensor, hidden_mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    pixel_mask = hidden_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return image.masked_fill(pixel_mask[:, None], 0.0)


def _variance_covariance(
    features: torch.Tensor,
    *,
    maximum_samples: int = 4096,
    variance_floor: float = 0.75,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = features.reshape(-1, features.shape[-1]).float()
    if features.shape[0] > maximum_samples:
        step = max(1, features.shape[0] // maximum_samples)
        features = features[::step][:maximum_samples]
    features = features - features.mean(dim=0, keepdim=True)
    standard_deviation = torch.sqrt(features.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(variance_floor - standard_deviation).mean()
    if features.shape[0] < 2:
        covariance = features.new_zeros(())
    else:
        covariance_matrix = features.transpose(0, 1) @ features / (features.shape[0] - 1)
        covariance_matrix.fill_diagonal_(0.0)
        covariance = covariance_matrix.square().sum() / features.shape[1]
    return variance, covariance, standard_deviation.mean()


def ink_jepa_loss(
    outputs: dict[str, torch.Tensor],
    source_image: torch.Tensor,
    hidden_mask: torch.Tensor,
    *,
    patch_size: int,
    contrastive_scale: torch.Tensor,
    field_weight: float = 1.0,
    page_weight: float = 0.35,
    ink_weight: float = 0.12,
    variance_weight: float = 0.08,
    covariance_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_local = outputs["predicted_local"][hidden_mask]
    target_local = outputs["target_local"][hidden_mask].detach()
    if predicted_local.numel() == 0:
        raise ValueError("predictive mask selected no retinal cells")
    normalized_prediction = F.normalize(predicted_local.float(), dim=-1)
    normalized_target = F.normalize(target_local.float(), dim=-1)
    field = (1.0 - (normalized_prediction * normalized_target).sum(dim=-1)).mean()

    predicted_page = F.normalize(outputs["predicted_page"].float(), dim=-1)
    target_page = F.normalize(outputs["target_page"].detach().float(), dim=-1)
    labels = torch.arange(predicted_page.shape[0], device=predicted_page.device)
    page_logits = contrastive_scale * predicted_page @ target_page.transpose(0, 1)
    page = F.cross_entropy(page_logits, labels)

    ink_patches = F.pixel_unshuffle(source_image.float(), patch_size).permute(0, 2, 3, 1)
    target_ink = ink_patches[hidden_mask]
    predicted_ink = outputs["predicted_ink"][hidden_mask].float()
    positive_weight = ((1.0 - target_ink).sum() / target_ink.sum().clamp_min(1.0)).clamp(1.0, 12.0)
    ink = F.binary_cross_entropy_with_logits(predicted_ink, target_ink, pos_weight=positive_weight)

    spread_features = torch.cat(
        (
            F.layer_norm(predicted_local.float(), (predicted_local.shape[-1],)),
            F.layer_norm(outputs["online_local"].float().reshape(-1, predicted_local.shape[-1]),
                         (predicted_local.shape[-1],)),
        ),
        dim=0,
    )
    variance, covariance, feature_std = _variance_covariance(spread_features)
    total = (
        field_weight * field
        + page_weight * page
        + ink_weight * ink
        + variance_weight * variance
        + covariance_weight * covariance
    )

    with torch.no_grad():
        page_accuracy = (page_logits.argmax(dim=1) == labels).float().mean()
        ink_binary = predicted_ink.sigmoid() >= 0.5
        target_binary = target_ink >= 0.5
        true_positive = (ink_binary & target_binary).sum().float()
        ink_f1 = 2.0 * true_positive / (ink_binary.sum() + target_binary.sum()).clamp_min(1)
        target_std = F.layer_norm(target_local.float(), (target_local.shape[-1],)).std(dim=0).mean()
    return total, {
        "field_cosine_loss": field.detach(),
        "page_contrastive": page.detach(),
        "page_retrieval_accuracy": page_accuracy.detach(),
        "ink_bce": ink.detach(),
        "ink_f1": ink_f1.detach(),
        "variance_penalty": variance.detach(),
        "covariance_penalty": covariance.detach(),
        "student_feature_std": feature_std.detach(),
        "target_feature_std": target_std.detach(),
        "mask_fraction": hidden_mask.float().mean().detach(),
    }


def ink_jepa_config_payload(config: InkJEPAConfig) -> dict[str, Any]:
    return asdict(config)


def ink_jepa_config_from_payload(payload: dict[str, Any]) -> InkJEPAConfig:
    return InkJEPAConfig(**payload)
