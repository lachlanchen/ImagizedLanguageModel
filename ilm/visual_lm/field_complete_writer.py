from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retinal_topology_router import (
    ChannelLayerNorm2d,
    block_downsample,
    block_upsample,
    quadrant_field_mask,
)
from .spatial_motor_plan import visual_complexity_masks, visual_complexity_score
from .visual_actuator import (
    StyleImageEncoder,
    multi_positive_nce,
    visual_actuator_retrieval_metrics,
    visual_positive_mask,
)


FIELD_COMPLETE_ROUTE = "field_complete"
TILED_GLOBAL_CONTROL_ROUTE = "tiled_global_control"
ROUTE_MODES = (FIELD_COMPLETE_ROUTE, TILED_GLOBAL_CONTROL_ROUTE)


class SpatialVisualRetina(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...

    def forward_with_field(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class FieldCompleteWriterConfig:
    """Decode complete output patches from corresponding retinal cells."""

    fovea_size: int = 32
    field_size: int = 4
    visual_dim: int = 192
    spatial_channels: int = 192
    style_dim: int = 64
    style_base_channels: int = 32
    hidden_channels: int = 128
    context_dim: int = 128
    pointwise_blocks: int = 3
    dropout: float = 0.05
    route_mode: str = FIELD_COMPLETE_ROUTE

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % self.field_size:
            raise ValueError("fovea_size must be divisible by field_size")
        if self.field_size < 2:
            raise ValueError("field_size must be at least two")
        if self.visual_dim < 32 or self.spatial_channels < 32:
            raise ValueError("field-complete condition is underspecified")
        if self.visual_dim != self.spatial_channels:
            raise ValueError(
                "visual_dim must equal spatial_channels for the tiled control"
            )
        if self.style_dim < 16 or self.style_base_channels < 8:
            raise ValueError("style condition is underspecified")
        if self.hidden_channels < 32 or self.context_dim < 32:
            raise ValueError("field-complete decoder is underspecified")
        if self.pointwise_blocks < 1:
            raise ValueError("at least one pointwise block is required")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"route_mode must be one of {ROUTE_MODES}")
        patch_area = self.patch_size**2
        if patch_area & (patch_area - 1):
            raise ValueError("patch area must be a power of two")

    @property
    def patch_size(self) -> int:
        return self.fovea_size // self.field_size

    @property
    def patch_area(self) -> int:
        return self.patch_size**2


def sylvester_hadamard(order: int) -> torch.Tensor:
    """Return a Sylvester Hadamard matrix with exactly representable entries."""

    if order < 2 or order & (order - 1):
        raise ValueError("Hadamard order must be a power of two and at least two")
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.shape[0] < order:
        top = torch.cat((matrix, matrix), dim=1)
        bottom = torch.cat((matrix, -matrix), dim=1)
        matrix = torch.cat((top, bottom), dim=0)
    return matrix


def zero_dc_hadamard_basis(patch_area: int) -> torch.Tensor:
    """Return all orthonormal Walsh modes except the constant DC column."""

    hadamard = sylvester_hadamard(patch_area)
    return hadamard[:, 1:] / float(patch_area**0.5)


class SharedGlobalConditioner(nn.Module):
    """Create one non-spatial context shared by every retinal cell."""

    def __init__(self, config: FieldCompleteWriterConfig):
        super().__init__()
        input_dim = config.visual_dim + config.style_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.context_dim),
            nn.SiLU(),
            nn.Linear(config.context_dim, config.context_dim),
            nn.LayerNorm(config.context_dim),
        )

    def forward(
        self,
        intended_visual: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(torch.cat((intended_visual, style), dim=-1))


class PointwiseGlobalBlock(nn.Module):
    """Transform cells independently under one spatially uniform context."""

    def __init__(
        self,
        channels: int,
        context_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = ChannelLayerNorm2d(channels)
        self.input = nn.Conv2d(channels, channels * 2, 1)
        self.context = nn.Linear(context_dim, channels * 4)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(channels * 2, channels, 1)

    def forward(
        self,
        field: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input(F.silu(self.norm(field)))
        scale, shift = self.context(F.silu(context)).chunk(2, dim=-1)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.output(self.dropout(F.silu(hidden)))
        return field + hidden


class FieldCompleteWriter(nn.Module):
    """Write each output patch from its corresponding continuous source cell."""

    def __init__(self, config: FieldCompleteWriterConfig):
        super().__init__()
        self.config = config
        self.style_encoder = StyleImageEncoder(config)
        self.global_conditioner = SharedGlobalConditioner(config)
        self.source_norm = ChannelLayerNorm2d(config.spatial_channels)
        self.source_projection = nn.Conv2d(
            config.spatial_channels,
            config.hidden_channels,
            1,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            PointwiseGlobalBlock(
                config.hidden_channels,
                config.context_dim,
                config.dropout,
            )
            for _ in range(config.pointwise_blocks)
        )
        self.output_norm = ChannelLayerNorm2d(config.hidden_channels)
        self.coarse_output = nn.Conv2d(config.hidden_channels, 1, 1)
        self.detail_output = nn.Conv2d(
            config.hidden_channels,
            config.patch_area - 1,
            1,
        )
        self.register_buffer(
            "detail_basis",
            zero_dc_hadamard_basis(config.patch_area),
            persistent=True,
        )

    def _validate_inputs(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
    ) -> None:
        batch = intended_visual.shape[0]
        if tuple(intended_visual.shape) != (batch, self.config.visual_dim):
            raise ValueError("intended visual state has the wrong shape")
        expected_field = (
            batch,
            self.config.spatial_channels,
            self.config.field_size,
            self.config.field_size,
        )
        if tuple(intended_field.shape) != expected_field:
            raise ValueError(f"intended field must have shape {expected_field}")
        expected_style = (
            batch,
            1,
            self.config.fovea_size,
            self.config.fovea_size,
        )
        if tuple(style_image.shape) != expected_style:
            raise ValueError(f"style image must have shape {expected_style}")

    def _local_source(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        field_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = intended_visual.shape[0]
        if self.config.route_mode == TILED_GLOBAL_CONTROL_ROUTE:
            source = F.normalize(intended_visual.float(), dim=-1)
            return source[:, :, None, None].expand(
                batch,
                self.config.spatial_channels,
                self.config.field_size,
                self.config.field_size,
            )
        if field_mask is None:
            return intended_field.float()
        expected = (batch, 1, self.config.field_size, self.config.field_size)
        if tuple(field_mask.shape) != expected:
            raise ValueError(f"field_mask must have shape {expected}")
        return intended_field.float() * field_mask.float()

    def _detail_logits(self, coefficients: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=coefficients.device.type, enabled=False):
            patch_channels = torch.einsum(
                "pc,bcij->bpij",
                self.detail_basis.float(),
                coefficients.float(),
            )
        return F.pixel_shuffle(patch_channels, self.config.patch_size)

    def logits_with_trace(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        field_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(intended_visual, intended_field, style_image)
        intended = F.normalize(intended_visual.float(), dim=-1)
        style = self.style_encoder(style_image)
        context = self.global_conditioner(intended, style)
        source = self._local_source(intended, intended_field, field_mask)
        hidden = self.source_projection(self.source_norm(source.float()))
        for block in self.blocks:
            hidden = block(hidden, context)
        hidden = F.silu(self.output_norm(hidden))
        coarse_cells = self.coarse_output(hidden).float()
        detail_coefficients = self.detail_output(hidden).float()
        coarse_logits = block_upsample(coarse_cells, self.config.patch_size)
        detail_logits = self._detail_logits(detail_coefficients)
        combined = coarse_logits + detail_logits
        return combined, {
            "coarse_cell_logits": coarse_cells,
            "coarse_logits": coarse_logits,
            "detail_coefficients": detail_coefficients,
            "detail_logits": detail_logits,
            "combined_logits": combined,
            "local_source": source,
            "global_context": context,
            "style_state": style,
        }

    def forward(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        field_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.logits_with_trace(
            intended_visual,
            intended_field,
            style_image,
            field_mask=field_mask,
        )
        return logits

    def plan(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        field_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(
            intended_visual,
            intended_field,
            style_image,
            field_mask=field_mask,
        ).sigmoid()


def image_patch_cell_variation(
    image: torch.Tensor,
    field_size: int,
) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("patch variation expects [batch, 1, height, width]")
    height, width = image.shape[-2:]
    if height != width:
        raise ValueError("patch variation expects square images")
    if width % field_size:
        raise ValueError("image size must be divisible by field_size")
    patch = width // field_size
    cells = (
        image.reshape(image.shape[0], 1, field_size, patch, field_size, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(image.shape[0], field_size * field_size, -1)
    )
    return (cells - cells[:, :1]).abs().max()


def _retinal_read_with_field(
    retina: SpatialVisualRetina,
    image: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    visual, field = retina.forward_with_field(image.float())
    return F.normalize(visual.float(), dim=-1), field.float()


def _retinal_read(retina: SpatialVisualRetina, image: torch.Tensor) -> torch.Tensor:
    return F.normalize(retina(image.float()).float(), dim=-1)


def _pixel_f1_rows(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_binary = generated >= 0.5
    target_binary = target >= 0.5
    true_positive = (generated_binary & target_binary).sum(dim=(1, 2, 3)).float()
    denominator = (
        generated_binary.sum(dim=(1, 2, 3))
        + target_binary.sum(dim=(1, 2, 3))
    ).clamp_min(1)
    return 2.0 * true_positive / denominator


def _soft_dice_rows(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (generated * target).sum(dim=(1, 2, 3)) + 1.0
    denominator = generated.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    return numerator / denominator


def _edge_field(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    horizontal = F.conv2d(image.float(), kernel_x.float(), padding=1)
    vertical = F.conv2d(image.float(), kernel_y.float(), padding=1)
    return torch.cat((horizontal, vertical), dim=1)


def _complexity_weights(target: torch.Tensor) -> torch.Tensor:
    masks = visual_complexity_masks(target)
    weights = target.new_ones(target.shape[0])
    weights = torch.where(masks["medium"], weights.new_full((), 1.25), weights)
    return torch.where(masks["dense"], weights.new_full((), 2.0), weights)


def _weighted_mean(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (rows * weights).sum() / weights.sum().clamp_min(1.0)


def _validate_training_images(
    writer: FieldCompleteWriter,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
) -> None:
    expected = (
        target_ink.shape[0],
        1,
        writer.config.fovea_size,
        writer.config.fovea_size,
    )
    for name, image in (
        ("target_ink", target_ink),
        ("semantic_reference", semantic_reference),
        ("style_reference", style_reference),
    ):
        if tuple(image.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if target_ink.shape[0] < 2:
        raise ValueError("field-complete training requires at least two examples")


def field_complete_writer_loss(
    writer: FieldCompleteWriter,
    retina: SpatialVisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    stroke_weight: float = 4.0,
    dice_weight: float = 1.0,
    pixel_l1_weight: float = 0.50,
    edge_weight: float = 0.25,
    identity_weight: float = 0.10,
    contrastive_weight: float = 0.05,
    field_margin_weight: float = 0.25,
    field_margin: float = 0.05,
    zero_margin_weight: float = 0.25,
    zero_margin: float = 0.05,
    coarse_weight: float = 0.50,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _validate_training_images(
        writer,
        target_ink,
        semantic_reference,
        style_reference,
    )
    with torch.no_grad():
        intended_visual, intended_field = _retinal_read_with_field(
            retina,
            semantic_reference,
        )
    logits, writer_trace = writer.logits_with_trace(
        intended_visual,
        intended_field,
        style_reference,
    )
    correct_ink = logits.sigmoid()
    target = target_ink.float()
    example_weights = _complexity_weights(target)
    pixel_weights = 1.0 + stroke_weight * target

    bce_rows = (
        pixel_weights
        * F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    ).mean(dim=(1, 2, 3))
    pixel_l1_rows = (pixel_weights * (correct_ink - target).abs()).mean(
        dim=(1, 2, 3)
    )
    dice_rows = _soft_dice_rows(correct_ink, target)
    edge_rows = (
        _edge_field(correct_ink).sub(_edge_field(target)).abs().mean(dim=(1, 2, 3))
    )
    coarse_target = block_downsample(target, writer.config.field_size)
    coarse_rows = F.binary_cross_entropy_with_logits(
        writer_trace["coarse_cell_logits"].float(),
        coarse_target,
        reduction="none",
    ).mean(dim=(1, 2, 3))

    topology_bce = _weighted_mean(bce_rows, example_weights)
    pixel_l1 = _weighted_mean(pixel_l1_rows, example_weights)
    dice_loss = _weighted_mean(1.0 - dice_rows, example_weights)
    edge_l1 = _weighted_mean(edge_rows, example_weights)
    coarse_loss = _weighted_mean(coarse_rows, example_weights)

    correct_visual = _retinal_read(retina, correct_ink)
    target_cosine_rows = (correct_visual * intended_visual).sum(dim=-1)
    identity_loss = _weighted_mean(1.0 - target_cosine_rows, example_weights)
    positive = visual_positive_mask(intended_visual, duplicate_similarity)
    identity_logits = logit_scale * correct_visual @ intended_visual.transpose(0, 1)
    identity_nce, identity_top1 = multi_positive_nce(identity_logits, positive)

    zero_mask = target.new_zeros(
        target.shape[0],
        1,
        writer.config.field_size,
        writer.config.field_size,
    )
    shuffled_ink = writer.plan(
        intended_visual,
        intended_field.roll(1, dims=0),
        style_reference,
    )
    zero_field_ink = writer.plan(
        intended_visual,
        intended_field,
        style_reference,
        field_mask=zero_mask,
    )
    if writer.config.route_mode == FIELD_COMPLETE_ROUTE:
        correct_error = (correct_ink - target).abs().mean(dim=(1, 2, 3))
        shuffled_error = (shuffled_ink - target).abs().mean(dim=(1, 2, 3))
        zero_error = (zero_field_ink - target).abs().mean(dim=(1, 2, 3))
        field_margin_loss = _weighted_mean(
            F.relu(field_margin + correct_error - shuffled_error),
            example_weights,
        )
        zero_margin_loss = _weighted_mean(
            F.relu(zero_margin + correct_error - zero_error),
            example_weights,
        )
    else:
        field_margin_loss = logits.new_zeros(())
        zero_margin_loss = logits.new_zeros(())

    total = (
        topology_bce
        + dice_weight * dice_loss
        + pixel_l1_weight * pixel_l1
        + edge_weight * edge_l1
        + identity_weight * identity_loss
        + contrastive_weight * identity_nce
        + field_margin_weight * field_margin_loss
        + zero_margin_weight * zero_margin_loss
        + coarse_weight * coarse_loss
    )
    metrics = {
        "topology_bce": topology_bce.detach(),
        "soft_dice": dice_rows.mean().detach(),
        "pixel_l1": F.l1_loss(correct_ink, target).detach(),
        "pixel_f1": _pixel_f1_rows(correct_ink, target).mean().detach(),
        "edge_l1": edge_l1.detach(),
        "coarse_loss": coarse_loss.detach(),
        "target_cosine": target_cosine_rows.mean().detach(),
        "identity_nce": identity_nce.detach(),
        "identity_top1": identity_top1.detach(),
        "field_margin_loss": field_margin_loss.detach(),
        "zero_margin_loss": zero_margin_loss.detach(),
        "field_condition_pixel_l1": F.l1_loss(correct_ink, shuffled_ink).detach(),
        "zero_field_condition_pixel_l1": F.l1_loss(
            correct_ink,
            zero_field_ink,
        ).detach(),
        "ink_fraction": correct_ink.mean().detach(),
        "detail_rms": writer_trace["detail_logits"]
        .square()
        .mean()
        .sqrt()
        .detach(),
        "complexity_mean": visual_complexity_score(target).mean().detach(),
    }
    trace = {
        "target_ink": target_ink.detach(),
        "semantic_reference": semantic_reference.detach(),
        "style_reference": style_reference.detach(),
        "correct_ink": correct_ink,
        "field_shuffled_ink": shuffled_ink,
        "zero_field_ink": zero_field_ink,
        "intended_visual": intended_visual.detach(),
        "intended_field": intended_field.detach(),
        "correct_visual": correct_visual.detach(),
        "complexity": visual_complexity_score(target).detach(),
        **{
            key: value
            for key, value in writer_trace.items()
            if key not in ("local_source", "global_context", "style_state")
        },
    }
    return total, metrics, trace


@torch.no_grad()
def evaluate_field_complete_writer_batch(
    writer: FieldCompleteWriter,
    retina: SpatialVisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    _validate_training_images(
        writer,
        target_ink,
        semantic_reference,
        style_reference,
    )
    intended_visual, intended_field = _retinal_read_with_field(
        retina,
        semantic_reference,
    )
    shuffled_visual = intended_visual.roll(1, dims=0)
    shuffled_field = intended_field.roll(1, dims=0)
    correct_logits, writer_trace = writer.logits_with_trace(
        intended_visual,
        intended_field,
        style_reference,
    )
    zero_mask = target_ink.new_zeros(
        target_ink.shape[0],
        1,
        writer.config.field_size,
        writer.config.field_size,
    )
    zero_field_logits, zero_trace = writer.logits_with_trace(
        intended_visual,
        intended_field,
        style_reference,
        field_mask=zero_mask,
    )
    branches = {
        "correct": correct_logits.sigmoid(),
        "field_shuffled": writer.plan(
            intended_visual,
            shuffled_field,
            style_reference,
        ),
        "global_shuffled": writer.plan(
            shuffled_visual,
            intended_field,
            style_reference,
        ),
        "both_shuffled": writer.plan(
            shuffled_visual,
            shuffled_field,
            style_reference,
        ),
        "zero_field": zero_field_logits.sigmoid(),
    }
    for quadrant in range(4):
        branches[f"occluded_q{quadrant}"] = writer.plan(
            intended_visual,
            intended_field,
            style_reference,
            field_mask=quadrant_field_mask(
                intended_field,
                quadrant,
                writer.config.field_size,
            ),
        )
    branch_visual = {name: _retinal_read(retina, ink) for name, ink in branches.items()}
    trace = {
        "target_ink": target_ink,
        "semantic_reference": semantic_reference,
        "style_reference": style_reference,
        "style_visual": _retinal_read(retina, style_reference),
        "intended_visual": intended_visual,
        "intended_field": intended_field,
        "complexity": visual_complexity_score(target_ink),
        "coarse_cell_logits": writer_trace["coarse_cell_logits"],
        "coarse_logits": writer_trace["coarse_logits"],
        "detail_coefficients": writer_trace["detail_coefficients"],
        "detail_logits": writer_trace["detail_logits"],
        "combined_logits": writer_trace["combined_logits"],
        "zero_source_logits": zero_trace["combined_logits"],
        "detail_basis": writer.detail_basis,
    }
    for name, ink in branches.items():
        trace[f"{name}_ink"] = ink
        trace[f"{name}_visual"] = branch_visual[name]
    metrics = summarize_field_complete_writer_trace(
        trace,
        field_size=writer.config.field_size,
        duplicate_similarity=duplicate_similarity,
        logit_scale=logit_scale,
    )
    return metrics, trace


def summarize_field_complete_writer_trace(
    trace: dict[str, torch.Tensor],
    *,
    field_size: int = 4,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> dict[str, torch.Tensor]:
    target = trace["target_ink"].float()
    intended = trace["intended_visual"]
    correct = trace["correct_ink"].float()
    basis = trace["detail_basis"].float()
    metrics = {
        **visual_actuator_retrieval_metrics(
            trace["correct_visual"],
            trace["both_shuffled_visual"],
            intended,
            duplicate_similarity=duplicate_similarity,
            logit_scale=logit_scale,
        ),
        "semantic_target_pixel_l1": F.l1_loss(
            trace["semantic_reference"].float(),
            target,
        ),
        "condition_pixel_l1": F.l1_loss(correct, trace["both_shuffled_ink"]),
        "field_condition_pixel_l1": F.l1_loss(
            correct,
            trace["field_shuffled_ink"],
        ),
        "zero_field_condition_pixel_l1": F.l1_loss(
            correct,
            trace["zero_field_ink"],
        ),
        "global_condition_pixel_l1": F.l1_loss(
            correct,
            trace["global_shuffled_ink"],
        ),
        "style_copy_cosine": (
            trace["correct_visual"] * trace["style_visual"]
        ).sum(dim=-1).mean(),
        "complexity_mean": trace["complexity"].float().mean(),
        "detail_rms": trace["detail_logits"].float().square().mean().sqrt(),
        "basis_dc_leakage_max": basis.sum(dim=0).abs().max(),
        "basis_gram_error_max": (
            basis.transpose(0, 1) @ basis
            - torch.eye(basis.shape[1], device=basis.device)
        )
        .abs()
        .max(),
        "detail_block_mean_abs_max": block_downsample(
            trace["detail_logits"].double(),
            field_size,
        )
        .abs()
        .max()
        .float(),
        "decomposition_error_max": (
            trace["combined_logits"].float()
            - trace["coarse_logits"].float()
            - trace["detail_logits"].float()
        )
        .abs()
        .max(),
        "zero_source_cell_variation_max": image_patch_cell_variation(
            trace["zero_source_logits"].float(),
            field_size,
        ),
    }
    metrics["both_shuffled_identity_top1"] = metrics["shuffled_identity_top1"]
    metrics["both_shuffled_target_cosine"] = metrics["shuffled_target_cosine"]

    score = trace["complexity"]
    masks = {
        "simple": score < 0.24,
        "medium": (score >= 0.24) & (score < 0.35),
        "dense": score >= 0.35,
    }
    for stratum, mask in masks.items():
        metrics[f"{stratum}_examples"] = mask.sum().float()
    for name in (
        "correct",
        "field_shuffled",
        "global_shuffled",
        "both_shuffled",
        "zero_field",
    ):
        ink = trace[f"{name}_ink"].float()
        visual = trace[f"{name}_visual"]
        pixel_f1 = _pixel_f1_rows(ink, target)
        pixel_l1 = (ink - target).abs().mean(dim=(1, 2, 3))
        soft_dice = _soft_dice_rows(ink, target)
        cosine = (visual * intended).sum(dim=-1)
        metrics[f"{name}_pixel_f1"] = pixel_f1.mean()
        metrics[f"{name}_pixel_l1"] = pixel_l1.mean()
        metrics[f"{name}_soft_dice"] = soft_dice.mean()
        metrics[f"{name}_target_cosine"] = cosine.mean()
        metrics[f"{name}_ink_fraction"] = ink.mean()
        for stratum, mask in masks.items():
            if bool(mask.any()):
                metrics[f"{name}_pixel_f1_{stratum}"] = pixel_f1[mask].mean()
                metrics[f"{name}_pixel_l1_{stratum}"] = pixel_l1[mask].mean()
            else:
                nan = pixel_f1.new_tensor(float("nan"))
                metrics[f"{name}_pixel_f1_{stratum}"] = nan
                metrics[f"{name}_pixel_l1_{stratum}"] = nan

    locality_values = []
    change_values = []
    output_half = target.shape[-1] // 2
    for quadrant in range(4):
        delta = (correct - trace[f"occluded_q{quadrant}_ink"].float()).abs()
        row, column = divmod(quadrant, 2)
        inside = delta[
            :,
            :,
            row * output_half : (row + 1) * output_half,
            column * output_half : (column + 1) * output_half,
        ].sum(dim=(1, 2, 3))
        total = delta.sum(dim=(1, 2, 3))
        locality = inside / (total + 1e-8)
        changed = total > 1e-8
        locality = torch.where(changed, locality, torch.zeros_like(locality))
        mean_change = delta.mean(dim=(1, 2, 3))
        metrics[f"occluded_q{quadrant}_locality"] = locality.mean()
        metrics[f"occluded_q{quadrant}_pixel_change"] = mean_change.mean()
        locality_values.append(locality)
        change_values.append(mean_change)
    metrics["occlusion_locality"] = torch.stack(locality_values).mean()
    metrics["occlusion_pixel_change"] = torch.stack(change_values).mean()
    return metrics


def field_complete_writer_config_payload(
    config: FieldCompleteWriterConfig,
) -> dict[str, Any]:
    return asdict(config)


def field_complete_writer_config_from_payload(
    payload: dict[str, Any],
) -> FieldCompleteWriterConfig:
    return FieldCompleteWriterConfig(**payload)
