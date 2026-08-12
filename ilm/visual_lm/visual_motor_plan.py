from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_actuator import (
    StyleImageEncoder,
    multi_positive_nce,
    visual_actuator_retrieval_metrics,
    visual_positive_mask,
)


class VisualRetina(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class VisualMotorPlanConfig:
    """Decode continuous visual intent into an explicit spatial stroke plan."""

    fovea_size: int = 32
    visual_dim: int = 192
    style_dim: int = 64
    style_base_channels: int = 32
    plan_base_channels: int = 128
    context_dim: int = 256
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")
        if self.visual_dim < 32 or self.style_dim < 16:
            raise ValueError("visual motor-plan condition is underspecified")
        if self.style_base_channels < 8 or self.plan_base_channels < 32:
            raise ValueError("visual motor-plan image path is underspecified")
        if self.plan_base_channels % 4:
            raise ValueError("plan_base_channels must be divisible by four")
        if self.context_dim < 64:
            raise ValueError("visual motor-plan context is underspecified")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ConditionedPlanBlock(nn.Module):
    def __init__(self, channels: int, context_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.context = nn.Linear(context_dim, channels * 2)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, field: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(field)))
        scale, shift = self.context(F.silu(context)).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(hidden)))
        return field + hidden


class PlanUpsample(nn.Module):
    def __init__(self, incoming: int, outgoing: int, context_dim: int, dropout: float):
        super().__init__()
        self.project = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.block = ConditionedPlanBlock(outgoing, context_dim, dropout)

    def forward(self, field: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        field = F.interpolate(field, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(self.project(field), context)


class ContinuousVisualMotorPlan(nn.Module):
    """Generate a topology-first ink image from continuous image-derived state."""

    def __init__(self, config: VisualMotorPlanConfig):
        super().__init__()
        self.config = config
        self.style_encoder = StyleImageEncoder(config)
        condition_dim = config.visual_dim + config.style_dim
        self.context = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, config.context_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.context_dim * 2, config.context_dim),
            nn.LayerNorm(config.context_dim),
        )
        base = config.plan_base_channels
        seed_size = config.fovea_size // 8
        self.seed_size = seed_size
        self.seed = nn.Linear(config.context_dim, base * seed_size * seed_size)
        self.seed_blocks = nn.ModuleList(
            (
                ConditionedPlanBlock(base, config.context_dim, config.dropout),
                ConditionedPlanBlock(base, config.context_dim, config.dropout),
            )
        )
        self.up1 = PlanUpsample(base, base, config.context_dim, config.dropout)
        self.up2 = PlanUpsample(base, base // 2, config.context_dim, config.dropout)
        self.up3 = PlanUpsample(base // 2, base // 4, config.context_dim, config.dropout)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(base // 4), base // 4),
            nn.SiLU(),
            nn.Conv2d(base // 4, 1, 3, padding=1),
        )

    def encode_condition(
        self,
        intended_visual: torch.Tensor,
        style_image: torch.Tensor,
    ) -> torch.Tensor:
        if intended_visual.ndim != 2 or intended_visual.shape[1] != self.config.visual_dim:
            raise ValueError("intended visual state has the wrong shape")
        if style_image.shape[0] != intended_visual.shape[0]:
            raise ValueError("style and intended visual batches must match")
        intended = F.normalize(intended_visual.float(), dim=-1)
        style = self.style_encoder(style_image)
        return self.context(torch.cat((intended, style), dim=-1))

    def forward(
        self,
        intended_visual: torch.Tensor,
        style_image: torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_condition(intended_visual, style_image)
        field = self.seed(context).reshape(
            context.shape[0],
            self.config.plan_base_channels,
            self.seed_size,
            self.seed_size,
        )
        for block in self.seed_blocks:
            field = block(field, context)
        field = self.up3(self.up2(self.up1(field, context), context), context)
        return self.output(field)

    def plan(
        self,
        intended_visual: torch.Tensor,
        style_image: torch.Tensor,
    ) -> torch.Tensor:
        return self(intended_visual, style_image).sigmoid()


def _retinal_read(retina: VisualRetina, ink: torch.Tensor) -> torch.Tensor:
    return F.normalize(retina(ink.float()).float(), dim=-1)


def _pixel_f1(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_binary = generated >= 0.5
    target_binary = target >= 0.5
    true_positive = (generated_binary & target_binary).sum(dim=(1, 2, 3)).float()
    denominator = (
        generated_binary.sum(dim=(1, 2, 3))
        + target_binary.sum(dim=(1, 2, 3))
    ).clamp_min(1)
    return 2.0 * true_positive / denominator


def _soft_dice(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (generated * target).sum(dim=(1, 2, 3)) + 1.0
    denominator = (
        generated.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    )
    return numerator / denominator


def _edge_field(image: torch.Tensor) -> torch.Tensor:
    kernel_x = image.new_tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
    ).unsqueeze(0)
    kernel_y = kernel_x.transpose(-1, -2)
    horizontal = F.conv2d(image.float(), kernel_x.float(), padding=1)
    vertical = F.conv2d(image.float(), kernel_y.float(), padding=1)
    return torch.cat((horizontal, vertical), dim=1)


def visual_motor_plan_loss(
    planner: ContinuousVisualMotorPlan,
    retina: VisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    stroke_weight: float = 4.0,
    dice_weight: float = 1.0,
    pixel_l1_weight: float = 0.50,
    edge_weight: float = 0.25,
    identity_weight: float = 0.05,
    contrastive_weight: float = 0.05,
    state_margin_weight: float = 0.10,
    state_margin: float = 0.03,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Supervise stroke topology before any stochastic surface refinement."""

    expected = (
        target_ink.shape[0],
        1,
        planner.config.fovea_size,
        planner.config.fovea_size,
    )
    for name, image in (
        ("target_ink", target_ink),
        ("semantic_reference", semantic_reference),
        ("style_reference", style_reference),
    ):
        if tuple(image.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if target_ink.shape[0] < 2:
        raise ValueError("visual motor-plan training requires at least two examples")

    with torch.no_grad():
        intended_visual = _retinal_read(retina, semantic_reference)
    logits = planner(intended_visual, style_reference)
    plan_ink = logits.sigmoid()
    target = target_ink.float()
    weights = 1.0 + stroke_weight * target
    topology_bce = (
        weights * F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    ).mean()
    pixel_l1 = (weights * (plan_ink.float() - target).abs()).mean()
    dice = _soft_dice(plan_ink.float(), target)
    dice_loss = 1.0 - dice.mean()
    edge_l1 = F.l1_loss(_edge_field(plan_ink), _edge_field(target))

    plan_visual = _retinal_read(retina, plan_ink)
    target_cosine_rows = (plan_visual * intended_visual).sum(dim=-1)
    identity_loss = 1.0 - target_cosine_rows.mean()
    positive = visual_positive_mask(intended_visual, duplicate_similarity)
    logits_identity = logit_scale * plan_visual @ intended_visual.transpose(0, 1)
    identity_nce, identity_top1 = multi_positive_nce(logits_identity, positive)

    shuffled_visual = intended_visual.roll(1, dims=0)
    shuffled_ink = planner.plan(shuffled_visual, style_reference)
    correct_rows = (plan_ink.float() - target).abs().mean(dim=(1, 2, 3))
    shuffled_rows = (shuffled_ink.float() - target).abs().mean(dim=(1, 2, 3))
    state_margin_loss = F.relu(state_margin + correct_rows - shuffled_rows).mean()

    total = (
        topology_bce
        + dice_weight * dice_loss
        + pixel_l1_weight * pixel_l1
        + edge_weight * edge_l1
        + identity_weight * identity_loss
        + contrastive_weight * identity_nce
        + state_margin_weight * state_margin_loss
    )
    metrics = {
        "topology_bce": topology_bce.detach(),
        "soft_dice": dice.mean().detach(),
        "pixel_l1": F.l1_loss(plan_ink.float(), target).detach(),
        "pixel_f1": _pixel_f1(plan_ink, target).mean().detach(),
        "edge_l1": edge_l1.detach(),
        "target_cosine": target_cosine_rows.mean().detach(),
        "identity_nce": identity_nce.detach(),
        "identity_top1": identity_top1.detach(),
        "state_margin_loss": state_margin_loss.detach(),
        "condition_pixel_l1": F.l1_loss(plan_ink, shuffled_ink).detach(),
        "ink_fraction": plan_ink.mean().detach(),
    }
    trace = {
        "target_ink": target_ink.detach(),
        "semantic_reference": semantic_reference.detach(),
        "style_reference": style_reference.detach(),
        "correct_ink": plan_ink,
        "shuffled_ink": shuffled_ink,
        "intended_visual": intended_visual.detach(),
        "correct_visual": plan_visual.detach(),
    }
    return total, metrics, trace


@torch.no_grad()
def evaluate_visual_motor_plan_batch(
    planner: ContinuousVisualMotorPlan,
    retina: VisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Intervene on intended state while target, style, and renderer stay fixed."""

    batch = target_ink.shape[0]
    expected = (batch, 1, planner.config.fovea_size, planner.config.fovea_size)
    for name, image in (
        ("target_ink", target_ink),
        ("semantic_reference", semantic_reference),
        ("style_reference", style_reference),
    ):
        if tuple(image.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if batch < 2:
        raise ValueError("visual motor-plan evaluation requires at least two examples")

    intended_visual = _retinal_read(retina, semantic_reference)
    shuffled_visual = intended_visual.roll(1, dims=0)
    correct_ink = planner.plan(intended_visual, style_reference)
    shuffled_ink = planner.plan(shuffled_visual, style_reference)
    correct_visual = _retinal_read(retina, correct_ink)
    shuffled_output_visual = _retinal_read(retina, shuffled_ink)
    style_visual = _retinal_read(retina, style_reference)
    metrics = {
        **visual_actuator_retrieval_metrics(
            correct_visual,
            shuffled_output_visual,
            intended_visual,
            duplicate_similarity=duplicate_similarity,
            logit_scale=logit_scale,
        ),
        "correct_pixel_l1": F.l1_loss(correct_ink, target_ink.float()),
        "shuffled_pixel_l1": F.l1_loss(shuffled_ink, target_ink.float()),
        "correct_pixel_f1": _pixel_f1(correct_ink, target_ink).mean(),
        "shuffled_pixel_f1": _pixel_f1(shuffled_ink, target_ink).mean(),
        "correct_soft_dice": _soft_dice(correct_ink, target_ink.float()).mean(),
        "shuffled_soft_dice": _soft_dice(shuffled_ink, target_ink.float()).mean(),
        "correct_ink_fraction": correct_ink.mean(),
        "shuffled_ink_fraction": shuffled_ink.mean(),
        "condition_pixel_l1": F.l1_loss(correct_ink, shuffled_ink),
        "style_copy_cosine": (correct_visual * style_visual).sum(dim=-1).mean(),
    }
    trace = {
        "target_ink": target_ink,
        "semantic_reference": semantic_reference,
        "style_reference": style_reference,
        "correct_ink": correct_ink,
        "shuffled_ink": shuffled_ink,
        "intended_visual": intended_visual,
        "shuffled_visual": shuffled_visual,
        "correct_visual": correct_visual,
        "shuffled_output_visual": shuffled_output_visual,
        "style_visual": style_visual,
    }
    return metrics, trace


def visual_motor_plan_config_payload(config: VisualMotorPlanConfig) -> dict[str, Any]:
    return asdict(config)


def visual_motor_plan_config_from_payload(payload: dict[str, Any]) -> VisualMotorPlanConfig:
    return VisualMotorPlanConfig(**payload)
