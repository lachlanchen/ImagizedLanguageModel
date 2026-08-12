from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visual_actuator import (
    multi_positive_nce,
    visual_actuator_retrieval_metrics,
    visual_positive_mask,
)
from .visual_motor_plan import (
    ConditionedPlanBlock,
    ContinuousVisualMotorPlan,
    VisualMotorPlanConfig,
)


class SpatialVisualRetina(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...

    def forward_with_field(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True)
class SpatialMotorPlanConfig(VisualMotorPlanConfig):
    """Add a continuous retinal field to a frozen global visual motor plan."""

    spatial_channels: int = 192
    spatial_hidden_channels: int = 128
    spatial_blocks: int = 2
    spatial_gate_init: float = -2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.spatial_channels < 16 or self.spatial_hidden_channels < 16:
            raise ValueError("spatial retinal path is underspecified")
        if self.spatial_blocks < 1:
            raise ValueError("spatial retinal path requires at least one block")

    def global_config(self) -> VisualMotorPlanConfig:
        return VisualMotorPlanConfig(
            fovea_size=self.fovea_size,
            visual_dim=self.visual_dim,
            style_dim=self.style_dim,
            style_base_channels=self.style_base_channels,
            plan_base_channels=self.plan_base_channels,
            context_dim=self.context_dim,
            dropout=self.dropout,
        )


class SpatialFieldAdapter(nn.Module):
    """Map a frozen retinal feature map into the global planner's seed basis."""

    def __init__(self, config: SpatialMotorPlanConfig):
        super().__init__()
        hidden = config.spatial_hidden_channels
        self.input = nn.Sequential(
            nn.GroupNorm(1, config.spatial_channels),
            nn.Conv2d(config.spatial_channels, hidden, 1),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            ConditionedPlanBlock(hidden, config.context_dim, config.dropout)
            for _ in range(config.spatial_blocks)
        )
        self.output = nn.Conv2d(hidden, config.plan_base_channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, field: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        hidden = self.input(field.float())
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output(hidden)


class SpatialRetinalMotorPlan(nn.Module):
    """Fuse global visual intent with a continuous, non-codebook retinal field."""

    def __init__(self, config: SpatialMotorPlanConfig):
        super().__init__()
        self.config = config
        self.global_plan = ContinuousVisualMotorPlan(config.global_config())
        self.global_plan.requires_grad_(False).eval()
        self.spatial_adapter = SpatialFieldAdapter(config)
        self.spatial_gate_logit = nn.Parameter(torch.tensor(config.spatial_gate_init))

    def train(self, mode: bool = True) -> "SpatialRetinalMotorPlan":
        super().train(mode)
        self.global_plan.eval()
        return self

    @property
    def spatial_gate(self) -> torch.Tensor:
        return self.spatial_gate_logit.sigmoid()

    def load_global_plan(self, state: dict[str, torch.Tensor]) -> None:
        self.global_plan.load_state_dict(state, strict=True)
        self.global_plan.requires_grad_(False).eval()

    def _validate_field(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
    ) -> None:
        expected_size = self.config.fovea_size // 8
        expected = (
            intended_visual.shape[0],
            self.config.spatial_channels,
            expected_size,
            expected_size,
        )
        if tuple(intended_field.shape) != expected:
            raise ValueError(f"intended spatial field must have shape {expected}")

    def logits_with_trace(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        spatial_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_field(intended_visual, intended_field)
        context = self.global_plan.encode_condition(intended_visual, style_image)
        batch = context.shape[0]
        if spatial_present is None:
            spatial_present = context.new_ones(batch)
        if spatial_present.ndim == 2 and spatial_present.shape[1] == 1:
            spatial_present = spatial_present[:, 0]
        if tuple(spatial_present.shape) != (batch,):
            raise ValueError("spatial_present must have shape [batch]")

        seed = self.global_plan.seed(context).reshape(
            batch,
            self.config.plan_base_channels,
            self.global_plan.seed_size,
            self.global_plan.seed_size,
        )
        residual = self.spatial_adapter(intended_field, context)
        applied = (
            spatial_present.float()[:, None, None, None]
            * self.spatial_gate
            * residual
        )
        field = seed + applied
        for block in self.global_plan.seed_blocks:
            field = block(field, context)
        field = self.global_plan.up3(
            self.global_plan.up2(
                self.global_plan.up1(field, context),
                context,
            ),
            context,
        )
        logits = self.global_plan.output(field)
        return logits, {
            "global_seed": seed,
            "spatial_residual": residual,
            "applied_spatial_residual": applied,
            "spatial_gate": self.spatial_gate.expand(batch),
        }

    def forward(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        spatial_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits, _ = self.logits_with_trace(
            intended_visual,
            intended_field,
            style_image,
            spatial_present=spatial_present,
        )
        return logits

    def plan(
        self,
        intended_visual: torch.Tensor,
        intended_field: torch.Tensor,
        style_image: torch.Tensor,
        *,
        spatial_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self(
            intended_visual,
            intended_field,
            style_image,
            spatial_present=spatial_present,
        ).sigmoid()


def visual_complexity_score(image: torch.Tensor) -> torch.Tensor:
    """Measure writing density using only binary image geometry."""

    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("visual complexity expects [batch, 1, height, width]")
    if min(image.shape[-2:]) < 8 or any(size % 4 for size in image.shape[-2:]):
        raise ValueError("visual complexity requires dimensions divisible by four")
    binary = (image.float() >= 0.5).float()
    ink = binary.mean(dim=(1, 2, 3))
    horizontal = (binary[:, :, :, 1:] - binary[:, :, :, :-1]).abs().mean(
        dim=(1, 2, 3)
    )
    vertical = (binary[:, :, 1:, :] - binary[:, :, :-1, :]).abs().mean(
        dim=(1, 2, 3)
    )
    occupied = (F.avg_pool2d(binary, 4, 4) > 0.05).float().mean(dim=(1, 2, 3))
    return ink + 0.5 * (horizontal + vertical) + 0.1 * occupied


def visual_complexity_masks(image: torch.Tensor) -> dict[str, torch.Tensor]:
    score = visual_complexity_score(image)
    return {
        "simple": score < 0.24,
        "medium": (score >= 0.24) & (score < 0.35),
        "dense": score >= 0.35,
    }


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


def _weighted_mean(rows: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (rows * weights).sum() / weights.sum().clamp_min(1.0)


def _complexity_weights(target: torch.Tensor) -> torch.Tensor:
    masks = visual_complexity_masks(target)
    weights = target.new_ones(target.shape[0])
    weights = torch.where(masks["medium"], weights.new_full((), 1.25), weights)
    return torch.where(masks["dense"], weights.new_full((), 2.0), weights)


def spatial_motor_plan_loss(
    planner: SpatialRetinalMotorPlan,
    retina: SpatialVisualRetina,
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
    spatial_margin_weight: float = 0.10,
    spatial_margin: float = 0.03,
    zero_margin_weight: float = 0.10,
    zero_margin: float = 0.01,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        raise ValueError("spatial motor-plan training requires at least two examples")

    with torch.no_grad():
        intended_visual, intended_field = _retinal_read_with_field(
            retina,
            semantic_reference,
        )
    logits, planner_trace = planner.logits_with_trace(
        intended_visual,
        intended_field,
        style_reference,
    )
    correct_ink = logits.sigmoid()
    spatial_shuffled_ink = planner.plan(
        intended_visual,
        intended_field.roll(1, dims=0),
        style_reference,
    )
    zero_field_ink = planner.plan(
        intended_visual,
        intended_field,
        style_reference,
        spatial_present=target_ink.new_zeros(target_ink.shape[0]),
    )

    target = target_ink.float()
    example_weights = _complexity_weights(target)
    pixel_weights = 1.0 + stroke_weight * target
    bce_rows = (
        pixel_weights
        * F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none")
    ).mean(dim=(1, 2, 3))
    pixel_l1_rows = (pixel_weights * (correct_ink.float() - target).abs()).mean(
        dim=(1, 2, 3)
    )
    dice_rows = _soft_dice_rows(correct_ink.float(), target)
    edge_rows = (
        _edge_field(correct_ink).sub(_edge_field(target)).abs().mean(dim=(1, 2, 3))
    )
    topology_bce = _weighted_mean(bce_rows, example_weights)
    pixel_l1 = _weighted_mean(pixel_l1_rows, example_weights)
    dice_loss = _weighted_mean(1.0 - dice_rows, example_weights)
    edge_l1 = _weighted_mean(edge_rows, example_weights)

    correct_visual = _retinal_read(retina, correct_ink)
    target_cosine_rows = (correct_visual * intended_visual).sum(dim=-1)
    identity_loss = _weighted_mean(1.0 - target_cosine_rows, example_weights)
    positive = visual_positive_mask(intended_visual, duplicate_similarity)
    identity_logits = logit_scale * correct_visual @ intended_visual.transpose(0, 1)
    identity_nce, identity_top1 = multi_positive_nce(identity_logits, positive)

    correct_error = (correct_ink.float() - target).abs().mean(dim=(1, 2, 3))
    spatial_error = (spatial_shuffled_ink.float() - target).abs().mean(dim=(1, 2, 3))
    zero_error = (zero_field_ink.float() - target).abs().mean(dim=(1, 2, 3))
    spatial_margin_loss = _weighted_mean(
        F.relu(spatial_margin + correct_error - spatial_error),
        example_weights,
    )
    zero_margin_loss = _weighted_mean(
        F.relu(zero_margin + correct_error - zero_error),
        example_weights,
    )

    total = (
        topology_bce
        + dice_weight * dice_loss
        + pixel_l1_weight * pixel_l1
        + edge_weight * edge_l1
        + identity_weight * identity_loss
        + contrastive_weight * identity_nce
        + spatial_margin_weight * spatial_margin_loss
        + zero_margin_weight * zero_margin_loss
    )
    applied = planner_trace["applied_spatial_residual"]
    metrics = {
        "topology_bce": topology_bce.detach(),
        "soft_dice": dice_rows.mean().detach(),
        "pixel_l1": correct_error.mean().detach(),
        "pixel_f1": _pixel_f1_rows(correct_ink, target).mean().detach(),
        "edge_l1": edge_l1.detach(),
        "target_cosine": target_cosine_rows.mean().detach(),
        "identity_nce": identity_nce.detach(),
        "identity_top1": identity_top1.detach(),
        "spatial_margin_loss": spatial_margin_loss.detach(),
        "zero_margin_loss": zero_margin_loss.detach(),
        "spatial_condition_pixel_l1": F.l1_loss(
            correct_ink,
            spatial_shuffled_ink,
        ).detach(),
        "zero_field_condition_pixel_l1": F.l1_loss(
            correct_ink,
            zero_field_ink,
        ).detach(),
        "spatial_gate": planner.spatial_gate.detach(),
        "spatial_residual_rms": applied.float().square().mean().sqrt().detach(),
        "ink_fraction": correct_ink.mean().detach(),
        "complexity_mean": visual_complexity_score(target).mean().detach(),
    }
    trace = {
        "target_ink": target_ink.detach(),
        "semantic_reference": semantic_reference.detach(),
        "style_reference": style_reference.detach(),
        "correct_ink": correct_ink,
        "spatial_shuffled_ink": spatial_shuffled_ink,
        "zero_field_ink": zero_field_ink,
        "intended_visual": intended_visual.detach(),
        "correct_visual": correct_visual.detach(),
        "complexity": visual_complexity_score(target).detach(),
        **{key: value for key, value in planner_trace.items() if key != "global_seed"},
    }
    return total, metrics, trace


@torch.no_grad()
def evaluate_spatial_motor_plan_batch(
    planner: SpatialRetinalMotorPlan,
    retina: SpatialVisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        raise ValueError("spatial motor-plan evaluation requires at least two examples")

    intended_visual, intended_field = _retinal_read_with_field(
        retina,
        semantic_reference,
    )
    shuffled_visual = intended_visual.roll(1, dims=0)
    shuffled_field = intended_field.roll(1, dims=0)
    correct_logits, planner_trace = planner.logits_with_trace(
        intended_visual,
        intended_field,
        style_reference,
    )
    branches = {
        "correct": correct_logits.sigmoid(),
        "spatial_shuffled": planner.plan(
            intended_visual,
            shuffled_field,
            style_reference,
        ),
        "global_shuffled": planner.plan(
            shuffled_visual,
            intended_field,
            style_reference,
        ),
        "both_shuffled": planner.plan(
            shuffled_visual,
            shuffled_field,
            style_reference,
        ),
        "zero_field": planner.plan(
            intended_visual,
            intended_field,
            style_reference,
            spatial_present=target_ink.new_zeros(batch),
        ),
    }
    branch_visual = {name: _retinal_read(retina, ink) for name, ink in branches.items()}
    style_visual = _retinal_read(retina, style_reference)
    trace = {
        "target_ink": target_ink,
        "semantic_reference": semantic_reference,
        "style_reference": style_reference,
        "intended_visual": intended_visual,
        "intended_field": intended_field,
        "style_visual": style_visual,
        "complexity": visual_complexity_score(target_ink),
        "spatial_gate": planner_trace["spatial_gate"],
        "applied_spatial_residual": planner_trace["applied_spatial_residual"],
    }
    for name, ink in branches.items():
        trace[f"{name}_ink"] = ink
        trace[f"{name}_visual"] = branch_visual[name]
    metrics = summarize_spatial_motor_plan_trace(
        trace,
        duplicate_similarity=duplicate_similarity,
        logit_scale=logit_scale,
    )
    return metrics, trace


def summarize_spatial_motor_plan_trace(
    trace: dict[str, torch.Tensor],
    *,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> dict[str, torch.Tensor]:
    target = trace["target_ink"].float()
    intended = trace["intended_visual"]
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
        "condition_pixel_l1": F.l1_loss(
            trace["correct_ink"],
            trace["both_shuffled_ink"],
        ),
        "spatial_condition_pixel_l1": F.l1_loss(
            trace["correct_ink"],
            trace["spatial_shuffled_ink"],
        ),
        "zero_field_condition_pixel_l1": F.l1_loss(
            trace["correct_ink"],
            trace["zero_field_ink"],
        ),
        "style_copy_cosine": (
            trace["correct_visual"] * trace["style_visual"]
        ).sum(dim=-1).mean(),
        "complexity_mean": trace["complexity"].float().mean(),
        "spatial_gate": trace["spatial_gate"].float().mean(),
        "spatial_residual_rms": trace["applied_spatial_residual"]
        .float()
        .square()
        .mean()
        .sqrt(),
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
        "spatial_shuffled",
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
                metrics[f"{name}_pixel_f1_{stratum}"] = pixel_f1.new_tensor(float("nan"))
                metrics[f"{name}_pixel_l1_{stratum}"] = pixel_l1.new_tensor(float("nan"))
    return metrics


def spatial_motor_plan_config_payload(config: SpatialMotorPlanConfig) -> dict[str, Any]:
    return asdict(config)


def spatial_motor_plan_config_from_payload(payload: dict[str, Any]) -> SpatialMotorPlanConfig:
    return SpatialMotorPlanConfig(**payload)
