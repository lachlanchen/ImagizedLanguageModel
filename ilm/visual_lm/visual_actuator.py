from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ink_writer import (
    FovealInkFlow,
    FovealWriterConfig,
    flow_training_state,
    foveal_flow_loss,
    integrate_foveal_ink,
)


class VisualRetina(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class VisualActuatorConfig:
    """Configuration for rendering a continuous retinal plan as pixels."""

    fovea_size: int = 32
    visual_dim: int = 192
    style_dim: int = 64
    style_base_channels: int = 32
    flow_base_channels: int = 64
    flow_context_dim: int = 256
    condition_dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.fovea_size < 16 or self.fovea_size % 8:
            raise ValueError("fovea_size must be a multiple of eight and at least 16")
        if self.visual_dim < 32 or self.style_dim < 16:
            raise ValueError("visual actuator condition is underspecified")
        if self.style_base_channels < 8 or self.flow_base_channels < 16:
            raise ValueError("visual actuator image path is underspecified")
        if self.flow_context_dim < 64:
            raise ValueError("visual actuator flow context is underspecified")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")


class StyleImageEncoder(nn.Module):
    """Compress a writing image into a continuous typography field."""

    def __init__(self, config: VisualActuatorConfig):
        super().__init__()
        base = config.style_base_channels
        self.fovea_size = config.fovea_size
        self.output_dim = config.style_dim
        self.field = nn.Sequential(
            nn.Conv2d(1, base, 5, stride=2, padding=2),
            nn.GroupNorm(4 if base % 4 == 0 else 1, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8 if (base * 2) % 8 == 0 else 1, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8 if (base * 4) % 8 == 0 else 1, base * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base * 4, config.style_dim),
            nn.LayerNorm(config.style_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        expected = (1, self.fovea_size, self.fovea_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"style image must have shape [batch, {expected}]")
        return self.output(self.field(image.float()))


class ContinuousVisualActuator(nn.Module):
    """Write pixels from a continuous intended state and a style image."""

    def __init__(self, config: VisualActuatorConfig):
        super().__init__()
        self.config = config
        self.style_encoder = StyleImageEncoder(config)
        self.writer = FovealInkFlow(
            FovealWriterConfig(
                fovea_size=config.fovea_size,
                condition_dim=config.visual_dim + config.style_dim,
                base_channels=config.flow_base_channels,
                context_dim=config.flow_context_dim,
                condition_dropout=config.condition_dropout,
            )
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
        style = self.style_encoder(style_image)
        intended = F.normalize(intended_visual.float(), dim=-1)
        return torch.cat((intended, style), dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        intended_visual: torch.Tensor,
        style_image: torch.Tensor,
        *,
        condition_present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(intended_visual, style_image)
        blank_plan = state.new_full(state.shape, -1.0)
        return self.writer(
            state,
            time,
            condition,
            blank_plan,
            condition_present=condition_present,
        )

    def sample(
        self,
        intended_visual: torch.Tensor,
        style_image: torch.Tensor,
        *,
        steps: int = 8,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(intended_visual, style_image)
        blank_plan = condition.new_full(
            (
                condition.shape[0],
                1,
                self.config.fovea_size,
                self.config.fovea_size,
            ),
            -1.0,
        )
        return integrate_foveal_ink(
            self.writer,
            condition,
            blank_plan,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            initial_noise=initial_noise,
        )


def visual_positive_mask(candidates: torch.Tensor, threshold: float) -> torch.Tensor:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("positive threshold must be in (0, 1]")
    normalized = F.normalize(candidates.float(), dim=-1)
    mask = normalized @ normalized.transpose(0, 1) >= threshold
    mask.fill_diagonal_(True)
    return mask


def multi_positive_nce(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape != positive_mask.shape:
        raise ValueError("multi-positive logits and mask must be matching matrices")
    if not positive_mask.any(dim=1).all():
        raise ValueError("every query must have a positive candidate")
    positive = logits.masked_fill(~positive_mask, -torch.inf)
    rows = torch.logsumexp(logits, dim=1) - torch.logsumexp(positive, dim=1)
    top1 = positive_mask.gather(1, logits.argmax(dim=1, keepdim=True)).float().mean()
    return rows.mean(), top1


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


def visual_actuator_retrieval_metrics(
    correct_visual: torch.Tensor,
    shuffled_visual: torch.Tensor,
    intended_visual: torch.Tensor,
    *,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
) -> dict[str, torch.Tensor]:
    """Compare generated visual states against one shared image-derived bank."""

    if not (
        correct_visual.shape == shuffled_visual.shape == intended_visual.shape
        and correct_visual.ndim == 2
    ):
        raise ValueError("actuator retrieval states must be matching matrices")
    correct = F.normalize(correct_visual.float(), dim=-1)
    shuffled = F.normalize(shuffled_visual.float(), dim=-1)
    intended = F.normalize(intended_visual.float(), dim=-1)
    positive = visual_positive_mask(intended, duplicate_similarity)
    correct_logits = logit_scale * correct @ intended.transpose(0, 1)
    shuffled_logits = logit_scale * shuffled @ intended.transpose(0, 1)
    correct_nce, correct_top1 = multi_positive_nce(correct_logits, positive)
    shuffled_nce, shuffled_top1 = multi_positive_nce(shuffled_logits, positive)
    correct_target_cosine = (correct * intended).sum(dim=-1).mean()
    shuffled_target_cosine = (shuffled * intended).sum(dim=-1).mean()
    return {
        "correct_identity_nce": correct_nce,
        "shuffled_identity_nce": shuffled_nce,
        "identity_nce_gain": shuffled_nce - correct_nce,
        "correct_identity_top1": correct_top1,
        "shuffled_identity_top1": shuffled_top1,
        "correct_target_cosine": correct_target_cosine,
        "shuffled_target_cosine": shuffled_target_cosine,
        "target_cosine_gain": correct_target_cosine - shuffled_target_cosine,
    }


def visual_actuator_loss(
    actuator: ContinuousVisualActuator,
    retina: VisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    endpoint_weight: float = 0.10,
    stroke_weight: float = 2.0,
    identity_weight: float = 0.25,
    contrastive_weight: float = 0.25,
    sampled_identity_weight: float = 0.50,
    sampled_pixel_weight: float = 0.10,
    sampled_batch_size: int = 16,
    sampled_steps: int = 2,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Train flow endpoints and deployed samples to obey a visual plan."""

    expected = (
        target_ink.shape[0],
        1,
        actuator.config.fovea_size,
        actuator.config.fovea_size,
    )
    for name, image in (
        ("target_ink", target_ink),
        ("semantic_reference", semantic_reference),
        ("style_reference", style_reference),
    ):
        if tuple(image.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if sampled_batch_size < 0 or sampled_steps < 1:
        raise ValueError("sampled actuator settings are invalid")

    with torch.no_grad():
        intended_visual = _retinal_read(retina, semantic_reference)
    target_state = target_ink.float() * 2.0 - 1.0
    flow_state, velocity, time, _ = flow_training_state(
        target_state,
        generator=generator,
    )
    keep = (
        torch.rand(target_state.shape[0], device=target_state.device, generator=generator)
        >= actuator.config.condition_dropout
    ).to(target_state.dtype)
    prediction = actuator(
        flow_state,
        time,
        intended_visual,
        style_reference,
        condition_present=keep,
    )
    flow_loss, flow_metrics = foveal_flow_loss(
        prediction,
        velocity,
        flow_state,
        target_state,
        time,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
    )

    endpoint_state = flow_state.float() - time.float()[:, None, None, None] * prediction.float()
    endpoint_ink = ((endpoint_state + 1.0) * 0.5).clamp(0, 1)
    endpoint_visual = _retinal_read(retina, endpoint_ink)
    endpoint_cosine_rows = (endpoint_visual * intended_visual).sum(dim=-1)
    condition_weight = keep.float() * (0.05 + 0.95 * time.float().pow(4))
    weight_sum = condition_weight.sum().clamp_min(1e-6)
    identity_loss = (
        (1.0 - endpoint_cosine_rows) * condition_weight
    ).sum() / weight_sum

    positive = visual_positive_mask(intended_visual, duplicate_similarity)
    endpoint_logits = logit_scale * endpoint_visual @ intended_visual.transpose(0, 1)
    endpoint_nce_rows = torch.logsumexp(endpoint_logits, dim=1) - torch.logsumexp(
        endpoint_logits.masked_fill(~positive, -torch.inf),
        dim=1,
    )
    endpoint_nce = (endpoint_nce_rows * condition_weight).sum() / weight_sum
    with torch.no_grad():
        endpoint_top1 = (
            positive.gather(1, endpoint_logits.argmax(dim=1, keepdim=True))[:, 0].float()
            * condition_weight
        ).sum() / weight_sum
        endpoint_target_cosine = (
            endpoint_cosine_rows * condition_weight
        ).sum() / weight_sum

    sampled_loss = flow_loss.new_zeros(())
    sampled_identity = flow_loss.new_zeros(())
    sampled_pixel = flow_loss.new_zeros(())
    sampled_top1 = flow_loss.new_zeros(())
    sampled_target_cosine = flow_loss.new_zeros(())
    sampled_ink = target_ink[:0]
    subset = torch.empty(0, device=target_ink.device, dtype=torch.long)
    if sampled_batch_size:
        count = min(target_ink.shape[0], sampled_batch_size)
        subset = torch.randperm(
            target_ink.shape[0],
            device=target_ink.device,
            generator=generator,
        )[:count]
        sampled_state = actuator.sample(
            intended_visual[subset],
            style_reference[subset],
            steps=sampled_steps,
            generator=generator,
        )
        sampled_ink = ((sampled_state + 1.0) * 0.5).clamp(0, 1)
        sampled_visual = _retinal_read(retina, sampled_ink)
        sampled_target = intended_visual[subset]
        sampled_positive = positive[subset]
        sampled_logits = logit_scale * sampled_visual @ intended_visual.transpose(0, 1)
        sampled_identity, sampled_top1 = multi_positive_nce(
            sampled_logits,
            sampled_positive,
        )
        sampled_target_cosine = (sampled_visual * sampled_target).sum(dim=-1).mean()
        sampled_pixel = F.l1_loss(sampled_ink, target_ink[subset].float())
        sampled_loss = (
            sampled_identity_weight
            * (sampled_identity + 1.0 - sampled_target_cosine)
            + sampled_pixel_weight * sampled_pixel
        )

    total = (
        flow_loss
        + identity_weight * identity_loss
        + contrastive_weight * endpoint_nce
        + sampled_loss
    )
    metrics = {
        **flow_metrics,
        "identity_loss": identity_loss.detach(),
        "endpoint_identity_nce": endpoint_nce.detach(),
        "endpoint_identity_top1": endpoint_top1.detach(),
        "endpoint_target_cosine": endpoint_target_cosine.detach(),
        "condition_keep_fraction": keep.float().mean().detach(),
        "sampled_identity_active": total.new_tensor(float(bool(sampled_batch_size))),
        "sampled_identity_nce": sampled_identity.detach(),
        "sampled_identity_top1": sampled_top1.detach(),
        "sampled_target_cosine": sampled_target_cosine.detach(),
        "sampled_pixel_l1": sampled_pixel.detach(),
        "sampled_pixel_f1": (
            _pixel_f1(sampled_ink, target_ink[subset]).mean().detach()
            if subset.numel()
            else total.new_zeros(())
        ),
        "sampled_ink_fraction": (
            sampled_ink.mean().detach() if sampled_ink.numel() else total.new_zeros(())
        ),
    }
    trace = {
        "endpoint_ink": endpoint_ink.detach(),
        "sampled_ink": sampled_ink,
        "sampled_subset": subset,
        "intended_visual": intended_visual.detach(),
    }
    return total, metrics, trace


@torch.no_grad()
def evaluate_visual_actuator_batch(
    actuator: ContinuousVisualActuator,
    retina: VisualRetina,
    target_ink: torch.Tensor,
    semantic_reference: torch.Tensor,
    style_reference: torch.Tensor,
    *,
    steps: int = 8,
    guidance_scale: float = 1.0,
    duplicate_similarity: float = 0.90,
    logit_scale: float = 12.5,
    generator: torch.Generator | None = None,
    initial_noise: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Measure whether an intended visual state causally controls generated ink."""

    batch = target_ink.shape[0]
    expected = (
        batch,
        1,
        actuator.config.fovea_size,
        actuator.config.fovea_size,
    )
    for name, image in (
        ("target_ink", target_ink),
        ("semantic_reference", semantic_reference),
        ("style_reference", style_reference),
    ):
        if tuple(image.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if batch < 2:
        raise ValueError("actuator causal evaluation requires at least two examples")

    intended_visual = _retinal_read(retina, semantic_reference)
    if initial_noise is None:
        initial_noise = torch.randn(
            expected,
            device=target_ink.device,
            dtype=target_ink.dtype,
            generator=generator,
        )
    elif tuple(initial_noise.shape) != expected:
        raise ValueError(f"initial actuator noise must have shape {expected}")
    shuffled_visual = intended_visual.roll(1, dims=0)

    correct_state = actuator.sample(
        intended_visual,
        style_reference,
        steps=steps,
        guidance_scale=guidance_scale,
        initial_noise=initial_noise.clone(),
    )
    shuffled_state = actuator.sample(
        shuffled_visual,
        style_reference,
        steps=steps,
        guidance_scale=guidance_scale,
        initial_noise=initial_noise.clone(),
    )
    correct_ink = ((correct_state + 1.0) * 0.5).clamp(0, 1)
    shuffled_ink = ((shuffled_state + 1.0) * 0.5).clamp(0, 1)
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


def visual_actuator_config_payload(config: VisualActuatorConfig) -> dict[str, Any]:
    return asdict(config)


def visual_actuator_config_from_payload(payload: dict[str, Any]) -> VisualActuatorConfig:
    return VisualActuatorConfig(**payload)
