from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .continuous_glyph_codec_training import glyph_sobel_edges
from .glyph_content_form import GlyphContentFormModel, GlyphContentFormOutput


@dataclass(frozen=True)
class GlyphContentFormLossWeights:
    content_contrastive: float = 1.0
    content_alignment: float = 0.5
    form_contrastive: float = 0.25
    self_surface: float = 1.0
    cross_surface: float = 0.5
    reference_surface: float = 0.75
    visual: float = 0.25
    independence: float = 0.05
    surface_mse: float = 0.10
    visual_edge: float = 0.25
    visual_ink: float = 0.25

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V40 loss weights must be non-negative")


V40_LOSS_WEIGHTS = GlyphContentFormLossWeights()


class WarmStartTrainableEMA:
    def __init__(self, model: torch.nn.Module, *, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("V40 EMA decay must be in [0,1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("V40 EMA requires trainable parameters")

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> float:
        effective_decay = min(
            self.decay,
            (1.0 + self.updates) / (10.0 + self.updates),
        )
        parameters = dict(model.named_parameters())
        if not set(parameters).issuperset(self.shadow):
            raise ValueError("V40 EMA parameter names changed")
        for name, shadow in self.shadow.items():
            shadow.lerp_(parameters[name].detach().float(), 1.0 - effective_decay)
        self.updates += 1
        return effective_decay

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            parameters[name].copy_(value.to(parameters[name]))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {
                name: value.detach().cpu().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if float(state["decay"]) != self.decay:
            raise ValueError("V40 EMA decay differs from checkpoint")
        source = state["shadow"]
        if not isinstance(source, Mapping) or set(source) != set(self.shadow):
            raise ValueError("V40 EMA parameter names differ from checkpoint")
        for name, value in source.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError("V40 EMA state must contain tensors")
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"V40 EMA tensor shape differs for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name]))
        self.updates = int(state["updates"])
        if self.updates < 0:
            raise ValueError("V40 EMA update count cannot be negative")


@dataclass
class GlyphContentFormLoss:
    loss: torch.Tensor
    content_contrastive: torch.Tensor
    content_alignment: torch.Tensor
    content_top1: torch.Tensor
    form_contrastive: torch.Tensor
    form_top1: torch.Tensor
    self_surface: torch.Tensor
    cross_surface: torch.Tensor
    reference_surface: torch.Tensor
    visual: torch.Tensor
    visual_pixel: torch.Tensor
    visual_edge: torch.Tensor
    visual_ink: torch.Tensor
    independence: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            key: float(value.detach())
            for key, value in self.__dict__.items()
            if isinstance(value, torch.Tensor)
        }


def symmetric_paired_contrastive_loss(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("V40 paired logits must be a square matrix")
    if len(logits) < 2:
        raise ValueError("V40 contrastive batches require at least two families")
    labels = torch.arange(len(logits), device=logits.device)
    loss = 0.5 * (
        F.cross_entropy(logits.float(), labels)
        + F.cross_entropy(logits.float().T, labels)
    )
    top1 = 0.5 * (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.argmax(dim=0) == labels).float().mean()
    )
    return loss, top1


def supervised_contrastive_loss(
    states: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 2 or labels.shape != (len(states),):
        raise ValueError("V40 supervised contrastive inputs do not align")
    if len(states) < 4 or not 0.01 <= temperature <= 1.0:
        raise ValueError("V40 form contrastive configuration is invalid")
    normalized = F.normalize(states.float(), dim=-1)
    logits = (normalized @ normalized.T) / temperature
    eye = torch.eye(len(states), dtype=torch.bool, device=states.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    if not bool(positives.any(dim=1).all()):
        raise ValueError("V40 every form state requires a same-stage positive")
    logits = logits.masked_fill(eye, -torch.inf)
    log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1)
    loss = -(
        log_probabilities.masked_fill(~positives, 0.0).sum(dim=1)
        / positive_count
    ).mean()
    nearest = logits.argmax(dim=1)
    top1 = labels[nearest].eq(labels).float().mean()
    return loss, top1


def _surface_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("V40 surface targets do not align")
    cosine = 1.0 - F.cosine_similarity(
        predicted.float(),
        target.float(),
        dim=-1,
    ).mean()
    mse = F.mse_loss(predicted.float(), target.float())
    return cosine + mse_weight * mse


def _visual_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    weights: GlyphContentFormLossWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.shape != targets.shape or targets.ndim != 4:
        raise ValueError("V40 visual targets do not align")
    pixel = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    probabilities = logits.float().sigmoid()
    edge = (
        glyph_sobel_edges(probabilities) - glyph_sobel_edges(targets.float())
    ).abs().mean()
    target_ink = (1.0 - targets.float()).flatten(1)
    predicted_ink = (1.0 - probabilities).flatten(1)
    overlap = 2.0 * (target_ink * predicted_ink).sum(dim=1)
    scale = target_ink.sum(dim=1) + predicted_ink.sum(dim=1)
    ink = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    visual = pixel + weights.visual_edge * edge + weights.visual_ink * ink
    return visual, pixel, edge, ink


def _content_form_independence(
    content_states: torch.Tensor,
    form_states: torch.Tensor,
) -> torch.Tensor:
    if content_states.ndim != 2 or form_states.ndim != 2:
        raise ValueError("V40 independence states must be matrices")
    if len(content_states) != len(form_states) or len(content_states) < 2:
        raise ValueError("V40 independence states must align")
    content = content_states.float() - content_states.float().mean(dim=0)
    form = form_states.float() - form_states.float().mean(dim=0)
    covariance = content.T @ form / (len(content) - 1)
    return covariance.square().sum()


def glyph_content_form_loss(
    model: GlyphContentFormModel,
    output: GlyphContentFormOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    stage_ids: torch.Tensor,
    weights: GlyphContentFormLossWeights = V40_LOSS_WEIGHTS,
) -> GlyphContentFormLoss:
    required = {
        "anchor_pixels",
        "positive_pixels",
        "anchor_style_pixels",
        "positive_style_pixels",
    }
    if not required.issubset(batch):
        raise ValueError("V40 batch lacks paired visual tensors")
    batch_size = output.anchor_content.shape[0]
    if stage_ids.shape != (4 * batch_size,):
        raise ValueError("V40 stage labels do not align with form states")
    stage_ids = stage_ids.to(output.anchor_content.device)

    content_logits = model.content_similarity_logits(
        output.anchor_content,
        output.positive_content,
    )
    content_contrastive, content_top1 = symmetric_paired_contrastive_loss(
        content_logits
    )
    content_alignment = (
        1.0
        - F.cosine_similarity(
            output.anchor_content.float(),
            output.positive_content.float(),
            dim=-1,
        ).mean()
    )

    form_states = torch.cat(
        (
            output.anchor_form,
            output.positive_form,
            output.anchor_reference_form,
            output.positive_reference_form,
        ),
        dim=0,
    )
    form_contrastive, form_top1 = supervised_contrastive_loss(
        form_states,
        stage_ids,
    )

    self_surface = 0.5 * (
        _surface_loss(
            output.anchor_self_surface,
            output.anchor_surface,
            mse_weight=weights.surface_mse,
        )
        + _surface_loss(
            output.positive_self_surface,
            output.positive_surface,
            mse_weight=weights.surface_mse,
        )
    )
    cross_surface = 0.5 * (
        _surface_loss(
            output.anchor_cross_surface,
            output.anchor_surface,
            mse_weight=weights.surface_mse,
        )
        + _surface_loss(
            output.positive_cross_surface,
            output.positive_surface,
            mse_weight=weights.surface_mse,
        )
    )
    reference_surface = 0.5 * (
        _surface_loss(
            output.anchor_reference_surface,
            output.anchor_surface,
            mse_weight=weights.surface_mse,
        )
        + _surface_loss(
            output.positive_reference_surface,
            output.positive_surface,
            mse_weight=weights.surface_mse,
        )
    )

    predicted_surfaces = torch.cat(
        (output.anchor_reference_surface, output.positive_reference_surface),
        dim=0,
    )
    target_pixels = torch.cat(
        (batch["anchor_pixels"], batch["positive_pixels"]),
        dim=0,
    ).to(predicted_surfaces.device)
    visual_logits = model.decode_surface(predicted_surfaces)
    visual, visual_pixel, visual_edge, visual_ink = _visual_losses(
        visual_logits,
        target_pixels,
        weights=weights,
    )

    independence = _content_form_independence(
        torch.cat((output.anchor_content, output.positive_content), dim=0),
        torch.cat((output.anchor_form, output.positive_form), dim=0),
    )
    loss = (
        weights.content_contrastive * content_contrastive
        + weights.content_alignment * content_alignment
        + weights.form_contrastive * form_contrastive
        + weights.self_surface * self_surface
        + weights.cross_surface * cross_surface
        + weights.reference_surface * reference_surface
        + weights.visual * visual
        + weights.independence * independence
    )
    return GlyphContentFormLoss(
        loss=loss,
        content_contrastive=content_contrastive,
        content_alignment=content_alignment,
        content_top1=content_top1,
        form_contrastive=form_contrastive,
        form_top1=form_top1,
        self_surface=self_surface,
        cross_surface=cross_surface,
        reference_surface=reference_surface,
        visual=visual,
        visual_pixel=visual_pixel,
        visual_edge=visual_edge,
        visual_ink=visual_ink,
        independence=independence,
    )


__all__ = [
    "GlyphContentFormLoss",
    "GlyphContentFormLossWeights",
    "V40_LOSS_WEIGHTS",
    "WarmStartTrainableEMA",
    "glyph_content_form_loss",
    "supervised_contrastive_loss",
    "symmetric_paired_contrastive_loss",
]
