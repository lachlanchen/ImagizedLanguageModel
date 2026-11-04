from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_flow_v43_training import select_writer_positions
from .canonical_glyph_language_training import dynamic_visual_contrastive_loss
from .hierarchical_raster_language import (
    HierarchicalRasterLanguageModel,
    RasterMixtureOutput,
    raster_to_generation_units,
)


@dataclass(frozen=True)
class HierarchicalRasterLanguageLossWeights:
    raster_nll: float = 1.0
    counterfactual: float = 0.75
    visual_contrastive: float = 0.05
    anchor: float = 0.025

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("V50 loss weights must be non-negative")


V50_LOSS_WEIGHTS = HierarchicalRasterLanguageLossWeights()


@dataclass
class HierarchicalRasterLanguageLoss:
    loss: torch.Tensor
    raster_nll: torch.Tensor
    coarse_nll: torch.Tensor
    detail_nll: torch.Tensor
    shuffled_raster_nll: torch.Tensor
    counterfactual: torch.Tensor
    counterfactual_accuracy: torch.Tensor
    counterfactual_nll_gain: torch.Tensor
    visual_contrastive: torch.Tensor
    anchor: torch.Tensor
    in_batch_top1: torch.Tensor
    teacher_pixel_f1: torch.Tensor
    mixture_entropy: torch.Tensor
    mixture_perplexity: torch.Tensor
    selected_positions: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "raster_nll_bits": float(self.raster_nll.detach()),
            "coarse_nll_bits": float(self.coarse_nll.detach()),
            "detail_nll_bits": float(self.detail_nll.detach()),
            "shuffled_raster_nll_bits": float(self.shuffled_raster_nll.detach()),
            "counterfactual": float(self.counterfactual.detach()),
            "counterfactual_accuracy": float(
                self.counterfactual_accuracy.detach()
            ),
            "counterfactual_nll_gain": float(
                self.counterfactual_nll_gain.detach()
            ),
            "visual_contrastive": float(self.visual_contrastive.detach()),
            "anchor": float(self.anchor.detach()),
            "in_batch_top1": float(self.in_batch_top1.detach()),
            "teacher_pixel_f1": float(self.teacher_pixel_f1.detach()),
            "mixture_entropy": float(self.mixture_entropy.detach()),
            "mixture_perplexity": float(self.mixture_perplexity.detach()),
            "selected_positions": float(self.selected_positions),
        }


def raster_mixture_nll(
    output: RasterMixtureOutput,
    target_patches: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_patches.shape != output.pixel_logits.shape[:2] + (
        output.pixel_logits.shape[-1],
    ):
        raise ValueError("V50 target patches do not align with mixture logits")
    target = target_patches.float().unsqueeze(-2)
    component_log_probability = -F.binary_cross_entropy_with_logits(
        output.pixel_logits.float(),
        target.expand_as(output.pixel_logits),
        reduction="none",
    ).sum(dim=-1)
    mixture_log_probability = output.mixture_logits.float().log_softmax(dim=-1)
    patch_log_probability = torch.logsumexp(
        mixture_log_probability + component_log_probability,
        dim=-1,
    )
    patch_nll_bits = -patch_log_probability / (
        output.pixel_logits.shape[-1] * torch.log(torch.tensor(2.0, device=target.device))
    )
    glyph_nll_bits = patch_nll_bits.mean(dim=-1)
    return patch_nll_bits, glyph_nll_bits


def _teacher_pixel_f1(
    output: RasterMixtureOutput,
    target_patches: torch.Tensor,
    *,
    coarse_units: int,
) -> torch.Tensor:
    components = output.mixture_logits.argmax(dim=-1)
    rows = torch.arange(len(components), device=components.device)[:, None]
    positions = torch.arange(components.shape[1], device=components.device)[None]
    logits = output.pixel_logits[rows, positions, components]
    predicted = logits[:, coarse_units:].sigmoid() >= 0.5
    target = target_patches[:, coarse_units:] >= 0.5
    true_positive = (predicted & target).flatten(1).sum(dim=-1).float()
    total = predicted.flatten(1).sum(dim=-1) + target.flatten(1).sum(dim=-1)
    return (2.0 * true_positive / total.clamp_min(1)).mean()


def _deranged_indices(count: int, device: torch.device) -> torch.Tensor:
    if count < 2:
        raise ValueError("V50 counterfactual training requires at least two positions")
    return torch.roll(torch.arange(count, device=device), shifts=1)


def hierarchical_raster_language_loss(
    model: HierarchicalRasterLanguageModel,
    output: Mapping[str, torch.Tensor],
    target_pixels: torch.Tensor,
    *,
    positions_per_stream: int,
    generator: torch.Generator,
    counterfactual_margin: float = 0.02,
    weights: HierarchicalRasterLanguageLossWeights = V50_LOSS_WEIGHTS,
) -> HierarchicalRasterLanguageLoss:
    if counterfactual_margin <= 0.0:
        raise ValueError("V50 counterfactual margin must be positive")
    hidden = output["hidden_states"]
    anchors = output["anchor_fields"]
    selected_hidden, selected_anchors, selected_targets = select_writer_positions(
        hidden,
        anchors,
        target_pixels,
        positions_per_stream=positions_per_stream,
        generator=generator,
    )
    target_patches = raster_to_generation_units(
        selected_targets,
        config=model.config,
    )
    correct_plan = model.plan_units(selected_anchors)
    correct = model.writer(
        selected_hidden,
        selected_anchors,
        target_patches,
        correct_plan,
    )
    correct_patch_nll, _ = raster_mixture_nll(correct, target_patches)
    coarse_units = model.config.coarse_units
    correct_coarse_nll = correct_patch_nll[:, :coarse_units].mean(dim=-1)
    correct_detail_nll = correct_patch_nll[:, coarse_units:].mean(dim=-1)
    correct_glyph_nll = 0.5 * (correct_coarse_nll + correct_detail_nll)

    shuffled_indices = _deranged_indices(len(selected_hidden), selected_hidden.device)
    shuffled_anchor = selected_anchors[shuffled_indices]
    shuffled = model.writer(
        selected_hidden[shuffled_indices],
        shuffled_anchor,
        target_patches,
        model.plan_units(shuffled_anchor),
    )
    shuffled_patch_nll, _ = raster_mixture_nll(shuffled, target_patches)
    shuffled_coarse_nll = shuffled_patch_nll[:, :coarse_units].mean(dim=-1)
    shuffled_detail_nll = shuffled_patch_nll[:, coarse_units:].mean(dim=-1)
    shuffled_glyph_nll = 0.5 * (
        shuffled_coarse_nll + shuffled_detail_nll
    )
    nll_gain = shuffled_coarse_nll - correct_coarse_nll
    counterfactual = F.relu(counterfactual_margin - nll_gain).mean()
    counterfactual_accuracy = (nll_gain > 0.0).float().mean()

    target_fields = model.reader.field.encode_unit(selected_targets).detach()
    contrastive, top1 = dynamic_visual_contrastive_loss(
        selected_anchors,
        target_fields,
        scale=model.reader.contrastive_scale,
    )
    anchor = (
        1.0 - (selected_anchors.float() * target_fields.float()).sum(dim=-1)
    ).mean()
    probabilities = correct.mixture_logits.float().softmax(dim=-1)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-8).log()
    ).sum(dim=-1).mean()
    raster_nll = correct_glyph_nll.mean()
    shuffled_raster_nll = shuffled_glyph_nll.mean()
    total = (
        weights.raster_nll * raster_nll
        + weights.counterfactual * counterfactual
        + weights.visual_contrastive * contrastive
        + weights.anchor * anchor
    )
    return HierarchicalRasterLanguageLoss(
        loss=total,
        raster_nll=raster_nll,
        coarse_nll=correct_coarse_nll.mean(),
        detail_nll=correct_detail_nll.mean(),
        shuffled_raster_nll=shuffled_raster_nll,
        counterfactual=counterfactual,
        counterfactual_accuracy=counterfactual_accuracy,
        counterfactual_nll_gain=nll_gain.mean(),
        visual_contrastive=contrastive,
        anchor=anchor,
        in_batch_top1=top1,
        teacher_pixel_f1=_teacher_pixel_f1(
            correct,
            target_patches,
            coarse_units=coarse_units,
        ),
        mixture_entropy=entropy,
        mixture_perplexity=entropy.exp(),
        selected_positions=len(selected_hidden),
    )
