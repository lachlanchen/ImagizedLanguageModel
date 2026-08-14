from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_flow_v43 import CanonicalGlyphFlowV43
from .canonical_glyph_language_training import dynamic_visual_contrastive_loss
from .ink_writer import flow_training_state, foveal_flow_loss


@dataclass
class V43LanguageLoss:
    loss: torch.Tensor
    natural_contrastive: torch.Tensor
    natural_anchor: torch.Tensor
    natural_pixel: torch.Tensor
    pair: torch.Tensor
    natural_top1: torch.Tensor
    pair_arm_accuracy: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "natural_contrastive": float(self.natural_contrastive.detach()),
            "natural_anchor": float(self.natural_anchor.detach()),
            "natural_pixel": float(self.natural_pixel.detach()),
            "pair": float(self.pair.detach()),
            "natural_top1": float(self.natural_top1.detach()),
            "pair_arm_accuracy": float(self.pair_arm_accuracy.detach()),
        }


@dataclass
class V43WriterLoss:
    loss: torch.Tensor
    flow_mse: torch.Tensor
    endpoint_l1: torch.Tensor
    endpoint_ink_f1: torch.Tensor
    target_ink_fraction: torch.Tensor
    condition_present_fraction: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "flow_mse": float(self.flow_mse.detach()),
            "endpoint_l1": float(self.endpoint_l1.detach()),
            "endpoint_ink_f1": float(self.endpoint_ink_f1.detach()),
            "target_ink_fraction": float(self.target_ink_fraction.detach()),
            "condition_present_fraction": float(
                self.condition_present_fraction.detach()
            ),
        }


def _decoded_anchor_pixel_loss(
    model: CanonicalGlyphFlowV43,
    anchors: torch.Tensor,
    target_pixels: torch.Tensor,
) -> torch.Tensor:
    target = (target_pixels.float() >= 0.5).float()
    full_fields = anchors.float() * (model.language_model.config.field_dim**0.5)
    logits = model.language_model.config.decoder_sharpness * (
        model.language_model.field.signed_spatial(full_fields)
    )
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    overlap = 2.0 * (probability * target).flatten(1).sum(dim=1)
    scale = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1)
    dice = (1.0 - (overlap + 1e-6) / (scale + 1e-6)).mean()
    return bce + 0.50 * dice


def canonical_glyph_flow_v43_language_loss(
    model: CanonicalGlyphFlowV43,
    natural_output: Mapping[str, torch.Tensor],
    natural_target_pixels: torch.Tensor,
    pair_contexts: torch.Tensor,
    pair_candidates: torch.Tensor,
    pair_assignment: torch.Tensor,
) -> V43LanguageLoss:
    anchors = natural_output["anchor_fields"]
    if anchors.shape[:2] != natural_target_pixels.shape[:2]:
        raise ValueError("V43 natural anchors and targets do not align")
    flat_anchors = anchors.flatten(0, 1)
    flat_pixels = natural_target_pixels.flatten(0, 1)
    flat_targets = (
        model.language_model.field.encode_unit(natural_target_pixels)
        .detach()
        .flatten(0, 1)
    )
    natural_contrastive, natural_top1 = dynamic_visual_contrastive_loss(
        flat_anchors,
        flat_targets,
        scale=model.language_model.contrastive_scale,
    )
    natural_anchor = (
        1.0 - (flat_anchors.float() * flat_targets.float()).sum(dim=-1)
    ).mean()
    natural_pixel = _decoded_anchor_pixel_loss(model, flat_anchors, flat_pixels)

    pair_logits = model.pair_logits(pair_contexts, pair_candidates)
    if pair_assignment.shape != pair_logits.shape[:2]:
        raise ValueError("V43 pair assignments do not align with pair logits")
    pair = F.cross_entropy(
        pair_logits.flatten(0, 1),
        pair_assignment.flatten(),
    )
    pair_arm_accuracy = (pair_logits.argmax(dim=-1) == pair_assignment).float().mean()
    loss = (
        natural_contrastive + 0.25 * natural_anchor + 0.20 * natural_pixel + 2.0 * pair
    )
    return V43LanguageLoss(
        loss=loss,
        natural_contrastive=natural_contrastive,
        natural_anchor=natural_anchor,
        natural_pixel=natural_pixel,
        pair=pair,
        natural_top1=natural_top1,
        pair_arm_accuracy=pair_arm_accuracy,
    )


def select_writer_positions(
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    positions_per_stream: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        hidden.shape[:2] != anchors.shape[:2]
        or hidden.shape[:2] != target_pixels.shape[:2]
    ):
        raise ValueError("V43 writer source streams do not align")
    if not 1 <= positions_per_stream <= hidden.shape[1]:
        raise ValueError("V43 writer position count is invalid")
    scores = torch.rand(
        hidden.shape[:2],
        device=hidden.device,
        generator=generator,
    )
    indices = scores.topk(positions_per_stream, dim=1).indices
    rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
    return (
        hidden[rows, indices].flatten(0, 1),
        anchors[rows, indices].flatten(0, 1),
        target_pixels[rows, indices].flatten(0, 1),
    )


def canonical_glyph_flow_v43_writer_loss(
    model: CanonicalGlyphFlowV43,
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    target_pixels: torch.Tensor,
    *,
    generator: torch.Generator,
    condition_dropout: float = 0.10,
    endpoint_weight: float = 0.10,
    stroke_weight: float = 2.0,
) -> V43WriterLoss:
    if hidden.ndim != 2 or hidden.shape[1] != model.language_model.config.model_dim:
        raise ValueError("V43 writer hidden state must be [N,model_dim]")
    if anchors.shape != (len(hidden), model.language_model.config.field_dim):
        raise ValueError("V43 writer anchors do not align")
    if target_pixels.shape != (len(hidden), 1, 32, 32):
        raise ValueError("V43 writer targets must be [N,1,32,32]")
    if not 0.0 <= condition_dropout < 1.0:
        raise ValueError("V43 writer condition dropout must lie in [0,1)")
    target_signed = target_pixels.float().mul(2.0).sub(1.0)
    state, velocity, times, _ = flow_training_state(
        target_signed,
        generator=generator,
    )
    plan = model.anchor_ink_plan(anchors).detach()
    condition_present = (
        torch.rand(
            len(hidden),
            device=hidden.device,
            generator=generator,
        )
        >= condition_dropout
    ).to(hidden.dtype)
    prediction = model.writer(
        state.to(hidden.dtype),
        times.to(hidden.dtype),
        hidden,
        plan.to(hidden.dtype),
        condition_present=condition_present,
    )
    loss, metrics = foveal_flow_loss(
        prediction,
        velocity,
        state,
        target_signed,
        times,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
    )
    return V43WriterLoss(
        loss=loss,
        flow_mse=metrics["flow_mse"],
        endpoint_l1=metrics["endpoint_l1"],
        endpoint_ink_f1=metrics["endpoint_ink_f1"],
        target_ink_fraction=metrics["target_ink_fraction"],
        condition_present_fraction=condition_present.float().mean().detach(),
    )
