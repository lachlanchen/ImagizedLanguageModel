from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .canonical_glyph_flow_v43_training import select_writer_positions
from .canonical_glyph_language_training import dynamic_visual_contrastive_loss
from .ink_writer import flow_training_state, foveal_flow_loss
from .visual_flow_language import VisualFlowLanguageModel


@dataclass(frozen=True)
class VisualFlowLanguageLossWeights:
    flow: float = 1.0
    visual_contrastive: float = 0.10
    anchor: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("visual-flow loss weights must be non-negative")


VISUAL_FLOW_LANGUAGE_LOSS_WEIGHTS = VisualFlowLanguageLossWeights()


@dataclass
class VisualFlowLanguageLoss:
    loss: torch.Tensor
    flow: torch.Tensor
    endpoint_l1: torch.Tensor
    endpoint_ink_f1: torch.Tensor
    visual_contrastive: torch.Tensor
    anchor: torch.Tensor
    in_batch_top1: torch.Tensor
    condition_present_fraction: torch.Tensor
    selected_positions: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "flow": float(self.flow.detach()),
            "endpoint_l1": float(self.endpoint_l1.detach()),
            "endpoint_ink_f1": float(self.endpoint_ink_f1.detach()),
            "visual_contrastive": float(self.visual_contrastive.detach()),
            "anchor": float(self.anchor.detach()),
            "in_batch_top1": float(self.in_batch_top1.detach()),
            "condition_present_fraction": float(
                self.condition_present_fraction.detach()
            ),
            "selected_positions": float(self.selected_positions),
        }


def visual_flow_language_loss(
    model: VisualFlowLanguageModel,
    output: Mapping[str, torch.Tensor],
    target_pixels: torch.Tensor,
    *,
    positions_per_stream: int,
    generator: torch.Generator,
    condition_dropout: float | None = None,
    endpoint_weight: float = 0.10,
    stroke_weight: float = 2.0,
    weights: VisualFlowLanguageLossWeights = VISUAL_FLOW_LANGUAGE_LOSS_WEIGHTS,
) -> VisualFlowLanguageLoss:
    hidden = output["hidden_states"]
    anchors = output["anchor_fields"]
    selected_hidden, selected_anchors, selected_targets = select_writer_positions(
        hidden,
        anchors,
        target_pixels,
        positions_per_stream=positions_per_stream,
        generator=generator,
    )
    target_signed = selected_targets.float().mul(2.0).sub(1.0)
    state, velocity, times, _ = flow_training_state(
        target_signed,
        generator=generator,
    )
    dropout = (
        model.config.condition_dropout
        if condition_dropout is None
        else float(condition_dropout)
    )
    if not 0.0 <= dropout < 1.0:
        raise ValueError("visual-flow condition dropout must lie in [0,1)")
    condition_present = (
        torch.rand(
            len(selected_hidden),
            device=selected_hidden.device,
            generator=generator,
        )
        >= dropout
    ).to(selected_hidden.dtype)
    zero_plan = torch.zeros_like(target_signed, dtype=selected_hidden.dtype)
    prediction = model.writer(
        state.to(selected_hidden.dtype),
        times.to(selected_hidden.dtype),
        selected_hidden,
        zero_plan,
        condition_present=condition_present,
    )
    flow, flow_metrics = foveal_flow_loss(
        prediction,
        velocity,
        state,
        target_signed,
        times,
        endpoint_weight=endpoint_weight,
        stroke_weight=stroke_weight,
    )
    target_fields = model.language_model.field.encode_unit(selected_targets).detach()
    contrastive, top1 = dynamic_visual_contrastive_loss(
        selected_anchors,
        target_fields,
        scale=model.language_model.contrastive_scale,
    )
    anchor = (
        1.0 - (selected_anchors.float() * target_fields.float()).sum(dim=-1)
    ).mean()
    total = (
        weights.flow * flow
        + weights.visual_contrastive * contrastive
        + weights.anchor * anchor
    )
    return VisualFlowLanguageLoss(
        loss=total,
        flow=flow_metrics["flow_mse"],
        endpoint_l1=flow_metrics["endpoint_l1"],
        endpoint_ink_f1=flow_metrics["endpoint_ink_f1"],
        visual_contrastive=contrastive,
        anchor=anchor,
        in_batch_top1=top1,
        condition_present_fraction=condition_present.float().mean(),
        selected_positions=len(selected_hidden),
    )
