from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from .canonical_glyph_binding_v44 import CanonicalGlyphBindingV44
from .canonical_glyph_language_training import dynamic_visual_contrastive_loss


@dataclass
class CanonicalGlyphBindingV44Loss:
    loss: torch.Tensor
    pair: torch.Tensor
    pair_arm_accuracy: torch.Tensor
    pair_anchor: torch.Tensor
    delta_cosine: torch.Tensor
    delta_norm: torch.Tensor
    order: torch.Tensor
    order_target_gain: torch.Tensor
    natural_contrastive: torch.Tensor
    natural_top1: torch.Tensor
    natural_target: torch.Tensor
    natural_distill: torch.Tensor
    residual_norm: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "pair": float(self.pair.detach()),
            "pair_arm_accuracy": float(self.pair_arm_accuracy.detach()),
            "pair_anchor": float(self.pair_anchor.detach()),
            "delta_cosine": float(self.delta_cosine.detach()),
            "delta_norm": float(self.delta_norm.detach()),
            "order": float(self.order.detach()),
            "order_target_gain": float(self.order_target_gain.detach()),
            "natural_contrastive": float(self.natural_contrastive.detach()),
            "natural_top1": float(self.natural_top1.detach()),
            "natural_target": float(self.natural_target.detach()),
            "natural_distill": float(self.natural_distill.detach()),
            "residual_norm": float(self.residual_norm.detach()),
        }


def shuffle_v44_pair_prefixes(
    contexts: torch.Tensor,
    *,
    suffix_cells: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if contexts.ndim != 6 or tuple(contexts.shape[1:]) != (
        2,
        64,
        1,
        32,
        32,
    ):
        raise ValueError("V44 pair contexts must be [B,2,64,1,32,32]")
    if suffix_cells != 4:
        raise ValueError("V44 prefix shuffle fixes a four-image suffix")
    prefix_cells = contexts.shape[2] - suffix_cells
    priorities = torch.rand(
        (len(contexts), 2, prefix_cells),
        device=contexts.device,
        generator=generator,
    )
    permutation = priorities.argsort(dim=2)
    gather_index = permutation[..., None, None, None].expand(
        -1,
        -1,
        -1,
        *contexts.shape[3:],
    )
    shuffled_prefix = torch.gather(
        contexts[:, :, :prefix_cells],
        dim=2,
        index=gather_index,
    )
    return torch.cat((shuffled_prefix, contexts[:, :, prefix_cells:]), dim=2)


def _assigned_fields(
    candidate_fields: torch.Tensor,
    assignment: torch.Tensor,
) -> torch.Tensor:
    if candidate_fields.ndim != 3 or candidate_fields.shape[1:] != (2, 1024):
        raise ValueError("V44 pair candidate fields must be [B,2,1024]")
    if assignment.shape != candidate_fields.shape[:2]:
        raise ValueError("V44 pair assignment does not align")
    index = assignment[..., None].expand(-1, -1, candidate_fields.shape[-1])
    return candidate_fields.gather(1, index)


def canonical_glyph_binding_v44_loss(
    model: CanonicalGlyphBindingV44,
    natural_output: Mapping[str, torch.Tensor],
    natural_target_pixels: torch.Tensor,
    pair_contexts: torch.Tensor,
    pair_candidates: torch.Tensor,
    pair_assignment: torch.Tensor,
    shuffled_pair_contexts: torch.Tensor,
) -> CanonicalGlyphBindingV44Loss:
    if natural_target_pixels.ndim != 4 or natural_target_pixels.shape[1:] != (
        1,
        32,
        32,
    ):
        raise ValueError("V44 natural terminal targets must be [B,1,32,32]")
    natural_anchor = natural_output["anchor_fields"][:, -1]
    natural_base = natural_output["base_anchor_fields"][:, -1]
    natural_target_fields = model.field.encode_unit(natural_target_pixels).detach()
    natural_contrastive, natural_top1 = dynamic_visual_contrastive_loss(
        natural_anchor,
        natural_target_fields,
        scale=model.contrastive_scale,
    )
    natural_target = (
        1.0 - (natural_anchor.float() * natural_target_fields.float()).sum(dim=-1)
    ).mean()
    natural_distill = (
        1.0 - (natural_anchor.float() * natural_base.float()).sum(dim=-1)
    ).mean()

    paired = model.pair_outputs(pair_contexts, pair_candidates)
    logits = paired["logits"]
    if pair_assignment.shape != logits.shape[:2]:
        raise ValueError("V44 pair assignment does not align with logits")
    pair = F.cross_entropy(logits.flatten(0, 1), pair_assignment.flatten())
    pair_arm_accuracy = (logits.argmax(dim=-1) == pair_assignment).float().mean()
    assigned = _assigned_fields(paired["candidate_fields"], pair_assignment)
    pair_anchor = (
        1.0 - (paired["anchors"].float() * assigned.float()).sum(dim=-1)
    ).mean()

    predicted_delta = paired["anchors"][:, 0].float() - paired["anchors"][:, 1].float()
    target_delta = assigned[:, 0].float() - assigned[:, 1].float()
    delta_cosine = (
        1.0 - F.cosine_similarity(predicted_delta, target_delta, dim=-1)
    ).mean()
    delta_norm = F.smooth_l1_loss(
        predicted_delta.norm(dim=-1),
        target_delta.norm(dim=-1),
    )

    shuffled_output = model.language(shuffled_pair_contexts.flatten(0, 1))
    shuffled_anchor = shuffled_output["anchor_fields"][:, -1].reshape_as(
        paired["anchors"]
    )
    ordered_target_score = (paired["anchors"].float() * assigned.float()).sum(dim=-1)
    shuffled_target_score = (shuffled_anchor.float() * assigned.float()).sum(dim=-1)
    order_target_gain = ordered_target_score - shuffled_target_score
    order = F.relu(0.05 - order_target_gain).mean()

    residual_norm = (
        paired["anchors"].float() - paired["base_anchors"].float()
    ).norm(dim=-1).mean()
    loss = (
        2.0 * pair
        + delta_cosine
        + 0.25 * delta_norm
        + 0.50 * pair_anchor
        + 0.50 * order
        + 0.75 * natural_contrastive
        + 0.25 * natural_target
        + 0.50 * natural_distill
    )
    return CanonicalGlyphBindingV44Loss(
        loss=loss,
        pair=pair,
        pair_arm_accuracy=pair_arm_accuracy,
        pair_anchor=pair_anchor,
        delta_cosine=delta_cosine,
        delta_norm=delta_norm,
        order=order,
        order_target_gain=order_target_gain.mean(),
        natural_contrastive=natural_contrastive,
        natural_top1=natural_top1,
        natural_target=natural_target,
        natural_distill=natural_distill,
        residual_norm=residual_norm,
    )
