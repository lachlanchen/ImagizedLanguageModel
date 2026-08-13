from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .spatial_visual_next_field import (
    SpatialVisualNextFieldModel,
    paired_assignment_loss,
    per_row_assignment_margin,
)


V30_ORDER_MARGIN = 0.10


def shuffle_visual_prefix(
    context: torch.Tensor,
    *,
    preserved_suffix: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not torch.is_floating_point(context):
        raise TypeError("V30 prefix shuffle requires floating image tensors")
    if context.ndim < 5 or tuple(context.shape[-3:]) != (1, 32, 32):
        raise ValueError("V30 prefix shuffle requires [...,T,1,32,32]")
    length = context.shape[-4]
    prefix = length - preserved_suffix
    if prefix < 2:
        raise ValueError("V30 prefix shuffle needs at least two prefix cells")
    leading = context.shape[:-4]
    flat = context.reshape(-1, length, 1, 32, 32)
    identity = torch.arange(prefix, device=context.device)
    orders: list[torch.Tensor] = []
    for _ in range(flat.shape[0]):
        order = torch.randperm(prefix, device=context.device, generator=generator)
        if torch.equal(order, identity):
            order = identity.roll(1)
        orders.append(order)
    order = torch.stack(orders)
    gather = order[:, :, None, None, None].expand(-1, -1, 1, 32, 32)
    shuffled_prefix = flat[:, :prefix].gather(1, gather)
    output = torch.cat((shuffled_prefix, flat[:, prefix:]), dim=1)
    return output.reshape(*leading, length, 1, 32, 32)


def target_log_probability(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 2 or targets.shape != logits.shape[:1]:
        raise ValueError("V30 target log-probability rows do not align")
    return logits.float().log_softmax(dim=-1).gather(
        1, targets[:, None]
    )[:, 0]


def aligned_field_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim < 3:
        raise ValueError("V30 aligned fields must have the same [...,16,C] shape")
    if predicted.shape[-2] != 16:
        raise ValueError("V30 aligned fields must contain 16 retinal cells")
    cosine = (predicted.float() * target.float()).sum(dim=-1)
    return (1.0 - cosine).mean()


def gather_target_fields(
    candidate_fields: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    if candidate_fields.ndim != 4 or candidate_fields.shape[2] != 16:
        raise ValueError("V30 pair candidate fields must be [B,K,16,C]")
    if assignments.ndim != 2 or assignments.shape[0] != candidate_fields.shape[0]:
        raise ValueError("V30 pair assignments must be [B,Q]")
    if assignments.dtype != torch.long:
        raise TypeError("V30 pair assignments must be int64")
    batch = torch.arange(candidate_fields.shape[0], device=candidate_fields.device)
    return candidate_fields[batch[:, None], assignments]


def natural_direction_objective(
    model: SpatialVisualNextFieldModel,
    context: torch.Tensor,
    targets: torch.Tensor,
    candidate_fields: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if context.ndim != 5 or targets.shape != context.shape[:1]:
        raise ValueError("V30 natural context and target rows do not align")
    full_prediction = model.predict_field(context)
    suffix_prediction = model.predict_field(
        context[:, -model.config.suffix_cells :]
    )
    shuffled_prediction = model.predict_field(shuffle_visual_prefix(context))
    full = model.score_encoded_shared(full_prediction, candidate_fields)
    suffix = model.score_encoded_shared(suffix_prediction, candidate_fields)
    shuffled = model.score_encoded_shared(shuffled_prediction, candidate_fields)
    full_loss = F.cross_entropy(full.float(), targets)
    target_fields = candidate_fields.index_select(0, targets)
    field_loss = aligned_field_loss(full_prediction, target_fields)
    full_logp = target_log_probability(full, targets)
    shuffled_logp = target_log_probability(shuffled, targets)
    order_loss = F.softplus(
        V30_ORDER_MARGIN - (full_logp - shuffled_logp)
    ).mean()
    losses = {
        "full": full_loss,
        "field": field_loss,
        "order": order_loss,
    }
    metrics = {
        "natural_full_loss": full_loss.detach(),
        "natural_field_loss": field_loss.detach(),
        "natural_order_loss": order_loss.detach(),
        "natural_full_top1": (full.argmax(dim=-1) == targets).float().mean(),
        "natural_suffix_top1": (suffix.argmax(dim=-1) == targets).float().mean(),
        "natural_shuffled_top1": (
            shuffled.argmax(dim=-1) == targets
        ).float().mean(),
        "natural_full_target_log_probability": full_logp.mean().detach(),
        "natural_suffix_target_log_probability": target_log_probability(
            suffix, targets
        ).mean().detach(),
        "natural_shuffled_target_log_probability": shuffled_logp.mean().detach(),
        "natural_order_gain": (full_logp - shuffled_logp).mean().detach(),
        "natural_positive_field_cosine": (1.0 - field_loss).detach(),
    }
    return losses, metrics


def natural_objective(
    model: SpatialVisualNextFieldModel,
    batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    candidate_features: tuple[torch.Tensor, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    directions = []
    for context_key, bank_view in (
        ("first_context", 1),
        ("second_context", 0),
    ):
        directions.append(
            natural_direction_objective(
                model,
                batch[context_key],
                targets,
                candidate_features[bank_view],
            )
        )
    losses = {
        name: torch.stack([item[0][name] for item in directions]).mean()
        for name in directions[0][0]
    }
    metrics = {
        name: torch.stack([item[1][name] for item in directions]).mean()
        for name in directions[0][1]
    }
    return losses, metrics


def pair_direction_objective(
    model: SpatialVisualNextFieldModel,
    contexts: torch.Tensor,
    candidates: torch.Tensor,
    assignments: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    candidate_fields = model.encode_route_candidates(candidates)
    full_prediction = model.predict_paired_fields(contexts)
    shuffled_prediction = model.predict_paired_fields(
        shuffle_visual_prefix(contexts)
    )
    full = model.score_encoded_paired(full_prediction, candidate_fields)
    shuffled = model.score_encoded_paired(
        shuffled_prediction, candidate_fields
    )
    suffix = model.score_exact_suffix_paired(contexts, candidates)
    assignment_loss, assignment_metrics = paired_assignment_loss(
        full, assignments
    )
    margin = per_row_assignment_margin(full, assignments)
    shuffled_margin = per_row_assignment_margin(shuffled, assignments)
    positive_loss = F.softplus(V30_ORDER_MARGIN - margin).mean()
    order_loss = F.softplus(
        V30_ORDER_MARGIN - (margin - shuffled_margin)
    ).mean()
    field_loss = aligned_field_loss(
        full_prediction,
        gather_target_fields(candidate_fields, assignments),
    )
    losses = {
        "assignment": assignment_loss,
        "field": field_loss,
        "positive": positive_loss,
        "order": order_loss,
    }
    metrics = {
        "pair_assignment_loss": assignment_loss.detach(),
        "pair_field_loss": field_loss.detach(),
        "pair_positive_loss": positive_loss.detach(),
        "pair_order_loss": order_loss.detach(),
        "pair_full_arm_accuracy": assignment_metrics["pair_arm_accuracy"],
        "pair_full_both_correct_rate": assignment_metrics[
            "pair_both_correct_rate"
        ],
        "pair_full_mean_margin": assignment_metrics["pair_mean_margin"],
        "pair_shuffled_mean_margin": shuffled_margin.mean().detach(),
        "pair_order_gain": (margin - shuffled_margin).mean().detach(),
        "pair_positive_field_cosine": (1.0 - field_loss).detach(),
        "pair_suffix_row_max_error": (
            suffix[:, 0] - suffix[:, 1]
        ).abs().amax().detach(),
    }
    return losses, metrics


def pair_objective(
    model: SpatialVisualNextFieldModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    directions = []
    for context_key, candidate_key, assignment_key in (
        ("contexts", "candidates", "assignment"),
        (
            "reference_contexts",
            "reference_candidates",
            "reference_assignment",
        ),
    ):
        directions.append(
            pair_direction_objective(
                model,
                batch[context_key],
                batch[candidate_key],
                batch[assignment_key],
            )
        )
    losses = {
        name: torch.stack([item[0][name] for item in directions]).mean()
        for name in directions[0][0]
    }
    metrics = {
        name: torch.stack([item[1][name] for item in directions]).mean()
        for name in directions[0][1]
    }
    return losses, metrics


def spatial_visual_training_microstep(
    model: SpatialVisualNextFieldModel,
    natural_batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    pair_batch: Mapping[str, torch.Tensor],
    candidate_features: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    natural_losses, natural_metrics = natural_objective(
        model, natural_batch, targets, candidate_features
    )
    pair_losses, pair_metrics = pair_objective(model, pair_batch)
    total = (
        natural_losses["full"]
        + natural_losses["field"]
        + 0.50 * natural_losses["order"]
        + 4.0 * pair_losses["assignment"]
        + pair_losses["field"]
        + pair_losses["positive"]
        + pair_losses["order"]
    )
    return total, {
        "loss": total.detach(),
        **natural_metrics,
        **pair_metrics,
    }


__all__ = [
    "V30_ORDER_MARGIN",
    "aligned_field_loss",
    "gather_target_fields",
    "natural_direction_objective",
    "natural_objective",
    "pair_direction_objective",
    "pair_objective",
    "shuffle_visual_prefix",
    "spatial_visual_training_microstep",
    "target_log_probability",
]
