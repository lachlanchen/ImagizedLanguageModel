from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .conditional_visual_density_ratio import (
    ConditionalVisualDensityRatioModel,
    paired_assignment_loss,
    per_row_assignment_margin,
    row_center_scores,
)


V29_ORDER_MARGIN = 0.10


def shuffle_visual_prefix(
    context: torch.Tensor,
    *,
    preserved_suffix: int = 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not torch.is_floating_point(context):
        raise TypeError("V29 prefix shuffle requires floating image tensors")
    if context.ndim < 5 or tuple(context.shape[-3:]) != (1, 32, 32):
        raise ValueError("V29 prefix shuffle requires [...,T,1,32,32]")
    length = context.shape[-4]
    prefix = length - preserved_suffix
    if prefix < 2:
        raise ValueError("V29 prefix shuffle needs at least two prefix cells")
    leading = context.shape[:-4]
    flat = context.reshape(-1, length, 1, 32, 32)
    orders: list[torch.Tensor] = []
    identity = torch.arange(prefix, device=context.device)
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


def _target_log_probability(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return logits.float().log_softmax(dim=-1).gather(
        1, targets[:, None]
    )[:, 0]


def natural_direction_objective(
    model: ConditionalVisualDensityRatioModel,
    context: torch.Tensor,
    targets: torch.Tensor,
    candidate_raw: torch.Tensor,
    candidate_semantic: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if context.ndim != 5 or targets.shape != context.shape[:1]:
        raise ValueError("V29 natural context and target rows do not align")
    full_state = model.encode_context(context)
    suffix_state = model.encode_context(context[:, -model.config.suffix_cells :])
    shuffled_state = model.encode_context(shuffle_visual_prefix(context))
    full = model.score_encoded_shared(
        full_state, candidate_raw, candidate_semantic
    )
    suffix = model.score_encoded_shared(
        suffix_state, candidate_raw, candidate_semantic
    )
    shuffled = model.score_encoded_shared(
        shuffled_state, candidate_raw, candidate_semantic
    )
    increment = row_center_scores(full - suffix)
    shuffled_increment = row_center_scores(shuffled - suffix)

    full_loss = F.cross_entropy(full.float(), targets)
    suffix_loss = F.cross_entropy(suffix.float(), targets)
    increment_loss = F.cross_entropy(increment, targets)
    increment_logp = _target_log_probability(increment, targets)
    shuffled_logp = _target_log_probability(shuffled_increment, targets)
    order_loss = F.softplus(
        V29_ORDER_MARGIN - (increment_logp - shuffled_logp)
    ).mean()
    losses = {
        "full": full_loss,
        "suffix": suffix_loss,
        "increment": increment_loss,
        "order": order_loss,
    }
    metrics = {
        "natural_full_loss": full_loss.detach(),
        "natural_suffix_loss": suffix_loss.detach(),
        "natural_increment_loss": increment_loss.detach(),
        "natural_order_loss": order_loss.detach(),
        "natural_full_top1": (full.argmax(dim=-1) == targets).float().mean(),
        "natural_suffix_top1": (suffix.argmax(dim=-1) == targets).float().mean(),
        "natural_increment_top1": (
            increment.argmax(dim=-1) == targets
        ).float().mean(),
        "natural_shuffled_increment_top1": (
            shuffled_increment.argmax(dim=-1) == targets
        ).float().mean(),
        "natural_increment_target_log_probability": increment_logp.mean().detach(),
        "natural_increment_order_gain": (
            increment_logp - shuffled_logp
        ).mean().detach(),
    }
    return losses, metrics


def natural_objective(
    model: ConditionalVisualDensityRatioModel,
    batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    candidate_features: tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    directions = []
    for context_key, bank_view in (
        ("first_context", 1),
        ("second_context", 0),
    ):
        raw, semantic = candidate_features[bank_view]
        directions.append(
            natural_direction_objective(
                model, batch[context_key], targets, raw, semantic
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
    model: ConditionalVisualDensityRatioModel,
    contexts: torch.Tensor,
    candidates: torch.Tensor,
    assignments: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full = model.score_paired_candidates(contexts, candidates)
    suffix = model.score_exact_suffix_paired(contexts, candidates)
    shuffled = model.score_paired_candidates(
        shuffle_visual_prefix(contexts), candidates
    )
    increment = row_center_scores(full - suffix)
    shuffled_increment = row_center_scores(shuffled - suffix)
    full_loss, full_metrics = paired_assignment_loss(full, assignments)
    increment_loss, increment_metrics = paired_assignment_loss(
        increment, assignments
    )
    increment_margin = per_row_assignment_margin(increment, assignments)
    shuffled_margin = per_row_assignment_margin(
        shuffled_increment, assignments
    )
    positive_loss = F.softplus(
        V29_ORDER_MARGIN - increment_margin
    ).mean()
    order_loss = F.softplus(
        V29_ORDER_MARGIN - (increment_margin - shuffled_margin)
    ).mean()
    losses = {
        "full": full_loss,
        "increment": increment_loss,
        "positive": positive_loss,
        "order": order_loss,
    }
    metrics = {
        "pair_full_loss": full_loss.detach(),
        "pair_increment_loss": increment_loss.detach(),
        "pair_positive_loss": positive_loss.detach(),
        "pair_order_loss": order_loss.detach(),
        "pair_full_arm_accuracy": full_metrics["pair_arm_accuracy"],
        "pair_full_both_correct_rate": full_metrics["pair_both_correct_rate"],
        "pair_full_mean_margin": full_metrics["pair_mean_margin"],
        "pair_increment_arm_accuracy": increment_metrics["pair_arm_accuracy"],
        "pair_increment_both_correct_rate": increment_metrics[
            "pair_both_correct_rate"
        ],
        "pair_increment_mean_margin": increment_margin.mean().detach(),
        "pair_shuffled_increment_mean_margin": shuffled_margin.mean().detach(),
        "pair_increment_order_gain": (
            increment_margin - shuffled_margin
        ).mean().detach(),
        "pair_suffix_row_max_error": (
            suffix[:, 0] - suffix[:, 1]
        ).abs().amax().detach(),
    }
    return losses, metrics


def pair_objective(
    model: ConditionalVisualDensityRatioModel,
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


def conditional_visual_training_microstep(
    model: ConditionalVisualDensityRatioModel,
    natural_batch: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    pair_batch: Mapping[str, torch.Tensor],
    candidate_features: tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    natural_losses, natural_metrics = natural_objective(
        model, natural_batch, targets, candidate_features
    )
    pair_losses, pair_metrics = pair_objective(model, pair_batch)
    total = (
        natural_losses["full"]
        + 0.50 * natural_losses["suffix"]
        + natural_losses["increment"]
        + 0.50 * natural_losses["order"]
        + pair_losses["full"]
        + 4.0 * pair_losses["increment"]
        + pair_losses["positive"]
        + pair_losses["order"]
    )
    return total, {
        "loss": total.detach(),
        **natural_metrics,
        **pair_metrics,
    }


__all__ = [
    "V29_ORDER_MARGIN",
    "conditional_visual_training_microstep",
    "natural_direction_objective",
    "natural_objective",
    "pair_direction_objective",
    "pair_objective",
    "shuffle_visual_prefix",
]
