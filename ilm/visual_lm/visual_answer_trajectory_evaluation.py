from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .visual_answer_trajectory import (
    V39_MAX_SEGMENTS,
    visual_segment_count_distribution,
)
from .visual_semantic_distillation_training import centered_effective_rank


@dataclass
class VisualAnswerTrajectoryEvaluationOutputs:
    read_state: torch.Tensor
    baseline_answer_state: torch.Tensor
    answer_state: torch.Tensor
    stage1_answer_state: torch.Tensor
    segment_states: torch.Tensor
    stage1_segment_states: torch.Tensor
    stop_logits: torch.Tensor
    active_probabilities: torch.Tensor
    lengths: torch.Tensor

    def validate(self) -> None:
        records, dimension = self.read_state.shape
        global_shape = (records, dimension)
        segment_shape = (records, V39_MAX_SEGMENTS, dimension)
        scalar_shape = (records, V39_MAX_SEGMENTS)
        if any(
            value.shape != global_shape
            for value in (
                self.baseline_answer_state,
                self.answer_state,
                self.stage1_answer_state,
            )
        ):
            raise ValueError("V39 evaluation global outputs do not align")
        if any(
            value.shape != segment_shape
            for value in (self.segment_states, self.stage1_segment_states)
        ):
            raise ValueError("V39 evaluation segment outputs do not align")
        if any(
            value.shape != scalar_shape
            for value in (self.stop_logits, self.active_probabilities, self.lengths)
        ):
            raise ValueError("V39 evaluation scalar outputs do not align")
        for value in self.__dict__.values():
            if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
                raise TypeError("V39 evaluation outputs must be floating tensors")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("V39 evaluation outputs are non-finite")


def indexed_retrieval_metrics(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    *,
    block_size: int = 256,
) -> dict[str, float | int]:
    if queries.ndim != 2 or candidates.ndim != 2:
        raise ValueError("V39 retrieval inputs must be matrices")
    if queries.shape[1] != candidates.shape[1] or labels.shape != (len(queries),):
        raise ValueError("V39 retrieval geometry does not align")
    if labels.dtype != torch.long or block_size < 1:
        raise ValueError("V39 retrieval labels or block size are invalid")
    if not bool(((labels >= 0) & (labels < len(candidates))).all()):
        raise ValueError("V39 retrieval label lies outside the candidate bank")
    query = F.normalize(queries.float(), dim=-1)
    candidate = F.normalize(candidates.float(), dim=-1)
    ranks: list[torch.Tensor] = []
    positive_cosines: list[torch.Tensor] = []
    for start in range(0, len(query), block_size):
        stop = min(len(query), start + block_size)
        similarities = query[start:stop] @ candidate.T
        local_labels = labels[start:stop]
        positive = similarities[
            torch.arange(stop - start, device=similarities.device),
            local_labels.to(similarities.device),
        ]
        ranks.append((similarities > positive.unsqueeze(1)).sum(dim=1).cpu() + 1)
        positive_cosines.append(positive.cpu())
    rank = torch.cat(ranks).float()
    positive = torch.cat(positive_cosines).float()
    return {
        "queries": len(query),
        "candidates": len(candidate),
        "top1": float((rank <= 1).float().mean()),
        "top5": float((rank <= min(5, len(candidate))).float().mean()),
        "mrr": float((1 / rank).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "positive_cosine": float(positive.mean()),
    }


def active_segment_geometry(
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if offsets.ndim != 1 or offsets.dtype != torch.long or len(offsets) < 2:
        raise ValueError("V39 segment offsets are invalid")
    counts = offsets[1:] - offsets[:-1]
    if not bool(((counts >= 1) & (counts <= V39_MAX_SEGMENTS)).all()):
        raise ValueError("V39 segment counts lie outside [1,16]")
    mask = torch.arange(V39_MAX_SEGMENTS).unsqueeze(0) < counts.unsqueeze(1)
    labels = torch.cat(
        [torch.arange(int(start), int(stop)) for start, stop in zip(offsets, offsets[1:])]
    )
    return counts, mask, labels


def trajectory_content_metrics(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    offsets: torch.Tensor,
    *,
    block_size: int = 256,
) -> dict[str, Any]:
    if predicted.ndim != 3 or predicted.shape[1] != V39_MAX_SEGMENTS:
        raise ValueError("V39 predicted trajectory has another shape")
    counts, mask, labels = active_segment_geometry(offsets)
    if predicted.shape[0] != len(counts) or targets.shape[0] != int(offsets[-1]):
        raise ValueError("V39 trajectory targets do not align")
    active_prediction = predicted[mask]
    retrieval = indexed_retrieval_metrics(
        active_prediction,
        targets,
        labels,
        block_size=block_size,
    )
    normalized_prediction = F.normalize(active_prediction.float(), dim=-1)
    normalized_targets = F.normalize(targets.float(), dim=-1)
    paired = F.cosine_similarity(
        normalized_prediction,
        normalized_targets[labels],
        dim=-1,
    )
    permuted = F.cosine_similarity(
        normalized_prediction,
        normalized_targets[labels.roll(1)],
        dim=-1,
    )
    adjacent_correct: list[torch.Tensor] = []
    adjacent_wrong: list[torch.Tensor] = []
    transition: list[torch.Tensor] = []
    cursor = 0
    for row, count_value in enumerate(counts.tolist()):
        if count_value < 2:
            cursor += count_value
            continue
        student = F.normalize(predicted[row, :count_value].float(), dim=-1)
        teacher = F.normalize(targets[cursor : cursor + count_value].float(), dim=-1)
        adjacent_correct.append((student[:-1] * teacher[:-1]).sum(dim=-1))
        adjacent_wrong.append((student[:-1] * teacher[1:]).sum(dim=-1))
        student_delta = F.normalize(student[1:] - student[:-1], dim=-1, eps=1e-2)
        teacher_delta = F.normalize(teacher[1:] - teacher[:-1], dim=-1, eps=1e-2)
        transition.append((student_delta * teacher_delta).sum(dim=-1))
        cursor += count_value
    correct_adjacent = torch.cat(adjacent_correct) if adjacent_correct else paired[:0]
    wrong_adjacent = torch.cat(adjacent_wrong) if adjacent_wrong else paired[:0]
    transition_cosine = torch.cat(transition) if transition else paired[:0]
    return {
        "retrieval": retrieval,
        "paired_cosine": float(paired.mean()),
        "permuted_cosine": float(permuted.mean()),
        "paired_beats_permuted": float((paired > permuted).float().mean()),
        "adjacent_pairs": len(correct_adjacent),
        "exact_position_cosine": (
            float(correct_adjacent.mean()) if len(correct_adjacent) else 0.0
        ),
        "next_position_cosine": (
            float(wrong_adjacent.mean()) if len(wrong_adjacent) else 0.0
        ),
        "exact_beats_next": (
            float((correct_adjacent > wrong_adjacent).float().mean())
            if len(correct_adjacent)
            else 0.0
        ),
        "transition_direction_cosine": (
            float(transition_cosine.mean()) if len(transition_cosine) else 0.0
        ),
    }


def stop_and_length_metrics(
    outputs: VisualAnswerTrajectoryEvaluationOutputs,
    offsets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> dict[str, float | int]:
    outputs.validate()
    counts, mask, _labels = active_segment_geometry(offsets)
    if len(counts) != len(outputs.stop_logits) or len(target_lengths) != int(offsets[-1]):
        raise ValueError("V39 stop/length targets do not align")
    distribution = visual_segment_count_distribution(outputs.stop_logits)
    counts = counts.to(distribution.probabilities.device)
    mode_count = distribution.mode
    median_count = distribution.median
    expected_count = distribution.expected
    active_expected_count = outputs.active_probabilities.float().sum(dim=1)
    target_log_probability = distribution.log_probabilities[
        torch.arange(len(counts), device=counts.device),
        counts - 1,
    ]
    active_lengths = outputs.lengths.float()[mask]
    return {
        "records": len(counts),
        "count_accuracy": float((mode_count == counts).float().mean()),
        "count_mae": float((mode_count.float() - counts.float()).abs().mean()),
        "mode_count_accuracy": float((mode_count == counts).float().mean()),
        "mode_count_mae": float(
            (mode_count.float() - counts.float()).abs().mean()
        ),
        "median_count_accuracy": float((median_count == counts).float().mean()),
        "median_count_mae": float(
            (median_count.float() - counts.float()).abs().mean()
        ),
        "expected_count_mae": float((expected_count - counts.float()).abs().mean()),
        "predicted_count_mean": float(mode_count.float().mean()),
        "median_count_mean": float(median_count.float().mean()),
        "expected_count_mean": float(expected_count.mean()),
        "target_count_mean": float(counts.float().mean()),
        "count_nll": float(-target_log_probability.mean()),
        "count_entropy": float(distribution.entropy.mean()),
        "active_expected_count_max_error": float(
            (active_expected_count - expected_count).abs().max()
        ),
        "length_mae": float((active_lengths - target_lengths.float()).abs().mean()),
    }


def trajectory_consistency_metrics(
    reference: VisualAnswerTrajectoryEvaluationOutputs,
    view: VisualAnswerTrajectoryEvaluationOutputs,
    offsets: torch.Tensor,
) -> dict[str, float]:
    reference.validate()
    view.validate()
    if reference.read_state.shape != view.read_state.shape:
        raise ValueError("V39 consistency views do not align")
    _counts, mask, _labels = active_segment_geometry(offsets)
    return {
        "read_cosine": float(
            F.cosine_similarity(reference.read_state.float(), view.read_state.float()).mean()
        ),
        "answer_cosine": float(
            F.cosine_similarity(
                reference.answer_state.float(),
                view.answer_state.float(),
            ).mean()
        ),
        "segment_cosine": float(
            F.cosine_similarity(
                reference.segment_states.float()[mask],
                view.segment_states.float()[mask],
            ).mean()
        ),
        "expected_count_mae": float(
            (
                reference.active_probabilities.float().sum(dim=1)
                - view.active_probabilities.float().sum(dim=1)
            )
            .abs()
            .mean()
        ),
    }


def output_effective_rank(
    outputs: VisualAnswerTrajectoryEvaluationOutputs,
    offsets: torch.Tensor,
    *,
    maximum_samples: int = 512,
) -> dict[str, float | int]:
    outputs.validate()
    _counts, mask, _labels = active_segment_geometry(offsets)
    answer = outputs.answer_state.float()
    segments = outputs.segment_states.float()[mask]
    answer = answer[
        torch.linspace(0, len(answer) - 1, min(len(answer), maximum_samples)).round().long()
    ]
    segments = segments[
        torch.linspace(0, len(segments) - 1, min(len(segments), maximum_samples))
        .round()
        .long()
    ]
    return {
        "answer_samples": len(answer),
        "segment_samples": len(segments),
        "answer_effective_rank": centered_effective_rank(answer),
        "segment_effective_rank": centered_effective_rank(segments),
    }


__all__ = [
    "VisualAnswerTrajectoryEvaluationOutputs",
    "active_segment_geometry",
    "indexed_retrieval_metrics",
    "output_effective_rank",
    "stop_and_length_metrics",
    "trajectory_consistency_metrics",
    "trajectory_content_metrics",
]
