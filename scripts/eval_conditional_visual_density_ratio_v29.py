#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.conditional_visual_density_ratio import (
    V29_ARCHITECTURE,
    ConditionalVisualDensityRatioModel,
    conditional_visual_density_ratio_boundary_receipt,
    conditional_visual_density_ratio_config_from_payload,
    row_center_scores,
)
from ilm.visual_lm.conditional_visual_density_ratio_data import (
    build_v29_candidate_statistics,
    conditional_visual_data_boundary_receipt,
)
from ilm.visual_lm.conditional_visual_density_ratio_training import (
    shuffle_visual_prefix,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualAuditDataset,
    build_factorized_audit_windows,
)
from ilm.visual_lm.joint_visual_compatibility_data import (
    JointVisualPairDataset,
    JointVisualRenderConfig,
    build_joint_suffix_pairs,
    joint_visual_pair_collate,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    file_sha256,
    iter_split_writing,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    VisualCharacterStatistics,
    render_visual_character_bank,
    visual_character_statistics_receipt,
)


ARCHITECTURE = V29_ARCHITECTURE
AUDIT_ARCHITECTURE = "conditional-visual-density-ratio-v29-development-audit"
PROTOCOL_DOCUMENT = "references/conditional_visual_density_ratio_v29_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/conditional_visual_density_ratio_v29_evidence/checkpoint_final.pt"
)
AUDIT_SEED = 20260924
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered V29 development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out", default="artifacts/conditional_visual_density_ratio_v29_audit"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=NATURAL_WINDOWS)
    parser.add_argument("--pair-windows", type=int, default=PAIR_WINDOWS)
    parser.add_argument("--bank-size", type=int, default=AUDIT_BANK_SIZE)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def v29_gate_report(
    natural: Mapping[str, float],
    suffix4: Mapping[str, float],
    *,
    frozen_images_instantiated: bool,
) -> tuple[dict[str, bool], dict[str, bool]]:
    permutation_clean = all(
        _strictly_below(
            suffix4[f"candidate_permutation_{name}_max_score_error"], 1e-5
        )
        and suffix4[f"candidate_permutation_{name}_accuracy_agreement"] == 1.0
        for name in ("full", "suffix4", "increment")
    )
    mechanism = {
        "increment_pair_arm_accuracy": _strictly_above(
            suffix4["increment_arm_accuracy"], 0.65
        ),
        "increment_pair_both_correct_rate": _strictly_above(
            suffix4["increment_both_correct_rate"], 0.40
        ),
        "increment_gain_over_shuffled": _strictly_above(
            suffix4["increment_arm_accuracy"]
            - suffix4["shuffled_increment_arm_accuracy"],
            0.10,
        ),
        "increment_margin_gain_over_shuffled": _strictly_above(
            suffix4["increment_mean_margin"]
            - suffix4["shuffled_increment_mean_margin"],
            0.05,
        ),
        "full_pair_arm_accuracy": _strictly_above(
            suffix4["full_arm_accuracy"], 0.65
        ),
        "full_pair_both_correct_rate": _strictly_above(
            suffix4["full_both_correct_rate"], 0.40
        ),
        "suffix4_control_at_chance": abs(
            suffix4["suffix4_arm_accuracy"] - 0.5
        )
        <= 1e-6,
        "suffix_pixels_and_scores_exact": (
            suffix4["suffix_pixel_equality"] == 1.0
            and suffix4["suffix_score_row_equality"] == 1.0
        ),
        "raw_retina_two_candidate_identity": (
            suffix4["raw_retina_two_candidate_identity_accuracy"] >= 0.99
        ),
        "frozen_semantic_cross_font_identity": (
            natural["frozen_semantic_cross_font_identity_top1"] >= 0.95
        ),
        "candidate_permutation_equivariant": permutation_clean,
        "student_boundary_and_bank_clean": (
            natural["student_boundary_clean"] == 1.0
            and natural["training_bank_absent_from_checkpoint"] == 1.0
        ),
        "peak_allocated_vram_below_18_gib": _strictly_below(
            natural["peak_allocated_vram_gib"], 18.0
        ),
        "frozen_images_not_instantiated": not frozen_images_instantiated,
    }
    language = {
        "full_top1_at_least_15_percent": natural["full_top1"] >= 0.15,
        "full_top1_gain_over_unigram": _strictly_above(
            natural["full_top1"] - natural["unigram_top1"], 0.03
        ),
        "full_top1_gain_over_bigram": _strictly_above(
            natural["full_top1"] - natural["bigram_top1"], 0.01
        ),
        "full_log_probability_gain_over_bigram": _strictly_above(
            natural["full_target_log_probability"]
            - natural["bigram_target_log_probability"],
            0.05,
        ),
        "full_log_probability_gain_over_suffix4": _strictly_above(
            natural["full_target_log_probability"]
            - natural["suffix4_target_log_probability"],
            0.03,
        ),
        "full_log_probability_gain_over_shuffled": _strictly_above(
            natural["full_target_log_probability"]
            - natural["shuffled_target_log_probability"],
            0.03,
        ),
    }
    return mechanism, language


def _checkpoint_bank_is_absent(checkpoint: Mapping[str, Any]) -> bool:
    if checkpoint.get("deployed_state_includes_training_candidate_images") is not False:
        return False
    model_state = checkpoint.get("model")
    if not isinstance(model_state, Mapping):
        return False
    return not any("bank" in str(name).lower() for name in model_state)


def student_boundary_is_clean(
    model: ConditionalVisualDensityRatioModel,
    checkpoint: Mapping[str, Any],
) -> bool:
    model_receipt = conditional_visual_density_ratio_boundary_receipt(model.config)
    data_receipt = conditional_visual_data_boundary_receipt()
    required_model_true = {
        "input_is_continuous_image_stream",
        "candidate_is_arbitrary_image",
        "output_is_candidate_conditioned_visual_energy",
        "retina_is_frozen",
        "semantic_adapters_are_frozen",
    }
    required_data_true = {
        "input_is_continuous_image_stream",
        "canonical_identity_derived_from_exact_pixels",
        "canonical_indices_are_temporary_loss_only",
        "pair_assignment_labels_are_positions",
        "pair_candidate_order_is_randomized",
        "pair_suffix_pixels_identical",
        "training_bank_is_host_only",
    }
    required_false = {
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_external_language_model",
        "candidate_bank_deployed",
    }
    extra_model_false = {
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_glyph_lookup",
        "candidate_bank_in_model_state",
    }
    return (
        model_receipt.get("architecture") == ARCHITECTURE
        and data_receipt.get("architecture") == ARCHITECTURE
        and all(model_receipt.get(key) is True for key in required_model_true)
        and all(data_receipt.get(key) is True for key in required_data_true)
        and all(model_receipt.get(key) is False for key in required_false)
        and all(data_receipt.get(key) is False for key in required_false)
        and all(model_receipt.get(key) is False for key in extra_model_false)
        and _checkpoint_bank_is_absent(checkpoint)
    )


def _device_images(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("V29 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


@torch.no_grad()
def encode_candidate_bank(
    model: ConditionalVisualDensityRatioModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V29 visual bank must be [identity,view,1,32,32]")
    flat = images.reshape(-1, 1, 32, 32)
    raw_chunks: list[torch.Tensor] = []
    semantic_chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            raw, semantic = model.encode_image_parts(batch, target=True)
        raw_chunks.append(raw)
        semantic_chunks.append(semantic)
    shape = (images.shape[0], images.shape[1], -1)
    return torch.cat(raw_chunks).reshape(*shape), torch.cat(
        semantic_chunks
    ).reshape(*shape)


def _cross_font_retrieval_accuracy(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    logits = queries.float() @ keys.float().transpose(0, 1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean())


def _top_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = logits.float().log_softmax(dim=1).gather(
        1, targets[:, None]
    )[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(log_probability.sum()),
    }


def _audit_shuffle(context: torch.Tensor, *, first_index: int) -> torch.Tensor:
    generator = torch.Generator(device=context.device).manual_seed(
        AUDIT_SEED + first_index * 104_729
    )
    return shuffle_visual_prefix(context, generator=generator)


def _trigram_rows(
    records: Sequence[Any],
    statistics: VisualCharacterStatistics,
) -> dict[str, Counter[int]]:
    rows: dict[str, Counter[int]] = defaultdict(Counter)
    index = statistics.index
    for _, _, writing in iter_split_writing(
        records,
        split="train",
        script_views_mode="original+simplified",
    ):
        for position in range(2, len(writing)):
            target = index.get(writing[position])
            if target is not None:
                rows[writing[position - 2 : position]][target] += 1
    return dict(rows)


def _baseline_metrics(
    statistics: VisualCharacterStatistics,
    trigram_rows: Mapping[str, Counter[int]],
    targets: Sequence[int],
    contexts: Sequence[str],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    width = len(statistics.characters)
    unigram = torch.tensor(statistics.counts, dtype=torch.float64) + alpha
    unigram /= unigram.sum()
    output = {
        f"{name}_{metric}": 0.0
        for name in ("unigram", "bigram", "trigram")
        for metric in (
            "correct_top1",
            "correct_top5",
            "target_log_probability_sum",
            "coverage",
        )
    }
    unigram_top = unigram.topk(min(5, width)).indices
    for target, context in zip(targets, contexts):
        output["unigram_correct_top1"] += float(unigram_top[0] == target)
        output["unigram_correct_top5"] += float((unigram_top == target).any())
        output["unigram_target_log_probability_sum"] += math.log(
            float(unigram[target])
        )
        output["unigram_coverage"] += 1.0
        bigram_sparse = statistics.bigram_rows.get(context[-1])
        if bigram_sparse:
            bigram = torch.full((width,), alpha, dtype=torch.float64)
            for index, count in bigram_sparse:
                bigram[index] += count
            bigram /= bigram.sum()
            output["bigram_coverage"] += 1.0
        else:
            bigram = unigram
        trigram_sparse = trigram_rows.get(context[-2:])
        if trigram_sparse:
            trigram = torch.full((width,), alpha, dtype=torch.float64)
            for index, count in trigram_sparse.items():
                trigram[index] += count
            trigram /= trigram.sum()
            output["trigram_coverage"] += 1.0
        else:
            trigram = unigram
        for name, row in (("bigram", bigram), ("trigram", trigram)):
            top = row.topk(min(5, width)).indices
            output[f"{name}_correct_top1"] += float(top[0] == target)
            output[f"{name}_correct_top5"] += float((top == target).any())
            output[f"{name}_target_log_probability_sum"] += math.log(
                float(row[target])
            )
    return output


@torch.no_grad()
def evaluate_natural_language(
    model: ConditionalVisualDensityRatioModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    trigram_rows: Mapping[str, Counter[int]],
    bank_images: torch.Tensor,
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_bank, semantic_bank = encode_candidate_bank(
        model, bank_images, device=device, precision=precision
    )
    raw_identity = 0.5 * (
        _cross_font_retrieval_accuracy(raw_bank[:, 0], raw_bank[:, 1])
        + _cross_font_retrieval_accuracy(raw_bank[:, 1], raw_bank[:, 0])
    )
    semantic_identity = 0.5 * (
        _cross_font_retrieval_accuracy(
            semantic_bank[:, 0], semantic_bank[:, 1]
        )
        + _cross_font_retrieval_accuracy(
            semantic_bank[:, 1], semantic_bank[:, 0]
        )
    )
    variants = (
        "full",
        "suffix4",
        "shuffled",
        "increment",
        "shuffled_increment",
    )
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
        }
        for name in variants
    }
    all_targets: list[int] = []
    all_contexts: list[str] = []
    examples = 0
    started = time.monotonic()
    raw_by_view = raw_bank.permute(1, 0, 2)
    semantic_by_view = semantic_bank.permute(1, 0, 2)
    for raw in loader:
        context = _device_images(raw["context"], device)
        targets = raw["target_index"].to(device)
        views = raw["candidate_view"].to(device)
        candidate_raw = raw_by_view[views]
        candidate_semantic = semantic_by_view[views]
        shuffled_context = _audit_shuffle(context, first_index=examples)
        with autocast_context(device, precision):
            full = model.score_encoded_batched(
                model.encode_context(context),
                candidate_raw,
                candidate_semantic,
            )
            suffix = model.score_encoded_batched(
                model.encode_context(context[:, -4:]),
                candidate_raw,
                candidate_semantic,
            )
            shuffled = model.score_encoded_batched(
                model.encode_context(shuffled_context),
                candidate_raw,
                candidate_semantic,
            )
        scores = {
            "full": full,
            "suffix4": suffix,
            "shuffled": shuffled,
            "increment": row_center_scores(full - suffix),
            "shuffled_increment": row_center_scores(shuffled - suffix),
        }
        for name, logits in scores.items():
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                totals[name][key] += value
        all_targets.extend(raw["target_index"].tolist())
        all_contexts.extend(raw["context_text"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("V29 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "context_arms_per_second": examples * len(variants) / max(elapsed, 1e-9),
        "raw_retina_cross_font_identity_top1": raw_identity,
        "frozen_semantic_cross_font_identity_top1": semantic_identity,
        "student_boundary_clean": float(
            student_boundary_is_clean(model, checkpoint)
        ),
        "training_bank_absent_from_checkpoint": float(
            _checkpoint_bank_is_absent(checkpoint)
        ),
    }
    for name in variants:
        metrics[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        metrics[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
    baseline = _baseline_metrics(
        statistics, trigram_rows, all_targets, all_contexts
    )
    for name in ("unigram", "bigram", "trigram"):
        metrics[f"{name}_top1"] = baseline[f"{name}_correct_top1"] / examples
        metrics[f"{name}_top5"] = baseline[f"{name}_correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            baseline[f"{name}_target_log_probability_sum"] / examples
        )
        metrics[f"{name}_context_coverage"] = (
            baseline[f"{name}_coverage"] / examples
        )
    evaluator_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    metrics["evaluator_peak_allocated_vram_gib"] = evaluator_peak
    metrics["peak_allocated_vram_gib"] = max(
        evaluator_peak, float(checkpoint.get("peak_allocated_vram_gib", 0.0))
    )
    return metrics


def _assignment_statistics(
    logits: torch.Tensor,
    assignments: torch.Tensor,
) -> dict[str, torch.Tensor]:
    correct = logits.gather(2, assignments[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignments)[:, :, None])[:, :, 0]
    margins = correct.float() - other.float()
    ties = margins == 0
    credit = (margins > 0).float() + 0.5 * ties.float()
    return {
        "accuracy_sum": credit.sum(),
        "row_accuracy_sum": credit.sum(dim=0),
        "strict_accuracy_sum": (margins > 0).float().sum(),
        "tie_sum": ties.float().sum(),
        "both_correct_sum": (margins > 0).all(dim=1).float().sum(),
        "margin_sum": margins.sum(),
        "arms": torch.tensor(float(margins.numel()), device=logits.device),
        "assignments": torch.tensor(float(margins.shape[0]), device=logits.device),
    }


def _accumulate_assignment(
    totals: dict[str, Any],
    prefix: str,
    statistics: Mapping[str, torch.Tensor],
) -> None:
    for key in (
        "accuracy_sum",
        "strict_accuracy_sum",
        "tie_sum",
        "both_correct_sum",
        "margin_sum",
        "arms",
        "assignments",
    ):
        name = f"{prefix}_{key}"
        totals[name] = totals.get(name, 0.0) + float(statistics[key])
    row_key = f"{prefix}_row_accuracy_sum"
    current = totals.get(row_key, torch.zeros(2, dtype=torch.float64))
    totals[row_key] = current + statistics["row_accuracy_sum"].detach().cpu().double()


def _align_candidates(
    candidates: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    gather = assignments[:, :, None, None, None].expand_as(candidates)
    return candidates.gather(1, gather)


@torch.no_grad()
def _retina_pair_identity(
    model: ConditionalVisualDensityRatioModel,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[float, float]:
    batch = first.shape[0]
    first_visual, _ = model.encode_image_parts(first, target=True)
    second_visual, _ = model.encode_image_parts(second, target=True)
    logits_forward = torch.einsum(
        "bqd,bkd->bqk", first_visual.float(), second_visual.float()
    )
    logits_reverse = logits_forward.transpose(1, 2)
    diagonal = torch.arange(2, device=first.device).expand(batch, 2)
    forward = _assignment_statistics(logits_forward, diagonal)
    reverse = _assignment_statistics(logits_reverse, diagonal)
    arms = float(forward["arms"] + reverse["arms"])
    accuracy = float(
        (forward["accuracy_sum"] + reverse["accuracy_sum"]) / arms
    )
    margin = float((forward["margin_sum"] + reverse["margin_sum"]) / arms)
    return accuracy, margin


def _pair_score_family(
    model: ConditionalVisualDensityRatioModel,
    contexts: torch.Tensor,
    shuffled_contexts: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, torch.Tensor]:
    full = model.score_paired_candidates(contexts, candidates)
    suffix = model.score_exact_suffix_paired(contexts, candidates)
    shuffled = model.score_paired_candidates(shuffled_contexts, candidates)
    return {
        "full": full,
        "suffix4": suffix,
        "shuffled": shuffled,
        "increment": row_center_scores(full - suffix),
        "shuffled_increment": row_center_scores(shuffled - suffix),
    }


@torch.no_grad()
def evaluate_suffix_pairs(
    model: ConditionalVisualDensityRatioModel,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, Any] = {
        "pairs": 0.0,
        "suffix_equal": 0.0,
        "suffix_checks": 0.0,
        "suffix_score_equal": 0.0,
        "suffix_score_checks": 0.0,
        "raw_identity_accuracy_sum": 0.0,
        "raw_identity_margin_sum": 0.0,
        "raw_identity_pairs": 0.0,
    }
    for score_name in ("full", "suffix4", "increment"):
        totals[f"permutation_{score_name}_max_error"] = 0.0
        totals[f"permutation_{score_name}_accuracy_equal"] = 0.0
        totals[f"permutation_{score_name}_checks"] = 0.0
    pair_index = 0
    for raw in loader:
        contexts = _device_images(raw["contexts"], device)
        candidates = _device_images(raw["candidates"], device)
        assignments = raw["assignment"].to(device)
        reference_contexts = _device_images(raw["reference_contexts"], device)
        reference_candidates = _device_images(raw["reference_candidates"], device)
        reference_assignments = raw["reference_assignment"].to(device)
        suffix = int(raw["metadata"][0]["suffix_cells"])
        for visual_contexts in (contexts, reference_contexts):
            equal = (
                visual_contexts[:, 0, -suffix:]
                == visual_contexts[:, 1, -suffix:]
            ).flatten(1).all(dim=1)
            totals["suffix_equal"] += float(equal.sum())
            totals["suffix_checks"] += float(equal.numel())

        for visual_contexts, visual_candidates, labels in (
            (contexts, candidates, assignments),
            (reference_contexts, reference_candidates, reference_assignments),
        ):
            shuffled_contexts = _audit_shuffle(
                visual_contexts, first_index=pair_index
            )
            with autocast_context(device, precision):
                scores = _pair_score_family(
                    model, visual_contexts, shuffled_contexts, visual_candidates
                )
                swapped = _pair_score_family(
                    model,
                    visual_contexts,
                    shuffled_contexts,
                    visual_candidates.flip(1),
                )
            suffix_equal = torch.equal(
                scores["suffix4"][:, 0], scores["suffix4"][:, 1]
            )
            totals["suffix_score_equal"] += float(suffix_equal)
            totals["suffix_score_checks"] += 1.0
            score_statistics: dict[str, dict[str, torch.Tensor]] = {}
            for name, logits in scores.items():
                stats = _assignment_statistics(logits, labels)
                score_statistics[name] = stats
                _accumulate_assignment(totals, name, stats)
            for name in ("full", "suffix4", "increment"):
                error = float((swapped[name] - scores[name].flip(-1)).abs().amax())
                swapped_stats = _assignment_statistics(swapped[name], 1 - labels)
                agreement = float(
                    swapped_stats["accuracy_sum"]
                    == score_statistics[name]["accuracy_sum"]
                )
                totals[f"permutation_{name}_max_error"] = max(
                    totals[f"permutation_{name}_max_error"], error
                )
                totals[f"permutation_{name}_accuracy_equal"] += agreement
                totals[f"permutation_{name}_checks"] += 1.0

        first_aligned = _align_candidates(reference_candidates, reference_assignments)
        second_aligned = _align_candidates(candidates, assignments)
        raw_accuracy, raw_margin = _retina_pair_identity(
            model, first_aligned, second_aligned
        )
        totals["raw_identity_accuracy_sum"] += raw_accuracy * contexts.shape[0]
        totals["raw_identity_margin_sum"] += raw_margin * contexts.shape[0]
        totals["raw_identity_pairs"] += contexts.shape[0]
        totals["pairs"] += contexts.shape[0]
        pair_index += contexts.shape[0]
    if not totals["pairs"]:
        raise ValueError("V29 suffix-pair audit loader is empty")
    metrics: dict[str, float] = {
        "pairs": totals["pairs"],
        "suffix_pixel_equality": totals["suffix_equal"] / totals["suffix_checks"],
        "suffix_score_row_equality": totals["suffix_score_equal"]
        / totals["suffix_score_checks"],
        "raw_retina_two_candidate_identity_accuracy": totals[
            "raw_identity_accuracy_sum"
        ]
        / totals["raw_identity_pairs"],
        "raw_retina_two_candidate_cosine_margin": totals[
            "raw_identity_margin_sum"
        ]
        / totals["raw_identity_pairs"],
    }
    for name in (
        "full",
        "suffix4",
        "shuffled",
        "increment",
        "shuffled_increment",
    ):
        arms = totals[f"{name}_arms"]
        assignments_count = totals[f"{name}_assignments"]
        rows = totals[f"{name}_row_accuracy_sum"] / assignments_count
        metrics[f"{name}_arm_accuracy"] = totals[f"{name}_accuracy_sum"] / arms
        metrics[f"{name}_strict_arm_accuracy"] = (
            totals[f"{name}_strict_accuracy_sum"] / arms
        )
        metrics[f"{name}_tie_rate"] = totals[f"{name}_tie_sum"] / arms
        metrics[f"{name}_both_correct_rate"] = (
            totals[f"{name}_both_correct_sum"] / assignments_count
        )
        metrics[f"{name}_mean_margin"] = totals[f"{name}_margin_sum"] / arms
        metrics[f"{name}_row0_accuracy"] = float(rows[0])
        metrics[f"{name}_row1_accuracy"] = float(rows[1])
        metrics[f"{name}_balanced_row_accuracy"] = float(rows.mean())
    for name in ("full", "suffix4", "increment"):
        metrics[f"candidate_permutation_{name}_max_score_error"] = totals[
            f"permutation_{name}_max_error"
        ]
        metrics[f"candidate_permutation_{name}_accuracy_agreement"] = totals[
            f"permutation_{name}_accuracy_equal"
        ] / totals[f"permutation_{name}_checks"]
    metrics["increment_minus_shuffled_arm_accuracy"] = (
        metrics["increment_arm_accuracy"]
        - metrics["shuffled_increment_arm_accuracy"]
    )
    metrics["increment_minus_shuffled_mean_margin"] = (
        metrics["increment_mean_margin"]
        - metrics["shuffled_increment_mean_margin"]
    )
    return metrics


class _NaturalAuditDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[Any],
        statistics: VisualCharacterStatistics,
    ) -> None:
        self.windows = tuple(windows)
        self.base = FactorizedVisualAuditDataset(windows, statistics.index)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        item["context_text"] = self.windows[index].context
        item["candidate_view"] = (index + 1) % 2
        return item


def _natural_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V29 natural audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "target_index": torch.tensor(
            [item["target_index"] for item in batch], dtype=torch.long
        ),
        "candidate_view": torch.tensor(
            [item["candidate_view"] for item in batch], dtype=torch.long
        ),
        "context_text": [item["context_text"] for item in batch],
    }


def _natural_loader(
    windows: Sequence[Any],
    statistics: VisualCharacterStatistics,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        _NaturalAuditDataset(windows, statistics),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=_natural_audit_collate,
    )


def _pair_loader(
    pairs: Sequence[Any],
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = JointVisualPairDataset(
        pairs,
        split="development",
        render_config=JointVisualRenderConfig(
            augment=False, script_views="original"
        ),
        seed=AUDIT_SEED,
        length=len(pairs),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_pair_collate,
    )


def run_development_audit(
    model: ConditionalVisualDensityRatioModel,
    checkpoint: Mapping[str, Any],
    *,
    manifest: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
    windows: int,
    pair_windows: int,
    bank_size: int,
) -> dict[str, Any]:
    smoke = bool(checkpoint.get("smoke_only"))
    strict = not smoke
    records = load_v25_records(manifest, strict_manifest=strict)
    statistics = build_v29_candidate_statistics(records, bank_size=bank_size)
    natural_windows = build_factorized_audit_windows(
        records,
        allowed_targets=set(statistics.characters),
        count=windows,
        seed=AUDIT_SEED,
    )
    suffix4_pairs = build_joint_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=pair_windows,
        seed=AUDIT_SEED,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
    )
    bank_images = render_visual_character_bank(statistics)
    natural = evaluate_natural_language(
        model,
        _natural_loader(
            natural_windows,
            statistics,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        statistics,
        _trigram_rows(records, statistics),
        bank_images,
        checkpoint,
        device=device,
        precision=precision,
    )
    suffix4 = evaluate_suffix_pairs(
        model,
        _pair_loader(
            suffix4_pairs,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        device=device,
        precision=precision,
    )
    frozen_images_instantiated = False
    mechanism_gates, language_gates = v29_gate_report(
        natural,
        suffix4,
        frozen_images_instantiated=frozen_images_instantiated,
    )
    selected = all(mechanism_gates.values()) and all(language_gates.values())
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint_architecture": checkpoint.get("architecture"),
        "smoke_only": smoke,
        "manifest": verify_v25_manifest(manifest, strict=strict),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": visual_character_statistics_receipt(statistics),
        "natural_windows": len(natural_windows),
        "suffix4_pairs": len(suffix4_pairs),
        "suffix4_pairs_require_different_identifiers": True,
        "audit_seed": AUDIT_SEED,
        "frozen_images_instantiated": frozen_images_instantiated,
        "natural": natural,
        "suffix4": suffix4,
        "mechanism_gates": mechanism_gates,
        "mechanism_selected": all(mechanism_gates.values()),
        "language_gates": language_gates,
        "language_selected": all(language_gates.values()),
        "frozen_evaluation_authorized": selected,
        "writer_training_authorized": selected,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[ConditionalVisualDensityRatioModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V29 conditional visual field")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise PermissionError("V29 smoke checkpoint requires --allow-smoke")
    model = ConditionalVisualDensityRatioModel(
        conditional_visual_density_ratio_config_from_payload(
            checkpoint["model_config"]
        )
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_model_checkpoint(
        args.checkpoint, device=device, allow_smoke=args.allow_smoke
    )
    if not checkpoint.get("smoke_only"):
        if args.windows != NATURAL_WINDOWS or args.pair_windows != PAIR_WINDOWS:
            raise ValueError("V29 evidence audit requires fixed window counts")
        if args.bank_size != AUDIT_BANK_SIZE or args.batch_size != 16:
            raise ValueError("V29 evidence audit requires fixed bank and batch sizes")
        if checkpoint.get("manifest", {}).get("sha256") != V25_MANIFEST_SHA256:
            raise ValueError("V29 checkpoint has the wrong manifest receipt")
    report = run_development_audit(
        model,
        checkpoint,
        manifest=args.manifest,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows=args.windows,
        pair_windows=args.pair_windows,
        bank_size=args.bank_size,
    )
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "device": str(device),
        }
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "development_audit.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
