#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.dense_visual_future_data import (
    JointVisualPairAuditDataset,
    build_joint_suffix_pairs,
    dense_visual_data_boundary_receipt,
    joint_visual_pair_collate,
)
from ilm.visual_lm.dense_visual_future_energy import (
    DenseVisualFutureModel,
    dense_visual_future_boundary_receipt,
    dense_visual_future_config_from_payload,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualAuditDataset,
    build_factorized_audit_windows,
)
from ilm.visual_lm.visual_cell_data import (
    iter_split_writing,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    VisualCharacterStatistics,
    build_visual_character_statistics,
    render_visual_character_bank,
    visual_character_statistics_receipt,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


ARCHITECTURE = "dense-visual-future-energy-v28"
AUDIT_ARCHITECTURE = "dense-visual-future-energy-v28-development-audit"
PROTOCOL_DOCUMENT = "references/dense_visual_future_energy_v28_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/dense_visual_future_energy_v28_evidence/checkpoint_final.pt"
)
DEFAULT_RETINA_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
AUDIT_SEED = 20260919
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered V28 development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--retina-checkpoint", default=DEFAULT_RETINA_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out", default="artifacts/dense_visual_future_energy_v28_audit"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=NATURAL_WINDOWS)
    parser.add_argument("--pair-windows", type=int, default=PAIR_WINDOWS)
    parser.add_argument("--bank-size", type=int, default=AUDIT_BANK_SIZE)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def v28_gate_report(
    natural: Mapping[str, float],
    suffix4: Mapping[str, float],
    *,
    frozen_images_instantiated: bool,
) -> tuple[dict[str, bool], dict[str, bool]]:
    mechanism = {
        "full_pair_arm_accuracy": _strictly_above(
            suffix4["full_arm_accuracy"], 0.65
        ),
        "full_pair_both_correct_rate": _strictly_above(
            suffix4["full_both_correct_rate"], 0.40
        ),
        "full_gain_over_suffix4": _strictly_above(
            suffix4["full_arm_accuracy"] - suffix4["suffix4_arm_accuracy"],
            0.15,
        ),
        "full_gain_over_shuffled": _strictly_above(
            suffix4["full_arm_accuracy"] - suffix4["shuffled_arm_accuracy"],
            0.05,
        ),
        "full_margin_gain_over_shuffled": _strictly_above(
            suffix4["full_mean_margin"] - suffix4["shuffled_mean_margin"],
            0.02,
        ),
        "last_control_at_chance": abs(
            suffix4["last_arm_accuracy"] - 0.5
        ) <= 1e-6,
        "suffix4_control_at_chance": abs(
            suffix4["suffix4_arm_accuracy"] - 0.5
        ) <= 1e-6,
        "suffix_pixels_exact": suffix4["suffix_pixel_equality"] == 1.0,
        "candidate_permutation_equivariant": _strictly_below(
            suffix4["candidate_permutation_max_score_error"], 1e-5
        )
        and suffix4["candidate_permutation_accuracy_agreement"] == 1.0,
        "ema_semantic_cross_font_identity": (
            natural["ema_semantic_cross_font_identity_top1"] >= 0.95
        ),
        "semantic_improves_same_scope_identity": (
            natural["ema_semantic_cross_font_identity_top1"]
            - natural["raw_retina_cross_font_identity_top1"]
            >= 0.02
        ),
        "student_boundary_clean": natural["student_boundary_clean"] == 1.0,
        "peak_allocated_vram_below_18_gib": _strictly_below(
            natural["peak_allocated_vram_gib"], 18.0
        ),
        "frozen_images_not_instantiated": not frozen_images_instantiated,
    }
    language = {
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
        "full_top1_at_least_15_percent": natural["full_top1"] >= 0.15,
    }
    return mechanism, language


def student_boundary_is_clean(model: DenseVisualFutureModel) -> bool:
    model_receipt = dense_visual_future_boundary_receipt(model.config)
    data_receipt = dense_visual_data_boundary_receipt()
    required_model_true = {
        "input_is_continuous_image_stream",
        "candidate_is_arbitrary_image",
        "output_is_continuous_future_distribution",
        "retina_is_frozen",
        "target_semantic_route_is_ema",
    }
    required_data_true = {
        "input_is_continuous_image_stream",
        "canonical_identity_derived_from_exact_pixels",
        "canonical_groups_are_temporary_loss_only",
        "pair_assignment_labels_are_positions",
        "pair_candidate_order_is_randomized",
        "pair_suffix_pixels_identical",
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
    model_extra_false = {
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_glyph_lookup",
    }
    return (
        model_receipt.get("architecture") == ARCHITECTURE
        and data_receipt.get("architecture") == ARCHITECTURE
        and all(model_receipt.get(key) is True for key in required_model_true)
        and all(data_receipt.get(key) is True for key in required_data_true)
        and all(model_receipt.get(key) is False for key in required_false)
        and all(data_receipt.get(key) is False for key in required_false)
        and all(model_receipt.get(key) is False for key in model_extra_false)
    )


def _device_images(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("V28 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


def _render_audit_bank(
    statistics: VisualCharacterStatistics,
) -> torch.Tensor:
    if not isinstance(statistics, VisualCharacterStatistics):
        raise TypeError("V28 audit-bank rendering requires full statistics")
    images = render_visual_character_bank(statistics)
    expected = (len(statistics.characters), 2, 1, 32, 32)
    if tuple(images.shape) != expected:
        raise ValueError(
            f"V28 audit bank has shape {tuple(images.shape)}, expected {expected}"
        )
    return images


@torch.no_grad()
def encode_candidate_bank(
    model: DenseVisualFutureModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    target: bool,
    batch_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V28 visual bank must be [identity,view,1,32,32]")
    flat = images.reshape(-1, 1, 32, 32)
    raw_chunks: list[torch.Tensor] = []
    semantic_chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            raw, semantic = model.encode_image_parts(batch, target=target)
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
    if queries.shape != keys.shape or queries.ndim != 2:
        raise ValueError("V28 cross-font retrieval matrices must align")
    logits = queries.float() @ keys.float().transpose(0, 1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean())


def _top_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = logits.log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(log_probability.detach().sum()),
    }


def _shuffle_prefix(
    context: torch.Tensor,
    *,
    first_index: int,
    preserved_suffix: int = 4,
) -> torch.Tensor:
    prefix = context.shape[1] - preserved_suffix
    if prefix < 2:
        return context
    output = context.clone()
    permutations = []
    for offset in range(context.shape[0]):
        generator = torch.Generator().manual_seed(
            AUDIT_SEED + (first_index + offset) * 104_729
        )
        permutations.append(torch.randperm(prefix, generator=generator))
    permutation = torch.stack(permutations).to(context.device)
    gather = permutation[:, :, None, None, None].expand(
        -1, -1, *context.shape[2:]
    )
    output[:, :prefix] = context[:, :prefix].gather(1, gather)
    return output


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
    output: dict[str, float] = {}
    for name in ("unigram", "bigram", "trigram"):
        output[f"{name}_correct_top1"] = 0.0
        output[f"{name}_correct_top5"] = 0.0
        output[f"{name}_target_log_probability_sum"] = 0.0
        output[f"{name}_coverage"] = 0.0
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
    model: DenseVisualFutureModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    trigram_rows: Mapping[str, Counter[int]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_target, semantic_target = encode_candidate_bank(
        model,
        bank_images,
        device=device,
        precision=precision,
        target=True,
    )
    _, semantic_online = encode_candidate_bank(
        model,
        bank_images,
        device=device,
        precision=precision,
        target=False,
    )
    raw_identity = 0.5 * (
        _cross_font_retrieval_accuracy(raw_target[:, 0], raw_target[:, 1])
        + _cross_font_retrieval_accuracy(raw_target[:, 1], raw_target[:, 0])
    )
    ema_identity = 0.5 * (
        _cross_font_retrieval_accuracy(
            semantic_target[:, 0], semantic_target[:, 1]
        )
        + _cross_font_retrieval_accuracy(
            semantic_target[:, 1], semantic_target[:, 0]
        )
    )
    online_to_ema = 0.5 * (
        _cross_font_retrieval_accuracy(
            semantic_online[:, 0], semantic_target[:, 1]
        )
        + _cross_font_retrieval_accuracy(
            semantic_online[:, 1], semantic_target[:, 0]
        )
    )
    variants = ("full", "last", "suffix4", "shuffled")
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
    for raw in loader:
        context = _device_images(raw["context"], device)
        targets = raw["target_index"].to(device)
        contexts = {
            "full": context,
            "last": context[:, -1:],
            "suffix4": context[:, -4:],
            "shuffled": _shuffle_prefix(context, first_index=examples),
        }
        for name, visual_context in contexts.items():
            with autocast_context(device, precision):
                state = model.encode_context(visual_context)[:, -1]
                distribution = model.future_distribution(state, horizon=1)
                logits = model.score_distribution_shared(
                    distribution,
                    raw_target[:, 1],
                    semantic_target[:, 1],
                )
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                totals[name][key] += value
        all_targets.extend(raw["target_index"].tolist())
        all_contexts.extend(raw["context_text"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("V28 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "context_examples_per_second": examples * len(variants) / max(elapsed, 1e-9),
        "raw_retina_cross_font_identity_top1": raw_identity,
        "ema_semantic_cross_font_identity_top1": ema_identity,
        "online_to_ema_semantic_cross_font_identity_top1": online_to_ema,
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
    metrics["student_boundary_clean"] = float(student_boundary_is_clean(model))
    evaluator_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    metrics["evaluator_peak_allocated_vram_gib"] = evaluator_peak
    metrics["peak_allocated_vram_gib"] = max(
        evaluator_peak, checkpoint_peak_vram_gib
    )
    return metrics


def _assignment_statistics(
    logits: torch.Tensor,
    assignments: torch.Tensor,
) -> dict[str, torch.Tensor]:
    correct = logits.gather(2, assignments[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignments)[:, :, None])[:, :, 0]
    margins = correct - other
    ties = margins == 0
    credit = (margins > 0).float() + 0.5 * ties.float()
    return {
        "accuracy_sum": credit.sum(),
        "strict_accuracy_sum": (margins > 0).float().sum(),
        "tie_sum": ties.float().sum(),
        "both_correct_sum": (margins > 0).all(dim=1).float().sum(),
        "margin_sum": margins.sum(),
        "arms": torch.tensor(float(margins.numel()), device=logits.device),
        "assignments": torch.tensor(float(margins.shape[0]), device=logits.device),
    }


def _shuffle_pair_prefix(
    contexts: torch.Tensor,
    *,
    first_index: int,
    preserved_suffix: int = 4,
) -> torch.Tensor:
    if contexts.ndim != 6 or contexts.shape[1] != 2:
        raise ValueError("V28 pair contexts must have shape [B,2,T,1,32,32]")
    prefix = contexts.shape[2] - preserved_suffix
    if prefix < 2:
        return contexts
    output = contexts.clone()
    for offset in range(contexts.shape[0]):
        generator = torch.Generator().manual_seed(
            AUDIT_SEED + (first_index + offset) * 104_729
        )
        permutation = torch.randperm(prefix, generator=generator).to(
            contexts.device
        )
        output[offset, :, :prefix] = contexts[offset, :, permutation]
    return output


def _align_candidates(
    candidates: torch.Tensor,
    assignments: torch.Tensor,
) -> torch.Tensor:
    gather = assignments[:, :, None, None, None].expand_as(candidates)
    return candidates.gather(1, gather)


@torch.no_grad()
def _retina_pair_identity(
    retina: torch.nn.Module,
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[float, float]:
    batch = first.shape[0]
    first_visual = torch.nn.functional.normalize(
        retina(first.reshape(batch * 2, 1, 32, 32)).float(), dim=-1
    ).reshape(batch, 2, -1)
    second_visual = torch.nn.functional.normalize(
        retina(second.reshape(batch * 2, 1, 32, 32)).float(), dim=-1
    ).reshape(batch, 2, -1)
    logits_forward = torch.einsum(
        "bqd,bkd->bqk", first_visual, second_visual
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


def _accumulate_assignment(
    totals: dict[str, float],
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


@torch.no_grad()
def evaluate_suffix_pairs(
    model: DenseVisualFutureModel,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        "pairs": 0.0,
        "suffix_equal": 0.0,
        "suffix_checks": 0.0,
        "permutation_max_error": 0.0,
        "permutation_accuracy_equal": 0.0,
        "permutation_checks": 0.0,
        "raw_identity_accuracy_sum": 0.0,
        "raw_identity_margin_sum": 0.0,
        "raw_identity_batches": 0.0,
    }
    pair_index = 0
    for raw in loader:
        contexts = _device_images(raw["contexts"], device)
        candidates = _device_images(raw["candidates"], device)
        assignments = raw["assignment"].to(device)
        reference_contexts = _device_images(raw["reference_contexts"], device)
        reference_candidates = _device_images(raw["reference_candidates"], device)
        reference_assignments = raw["reference_assignment"].to(device)
        suffix = int(raw["metadata"][0]["suffix_cells"])
        first_equal = (
            contexts[:, 0, -suffix:] == contexts[:, 1, -suffix:]
        ).flatten(1).all(dim=1)
        second_equal = (
            reference_contexts[:, 0, -suffix:]
            == reference_contexts[:, 1, -suffix:]
        ).flatten(1).all(dim=1)
        totals["suffix_equal"] += float(first_equal.sum() + second_equal.sum())
        totals["suffix_checks"] += float(2 * contexts.shape[0])

        view_inputs = (
            (contexts, candidates, assignments),
            (reference_contexts, reference_candidates, reference_assignments),
        )
        for visual_contexts, visual_candidates, labels in view_inputs:
            variants = {
                "full": visual_contexts,
                "last": visual_contexts[:, :, -1:],
                "suffix4": visual_contexts[:, :, -4:],
                "shuffled": _shuffle_pair_prefix(
                    visual_contexts, first_index=pair_index
                ),
            }
            full_logits: torch.Tensor | None = None
            full_stats: dict[str, torch.Tensor] | None = None
            for name, model_context in variants.items():
                with autocast_context(device, precision):
                    logits = model.score_paired_candidates(
                        model_context, visual_candidates
                    )
                stats = _assignment_statistics(logits, labels)
                _accumulate_assignment(totals, name, stats)
                if name == "full":
                    full_logits = logits
                    full_stats = stats
            if full_logits is None or full_stats is None:
                raise RuntimeError("V28 full pair arm was not evaluated")
            with autocast_context(device, precision):
                swapped_logits = model.score_paired_candidates(
                    visual_contexts, visual_candidates.flip(1)
                )
            expected = full_logits.flip(-1)
            error = float((swapped_logits - expected).abs().amax())
            swapped_stats = _assignment_statistics(swapped_logits, 1 - labels)
            accuracy_equal = float(
                swapped_stats["accuracy_sum"] == full_stats["accuracy_sum"]
            )
            totals["permutation_max_error"] = max(
                totals["permutation_max_error"], error
            )
            totals["permutation_accuracy_equal"] += accuracy_equal
            totals["permutation_checks"] += 1.0

        first_aligned = _align_candidates(
            reference_candidates, reference_assignments
        )
        second_aligned = _align_candidates(candidates, assignments)
        raw_accuracy, raw_margin = _retina_pair_identity(
            model.retina, first_aligned, second_aligned
        )
        totals["raw_identity_accuracy_sum"] += raw_accuracy * contexts.shape[0]
        totals["raw_identity_margin_sum"] += raw_margin * contexts.shape[0]
        totals["raw_identity_batches"] += contexts.shape[0]
        totals["pairs"] += contexts.shape[0]
        pair_index += contexts.shape[0]
    if not totals["pairs"]:
        raise ValueError("V28 suffix-pair audit loader is empty")
    metrics: dict[str, float] = {
        "pairs": totals["pairs"],
        "suffix_pixel_equality": totals["suffix_equal"]
        / totals["suffix_checks"],
        "candidate_permutation_max_score_error": totals[
            "permutation_max_error"
        ],
        "candidate_permutation_accuracy_agreement": totals[
            "permutation_accuracy_equal"
        ]
        / totals["permutation_checks"],
        "raw_retina_two_candidate_identity_accuracy": totals[
            "raw_identity_accuracy_sum"
        ]
        / totals["raw_identity_batches"],
        "raw_retina_two_candidate_cosine_margin": totals[
            "raw_identity_margin_sum"
        ]
        / totals["raw_identity_batches"],
    }
    for name in ("full", "last", "suffix4", "shuffled"):
        arms = totals[f"{name}_arms"]
        assignments = totals[f"{name}_assignments"]
        metrics[f"{name}_arm_accuracy"] = totals[f"{name}_accuracy_sum"] / arms
        metrics[f"{name}_strict_arm_accuracy"] = (
            totals[f"{name}_strict_accuracy_sum"] / arms
        )
        metrics[f"{name}_tie_rate"] = totals[f"{name}_tie_sum"] / arms
        metrics[f"{name}_both_correct_rate"] = (
            totals[f"{name}_both_correct_sum"] / assignments
        )
        metrics[f"{name}_mean_margin"] = totals[f"{name}_margin_sum"] / arms
    metrics["full_minus_suffix4_arm_accuracy"] = (
        metrics["full_arm_accuracy"] - metrics["suffix4_arm_accuracy"]
    )
    metrics["full_minus_shuffled_arm_accuracy"] = (
        metrics["full_arm_accuracy"] - metrics["shuffled_arm_accuracy"]
    )
    metrics["full_minus_shuffled_mean_margin"] = (
        metrics["full_mean_margin"] - metrics["shuffled_mean_margin"]
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
        return item


def _natural_audit_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty V28 natural audit batch")
    return {
        "context": torch.stack([item["context"] for item in batch]),
        "target_index": torch.tensor(
            [item["target_index"] for item in batch], dtype=torch.long
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
    dataset = _NaturalAuditDataset(windows, statistics)
    return DataLoader(
        dataset,
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
    return DataLoader(
        JointVisualPairAuditDataset(pairs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_pair_collate,
    )


def run_development_audit(
    model: DenseVisualFutureModel,
    checkpoint: Mapping[str, Any],
    *,
    manifest: str,
    retina_checkpoint: str,
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
    statistics = build_visual_character_statistics(records, bank_size=bank_size)
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
    bank_images = _render_audit_bank(statistics)
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
        device=device,
        precision=precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
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
    mechanism_gates, language_gates = v28_gate_report(
        natural,
        suffix4,
        frozen_images_instantiated=frozen_images_instantiated,
    )
    retina_digest = file_sha256(retina_checkpoint)
    if strict and retina_digest != EXPECTED_RETINA_SHA256:
        raise ValueError("V28 development audit received the wrong V16 retina")
    selected = all(mechanism_gates.values()) and all(language_gates.values())
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint_architecture": checkpoint.get("architecture"),
        "smoke_only": smoke,
        "manifest": verify_v25_manifest(manifest, strict=strict),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": visual_character_statistics_receipt(statistics),
        "retina_checkpoint": retina_checkpoint,
        "retina_sha256": retina_digest,
        "natural_windows": len(natural_windows),
        "suffix4_pairs": len(suffix4_pairs),
        "suffix4_pairs_require_different_identifiers": True,
        "frozen_images_instantiated": frozen_images_instantiated,
        "natural": natural,
        "suffix4": suffix4,
        "mechanism_gates": mechanism_gates,
        "mechanism_selected": all(mechanism_gates.values()),
        "language_gates": language_gates,
        "language_selected": selected,
        "frozen_evaluation_authorized": selected,
        "writer_training_authorized": selected,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
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
) -> tuple[DenseVisualFutureModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V28 dense visual future model")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise PermissionError("V28 smoke checkpoint requires --allow-smoke")
    model = DenseVisualFutureModel(
        dense_visual_future_config_from_payload(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def main() -> None:
    args = parse_args()
    if min(args.windows, args.pair_windows, args.bank_size, args.batch_size) < 1:
        raise ValueError("V28 audit sizes must be positive")
    seed_everything(AUDIT_SEED)
    device = choose_device(args.device)
    model, checkpoint = load_model_checkpoint(
        args.checkpoint, device=device, allow_smoke=args.allow_smoke
    )
    started = time.monotonic()
    report = run_development_audit(
        model,
        checkpoint,
        manifest=args.manifest,
        retina_checkpoint=args.retina_checkpoint,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows=args.windows,
        pair_windows=args.pair_windows,
        bank_size=args.bank_size,
    )
    report["elapsed_seconds"] = time.monotonic() - started
    report["device"] = str(device)
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_sha256"] = file_sha256(args.checkpoint)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "development_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
