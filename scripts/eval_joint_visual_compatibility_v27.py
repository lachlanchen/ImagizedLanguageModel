#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualAuditDataset,
    build_factorized_audit_windows,
    factorized_visual_audit_collate,
)
from ilm.visual_lm.joint_visual_compatibility import (
    JointVisualCompatibilityModel,
    joint_visual_compatibility_boundary_receipt,
    joint_visual_compatibility_config_from_payload,
)
from ilm.visual_lm.joint_visual_compatibility_data import (
    JointVisualPairAuditDataset,
    build_joint_suffix_pairs,
    joint_visual_data_boundary_receipt,
    joint_visual_pair_collate,
)
from ilm.visual_lm.visual_cell_data import (
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


ARCHITECTURE = "joint-visual-compatibility-v27"
AUDIT_ARCHITECTURE = "joint-visual-compatibility-v27-development-audit"
PROTOCOL_DOCUMENT = "references/joint_visual_compatibility_v27_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/joint_visual_compatibility_v27_evidence/checkpoint_final.pt"
)
DEFAULT_RETINA_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
AUDIT_SEED = 20260915
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered V27 image-only development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--retina-checkpoint", default=DEFAULT_RETINA_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out", default="artifacts/joint_visual_compatibility_v27_audit"
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


def v27_gate_report(
    natural: Mapping[str, float],
    suffix4: Mapping[str, float],
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
        "raw_retina_cross_font_identity": (
            suffix4["raw_retina_cross_font_identity_accuracy"] >= 0.99
        ),
        "learned_cross_font_identity": (
            natural["learned_candidate_cross_font_identity_top1"] >= 0.99
        ),
        "student_boundary_clean": natural["student_boundary_clean"] == 1.0,
        "peak_allocated_vram_below_18_gib": _strictly_below(
            natural["peak_allocated_vram_gib"], 18.0
        ),
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
    }
    return mechanism, language


def student_boundary_is_clean(model: JointVisualCompatibilityModel) -> bool:
    model_receipt = joint_visual_compatibility_boundary_receipt(model.config)
    data_receipt = joint_visual_data_boundary_receipt()
    required_model_true = {
        "input_is_continuous_image_stream",
        "candidate_is_arbitrary_image",
        "output_is_continuous_compatibility",
        "target_route_is_ema",
    }
    required_data_true = {
        "canonical_identity_derived_from_exact_pixels",
        "pair_assignment_labels_are_positions",
        "pair_candidate_order_is_randomized",
        "pair_suffix_pixels_identical",
        "input_is_continuous_image_stream",
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
        raise TypeError("V27 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


@torch.no_grad()
def encode_candidate_bank(
    model: JointVisualCompatibilityModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    target: bool,
    batch_size: int = 128,
) -> torch.Tensor:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V27 visual bank must be [identity,view,1,32,32]")
    flat = images.reshape(-1, 1, 32, 32)
    chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            chunks.append(model.encode_candidates(batch, target=target))
    return torch.cat(chunks).reshape(images.shape[0], images.shape[1], -1)


def _identity_bank_logits(
    query: torch.Tensor,
    bank: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    similarities = torch.einsum("bd,nvd->bnv", query.float(), bank.float())
    return scale.float() * similarities.amax(dim=2)


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


def _baseline_metrics(
    statistics: VisualCharacterStatistics,
    targets: Sequence[int],
    last_characters: Sequence[str],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    width = len(statistics.characters)
    unigram = torch.tensor(statistics.counts, dtype=torch.float64) + alpha
    unigram /= unigram.sum()
    output: dict[str, float] = {}
    for name in ("unigram", "bigram"):
        output[f"{name}_correct_top1"] = 0.0
        output[f"{name}_correct_top5"] = 0.0
        output[f"{name}_target_log_probability_sum"] = 0.0
    unigram_top = unigram.topk(min(5, width)).indices
    for target, previous in zip(targets, last_characters):
        output["unigram_correct_top1"] += float(unigram_top[0] == target)
        output["unigram_correct_top5"] += float((unigram_top == target).any())
        output["unigram_target_log_probability_sum"] += math.log(
            float(unigram[target])
        )
        sparse = statistics.bigram_rows.get(previous)
        if sparse:
            row = torch.full((width,), alpha, dtype=torch.float64)
            for index, count in sparse:
                row[index] += count
            row /= row.sum()
        else:
            row = unigram
        top = row.topk(min(5, width)).indices
        output["bigram_correct_top1"] += float(top[0] == target)
        output["bigram_correct_top5"] += float((top == target).any())
        output["bigram_target_log_probability_sum"] += math.log(float(row[target]))
    return output


def _cross_font_retrieval_accuracy(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    if queries.shape != keys.shape or queries.ndim != 2:
        raise ValueError("cross-font retrieval matrices must align")
    logits = queries.float() @ keys.float().transpose(0, 1)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean())


@torch.no_grad()
def evaluate_natural_language(
    model: JointVisualCompatibilityModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_bank = encode_candidate_bank(
        model,
        bank_images,
        device=device,
        precision=precision,
        target=True,
    )
    online_bank = encode_candidate_bank(
        model,
        bank_images,
        device=device,
        precision=precision,
        target=False,
    )
    online_to_ema = 0.5 * (
        _cross_font_retrieval_accuracy(online_bank[:, 0], target_bank[:, 1])
        + _cross_font_retrieval_accuracy(online_bank[:, 1], target_bank[:, 0])
    )
    ema_to_ema = 0.5 * (
        _cross_font_retrieval_accuracy(target_bank[:, 0], target_bank[:, 1])
        + _cross_font_retrieval_accuracy(target_bank[:, 1], target_bank[:, 0])
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
    all_last_characters: list[str] = []
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
                query = model.encode_context(visual_context)
            logits = _identity_bank_logits(
                query, target_bank, model.compatibility_scale
            )
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                totals[name][key] += value
        all_targets.extend(raw["target_index"].tolist())
        all_last_characters.extend(raw["last_character"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("V27 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "context_examples_per_second": examples * len(variants) / max(elapsed, 1e-9),
        "online_to_ema_cross_font_identity_top1": online_to_ema,
        "ema_to_ema_cross_font_identity_top1": ema_to_ema,
        "learned_candidate_cross_font_identity_top1": min(
            online_to_ema, ema_to_ema
        ),
    }
    for name in variants:
        metrics[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        metrics[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
    baseline = _baseline_metrics(statistics, all_targets, all_last_characters)
    for name in ("unigram", "bigram"):
        metrics[f"{name}_top1"] = baseline[f"{name}_correct_top1"] / examples
        metrics[f"{name}_top5"] = baseline[f"{name}_correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            baseline[f"{name}_target_log_probability_sum"] / examples
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
        raise ValueError("V27 pair contexts must have shape [B,2,T,1,32,32]")
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
    model: JointVisualCompatibilityModel,
    raw_retina: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    raw_retina.eval()
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
                raise RuntimeError("V27 full pair arm was not evaluated")
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
            raw_retina, first_aligned, second_aligned
        )
        totals["raw_identity_accuracy_sum"] += raw_accuracy * contexts.shape[0]
        totals["raw_identity_margin_sum"] += raw_margin * contexts.shape[0]
        totals["raw_identity_batches"] += contexts.shape[0]
        totals["pairs"] += contexts.shape[0]
        pair_index += contexts.shape[0]
    if not totals["pairs"]:
        raise ValueError("V27 suffix-pair audit loader is empty")
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
        "raw_retina_cross_font_identity_accuracy": totals[
            "raw_identity_accuracy_sum"
        ]
        / totals["raw_identity_batches"],
        "raw_retina_cross_font_cosine_margin": totals[
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


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[JointVisualCompatibilityModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V27 visual compatibility model")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise ValueError("smoke-only V27 checkpoint requires --allow-smoke")
    config = joint_visual_compatibility_config_from_payload(
        checkpoint["model_config"]
    )
    model = JointVisualCompatibilityModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def load_raw_v16_retina(
    model: JointVisualCompatibilityModel,
    path: str | Path,
    *,
    device: torch.device,
    require_expected_hash: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    digest = file_sha256(path)
    if require_expected_hash and digest != EXPECTED_RETINA_SHA256:
        raise ValueError(
            f"V27 requires raw V16 retina {EXPECTED_RETINA_SHA256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "predictive-visual-field-state-flow-v1":
        raise ValueError("V27 raw-retina source has the wrong architecture")
    state = {
        name.removeprefix("retina."): value
        for name, value in checkpoint["model"].items()
        if name.startswith("retina.")
    }
    raw_retina = copy.deepcopy(model.target_retina)
    raw_retina.load_state_dict(state, strict=True)
    raw_retina.requires_grad_(False).eval().to(device)
    return raw_retina, {
        "checkpoint": str(path),
        "sha256": digest,
        "source_architecture": checkpoint["architecture"],
        "source_step": checkpoint.get("global_step"),
    }


def _natural_loader(
    records: Sequence[Any],
    statistics: VisualCharacterStatistics,
    *,
    windows: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, tuple[Any, ...]]:
    selected = build_factorized_audit_windows(
        records,
        allowed_targets=set(statistics.characters),
        count=windows,
        seed=AUDIT_SEED,
    )
    dataset = FactorizedVisualAuditDataset(selected, statistics.index)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=factorized_visual_audit_collate,
    ), selected


def _pair_loader(
    records: Sequence[Any],
    statistics: VisualCharacterStatistics,
    *,
    count: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, tuple[Any, ...]]:
    pairs = build_joint_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=count,
        seed=AUDIT_SEED,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
    )
    dataset = JointVisualPairAuditDataset(pairs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_pair_collate,
    ), pairs


def run_development_audit(
    model: JointVisualCompatibilityModel,
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
    strict = not bool(checkpoint.get("exploratory"))
    records = load_v25_records(manifest, strict_manifest=strict)
    statistics = build_visual_character_statistics(records, bank_size=bank_size)
    natural_loader, natural_windows = _natural_loader(
        records,
        statistics,
        windows=windows,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    pair_loader, suffix4_pairs = _pair_loader(
        records,
        statistics,
        count=pair_windows,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bank_images = render_visual_character_bank(statistics)
    raw_retina, raw_retina_receipt = load_raw_v16_retina(
        model,
        retina_checkpoint,
        device=device,
        require_expected_hash=strict,
    )
    natural = evaluate_natural_language(
        model,
        natural_loader,
        statistics,
        bank_images,
        device=device,
        precision=precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    suffix4 = evaluate_suffix_pairs(
        model,
        raw_retina,
        pair_loader,
        device=device,
        precision=precision,
    )
    mechanism_gates, language_gates = v27_gate_report(natural, suffix4)
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_smoke_only": bool(checkpoint.get("smoke_only")),
        "manifest": verify_v25_manifest(manifest, strict=strict),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": visual_character_statistics_receipt(statistics),
        "raw_retina": raw_retina_receipt,
        "natural_windows": len(natural_windows),
        "suffix4_pairs": len(suffix4_pairs),
        "suffix4_pairs_require_different_identifiers": True,
        "frozen_images_instantiated": False,
        "natural": natural,
        "suffix4": suffix4,
        "mechanism_gates": mechanism_gates,
        "mechanism_selected": all(mechanism_gates.values()),
        "language_gates": language_gates,
        "language_selected": all(mechanism_gates.values())
        and all(language_gates.values()),
        "frozen_evaluation_authorized": all(mechanism_gates.values())
        and all(language_gates.values()),
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


def main() -> None:
    args = parse_args()
    if min(args.windows, args.pair_windows, args.bank_size, args.batch_size) < 1:
        raise ValueError("V27 audit sizes must be positive")
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
