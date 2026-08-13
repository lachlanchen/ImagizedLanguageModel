#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

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
from ilm.visual_lm.spatial_visual_next_field import (
    V30_ARCHITECTURE,
    V30_GLOBAL_ROUTE,
    V30_SPATIAL_ROUTE,
    SpatialVisualNextFieldModel,
    model_state_sha256,
    spatial_visual_next_field_boundary_receipt,
    spatial_visual_next_field_config_from_payload,
    spatially_permute_candidate_fields,
)
from ilm.visual_lm.spatial_visual_next_field_data import (
    build_v30_candidate_statistics,
    spatial_visual_data_boundary_receipt,
)
from ilm.visual_lm.spatial_visual_next_field_training import shuffle_visual_prefix
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
)


ARCHITECTURE = V30_ARCHITECTURE
AUDIT_ARCHITECTURE = "spatial-visual-next-field-v30-development-audit"
PROTOCOL_DOCUMENT = "references/spatial_visual_next_field_v30_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "81d2b2af1eb3a305b4acd1028c004ddddc607e826eea1d50b6d137d32ed180a5"
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_SPATIAL_CHECKPOINT = (
    "artifacts/spatial_visual_next_field_v30_spatial_evidence/checkpoint_final.pt"
)
DEFAULT_GLOBAL_CHECKPOINT = (
    "artifacts/spatial_visual_next_field_v30_global_control_evidence/"
    "checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/spatial_visual_next_field_v30_evidence"
AUDIT_SEED = 20261014
MODEL_SEED = 20261010
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
GATE_EPSILON = 1e-12
SCORE_NAMES = ("full", "suffix4", "shuffled", "spatial_permuted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered joint V30 route audit."
    )
    parser.add_argument("--spatial-checkpoint", default=DEFAULT_SPATIAL_CHECKPOINT)
    parser.add_argument("--global-checkpoint", default=DEFAULT_GLOBAL_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
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


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def _all_finite(values: Mapping[str, Any]) -> bool:
    return all(
        math.isfinite(float(value))
        for value in values.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _checkpoint_bank_is_absent(checkpoint: Mapping[str, Any]) -> bool:
    if checkpoint.get("deployed_state_includes_training_candidate_images") is not False:
        return False
    if checkpoint.get("deployed_state_includes_training_form_labels") is not False:
        return False
    state = checkpoint.get("model")
    return isinstance(state, Mapping) and not any(
        "bank" in str(name).lower() for name in state
    )


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def final_checkpoint_is_clean(checkpoint: Mapping[str, Any]) -> bool:
    non_model_tensors = any(
        _contains_tensor(value) for key, value in checkpoint.items() if key != "model"
    )
    bank = checkpoint.get("candidate_bank_receipt", {})
    return (
        checkpoint.get("optimizer") is None
        and checkpoint.get("rng_state") is None
        and checkpoint.get("resumable") is False
        and not non_model_tensors
        and _checkpoint_bank_is_absent(checkpoint)
        and bank.get("images_in_checkpoint") is False
        and bank.get("forms_in_checkpoint") is False
        and bank.get("inference_requires_bank") is False
    )


def student_boundary_is_clean(
    model: SpatialVisualNextFieldModel,
    checkpoint: Mapping[str, Any],
) -> bool:
    model_receipt = spatial_visual_next_field_boundary_receipt(model.config)
    data_receipt = spatial_visual_data_boundary_receipt()
    required_model_true = {
        "input_is_continuous_image_stream",
        "output_is_candidate_independent_continuous_field",
        "candidate_is_arbitrary_image",
        "retina_is_frozen",
        "semantic_adapters_are_frozen",
    }
    required_data_true = {
        "input_is_continuous_image_stream",
        "output_is_candidate_independent_continuous_field",
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
    model_only_false = {
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_glyph_lookup",
        "candidate_bank_in_model_state",
    }
    route_reduction = model_receipt.get(
        "candidate_reduction_occurs_after_local_interaction"
    )
    expected_reduction = model.config.route_mode == V30_SPATIAL_ROUTE
    return (
        model_receipt.get("architecture") == ARCHITECTURE
        and data_receipt.get("architecture") == ARCHITECTURE
        and checkpoint.get("route_mode") == model.config.route_mode
        and route_reduction is expected_reduction
        and all(model_receipt.get(key) is True for key in required_model_true)
        and all(data_receipt.get(key) is True for key in required_data_true)
        and all(model_receipt.get(key) is False for key in required_false)
        and all(data_receipt.get(key) is False for key in required_false)
        and all(model_receipt.get(key) is False for key in model_only_false)
        and _checkpoint_bank_is_absent(checkpoint)
    )


def _device_images(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("V30 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


def _update_tensor_digest(
    digest: Any,
    name: str,
    value: torch.Tensor,
) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())


def _tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    _update_tensor_digest(digest, "tensor", value)
    return digest.hexdigest()


@torch.no_grad()
def encode_candidate_bank(
    model: SpatialVisualNextFieldModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> torch.Tensor:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V30 visual bank must be [identity,view,1,32,32]")
    flat = images.reshape(-1, 1, 32, 32)
    chunks: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            chunks.append(model.encode_route_candidates(batch))
    return torch.cat(chunks).reshape(images.shape[0], images.shape[1], 16, -1)


def _cross_font_retrieval_accuracy(
    queries: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    logits = (
        torch.einsum("npc,mpc->nm", queries.float(), keys.float()) / queries.shape[1]
    )
    targets = torch.arange(logits.shape[0], device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean())


def _top_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = (
        logits.float().log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    )
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
        output["unigram_target_log_probability_sum"] += math.log(float(unigram[target]))
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
            output[f"{name}_target_log_probability_sum"] += math.log(float(row[target]))
    return output


@torch.no_grad()
def evaluate_natural_language(
    model: SpatialVisualNextFieldModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    trigram_rows: Mapping[str, Counter[int]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    bank = encode_candidate_bank(model, bank_images, device=device, precision=precision)
    visibility = 0.5 * (
        _cross_font_retrieval_accuracy(bank[:, 0], bank[:, 1])
        + _cross_font_retrieval_accuracy(bank[:, 1], bank[:, 0])
    )
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
        }
        for name in SCORE_NAMES
    }
    all_targets: list[int] = []
    all_contexts: list[str] = []
    examples = 0
    output_shape_clean = True
    scores_finite = True
    spatial_permutation_max_error = 0.0
    pixel_digest = hashlib.sha256()
    started = time.monotonic()
    bank_by_view = bank.permute(1, 0, 2, 3)
    for raw in loader:
        _update_tensor_digest(pixel_digest, "context", raw["context"])
        _update_tensor_digest(pixel_digest, "target_index", raw["target_index"])
        _update_tensor_digest(pixel_digest, "candidate_view", raw["candidate_view"])
        context = _device_images(raw["context"], device)
        targets = raw["target_index"].to(device)
        views = raw["candidate_view"].to(device)
        candidate_fields = bank_by_view[views]
        permuted_fields = spatially_permute_candidate_fields(candidate_fields)
        shuffled_context = _audit_shuffle(context, first_index=examples)
        with autocast_context(device, precision):
            full_prediction = model.predict_field(context)
            suffix_prediction = model.predict_field(context[:, -4:])
            shuffled_prediction = model.predict_field(shuffled_context)
            scores = {
                "full": model.score_encoded_batched(full_prediction, candidate_fields),
                "suffix4": model.score_encoded_batched(
                    suffix_prediction, candidate_fields
                ),
                "shuffled": model.score_encoded_batched(
                    shuffled_prediction, candidate_fields
                ),
                "spatial_permuted": model.score_encoded_batched(
                    full_prediction, permuted_fields
                ),
            }
        output_shape_clean = output_shape_clean and full_prediction.shape[1:] == (
            16,
            192,
        )
        for name, logits in scores.items():
            scores_finite = scores_finite and bool(torch.isfinite(logits).all())
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                totals[name][key] += value
        spatial_permutation_max_error = max(
            spatial_permutation_max_error,
            float((scores["spatial_permuted"] - scores["full"]).abs().amax()),
        )
        all_targets.extend(raw["target_index"].tolist())
        all_contexts.extend(raw["context_text"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("V30 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, Any] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "context_arms_per_second": examples * len(SCORE_NAMES) / max(elapsed, 1e-9),
        "candidate_cross_font_identity_top1": visibility,
        "spatial_permutation_max_score_error": spatial_permutation_max_error,
        "candidate_independent_output_shape_clean": float(output_shape_clean),
        "all_scores_finite": float(scores_finite),
        "rendered_batch_sha256": pixel_digest.hexdigest(),
    }
    for name in SCORE_NAMES:
        metrics[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        metrics[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
    baseline = _baseline_metrics(statistics, trigram_rows, all_targets, all_contexts)
    for name in ("unigram", "bigram", "trigram"):
        metrics[f"{name}_top1"] = baseline[f"{name}_correct_top1"] / examples
        metrics[f"{name}_top5"] = baseline[f"{name}_correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            baseline[f"{name}_target_log_probability_sum"] / examples
        )
        metrics[f"{name}_context_coverage"] = baseline[f"{name}_coverage"] / examples
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


@torch.no_grad()
def _pair_score_family(
    model: SpatialVisualNextFieldModel,
    contexts: torch.Tensor,
    shuffled_contexts: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, torch.Tensor]:
    fields = model.encode_route_candidates(candidates)
    full_prediction = model.predict_paired_fields(contexts)
    shuffled_prediction = model.predict_paired_fields(shuffled_contexts)
    return {
        "full": model.score_encoded_paired(full_prediction, fields),
        "suffix4": model.score_exact_suffix_paired(contexts, candidates),
        "shuffled": model.score_encoded_paired(shuffled_prediction, fields),
        "spatial_permuted": model.score_encoded_paired(
            full_prediction,
            spatially_permute_candidate_fields(fields),
        ),
    }


@torch.no_grad()
def evaluate_suffix_pairs(
    model: SpatialVisualNextFieldModel,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    totals: dict[str, Any] = {
        "pairs": 0.0,
        "suffix_equal": 0.0,
        "suffix_checks": 0.0,
        "suffix_score_max_error": 0.0,
        "spatial_permutation_max_error": 0.0,
        "scores_finite": 1.0,
    }
    for score_name in SCORE_NAMES:
        totals[f"permutation_{score_name}_max_error"] = 0.0
        totals[f"permutation_{score_name}_accuracy_equal"] = 0.0
        totals[f"permutation_{score_name}_checks"] = 0.0
    pair_index = 0
    pixel_digest = hashlib.sha256()
    for raw in loader:
        for key in (
            "contexts",
            "candidates",
            "assignment",
            "reference_contexts",
            "reference_candidates",
            "reference_assignment",
        ):
            _update_tensor_digest(pixel_digest, key, raw[key])
        contexts = _device_images(raw["contexts"], device)
        candidates = _device_images(raw["candidates"], device)
        assignments = raw["assignment"].to(device)
        reference_contexts = _device_images(raw["reference_contexts"], device)
        reference_candidates = _device_images(raw["reference_candidates"], device)
        reference_assignments = raw["reference_assignment"].to(device)
        suffix = int(raw["metadata"][0]["suffix_cells"])
        for visual_contexts in (contexts, reference_contexts):
            equal = (
                (visual_contexts[:, 0, -suffix:] == visual_contexts[:, 1, -suffix:])
                .flatten(1)
                .all(dim=1)
            )
            totals["suffix_equal"] += float(equal.sum())
            totals["suffix_checks"] += float(equal.numel())

        for visual_contexts, visual_candidates, labels in (
            (contexts, candidates, assignments),
            (reference_contexts, reference_candidates, reference_assignments),
        ):
            shuffled_contexts = _audit_shuffle(visual_contexts, first_index=pair_index)
            with autocast_context(device, precision):
                scores = _pair_score_family(
                    model,
                    visual_contexts,
                    shuffled_contexts,
                    visual_candidates,
                )
                swapped = _pair_score_family(
                    model,
                    visual_contexts,
                    shuffled_contexts,
                    visual_candidates.flip(1),
                )
            totals["suffix_score_max_error"] = max(
                totals["suffix_score_max_error"],
                float((scores["suffix4"][:, 0] - scores["suffix4"][:, 1]).abs().amax()),
            )
            totals["spatial_permutation_max_error"] = max(
                totals["spatial_permutation_max_error"],
                float((scores["spatial_permuted"] - scores["full"]).abs().amax()),
            )
            score_statistics: dict[str, dict[str, torch.Tensor]] = {}
            for name, logits in scores.items():
                totals["scores_finite"] *= float(torch.isfinite(logits).all())
                stats = _assignment_statistics(logits, labels)
                score_statistics[name] = stats
                _accumulate_assignment(totals, name, stats)
            for name in SCORE_NAMES:
                totals["scores_finite"] *= float(torch.isfinite(swapped[name]).all())
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

        totals["pairs"] += contexts.shape[0]
        pair_index += contexts.shape[0]
    if not totals["pairs"]:
        raise ValueError("V30 suffix-pair audit loader is empty")
    metrics: dict[str, Any] = {
        "pairs": totals["pairs"],
        "suffix_pixel_equality": totals["suffix_equal"] / totals["suffix_checks"],
        "suffix_score_row_max_error": totals["suffix_score_max_error"],
        "spatial_permutation_max_score_error": totals["spatial_permutation_max_error"],
        "all_scores_finite": totals["scores_finite"],
        "rendered_batch_sha256": pixel_digest.hexdigest(),
    }
    for name in SCORE_NAMES:
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
        metrics[f"candidate_permutation_{name}_max_score_error"] = totals[
            f"permutation_{name}_max_error"
        ]
        metrics[f"candidate_permutation_{name}_accuracy_agreement"] = (
            totals[f"permutation_{name}_accuracy_equal"]
            / totals[f"permutation_{name}_checks"]
        )
    metrics["full_minus_shuffled_arm_accuracy"] = (
        metrics["full_arm_accuracy"] - metrics["shuffled_arm_accuracy"]
    )
    metrics["full_minus_shuffled_mean_margin"] = (
        metrics["full_mean_margin"] - metrics["shuffled_mean_margin"]
    )
    metrics["full_minus_spatial_permuted_arm_accuracy"] = (
        metrics["full_arm_accuracy"] - metrics["spatial_permuted_arm_accuracy"]
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
        raise ValueError("cannot collate an empty V30 natural audit batch")
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
        render_config=JointVisualRenderConfig(augment=False, script_views="original"),
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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _natural_window_receipt(windows: Sequence[Any]) -> dict[str, Any]:
    payload = [
        {
            "identifier": item.identifier,
            "script_view": item.script_view,
            "context": item.context,
            "future": item.future,
        }
        for item in windows
    ]
    return {"count": len(payload), "sha256": _json_sha256(payload)}


def _pair_receipt(pairs: Sequence[Any]) -> dict[str, Any]:
    payload = [
        {
            "suffix_cells": item.suffix_cells,
            "identifier_a": item.identifier_a,
            "identifier_b": item.identifier_b,
            "script_view_a": item.script_view_a,
            "script_view_b": item.script_view_b,
            "context_a": item.context_a,
            "context_b": item.context_b,
            "target_a": item.target_a,
            "target_b": item.target_b,
        }
        for item in pairs
    ]
    return {
        "count": len(payload),
        "suffix_cells": 4,
        "require_different_identifiers": True,
        "sha256": _json_sha256(payload),
    }


def _statistics_receipt(
    statistics: VisualCharacterStatistics,
    bank_images: torch.Tensor,
) -> dict[str, Any]:
    return {
        "bank_size": len(statistics.characters),
        "character_sequence_sha256": hashlib.sha256(
            "".join(statistics.characters).encode("utf-8")
        ).hexdigest(),
        "counts_sha256": _json_sha256(list(statistics.counts)),
        "rendered_images_sha256": _tensor_sha256(bank_images),
        "development_views": int(bank_images.shape[1]),
        "student_receives_labels": False,
        "evaluator_only": True,
    }


def _model_state_is_finite(checkpoint: Mapping[str, Any]) -> bool:
    state = checkpoint.get("model", {})
    return isinstance(state, Mapping) and all(
        isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
        for value in state.values()
    )


def _training_metrics_are_finite(checkpoint: Mapping[str, Any]) -> bool:
    metrics = checkpoint.get("training_metrics", {})
    return isinstance(metrics, Mapping) and bool(metrics) and _all_finite(metrics)


def _route_integrity(
    model: SpatialVisualNextFieldModel,
    checkpoint: Mapping[str, Any],
    natural: Mapping[str, Any],
    suffix4: Mapping[str, Any],
    *,
    peak_allocated_vram_gib: float,
) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "model_state_finite": _model_state_is_finite(checkpoint),
        "training_metrics_finite": _training_metrics_are_finite(checkpoint),
        "scores_finite": (
            natural["all_scores_finite"] == 1.0 and suffix4["all_scores_finite"] == 1.0
        ),
        "student_boundary_clean": student_boundary_is_clean(model, checkpoint),
        "final_checkpoint_clean": final_checkpoint_is_clean(checkpoint),
        "candidate_independent_output_shape_clean": (
            natural["candidate_independent_output_shape_clean"] == 1.0
        ),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "parameter_cap_clean": total < 20_000_000 and trainable < 18_500_000,
        "peak_allocated_vram_gib": peak_allocated_vram_gib,
        "memory_cap_clean": _strictly_below(peak_allocated_vram_gib, 18.0),
        "step": int(checkpoint.get("step", -1)),
        "finite_updates_verified": int(checkpoint.get("finite_updates_verified", -1)),
    }


def _candidate_column_gate(metrics: Mapping[str, Any]) -> bool:
    return all(
        _strictly_below(metrics[f"candidate_permutation_{name}_max_score_error"], 1e-5)
        and metrics[f"candidate_permutation_{name}_accuracy_agreement"] == 1.0
        for name in SCORE_NAMES
    )


def _common_integrity_gates(route: Mapping[str, Any]) -> dict[str, bool]:
    integrity = route["integrity"]
    return {
        "all_model_training_and_score_values_finite": (
            integrity["model_state_finite"]
            and integrity["training_metrics_finite"]
            and integrity["scores_finite"]
        ),
        "student_boundary_clean": integrity["student_boundary_clean"],
        "final_checkpoint_clean": integrity["final_checkpoint_clean"],
        "candidate_independent_output_shape_clean": integrity[
            "candidate_independent_output_shape_clean"
        ],
        "parameter_caps_clean": integrity["parameter_cap_clean"],
        "peak_allocated_vram_below_18_gib": integrity["memory_cap_clean"],
    }


def v30_gate_report(
    spatial: Mapping[str, Any],
    global_control: Mapping[str, Any],
    matched: Mapping[str, Any],
    *,
    frozen_images_instantiated: bool,
) -> tuple[
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
    dict[str, bool],
]:
    spatial_natural = spatial["natural"]
    spatial_pairs = spatial["suffix4"]
    global_natural = global_control["natural"]
    global_pairs = global_control["suffix4"]
    spatial_common = _common_integrity_gates(spatial) | {
        "aligned_spatial_cross_font_identity": (
            spatial_natural["candidate_cross_font_identity_top1"] >= 0.95
        ),
        "suffix_pixels_exact": spatial_pairs["suffix_pixel_equality"] == 1.0,
        "suffix_score_rows_exact": _strictly_below(
            spatial_pairs["suffix_score_row_max_error"], 1e-6
        ),
        "candidate_column_permutation_equivariant": _candidate_column_gate(
            spatial_pairs
        ),
        "full_pair_arm_accuracy": _strictly_above(
            spatial_pairs["full_arm_accuracy"], 0.65
        ),
        "full_pair_both_correct": _strictly_above(
            spatial_pairs["full_both_correct_rate"], 0.40
        ),
        "pair_accuracy_gain_over_shuffled": _strictly_above(
            spatial_pairs["full_minus_shuffled_arm_accuracy"], 0.10
        ),
        "pair_margin_gain_over_shuffled": _strictly_above(
            spatial_pairs["full_minus_shuffled_mean_margin"], 0.05
        ),
        "natural_logp_gain_over_shuffled": _strictly_above(
            spatial_natural["full_target_log_probability"]
            - spatial_natural["shuffled_target_log_probability"],
            0.03,
        ),
        "natural_logp_gain_over_spatial_permutation": _strictly_above(
            spatial_natural["full_target_log_probability"]
            - spatial_natural["spatial_permuted_target_log_probability"],
            0.05,
        ),
        "pair_accuracy_gain_over_spatial_permutation": _strictly_above(
            spatial_pairs["full_minus_spatial_permuted_arm_accuracy"], 0.05
        ),
        "frozen_images_not_instantiated": not frozen_images_instantiated,
    }
    global_integrity = _common_integrity_gates(global_control) | {
        "global_semantic_cross_font_identity": (
            global_natural["candidate_cross_font_identity_top1"] >= 0.95
        ),
        "suffix_pixels_exact": global_pairs["suffix_pixel_equality"] == 1.0,
        "suffix_score_rows_exact": _strictly_below(
            global_pairs["suffix_score_row_max_error"], 1e-6
        ),
        "candidate_column_permutation_equivariant": _candidate_column_gate(
            global_pairs
        ),
        "candidate_spatial_permutation_invariant": _strictly_below(
            max(
                global_natural["spatial_permutation_max_score_error"],
                global_pairs["spatial_permutation_max_score_error"],
            ),
            1e-6,
        ),
        "frozen_images_not_instantiated": not frozen_images_instantiated,
    }
    matched_gates = {
        "initialized_parameter_states_exact": matched[
            "initialized_parameter_states_exact"
        ],
        "final_parameter_counts_exact": matched["final_parameter_counts_exact"],
        "source_and_data_receipts_exact": matched["source_and_data_receipts_exact"],
        "audit_windows_and_pixels_exact": matched["audit_windows_and_pixels_exact"],
        "both_arms_completed_8000_finite_updates": matched[
            "both_arms_completed_8000_finite_updates"
        ],
        "spatial_pair_accuracy_gain_over_global": _strictly_above(
            spatial_pairs["full_arm_accuracy"] - global_pairs["full_arm_accuracy"],
            0.05,
        ),
        "spatial_pair_both_correct_gain_over_global": _strictly_above(
            spatial_pairs["full_both_correct_rate"]
            - global_pairs["full_both_correct_rate"],
            0.05,
        ),
        "spatial_natural_top1_gain_over_global": _strictly_above(
            spatial_natural["full_top1"] - global_natural["full_top1"], 0.01
        ),
        "spatial_natural_logp_gain_over_global": _strictly_above(
            spatial_natural["full_target_log_probability"]
            - global_natural["full_target_log_probability"],
            0.05,
        ),
    }
    language = {
        "natural_full_top1_at_least_15_percent": spatial_natural["full_top1"] >= 0.15,
        "full_top1_gain_over_suffix4": _strictly_above(
            spatial_natural["full_top1"] - spatial_natural["suffix4_top1"], 0.03
        ),
        "full_top1_gain_over_shuffled": _strictly_above(
            spatial_natural["full_top1"] - spatial_natural["shuffled_top1"],
            0.03,
        ),
        "full_top1_gain_over_unigram": _strictly_above(
            spatial_natural["full_top1"] - spatial_natural["unigram_top1"],
            0.03,
        ),
        "full_top1_gain_over_bigram": _strictly_above(
            spatial_natural["full_top1"] - spatial_natural["bigram_top1"],
            0.01,
        ),
        "full_logp_gain_over_bigram": _strictly_above(
            spatial_natural["full_target_log_probability"]
            - spatial_natural["bigram_target_log_probability"],
            0.05,
        ),
        "full_pair_arm_accuracy": _strictly_above(
            spatial_pairs["full_arm_accuracy"], 0.65
        ),
        "full_pair_both_correct": _strictly_above(
            spatial_pairs["full_both_correct_rate"], 0.40
        ),
    }
    return spatial_common, global_integrity, matched_gates, language


def load_model_checkpoint(
    path: str | Path,
    *,
    expected_route: str,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[SpatialVisualNextFieldModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{path} is not a V30 checkpoint")
    if checkpoint.get("route_mode") != expected_route:
        raise ValueError(f"{path} is not the {expected_route} arm")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise PermissionError("V30 smoke checkpoints require --allow-smoke")
    model = SpatialVisualNextFieldModel(
        spatial_visual_next_field_config_from_payload(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def _reconstruct_initial_state(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    initialization = checkpoint.get("initialization", {})
    source_path = initialization.get("checkpoint")
    if not isinstance(source_path, str):
        raise ValueError("V30 checkpoint has no V29 initialization path")
    source_digest = file_sha256(source_path)
    if source_digest != initialization.get("sha256"):
        raise ValueError("V30 V29 source no longer matches its receipt")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("architecture") != "conditional-visual-density-ratio-v29":
        raise ValueError("V30 initialization source is not V29")
    config = spatial_visual_next_field_config_from_payload(checkpoint["model_config"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = SpatialVisualNextFieldModel(config)
        loaded = model.load_v29_backbone_state(source["model"])
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }
    digest = model_state_sha256(state)
    return state, {
        "sha256": digest,
        "checkpoint_receipt_sha256": initialization.get(
            "initialized_model_state_sha256"
        ),
        "source_checkpoint_sha256": source_digest,
        "source_architecture": loaded["source_architecture"],
        "state_keys": len(state),
    }


def _compare_initial_states(
    spatial_checkpoint: Mapping[str, Any],
    global_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    spatial_state, spatial_receipt = _reconstruct_initial_state(spatial_checkpoint)
    global_state, global_receipt = _reconstruct_initial_state(global_checkpoint)
    keys_equal = tuple(spatial_state) == tuple(global_state)
    shapes_equal = keys_equal and all(
        spatial_state[name].shape == global_state[name].shape for name in spatial_state
    )
    dtypes_equal = keys_equal and all(
        spatial_state[name].dtype == global_state[name].dtype for name in spatial_state
    )
    values_equal = keys_equal and all(
        torch.equal(spatial_state[name], global_state[name]) for name in spatial_state
    )
    receipt_hashes_valid = (
        spatial_receipt["sha256"] == spatial_receipt["checkpoint_receipt_sha256"]
        and global_receipt["sha256"] == global_receipt["checkpoint_receipt_sha256"]
    )
    return {
        "keys_equal": keys_equal,
        "shapes_equal": shapes_equal,
        "dtypes_equal": dtypes_equal,
        "values_equal": values_equal,
        "state_sha256_equal": (spatial_receipt["sha256"] == global_receipt["sha256"]),
        "checkpoint_receipts_valid": receipt_hashes_valid,
        "spatial": spatial_receipt,
        "global_control": global_receipt,
    }


def _matched_receipts(
    spatial_checkpoint: Mapping[str, Any],
    global_checkpoint: Mapping[str, Any],
    spatial_report: Mapping[str, Any],
    global_report: Mapping[str, Any],
    initialization: Mapping[str, Any],
    *,
    audit_windows: Mapping[str, Any],
    audit_pairs: Mapping[str, Any],
) -> dict[str, Any]:
    initial_exact = all(
        initialization[name]
        for name in (
            "keys_equal",
            "shapes_equal",
            "dtypes_equal",
            "values_equal",
            "state_sha256_equal",
            "checkpoint_receipts_valid",
        )
    )
    source_keys = (
        "sha256",
        "expected_sha256",
        "retina_sha256",
        "source_architecture",
        "loaded_modules",
        "discarded_candidate_critic",
    )
    source_equal = all(
        spatial_checkpoint.get("initialization", {}).get(key)
        == global_checkpoint.get("initialization", {}).get(key)
        for key in source_keys
    )
    receipt_keys = (
        "manifest",
        "partition",
        "training_pairs",
        "candidate_bank_receipt",
        "fonts",
        "render_config",
        "data_boundary",
    )
    data_equal = all(
        spatial_checkpoint.get(key) == global_checkpoint.get(key)
        for key in receipt_keys
    )
    protocol_source_equal = spatial_checkpoint.get("protocol", {}).get(
        "source_files_sha256"
    ) == global_checkpoint.get("protocol", {}).get("source_files_sha256")
    optimization_equal = spatial_checkpoint.get("protocol", {}).get(
        "fixed_optimization"
    ) == global_checkpoint.get("protocol", {}).get(
        "fixed_optimization"
    ) and spatial_checkpoint.get("protocol", {}).get(
        "fixed_evidence"
    ) == global_checkpoint.get("protocol", {}).get("fixed_evidence")
    pixel_equal = (
        spatial_report["natural"]["rendered_batch_sha256"]
        == global_report["natural"]["rendered_batch_sha256"]
        and spatial_report["suffix4"]["rendered_batch_sha256"]
        == global_report["suffix4"]["rendered_batch_sha256"]
    )
    spatial_integrity = spatial_report["integrity"]
    global_integrity = global_report["integrity"]
    finite_updates = all(
        item["step"] == 8_000
        and item["finite_updates_verified"] == 8_000
        and item["model_state_finite"]
        and item["training_metrics_finite"]
        for item in (spatial_integrity, global_integrity)
    )
    return {
        "initialized_parameter_states_exact": initial_exact,
        "final_parameter_counts_exact": (
            spatial_integrity["total_parameters"]
            == global_integrity["total_parameters"]
            and spatial_integrity["trainable_parameters"]
            == global_integrity["trainable_parameters"]
        ),
        "source_and_data_receipts_exact": (
            source_equal and data_equal and protocol_source_equal and optimization_equal
        ),
        "audit_windows_and_pixels_exact": pixel_equal,
        "both_arms_completed_8000_finite_updates": finite_updates,
        "initialization": dict(initialization),
        "source_receipts_equal": source_equal,
        "data_receipts_equal": data_equal,
        "protocol_source_hashes_equal": protocol_source_equal,
        "optimization_receipts_equal": optimization_equal,
        "rendered_audit_pixels_equal": pixel_equal,
        "natural_windows": dict(audit_windows),
        "suffix4_pairs": dict(audit_pairs),
    }


@torch.no_grad()
def _evaluate_route(
    model: SpatialVisualNextFieldModel,
    checkpoint: Mapping[str, Any],
    records: Sequence[Any],
    statistics: VisualCharacterStatistics,
    trigram_rows: Mapping[str, Counter[int]],
    natural_windows: Sequence[Any],
    pairs: Sequence[Any],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    natural = evaluate_natural_language(
        model,
        _natural_loader(
            natural_windows,
            statistics,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        statistics,
        trigram_rows,
        bank_images,
        device=device,
        precision=precision,
    )
    suffix4 = evaluate_suffix_pairs(
        model,
        _pair_loader(pairs, batch_size=batch_size, num_workers=num_workers),
        device=device,
        precision=precision,
    )
    evaluator_peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    peak = max(
        evaluator_peak,
        float(checkpoint.get("peak_allocated_vram_gib", 0.0)),
    )
    return {
        "route_mode": model.config.route_mode,
        "natural": natural,
        "suffix4": suffix4,
        "integrity": _route_integrity(
            model,
            checkpoint,
            natural,
            suffix4,
            peak_allocated_vram_gib=peak,
        ),
        "evaluator_peak_allocated_vram_gib": evaluator_peak,
    }


def run_joint_audit(
    spatial_model: SpatialVisualNextFieldModel,
    spatial_checkpoint: Mapping[str, Any],
    global_model: SpatialVisualNextFieldModel,
    global_checkpoint: Mapping[str, Any],
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
    smoke = bool(spatial_checkpoint.get("smoke_only"))
    if smoke != bool(global_checkpoint.get("smoke_only")):
        raise ValueError("V30 route checkpoints mix smoke and evidence states")
    strict = not smoke
    records = load_v25_records(manifest, strict_manifest=strict)
    statistics = build_v30_candidate_statistics(records, bank_size=bank_size)
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
    trigram_rows = _trigram_rows(records, statistics)
    audit_window_receipt = _natural_window_receipt(natural_windows)
    audit_pair_receipt = _pair_receipt(suffix4_pairs)
    statistics_receipt = _statistics_receipt(statistics, bank_images)
    initialization = _compare_initial_states(spatial_checkpoint, global_checkpoint)
    started = time.monotonic()
    spatial = _evaluate_route(
        spatial_model,
        spatial_checkpoint,
        records,
        statistics,
        trigram_rows,
        natural_windows,
        suffix4_pairs,
        bank_images,
        device=device,
        precision=precision,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    global_control = _evaluate_route(
        global_model,
        global_checkpoint,
        records,
        statistics,
        trigram_rows,
        natural_windows,
        suffix4_pairs,
        bank_images,
        device=device,
        precision=precision,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    matched = _matched_receipts(
        spatial_checkpoint,
        global_checkpoint,
        spatial,
        global_control,
        initialization,
        audit_windows=audit_window_receipt,
        audit_pairs=audit_pair_receipt,
    )
    frozen_images_instantiated = False
    spatial_gates, global_gates, matched_gates, language_gates = v30_gate_report(
        spatial,
        global_control,
        matched,
        frozen_images_instantiated=frozen_images_instantiated,
    )
    selected = all(
        all(group.values())
        for group in (
            spatial_gates,
            global_gates,
            matched_gates,
            language_gates,
        )
    )
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint_architecture": ARCHITECTURE,
        "smoke_only": smoke,
        "manifest": verify_v25_manifest(manifest, strict=strict),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": statistics_receipt,
        "natural_windows": audit_window_receipt,
        "suffix4_pairs": audit_pair_receipt,
        "audit_seed": AUDIT_SEED,
        "frozen_images_instantiated": frozen_images_instantiated,
        "spatial": spatial,
        "global_control": global_control,
        "matched": matched,
        "spatial_common_gates": spatial_gates,
        "spatial_common_selected": all(spatial_gates.values()),
        "global_integrity_gates": global_gates,
        "global_integrity_selected": all(global_gates.values()),
        "matched_arm_gates": matched_gates,
        "matched_arms_selected": all(matched_gates.values()),
        "spatial_language_gates": language_gates,
        "spatial_language_selected": all(language_gates.values()),
        "spatial_mechanism_selected": selected,
        "frozen_evaluation_authorized": selected,
        "writer_training_authorized": False,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "evaluation_seconds": time.monotonic() - started,
    }


def _verify_source_receipt(checkpoint: Mapping[str, Any]) -> None:
    source_files = checkpoint.get("protocol", {}).get("source_files_sha256", {})
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("V30 checkpoint has no source-file receipt")
    for path, expected in source_files.items():
        if file_sha256(path) != expected:
            raise ValueError(f"V30 source file changed after training: {path}")


def _verify_evidence_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("smoke_only") or checkpoint.get("exploratory"):
        raise ValueError("V30 evidence audit rejects smoke or exploratory weights")
    if checkpoint.get("step") != 8_000:
        raise ValueError("V30 evidence checkpoint did not finish 8,000 updates")
    if checkpoint.get("finite_updates_verified") != 8_000:
        raise ValueError("V30 checkpoint lacks 8,000 finite-update receipts")
    if checkpoint.get("manifest", {}).get("sha256") != V25_MANIFEST_SHA256:
        raise ValueError("V30 checkpoint has the wrong corpus receipt")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V30 checkpoint has the wrong protocol receipt")
    if protocol.get("expected_protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V30 checkpoint did not pin the V30 protocol")
    _verify_source_receipt(checkpoint)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    spatial_model, spatial_checkpoint = load_model_checkpoint(
        args.spatial_checkpoint,
        expected_route=V30_SPATIAL_ROUTE,
        device=device,
        allow_smoke=args.allow_smoke,
    )
    global_model, global_checkpoint = load_model_checkpoint(
        args.global_checkpoint,
        expected_route=V30_GLOBAL_ROUTE,
        device=device,
        allow_smoke=args.allow_smoke,
    )
    smoke = bool(spatial_checkpoint.get("smoke_only"))
    if smoke != bool(global_checkpoint.get("smoke_only")):
        raise ValueError("V30 audit cannot mix smoke and evidence checkpoints")
    protocol_sha256 = file_sha256(PROTOCOL_DOCUMENT)
    if not smoke:
        if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("V30 protocol changed after preregistration")
        if args.windows != NATURAL_WINDOWS or args.pair_windows != PAIR_WINDOWS:
            raise ValueError("V30 evidence audit requires fixed window counts")
        if args.bank_size != AUDIT_BANK_SIZE or args.batch_size != 16:
            raise ValueError("V30 evidence audit requires fixed bank and batch sizes")
        if args.out != DEFAULT_OUTPUT:
            raise ValueError(f"V30 evidence audit requires --out={DEFAULT_OUTPUT}")
        if device.type != "cuda" or device.index not in (None, 0):
            raise ValueError("V30 evidence audit requires CUDA device 0")
        if args.precision != "bf16":
            raise ValueError("V30 evidence audit requires BF16 autocast")
        _verify_evidence_checkpoint(spatial_checkpoint)
        _verify_evidence_checkpoint(global_checkpoint)
    report = run_joint_audit(
        spatial_model,
        spatial_checkpoint,
        global_model,
        global_checkpoint,
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
            "spatial_checkpoint": str(args.spatial_checkpoint),
            "spatial_checkpoint_sha256": file_sha256(args.spatial_checkpoint),
            "global_checkpoint": str(args.global_checkpoint),
            "global_checkpoint_sha256": file_sha256(args.global_checkpoint),
            "device": str(device),
            "precision": args.precision,
        }
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "development_audit.json"
    atomic_write_json(report, audit_path)
    comparison = {
        "architecture": AUDIT_ARCHITECTURE,
        "development_audit": str(audit_path),
        "development_audit_sha256": file_sha256(audit_path),
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": report["manifest"]["sha256"],
        "spatial_checkpoint_sha256": report["spatial_checkpoint_sha256"],
        "global_checkpoint_sha256": report["global_checkpoint_sha256"],
        "spatial_initial_state_sha256": report["matched"]["initialization"]["spatial"][
            "sha256"
        ],
        "global_initial_state_sha256": report["matched"]["initialization"][
            "global_control"
        ]["sha256"],
        "training_candidate_bank_receipt_sha256": _json_sha256(
            spatial_checkpoint["candidate_bank_receipt"]
        ),
        "natural_audit_receipt": report["natural_windows"],
        "suffix4_audit_receipt": report["suffix4_pairs"],
        "audit_bank_receipt": report["statistics"],
        "spatial_mechanism_selected": report["spatial_mechanism_selected"],
        "frozen_images_instantiated": False,
        "frozen_evaluation_authorized": report["frozen_evaluation_authorized"],
        "writer_training_authorized": False,
    }
    atomic_write_json(comparison, output / "comparison_receipt.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
