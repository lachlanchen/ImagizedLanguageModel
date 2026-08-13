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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.conditional_visual_field_flow import (
    V31_ARCHITECTURE,
    V31_AUDIT_PROBE_TIMES,
    V31_GLOBAL_ROUTE,
    V31_SPATIAL_ROUTE,
    ConditionalVisualFieldFlowModel,
    conditional_visual_field_flow_boundary_receipt,
    conditional_visual_field_flow_config_from_payload,
    model_state_sha256,
    spatially_permute_v31_fields,
)
from ilm.visual_lm.conditional_visual_field_flow_data import (
    build_v31_candidate_statistics,
    conditional_visual_field_flow_data_boundary_receipt,
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
from ilm.visual_lm.spatial_visual_next_field import per_row_assignment_margin
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


ARCHITECTURE = V31_ARCHITECTURE
AUDIT_ARCHITECTURE = "conditional-visual-field-flow-v31-development-audit"
PROTOCOL_DOCUMENT = "references/conditional_visual_field_flow_v31_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "92b6f70975dffe25723e332268b8929fa547b9d848a296f9ed80968cf798f8f7"
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_SPATIAL_CHECKPOINT = (
    "artifacts/conditional_visual_field_flow_v31_spatial_evidence/"
    "checkpoint_final.pt"
)
DEFAULT_GLOBAL_CHECKPOINT = (
    "artifacts/conditional_visual_field_flow_v31_global_control_evidence/"
    "checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/conditional_visual_field_flow_v31_evidence"
AUDIT_SEED = 20261114
MODEL_SEED = 20261110
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
AUDIT_BATCH_SIZE = 16
PATH_SCORE_NAMES = ("full", "suffix4", "shuffled", "spatial_permuted")
SAMPLE_SCORE_NAMES = ("full", "shuffled", "spatial_permuted")
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered joint V31 visual-flow audit."
    )
    parser.add_argument("--spatial-checkpoint", default=DEFAULT_SPATIAL_CHECKPOINT)
    parser.add_argument("--global-checkpoint", default=DEFAULT_GLOBAL_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=AUDIT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=NATURAL_WINDOWS)
    parser.add_argument("--pair-windows", type=int, default=PAIR_WINDOWS)
    parser.add_argument("--bank-size", type=int, default=AUDIT_BANK_SIZE)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


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


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, Mapping):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _checkpoint_bank_is_absent(checkpoint: Mapping[str, Any]) -> bool:
    if checkpoint.get("deployed_state_includes_training_candidate_images") is not False:
        return False
    if checkpoint.get("deployed_state_includes_training_form_labels") is not False:
        return False
    state = checkpoint.get("model")
    return isinstance(state, Mapping) and not any(
        "bank" in str(name).lower() for name in state
    )


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
    model: ConditionalVisualFieldFlowModel,
    checkpoint: Mapping[str, Any],
) -> bool:
    expected_model = conditional_visual_field_flow_boundary_receipt(model.config)
    expected_data = conditional_visual_field_flow_data_boundary_receipt()
    boundary = checkpoint.get("model_boundary", {})
    data = checkpoint.get("data_boundary", {})
    forbidden = (
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_vocabulary_output",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
    )
    return (
        boundary == expected_model
        and data == expected_data
        and boundary.get("autonomous_sampler_requires_candidates") is False
        and boundary.get("candidate_bank_deployed") is False
        and all(boundary.get(name) is False for name in forbidden)
    )


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


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def encode_candidate_bank(
    model: ConditionalVisualFieldFlowModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> torch.Tensor:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V31 visual bank must be [identity,view,1,32,32]")
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
    logits = torch.einsum("npc,mpc->nm", queries.float(), keys.float()) / 16
    targets = torch.arange(logits.shape[0], device=logits.device)
    return float((logits.argmax(dim=1) == targets).float().mean())


def _top_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    logp = logits.float().log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(logp.sum()),
    }


def _audit_shuffle(context: torch.Tensor, *, first_index: int) -> torch.Tensor:
    generator = torch.Generator(device=context.device).manual_seed(
        AUDIT_SEED + first_index * 104_729
    )
    return shuffle_visual_prefix(context, generator=generator)


def _fixed_probe_fields(
    model: ConditionalVisualFieldFlowModel,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(AUDIT_SEED)
    vectors = torch.randn(
        8,
        model.config.field_channels,
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    bases = vectors[:, None].expand(-1, 16, -1).clone()
    times = torch.tensor(V31_AUDIT_PROBE_TIMES, device=device, dtype=torch.float32)
    return bases, times, vectors


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


def _score_conditions_by_view(
    model: ConditionalVisualFieldFlowModel,
    condition: torch.Tensor,
    candidate_fields: torch.Tensor,
    views: torch.Tensor,
    probe_bases: torch.Tensor,
    probe_times: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        condition.shape[0],
        candidate_fields.shape[1],
        device=condition.device,
        dtype=torch.float32,
    )
    for view in range(candidate_fields.shape[0]):
        rows = (views == view).nonzero(as_tuple=False).flatten()
        if rows.numel():
            output[rows] = model.path_score_encoded_shared(
                condition.index_select(0, rows),
                candidate_fields[view],
                probe_bases,
                probe_times,
                chunk_size=32,
            )
    return output


def _sample_similarity_by_view(
    samples: torch.Tensor,
    candidate_fields: torch.Tensor,
    views: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        samples.shape[0],
        samples.shape[1],
        candidate_fields.shape[1],
        device=samples.device,
        dtype=torch.float32,
    )
    for view in range(candidate_fields.shape[0]):
        rows = (views == view).nonzero(as_tuple=False).flatten()
        if rows.numel():
            output[rows] = (
                torch.einsum(
                    "bkpc,npc->bkn",
                    samples.index_select(0, rows).float(),
                    candidate_fields[view].float(),
                )
                / 16
            )
    return output


def _sample_metrics(
    samples: torch.Tensor,
    shuffled_samples: torch.Tensor,
) -> tuple[float, int, float]:
    similarities = torch.einsum("bkpc,blpc->bkl", samples.float(), samples.float()) / 16
    mask = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
    diversity = 1.0 - similarities[mask]
    displacement = 1.0 - (samples.float() * shuffled_samples.float()).sum(dim=-1).mean(
        dim=-1
    )
    return float(diversity.sum()), int(diversity.numel()), float(displacement.sum())


@torch.no_grad()
def evaluate_natural_language(
    model: ConditionalVisualFieldFlowModel,
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
    bank_by_view = bank.permute(1, 0, 2, 3)
    permuted_by_view = spatially_permute_v31_fields(bank_by_view)
    visibility = 0.5 * (
        _cross_font_retrieval_accuracy(bank[:, 0], bank[:, 1])
        + _cross_font_retrieval_accuracy(bank[:, 1], bank[:, 0])
    )
    probe_bases, probe_times, sample_vectors = _fixed_probe_fields(model, device=device)
    path_totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
        }
        for name in PATH_SCORE_NAMES
    }
    sample_totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
        }
        for name in SAMPLE_SCORE_NAMES
    }
    examples = 0
    all_targets: list[int] = []
    all_contexts: list[str] = []
    sample_examples: list[dict[str, int]] = []
    best_rank_sum = 0.0
    diversity_sum = 0.0
    diversity_count = 0
    displacement_sum = 0.0
    path_spatial_error = 0.0
    sample_spatial_error = 0.0
    scores_finite = True
    sample_shape_clean = True
    pixel_digest = hashlib.sha256()
    started = time.monotonic()
    for raw in loader:
        _update_tensor_digest(pixel_digest, "context", raw["context"])
        _update_tensor_digest(pixel_digest, "target_index", raw["target_index"])
        _update_tensor_digest(pixel_digest, "candidate_view", raw["candidate_view"])
        context = raw["context"].to(device, non_blocking=True)
        targets = raw["target_index"].to(device)
        views = raw["candidate_view"].to(device)
        shuffled_context = _audit_shuffle(context, first_index=examples)
        with autocast_context(device, precision):
            conditions = {
                "full": model.context_condition(context),
                "suffix4": model.context_condition(context[:, -4:]),
                "shuffled": model.context_condition(shuffled_context),
            }
            path_scores = {
                name: _score_conditions_by_view(
                    model,
                    condition,
                    bank_by_view,
                    views,
                    probe_bases,
                    probe_times,
                )
                for name, condition in conditions.items()
            }
            path_scores["spatial_permuted"] = _score_conditions_by_view(
                model,
                conditions["full"],
                permuted_by_view,
                views,
                probe_bases,
                probe_times,
            )
            full_samples = model.sample_encoded(
                conditions["full"], sample_vectors, steps=8, solver="heun"
            )
            shuffled_samples = model.sample_encoded(
                conditions["shuffled"], sample_vectors, steps=8, solver="heun"
            )
        sample_shape_clean = sample_shape_clean and tuple(full_samples.shape[1:]) == (
            8,
            16,
            model.config.field_channels,
        )
        scores_finite = (
            scores_finite
            and bool(torch.isfinite(full_samples).all())
            and bool(torch.isfinite(shuffled_samples).all())
        )
        full_similarity = _sample_similarity_by_view(full_samples, bank_by_view, views)
        shuffled_similarity = _sample_similarity_by_view(
            shuffled_samples, bank_by_view, views
        )
        permuted_similarity = _sample_similarity_by_view(
            full_samples, permuted_by_view, views
        )
        sample_scores = {
            "full": torch.logsumexp(16.0 * full_similarity, dim=1) - math.log(8),
            "shuffled": (
                torch.logsumexp(16.0 * shuffled_similarity, dim=1) - math.log(8)
            ),
            "spatial_permuted": (
                torch.logsumexp(16.0 * permuted_similarity, dim=1) - math.log(8)
            ),
        }
        for name, logits in path_scores.items():
            scores_finite = scores_finite and bool(torch.isfinite(logits).all())
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                path_totals[name][key] += value
        for name, logits in sample_scores.items():
            scores_finite = scores_finite and bool(torch.isfinite(logits).all())
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                sample_totals[name][key] += value
        target_similarity = full_similarity.gather(
            2,
            targets[:, None, None].expand(-1, full_similarity.shape[1], 1),
        )
        ranks = 1 + (full_similarity > target_similarity).sum(dim=-1)
        best_rank_sum += float(ranks.min(dim=1).values.float().sum())
        diversity, count, displacement = _sample_metrics(full_samples, shuffled_samples)
        diversity_sum += diversity
        diversity_count += count
        displacement_sum += displacement
        path_spatial_error = max(
            path_spatial_error,
            float((path_scores["full"] - path_scores["spatial_permuted"]).abs().amax()),
        )
        sample_spatial_error = max(
            sample_spatial_error,
            float(
                (sample_scores["full"] - sample_scores["spatial_permuted"]).abs().amax()
            ),
        )
        if len(sample_examples) < 8:
            predictions = sample_scores["full"].argmax(dim=1)
            shuffled_predictions = sample_scores["shuffled"].argmax(dim=1)
            for row in range(min(context.shape[0], 8 - len(sample_examples))):
                sample_examples.append(
                    {
                        "target": int(targets[row]),
                        "full_nearest": int(predictions[row]),
                        "shuffled_nearest": int(shuffled_predictions[row]),
                        "candidate_view": int(views[row]),
                    }
                )
        all_targets.extend(raw["target_index"].tolist())
        all_contexts.extend(raw["context_text"])
        examples += context.shape[0]
    if not examples:
        raise ValueError("V31 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, Any] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "candidate_cross_font_identity_top1": visibility,
        "path_spatial_permutation_max_score_error": path_spatial_error,
        "sample_spatial_permutation_max_score_error": sample_spatial_error,
        "all_scores_and_samples_finite": float(scores_finite),
        "candidate_independent_sample_shape_clean": float(sample_shape_clean),
        "sample_mean_pairwise_cosine_distance": diversity_sum / diversity_count,
        "same_noise_full_shuffled_sample_displacement": displacement_sum / examples,
        "sample_best_of_eight_mean_target_rank": best_rank_sum / examples,
        "sample_examples": sample_examples,
        "rendered_batch_sha256": pixel_digest.hexdigest(),
    }
    for name in PATH_SCORE_NAMES:
        metrics[f"path_{name}_top1"] = path_totals[name]["correct_top1"] / examples
        metrics[f"path_{name}_top5"] = path_totals[name]["correct_top5"] / examples
        metrics[f"path_{name}_target_log_probability"] = (
            path_totals[name]["target_log_probability_sum"] / examples
        )
    for name in SAMPLE_SCORE_NAMES:
        metrics[f"sample_{name}_top1"] = sample_totals[name]["correct_top1"] / examples
        metrics[f"sample_{name}_top5"] = sample_totals[name]["correct_top5"] / examples
        metrics[f"sample_{name}_target_log_probability"] = (
            sample_totals[name]["target_log_probability_sum"] / examples
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
    margins = per_row_assignment_margin(logits, assignments)
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


def _paired_conditions(
    model: ConditionalVisualFieldFlowModel,
    contexts: torch.Tensor,
) -> torch.Tensor:
    batch, queries = contexts.shape[:2]
    return model.context_condition(
        contexts.reshape(batch * queries, *contexts.shape[2:])
    ).reshape(batch, queries, model.config.model_dim)


@torch.no_grad()
def _pair_path_scores_from_images(
    model: ConditionalVisualFieldFlowModel,
    contexts: torch.Tensor,
    shuffled_contexts: torch.Tensor,
    candidates: torch.Tensor,
    probe_bases: torch.Tensor,
    probe_times: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = model.encode_route_candidates(candidates)
    full_condition = _paired_conditions(model, contexts)
    shuffled_condition = _paired_conditions(model, shuffled_contexts)
    suffix = contexts[:, :1, -4:]
    suffix_condition = _paired_conditions(model, suffix)
    suffix_score = model.path_score_encoded_batched(
        suffix_condition, fields, probe_bases, probe_times
    ).expand(-1, 2, -1)
    scores = {
        "full": model.path_score_encoded_batched(
            full_condition, fields, probe_bases, probe_times
        ),
        "suffix4": suffix_score,
        "shuffled": model.path_score_encoded_batched(
            shuffled_condition, fields, probe_bases, probe_times
        ),
        "spatial_permuted": model.path_score_encoded_batched(
            full_condition,
            spatially_permute_v31_fields(fields),
            probe_bases,
            probe_times,
        ),
    }
    return scores, full_condition, shuffled_condition, fields


@torch.no_grad()
def _pair_sample_scores(
    model: ConditionalVisualFieldFlowModel,
    full_condition: torch.Tensor,
    shuffled_condition: torch.Tensor,
    candidate_fields: torch.Tensor,
    sample_vectors: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch, queries = full_condition.shape[:2]
    full_samples = model.sample_encoded(
        full_condition.reshape(batch * queries, model.config.model_dim),
        sample_vectors,
        steps=8,
        solver="heun",
    ).reshape(batch, queries, 8, 16, model.config.field_channels)
    shuffled_samples = model.sample_encoded(
        shuffled_condition.reshape(batch * queries, model.config.model_dim),
        sample_vectors,
        steps=8,
        solver="heun",
    ).reshape(batch, queries, 8, 16, model.config.field_channels)
    return (
        {
            "full": model.sample_score_encoded_batched(full_samples, candidate_fields),
            "shuffled": model.sample_score_encoded_batched(
                shuffled_samples, candidate_fields
            ),
            "spatial_permuted": model.sample_score_encoded_batched(
                full_samples, spatially_permute_v31_fields(candidate_fields)
            ),
        },
        full_samples,
        shuffled_samples,
    )


@torch.no_grad()
def evaluate_suffix_pairs(
    model: ConditionalVisualFieldFlowModel,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    probe_bases, probe_times, sample_vectors = _fixed_probe_fields(model, device=device)
    totals: dict[str, Any] = {
        "pairs": 0.0,
        "suffix_equal": 0.0,
        "suffix_checks": 0.0,
        "suffix_score_max_error": 0.0,
        "path_spatial_permutation_max_error": 0.0,
        "sample_spatial_permutation_max_error": 0.0,
        "scores_finite": 1.0,
        "sample_diversity_sum": 0.0,
        "sample_diversity_count": 0.0,
        "sample_displacement_sum": 0.0,
        "sample_displacement_count": 0.0,
    }
    score_prefixes = [f"path_{name}" for name in PATH_SCORE_NAMES] + [
        f"sample_{name}" for name in SAMPLE_SCORE_NAMES
    ]
    for prefix in score_prefixes:
        totals[f"permutation_{prefix}_max_error"] = 0.0
        totals[f"permutation_{prefix}_accuracy_equal"] = 0.0
        totals[f"permutation_{prefix}_checks"] = 0.0
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
        directions = (
            (
                raw["contexts"].to(device, non_blocking=True),
                raw["candidates"].to(device, non_blocking=True),
                raw["assignment"].to(device),
            ),
            (
                raw["reference_contexts"].to(device, non_blocking=True),
                raw["reference_candidates"].to(device, non_blocking=True),
                raw["reference_assignment"].to(device),
            ),
        )
        suffix = int(raw["metadata"][0]["suffix_cells"])
        for contexts, candidates, labels in directions:
            equal = (
                (contexts[:, 0, -suffix:] == contexts[:, 1, -suffix:])
                .flatten(1)
                .all(dim=1)
            )
            totals["suffix_equal"] += float(equal.sum())
            totals["suffix_checks"] += float(equal.numel())
            shuffled = _audit_shuffle(contexts, first_index=pair_index)
            with autocast_context(device, precision):
                path_scores, full_condition, shuffled_condition, fields = (
                    _pair_path_scores_from_images(
                        model,
                        contexts,
                        shuffled,
                        candidates,
                        probe_bases,
                        probe_times,
                    )
                )
                swapped_path, _, _, _ = _pair_path_scores_from_images(
                    model,
                    contexts,
                    shuffled,
                    candidates.flip(1),
                    probe_bases,
                    probe_times,
                )
                sample_scores, full_samples, shuffled_samples = _pair_sample_scores(
                    model,
                    full_condition,
                    shuffled_condition,
                    fields,
                    sample_vectors,
                )
            swapped_sample = {
                "full": model.sample_score_encoded_batched(
                    full_samples, fields.flip(1)
                ),
                "shuffled": model.sample_score_encoded_batched(
                    shuffled_samples, fields.flip(1)
                ),
                "spatial_permuted": model.sample_score_encoded_batched(
                    full_samples,
                    spatially_permute_v31_fields(fields.flip(1)),
                ),
            }
            totals["suffix_score_max_error"] = max(
                totals["suffix_score_max_error"],
                float(
                    (path_scores["suffix4"][:, 0] - path_scores["suffix4"][:, 1])
                    .abs()
                    .amax()
                ),
            )
            totals["path_spatial_permutation_max_error"] = max(
                totals["path_spatial_permutation_max_error"],
                float(
                    (path_scores["spatial_permuted"] - path_scores["full"]).abs().amax()
                ),
            )
            totals["sample_spatial_permutation_max_error"] = max(
                totals["sample_spatial_permutation_max_error"],
                float(
                    (sample_scores["spatial_permuted"] - sample_scores["full"])
                    .abs()
                    .amax()
                ),
            )
            flattened_full = full_samples.reshape(
                -1, 8, 16, model.config.field_channels
            )
            flattened_shuffled = shuffled_samples.reshape_as(flattened_full)
            diversity, count, displacement = _sample_metrics(
                flattened_full, flattened_shuffled
            )
            totals["sample_diversity_sum"] += diversity
            totals["sample_diversity_count"] += count
            totals["sample_displacement_sum"] += displacement
            totals["sample_displacement_count"] += flattened_full.shape[0]
            for family, scores, swapped in (
                ("path", path_scores, swapped_path),
                ("sample", sample_scores, swapped_sample),
            ):
                names = PATH_SCORE_NAMES if family == "path" else SAMPLE_SCORE_NAMES
                for name in names:
                    prefix = f"{family}_{name}"
                    logits = scores[name]
                    totals["scores_finite"] *= float(torch.isfinite(logits).all())
                    stats = _assignment_statistics(logits, labels)
                    _accumulate_assignment(totals, prefix, stats)
                    swapped_logits = swapped[name]
                    totals["scores_finite"] *= float(
                        torch.isfinite(swapped_logits).all()
                    )
                    error = float((swapped_logits - logits.flip(-1)).abs().amax())
                    swapped_stats = _assignment_statistics(swapped_logits, 1 - labels)
                    agreement = float(
                        swapped_stats["accuracy_sum"] == stats["accuracy_sum"]
                    )
                    totals[f"permutation_{prefix}_max_error"] = max(
                        totals[f"permutation_{prefix}_max_error"], error
                    )
                    totals[f"permutation_{prefix}_accuracy_equal"] += agreement
                    totals[f"permutation_{prefix}_checks"] += 1.0
        totals["pairs"] += directions[0][0].shape[0]
        pair_index += directions[0][0].shape[0]
    if not totals["pairs"]:
        raise ValueError("V31 suffix-pair audit loader is empty")
    metrics: dict[str, Any] = {
        "pairs": totals["pairs"],
        "suffix_pixel_equality": totals["suffix_equal"] / totals["suffix_checks"],
        "suffix_path_score_row_max_error": totals["suffix_score_max_error"],
        "path_spatial_permutation_max_score_error": totals[
            "path_spatial_permutation_max_error"
        ],
        "sample_spatial_permutation_max_score_error": totals[
            "sample_spatial_permutation_max_error"
        ],
        "all_scores_and_samples_finite": totals["scores_finite"],
        "sample_mean_pairwise_cosine_distance": (
            totals["sample_diversity_sum"] / totals["sample_diversity_count"]
        ),
        "same_noise_full_shuffled_sample_displacement": (
            totals["sample_displacement_sum"] / totals["sample_displacement_count"]
        ),
        "rendered_batch_sha256": pixel_digest.hexdigest(),
    }
    for prefix in score_prefixes:
        arms = totals[f"{prefix}_arms"]
        assignments_count = totals[f"{prefix}_assignments"]
        rows = totals[f"{prefix}_row_accuracy_sum"] / assignments_count
        metrics[f"{prefix}_arm_accuracy"] = totals[f"{prefix}_accuracy_sum"] / arms
        metrics[f"{prefix}_strict_arm_accuracy"] = (
            totals[f"{prefix}_strict_accuracy_sum"] / arms
        )
        metrics[f"{prefix}_tie_rate"] = totals[f"{prefix}_tie_sum"] / arms
        metrics[f"{prefix}_both_correct_rate"] = (
            totals[f"{prefix}_both_correct_sum"] / assignments_count
        )
        metrics[f"{prefix}_mean_margin"] = totals[f"{prefix}_margin_sum"] / arms
        metrics[f"{prefix}_row0_accuracy"] = float(rows[0])
        metrics[f"{prefix}_row1_accuracy"] = float(rows[1])
        metrics[f"candidate_permutation_{prefix}_max_score_error"] = totals[
            f"permutation_{prefix}_max_error"
        ]
        metrics[f"candidate_permutation_{prefix}_accuracy_agreement"] = (
            totals[f"permutation_{prefix}_accuracy_equal"]
            / totals[f"permutation_{prefix}_checks"]
        )
    metrics["path_full_minus_shuffled_arm_accuracy"] = (
        metrics["path_full_arm_accuracy"] - metrics["path_shuffled_arm_accuracy"]
    )
    metrics["path_full_minus_shuffled_mean_margin"] = (
        metrics["path_full_mean_margin"] - metrics["path_shuffled_mean_margin"]
    )
    metrics["path_full_minus_spatial_permuted_arm_accuracy"] = (
        metrics["path_full_arm_accuracy"]
        - metrics["path_spatial_permuted_arm_accuracy"]
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
        raise ValueError("cannot collate an empty V31 natural audit batch")
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
    model: ConditionalVisualFieldFlowModel,
    checkpoint: Mapping[str, Any],
    natural: Mapping[str, Any],
    suffix4: Mapping[str, Any],
    *,
    peak_allocated_vram_gib: float,
) -> dict[str, Any]:
    return {
        "model_state_finite": _model_state_is_finite(checkpoint),
        "training_metrics_finite": _training_metrics_are_finite(checkpoint),
        "natural_metrics_finite": _all_finite(natural),
        "pair_metrics_finite": _all_finite(suffix4),
        "natural_scores_and_samples_finite": bool(
            natural["all_scores_and_samples_finite"]
        ),
        "pair_scores_and_samples_finite": bool(
            suffix4["all_scores_and_samples_finite"]
        ),
        "student_boundary_clean": student_boundary_is_clean(model, checkpoint),
        "final_checkpoint_clean": final_checkpoint_is_clean(checkpoint),
        "candidate_independent_sample_shape_clean": bool(
            natural["candidate_independent_sample_shape_clean"]
        ),
        "candidate_bank_absent": _checkpoint_bank_is_absent(checkpoint),
        "total_parameters": int(checkpoint.get("total_parameters", -1)),
        "trainable_parameters": int(checkpoint.get("trainable_parameters", -1)),
        "peak_allocated_vram_gib": peak_allocated_vram_gib,
        "step": int(checkpoint.get("step", -1)),
        "finite_updates_verified": int(checkpoint.get("finite_updates_verified", -1)),
        "protocol_sha256": checkpoint.get("protocol", {}).get("protocol_sha256"),
        "frozen_images_instantiated": bool(
            checkpoint.get("frozen_images_instantiated", True)
        ),
    }


def _reconstruct_initial_state(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    initialization = checkpoint.get("initialization", {})
    source_path = initialization.get("checkpoint")
    if not isinstance(source_path, str):
        raise ValueError("V31 checkpoint has no V30 initialization path")
    source_digest = file_sha256(source_path)
    if source_digest != initialization.get("sha256"):
        raise ValueError("V31 V30 source no longer matches its receipt")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("architecture") != "spatial-visual-next-field-v30":
        raise ValueError("V31 initialization source is not V30")
    config = conditional_visual_field_flow_config_from_payload(
        checkpoint["model_config"]
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MODEL_SEED)
        model = ConditionalVisualFieldFlowModel(config)
        loaded = model.load_v30_backbone_state(source["model"])
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
        "source_route": loaded["source_route"],
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
    receipts_valid = (
        spatial_receipt["sha256"] == spatial_receipt["checkpoint_receipt_sha256"]
        and global_receipt["sha256"] == global_receipt["checkpoint_receipt_sha256"]
    )
    return {
        "keys_equal": keys_equal,
        "shapes_equal": shapes_equal,
        "dtypes_equal": dtypes_equal,
        "values_equal": values_equal,
        "state_sha256_equal": spatial_receipt["sha256"] == global_receipt["sha256"],
        "checkpoint_receipts_valid": receipts_valid,
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
        "source_route",
        "loaded_modules",
        "discarded_v30_field_decoder",
        "discarded_v30_logit_scale",
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
    protocol_sources_equal = spatial_checkpoint.get("protocol", {}).get(
        "source_files_sha256"
    ) == global_checkpoint.get("protocol", {}).get("source_files_sha256")
    optimization_equal = spatial_checkpoint.get("protocol", {}).get(
        "fixed_optimization"
    ) == global_checkpoint.get("protocol", {}).get(
        "fixed_optimization"
    ) and spatial_checkpoint.get("protocol", {}).get(
        "fixed_evidence"
    ) == global_checkpoint.get("protocol", {}).get("fixed_evidence")
    rendered_equal = (
        spatial_report["natural"]["rendered_batch_sha256"]
        == global_report["natural"]["rendered_batch_sha256"]
        and spatial_report["suffix4"]["rendered_batch_sha256"]
        == global_report["suffix4"]["rendered_batch_sha256"]
    )
    finite_updates = all(
        item["step"] == 10_000
        and item["finite_updates_verified"] == 10_000
        and item["model_state_finite"]
        and item["training_metrics_finite"]
        for item in (
            spatial_report["integrity"],
            global_report["integrity"],
        )
    )
    return {
        "initialized_parameter_states_exact": initial_exact,
        "final_parameter_counts_exact": (
            spatial_report["integrity"]["total_parameters"]
            == global_report["integrity"]["total_parameters"]
            and spatial_report["integrity"]["trainable_parameters"]
            == global_report["integrity"]["trainable_parameters"]
        ),
        "source_and_data_receipts_exact": (
            source_equal
            and data_equal
            and protocol_sources_equal
            and optimization_equal
        ),
        "audit_windows_and_pixels_exact": rendered_equal,
        "both_arms_completed_10000_finite_updates": finite_updates,
        "initialization": dict(initialization),
        "source_receipts_equal": source_equal,
        "data_receipts_equal": data_equal,
        "protocol_source_hashes_equal": protocol_sources_equal,
        "optimization_receipts_equal": optimization_equal,
        "rendered_audit_pixels_equal": rendered_equal,
        "natural_windows": dict(audit_windows),
        "suffix4_pairs": dict(audit_pairs),
    }


def v31_gate_report(
    spatial: Mapping[str, Any],
    global_control: Mapping[str, Any],
    matched: Mapping[str, Any],
    *,
    frozen_images_instantiated: bool,
) -> tuple[dict[str, bool], dict[str, bool], dict[str, bool], dict[str, bool]]:
    sn = spatial["natural"]
    sp = spatial["suffix4"]
    si = spatial["integrity"]
    gn = global_control["natural"]
    gp = global_control["suffix4"]
    gi = global_control["integrity"]
    candidate_permutation = all(
        _strictly_below(sp[f"candidate_permutation_path_{name}_max_score_error"], 1e-5)
        and sp[f"candidate_permutation_path_{name}_accuracy_agreement"] == 1.0
        for name in PATH_SCORE_NAMES
    )
    spatial_common = {
        "all_values_finite": all(
            (
                si["model_state_finite"],
                si["training_metrics_finite"],
                si["natural_metrics_finite"],
                si["pair_metrics_finite"],
                si["natural_scores_and_samples_finite"],
                si["pair_scores_and_samples_finite"],
            )
        ),
        "student_boundary_and_checkpoint_clean": (
            si["student_boundary_clean"]
            and si["final_checkpoint_clean"]
            and not frozen_images_instantiated
            and not si["frozen_images_instantiated"]
        ),
        "candidate_independent_sample_shape_clean": si[
            "candidate_independent_sample_shape_clean"
        ],
        "candidate_bank_absent": si["candidate_bank_absent"],
        "parameter_caps": (
            si["total_parameters"] < 20_000_000
            and si["trainable_parameters"] < 18_500_000
        ),
        "peak_vram_below_18_gib": si["peak_allocated_vram_gib"] < 18.0,
        "spatial_candidate_visibility": sn["candidate_cross_font_identity_top1"]
        >= 0.95,
        "suffix_pixels_exact": sp["suffix_pixel_equality"] == 1.0,
        "suffix_score_rows_exact": _strictly_below(
            sp["suffix_path_score_row_max_error"], 1e-6
        ),
        "candidate_column_equivariance": candidate_permutation,
        "path_pair_arm_accuracy": _strictly_above(sp["path_full_arm_accuracy"], 0.65),
        "path_pair_both_correct": _strictly_above(
            sp["path_full_both_correct_rate"], 0.40
        ),
        "path_pair_full_minus_shuffle_accuracy": _strictly_above(
            sp["path_full_minus_shuffled_arm_accuracy"], 0.10
        ),
        "path_pair_full_minus_shuffle_margin": _strictly_above(
            sp["path_full_minus_shuffled_mean_margin"], 0.05
        ),
        "natural_path_full_minus_shuffle_logp": _strictly_above(
            sn["path_full_target_log_probability"]
            - sn["path_shuffled_target_log_probability"],
            0.03,
        ),
        "natural_spatial_permutation_effect": _strictly_above(
            sn["path_full_target_log_probability"]
            - sn["path_spatial_permuted_target_log_probability"],
            0.05,
        ),
        "pair_spatial_permutation_effect": _strictly_above(
            sp["path_full_minus_spatial_permuted_arm_accuracy"], 0.05
        ),
        "sample_diversity_in_range": (
            _strictly_above(sn["sample_mean_pairwise_cosine_distance"], 1e-4)
            and _strictly_below(sn["sample_mean_pairwise_cosine_distance"], 1.5)
        ),
        "sample_condition_sensitivity": _strictly_above(
            sn["same_noise_full_shuffled_sample_displacement"], 0.01
        ),
    }
    global_candidate_permutation = all(
        _strictly_below(gp[f"candidate_permutation_path_{name}_max_score_error"], 1e-5)
        and gp[f"candidate_permutation_path_{name}_accuracy_agreement"] == 1.0
        for name in PATH_SCORE_NAMES
    )
    global_integrity = {
        "all_values_finite": all(
            (
                gi["model_state_finite"],
                gi["training_metrics_finite"],
                gi["natural_metrics_finite"],
                gi["pair_metrics_finite"],
                gi["natural_scores_and_samples_finite"],
                gi["pair_scores_and_samples_finite"],
            )
        ),
        "boundary_checkpoint_parameters_resources_clean": (
            gi["student_boundary_clean"]
            and gi["final_checkpoint_clean"]
            and gi["candidate_bank_absent"]
            and gi["candidate_independent_sample_shape_clean"]
            and gi["total_parameters"] < 20_000_000
            and gi["trainable_parameters"] < 18_500_000
            and gi["peak_allocated_vram_gib"] < 18.0
        ),
        "global_candidate_visibility": gn["candidate_cross_font_identity_top1"] >= 0.95,
        "suffix_pixels_and_scores_exact": (
            gp["suffix_pixel_equality"] == 1.0
            and _strictly_below(gp["suffix_path_score_row_max_error"], 1e-6)
        ),
        "candidate_column_equivariance": global_candidate_permutation,
        "spatial_permutation_exactly_invariant": (
            _strictly_below(gn["path_spatial_permutation_max_score_error"], 1e-6)
            and _strictly_below(gn["sample_spatial_permutation_max_score_error"], 1e-6)
            and _strictly_below(gp["path_spatial_permutation_max_score_error"], 1e-6)
            and _strictly_below(gp["sample_spatial_permutation_max_score_error"], 1e-6)
        ),
    }
    matched_gates = {
        "initialized_states_exact": matched["initialized_parameter_states_exact"],
        "final_parameter_counts_exact": matched["final_parameter_counts_exact"],
        "source_data_and_audit_receipts_exact": (
            matched["source_and_data_receipts_exact"]
            and matched["audit_windows_and_pixels_exact"]
        ),
        "both_arms_completed_10000_finite_updates": matched[
            "both_arms_completed_10000_finite_updates"
        ],
        "spatial_pair_accuracy_gain": _strictly_above(
            sp["path_full_arm_accuracy"] - gp["path_full_arm_accuracy"], 0.05
        ),
        "spatial_pair_both_correct_gain": _strictly_above(
            sp["path_full_both_correct_rate"] - gp["path_full_both_correct_rate"],
            0.05,
        ),
        "spatial_natural_top1_gain": _strictly_above(
            sn["path_full_top1"] - gn["path_full_top1"], 0.01
        ),
        "spatial_natural_logp_gain": _strictly_above(
            sn["path_full_target_log_probability"]
            - gn["path_full_target_log_probability"],
            0.05,
        ),
    }
    language = {
        "path_natural_top1_at_least_15_percent": sn["path_full_top1"] >= 0.15,
        "path_full_beats_suffix_top1": _strictly_above(
            sn["path_full_top1"] - sn["path_suffix4_top1"], 0.03
        ),
        "path_full_beats_shuffle_top1": _strictly_above(
            sn["path_full_top1"] - sn["path_shuffled_top1"], 0.03
        ),
        "path_full_beats_unigram_top1": _strictly_above(
            sn["path_full_top1"] - sn["unigram_top1"], 0.03
        ),
        "path_full_beats_bigram_top1": _strictly_above(
            sn["path_full_top1"] - sn["bigram_top1"], 0.01
        ),
        "path_full_beats_bigram_logp": _strictly_above(
            sn["path_full_target_log_probability"]
            - sn["bigram_target_log_probability"],
            0.05,
        ),
        "path_exact_suffix_binding": (
            _strictly_above(sp["path_full_arm_accuracy"], 0.65)
            and _strictly_above(sp["path_full_both_correct_rate"], 0.40)
        ),
        "sample_natural_top1_at_least_5_percent": sn["sample_full_top1"] >= 0.05,
        "sample_full_beats_shuffle_top1": _strictly_above(
            sn["sample_full_top1"] - sn["sample_shuffled_top1"], 0.02
        ),
        "sample_exact_suffix_pair_accuracy": _strictly_above(
            sp["sample_full_arm_accuracy"], 0.60
        ),
    }
    return spatial_common, global_integrity, matched_gates, language


def load_model_checkpoint(
    path: str | Path,
    *,
    expected_route: str,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[ConditionalVisualFieldFlowModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{path} is not a V31 checkpoint")
    if checkpoint.get("route_mode") != expected_route:
        raise ValueError(f"{path} is not the {expected_route} arm")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise PermissionError("V31 smoke checkpoints require --allow-smoke")
    model = ConditionalVisualFieldFlowModel(
        conditional_visual_field_flow_config_from_payload(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def _evaluate_route(
    model: ConditionalVisualFieldFlowModel,
    checkpoint: Mapping[str, Any],
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
    peak = max(evaluator_peak, float(checkpoint.get("peak_allocated_vram_gib", 0.0)))
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


def _write_sample_contact_sheet(
    examples: Sequence[Mapping[str, int]],
    bank_images: torch.Tensor,
    path: Path,
) -> None:
    rows = min(8, len(examples))
    if not rows:
        return
    scale = 3
    tile = 32 * scale
    label_height = 22
    columns = ("target", "full_nearest", "shuffled_nearest")
    canvas = Image.new(
        "RGB", (tile * len(columns), label_height + rows * tile), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("Target", "Full sample", "Shuffled sample")):
        draw.text((column * tile + 4, 5), label, fill="black")
    for row, example in enumerate(examples[:rows]):
        view = int(example["candidate_view"])
        for column, name in enumerate(columns):
            index = int(example[name])
            pixels = (
                bank_images[index, view, 0].detach().cpu().clamp(0, 1).mul(255)
            ).to(torch.uint8)
            image = Image.fromarray(pixels.numpy(), mode="L").convert("RGB")
            image = image.resize((tile, tile), Image.Resampling.NEAREST)
            canvas.paste(image, (column * tile, label_height + row * tile))
            draw.text(
                (column * tile + 3, label_height + row * tile + 3),
                str(index),
                fill=(180, 0, 0),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_joint_audit(
    spatial_model: ConditionalVisualFieldFlowModel,
    spatial_checkpoint: Mapping[str, Any],
    global_model: ConditionalVisualFieldFlowModel,
    global_checkpoint: Mapping[str, Any],
    *,
    manifest: str,
    output: Path,
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
        raise ValueError("V31 route checkpoints mix smoke and evidence states")
    strict = not smoke
    records = load_v25_records(manifest, strict_manifest=strict)
    statistics = build_v31_candidate_statistics(records, bank_size=bank_size)
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
    probes, times, vectors = _fixed_probe_fields(spatial_model, device=device)
    probe_receipt = {
        "audit_seed": AUDIT_SEED,
        "path_times": [float(value) for value in times.cpu()],
        "coherent_probe_fields_sha256": _tensor_sha256(probes),
        "sample_base_vectors_sha256": _tensor_sha256(vectors),
        "sample_count": 8,
        "heun_steps": 8,
    }
    started = time.monotonic()
    spatial = _evaluate_route(
        spatial_model,
        spatial_checkpoint,
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
    matched["probe_and_sample_receipts_exact"] = True
    frozen_images_instantiated = False
    spatial_gates, global_gates, matched_gates, language_gates = v31_gate_report(
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
    contact_sheet = output / "autonomous_sample_nearest_images.png"
    _write_sample_contact_sheet(
        spatial["natural"]["sample_examples"], bank_images, contact_sheet
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
        "probe_and_sample_receipt": probe_receipt,
        "sample_contact_sheet": str(contact_sheet),
        "sample_contact_sheet_sha256": (
            file_sha256(contact_sheet) if contact_sheet.exists() else None
        ),
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
        "spatial_language_and_generation_gates": language_gates,
        "spatial_language_and_generation_selected": all(language_gates.values()),
        "spatial_mechanism_selected": selected,
        "frozen_evaluation_authorized": selected,
        "writer_training_authorized": False,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "evaluation_seconds": time.monotonic() - started,
    }


def _verify_source_receipt(checkpoint: Mapping[str, Any]) -> None:
    source_files = checkpoint.get("protocol", {}).get("source_files_sha256", {})
    if not isinstance(source_files, Mapping) or not source_files:
        raise ValueError("V31 checkpoint has no source-file receipt")
    for path, expected in source_files.items():
        if file_sha256(path) != expected:
            raise ValueError(f"V31 source file changed after training: {path}")


def _verify_evidence_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("smoke_only") or checkpoint.get("exploratory"):
        raise ValueError("V31 evidence audit rejects smoke or exploratory weights")
    if checkpoint.get("step") != 10_000:
        raise ValueError("V31 evidence checkpoint did not finish 10,000 updates")
    if checkpoint.get("finite_updates_verified") != 10_000:
        raise ValueError("V31 checkpoint lacks 10,000 finite-update receipts")
    if checkpoint.get("manifest", {}).get("sha256") != V25_MANIFEST_SHA256:
        raise ValueError("V31 checkpoint has the wrong corpus receipt")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V31 checkpoint has the wrong protocol receipt")
    if protocol.get("expected_protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V31 checkpoint did not pin the V31 protocol")
    _verify_source_receipt(checkpoint)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    spatial_model, spatial_checkpoint = load_model_checkpoint(
        args.spatial_checkpoint,
        expected_route=V31_SPATIAL_ROUTE,
        device=device,
        allow_smoke=args.allow_smoke,
    )
    global_model, global_checkpoint = load_model_checkpoint(
        args.global_checkpoint,
        expected_route=V31_GLOBAL_ROUTE,
        device=device,
        allow_smoke=args.allow_smoke,
    )
    smoke = bool(spatial_checkpoint.get("smoke_only"))
    if smoke != bool(global_checkpoint.get("smoke_only")):
        raise ValueError("V31 audit cannot mix smoke and evidence checkpoints")
    protocol_sha256 = file_sha256(PROTOCOL_DOCUMENT)
    if not smoke:
        if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
            raise ValueError("V31 protocol changed after preregistration")
        if args.windows != NATURAL_WINDOWS or args.pair_windows != PAIR_WINDOWS:
            raise ValueError("V31 evidence audit requires fixed window counts")
        if args.bank_size != AUDIT_BANK_SIZE or args.batch_size != AUDIT_BATCH_SIZE:
            raise ValueError("V31 evidence audit requires fixed bank and batch sizes")
        if args.out != DEFAULT_OUTPUT:
            raise ValueError(f"V31 evidence audit requires --out={DEFAULT_OUTPUT}")
        if device.type != "cuda" or device.index not in (None, 0):
            raise ValueError("V31 evidence audit requires CUDA device 0")
        if args.precision != "bf16":
            raise ValueError("V31 evidence audit requires BF16 autocast")
        _verify_evidence_checkpoint(spatial_checkpoint)
        _verify_evidence_checkpoint(global_checkpoint)
    else:
        _verify_source_receipt(spatial_checkpoint)
        _verify_source_receipt(global_checkpoint)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    report = run_joint_audit(
        spatial_model,
        spatial_checkpoint,
        global_model,
        global_checkpoint,
        manifest=args.manifest,
        output=output,
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
        "probe_and_sample_receipt": report["probe_and_sample_receipt"],
        "sample_contact_sheet_sha256": report["sample_contact_sheet_sha256"],
        "spatial_mechanism_selected": report["spatial_mechanism_selected"],
        "frozen_images_instantiated": False,
        "frozen_evaluation_authorized": report["frozen_evaluation_authorized"],
        "writer_training_authorized": False,
    }
    atomic_write_json(comparison, output / "comparison_receipt.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
