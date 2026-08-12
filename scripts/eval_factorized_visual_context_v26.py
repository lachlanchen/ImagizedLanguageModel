#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.factorized_visual_context import (
    FactorizedVisualContextModel,
    factorized_visual_context_boundary_receipt,
    factorized_visual_context_config_from_payload,
    particle_candidate_scores,
    particle_energy_score,
    particle_target_scores,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualAuditDataset,
    FactorizedVisualPairAuditDataset,
    build_factorized_audit_windows,
    build_factorized_suffix_pairs,
    factorized_visual_audit_collate,
    factorized_visual_pair_collate,
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


ARCHITECTURE = "factorized-visual-context-v26"
AUDIT_ARCHITECTURE = "factorized-visual-context-v26-development-audit"
PROTOCOL_DOCUMENT = "references/factorized_visual_context_v26_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/factorized_visual_context_v26_evidence/checkpoint_final.pt"
)
AUDIT_SEED = 20260911
NATURAL_WINDOWS = 2_048
PAIR_WINDOWS = 512
AUDIT_BANK_SIZE = 1_024
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the preregistered V26 image-only development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out",
        default="artifacts/factorized_visual_context_v26_development_audit",
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


def v26_gate_report(
    natural: Mapping[str, float],
    suffix4: Mapping[str, float],
) -> tuple[dict[str, bool], dict[str, bool]]:
    mechanism = {
        "full_top1_gain_over_last": _strictly_above(
            natural["full_top1"] - natural["last_top1"], 0.02
        ),
        "full_top1_gain_over_unigram": _strictly_above(
            natural["full_top1"] - natural["unigram_top1"], 0.03
        ),
        "full_log_probability_gain_over_last": _strictly_above(
            natural["full_target_log_probability"]
            - natural["last_target_log_probability"],
            0.10,
        ),
        "full_log_probability_gain_over_suffix4": _strictly_above(
            natural["full_target_log_probability"]
            - natural["suffix_4_target_log_probability"],
            0.03,
        ),
        "ordered_prefix_log_probability_gain": _strictly_above(
            natural["full_target_log_probability"]
            - natural["shuffled_prefix_target_log_probability"],
            0.03,
        ),
        "suffix4_pair_ranking_accuracy": _strictly_above(
            suffix4["pair_ranking_accuracy"], 0.65
        ),
        "swapped_residual_target_accuracy": _strictly_above(
            suffix4["swapped_residual_target_accuracy"], 0.65
        ),
        "suffix4_pixels_exact": suffix4["suffix_pixel_equality"] == 1.0,
        "retina_bank_oracle_top1": natural["retina_bank_oracle_top1"] >= 0.99,
        "student_boundary_clean": natural["student_boundary_clean"] == 1.0,
        "peak_allocated_vram_below_18_gib": _strictly_below(
            natural["peak_allocated_vram_gib"], 18.0
        ),
    }
    language = {
        "full_top1_gain_over_symbolic_bigram": _strictly_above(
            natural["full_top1"] - natural["bigram_top1"], 0.01
        ),
        "full_log_probability_gain_over_symbolic_bigram": _strictly_above(
            natural["full_target_log_probability"]
            - natural["bigram_target_log_probability"],
            0.05,
        ),
    }
    return mechanism, language


def student_boundary_is_clean(model: FactorizedVisualContextModel) -> bool:
    receipt = factorized_visual_context_boundary_receipt(model.config)
    required_true = {
        "output_is_continuous_visual_distribution",
        "last_appearance_is_factorized",
        "earlier_history_is_factorized",
        "history_can_be_zeroed_or_swapped",
        "uses_continuous_particle_noise",
    }
    required_false = {
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_character_ids",
        "uses_vocabulary_embedding",
        "uses_ocr",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "candidate_bank_deployed",
    }
    return (
        receipt.get("architecture") == ARCHITECTURE
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def _device_images(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not torch.is_floating_point(value):
        raise TypeError("V26 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


@torch.no_grad()
def encode_visual_bank(
    model: FactorizedVisualContextModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> torch.Tensor:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("V26 visual bank must be [identity,view,1,32,32]")
    chunks: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            chunks.append(model.encode_cells(batch, target=True))
    return torch.cat(chunks)


def _identity_scores(
    particles: torch.Tensor,
    bank_visual: torch.Tensor,
) -> torch.Tensor:
    identities, views, dimension = bank_visual.shape
    flat = bank_visual.reshape(identities * views, dimension)
    scores = particle_candidate_scores(particles, flat)
    return scores.reshape(particles.shape[0], identities, views).amax(dim=2)


def _top_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = logits.log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(log_probability.detach().sum()),
    }


def _particle_spread(particles: torch.Tensor) -> torch.Tensor:
    count = particles.shape[1]
    similarity = torch.einsum(
        "bkd,bjd->bkj", particles.float(), particles.float()
    )
    distance = (2.0 - 2.0 * similarity.clamp(-1, 1)).clamp_min(0).sqrt()
    mask = ~torch.eye(count, device=particles.device, dtype=torch.bool)
    return distance[:, mask].mean(dim=1)


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


@torch.no_grad()
def evaluate_natural_language(
    model: FactorizedVisualContextModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_visual: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    variants = (
        "full",
        "last",
        "suffix_2",
        "suffix_4",
        "suffix_8",
        "suffix_16",
        "suffix_32",
        "zeroed_prefix",
        "shuffled_prefix",
    )
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
            "target_score_sum": 0.0,
            "nearest_target_cosine_sum": 0.0,
            "energy_score_sum": 0.0,
            "particle_spread_sum": 0.0,
        }
        for name in variants
    }
    all_targets: list[int] = []
    all_last_characters: list[str] = []
    oracle_correct = 0.0
    oracle_top5 = 0.0
    examples = 0
    started = time.monotonic()
    for raw in loader:
        context = _device_images(raw["context"], device)
        reference_target = _device_images(raw["reference_future"][:, :1], device)
        targets = raw["target_index"].to(device)
        with autocast_context(device, precision):
            target_visual = model.encode_cells(reference_target, target=True)[:, 0]
        zeroed = context.clone()
        zeroed[:, :-4] = 0
        contexts = {
            "full": context,
            "last": context[:, -1:],
            "suffix_2": context[:, -2:],
            "suffix_4": context[:, -4:],
            "suffix_8": context[:, -8:],
            "suffix_16": context[:, -16:],
            "suffix_32": context[:, -32:],
            "zeroed_prefix": zeroed,
            "shuffled_prefix": _shuffle_prefix(context, first_index=examples),
        }
        for name, model_context in contexts.items():
            with autocast_context(device, precision):
                output = model.language(model_context, horizons=(1,))
            particles = output["particles"][:, 0]
            raw_scores = _identity_scores(particles, bank_visual)
            logits = model.contrastive_scale.float() * raw_scores
            top = _top_metrics(logits, targets)
            for key, value in top.items():
                totals[name][key] += value
            target_score = particle_target_scores(particles, target_visual)
            target_cosine = torch.einsum(
                "bkd,bd->bk", particles.float(), target_visual.float()
            ).amax(dim=1)
            energy = particle_energy_score(
                particles[:, None], target_visual[:, None]
            )[:, 0]
            totals[name]["target_score_sum"] += float(target_score.sum())
            totals[name]["nearest_target_cosine_sum"] += float(target_cosine.sum())
            totals[name]["energy_score_sum"] += float(energy.sum())
            totals[name]["particle_spread_sum"] += float(
                _particle_spread(particles).sum()
            )

        oracle_similarity = torch.einsum(
            "bd,nvd->bnv", target_visual.float(), bank_visual.float()
        ).amax(dim=2)
        oracle_top = oracle_similarity.topk(min(5, bank_visual.shape[0]), dim=1).indices
        oracle_correct += float((oracle_top[:, 0] == targets).sum())
        oracle_top5 += float((oracle_top == targets[:, None]).any(dim=1).sum())
        all_targets.extend(raw["target_index"].tolist())
        all_last_characters.extend(raw["last_character"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("V26 natural audit loader is empty")
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {
        "examples": float(examples),
        "evaluation_seconds": elapsed,
        "context_examples_per_second": examples * len(variants) / max(elapsed, 1e-9),
    }
    for name in variants:
        metrics[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        metrics[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
        metrics[f"{name}_target_score"] = totals[name]["target_score_sum"] / examples
        metrics[f"{name}_nearest_target_cosine"] = (
            totals[name]["nearest_target_cosine_sum"] / examples
        )
        metrics[f"{name}_energy_score"] = totals[name]["energy_score_sum"] / examples
        metrics[f"{name}_particle_spread"] = (
            totals[name]["particle_spread_sum"] / examples
        )
    metrics["suffix_64_top1"] = metrics["full_top1"]
    metrics["suffix_64_top5"] = metrics["full_top5"]
    metrics["suffix_64_target_log_probability"] = metrics[
        "full_target_log_probability"
    ]
    baseline = _baseline_metrics(statistics, all_targets, all_last_characters)
    for name in ("unigram", "bigram"):
        metrics[f"{name}_top1"] = baseline[f"{name}_correct_top1"] / examples
        metrics[f"{name}_top5"] = baseline[f"{name}_correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            baseline[f"{name}_target_log_probability_sum"] / examples
        )
    metrics["retina_bank_oracle_top1"] = oracle_correct / examples
    metrics["retina_bank_oracle_top5"] = oracle_top5 / examples
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


@torch.no_grad()
def evaluate_suffix_pairs(
    model: FactorizedVisualContextModel,
    loader: Iterable[dict[str, Any]],
    bank_visual: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals = {
        "pairs": 0.0,
        "ranking_correct": 0.0,
        "ranking_arms": 0.0,
        "both_correct": 0.0,
        "margin_sum": 0.0,
        "top1_switches": 0.0,
        "swapped_correct": 0.0,
        "swapped_arms": 0.0,
        "swapped_margin_sum": 0.0,
        "suffix_equal": 0.0,
        "suffix_checks": 0.0,
        "appearance_difference_sum": 0.0,
        "history_difference_sum": 0.0,
    }
    for raw in loader:
        context_a = _device_images(raw["context_a"], device)
        context_b = _device_images(raw["context_b"], device)
        target_a = _device_images(raw["reference_target_a"], device)
        target_b = _device_images(raw["reference_target_b"], device)
        target_index_a = raw["target_index_a"].to(device)
        target_index_b = raw["target_index_b"].to(device)
        suffix_cells = int(raw["metadata"][0]["suffix_cells"])
        suffix_equal = (
            context_a[:, -suffix_cells:].cpu()
            == context_b[:, -suffix_cells:].cpu()
        ).flatten(1).all(dim=1)
        with autocast_context(device, precision):
            parts_a = model.factorize(context_a)
            parts_b = model.factorize(context_b)
            particles_a = model.predict_particles_from_state(
                parts_a["fused_state"], horizons=(1,)
            )[:, 0]
            particles_b = model.predict_particles_from_state(
                parts_b["fused_state"], horizons=(1,)
            )[:, 0]
            visual_a = model.encode_cells(target_a, target=True)[:, 0]
            visual_b = model.encode_cells(target_b, target=True)[:, 0]

        score_aa = particle_target_scores(particles_a, visual_a)
        score_ab = particle_target_scores(particles_a, visual_b)
        score_bb = particle_target_scores(particles_b, visual_b)
        score_ba = particle_target_scores(particles_b, visual_a)
        margin_a = score_aa - score_ab
        margin_b = score_bb - score_ba
        correct_a = margin_a > 0
        correct_b = margin_b > 0

        normal_top_a = _identity_scores(particles_a, bank_visual).argmax(dim=1)
        normal_top_b = _identity_scores(particles_b, bank_visual).argmax(dim=1)

        with autocast_context(device, precision):
            fused_swap_a = model.fuse_parts(
                parts_a["appearance_state"], parts_b["history_residual"]
            )["fused_state"]
            fused_swap_b = model.fuse_parts(
                parts_b["appearance_state"], parts_a["history_residual"]
            )["fused_state"]
            swap_a = model.predict_particles_from_state(
                fused_swap_a, horizons=(1,)
            )[:, 0]
            swap_b = model.predict_particles_from_state(
                fused_swap_b, horizons=(1,)
            )[:, 0]
        swap_a_donor = particle_target_scores(swap_a, visual_b)
        swap_a_host = particle_target_scores(swap_a, visual_a)
        swap_b_donor = particle_target_scores(swap_b, visual_a)
        swap_b_host = particle_target_scores(swap_b, visual_b)
        swap_margin_a = swap_a_donor - swap_a_host
        swap_margin_b = swap_b_donor - swap_b_host

        batch = context_a.shape[0]
        totals["pairs"] += batch
        totals["ranking_correct"] += float(correct_a.sum() + correct_b.sum())
        totals["ranking_arms"] += 2 * batch
        totals["both_correct"] += float((correct_a & correct_b).sum())
        totals["margin_sum"] += float(margin_a.sum() + margin_b.sum())
        totals["top1_switches"] += float((normal_top_a != normal_top_b).sum())
        totals["swapped_correct"] += float(
            (swap_margin_a > 0).sum() + (swap_margin_b > 0).sum()
        )
        totals["swapped_arms"] += 2 * batch
        totals["swapped_margin_sum"] += float(
            swap_margin_a.sum() + swap_margin_b.sum()
        )
        totals["suffix_equal"] += float(suffix_equal.sum())
        totals["suffix_checks"] += batch
        totals["appearance_difference_sum"] += float(
            (parts_a["appearance_state"] - parts_b["appearance_state"])
            .float()
            .norm(dim=1)
            .sum()
        )
        totals["history_difference_sum"] += float(
            (parts_a["history_residual"] - parts_b["history_residual"])
            .float()
            .norm(dim=1)
            .sum()
        )
        if torch.any(target_index_a == target_index_b):
            raise RuntimeError("V26 pair evaluator received identical targets")
    pairs = totals["pairs"]
    if not pairs:
        raise ValueError("V26 suffix-pair audit loader is empty")
    return {
        "pairs": pairs,
        "pair_ranking_accuracy": totals["ranking_correct"] / totals["ranking_arms"],
        "pair_both_correct_rate": totals["both_correct"] / pairs,
        "mean_pair_margin": totals["margin_sum"] / totals["ranking_arms"],
        "top1_switch_rate": totals["top1_switches"] / pairs,
        "swapped_residual_target_accuracy": (
            totals["swapped_correct"] / totals["swapped_arms"]
        ),
        "swapped_residual_mean_margin": (
            totals["swapped_margin_sum"] / totals["swapped_arms"]
        ),
        "suffix_pixel_equality": totals["suffix_equal"] / totals["suffix_checks"],
        "mean_appearance_state_difference": (
            totals["appearance_difference_sum"] / pairs
        ),
        "mean_history_residual_difference": (
            totals["history_difference_sum"] / pairs
        ),
    }


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[FactorizedVisualContextModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V26 factorized visual context")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise ValueError("smoke-only V26 checkpoint requires --allow-smoke")
    config = factorized_visual_context_config_from_payload(
        checkpoint["model_config"]
    )
    model = FactorizedVisualContextModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


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
    suffix_cells: int,
    count: int,
    seed: int,
    require_different_identifiers: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, tuple[Any, ...]]:
    pairs = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=suffix_cells,
        count=count,
        seed=seed,
        require_different_identifiers=require_different_identifiers,
        allowed_targets=set(statistics.characters),
    )
    dataset = FactorizedVisualPairAuditDataset(
        pairs, character_index=statistics.index
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=factorized_visual_pair_collate,
    ), pairs


def run_development_audit(
    model: FactorizedVisualContextModel,
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
    records = load_v25_records(manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(records, bank_size=bank_size)
    natural_loader, natural_windows = _natural_loader(
        records,
        statistics,
        windows=windows,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    suffix4_loader, suffix4_pairs = _pair_loader(
        records,
        statistics,
        suffix_cells=4,
        count=pair_windows,
        seed=AUDIT_SEED,
        require_different_identifiers=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    suffix8_loader, suffix8_pairs = _pair_loader(
        records,
        statistics,
        suffix_cells=8,
        count=pair_windows,
        seed=AUDIT_SEED + 1,
        require_different_identifiers=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bank_visual = encode_visual_bank(
        model,
        render_visual_character_bank(statistics),
        device=device,
        precision=precision,
    )
    natural = evaluate_natural_language(
        model,
        natural_loader,
        statistics,
        bank_visual,
        device=device,
        precision=precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    suffix4 = evaluate_suffix_pairs(
        model,
        suffix4_loader,
        bank_visual,
        device=device,
        precision=precision,
    )
    suffix8 = evaluate_suffix_pairs(
        model,
        suffix8_loader,
        bank_visual,
        device=device,
        precision=precision,
    )
    mechanism_gates, language_gates = v26_gate_report(natural, suffix4)
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_smoke_only": bool(checkpoint.get("smoke_only")),
        "manifest": verify_v25_manifest(manifest, strict=True),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": visual_character_statistics_receipt(statistics),
        "natural_windows": len(natural_windows),
        "suffix4_pairs": len(suffix4_pairs),
        "suffix8_pairs": len(suffix8_pairs),
        "suffix4_pairs_require_different_identifiers": True,
        "suffix8_pairs_are_diagnostic_only": True,
        "frozen_images_instantiated": False,
        "natural": natural,
        "suffix4": suffix4,
        "suffix8_diagnostic": suffix8,
        "mechanism_gates": mechanism_gates,
        "mechanism_selected": all(mechanism_gates.values()),
        "language_gates": language_gates,
        "language_selected": all(mechanism_gates.values())
        and all(language_gates.values()),
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
    }


def main() -> None:
    args = parse_args()
    if min(args.windows, args.pair_windows, args.bank_size, args.batch_size) < 1:
        raise ValueError("V26 audit sizes must be positive")
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
