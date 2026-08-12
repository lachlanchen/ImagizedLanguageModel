#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    pack_visual_cells,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    VisualCellAuditDataset,
    VisualCharacterStatistics,
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    render_visual_character_bank,
    visual_cell_audit_collate,
    visual_character_statistics_receipt,
)
from ilm.visual_lm.visual_cell_stream import (
    VisualCellStreamModel,
    visual_cell_model_boundary_receipt,
    visual_cell_model_config_from_payload,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


ARCHITECTURE = "visual-cell-stream-v25"
AUDIT_ARCHITECTURE = "visual-cell-stream-v25-development-audit"
PROTOCOL_DOCUMENT = "references/visual_cell_stream_v25_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CHECKPOINT = (
    "artifacts/visual_cell_stream_v25_evidence/checkpoint_writer_final.pt"
)
LANGUAGE_AUDIT_WINDOWS = 2_048
AUDIT_BANK_SIZE = 1_024
AUDIT_SEED = 20260831
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed V25 natural-Chinese development audit."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--out",
        default="artifacts/visual_cell_stream_v25_development_audit",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--windows", type=int, default=LANGUAGE_AUDIT_WINDOWS)
    parser.add_argument("--bank-size", type=int, default=AUDIT_BANK_SIZE)
    parser.add_argument("--writer-samples", type=int, default=256)
    parser.add_argument("--autonomous-samples", type=int, default=16)
    parser.add_argument("--flow-steps", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def language_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "full_history_top1_gain_over_last": _strictly_above(
            metrics["full_top1"] - metrics["last_top1"], 0.03
        ),
        "full_history_top1_gain_over_unigram": _strictly_above(
            metrics["full_top1"] - metrics["unigram_top1"], 0.03
        ),
        "full_history_log_probability_gain_over_last": _strictly_above(
            metrics["full_target_log_probability"]
            - metrics["last_target_log_probability"],
            0.05,
        ),
        "ordered_history_top1_gain_over_shuffled": _strictly_above(
            metrics["full_top1"] - metrics["shuffled_top1"], 0.015
        ),
        "counterfactual_switch_accuracy": _strictly_above(
            metrics["counterfactual_switch_accuracy"], 0.55
        ),
        "full_history_target_cosine": _strictly_above(
            metrics["full_target_cosine"], 0.55
        ),
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "peak_allocated_vram_below_18_gib": _strictly_below(
            metrics["peak_allocated_vram_gib"], 18.0
        ),
    }


def writer_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "generated_identity_top1": _strictly_above(
            metrics["generated_identity_top1"], 0.20
        ),
        "reread_target_cosine": _strictly_above(
            metrics["reread_target_cosine"], 0.60
        ),
        "generated_pixel_f1": _strictly_above(
            metrics["generated_pixel_f1"], 0.45
        ),
        "blank_rate": _strictly_below(metrics["blank_rate"], 0.05),
        "position16_density_ratio_lower": metrics[
            "autonomous_position16_density_ratio"
        ]
        >= 0.50,
        "position16_density_ratio_upper": metrics[
            "autonomous_position16_density_ratio"
        ]
        <= 1.50,
        "rereads_generated_pixels": metrics["rereads_generated_pixels"] == 1.0,
    }


def student_boundary_is_clean(model: VisualCellStreamModel) -> bool:
    receipt = visual_cell_model_boundary_receipt(model.config)
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_continuous_image",
        "causal_over_visual_time",
        "uses_continuous_flow_time",
        "rereads_generated_pixels",
        "each_time_slice_is_a_clean_2d_cell",
        "geometric_depth_is_one",
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
        raise TypeError("V25 student calls accept floating image tensors only")
    return value.to(device, non_blocking=True)


@torch.no_grad()
def encode_visual_bank(
    model: VisualCellStreamModel,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 128,
) -> torch.Tensor:
    if images.ndim != 5 or tuple(images.shape[2:]) != (1, 32, 32):
        raise ValueError("visual audit bank must be [identities,views,1,32,32]")
    chunks: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            chunks.append(model.encode_cells(batch, target=True))
    return torch.cat(chunks)


def _candidate_logits(
    proposal: torch.Tensor,
    bank_visual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return scale.float() * torch.einsum(
        "bd,nvd->bnv", proposal.float(), bank_visual.float()
    ).amax(dim=2)


def _shuffle_history(
    context: torch.Tensor,
    *,
    first_index: int,
) -> torch.Tensor:
    shuffled = context.clone()
    permutations = []
    for offset in range(context.shape[0]):
        generator = torch.Generator().manual_seed(
            AUDIT_SEED + (first_index + offset) * 104_729
        )
        permutations.append(torch.randperm(context.shape[1] - 1, generator=generator))
    permutation = torch.stack(permutations).to(context.device)
    gather = permutation[:, :, None, None, None].expand(
        -1, -1, *context.shape[2:]
    )
    shuffled[:, :-1] = context[:, :-1].gather(1, gather)
    return shuffled


def _top_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    top = logits.topk(min(5, logits.shape[1]), dim=1).indices
    log_probability = logits.log_softmax(dim=1).gather(1, targets[:, None])[:, 0]
    return {
        "correct_top1": float((top[:, 0] == targets).sum()),
        "correct_top5": float((top == targets[:, None]).any(dim=1).sum()),
        "target_log_probability_sum": float(log_probability.sum()),
    }


def _baseline_metrics(
    statistics: VisualCharacterStatistics,
    targets: Sequence[int],
    last_characters: Sequence[str],
    *,
    alpha: float = 0.10,
) -> dict[str, float]:
    if len(targets) != len(last_characters):
        raise ValueError("baseline targets and contexts must align")
    width = len(statistics.characters)
    unigram = torch.tensor(statistics.counts, dtype=torch.float64) + alpha
    unigram /= unigram.sum()
    unigram_top = unigram.topk(min(5, width)).indices
    output = {
        "unigram_correct_top1": 0.0,
        "unigram_correct_top5": 0.0,
        "unigram_target_log_probability_sum": 0.0,
        "bigram_correct_top1": 0.0,
        "bigram_correct_top5": 0.0,
        "bigram_target_log_probability_sum": 0.0,
    }
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
        bigram_top = row.topk(min(5, width)).indices
        output["bigram_correct_top1"] += float(bigram_top[0] == target)
        output["bigram_correct_top5"] += float((bigram_top == target).any())
        output["bigram_target_log_probability_sum"] += math.log(float(row[target]))
    return output


def _counterfactual_pairs(
    targets: Sequence[int],
    last_characters: Sequence[str],
    *,
    maximum: int = 512,
) -> list[tuple[int, int]]:
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (target, previous) in enumerate(zip(targets, last_characters)):
        grouped[previous][target].append(index)
    pairs: list[tuple[int, int]] = []
    for previous in sorted(grouped):
        target_groups = grouped[previous]
        labels = sorted(target_groups)
        if len(labels) < 2:
            continue
        for offset, first_label in enumerate(labels):
            second_label = labels[(offset + 1) % len(labels)]
            first_items = target_groups[first_label]
            second_items = target_groups[second_label]
            for first, second in zip(first_items, second_items):
                pairs.append((first, second))
                if len(pairs) == maximum:
                    return pairs
    return pairs


@torch.no_grad()
def evaluate_language(
    model: VisualCellStreamModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_visual: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    checkpoint_peak_vram_gib: float = 0.0,
) -> dict[str, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    variants = ("full", "last", "shuffled", "blank")
    totals = {
        name: {
            "correct_top1": 0.0,
            "correct_top5": 0.0,
            "target_log_probability_sum": 0.0,
            "target_cosine_sum": 0.0,
        }
        for name in variants
    }
    all_full_logits: list[torch.Tensor] = []
    all_targets: list[int] = []
    all_last_characters: list[str] = []
    examples = 0
    for raw in loader:
        context = _device_images(raw["context"], device)
        targets = raw["target_index"].to(device)
        reference_target = _device_images(
            raw["reference_continuation"][:, :1], device
        )
        with autocast_context(device, precision):
            target_visual = model.encode_cells(reference_target, target=True)[:, 0]
        contexts = {
            "full": context,
            "last": context[:, -1:],
            "shuffled": _shuffle_history(context, first_index=examples),
            "blank": torch.zeros_like(context),
        }
        for name, model_context in contexts.items():
            with autocast_context(device, precision):
                output = model.language(model_context)
            proposal = output["proposed_visual"][:, -1]
            logits = _candidate_logits(proposal, bank_visual, model.contrastive_scale)
            batch_metrics = _top_metrics(logits, targets)
            for key, value in batch_metrics.items():
                totals[name][key] += value
            totals[name]["target_cosine_sum"] += float(
                (proposal.float() * target_visual.float()).sum(dim=1).sum()
            )
            if name == "full":
                all_full_logits.append(logits.cpu())
        all_targets.extend(raw["target_index"].tolist())
        all_last_characters.extend(raw["last_character"])
        examples += context.shape[0]
    if examples == 0:
        raise ValueError("language audit loader is empty")

    metrics: dict[str, float] = {"examples": float(examples)}
    for name in variants:
        metrics[f"{name}_top1"] = totals[name]["correct_top1"] / examples
        metrics[f"{name}_top5"] = totals[name]["correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            totals[name]["target_log_probability_sum"] / examples
        )
        metrics[f"{name}_target_cosine"] = (
            totals[name]["target_cosine_sum"] / examples
        )

    baselines = _baseline_metrics(statistics, all_targets, all_last_characters)
    for name in ("unigram", "bigram"):
        metrics[f"{name}_top1"] = baselines[f"{name}_correct_top1"] / examples
        metrics[f"{name}_top5"] = baselines[f"{name}_correct_top5"] / examples
        metrics[f"{name}_target_log_probability"] = (
            baselines[f"{name}_target_log_probability_sum"] / examples
        )

    full_logits = torch.cat(all_full_logits)
    pairs = _counterfactual_pairs(all_targets, all_last_characters)
    switch_correct = 0
    top1_switches = 0
    full_top1 = full_logits.argmax(dim=1)
    for first, second in pairs:
        first_target = all_targets[first]
        second_target = all_targets[second]
        first_prefers = full_logits[first, first_target] > full_logits[first, second_target]
        second_prefers = (
            full_logits[second, second_target] > full_logits[second, first_target]
        )
        switch_correct += int(first_prefers and second_prefers)
        top1_switches += int(full_top1[first] != full_top1[second])
    metrics["counterfactual_pairs"] = float(len(pairs))
    metrics["counterfactual_switch_accuracy"] = (
        switch_correct / len(pairs) if pairs else 0.0
    )
    metrics["counterfactual_top1_switch_rate"] = (
        top1_switches / len(pairs) if pairs else 0.0
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


def _pixel_f1(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    generated_ink = generated >= 0.5
    target_ink = target >= 0.5
    true_positive = (generated_ink & target_ink).flatten(1).sum(dim=1).float()
    precision = true_positive / generated_ink.flatten(1).sum(dim=1).float().clamp_min(1)
    recall = true_positive / target_ink.flatten(1).sum(dim=1).float().clamp_min(1)
    return 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)


@torch.no_grad()
def evaluate_writer(
    model: VisualCellStreamModel,
    loader: Iterable[dict[str, Any]],
    statistics: VisualCharacterStatistics,
    bank_visual: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    samples: int,
    autonomous_samples: int,
    candidates: int,
    flow_steps: int,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.eval()
    if samples < 1 or autonomous_samples < 1:
        raise ValueError("writer audit sample counts must be positive")
    index = statistics.index
    totals = {
        "correct": 0.0,
        "cosine": 0.0,
        "pixel_f1": 0.0,
        "blank": 0.0,
        "repeat": 0.0,
        "repeat_eligible": 0.0,
        "examples": 0.0,
        "generated_density": 0.0,
        "target_density": 0.0,
        "autonomous_examples": 0.0,
        "autonomous_adjacent_repeat": 0.0,
        "autonomous_adjacent_pairs": 0.0,
    }
    rereads = True
    sample_payload: dict[str, torch.Tensor] | None = None
    generator = torch.Generator(device=device).manual_seed(AUDIT_SEED + 17)
    for raw in loader:
        remaining = samples - int(totals["examples"])
        if remaining <= 0 and totals["autonomous_examples"] >= autonomous_samples:
            break
        take = min(raw["context"].shape[0], max(0, remaining))
        if take:
            context = _device_images(raw["context"][:take], device)
            reference = _device_images(
                raw["reference_continuation"][:take, :1], device
            )[:, 0]
            targets = raw["target_index"][:take].to(device)
            with autocast_context(device, precision):
                _, trace = model.generate(
                    context,
                    new_cells=1,
                    candidates=candidates,
                    flow_steps=flow_steps,
                    generator=generator,
                )
            generated = trace["generated_cells"][:, 0]
            generated_visual = trace["reread_visual"][:, 0]
            logits = _candidate_logits(
                generated_visual, bank_visual, model.contrastive_scale
            )
            with autocast_context(device, precision):
                target_visual = model.encode_cells(
                    reference[:, None], target=True
                )[:, 0]
            totals["correct"] += float((logits.argmax(dim=1) == targets).sum())
            totals["cosine"] += float(
                (generated_visual.float() * target_visual.float()).sum(dim=1).sum()
            )
            totals["pixel_f1"] += float(_pixel_f1(generated, reference).sum())
            density = generated.float().mean(dim=(1, 2, 3))
            totals["blank"] += float((density < 0.005).sum())
            for row, previous in enumerate(raw["last_character"][:take]):
                previous_index = index.get(previous)
                if previous_index is not None:
                    totals["repeat_eligible"] += 1.0
                    totals["repeat"] += float(
                        logits[row].argmax().item() == previous_index
                    )
            rereads = rereads and bool(trace["reread_generated_pixels"])
            totals["examples"] += take
            if sample_payload is None:
                sample_payload = {
                    "context": context[0, -16:].cpu(),
                    "generated": generated[0:1].cpu(),
                    "reference": reference[0:1].cpu(),
                }

        autonomous_remaining = autonomous_samples - int(
            totals["autonomous_examples"]
        )
        autonomous_take = min(raw["context"].shape[0], max(0, autonomous_remaining))
        if autonomous_take:
            context = _device_images(raw["context"][:autonomous_take], device)
            reference_continuation = _device_images(
                raw["reference_continuation"][:autonomous_take], device
            )
            with autocast_context(device, precision):
                _, trace = model.generate(
                    context,
                    new_cells=16,
                    candidates=candidates,
                    flow_steps=flow_steps,
                    generator=generator,
                )
            generated = trace["generated_cells"]
            density = generated[:, -1].float().mean(dim=(1, 2, 3))
            target_density = reference_continuation[:, -1].float().mean(
                dim=(1, 2, 3)
            )
            totals["generated_density"] += float(density.sum())
            totals["target_density"] += float(target_density.sum())
            adjacent = (
                trace["reread_visual"][:, 1:].float()
                * trace["reread_visual"][:, :-1].float()
            ).sum(dim=-1)
            totals["autonomous_adjacent_repeat"] += float((adjacent > 0.995).sum())
            totals["autonomous_adjacent_pairs"] += float(adjacent.numel())
            rereads = rereads and bool(trace["reread_generated_pixels"])
            totals["autonomous_examples"] += autonomous_take
            if sample_payload is not None and "autonomous" not in sample_payload:
                sample_payload["autonomous"] = generated[0].cpu()

    examples = totals["examples"]
    autonomous = totals["autonomous_examples"]
    if examples != samples or autonomous != autonomous_samples:
        raise ValueError("writer audit loader did not supply the requested samples")
    return {
        "examples": examples,
        "generated_identity_top1": totals["correct"] / examples,
        "reread_target_cosine": totals["cosine"] / examples,
        "generated_pixel_f1": totals["pixel_f1"] / examples,
        "blank_rate": totals["blank"] / examples,
        "repeated_cell_rate": (
            totals["repeat"] / totals["repeat_eligible"]
            if totals["repeat_eligible"]
            else 0.0
        ),
        "repeated_cell_eligible": totals["repeat_eligible"],
        "autonomous_examples": autonomous,
        "autonomous_position16_generated_density": (
            totals["generated_density"] / autonomous
        ),
        "autonomous_position16_target_density": totals["target_density"] / autonomous,
        "autonomous_position16_density_ratio": (
            totals["generated_density"] / max(totals["target_density"], 1e-12)
        ),
        "autonomous_adjacent_repeat_rate": (
            totals["autonomous_adjacent_repeat"]
            / max(totals["autonomous_adjacent_pairs"], 1.0)
        ),
        "rereads_generated_pixels": float(rereads),
    }, sample_payload


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    allow_smoke: bool,
) -> tuple[VisualCellStreamModel, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V25 visual-cell stream")
    if checkpoint.get("smoke_only") and not allow_smoke:
        raise ValueError("smoke-only checkpoint requires --allow-smoke")
    config = visual_cell_model_config_from_payload(checkpoint["model_config"])
    model = VisualCellStreamModel(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), checkpoint


def build_audit_loader(
    records: Sequence[Any],
    statistics: VisualCharacterStatistics,
    *,
    windows: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, tuple[Any, ...]]:
    selected = build_visual_cell_audit_windows(
        records,
        statistics,
        count=windows,
        continuation_cells=16,
        seed=AUDIT_SEED,
    )
    dataset = VisualCellAuditDataset(selected, statistics)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=visual_cell_audit_collate,
    )
    return loader, selected


def _save_sample(payload: dict[str, torch.Tensor] | None, path: Path) -> None:
    if payload is None:
        return
    rows = [payload["context"], payload["reference"], payload["generated"]]
    if "autonomous" in payload:
        rows.append(payload["autonomous"])
    width = max(row.shape[0] for row in rows)
    padded = []
    for row in rows:
        if row.shape[0] < width:
            row = torch.cat(
                (row, torch.zeros(width - row.shape[0], 1, 32, 32)), dim=0
            )
        padded.append(row)
    pack_visual_cells(torch.cat(padded), columns=width, gutter=1).save(path)


def run_development_audit(
    model: VisualCellStreamModel,
    checkpoint: dict[str, Any],
    *,
    manifest: str,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
    windows: int,
    bank_size: int,
    writer_samples: int,
    autonomous_samples: int,
    candidates: int,
    flow_steps: int,
    evaluate_pixels: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    records = load_v25_records(manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(records, bank_size=bank_size)
    loader, audit_windows = build_audit_loader(
        records,
        statistics,
        windows=windows,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    bank_images = render_visual_character_bank(statistics)
    bank_visual = encode_visual_bank(
        model,
        bank_images,
        device=device,
        precision=precision,
    )
    language = evaluate_language(
        model,
        loader,
        statistics,
        bank_visual,
        device=device,
        precision=precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    language_gates = language_gate_report(language)
    writer: dict[str, float] | None = None
    writer_gates: dict[str, bool] | None = None
    sample = None
    if evaluate_pixels:
        writer, sample = evaluate_writer(
            model,
            loader,
            statistics,
            bank_visual,
            device=device,
            precision=precision,
            samples=writer_samples,
            autonomous_samples=autonomous_samples,
            candidates=candidates,
            flow_steps=flow_steps,
        )
        writer_gates = writer_gate_report(writer)
    return {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint": str(checkpoint.get("checkpoint_path", "in-memory")),
        "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_smoke_only": bool(checkpoint.get("smoke_only")),
        "manifest": verify_v25_manifest(manifest, strict=True),
        "partition": visual_cell_partition_receipt(records),
        "fonts": visual_cell_font_manifest(),
        "statistics": visual_character_statistics_receipt(statistics),
        "audit_windows": len(audit_windows),
        "frozen_images_instantiated": False,
        "language": language,
        "language_gates": language_gates,
        "language_selected": all(language_gates.values()),
        "symbolic_bigram_is_benchmark_only": True,
        "beats_symbolic_bigram_top1": language["full_top1"] > language["bigram_top1"],
        "writer": writer,
        "writer_gates": writer_gates,
        "writer_selected": bool(writer_gates) and all(writer_gates.values()),
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
    }, sample


def main() -> None:
    args = parse_args()
    if args.windows < 1 or args.bank_size < 2:
        raise ValueError("audit windows and bank size must be positive")
    seed_everything(AUDIT_SEED)
    device = choose_device(args.device)
    model, checkpoint = load_model_checkpoint(
        args.checkpoint,
        device=device,
        allow_smoke=args.allow_smoke,
    )
    checkpoint["checkpoint_path"] = str(args.checkpoint)
    evaluate_pixels = checkpoint.get("stage") == "writer"
    started = time.monotonic()
    report, sample = run_development_audit(
        model,
        checkpoint,
        manifest=args.manifest,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        windows=args.windows,
        bank_size=args.bank_size,
        writer_samples=args.writer_samples,
        autonomous_samples=args.autonomous_samples,
        candidates=args.candidates,
        flow_steps=args.flow_steps,
        evaluate_pixels=evaluate_pixels,
    )
    report["elapsed_seconds"] = time.monotonic() - started
    report["device"] = str(device)
    report["checkpoint_sha256"] = file_sha256(args.checkpoint)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "development_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_sample(sample, output / "writer_sample.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
