#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_binding_v44 import (
    CanonicalGlyphBindingV44,
    V44_ARCHITECTURE,
    canonical_glyph_binding_v44_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_binding_v44_evaluation import V44_AUDIT_SEED
from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphPairAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    canonical_glyph_pair_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.factorized_visual_context_data import build_factorized_suffix_pairs
from ilm.visual_lm.visual_cell_data import load_v25_records, verify_v25_manifest
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from scripts.train_canonical_glyph_binding_v44 import (
    DEFAULT_BASE,
    DEFAULT_MANIFEST,
    _atomic_json,
    _base_matches,
    _resolve_device,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = "artifacts/canonical_glyph_binding_v44_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/canonical_glyph_binding_v44_20260814/diagnostic"
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0, 1.50, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-result V44 residual direction and scale diagnosis."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--development-pairs", type=int, default=512)
    parser.add_argument("--train-pairs", type=int, default=1_024)
    return parser.parse_args()


def _interpolate(
    base: torch.Tensor,
    adapted: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    return F.normalize(base.float() + alpha * (adapted.float() - base.float()), dim=-1)


def _empty_sweep() -> dict[float, dict[str, float]]:
    return {
        alpha: {
            "examples": 0.0,
            "correct": 0.0,
            "target_log_probability": 0.0,
            "target_cosine": 0.0,
            "mean_field_cosine": 0.0,
        }
        for alpha in ALPHAS
    }


@torch.no_grad()
def natural_alpha_sweep(
    model: CanonicalGlyphBindingV44,
    loader: Iterable[dict[str, Any]],
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, dict[str, float]]:
    bank_fields = model.field.encode_unit(bank_images.to(device)).float()
    mean_field = F.normalize(bank_fields.mean(dim=0), dim=-1)
    totals = _empty_sweep()
    for raw in loader:
        context = raw["context"].to(device, non_blocking=True)
        targets = raw["target_index"].to(device, non_blocking=True)
        target_fields = bank_fields[targets]
        with autocast_context(device, precision):
            output = model.language(context)
        base = output["base_anchor_fields"][:, -1]
        adapted = output["anchor_fields"][:, -1]
        rows = torch.arange(len(context), device=device)
        for alpha in ALPHAS:
            anchor = _interpolate(base, adapted, alpha)
            logits = model.contrastive_scale.float() * (
                anchor @ bank_fields.transpose(0, 1)
            )
            log_probability = logits.log_softmax(dim=-1)[rows, targets]
            total = totals[alpha]
            total["examples"] += len(context)
            total["correct"] += float((logits.argmax(dim=-1) == targets).sum())
            total["target_log_probability"] += float(log_probability.sum())
            total["target_cosine"] += float((anchor * target_fields).sum())
            total["mean_field_cosine"] += float((anchor * mean_field).sum())
    report: dict[str, dict[str, float]] = {}
    for alpha, total in totals.items():
        count = total["examples"]
        report[f"{alpha:.2f}"] = {
            "examples": count,
            "top1": total["correct"] / count,
            "target_log_probability": total["target_log_probability"] / count,
            "target_cosine": total["target_cosine"] / count,
            "mean_field_cosine": total["mean_field_cosine"] / count,
        }
    return report


def _pair_metrics(
    logits: torch.Tensor,
    assignment: torch.Tensor,
) -> tuple[float, float, float]:
    correct = logits.gather(2, assignment[:, :, None])[:, :, 0]
    other = logits.gather(2, (1 - assignment)[:, :, None])[:, :, 0]
    margins = correct - other
    accuracy = (margins > 0).float().mean()
    both = (margins > 0).all(dim=1).float().mean()
    return float(accuracy), float(both), float(margins.mean())


@torch.no_grad()
def pair_alpha_sweep(
    model: CanonicalGlyphBindingV44,
    loader: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    totals = {
        alpha: {"arms": 0.0, "correct": 0.0, "pairs": 0.0, "both": 0.0, "margin": 0.0}
        for alpha in ALPHAS
    }
    alignment = {
        "pairs": 0.0,
        "base_delta_cosine": 0.0,
        "adapted_delta_cosine": 0.0,
        "learned_update_delta_cosine": 0.0,
        "base_delta_norm": 0.0,
        "adapted_delta_norm": 0.0,
        "target_delta_norm": 0.0,
        "learned_update_delta_norm": 0.0,
    }
    for raw in loader:
        contexts = raw["contexts"].to(device, non_blocking=True)
        candidates = raw["candidates"].to(device, non_blocking=True)
        assignment = raw["assignment"].to(device, non_blocking=True)
        batch = len(contexts)
        with autocast_context(device, precision):
            output = model.language(contexts.flatten(0, 1))
        base = output["base_anchor_fields"][:, -1].reshape(batch, 2, -1).float()
        adapted = output["anchor_fields"][:, -1].reshape(batch, 2, -1).float()
        fields = model.field.encode_unit(candidates).float()
        assigned = fields.gather(1, assignment[..., None].expand_as(fields))
        count = batch
        for alpha in ALPHAS:
            anchor = _interpolate(base, adapted, alpha)
            logits = model.contrastive_scale.float() * torch.einsum(
                "bid,bjd->bij", anchor, fields
            )
            accuracy, both, margin = _pair_metrics(logits, assignment)
            total = totals[alpha]
            total["arms"] += 2 * count
            total["correct"] += accuracy * 2 * count
            total["pairs"] += count
            total["both"] += both * count
            total["margin"] += margin * 2 * count

        base_delta = base[:, 0] - base[:, 1]
        adapted_delta = adapted[:, 0] - adapted[:, 1]
        update_delta = (adapted - base)[:, 0] - (adapted - base)[:, 1]
        target_delta = assigned[:, 0] - assigned[:, 1]
        alignment["pairs"] += count
        for name, value in (
            ("base_delta", base_delta),
            ("adapted_delta", adapted_delta),
            ("learned_update_delta", update_delta),
        ):
            alignment[f"{name}_cosine"] += float(
                F.cosine_similarity(value, target_delta, dim=-1).sum()
            )
            alignment[f"{name}_norm"] += float(value.norm(dim=-1).sum())
        alignment["target_delta_norm"] += float(target_delta.norm(dim=-1).sum())

    sweep: dict[str, dict[str, float]] = {}
    for alpha, total in totals.items():
        sweep[f"{alpha:.2f}"] = {
            "pairs": total["pairs"],
            "arm_accuracy": total["correct"] / total["arms"],
            "both_correct_rate": total["both"] / total["pairs"],
            "mean_margin": total["margin"] / total["arms"],
        }
    count = alignment.pop("pairs")
    normalized_alignment = {
        key: value / count for key, value in alignment.items()
    }
    return {"alpha_sweep": sweep, "delta_alignment": normalized_alignment}


def _pair_loader(
    pairs,
    *,
    render_config: CanonicalGlyphRenderConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        CanonicalGlyphPairAuditDataset(pairs, render_config=render_config),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V44_AUDIT_SEED + 99)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V44_ARCHITECTURE or checkpoint.get(
        "smoke_only"
    ):
        raise ValueError("V44 diagnosis requires the production checkpoint")
    base_payload = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    model = CanonicalGlyphBindingV44(
        canonical_glyph_language_config_from_payload(checkpoint["language_config"]),
        canonical_glyph_binding_v44_config_from_payload(checkpoint["v44_config"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_base()
    if not _base_matches(model, base_payload["model"]):
        raise RuntimeError("V44 diagnosis found a changed V42 base")
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    manifest_receipt = verify_v25_manifest(args.manifest, strict=True)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V44 diagnostic corpus differs from training")
    records = load_v25_records(args.manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(
        records,
        bank_size=args.bank_size,
        script_views_mode=render_config.script_views,
    )
    windows = build_visual_cell_audit_windows(
        records,
        statistics,
        count=args.windows,
        continuation_cells=16,
        seed=V44_AUDIT_SEED,
        script_views_mode=render_config.script_views,
    )
    audit_loader = DataLoader(
        CanonicalGlyphAuditDataset(windows, statistics, render_config=render_config),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_audit_collate,
    )
    development_pairs = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=args.development_pairs,
        seed=V44_AUDIT_SEED + 1,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
        script_views_mode=render_config.script_views,
    )
    pool = checkpoint["pair_pool"]
    train_count = int(pool["training"]["count"])
    holdout_count = int(pool["holdout"]["count"])
    all_train_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=train_count + holdout_count,
        seed=int(pool["seed"]),
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    unseen_pairs = all_train_pairs[train_count : train_count + args.train_pairs]
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )
    started = time.perf_counter()
    report = {
        "experiment": "canonical-glyph-binding-v44-post-result-diagnostic",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "development_decision_unchanged": True,
        "natural_alpha_sweep": natural_alpha_sweep(
            model,
            audit_loader,
            bank_images,
            device=device,
            precision=args.precision,
        ),
        "development_pair_diagnostic": pair_alpha_sweep(
            model,
            _pair_loader(
                development_pairs,
                render_config=render_config,
                args=args,
                device=device,
            ),
            device=device,
            precision=args.precision,
        ),
        "unseen_train_pair_diagnostic": pair_alpha_sweep(
            model,
            _pair_loader(
                unseen_pairs,
                render_config=render_config,
                args=args,
                device=device,
            ),
            device=device,
            precision=args.precision,
        ),
        "alphas": list(ALPHAS),
        "writer_opened": False,
        "frozen_partition_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(report, output / "diagnostic_report.json")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
