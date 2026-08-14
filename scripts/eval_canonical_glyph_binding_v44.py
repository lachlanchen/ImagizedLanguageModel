#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_binding_v44 import (
    CanonicalGlyphBindingV44,
    V44_ARCHITECTURE,
    canonical_glyph_binding_v44_boundary_receipt,
    canonical_glyph_binding_v44_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_binding_v44_evaluation import (
    V44_AUDIT_SEED,
    canonical_glyph_binding_v44_boundary_is_clean,
    canonical_glyph_binding_v44_gate_report,
)
from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
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
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    evaluate_canonical_language,
    evaluate_counterfactual_pairs,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    visual_character_statistics_receipt,
)
from scripts.train_canonical_glyph_binding_v44 import (
    DEFAULT_BASE,
    DEFAULT_MANIFEST,
    PINNED_V42_SHA256,
    PROTOCOL_DOCUMENT,
    _atomic_json,
    _base_matches,
    _resolve_device,
    pair_sequence_receipt,
    tensor_state_sha256,
)
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_CHECKPOINT = "artifacts/canonical_glyph_binding_v44_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/canonical_glyph_binding_v44_20260814/development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed matched-base V44 development audit."
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
    parser.add_argument("--consumed-pairs", type=int, default=1_024)
    parser.add_argument("--unseen-pairs", type=int, default=1_024)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def _pair_loader(
    pairs,
    *,
    render_config: CanonicalGlyphRenderConfig,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = CanonicalGlyphPairAuditDataset(
        pairs,
        render_config=render_config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.development_pairs,
        args.consumed_pairs,
        args.unseen_pairs,
    ) < 1:
        raise ValueError("V44 audit sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V44_AUDIT_SEED)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V44_ARCHITECTURE:
        raise ValueError("audit checkpoint is not V44")
    smoke_only = bool(checkpoint.get("smoke_only"))
    if smoke_only and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    if not smoke_only and (
        args.windows != 2_048
        or args.bank_size != 1_024
        or args.development_pairs != 512
        or args.consumed_pairs != 1_024
        or args.unseen_pairs != 1_024
    ):
        raise ValueError("V44 production audit sizes are preregistered and fixed")

    base_checkpoint_sha256 = file_sha256(args.base_checkpoint)
    if base_checkpoint_sha256 != PINNED_V42_SHA256:
        raise ValueError("V44 audit requires the pinned V42 base checkpoint")
    if checkpoint["base_v42_checkpoint_sha256"] != base_checkpoint_sha256:
        raise ValueError("V44 audit base checkpoint differs from training")
    base_payload = torch.load(
        args.base_checkpoint, map_location="cpu", weights_only=False
    )
    expected_base_state = base_payload["model"]
    expected_base_state_sha256 = tensor_state_sha256(expected_base_state)
    if checkpoint["base_v42_state_sha256"] != expected_base_state_sha256:
        raise ValueError("V44 stored base-state receipt differs from V42")
    language_config = canonical_glyph_language_config_from_payload(
        checkpoint["language_config"]
    )
    model = CanonicalGlyphBindingV44(
        language_config,
        canonical_glyph_binding_v44_config_from_payload(checkpoint["v44_config"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.freeze_base()
    base_model = CanonicalGlyphLanguageModel(language_config)
    base_model.load_state_dict(expected_base_state, strict=True)
    base_model.requires_grad_(False).eval()
    base_state_exact = _base_matches(model, expected_base_state)
    if not base_state_exact:
        raise RuntimeError("V44 checkpoint does not preserve the exact V42 base")
    model.to(device).eval()
    base_model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not smoke_only
    manifest_receipt = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V44 audit corpus differs from training")
    records = load_v25_records(args.manifest, strict_manifest=strict_manifest)
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
    audit_dataset = CanonicalGlyphAuditDataset(
        windows,
        statistics,
        render_config=render_config,
    )
    audit_loader = DataLoader(
        audit_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_audit_collate,
    )
    development_pair_records = build_factorized_suffix_pairs(
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
    train_pair_count = int(pool["training"]["count"])
    holdout_pair_count = int(pool["holdout"]["count"])
    all_train_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=train_pair_count + holdout_pair_count,
        seed=int(pool["seed"]),
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    training_pairs = all_train_pairs[:train_pair_count]
    holdout_pairs = all_train_pairs[train_pair_count:]
    if pair_sequence_receipt(training_pairs) != pool["training"]:
        raise ValueError("V44 reconstructed training-pair receipt differs")
    if pair_sequence_receipt(holdout_pairs) != pool["holdout"]:
        raise ValueError("V44 reconstructed holdout-pair receipt differs")
    if args.consumed_pairs > len(training_pairs) or args.unseen_pairs > len(
        holdout_pairs
    ):
        raise ValueError("V44 requested pair audit exceeds its fixed pools")
    consumed_records = training_pairs[: args.consumed_pairs]
    unseen_records = holdout_pairs[: args.unseen_pairs]

    development_loader = _pair_loader(
        development_pair_records,
        render_config=render_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    consumed_loader = _pair_loader(
        consumed_records,
        render_config=render_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    unseen_loader = _pair_loader(
        unseen_records,
        render_config=render_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )

    started = time.perf_counter()
    training_peak = float(checkpoint.get("peak_allocated_vram_gib", 0.0))
    base_language = evaluate_canonical_language(
        base_model,
        audit_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=0.0,
    )
    language = evaluate_canonical_language(
        model,
        audit_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=training_peak,
    )
    base_development_pairs = evaluate_counterfactual_pairs(
        base_model,
        development_loader,
        device=device,
        precision=args.precision,
    )
    development_pairs = evaluate_counterfactual_pairs(
        model,
        development_loader,
        device=device,
        precision=args.precision,
    )
    consumed_pairs = evaluate_counterfactual_pairs(
        model,
        consumed_loader,
        device=device,
        precision=args.precision,
    )
    unseen_pairs = evaluate_counterfactual_pairs(
        model,
        unseen_loader,
        device=device,
        precision=args.precision,
    )
    boundary_clean = canonical_glyph_binding_v44_boundary_is_clean(model)
    boundary = canonical_glyph_binding_v44_boundary_receipt(model)
    peak = max(
        training_peak,
        float(language["peak_allocated_vram_gib"]),
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0,
    )
    gates = canonical_glyph_binding_v44_gate_report(
        base_language,
        language,
        base_development_pairs,
        development_pairs,
        consumed_pairs,
        unseen_pairs,
        boundary_clean=boundary_clean,
        base_state_exact=base_state_exact,
        adapter_parameters=int(boundary["adapter_parameters"]),
        peak_allocated_vram_gib=peak,
    )
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "experiment": "canonical-glyph-binding-v44-development",
        "claim_status": "passed" if all(gates.values()) else "rejected-or-partial",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_checkpoint_sha256": base_checkpoint_sha256,
        "base_state_sha256": expected_base_state_sha256,
        "base_state_exact": base_state_exact,
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
        },
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "statistics": visual_character_statistics_receipt(statistics),
        "matched_base_language": base_language,
        "language": language,
        "matched_base_development_pairs": base_development_pairs,
        "development_pairs": development_pairs,
        "consumed_training_pairs": consumed_pairs,
        "unseen_training_pairs": unseen_pairs,
        "pair_receipts": {
            "development_count": len(development_pair_records),
            "consumed": pair_sequence_receipt(consumed_records),
            "unseen": pair_sequence_receipt(unseen_records),
        },
        "gates": gates,
        "boundary_clean": boundary_clean,
        "model_boundary": boundary,
        "training_peak_allocated_vram_gib": training_peak,
        "peak_allocated_vram_gib": peak,
        "elapsed_seconds": elapsed,
        "writer_opened": False,
        "frozen_partition_opened": False,
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "claim_status": report["claim_status"],
            "gates": gates,
            "language": language,
            "development_pairs": development_pairs,
            "unseen_training_pairs": unseen_pairs,
            "checkpoint_sha256": report["checkpoint_sha256"],
        },
        output / "development_summary.json",
    )
    print(json.dumps(gates, sort_keys=True), flush=True)
    print(json.dumps(language, sort_keys=True), flush=True)
    print(json.dumps(development_pairs, sort_keys=True), flush=True)
    print(json.dumps(unseen_pairs, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
