#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphPairAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    canonical_glyph_pair_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.scaled_retinal_glyph_language_v46 import (
    V46_ARCHITECTURE,
    V46_PROTOCOL,
    V46_REFERENCE_RADIUS,
    V46_REQUIRED_V45_FIELD_STATE_SHA256,
    ScaledRetinalGlyphLanguageModelV46,
    scaled_retinal_glyph_language_v46_boundary_receipt,
    scaled_retinal_glyph_language_v46_config_from_payload,
)
from ilm.visual_lm.scaled_retinal_glyph_language_v46_evaluation import (
    V46_AUDIT_SEED,
    V46_REQUIRED_TRAINABLE_PARAMETERS,
    evaluate_scaled_retinal_counterfactual_pairs_v46,
    evaluate_scaled_retinal_generated_fields_v46,
    evaluate_scaled_retinal_language_v46,
    finite_metric_tree,
    scaled_retinal_field_roundtrip_receipt,
    scaled_retinal_language_v46_boundary_is_clean,
    scaled_retinal_language_v46_gate_report,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    pack_visual_cells,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
    visual_character_statistics_receipt,
)
from scripts.train_canonical_glyph_language_v42 import _atomic_json
from scripts.train_scaled_retinal_glyph_language_v46 import (
    DEFAULT_FIELD,
    DEFAULT_MANIFEST,
    FIXED_OPTIMIZATION,
    SOURCE_FILES,
    load_verified_v45_field,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = (
    "artifacts/scaled_retinal_glyph_language_v46_20260815/checkpoint_final.pt"
)
DEFAULT_OUTPUT = (
    "artifacts/scaled_retinal_glyph_language_v46_20260815/development"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen V46 scaled-retinal language core."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--field", default=DEFAULT_FIELD)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--pairs", type=int, default=512)
    parser.add_argument("--generated-examples", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def _save_sample_sheet(
    model: ScaledRetinalGlyphLanguageModelV46,
    dataset: CanonicalGlyphAuditDataset,
    path: Path,
    *,
    device: torch.device,
    precision: str,
    samples: int,
    noise_scale: float,
) -> None:
    count = min(16, len(dataset))
    context = torch.stack([dataset[index]["context"] for index in range(count)]).to(
        device
    )
    target = torch.stack(
        [dataset[index]["continuation"][0] for index in range(count)]
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(V46_AUDIT_SEED + 77)
    with autocast_context(device, precision):
        generated, _ = model.sample_next(
            context,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
    interleaved = torch.stack((target, generated), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(interleaved.cpu(), columns=8, gutter=2).save(path)


def _source_receipt_matches(checkpoint: Mapping[str, Any]) -> bool:
    stored = checkpoint.get("protocol", {}).get("source_files_sha256", {})
    if not isinstance(stored, Mapping):
        return False
    current = {
        path: file_sha256(path)
        for path in SOURCE_FILES
        if Path(path).exists()
    }
    return dict(stored) == current


def _fixed_evaluation_arguments(args: argparse.Namespace) -> bool:
    return (
        args.precision == "bf16"
        and args.windows == 2_048
        and args.bank_size == 1_024
        and args.pairs == 512
        and args.generated_examples == 256
        and args.samples == 4
        and abs(args.noise_scale - 1.0) <= 1e-12
    )


def _fixed_training_arguments(checkpoint: Mapping[str, Any]) -> bool:
    effective = checkpoint.get("protocol", {}).get("effective_arguments", {})
    if not isinstance(effective, Mapping):
        return False
    return all(effective.get(key) == value for key, value in FIXED_OPTIMIZATION.items())


def main() -> None:
    args = parse_args()
    if min(
        args.batch_size,
        args.windows,
        args.bank_size,
        args.pairs,
        args.generated_examples,
        args.samples,
    ) < 1:
        raise ValueError("V46 audit sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    if device.type == "cuda":
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
    seed_everything(V46_AUDIT_SEED)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture") != V46_ARCHITECTURE:
        raise ValueError("audit checkpoint is not V46")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    retinal_field, retinal_field_receipt = load_verified_v45_field(args.field)
    if checkpoint.get("retinal_field") != retinal_field_receipt:
        raise ValueError("V46 audit field differs from its training field")
    model = ScaledRetinalGlyphLanguageModelV46(
        scaled_retinal_glyph_language_v46_config_from_payload(
            checkpoint["model_config"]
        ),
        retinal_field,
        v45_checkpoint_sha256=retinal_field_receipt["checkpoint_sha256"],
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("exploratory") and not checkpoint.get(
        "smoke_only"
    )
    manifest_receipt = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V46 audit corpus differs from its training corpus")
    records = load_v25_records(args.manifest, strict_manifest=strict_manifest)
    partition = visual_cell_partition_receipt(records)
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
        seed=V46_AUDIT_SEED,
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
    pair_records = build_factorized_suffix_pairs(
        records,
        split="development",
        suffix_cells=4,
        count=args.pairs,
        seed=V46_AUDIT_SEED + 1,
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
        script_views_mode=render_config.script_views,
    )
    pair_dataset = CanonicalGlyphPairAuditDataset(
        pair_records,
        render_config=render_config,
    )
    pair_loader = DataLoader(
        pair_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=canonical_glyph_pair_audit_collate,
    )
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )

    started = time.perf_counter()
    language = evaluate_scaled_retinal_language_v46(
        model,
        audit_loader,
        statistics,
        bank_images,
        device=device,
        precision=args.precision,
        checkpoint_peak_vram_gib=float(
            checkpoint.get("peak_allocated_vram_gib", 0.0)
        ),
    )
    pairs = evaluate_scaled_retinal_counterfactual_pairs_v46(
        model,
        pair_loader,
        device=device,
        precision=args.precision,
    )
    generated = evaluate_scaled_retinal_generated_fields_v46(
        model,
        audit_loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
        samples=args.samples,
        noise_scale=args.noise_scale,
    )
    roundtrip = scaled_retinal_field_roundtrip_receipt(
        model,
        [bank_images],
    )
    audit_elapsed = time.perf_counter() - started
    training_elapsed = float(checkpoint.get("training_elapsed_seconds", 0.0))
    total_elapsed = training_elapsed + audit_elapsed
    model_boundary = scaled_retinal_glyph_language_v46_boundary_receipt(model)
    boundary_clean = scaled_retinal_language_v46_boundary_is_clean(model)
    preliminary = {
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
        "roundtrip": roundtrip,
    }
    protocol_integrity = {
        "production_checkpoint": (
            checkpoint.get("update") == FIXED_OPTIMIZATION["steps"]
            and not checkpoint.get("smoke_only")
            and not checkpoint.get("exploratory")
        ),
        "fixed_training_arguments": _fixed_training_arguments(checkpoint),
        "fixed_evaluation_arguments": _fixed_evaluation_arguments(args),
        "protocol_document_matches": (
            checkpoint.get("protocol", {}).get("document") == V46_PROTOCOL
            and checkpoint.get("protocol", {}).get("sha256")
            == file_sha256(V46_PROTOCOL)
        ),
        "source_files_match": _source_receipt_matches(checkpoint),
        "retinal_field_receipt_matches": (
            checkpoint.get("retinal_field") == retinal_field_receipt
            and retinal_field_receipt["field_state_sha256"]
            == V46_REQUIRED_V45_FIELD_STATE_SHA256
            and model_boundary["v45_field_state_sha256"]
            == V46_REQUIRED_V45_FIELD_STATE_SHA256
            and abs(
                retinal_field_receipt["reference_radius"] - V46_REFERENCE_RADIUS
            )
            <= 1e-12
            and abs(model_boundary["reference_radius"] - V46_REFERENCE_RADIUS)
            <= 1e-12
        ),
        "parameter_count_matches": (
            model_boundary["trainable_parameters"]
            == V46_REQUIRED_TRAINABLE_PARAMETERS
        ),
        "roundtrip_exact": (
            roundtrip["all_finite"] is True
            and roundtrip["maximum_dct_absolute_error"] < 2e-8
            and roundtrip["binary_pixel_accuracy"] == 1.0
        ),
        "metrics_finite": finite_metric_tree(preliminary),
        "frozen_partition_opened": bool(
            partition.get("frozen_images_instantiated", False)
        ),
        "total_elapsed_below_30_minutes": total_elapsed < 30.0 * 60.0,
    }
    protocol_integrity_clean = (
        all(
            value
            for key, value in protocol_integrity.items()
            if key != "frozen_partition_opened"
        )
        and protocol_integrity["frozen_partition_opened"] is False
    )
    gates = scaled_retinal_language_v46_gate_report(
        language,
        pairs,
        generated,
        boundary_clean=boundary_clean,
        protocol_integrity_clean=protocol_integrity_clean,
    )
    report = {
        "experiment": "scaled-retinal-glyph-language-v46-development-audit",
        "architecture": V46_ARCHITECTURE,
        "claim_status": (
            "qualified-scaled-retinal-causal-reader"
            if all(gates.values())
            else "non-qualifying-development-result"
        ),
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_update": checkpoint["update"],
        "checkpoint_smoke_only": bool(checkpoint.get("smoke_only")),
        "checkpoint_exploratory": bool(checkpoint.get("exploratory")),
        "manifest": manifest_receipt,
        "partition": partition,
        "statistics": visual_character_statistics_receipt(statistics),
        "retinal_field": retinal_field_receipt,
        "data_boundary": checkpoint["data_boundary"],
        "model_boundary": model_boundary,
        "boundary_clean": boundary_clean,
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
        "roundtrip": roundtrip,
        "protocol_integrity": protocol_integrity,
        "protocol_integrity_clean": protocol_integrity_clean,
        "gates": gates,
        "gates_passed": sum(gates.values()),
        "gates_total": len(gates),
        "all_gates_pass": all(gates.values()),
        "frozen_partition_opened": protocol_integrity[
            "frozen_partition_opened"
        ],
        "training_elapsed_seconds": training_elapsed,
        "audit_elapsed_seconds": audit_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "effective_arguments": vars(args),
    }
    _atomic_json(report, output / "development_report.json")
    _atomic_json(
        {
            "claim_status": report["claim_status"],
            "all_gates_pass": report["all_gates_pass"],
            "gates_passed": report["gates_passed"],
            "gates_total": report["gates_total"],
            "checkpoint_update": checkpoint["update"],
            "full_top1": language["full_top1"],
            "full_target_log_probability": language[
                "full_target_log_probability"
            ],
            "full_minus_shuffled_top1": (
                language["full_top1"] - language["shuffled_top1"]
            ),
            "counterfactual_arm_accuracy": pairs["full_arm_accuracy"],
            "generated_identity_top1": generated["generated_identity_top1"],
            "generated_pixel_f1": generated["generated_pixel_f1"],
            "peak_allocated_vram_gib": language["peak_allocated_vram_gib"],
            "total_elapsed_seconds": total_elapsed,
        },
        output / "development_summary.json",
    )
    _save_sample_sheet(
        model,
        audit_dataset,
        output / "target_generated_pairs.png",
        device=device,
        precision=args.precision,
        samples=args.samples,
        noise_scale=args.noise_scale,
    )
    print(json.dumps(gates, sort_keys=True), flush=True)
    print(json.dumps(language, sort_keys=True), flush=True)
    print(json.dumps(generated, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
