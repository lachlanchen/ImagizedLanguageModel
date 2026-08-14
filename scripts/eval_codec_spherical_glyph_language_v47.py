#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from ilm.visual_lm.codec_spherical_glyph_language_v47 import (
    V47_ARCHITECTURE,
    V47_PROTOCOL,
    V47_REQUIRED_CODEC_CHECKPOINT_SHA256,
    V47_REQUIRED_CODEC_STATE_SHA256,
    CodecSphericalGlyphLanguageModelV47,
    codec_spherical_glyph_language_v47_boundary_receipt,
    codec_spherical_glyph_language_v47_config_from_payload,
    tensor_state_sha256,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_data import (
    V47_PAIR_COUNT,
    V47_PAIR_SEQUENCE_SHA256,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    V47_AUDIT_SEED,
    codec_spherical_field_preflight_v47,
    codec_spherical_language_v47_boundary_is_clean,
    codec_spherical_language_v47_gate_report,
    evaluate_codec_spherical_counterfactual_pairs_v47,
    evaluate_codec_spherical_generated_fields_v47,
    evaluate_codec_spherical_language_v47,
    finite_metric_tree,
)
from ilm.visual_lm.continuous_glyph_codec import (
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
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
from scripts.train_codec_spherical_glyph_language_v47 import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT,
    FIXED_OPTIMIZATION,
    SOURCE_FILES,
    _render_held_font_banks,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


DEFAULT_CHECKPOINT = f"{DEFAULT_OUTPUT}/checkpoint_final.pt"
DEFAULT_AUDIT_OUTPUT = f"{DEFAULT_OUTPUT}/development"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen V47 codec-spherical language core."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_AUDIT_OUTPUT)
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


def _resolve_device(value: str) -> torch.device:
    device = choose_device(value)
    if device.type != "cuda":
        return device
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


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


@torch.no_grad()
def _save_sample_sheet(
    model: CodecSphericalGlyphLanguageModelV47,
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
    generator = torch.Generator(device=device).manual_seed(V47_AUDIT_SEED + 77)
    with autocast_context(device, precision):
        generated, trace = model.sample_next(
            context,
            samples=samples,
            generator=generator,
            noise_scale=noise_scale,
        )
    reread_render = model.field.binary(trace["selected_reread_fields"])
    interleaved = torch.stack((target, generated, reread_render), dim=1).flatten(0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    pack_visual_cells(interleaved.cpu(), columns=12, gutter=2).save(path)


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
        raise ValueError("V47 audit sizes must be positive")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(V47_AUDIT_SEED)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint.get("architecture") != V47_ARCHITECTURE:
        raise ValueError("audit checkpoint is not V47")
    if checkpoint.get("smoke_only") and not args.allow_smoke:
        raise PermissionError("smoke checkpoints require --allow-smoke")
    codec_receipt = checkpoint.get("codec")
    if not isinstance(codec_receipt, Mapping):
        raise ValueError("V47 checkpoint lacks its embedded codec receipt")
    codec = ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
    model = CodecSphericalGlyphLanguageModelV47(
        codec_spherical_glyph_language_v47_config_from_payload(
            checkpoint["model_config"]
        ),
        codec,
        codec_checkpoint_sha256=str(codec_receipt["checkpoint_sha256"]),
        codec_state_sha256=str(codec_receipt["ema_tensor_state_sha256"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    embedded_codec_state_sha256 = tensor_state_sha256(model.field.codec.state_dict())
    model.to(device).eval()
    render_config = CanonicalGlyphRenderConfig(**checkpoint["render_config"])

    strict_manifest = not checkpoint.get("exploratory") and not checkpoint.get(
        "smoke_only"
    )
    manifest_receipt = verify_v25_manifest(args.manifest, strict=strict_manifest)
    if manifest_receipt["sha256"] != checkpoint["manifest"]["sha256"]:
        raise ValueError("V47 audit corpus differs from its training corpus")
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
        seed=V47_AUDIT_SEED,
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
        seed=V47_AUDIT_SEED + 1,
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
    held_banks = _render_held_font_banks(statistics.characters, render_config)

    started = time.perf_counter()
    preflight = codec_spherical_field_preflight_v47(
        model,
        bank_images,
        held_banks,
        device=device,
    )
    language = evaluate_codec_spherical_language_v47(
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
    pairs = evaluate_codec_spherical_counterfactual_pairs_v47(
        model,
        pair_loader,
        device=device,
        precision=args.precision,
    )
    generated = evaluate_codec_spherical_generated_fields_v47(
        model,
        audit_loader,
        bank_images,
        device=device,
        precision=args.precision,
        maximum_examples=args.generated_examples,
        samples=args.samples,
        noise_scale=args.noise_scale,
    )
    audit_elapsed = time.perf_counter() - started
    training_elapsed = float(checkpoint.get("training_elapsed_seconds", 0.0))
    total_elapsed = training_elapsed + audit_elapsed
    model_boundary = codec_spherical_glyph_language_v47_boundary_receipt(model)
    boundary_clean = codec_spherical_language_v47_boundary_is_clean(model)
    preliminary = {
        "field_preflight": preflight,
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
    }
    pair_sequence = checkpoint.get("pair_sequence", {})
    protocol_integrity = {
        "production_checkpoint": (
            checkpoint.get("update") == FIXED_OPTIMIZATION["steps"]
            and not checkpoint.get("smoke_only")
            and not checkpoint.get("exploratory")
        ),
        "fixed_training_arguments": _fixed_training_arguments(checkpoint),
        "fixed_evaluation_arguments": _fixed_evaluation_arguments(args),
        "protocol_document_matches": (
            checkpoint.get("protocol", {}).get("document") == V47_PROTOCOL
            and checkpoint.get("protocol", {}).get("sha256")
            == file_sha256(V47_PROTOCOL)
        ),
        "source_files_match": _source_receipt_matches(checkpoint),
        "embedded_codec_matches": (
            codec_receipt.get("checkpoint_sha256")
            == V47_REQUIRED_CODEC_CHECKPOINT_SHA256
            and codec_receipt.get("ema_tensor_state_sha256")
            == V47_REQUIRED_CODEC_STATE_SHA256
            and embedded_codec_state_sha256 == V47_REQUIRED_CODEC_STATE_SHA256
            and model_boundary["codec_state_sha256"]
            == V47_REQUIRED_CODEC_STATE_SHA256
        ),
        "pair_sequence_matches": (
            isinstance(pair_sequence, Mapping)
            and pair_sequence.get("count") == V47_PAIR_COUNT
            and pair_sequence.get("unique_suffixes") == V47_PAIR_COUNT
            and pair_sequence.get("sha256") == V47_PAIR_SEQUENCE_SHA256
            and checkpoint.get("pair_rows_consumed") == V47_PAIR_COUNT
        ),
        "field_preflight_passes": bool(preflight["pass"]),
        "parameter_budget": (
            model_boundary["total_parameters"] < 32_000_000
            and model_boundary["trainable_parameters"] < 25_000_000
        ),
        "all_runtime_fields_finite": (
            language["anchor_finite_rate"] == 1.0
            and generated["proposal_finite_rate"] == 1.0
            and generated["reread_finite_rate"] == 1.0
        ),
        "metrics_finite": finite_metric_tree(preliminary),
        "frozen_partition_opened": bool(
            partition.get("frozen_images_instantiated", False)
        ),
        "total_elapsed_below_35_minutes": total_elapsed < 35.0 * 60.0,
    }
    protocol_integrity_clean = (
        all(
            value
            for key, value in protocol_integrity.items()
            if key != "frozen_partition_opened"
        )
        and protocol_integrity["frozen_partition_opened"] is False
    )
    gates = codec_spherical_language_v47_gate_report(
        preflight,
        language,
        pairs,
        generated,
        boundary_clean=boundary_clean,
        protocol_integrity_clean=protocol_integrity_clean,
        updates_complete=checkpoint.get("update") == FIXED_OPTIMIZATION["steps"],
        pair_rows_consumed=int(checkpoint.get("pair_rows_consumed", 0)),
        total_parameters=int(model_boundary["total_parameters"]),
        trainable_parameters=int(model_boundary["trainable_parameters"]),
        total_elapsed_seconds=total_elapsed,
    )
    report = {
        "experiment": "codec-spherical-glyph-language-v47-development-audit",
        "architecture": V47_ARCHITECTURE,
        "claim_status": (
            "qualified-codec-spherical-causal-reader"
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
        "codec": dict(codec_receipt),
        "embedded_codec_state_sha256": embedded_codec_state_sha256,
        "pair_sequence": pair_sequence,
        "data_boundary": checkpoint["data_boundary"],
        "model_boundary": model_boundary,
        "boundary_clean": boundary_clean,
        "field_preflight": preflight,
        "language": language,
        "counterfactual_pairs": pairs,
        "generated": generated,
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
            "selected_proposal_reread_cosine": generated[
                "mean_selected_proposal_to_visible_reread_cosine"
            ],
            "peak_allocated_vram_gib": language["peak_allocated_vram_gib"],
            "total_elapsed_seconds": total_elapsed,
        },
        output / "development_summary.json",
    )
    _save_sample_sheet(
        model,
        audit_dataset,
        output / "target_generated_reread_triplets.png",
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
