#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    V42_ARCHITECTURE,
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    finite_metric_tree,
)
from ilm.visual_lm.predictive_state_diagnostic import (
    DIRECT_ACTUATOR_THRESHOLDS,
    audit_window_digest,
    build_partition_audit_windows,
    collect_direct_actuator_predictions,
    evaluate_direct_actuator_predictions,
    select_direct_actuator_threshold,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
)
from ilm.visual_lm.visual_cell_eval_data import build_visual_character_statistics
from scripts.diagnose_v42_v47_predictive_state import (
    DEFAULT_MANIFEST,
    DEFAULT_V42_CHECKPOINT,
    DEVELOPMENT_WINDOW_SEED,
    TRAIN_WINDOW_SEED,
    _checkpoint_receipt,
    _instantiate_v42,
    _load_checkpoint,
    _resolve_device,
)
from scripts.train_canonical_glyph_language_v42 import _atomic_json
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_OUTPUT = (
    "artifacts/v42_v47_predictive_state_diagnostic_20260815/direct_actuator"
)
EXPECTED_TRAIN_WINDOW_SHA256 = (
    "42fc41a00a90113dfc1df44f1f435d84db61dfef5fdfc1c4d3f0e6d966958b38"
)
EXPECTED_DEVELOPMENT_WINDOW_SHA256 = (
    "e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630"
)
EXPECTED_BANK_SHA256 = (
    "a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test V42's direct image-field actuator without its sampler."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", default=DEFAULT_V42_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _bank_digest(characters: tuple[str, ...]) -> str:
    return hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()


def _loader(
    dataset: CanonicalGlyphAuditDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=canonical_glyph_audit_collate,
    )


def _parameter_receipt(model: torch.nn.Module) -> dict[str, int]:
    all_parameters = sum(parameter.numel() for parameter in model.parameters())
    generator_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("generator.")
    )
    return {
        "v42_total": all_parameters,
        "v42_generator": generator_parameters,
        "direct_reader_and_actuator": all_parameters - generator_parameters,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("direct-actuator batch settings are invalid")
    started = time.perf_counter()
    seed_everything(TRAIN_WINDOW_SEED)
    device = _resolve_device(args.device)

    checkpoint = _load_checkpoint(args.checkpoint, V42_ARCHITECTURE)
    checkpoint_receipt = _checkpoint_receipt(args.checkpoint, checkpoint)
    manifest = verify_v25_manifest(args.manifest, strict=True)
    if manifest["sha256"] != checkpoint_receipt["manifest_sha256"]:
        raise ValueError("direct-actuator corpus differs from V42 training")
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    records = load_v25_records(args.manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(
        records,
        bank_size=1_024,
        script_views_mode=render.script_views,
    )
    if _bank_digest(statistics.characters) != EXPECTED_BANK_SHA256:
        raise ValueError("direct-actuator evaluator bank changed")

    train_windows = build_partition_audit_windows(
        records,
        statistics,
        split="train",
        count=2_048,
        seed=TRAIN_WINDOW_SEED,
        script_views_mode=render.script_views,
    )
    development_windows = build_partition_audit_windows(
        records,
        statistics,
        split="development",
        count=2_048,
        seed=DEVELOPMENT_WINDOW_SEED,
        script_views_mode=render.script_views,
    )
    train_digest = audit_window_digest(train_windows)
    development_digest = audit_window_digest(development_windows)
    if train_digest != EXPECTED_TRAIN_WINDOW_SHA256:
        raise ValueError("direct-actuator training windows changed")
    if development_digest != EXPECTED_DEVELOPMENT_WINDOW_SHA256:
        raise ValueError("direct-actuator development windows changed")

    model, boundary, boundary_clean = _instantiate_v42(checkpoint)
    parameter_receipt = _parameter_receipt(model)
    model.to(device).eval()
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render,
    ).to(device)
    bank_fields = model.field.encode_unit(bank_images)

    train_dataset = CanonicalGlyphAuditDataset(
        train_windows,
        statistics,
        render_config=render,
    )
    train_predictions = collect_direct_actuator_predictions(
        model,
        _loader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        ),
        device=device,
        precision=args.precision,
    )
    selected = select_direct_actuator_threshold(
        train_predictions["signed_images"],
        train_predictions["target_pixels"],
        thresholds=DIRECT_ACTUATOR_THRESHOLDS,
    )
    train_zero = evaluate_direct_actuator_predictions(
        model,
        train_predictions,
        bank_fields,
        threshold=0.0,
    )
    train_calibrated = evaluate_direct_actuator_predictions(
        model,
        train_predictions,
        bank_fields,
        threshold=selected["threshold"],
    )
    del train_predictions, train_dataset

    development_dataset = CanonicalGlyphAuditDataset(
        development_windows,
        statistics,
        render_config=render,
    )
    development_predictions = collect_direct_actuator_predictions(
        model,
        _loader(
            development_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        ),
        device=device,
        precision=args.precision,
    )
    development_zero = evaluate_direct_actuator_predictions(
        model,
        development_predictions,
        bank_fields,
        threshold=0.0,
    )
    development_calibrated = evaluate_direct_actuator_predictions(
        model,
        development_predictions,
        bank_fields,
        threshold=selected["threshold"],
    )
    gradients_absent = not any(
        parameter.grad is not None for parameter in model.parameters()
    )

    report: dict[str, Any] = {
        "experiment": "v42-direct-field-actuator-post-result-intervention",
        "status": "post-result-exploratory-no-gate-changes",
        "effective_arguments": vars(args),
        "checkpoint": checkpoint_receipt,
        "manifest_sha256": manifest["sha256"],
        "bank_sha256": EXPECTED_BANK_SHA256,
        "windows": {
            "train": {
                "count": len(train_windows),
                "sha256": train_digest,
            },
            "development": {
                "count": len(development_windows),
                "sha256": development_digest,
            },
        },
        "threshold_selection": {
            "partition": "train",
            "candidate_thresholds": list(DIRECT_ACTUATOR_THRESHOLDS),
            **selected,
        },
        "metrics": {
            "train_threshold_zero": train_zero,
            "train_calibrated": train_calibrated,
            "development_threshold_zero": development_zero,
            "development_calibrated": development_calibrated,
        },
        "parameters": parameter_receipt,
        "model_boundary": boundary,
        "integrity": {
            "production_checkpoint_only": True,
            "strict_checkpoint_load": True,
            "exact_registered_train_windows": True,
            "exact_registered_development_windows": True,
            "exact_registered_evaluator_bank": True,
            "threshold_selected_on_train_only": True,
            "development_excluded_from_selection": True,
            "learned_generator_not_invoked": True,
            "student_inputs_are_float_images": True,
            "evaluator_metadata_excluded_from_student": True,
            "frozen_partition_remained_closed": True,
            "boundary_clean": bool(boundary_clean),
            "gradients_absent": gradients_absent,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": file_sha256(__file__),
    }
    report["integrity"]["all_finite"] = finite_metric_tree(report)
    report["integrity_clean"] = all(report["integrity"].values())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "direct_actuator_report.json")
    if not report["integrity_clean"]:
        raise RuntimeError("direct-actuator intervention failed integrity")
    print(
        json.dumps(
            {
                "report": str(output / "direct_actuator_report.json"),
                "selected_threshold": selected["threshold"],
                "development_zero_f1": development_zero["visible_pixel_f1"],
                "development_zero_top1": development_zero[
                    "visible_identity_top1"
                ],
                "development_calibrated_f1": development_calibrated[
                    "visible_pixel_f1"
                ],
                "development_calibrated_top1": development_calibrated[
                    "visible_identity_top1"
                ],
                "parameters_without_generator": parameter_receipt[
                    "direct_reader_and_actuator"
                ],
                "integrity_clean": report["integrity_clean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
