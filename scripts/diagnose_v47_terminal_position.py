#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from ilm.visual_lm.codec_spherical_glyph_language_v47 import V47_ARCHITECTURE
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    finite_metric_tree,
)
from ilm.visual_lm.predictive_state_diagnostic import (
    audit_window_digest,
    build_partition_audit_windows,
    evaluate_context_length_curve,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
)
from ilm.visual_lm.visual_cell_eval_data import build_visual_character_statistics
from scripts.diagnose_v42_v47_predictive_state import (
    DEFAULT_MANIFEST,
    DEFAULT_V42_CHECKPOINT,
    DEFAULT_V47_CHECKPOINT,
    DEVELOPMENT_WINDOW_SEED,
    _checkpoint_receipt,
    _instantiate_v42,
    _instantiate_v47,
    _load_checkpoint,
    _resolve_device,
)
from scripts.train_canonical_glyph_language_v42 import _atomic_json
from scripts.train_visual_state_actuator import file_sha256, seed_everything


DEFAULT_OUTPUT = (
    "artifacts/v42_v47_predictive_state_diagnostic_20260815/terminal_position"
)
EXPECTED_WINDOW_SHA256 = (
    "e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630"
)
EXPECTED_BANK_SHA256 = (
    "a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad"
)
TERMINAL_LENGTHS = tuple(range(56, 65))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory V42/V47 terminal-position intervention."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v42-checkpoint", default=DEFAULT_V42_CHECKPOINT)
    parser.add_argument("--v47-checkpoint", default=DEFAULT_V47_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _bank_digest(characters: tuple[str, ...]) -> str:
    import hashlib

    return hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()


def _evaluate(
    model: Any,
    dataset: CanonicalGlyphAuditDataset,
    bank_images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, dict[str, float]]:
    model.to(device).eval()
    bank_fields = model.field.encode_unit(bank_images.to(device))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=canonical_glyph_audit_collate,
    )
    return evaluate_context_length_curve(
        model,
        loader,
        bank_fields,
        lengths=TERMINAL_LENGTHS,
        device=device,
        precision=precision,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("terminal-position batch settings are invalid")
    started = time.perf_counter()
    seed_everything(DEVELOPMENT_WINDOW_SEED)
    device = _resolve_device(args.device)

    v42_checkpoint = _load_checkpoint(args.v42_checkpoint, V42_ARCHITECTURE)
    v47_checkpoint = _load_checkpoint(args.v47_checkpoint, V47_ARCHITECTURE)
    v42_receipt = _checkpoint_receipt(args.v42_checkpoint, v42_checkpoint)
    v47_receipt = _checkpoint_receipt(args.v47_checkpoint, v47_checkpoint)
    manifest = verify_v25_manifest(args.manifest, strict=True)
    if not (
        manifest["sha256"]
        == v42_receipt["manifest_sha256"]
        == v47_receipt["manifest_sha256"]
    ):
        raise ValueError("terminal-position manifests do not match")
    render = CanonicalGlyphRenderConfig(**v42_checkpoint["render_config"])
    if render != CanonicalGlyphRenderConfig(**v47_checkpoint["render_config"]):
        raise ValueError("terminal-position renderers do not match")
    records = load_v25_records(args.manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(
        records,
        bank_size=1_024,
        script_views_mode=render.script_views,
    )
    if _bank_digest(statistics.characters) != EXPECTED_BANK_SHA256:
        raise ValueError("terminal-position evaluator bank changed")
    windows = build_partition_audit_windows(
        records,
        statistics,
        split="development",
        count=2_048,
        seed=DEVELOPMENT_WINDOW_SEED,
        script_views_mode=render.script_views,
    )
    window_sha256 = audit_window_digest(windows)
    if window_sha256 != EXPECTED_WINDOW_SHA256:
        raise ValueError("terminal-position development windows changed")
    dataset = CanonicalGlyphAuditDataset(
        windows,
        statistics,
        render_config=render,
    )
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=render,
    )

    v42_model, _v42_boundary, v42_clean = _instantiate_v42(v42_checkpoint)
    v42_curve = _evaluate(
        v42_model,
        dataset,
        bank_images,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    v42_gradients_absent = not any(
        parameter.grad is not None for parameter in v42_model.parameters()
    )
    del v42_model, v42_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    v47_model, _v47_boundary, v47_clean = _instantiate_v47(v47_checkpoint)
    v47_curve = _evaluate(
        v47_model,
        dataset,
        bank_images,
        device=device,
        precision=args.precision,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    v47_gradients_absent = not any(
        parameter.grad is not None for parameter in v47_model.parameters()
    )
    del v47_model, v47_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report: dict[str, Any] = {
        "experiment": "v47-terminal-position-exploratory-intervention",
        "status": "post-result-exploratory-no-gate-changes",
        "effective_arguments": vars(args),
        "lengths": list(TERMINAL_LENGTHS),
        "manifest_sha256": manifest["sha256"],
        "window_sha256": window_sha256,
        "bank_sha256": EXPECTED_BANK_SHA256,
        "windows": len(windows),
        "checkpoints": {"v42": v42_receipt, "v47": v47_receipt},
        "curves": {"v42": v42_curve, "v47": v47_curve},
        "integrity": {
            "exact_registered_development_windows": True,
            "exact_registered_evaluator_bank": True,
            "frozen_partition_remained_closed": True,
            "evaluator_metadata_excluded_from_student": True,
            "boundaries_clean": bool(v42_clean and v47_clean),
            "gradients_absent": bool(
                v42_gradients_absent and v47_gradients_absent
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": file_sha256(__file__),
    }
    report["integrity"]["all_finite"] = finite_metric_tree(report)
    report["integrity_clean"] = all(report["integrity"].values())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "terminal_position_report.json")
    if not report["integrity_clean"]:
        raise RuntimeError("terminal-position intervention failed integrity")
    print(
        json.dumps(
            {
                "report": str(output / "terminal_position_report.json"),
                "elapsed_seconds": report["elapsed_seconds"],
                "v42_top1_63": v42_curve["63"]["top1"],
                "v42_top1_64": v42_curve["64"]["top1"],
                "v47_top1_63": v47_curve["63"]["top1"],
                "v47_top1_64": v47_curve["64"]["top1"],
                "integrity_clean": report["integrity_clean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
