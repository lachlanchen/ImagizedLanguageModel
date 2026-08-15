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

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    V42_ARCHITECTURE,
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
    render_canonical_character_bank,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import (
    canonical_language_boundary_is_clean,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47 import (
    V47_ARCHITECTURE,
    CodecSphericalGlyphLanguageModelV47,
    codec_spherical_glyph_language_v47_boundary_receipt,
    codec_spherical_glyph_language_v47_config_from_payload,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    codec_spherical_language_v47_boundary_is_clean,
    finite_metric_tree,
)
from ilm.visual_lm.continuous_glyph_codec import (
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
)
from ilm.visual_lm.predictive_state_diagnostic import (
    audit_window_digest,
    build_partition_audit_windows,
    evaluate_predictive_state,
    field_geometry,
    partition_generalization_gaps,
)
from ilm.visual_lm.visual_cell_data import (
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_character_statistics,
)
from scripts.train_canonical_glyph_language_v42 import _atomic_json
from scripts.train_visual_state_actuator import (
    choose_device,
    file_sha256,
    seed_everything,
)


PROTOCOL = "references/v42_v47_predictive_state_diagnostic_plan.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_V42_CHECKPOINT = (
    "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
)
DEFAULT_V47_CHECKPOINT = (
    "artifacts/codec_spherical_glyph_language_v47_20260815/checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/v42_v47_predictive_state_diagnostic_20260815"
TRAIN_WINDOW_SEED = 20264810
DEVELOPMENT_WINDOW_SEED = 20264811
SHUFFLE_SEED = 20264812


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the visual predictive states of V42 and V47."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v42-checkpoint", default=DEFAULT_V42_CHECKPOINT)
    parser.add_argument("--v47-checkpoint", default=DEFAULT_V47_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--windows", type=int, default=2_048)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
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


def _load_checkpoint(path: str, architecture: str) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("diagnostic checkpoint must be a mapping")
    if checkpoint.get("architecture") != architecture:
        raise ValueError(f"checkpoint {path} is not {architecture}")
    if checkpoint.get("smoke_only") or checkpoint.get("exploratory"):
        raise PermissionError("diagnostic requires production, non-exploratory checkpoints")
    return checkpoint


def _checkpoint_receipt(path: str, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": file_sha256(path),
        "architecture": checkpoint["architecture"],
        "update": int(checkpoint["update"]),
        "manifest_sha256": checkpoint["manifest"]["sha256"],
        "smoke_only": bool(checkpoint.get("smoke_only")),
        "exploratory": bool(checkpoint.get("exploratory")),
    }


def _instantiate_v42(
    checkpoint: Mapping[str, Any],
) -> tuple[CanonicalGlyphLanguageModel, dict[str, Any], bool]:
    model = CanonicalGlyphLanguageModel(
        canonical_glyph_language_config_from_payload(checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    receipt = canonical_glyph_language_boundary_receipt(model)
    return model, receipt, canonical_language_boundary_is_clean(model)


def _instantiate_v47(
    checkpoint: Mapping[str, Any],
) -> tuple[CodecSphericalGlyphLanguageModelV47, dict[str, Any], bool]:
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
    receipt = codec_spherical_glyph_language_v47_boundary_receipt(model)
    return model, receipt, codec_spherical_language_v47_boundary_is_clean(model)


def _loader(
    dataset: CanonicalGlyphAuditDataset,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=canonical_glyph_audit_collate,
    )


@torch.no_grad()
def _evaluate_model(
    model: Any,
    *,
    model_name: str,
    boundary_receipt: Mapping[str, Any],
    boundary_clean: bool,
    datasets: Mapping[str, CanonicalGlyphAuditDataset],
    bank_images: torch.Tensor,
    checkpoint_receipt: Mapping[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bank_fields = model.field.encode_unit(bank_images.to(device))
    geometry = field_geometry(bank_fields)
    partitions: dict[str, Any] = {}
    for partition_name, dataset in datasets.items():
        partitions[partition_name] = evaluate_predictive_state(
            model,
            _loader(dataset, args=args, device=device),
            bank_fields,
            device=device,
            precision=args.precision,
            shuffle_seed=SHUFFLE_SEED,
        )
    gradients_present = any(
        parameter.grad is not None for parameter in model.parameters()
    )
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    return {
        "model": model_name,
        "checkpoint": dict(checkpoint_receipt),
        "boundary": dict(boundary_receipt),
        "boundary_clean": bool(boundary_clean),
        "field_geometry": geometry,
        "partitions": partitions,
        "gradients_present_after_diagnostic": gradients_present,
        "peak_allocated_vram_gib": peak_vram,
    }


def _bank_digest(characters: tuple[str, ...]) -> str:
    import hashlib

    return hashlib.sha256("".join(characters).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    if min(args.windows, args.bank_size, args.batch_size) < 1:
        raise ValueError("diagnostic sizes must be positive")
    if args.num_workers < 0:
        raise ValueError("diagnostic worker count cannot be negative")
    started = time.perf_counter()
    seed_everything(TRAIN_WINDOW_SEED)
    device = _resolve_device(args.device)

    v42_checkpoint = _load_checkpoint(args.v42_checkpoint, V42_ARCHITECTURE)
    v47_checkpoint = _load_checkpoint(args.v47_checkpoint, V47_ARCHITECTURE)
    v42_receipt = _checkpoint_receipt(args.v42_checkpoint, v42_checkpoint)
    v47_receipt = _checkpoint_receipt(args.v47_checkpoint, v47_checkpoint)
    if v42_receipt["manifest_sha256"] != v47_receipt["manifest_sha256"]:
        raise ValueError("V42 and V47 were not trained on the same manifest")
    v42_render = CanonicalGlyphRenderConfig(**v42_checkpoint["render_config"])
    v47_render = CanonicalGlyphRenderConfig(**v47_checkpoint["render_config"])
    if v42_render != v47_render:
        raise ValueError("V42 and V47 do not share the same canonical renderer")

    manifest_receipt = verify_v25_manifest(args.manifest, strict=True)
    if manifest_receipt["sha256"] != v42_receipt["manifest_sha256"]:
        raise ValueError("diagnostic manifest differs from both checkpoints")
    records = load_v25_records(args.manifest, strict_manifest=True)
    statistics = build_visual_character_statistics(
        records,
        bank_size=args.bank_size,
        script_views_mode=v42_render.script_views,
    )
    train_windows = build_partition_audit_windows(
        records,
        statistics,
        split="train",
        count=args.windows,
        seed=TRAIN_WINDOW_SEED,
        script_views_mode=v42_render.script_views,
    )
    development_windows = build_partition_audit_windows(
        records,
        statistics,
        split="development",
        count=args.windows,
        seed=DEVELOPMENT_WINDOW_SEED,
        script_views_mode=v42_render.script_views,
    )
    datasets = {
        "train_partition": CanonicalGlyphAuditDataset(
            train_windows,
            statistics,
            render_config=v42_render,
        ),
        "development_partition": CanonicalGlyphAuditDataset(
            development_windows,
            statistics,
            render_config=v42_render,
        ),
    }
    bank_images = render_canonical_character_bank(
        statistics,
        render_config=v42_render,
    )
    if tuple(bank_images.shape) != (args.bank_size, 1, 32, 32):
        raise ValueError("diagnostic evaluator bank has the wrong raster shape")
    if not bank_images.is_floating_point() or not bool(torch.isfinite(bank_images).all()):
        raise ValueError("diagnostic evaluator bank is not finite floating imagery")

    models: dict[str, Any] = {}
    v42_model, v42_boundary, v42_clean = _instantiate_v42(v42_checkpoint)
    models["v42"] = _evaluate_model(
        v42_model,
        model_name="V42 direct canonical field",
        boundary_receipt=v42_boundary,
        boundary_clean=v42_clean,
        datasets=datasets,
        bank_images=bank_images,
        checkpoint_receipt=v42_receipt,
        args=args,
        device=device,
    )
    del v42_model, v42_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    v47_model, v47_boundary, v47_clean = _instantiate_v47(v47_checkpoint)
    models["v47"] = _evaluate_model(
        v47_model,
        model_name="V47 codec-spherical field",
        boundary_receipt=v47_boundary,
        boundary_clean=v47_clean,
        datasets=datasets,
        bank_images=bank_images,
        checkpoint_receipt=v47_receipt,
        args=args,
        device=device,
    )
    del v47_model, v47_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gaps = partition_generalization_gaps(models)
    elapsed = time.perf_counter() - started
    report: dict[str, Any] = {
        "experiment": "v42-v47-predictive-state-post-result-diagnostic",
        "protocol": {
            "path": PROTOCOL,
            "sha256": file_sha256(PROTOCOL),
            "status": "post-result-diagnostic-no-gate-changes",
        },
        "effective_arguments": vars(args),
        "seeds": {
            "train_windows": TRAIN_WINDOW_SEED,
            "development_windows": DEVELOPMENT_WINDOW_SEED,
            "shuffle": SHUFFLE_SEED,
        },
        "manifest": manifest_receipt,
        "partition": visual_cell_partition_receipt(records),
        "renderer": {
            "cell_size": v42_render.cell_size,
            "font_size": v42_render.font_size,
            "font_path": v42_render.font_path,
            "script_views": v42_render.script_views,
        },
        "evaluator_bank": {
            "size": len(statistics.characters),
            "ordered_character_sha256": _bank_digest(statistics.characters),
            "built_from_training_partition": True,
            "student_receives_bank": False,
            "student_receives_labels": False,
            "uses_bank_only_after_model_prediction": True,
        },
        "windows": {
            "train_partition": {
                "count": len(train_windows),
                "sha256": audit_window_digest(train_windows),
                "split": "train",
            },
            "development_partition": {
                "count": len(development_windows),
                "sha256": audit_window_digest(development_windows),
                "split": "development",
            },
            "frozen_partition_opened": False,
        },
        "models": models,
        "development_minus_train": gaps,
        "integrity": {
            "checkpoints_load_strictly": True,
            "manifest_matches_both_checkpoints": True,
            "production_checkpoints_only": True,
            "renderer_identical": True,
            "windows_complete": (
                len(train_windows) == args.windows
                and len(development_windows) == args.windows
            ),
            "student_inputs_are_float_images": True,
            "evaluator_metadata_excluded_from_student": True,
            "frozen_partition_remained_closed": True,
            "gradients_absent": not any(
                result["gradients_present_after_diagnostic"]
                for result in models.values()
            ),
            "boundaries_clean": all(
                result["boundary_clean"] for result in models.values()
            ),
        },
        "elapsed_seconds": elapsed,
    }
    report["integrity"]["all_finite"] = finite_metric_tree(report)
    report["integrity_clean"] = all(report["integrity"].values())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(report, output / "diagnostic_report.json")
    if not report["integrity_clean"]:
        failed = sorted(
            key for key, passed in report["integrity"].items() if not passed
        )
        raise RuntimeError(
            "predictive-state diagnostic failed integrity conditions: "
            + ", ".join(failed)
        )
    print(json.dumps({
        "report": str(output / "diagnostic_report.json"),
        "elapsed_seconds": elapsed,
        "v42_development_top1_64": models["v42"]["partitions"]["development_partition"]["context_curve"]["64"]["top1"],
        "v47_development_top1_64": models["v47"]["partitions"]["development_partition"]["context_curve"]["64"]["top1"],
        "integrity_clean": report["integrity_clean"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
