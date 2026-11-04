#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
    canonical_glyph_language_config_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.hierarchical_raster_language import (
    HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
    HierarchicalRasterLanguageConfig,
    HierarchicalRasterLanguageModel,
    hierarchical_raster_language_boundary_receipt,
    hierarchical_raster_language_config_from_payload,
    hierarchical_raster_language_config_payload,
)
from ilm.visual_lm.hierarchical_raster_language_training import (
    V50_LOSS_WEIGHTS,
    hierarchical_raster_language_loss,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from scripts.train_canonical_glyph_binding_v43 import (
    _append_jsonl,
    _atomic_json,
    _atomic_save,
    _resolve_device,
    _scheduled_lr,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/hierarchical_raster_language_v50_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_READER = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/hierarchical_raster_language_v50_20261104"
SOURCE_FILES = (
    "ilm/visual_lm/hierarchical_raster_language.py",
    "ilm/visual_lm/hierarchical_raster_language_training.py",
    "scripts/train_hierarchical_raster_language_v50.py",
)
FIXED_OPTIMIZATION = {
    "steps": 5_000,
    "reader_freeze_steps": 750,
    "batch_size": 8,
    "positions_per_stream": 8,
    "reader_learning_rate": 1e-5,
    "writer_learning_rate": 2e-4,
    "warmup_steps": 200,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "counterfactual_margin": 0.02,
    "seed": 20265010,
    "dataset_seed": 20265011,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the image-only hierarchical raster language model: a global "
            "visual reader and a local patch-autoregressive raster writer."
        )
    )
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--resume")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--reader-freeze-steps",
        type=int,
        default=FIXED_OPTIMIZATION["reader_freeze_steps"],
    )
    parser.add_argument(
        "--batch-size", type=int, default=FIXED_OPTIMIZATION["batch_size"]
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--positions-per-stream",
        type=int,
        default=FIXED_OPTIMIZATION["positions_per_stream"],
    )
    parser.add_argument(
        "--reader-learning-rate",
        type=float,
        default=FIXED_OPTIMIZATION["reader_learning_rate"],
    )
    parser.add_argument(
        "--writer-learning-rate",
        type=float,
        default=FIXED_OPTIMIZATION["writer_learning_rate"],
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=FIXED_OPTIMIZATION["warmup_steps"]
    )
    parser.add_argument(
        "--minimum-lr-ratio",
        type=float,
        default=FIXED_OPTIMIZATION["minimum_lr_ratio"],
    )
    parser.add_argument(
        "--weight-decay", type=float, default=FIXED_OPTIMIZATION["weight_decay"]
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=FIXED_OPTIMIZATION["gradient_clip"]
    )
    parser.add_argument(
        "--counterfactual-margin",
        type=float,
        default=FIXED_OPTIMIZATION["counterfactual_margin"],
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--local-dim", type=int, default=256)
    parser.add_argument("--local-layers", type=int, default=4)
    parser.add_argument("--local-heads", type=int, default=8)
    parser.add_argument("--mixtures", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if args.smoke:
        payload.update(
            {
                "steps": min(args.steps, 2),
                "reader_freeze_steps": min(args.reader_freeze_steps, 1),
                "batch_size": min(args.batch_size, 2),
                "positions_per_stream": min(args.positions_per_stream, 2),
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


def _writer_config(arguments: Mapping[str, Any]) -> HierarchicalRasterLanguageConfig:
    return HierarchicalRasterLanguageConfig(
        patch_size=int(arguments["patch_size"]),
        local_dim=int(arguments["local_dim"]),
        local_layers=int(arguments["local_layers"]),
        local_heads=int(arguments["local_heads"]),
        mixtures=int(arguments["mixtures"]),
    )


def _set_reader_trainable(
    model: HierarchicalRasterLanguageModel,
    trainable: bool,
) -> None:
    for parameter in model.reader.parameters():
        parameter.requires_grad = bool(trainable)


def _load_model(
    arguments: Mapping[str, Any],
) -> tuple[
    HierarchicalRasterLanguageModel,
    dict[str, Any] | None,
    str,
    int,
]:
    reader_sha256 = file_sha256(arguments["reader_checkpoint"])
    reader = torch.load(
        arguments["reader_checkpoint"], map_location="cpu", weights_only=False
    )
    if reader.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V50 requires the measured V42 visual reader")
    resume_payload: dict[str, Any] | None = None
    start_update = 0
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("architecture") != (
            HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE
        ):
            raise ValueError("resume checkpoint is not V50")
        if resume_payload.get("reader_initialization_sha256") != reader_sha256:
            raise ValueError("V50 reader initialization changed across resume")
        reader_config = canonical_glyph_language_config_from_payload(
            resume_payload["reader_config"]
        )
        writer_config = hierarchical_raster_language_config_from_payload(
            resume_payload["writer_config"]
        )
        start_update = int(resume_payload["update"])
    else:
        reader_config = canonical_glyph_language_config_from_payload(
            reader["model_config"]
        )
        writer_config = _writer_config(arguments)
    model = HierarchicalRasterLanguageModel(reader_config, writer_config)
    if resume_payload is None:
        model.load_v42_reader(reader["model"])
    else:
        model.load_state_dict(resume_payload["model"], strict=True)
    return model, resume_payload, reader_sha256, start_update


def _checkpoint(
    model: HierarchicalRasterLanguageModel,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    reader_sha256: str,
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
        "architecture": HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
        "reader_config": canonical_glyph_language_config_payload(
            model.reader_config
        ),
        "writer_config": hierarchical_raster_language_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "update": update,
        "reader_initialization_sha256": reader_sha256,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": canonical_glyph_data_boundary_receipt(),
        "model_boundary": hierarchical_raster_language_boundary_receipt(model),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": {
                path: file_sha256(path) for path in SOURCE_FILES
            },
            "fixed_optimization": FIXED_OPTIMIZATION,
            "effective_arguments": dict(arguments),
        },
        "loss_weights": V50_LOSS_WEIGHTS.__dict__,
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": bool(arguments["exploratory"]),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return payload


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    if min(
        arguments["steps"],
        arguments["batch_size"],
        arguments["positions_per_stream"],
    ) < 1:
        raise ValueError("V50 training sizes must be positive")
    if not 0 <= arguments["reader_freeze_steps"] < arguments["steps"]:
        raise ValueError("V50 reader freeze must end before training ends")
    if arguments["positions_per_stream"] > 64:
        raise ValueError("V50 selected positions cannot exceed the visual stream")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model, resume_payload, reader_sha256, start_update = _load_model(arguments)
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested V50 updates")
    model.to(device)
    reader_parameters = list(model.reader.parameters())
    writer_parameters = list(model.writer.parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": reader_parameters,
                "lr": arguments["reader_learning_rate"],
                "name": "reader",
            },
            {
                "params": writer_parameters,
                "lr": arguments["writer_learning_rate"],
                "name": "writer",
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    strict = not arguments["exploratory"] and not arguments["smoke"]
    manifest_receipt = verify_v25_manifest(arguments["manifest"], strict=strict)
    if strict and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V50 production requires the frozen corpus manifest")
    records = load_v25_records(arguments["manifest"], strict_manifest=strict)
    render_config = CanonicalGlyphRenderConfig()
    partition_receipt = visual_cell_partition_receipt(records)
    total_examples = arguments["steps"] * arguments["batch_size"]
    consumed_examples = start_update * arguments["batch_size"]
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_examples,
    )
    loader = DataLoader(
        Subset(dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_collate,
    )
    iterator = iter(loader)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and arguments["precision"] == "fp16",
    )
    training_generator = torch.Generator(device=device)
    if resume_payload is None:
        training_generator.manual_seed(arguments["seed"])
    else:
        training_generator.set_state(resume_payload["training_generator_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in resume_payload:
            torch.cuda.set_rng_state(resume_payload["cuda_rng_state"], device)

    log_path = output / "training_metrics.jsonl"
    started = time.perf_counter()
    previous_elapsed = (
        float(resume_payload.get("elapsed_seconds", 0.0))
        if resume_payload is not None
        else 0.0
    )
    metrics: dict[str, float] = {}
    stop_requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for update in range(start_update + 1, arguments["steps"] + 1):
            reader_trainable = update > arguments["reader_freeze_steps"]
            _set_reader_trainable(model, reader_trainable)
            model.train()
            if not reader_trainable:
                model.reader.eval()
            writer_schedule = _scheduled_lr(
                update,
                base=1.0,
                warmup=arguments["warmup_steps"],
                total=arguments["steps"],
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            joint_update = max(1, update - arguments["reader_freeze_steps"])
            joint_total = arguments["steps"] - arguments["reader_freeze_steps"]
            reader_schedule = _scheduled_lr(
                joint_update,
                base=1.0,
                warmup=min(arguments["warmup_steps"], max(0, joint_total - 1)),
                total=joint_total,
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            optimizer.param_groups[0]["lr"] = (
                reader_schedule * arguments["reader_learning_rate"]
                if reader_trainable
                else 0.0
            )
            optimizer.param_groups[1]["lr"] = (
                writer_schedule * arguments["writer_learning_rate"]
            )
            raw = canonical_glyph_student_batch(next(iterator))
            batch = {
                key: value.to(device, non_blocking=True) for key, value in raw.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                if reader_trainable:
                    language = model.language(batch["context"])
                else:
                    with torch.no_grad():
                        language = model.language(batch["context"])
                loss = hierarchical_raster_language_loss(
                    model,
                    language,
                    batch["target"],
                    positions_per_stream=arguments["positions_per_stream"],
                    generator=training_generator,
                    counterfactual_margin=arguments["counterfactual_margin"],
                )
            if scaler.is_enabled():
                scaler.scale(loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.loss.backward()
            active_parameters = writer_parameters + (
                reader_parameters if reader_trainable else []
            )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                active_parameters,
                arguments["gradient_clip"],
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            elapsed = previous_elapsed + time.perf_counter() - started
            metrics = loss.detached_metrics()
            metrics.update(
                {
                    "update": float(update),
                    "reader_trainable": float(reader_trainable),
                    "reader_learning_rate": optimizer.param_groups[0]["lr"],
                    "writer_learning_rate": optimizer.param_groups[1]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": elapsed,
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(log_path, metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
            should_save = (
                update == arguments["steps"]
                or update % arguments["save_every"] == 0
                or stop_requested
            )
            if should_save:
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint(
                    model,
                    optimizer,
                    training_generator,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    reader_sha256=reader_sha256,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=metrics,
                    device=device,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    boundary = payload["model_boundary"]
                    _atomic_json(
                        {
                            "architecture": HIERARCHICAL_RASTER_LANGUAGE_ARCHITECTURE,
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "total_parameters": boundary["total_parameters"],
                            "reader_parameters": boundary["reader_parameters"],
                            "writer_parameters": boundary["writer_parameters"],
                            "training_metrics": metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V50 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
