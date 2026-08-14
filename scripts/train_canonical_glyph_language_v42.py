#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language import (
    CanonicalGlyphLanguageConfig,
    CanonicalGlyphLanguageModel,
    canonical_glyph_language_boundary_receipt,
    canonical_glyph_language_config_from_payload,
    canonical_glyph_language_config_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    V42_ARCHITECTURE,
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.canonical_glyph_language_training import (
    V42_LOSS_WEIGHTS,
    canonical_glyph_language_loss,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/canonical_glyph_language_v42_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = "artifacts/canonical_glyph_language_v42_20260814"
SOURCE_FILES = (
    "ilm/visual_lm/canonical_glyph_language.py",
    "ilm/visual_lm/canonical_glyph_language_data.py",
    "ilm/visual_lm/canonical_glyph_language_training.py",
    "ilm/visual_lm/canonical_glyph_language_evaluation.py",
    "scripts/train_canonical_glyph_language_v42.py",
    "scripts/eval_canonical_glyph_language_v42.py",
)
FIXED_OPTIMIZATION = {
    "steps": 10_000,
    "batch_size": 8,
    "gradient_accumulation": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "maximum_contrastive_positions": 512,
    "maximum_energy_positions": 128,
    "energy_samples": 4,
    "seed": 20264200,
    "dataset_seed": 20264201,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V42 image-only canonical Chinese language core."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--batch-size", type=int, default=FIXED_OPTIMIZATION["batch_size"]
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=FIXED_OPTIMIZATION["gradient_accumulation"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=FIXED_OPTIMIZATION["learning_rate"],
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
        "--maximum-contrastive-positions",
        type=int,
        default=FIXED_OPTIMIZATION["maximum_contrastive_positions"],
    )
    parser.add_argument(
        "--maximum-energy-positions",
        type=int,
        default=FIXED_OPTIMIZATION["maximum_energy_positions"],
    )
    parser.add_argument(
        "--energy-samples",
        type=int,
        default=FIXED_OPTIMIZATION["energy_samples"],
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _scheduled_lr(
    step: int,
    *,
    base: float,
    warmup: int,
    total: int,
    minimum_ratio: float,
) -> float:
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if args.smoke:
        payload.update(
            {
                "steps": min(args.steps, 2),
                "batch_size": min(args.batch_size, 2),
                "gradient_accumulation": 1,
                "num_workers": 0,
                "warmup_steps": 1,
                "maximum_contrastive_positions": min(
                    args.maximum_contrastive_positions, 64
                ),
                "maximum_energy_positions": min(
                    args.maximum_energy_positions, 8
                ),
                "energy_samples": max(2, min(args.energy_samples, 2)),
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


def _protocol_receipt(arguments: Mapping[str, Any]) -> dict[str, Any]:
    source_hashes = {
        path: file_sha256(path) for path in SOURCE_FILES if Path(path).exists()
    }
    return {
        "document": PROTOCOL_DOCUMENT,
        "sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": source_hashes,
        "fixed_optimization": FIXED_OPTIMIZATION,
        "effective_arguments": dict(arguments),
    }


def _checkpoint(
    model: CanonicalGlyphLanguageModel,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "experiment": "canonical-glyph-language-v42",
        "architecture": V42_ARCHITECTURE,
        "model_config": canonical_glyph_language_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": canonical_glyph_data_boundary_receipt(),
        "model_boundary": canonical_glyph_language_boundary_receipt(model),
        "protocol": _protocol_receipt(arguments),
        "loss_weights": V42_LOSS_WEIGHTS.__dict__,
        "training_elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": bool(arguments["exploratory"]),
    }


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    if arguments["steps"] < 1 or arguments["batch_size"] < 1:
        raise ValueError("V42 training sizes must be positive")
    if arguments["gradient_accumulation"] < 1 or arguments["energy_samples"] < 2:
        raise ValueError("V42 accumulation and energy sampling are invalid")
    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        torch.cuda.set_device(cuda_index)
        device = torch.device("cuda", cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"],
        strict=not arguments["exploratory"] and not arguments["smoke"],
    )
    if (
        not arguments["exploratory"]
        and not arguments["smoke"]
        and manifest_receipt["sha256"] != V25_MANIFEST_SHA256
    ):
        raise ValueError("V42 production requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"],
        strict_manifest=not arguments["exploratory"] and not arguments["smoke"],
    )
    render_config = CanonicalGlyphRenderConfig()
    partition_receipt = visual_cell_partition_receipt(records)

    model_config = CanonicalGlyphLanguageConfig()
    start_update = 0
    resume_payload: dict[str, Any] | None = None
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("architecture") != V42_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V42")
        model_config = canonical_glyph_language_config_from_payload(
            resume_payload["model_config"]
        )
        start_update = int(resume_payload["update"])
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested V42 updates")
    model = CanonicalGlyphLanguageModel(model_config).to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    total_examples = (
        arguments["steps"]
        * arguments["gradient_accumulation"]
        * arguments["batch_size"]
    )
    consumed_examples = (
        start_update
        * arguments["gradient_accumulation"]
        * arguments["batch_size"]
    )
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_examples,
    )
    remaining = Subset(dataset, range(consumed_examples, total_examples))
    loader = DataLoader(
        remaining,
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
    training_generator = torch.Generator(device=device).manual_seed(
        arguments["seed"] + start_update * 1_000_003
    )
    log_path = output / "training_metrics.jsonl"
    started = time.perf_counter()
    final_metrics: dict[str, float] = {}
    stop_requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for update in range(start_update + 1, arguments["steps"] + 1):
            model.train()
            learning_rate = _scheduled_lr(
                update,
                base=arguments["learning_rate"],
                warmup=arguments["warmup_steps"],
                total=arguments["steps"],
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            for _ in range(arguments["gradient_accumulation"]):
                raw = next(iterator)
                student = canonical_glyph_student_batch(raw)
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in student.items()
                }
                with autocast_context(device, arguments["precision"]):
                    prediction = model(batch["context"])
                    loss = canonical_glyph_language_loss(
                        model,
                        prediction,
                        batch["target"],
                        generator=training_generator,
                        maximum_contrastive_positions=arguments[
                            "maximum_contrastive_positions"
                        ],
                        maximum_energy_positions=arguments[
                            "maximum_energy_positions"
                        ],
                        energy_samples=arguments["energy_samples"],
                    )
                    scaled_loss = loss.loss / arguments["gradient_accumulation"]
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for key, value in loss.detached_metrics().items():
                    totals[key] = totals.get(key, 0.0) + value
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), arguments["gradient_clip"]
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            final_metrics = {
                key: value / arguments["gradient_accumulation"]
                for key, value in totals.items()
            }
            final_metrics.update(
                {
                    "update": float(update),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            should_save = (
                update == arguments["steps"]
                or update % arguments["save_every"] == 0
                or stop_requested
            )
            if should_save:
                elapsed = time.perf_counter() - started
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint(
                    model,
                    optimizer,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=final_metrics,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V42_ARCHITECTURE,
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "total_parameters": payload["model_boundary"][
                                "total_parameters"
                            ],
                            "trainable_parameters": payload["model_boundary"][
                                "trainable_parameters"
                            ],
                            "training_metrics": final_metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V42 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
