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

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphRenderConfig,
    canonical_glyph_render_config_payload,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_future_block_language_v48 import (
    VisualFutureBlockLanguageConfigV48,
    VisualFutureBlockLanguageModelV48,
    visual_future_block_language_boundary_receipt_v48,
    visual_future_block_language_config_from_payload_v48,
    visual_future_block_language_config_payload_v48,
)
from ilm.visual_lm.visual_future_block_language_v48_data import (
    V48_ARCHITECTURE,
    VisualFutureBlockLanguageDataset,
    visual_future_block_collate,
    visual_future_block_data_boundary_receipt,
    visual_future_block_student_batch,
)
from ilm.visual_lm.visual_future_block_language_v48_training import (
    V48_LOSS_WEIGHTS,
    visual_future_block_language_loss_v48,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/visual_future_block_language_v48_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_OUTPUT = "artifacts/visual_future_block_language_v48_20260815"
SOURCE_FILES = (
    "ilm/visual_lm/visual_future_block_language_v48.py",
    "ilm/visual_lm/visual_future_block_language_v48_data.py",
    "ilm/visual_lm/visual_future_block_language_v48_training.py",
    "ilm/visual_lm/visual_future_block_language_v48_evaluation.py",
    "scripts/train_visual_future_block_language_v48.py",
    "scripts/eval_visual_future_block_language_v48.py",
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
    "maximum_positions": 512,
    "seed": 20264800,
    "dataset_seed": 20264801,
    "log_every": 50,
    "save_every": 1_000,
}
V48_PRODUCTION_PARAMETERS = 16_278_401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen V48 image-only future-block language core."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
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
        "--weight-decay",
        type=float,
        default=FIXED_OPTIMIZATION["weight_decay"],
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=FIXED_OPTIMIZATION["gradient_clip"],
    )
    parser.add_argument(
        "--maximum-positions",
        type=int,
        default=FIXED_OPTIMIZATION["maximum_positions"],
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument(
        "--log-every", type=int, default=FIXED_OPTIMIZATION["log_every"]
    )
    parser.add_argument(
        "--save-every", type=int, default=FIXED_OPTIMIZATION["save_every"]
    )
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
                "out": (
                    f"{DEFAULT_OUTPUT}_smoke"
                    if args.out == DEFAULT_OUTPUT
                    else args.out
                ),
                "steps": min(args.steps, 2),
                "batch_size": min(args.batch_size, 2),
                "gradient_accumulation": 1,
                "num_workers": 0,
                "warmup_steps": 1,
                "maximum_positions": min(args.maximum_positions, 64),
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


def _assert_production_arguments(arguments: Mapping[str, Any]) -> None:
    if arguments["smoke"]:
        return
    if arguments["exploratory"]:
        if arguments["out"] == DEFAULT_OUTPUT:
            raise ValueError("V48 exploratory runs require a distinct output path")
        return
    if arguments["precision"] != "bf16":
        raise ValueError("V48 production fixes BF16 precision")
    for key, expected in FIXED_OPTIMIZATION.items():
        if arguments[key] != expected:
            raise ValueError(
                f"V48 production fixes {key}={expected!r}, "
                f"got {arguments[key]!r}"
            )


def _resolve_device(value: str) -> torch.device:
    device = choose_device(value)
    if device.type != "cuda":
        return device
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _source_hashes(*, require_all: bool) -> dict[str, str]:
    missing = [path for path in SOURCE_FILES if not Path(path).is_file()]
    if require_all and missing:
        raise FileNotFoundError(f"V48 production source files are missing: {missing}")
    return {
        path: file_sha256(path) for path in SOURCE_FILES if Path(path).is_file()
    }


def _protocol_receipt(arguments: Mapping[str, Any]) -> dict[str, Any]:
    production = not arguments["smoke"] and not arguments["exploratory"]
    return {
        "document": PROTOCOL_DOCUMENT,
        "sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": _source_hashes(require_all=production),
        "fixed_optimization": FIXED_OPTIMIZATION,
        "effective_arguments": dict(arguments),
    }


def _global_rng_receipt(device: torch.device) -> dict[str, torch.Tensor]:
    receipt = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        receipt["cuda"] = torch.cuda.get_rng_state(device)
    return receipt


def _restore_global_rng(
    receipt: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> None:
    cpu_state = receipt.get("cpu")
    if not isinstance(cpu_state, torch.Tensor):
        raise ValueError("V48 resume checkpoint lacks the CPU RNG state")
    torch.random.set_rng_state(cpu_state)
    if device.type == "cuda":
        cuda_state = receipt.get("cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("V48 resume checkpoint lacks the CUDA RNG state")
        torch.cuda.set_rng_state(cuda_state, device)


def _checkpoint(
    model: VisualFutureBlockLanguageModelV48,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    training_generator: torch.Generator,
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "experiment": V48_ARCHITECTURE,
        "architecture": V48_ARCHITECTURE,
        "model_config": visual_future_block_language_config_payload_v48(
            model.config
        ),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "global_rng_state": _global_rng_receipt(device),
        "update": update,
        "segments_consumed": (
            update
            * int(arguments["gradient_accumulation"])
            * int(arguments["batch_size"])
        ),
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": visual_future_block_data_boundary_receipt(),
        "model_boundary": visual_future_block_language_boundary_receipt_v48(model),
        "protocol": _protocol_receipt(arguments),
        "loss_weights": V48_LOSS_WEIGHTS.__dict__,
        "training_elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": bool(arguments["exploratory"]),
    }


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    _assert_production_arguments(arguments)
    positive = (
        "steps",
        "batch_size",
        "gradient_accumulation",
        "learning_rate",
        "warmup_steps",
        "minimum_lr_ratio",
        "gradient_clip",
        "maximum_positions",
        "log_every",
        "save_every",
    )
    if any(float(arguments[key]) <= 0 for key in positive):
        raise ValueError("V48 training values must be positive")
    if float(arguments["weight_decay"]) < 0.0:
        raise ValueError("V48 weight decay cannot be negative")
    if int(arguments["num_workers"]) < 0:
        raise ValueError("V48 worker count cannot be negative")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    production = not arguments["smoke"] and not arguments["exploratory"]
    if production and (device.type != "cuda" or device.index != 0):
        raise ValueError("V48 production is frozen to CUDA device 0")
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"],
        strict=production,
    )
    if production and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V48 production requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"],
        strict_manifest=production,
    )
    render_config = CanonicalGlyphRenderConfig()
    partition_receipt = visual_cell_partition_receipt(records)

    resume_payload: dict[str, Any] | None = None
    model_config = VisualFutureBlockLanguageConfigV48()
    start_update = 0
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"],
            map_location="cpu",
            weights_only=False,
        )
        if resume_payload.get("architecture") != V48_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V48")
        if resume_payload.get("manifest", {}).get("sha256") != manifest_receipt[
            "sha256"
        ]:
            raise ValueError("V48 resume checkpoint has a different corpus")
        model_config = visual_future_block_language_config_from_payload_v48(
            resume_payload["model_config"]
        )
        start_update = int(resume_payload["update"])
        if production:
            saved_sources = resume_payload.get("protocol", {}).get(
                "source_files_sha256"
            )
            if saved_sources != _source_hashes(require_all=True):
                raise ValueError("V48 production source changed across resume")
            if resume_payload.get("protocol", {}).get("sha256") != file_sha256(
                PROTOCOL_DOCUMENT
            ):
                raise ValueError("V48 protocol changed across resume")
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested V48 updates")

    model = VisualFutureBlockLanguageModelV48(model_config)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
    model.to(device)
    boundary = visual_future_block_language_boundary_receipt_v48(model)
    expected_parameters = V48_PRODUCTION_PARAMETERS
    if boundary["trainable_parameters"] >= 17_000_000:
        raise ValueError("V48 model exceeds its parameter budget")
    if production and boundary["trainable_parameters"] != expected_parameters:
        raise ValueError("V48 production parameter count changed")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and arguments["precision"] == "fp16",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])
        scaler.load_state_dict(resume_payload.get("scaler", {}))

    total_examples = (
        int(arguments["steps"])
        * int(arguments["gradient_accumulation"])
        * int(arguments["batch_size"])
    )
    consumed_examples = (
        start_update
        * int(arguments["gradient_accumulation"])
        * int(arguments["batch_size"])
    )
    dataset = VisualFutureBlockLanguageDataset(
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
        collate_fn=visual_future_block_collate,
    )
    iterator = iter(loader)
    training_generator = torch.Generator(device=device)
    if resume_payload is not None:
        generator_state = resume_payload.get("training_generator_state")
        if not isinstance(generator_state, torch.Tensor):
            raise ValueError("V48 resume checkpoint lacks its sampling RNG")
        training_generator.set_state(generator_state)
        _restore_global_rng(resume_payload.get("global_rng_state", {}), device=device)
    else:
        training_generator.manual_seed(arguments["seed"])

    prior_elapsed = (
        float(resume_payload.get("training_elapsed_seconds", 0.0))
        if resume_payload is not None
        else 0.0
    )
    prior_peak = (
        float(resume_payload.get("peak_allocated_vram_gib", 0.0))
        if resume_payload is not None
        else 0.0
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
        for update in range(start_update + 1, int(arguments["steps"]) + 1):
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
            for _ in range(int(arguments["gradient_accumulation"])):
                student = visual_future_block_student_batch(next(iterator))
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in student.items()
                }
                with autocast_context(device, arguments["precision"]):
                    prediction = model(batch["context"])
                    measured = visual_future_block_language_loss_v48(
                        model,
                        prediction,
                        batch["future_pixels"],
                        generator=training_generator,
                        maximum_positions=arguments["maximum_positions"],
                    )
                    scaled_loss = (
                        measured.loss / arguments["gradient_accumulation"]
                    )
                if not bool(torch.isfinite(measured.loss)):
                    raise FloatingPointError("V48 training loss is non-finite")
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                for key, value in measured.detached_metrics().items():
                    totals[key] = totals.get(key, 0.0) + value
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), arguments["gradient_clip"]
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V48 gradient norm is non-finite")
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
                    "segments_consumed": float(
                        update
                        * int(arguments["gradient_accumulation"])
                        * int(arguments["batch_size"])
                    ),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": (
                        prior_elapsed + time.perf_counter() - started
                    ),
                }
            )
            if update == 1 or update % int(arguments["log_every"]) == 0:
                _append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            should_save = (
                update == int(arguments["steps"])
                or update % int(arguments["save_every"]) == 0
                or stop_requested
            )
            if should_save:
                elapsed = prior_elapsed + time.perf_counter() - started
                current_peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                peak = max(prior_peak, current_peak)
                payload = _checkpoint(
                    model,
                    optimizer,
                    scaler,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    training_generator=training_generator,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=final_metrics,
                    device=device,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == int(arguments["steps"]):
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V48_ARCHITECTURE,
                            "update": update,
                            "segments_consumed": payload["segments_consumed"],
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "total_parameters": boundary["total_parameters"],
                            "trainable_parameters": boundary[
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
                    f"V48 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
