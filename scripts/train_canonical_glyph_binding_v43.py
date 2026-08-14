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

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    CanonicalGlyphFlowV43Config,
    canonical_glyph_flow_v43_boundary_receipt,
    canonical_glyph_flow_v43_config_payload,
)
from ilm.visual_lm.canonical_glyph_flow_v43_data import (
    V43_ARCHITECTURE,
    CanonicalGlyphPairTrainingDataset,
    canonical_glyph_flow_v43_data_boundary_receipt,
    canonical_glyph_pair_student_batch,
    canonical_glyph_pair_training_collate,
)
from ilm.visual_lm.canonical_glyph_flow_v43_training import (
    canonical_glyph_flow_v43_language_loss,
)
from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
    canonical_glyph_language_config_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/canonical_glyph_flow_v43_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_BASE = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/canonical_glyph_flow_v43_20260814/binding"
PINNED_V42_SHA256 = "a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870"
SOURCE_FILES = (
    "ilm/visual_lm/canonical_glyph_flow_v43.py",
    "ilm/visual_lm/canonical_glyph_flow_v43_data.py",
    "ilm/visual_lm/canonical_glyph_flow_v43_evaluation.py",
    "ilm/visual_lm/canonical_glyph_flow_v43_training.py",
    "scripts/train_canonical_glyph_binding_v43.py",
    "scripts/train_canonical_glyph_writer_v43.py",
    "scripts/eval_canonical_glyph_flow_v43.py",
)
FIXED_OPTIMIZATION = {
    "steps": 3_000,
    "batch_size": 8,
    "pair_pool": 5_000,
    "pair_suffix_cells": 4,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "seed": 20264300,
    "dataset_seed": 20264301,
    "pair_seed": 20264302,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune V42 on train-only visual suffix pairs for V43."
    )
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--resume")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--batch-size", type=int, default=FIXED_OPTIMIZATION["batch_size"]
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pair-pool", type=int, default=FIXED_OPTIMIZATION["pair_pool"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=FIXED_OPTIMIZATION["learning_rate"]
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
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument(
        "--pair-seed", type=int, default=FIXED_OPTIMIZATION["pair_seed"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
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
                "pair_pool": min(args.pair_pool, 32),
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


def _resolve_device(value: str) -> torch.device:
    device = choose_device(value)
    if device.type != "cuda":
        return device
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _checkpoint(
    model: CanonicalGlyphFlowV43,
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
    base_sha256: str,
) -> dict[str, Any]:
    source_hashes = {
        path: file_sha256(path) for path in SOURCE_FILES if Path(path).exists()
    }
    return {
        "experiment": "canonical-glyph-flow-v43-binding",
        "architecture": V43_ARCHITECTURE,
        "stage": "binding",
        "language_config": canonical_glyph_language_config_payload(
            model.language_model.config
        ),
        "v43_config": canonical_glyph_flow_v43_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "language_model": model.language_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "base_v42_sha256": base_sha256,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "pair_pool": {
            "count": int(arguments["pair_pool"]),
            "suffix_cells": 4,
            "seed": int(arguments["pair_seed"]),
            "split": "train",
            "candidate_columns_permuted": True,
        },
        "data_boundary": canonical_glyph_flow_v43_data_boundary_receipt(),
        "model_boundary": canonical_glyph_flow_v43_boundary_receipt(model),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": source_hashes,
            "fixed_optimization": FIXED_OPTIMIZATION,
            "effective_arguments": dict(arguments),
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
    }


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    if min(arguments["steps"], arguments["batch_size"], arguments["pair_pool"]) < 1:
        raise ValueError("V43 binding sizes must be positive")
    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    base_sha256 = file_sha256(arguments["base_checkpoint"])
    if base_sha256 != PINNED_V42_SHA256:
        raise ValueError(f"V43 requires the pinned V42 checkpoint, got {base_sha256}")
    base = torch.load(
        arguments["base_checkpoint"], map_location="cpu", weights_only=False
    )
    language_config = canonical_glyph_language_config_from_payload(base["model_config"])
    model = CanonicalGlyphFlowV43(
        language_config,
        CanonicalGlyphFlowV43Config(),
    )
    model.language_model.load_state_dict(base["model"], strict=True)
    model.freeze_writer()
    model.unfreeze_language_core()

    start_update = 0
    resume_payload: dict[str, Any] | None = None
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if (
            resume_payload.get("architecture") != V43_ARCHITECTURE
            or resume_payload.get("stage") != "binding"
        ):
            raise ValueError("resume checkpoint is not V43 binding")
        model.language_model.load_state_dict(
            resume_payload["language_model"], strict=True
        )
        start_update = int(resume_payload["update"])
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested updates")
    model.to(device)

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"], strict=not arguments["smoke"]
    )
    if not arguments["smoke"] and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V43 binding requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"], strict_manifest=not arguments["smoke"]
    )
    render_config = CanonicalGlyphRenderConfig()
    pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=arguments["pair_pool"],
        seed=arguments["pair_seed"],
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    total_examples = arguments["steps"] * arguments["batch_size"]
    consumed_examples = start_update * arguments["batch_size"]
    natural_dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_examples,
    )
    pair_dataset = CanonicalGlyphPairTrainingDataset(
        pairs,
        render_config=render_config,
        seed=arguments["pair_seed"],
        length=total_examples,
    )
    natural_loader = DataLoader(
        Subset(natural_dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_collate,
    )
    pair_loader = DataLoader(
        Subset(pair_dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_pair_training_collate,
    )
    natural_iterator = iter(natural_loader)
    pair_iterator = iter(pair_loader)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and arguments["precision"] == "fp16",
    )
    started = time.perf_counter()
    metrics: dict[str, float] = {}
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
            natural = canonical_glyph_student_batch(next(natural_iterator))
            paired = canonical_glyph_pair_student_batch(next(pair_iterator))
            natural = {
                key: value.to(device, non_blocking=True)
                for key, value in natural.items()
            }
            paired = {
                key: value.to(device, non_blocking=True)
                for key, value in paired.items()
            }
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                natural_output = model.language_model(natural["context"])
                loss = canonical_glyph_flow_v43_language_loss(
                    model,
                    natural_output,
                    natural["target"],
                    paired["contexts"],
                    paired["candidates"],
                    paired["assignment"],
                )
            if scaler.is_enabled():
                scaler.scale(loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, arguments["gradient_clip"]
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            metrics = loss.detached_metrics()
            metrics.update(
                {
                    "update": float(update),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(output / "training_metrics.jsonl", metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
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
                    partition_receipt=visual_cell_partition_receipt(records),
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=metrics,
                    base_sha256=base_sha256,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V43_ARCHITECTURE,
                            "stage": "binding",
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "trainable_parameters": sum(
                                parameter.numel() for parameter in trainable
                            ),
                            "training_metrics": metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V43 binding stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
