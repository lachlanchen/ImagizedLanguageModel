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

from ilm.visual_lm.canonical_glyph_flow_v43 import (
    CanonicalGlyphFlowV43,
    canonical_glyph_flow_v43_boundary_receipt,
    canonical_glyph_flow_v43_config_from_payload,
    canonical_glyph_flow_v43_config_payload,
)
from ilm.visual_lm.canonical_glyph_flow_v43_data import V43_ARCHITECTURE
from ilm.visual_lm.canonical_glyph_flow_v43_training import (
    canonical_glyph_flow_v43_writer_loss,
    select_writer_positions,
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
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from scripts.train_canonical_glyph_binding_v43 import (
    PROTOCOL_DOCUMENT,
    SOURCE_FILES,
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


DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_BINDING = (
    "artifacts/canonical_glyph_flow_v43_20260814/binding/checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/canonical_glyph_flow_v43_20260814/writer"
FIXED_OPTIMIZATION = {
    "steps": 5_000,
    "batch_size": 8,
    "positions_per_stream": 16,
    "learning_rate": 2e-4,
    "warmup_steps": 200,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "condition_dropout": 0.10,
    "endpoint_weight": 0.10,
    "stroke_weight": 2.0,
    "seed": 20264320,
    "dataset_seed": 20264321,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V43 bank-free conditional spatial glyph writer."
    )
    parser.add_argument("--binding-checkpoint", default=DEFAULT_BINDING)
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
        "--positions-per-stream",
        type=int,
        default=FIXED_OPTIMIZATION["positions_per_stream"],
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
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=FIXED_OPTIMIZATION["condition_dropout"],
    )
    parser.add_argument(
        "--endpoint-weight",
        type=float,
        default=FIXED_OPTIMIZATION["endpoint_weight"],
    )
    parser.add_argument(
        "--stroke-weight", type=float, default=FIXED_OPTIMIZATION["stroke_weight"]
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if args.smoke:
        payload.update(
            {
                "steps": min(args.steps, 2),
                "batch_size": min(args.batch_size, 2),
                "positions_per_stream": min(args.positions_per_stream, 2),
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


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
    binding_sha256: str,
) -> dict[str, Any]:
    return {
        "experiment": "canonical-glyph-flow-v43-writer",
        "architecture": V43_ARCHITECTURE,
        "stage": "writer",
        "language_config": canonical_glyph_language_config_payload(
            model.language_model.config
        ),
        "v43_config": canonical_glyph_flow_v43_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": update,
        "binding_checkpoint_sha256": binding_sha256,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "model_boundary": canonical_glyph_flow_v43_boundary_receipt(model),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": {
                path: file_sha256(path) for path in SOURCE_FILES if Path(path).exists()
            },
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
    if (
        min(
            arguments["steps"],
            arguments["batch_size"],
            arguments["positions_per_stream"],
        )
        < 1
    ):
        raise ValueError("V43 writer sizes must be positive")
    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    binding_sha256 = file_sha256(arguments["binding_checkpoint"])
    binding = torch.load(
        arguments["binding_checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        binding.get("architecture") != V43_ARCHITECTURE
        or binding.get("stage") != "binding"
    ):
        raise ValueError("V43 writer requires a V43 binding checkpoint")
    if binding.get("smoke_only") and not arguments["smoke"]:
        raise PermissionError("production V43 writer cannot use a smoke binding")
    model = CanonicalGlyphFlowV43(
        canonical_glyph_language_config_from_payload(binding["language_config"]),
        canonical_glyph_flow_v43_config_from_payload(binding["v43_config"]),
    )
    model.language_model.load_state_dict(binding["language_model"], strict=True)
    model.freeze_language()
    model.unfreeze_writer()

    start_update = 0
    resume_payload: dict[str, Any] | None = None
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if (
            resume_payload.get("architecture") != V43_ARCHITECTURE
            or resume_payload.get("stage") != "writer"
            or resume_payload.get("binding_checkpoint_sha256") != binding_sha256
        ):
            raise ValueError("resume checkpoint is not this V43 writer run")
        model.load_state_dict(resume_payload["model"], strict=True)
        start_update = int(resume_payload["update"])
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested updates")
    model.to(device)
    trainable = [
        parameter for parameter in model.writer.parameters() if parameter.requires_grad
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
        raise ValueError("V43 writer requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"], strict_manifest=not arguments["smoke"]
    )
    render_config = CanonicalGlyphRenderConfig()
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
    training_generator = torch.Generator(device=device).manual_seed(
        arguments["seed"] + start_update * 1_000_003
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
            raw = canonical_glyph_student_batch(next(iterator))
            batch = {
                key: value.to(device, non_blocking=True) for key, value in raw.items()
            }
            with torch.no_grad(), autocast_context(device, arguments["precision"]):
                language = model.language_model(batch["context"])
                hidden, anchors, target = select_writer_positions(
                    language["hidden_states"],
                    language["anchor_fields"],
                    batch["target"],
                    positions_per_stream=arguments["positions_per_stream"],
                    generator=training_generator,
                )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                loss = canonical_glyph_flow_v43_writer_loss(
                    model,
                    hidden.detach(),
                    anchors.detach(),
                    target,
                    generator=training_generator,
                    condition_dropout=arguments["condition_dropout"],
                    endpoint_weight=arguments["endpoint_weight"],
                    stroke_weight=arguments["stroke_weight"],
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
                    binding_sha256=binding_sha256,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V43_ARCHITECTURE,
                            "stage": "writer",
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "writer_parameters": sum(
                                parameter.numel() for parameter in trainable
                            ),
                            "total_parameters": sum(
                                parameter.numel() for parameter in model.parameters()
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
                    f"V43 writer stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
