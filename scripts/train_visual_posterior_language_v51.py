#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_posterior_language_v51 import (
    V51_ARCHITECTURE,
    VisualGlyphScorePrior,
    VisualPosteriorLanguageConfig,
    visual_posterior_language_boundary_receipt,
    visual_posterior_language_config_from_payload,
    visual_posterior_language_config_payload,
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


PROTOCOL_DOCUMENT = "references/visual_posterior_language_v51_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_READER = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/visual_posterior_language_v51_20261104"
SOURCE_FILES = (
    "ilm/visual_lm/visual_posterior_language_v51.py",
    "scripts/train_visual_posterior_language_v51.py",
)
FIXED_OPTIMIZATION = {
    "steps": 2_000,
    "batch_size": 8,
    "positions_per_stream": 16,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": 20265110,
    "dataset_seed": 20265111,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the unconditional 32x32 writing score prior used by the "
            "V51 visual-posterior language sampler."
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
        "--ema-decay", type=float, default=FIXED_OPTIMIZATION["ema_decay"]
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
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
                "batch_size": min(args.batch_size, 2),
                "positions_per_stream": min(args.positions_per_stream, 2),
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
                "base_channels": min(args.base_channels, 16),
                "unet_depth": min(args.unet_depth, 1),
                "diffusion_steps": max(20, min(args.diffusion_steps, 20)),
            }
        )
    return payload


def _model_config(arguments: Mapping[str, Any]) -> VisualPosteriorLanguageConfig:
    diffusion_steps = int(arguments["diffusion_steps"])
    return VisualPosteriorLanguageConfig(
        diffusion_steps=diffusion_steps,
        base_channels=int(arguments["base_channels"]),
        unet_depth=int(arguments["unet_depth"]),
        image_to_image_start=min(30, diffusion_steps - 1),
        sampling_steps=min(16, diffusion_steps),
    )


def _load_model(
    arguments: Mapping[str, Any],
) -> tuple[VisualGlyphScorePrior, VisualGlyphScorePrior, dict[str, Any] | None, int]:
    resume_payload: dict[str, Any] | None = None
    start_update = 0
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("architecture") != V51_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V51")
        config = visual_posterior_language_config_from_payload(
            resume_payload["model_config"]
        )
        start_update = int(resume_payload["update"])
    else:
        config = _model_config(arguments)
    model = VisualGlyphScorePrior(config)
    ema = copy.deepcopy(model).requires_grad_(False)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
        ema.load_state_dict(resume_payload["ema_model"], strict=True)
    return model, ema, resume_payload, start_update


@torch.no_grad()
def _update_ema(
    ema: VisualGlyphScorePrior,
    model: VisualGlyphScorePrior,
    *,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _select_cells(
    streams: torch.Tensor,
    *,
    positions_per_stream: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if streams.ndim != 5 or tuple(streams.shape[2:]) != (1, 32, 32):
        raise ValueError("V51 training streams must be [B,T,1,32,32]")
    if not 1 <= positions_per_stream <= streams.shape[1]:
        raise ValueError("V51 selected positions are outside the stream")
    choices = torch.stack(
        [
            torch.randperm(
                streams.shape[1],
                device=streams.device,
                generator=generator,
            )[:positions_per_stream]
            for _ in range(len(streams))
        ]
    )
    rows = torch.arange(len(streams), device=streams.device)[:, None]
    return streams[rows, choices].flatten(0, 1)


def _checkpoint(
    model: VisualGlyphScorePrior,
    ema: VisualGlyphScorePrior,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    reader_checkpoint_sha256: str,
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": V51_ARCHITECTURE,
        "architecture": V51_ARCHITECTURE,
        "model_config": visual_posterior_language_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "ema_model": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "update": update,
        "reader_checkpoint_sha256": reader_checkpoint_sha256,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": canonical_glyph_data_boundary_receipt(),
        "model_boundary": visual_posterior_language_boundary_receipt(model),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": {
                path: file_sha256(path) for path in SOURCE_FILES
            },
            "fixed_optimization": FIXED_OPTIMIZATION,
            "effective_arguments": dict(arguments),
        },
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
        raise ValueError("V51 training sizes must be positive")
    if arguments["positions_per_stream"] > 64:
        raise ValueError("V51 selected positions cannot exceed 64")
    if not 0.0 < arguments["ema_decay"] < 1.0:
        raise ValueError("V51 EMA decay must lie in (0,1)")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reader_checkpoint = torch.load(
        arguments["reader_checkpoint"], map_location="cpu", weights_only=False
    )
    if reader_checkpoint.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V51 requires the measured V42 reader checkpoint")
    reader_checkpoint_sha256 = file_sha256(arguments["reader_checkpoint"])

    model, ema, resume_payload, start_update = _load_model(arguments)
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested V51 updates")
    model.to(device)
    ema.to(device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    strict = not arguments["exploratory"] and not arguments["smoke"]
    manifest_receipt = verify_v25_manifest(arguments["manifest"], strict=strict)
    if strict and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V51 production requires the frozen corpus manifest")
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
            model.train()
            schedule = _scheduled_lr(
                update,
                base=arguments["learning_rate"],
                warmup=arguments["warmup_steps"],
                total=arguments["steps"],
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            optimizer.param_groups[0]["lr"] = schedule
            raw = canonical_glyph_student_batch(next(iterator))
            targets = raw["target"].to(device, non_blocking=True)
            clean = _select_cells(
                targets,
                positions_per_stream=arguments["positions_per_stream"],
                generator=training_generator,
            )
            timesteps = torch.randint(
                0,
                model.config.diffusion_steps,
                (len(clean),),
                device=device,
                generator=training_generator,
            )
            noise = torch.randn(
                clean.shape,
                device=device,
                dtype=clean.dtype,
                generator=training_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                loss = model.training_loss(clean, timesteps, noise)
            if scaler.is_enabled():
                scaler.scale(loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), arguments["gradient_clip"]
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _update_ema(ema, model, decay=arguments["ema_decay"])
            elapsed = previous_elapsed + time.perf_counter() - started
            metrics = loss.detached_metrics()
            metrics.update(
                {
                    "update": float(update),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "selected_images": float(len(clean)),
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
                    ema,
                    optimizer,
                    training_generator,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    reader_checkpoint_sha256=reader_checkpoint_sha256,
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
                            "architecture": V51_ARCHITECTURE,
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "prior_parameters": boundary[
                                "total_prior_parameters"
                            ],
                            "training_metrics": metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V51 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
