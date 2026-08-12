#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa import (
    InkJEPA,
    InkJEPAConfig,
    hide_retinal_regions,
    ink_jepa_config_payload,
    ink_jepa_loss,
    sample_predictive_masks,
)
from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    VisualGrammarDataset,
    load_visual_grammar_manifest,
    visual_grammar_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the image-only Retinal Predictive Field foundation model."
    )
    data = parser.add_argument_group("data")
    data.add_argument("--manifest", default="data/visual_grammar/chinese_mvp.jsonl")
    data.add_argument("--validation-fraction", type=float, default=0.03)
    data.add_argument("--samples-per-epoch", type=int, default=20_000)
    data.add_argument("--validation-samples", type=int, default=512)
    data.add_argument("--image-height", type=int, default=192)
    data.add_argument("--image-width", type=int, default=768)
    data.add_argument("--font-size", type=int, default=25)
    data.add_argument("--minimum-font-size", type=int, default=21)
    data.add_argument("--margin", type=int, default=12)

    model = parser.add_argument_group("model")
    model.add_argument("--patch-size", type=int, default=8)
    model.add_argument("--model-dim", type=int, default=128)
    model.add_argument("--representation-dim", type=int, default=128)
    model.add_argument("--encoder-layers", type=int, default=5)
    model.add_argument("--predictor-dim", type=int, default=128)
    model.add_argument("--predictor-layers", type=int, default=2)
    model.add_argument("--heads", type=int, default=4)
    model.add_argument("--mlp-ratio", type=float, default=3.0)
    model.add_argument("--pool-slots", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.0)

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--out", default="artifacts/ink_jepa_chinese_mvp")
    optimization.add_argument("--epochs", type=int, default=20)
    optimization.add_argument("--maximum-steps", type=int, default=2_000)
    optimization.add_argument("--batch-size", type=int, default=8)
    optimization.add_argument("--num-workers", type=int, default=6)
    optimization.add_argument("--lr", type=float, default=3e-4)
    optimization.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    optimization.add_argument("--warmup-steps", type=int, default=150)
    optimization.add_argument("--weight-decay", type=float, default=0.05)
    optimization.add_argument("--ema-start", type=float, default=0.996)
    optimization.add_argument("--gradient-clip", type=float, default=1.0)
    optimization.add_argument("--field-weight", type=float, default=1.0)
    optimization.add_argument("--page-weight", type=float, default=0.35)
    optimization.add_argument("--ink-weight", type=float, default=0.12)
    optimization.add_argument("--variance-weight", type=float, default=0.08)
    optimization.add_argument("--covariance-weight", type=float, default=0.01)
    optimization.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    optimization.add_argument("--device", default="auto")
    optimization.add_argument("--seed", type=int, default=20260812)
    optimization.add_argument("--log-every", type=int, default=20)
    optimization.add_argument("--validate-every", type=int, default=200)
    optimization.add_argument("--save-every", type=int, default=400)
    optimization.add_argument("--validation-batches", type=int, default=16)
    optimization.add_argument("--resume", default=None)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def scheduled_lr(step: int, *, base: float, warmup: int, total: int, minimum_ratio: float) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def scheduled_momentum(step: int, *, start: float, total: int) -> float:
    progress = min(1.0, step / max(1, total))
    return 1.0 - (1.0 - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: InkJEPA,
    optimizer: torch.optim.Optimizer,
    *,
    render_config: RetinalRenderConfig,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    elapsed_seconds: float,
    rights: Counter[str],
) -> dict[str, Any]:
    return {
        "architecture": "ink-jepa-retinal-predictive-field-v1",
        "model_config": ink_jepa_config_payload(model.config),
        "render_config": {
            "height": render_config.height,
            "width": render_config.width,
            "font_size": render_config.font_size,
            "minimum_font_size": render_config.minimum_font_size,
            "margin": render_config.margin,
            "cell_padding": render_config.cell_padding,
            "augment": render_config.augment,
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "data_rights": dict(rights),
        "deployment_contract": {
            "input": "continuous writing image",
            "state": "continuous retinal field",
            "forbidden": ["token_ids", "unicode_ids", "ocr_strings", "text", "external_model_calls"],
            "teacher_required_at_inference": False,
        },
    }


def loss_for_batch(
    model: InkJEPA,
    batch: dict[str, Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    source = batch["view_a"].to(device, non_blocking=True)
    target = batch["view_b"].to(device, non_blocking=True)
    hidden_mask, modes = sample_predictive_masks(
        source.shape[0],
        model.config.grid_height,
        model.config.grid_width,
        device=device,
        generator=generator,
    )
    context = hide_retinal_regions(source, hidden_mask, model.config.patch_size)
    outputs = model(context, target, hidden_mask)
    loss, metrics = ink_jepa_loss(
        outputs,
        source,
        hidden_mask,
        patch_size=model.config.patch_size,
        contrastive_scale=model.contrastive_scale,
        field_weight=args.field_weight,
        page_weight=args.page_weight,
        ink_weight=args.ink_weight,
        variance_weight=args.variance_weight,
        covariance_weight=args.covariance_weight,
    )
    return loss, metrics, modes


@torch.no_grad()
def validate(
    model: InkJEPA,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    mode_counts = torch.zeros(3, dtype=torch.float64)
    examples = 0
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 9973 + 71)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        with autocast_context(device, args.precision):
            loss, metrics, modes = loss_for_batch(
                model,
                batch,
                device=device,
                args=args,
                generator=generator,
            )
        batch_size = len(batch["metadata"])
        values = {"loss": loss.detach(), **metrics}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        mode_counts += torch.bincount(modes.cpu(), minlength=3).double()
        examples += batch_size
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report.update(
        {
            "examples": float(examples),
            "mask_block_examples": float(mode_counts[0]),
            "future_line_examples": float(mode_counts[1]),
            "line_suffix_examples": float(mode_counts[2]),
        }
    )
    return report


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"

    records = load_visual_grammar_manifest(args.manifest)
    rights = Counter(record.rights for record in records)
    render_config = RetinalRenderConfig(
        height=args.image_height,
        width=args.image_width,
        font_size=args.font_size,
        minimum_font_size=args.minimum_font_size,
        margin=args.margin,
        augment=True,
    )
    train_dataset = VisualGrammarDataset(
        records,
        render_config=render_config,
        split="train",
        validation_fraction=args.validation_fraction,
        length=args.samples_per_epoch,
        seed=args.seed,
    )
    validation_dataset = VisualGrammarDataset(
        records,
        render_config=render_config,
        split="validation",
        validation_fraction=args.validation_fraction,
        length=args.validation_samples,
        seed=args.seed + 1_000_003,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": visual_grammar_collate,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)

    config = InkJEPAConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        representation_dim=args.representation_dim,
        encoder_layers=args.encoder_layers,
        predictor_dim=args.predictor_dim,
        predictor_layers=args.predictor_layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        pool_slots=args.pool_slots,
        dropout=args.dropout,
    )
    model = InkJEPA(config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    global_step = 0
    start_epoch = 0
    elapsed_before = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != "ink-jepa-retinal-predictive-field-v1":
            raise ValueError("resume checkpoint is not an InkJEPA retinal field")
        if checkpoint.get("model_config") != ink_jepa_config_payload(config):
            raise ValueError("resume checkpoint model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    startup = {
        "stage": "startup",
        "architecture": "ink-jepa-retinal-predictive-field-v1",
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "records": len(records),
        "train_records": len(train_dataset.records),
        "validation_records": len(validation_dataset.records),
        "planned_steps": planned_steps,
        "device": str(device),
        "rights": dict(rights),
    }
    print(json.dumps(startup, ensure_ascii=False), flush=True)
    append_jsonl(log_path, startup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"stage": "signal", "signal": signum, "action": "save_then_stop"}), flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    started = time.perf_counter()
    interval_started = started
    running: dict[str, float] = {}
    running_examples = 0
    epoch = start_epoch
    model.train()
    generator = torch.Generator(device=device).manual_seed(args.seed + global_step * 17)
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if stop_requested or (args.maximum_steps is not None and global_step >= args.maximum_steps):
                break
            global_step += 1
            learning_rate = scheduled_lr(
                global_step,
                base=args.lr,
                warmup=args.warmup_steps,
                total=planned_steps,
                minimum_ratio=args.minimum_lr_ratio,
            )
            set_lr(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss, metrics, _ = loss_for_batch(
                    model,
                    batch,
                    device=device,
                    args=args,
                    generator=generator,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            momentum = scheduled_momentum(global_step, start=args.ema_start, total=planned_steps)
            model.update_target(momentum)

            batch_size = len(batch["metadata"])
            values = {"loss": loss.detach(), **metrics}
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + float(value) * batch_size
            running_examples += batch_size

            if global_step % args.log_every == 0:
                now = time.perf_counter()
                report = {
                    "stage": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "ema_momentum": momentum,
                    "gradient_norm": float(gradient_norm),
                    "examples_per_second": running_examples / max(1e-6, now - interval_started),
                    **{key: value / max(1, running_examples) for key, value in running.items()},
                }
                if device.type == "cuda":
                    report["peak_cuda_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)
                running.clear()
                running_examples = 0
                interval_started = now

            if global_step % args.validate_every == 0:
                validation = validate(
                    model,
                    validation_loader,
                    device=device,
                    args=args,
                    step=global_step,
                )
                report = {"stage": "validation", "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    render_config=render_config,
                    args=args,
                    epoch=epoch,
                    step=global_step,
                    elapsed_seconds=elapsed,
                    rights=rights,
                )
                atomic_save(payload, output / f"checkpoint_step_{global_step:07d}.pt")
                atomic_save(payload, output / "checkpoint_latest.pt")
        if stop_requested or (args.maximum_steps is not None and global_step >= args.maximum_steps):
            break

    elapsed = elapsed_before + time.perf_counter() - started
    payload = checkpoint_payload(
        model,
        optimizer,
        render_config=render_config,
        args=args,
        epoch=epoch,
        step=global_step,
        elapsed_seconds=elapsed,
        rights=rights,
    )
    atomic_save(payload, output / "checkpoint_latest.pt")
    final = {
        "stage": "complete" if not stop_requested else "stopped",
        "step": global_step,
        "elapsed_seconds": elapsed,
        "checkpoint": str(output / "checkpoint_latest.pt"),
    }
    print(json.dumps(final), flush=True)
    append_jsonl(log_path, final)


if __name__ == "__main__":
    main()
