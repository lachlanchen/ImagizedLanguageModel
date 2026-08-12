#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.folio import (
    FolioRetina,
    FolioRetinaConfig,
    folio_config_payload,
    folio_distillation_loss,
)
from ilm.visual_lm.folio_data import (
    FolioRenderConfig,
    FolioSemanticDataset,
    folio_semantic_collate,
    load_teacher_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill an ordered image-only retina into a continuous multilingual semantic field."
    )
    data = parser.add_argument_group("data")
    data.add_argument("--teacher-cache", default="data/teacher/folio_bge_m3.pt")
    data.add_argument("--validation-fraction", type=float, default=0.05)
    data.add_argument("--samples-per-epoch", type=int, default=None)
    data.add_argument("--image-height", type=int, default=192)
    data.add_argument("--image-width", type=int, default=768)
    data.add_argument("--font-size", type=int, default=28)
    data.add_argument("--minimum-font-size", type=int, default=18)
    data.add_argument("--margin", type=int, default=16)

    model = parser.add_argument_group("model")
    model.add_argument("--model-dim", type=int, default=256)
    model.add_argument("--layers", type=int, default=8)
    model.add_argument("--heads", type=int, default=8)
    model.add_argument("--mlp-ratio", type=float, default=3.0)
    model.add_argument("--dropout", type=float, default=0.0)

    train = parser.add_argument_group("optimization")
    train.add_argument("--out", default="artifacts/visual_folio")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--maximum-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--minimum-lr-ratio", type=float, default=0.05)
    train.add_argument("--warmup-steps", type=int, default=200)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--view-weight", type=float, default=0.20)
    train.add_argument("--relational-weight", type=float, default=0.50)
    train.add_argument("--contrastive-weight", type=float, default=0.20)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=729)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--validate-every", type=int, default=250)
    train.add_argument("--save-every", type=int, default=500)
    train.add_argument("--validation-batches", type=int, default=8)
    train.add_argument("--resume", default=None)
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


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def learning_rate(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float,
) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def make_checkpoint(
    model: FolioRetina,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    render_config: FolioRenderConfig,
    teacher_cache: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": "visual-folio-retina-v1",
        "model_config": folio_config_payload(model.config),
        "render_config": {
            "height": render_config.height,
            "width": render_config.width,
            "font_size": render_config.font_size,
            "minimum_font_size": render_config.minimum_font_size,
            "margin": render_config.margin,
            "augment": render_config.augment,
        },
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "teacher_provenance": {
            "model": teacher_cache.get("teacher_model"),
            "role": teacher_cache.get("teacher_role"),
            "sources": teacher_cache.get("sources", []),
        },
        "deployment_contract": {
            "input": "continuous writing image",
            "output": "continuous semantic field",
            "forbidden": ["text", "token_ids", "unicode_ids", "ocr_strings", "external_model_calls"],
            "teacher_required_at_inference": False,
        },
    }


@torch.no_grad()
def validate(
    model: FolioRetina,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    maximum_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        first_image = batch["view_a"].to(device, non_blocking=True)
        second_image = batch["view_b"].to(device, non_blocking=True)
        teacher = batch["teacher"].to(device, non_blocking=True)
        batch_size = first_image.shape[0]
        with autocast_context(device, precision):
            fields = model(torch.cat((first_image, second_image), dim=0))
            first, second = fields.split(batch_size, dim=0)
            loss, metrics = folio_distillation_loss(
                first,
                second,
                teacher,
                contrastive_scale=model.contrastive_scale,
                view_weight=args.view_weight,
                relational_weight=args.relational_weight,
                contrastive_weight=args.contrastive_weight,
            )
        values = {"loss": loss.detach(), **metrics}
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        batches += 1
        examples += batch_size
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report["examples"] = float(examples)
    report["batches"] = float(batches)
    return report


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    teacher_cache = load_teacher_cache(args.teacher_cache)
    output_dimension = int(teacher_cache["embeddings"].shape[1])
    render_config = FolioRenderConfig(
        height=args.image_height,
        width=args.image_width,
        font_size=args.font_size,
        minimum_font_size=args.minimum_font_size,
        margin=args.margin,
        augment=True,
    )
    train_dataset = FolioSemanticDataset(
        teacher_cache,
        render_config=render_config,
        split="train",
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        length=args.samples_per_epoch,
    )
    validation_dataset = FolioSemanticDataset(
        teacher_cache,
        render_config=render_config,
        split="validation",
        validation_fraction=args.validation_fraction,
        seed=args.seed + 1_000_003,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": folio_semantic_collate,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        **loader_options,
    )
    config = FolioRetinaConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        output_dim=output_dimension,
        dropout=args.dropout,
    )
    model = FolioRetina(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
        if checkpoint.get("architecture") != "visual-folio-retina-v1":
            raise ValueError("resume checkpoint is not a visual folio retina")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    steps_per_epoch = max(1, len(train_loader))
    planned_steps = args.maximum_steps or args.epochs * steps_per_epoch
    parameters = sum(parameter.numel() for parameter in model.parameters())
    startup = {
        "stage": "startup",
        "architecture": "visual-folio-retina-v1",
        "parameters": parameters,
        "train_documents": len(train_dataset.indices),
        "validation_documents": len(validation_dataset.indices),
        "teacher_dimensions": output_dimension,
        "planned_steps": planned_steps,
        "device": str(device),
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(log_path, startup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    started = time.perf_counter()
    running: dict[str, float] = {}
    running_examples = 0
    stop = False
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if args.maximum_steps is not None and global_step >= args.maximum_steps:
                stop = True
                break
            global_step += 1
            current_lr = learning_rate(
                global_step,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                total_steps=planned_steps,
                minimum_ratio=args.minimum_lr_ratio,
            )
            set_optimizer_lr(optimizer, current_lr)
            first_image = batch["view_a"].to(device, non_blocking=True)
            second_image = batch["view_b"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            batch_size = first_image.shape[0]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                fields = model(torch.cat((first_image, second_image), dim=0))
                first, second = fields.split(batch_size, dim=0)
                loss, metrics = folio_distillation_loss(
                    first,
                    second,
                    teacher,
                    contrastive_scale=model.contrastive_scale,
                    view_weight=args.view_weight,
                    relational_weight=args.relational_weight,
                    contrastive_weight=args.contrastive_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            values = {"loss": float(loss.detach()), **{key: float(value) for key, value in metrics.items()}}
            values["gradient_norm"] = float(gradient_norm)
            for key, value in values.items():
                running[key] = running.get(key, 0.0) + value * batch_size
            running_examples += batch_size

            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                report: dict[str, Any] = {
                    "stage": "train",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": current_lr,
                    "elapsed_seconds": elapsed,
                    "examples_per_second": running_examples / max(1e-9, elapsed - elapsed_before),
                }
                report.update({key: value / max(1, running_examples) for key, value in running.items()})
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)
                running = {}
                running_examples = 0
                started = time.perf_counter()
                elapsed_before = elapsed

            if global_step % args.validate_every == 0:
                validation = validate(
                    model,
                    validation_loader,
                    device=device,
                    precision=args.precision,
                    maximum_batches=args.validation_batches,
                    args=args,
                )
                report = {"stage": "validation", "epoch": epoch + 1, "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                checkpoint = make_checkpoint(
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    elapsed_seconds=elapsed,
                    args=args,
                    render_config=render_config,
                    teacher_cache=teacher_cache,
                )
                atomic_save(checkpoint, output / f"checkpoint_step_{global_step:07d}.pt")
                atomic_save(checkpoint, output / "checkpoint_latest.pt")
        if stop:
            break

    elapsed = elapsed_before + time.perf_counter() - started
    validation = validate(
        model,
        validation_loader,
        device=device,
        precision=args.precision,
        maximum_batches=args.validation_batches,
        args=args,
    )
    final_report = {
        "stage": "complete",
        "epoch": min(args.epochs, epoch + 1),
        "step": global_step,
        "elapsed_seconds": elapsed,
        "parameters": parameters,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "validation": validation,
    }
    print(json.dumps(final_report), flush=True)
    append_jsonl(log_path, final_report)
    checkpoint = make_checkpoint(
        model,
        optimizer,
        epoch=min(args.epochs, epoch + 1),
        global_step=global_step,
        elapsed_seconds=elapsed,
        args=args,
        render_config=render_config,
        teacher_cache=teacher_cache,
    )
    checkpoint["final_report"] = final_report
    atomic_save(checkpoint, output / "checkpoint_latest.pt")


if __name__ == "__main__":
    main()
