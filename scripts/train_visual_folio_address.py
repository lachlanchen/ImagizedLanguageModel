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

from ilm.visual_lm.folio_address import (
    FolioAddressConfig,
    FolioAddressRetina,
    folio_address_config_payload,
    folio_address_loss,
    interference_addresses,
    make_interference_transform,
)
from ilm.visual_lm.folio_data import (
    FolioRenderConfig,
    FolioSemanticDataset,
    folio_semantic_collate,
    load_teacher_cache,
    semantic_residual_fields,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lossless image retina to emit continuous semantic interference addresses."
    )
    data = parser.add_argument_group("data")
    data.add_argument("--teacher-cache", required=True)
    data.add_argument("--validation-fraction", type=float, default=0.05)
    data.add_argument("--samples-per-epoch", type=int, default=None)
    data.add_argument("--image-height", type=int, default=192)
    data.add_argument("--image-width", type=int, default=768)
    data.add_argument("--font-size", type=int, default=28)
    data.add_argument("--minimum-font-size", type=int, default=18)
    data.add_argument("--margin", type=int, default=16)

    model = parser.add_argument_group("model")
    model.add_argument("--patch-size", type=int, default=8)
    model.add_argument("--model-dim", type=int, default=192)
    model.add_argument("--layers", type=int, default=6)
    model.add_argument("--heads", type=int, default=6)
    model.add_argument("--mlp-ratio", type=float, default=3.0)
    model.add_argument("--carriers", type=int, default=256)
    model.add_argument("--glance-slots", type=int, default=4)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--frequency-scale", type=float, default=2.0)

    train = parser.add_argument_group("optimization")
    train.add_argument("--out", default="artifacts/visual_folio_address")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--maximum-steps", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-workers", type=int, default=8)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    train.add_argument("--warmup-steps", type=int, default=150)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--alignment-weight", type=float, default=0.50)
    train.add_argument("--view-weight", type=float, default=0.10)
    train.add_argument("--relational-weight", type=float, default=0.10)
    train.add_argument("--contrastive-weight", type=float, default=1.0)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=1739)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--validate-every", type=int, default=250)
    train.add_argument("--save-every", type=int, default=500)
    train.add_argument("--validation-batches", type=int, default=8)
    train.add_argument("--resume", default=None)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def scheduled_lr(step: int, *, base: float, warmup: int, total: int, minimum_ratio: float) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def set_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def checkpoint_payload(
    model: FolioAddressRetina,
    optimizer: torch.optim.Optimizer,
    *,
    transform: dict[str, Any],
    render_config: FolioRenderConfig,
    teacher_cache: dict[str, Any],
    args: argparse.Namespace,
    epoch: int,
    step: int,
    elapsed: float,
) -> dict[str, Any]:
    return {
        "architecture": "visual-folio-interference-retina-v2",
        "model_config": folio_address_config_payload(model.config),
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
        "interference_transform": transform,
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed,
        "arguments": vars(args),
        "teacher_provenance": {
            "model": teacher_cache.get("teacher_model"),
            "role": teacher_cache.get("teacher_role"),
            "sources": teacher_cache.get("sources", []),
            "target_transform": "center_l2_then_random_fourier_phase_v1",
        },
        "deployment_contract": {
            "input": "continuous writing image",
            "output": "continuous phase-pair address",
            "forbidden": ["text", "token_ids", "unicode_ids", "ocr_strings", "external_model_calls"],
            "teacher_required_at_inference": False,
        },
    }


def labels_from_batch(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [int(metadata["document_index"]) for metadata in batch["metadata"]],
        device=device,
        dtype=torch.long,
    )


@torch.no_grad()
def validate(
    model: FolioAddressRetina,
    loader: DataLoader,
    *,
    target_bank: torch.Tensor,
    device: torch.device,
    precision: str,
    maximum_batches: int,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        first_image = batch["view_a"].to(device, non_blocking=True)
        second_image = batch["view_b"].to(device, non_blocking=True)
        target = batch["teacher"].to(device, non_blocking=True)
        labels = labels_from_batch(batch, device)
        batch_size = first_image.shape[0]
        with autocast_context(device, precision):
            fields = model(torch.cat((first_image, second_image), dim=0))
            first, second = fields.split(batch_size, dim=0)
            loss, metrics = folio_address_loss(
                first,
                second,
                target,
                target_bank,
                labels,
                contrastive_scale=model.contrastive_scale,
                alignment_weight=args.alignment_weight,
                view_weight=args.view_weight,
                relational_weight=args.relational_weight,
                contrastive_weight=args.contrastive_weight,
            )
        for key, value in {"loss": loss.detach(), **metrics}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        examples += batch_size
        batches += 1
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report.update({"examples": float(examples), "batches": float(batches)})
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
    cache = load_teacher_cache(args.teacher_cache)
    residuals, _ = semantic_residual_fields(cache)
    transform = make_interference_transform(
        residuals.shape[1],
        args.carriers,
        seed=args.seed + 991,
        frequency_scale=args.frequency_scale,
    )
    target_fields = interference_addresses(residuals, transform).cpu()
    render_config = FolioRenderConfig(
        height=args.image_height,
        width=args.image_width,
        font_size=args.font_size,
        minimum_font_size=args.minimum_font_size,
        margin=args.margin,
        augment=True,
    )
    train_dataset = FolioSemanticDataset(
        cache,
        render_config=render_config,
        split="train",
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        length=args.samples_per_epoch,
        target_fields=target_fields,
    )
    validation_dataset = FolioSemanticDataset(
        cache,
        render_config=render_config,
        split="validation",
        validation_fraction=args.validation_fraction,
        seed=args.seed + 1_000_003,
        target_fields=target_fields,
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
    config = FolioAddressConfig(
        image_height=args.image_height,
        image_width=args.image_width,
        patch_size=args.patch_size,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        carriers=args.carriers,
        glance_slots=args.glance_slots,
        dropout=args.dropout,
    )
    model = FolioAddressRetina(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    target_bank = target_fields.to(device)
    global_step = 0
    start_epoch = 0
    elapsed_before = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != "visual-folio-interference-retina-v2":
            raise ValueError("resume checkpoint is not an interference retina")
        if checkpoint["model_config"] != folio_address_config_payload(config):
            raise ValueError("resume checkpoint model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        transform = checkpoint["interference_transform"]
        target_fields = interference_addresses(residuals, transform).cpu()
        target_bank = target_fields.to(device)
        train_dataset.embeddings = target_fields
        validation_dataset.embeddings = target_fields
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    steps_per_epoch = max(1, len(train_loader))
    planned_steps = args.maximum_steps or args.epochs * steps_per_epoch
    parameters = sum(parameter.numel() for parameter in model.parameters())
    startup = {
        "stage": "startup",
        "architecture": "visual-folio-interference-retina-v2",
        "parameters": parameters,
        "train_documents": len(train_dataset.indices),
        "validation_documents": len(validation_dataset.indices),
        "target_documents": int(target_bank.shape[0]),
        "address_dimensions": int(target_bank.shape[1]),
        "planned_steps": planned_steps,
        "device": str(device),
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(log_path, startup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    total_started = time.perf_counter()
    interval_started = total_started
    running: dict[str, float] = {}
    running_examples = 0
    stop = False
    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if args.maximum_steps is not None and global_step >= args.maximum_steps:
                stop = True
                break
            global_step += 1
            current_lr = scheduled_lr(
                global_step,
                base=args.lr,
                warmup=args.warmup_steps,
                total=planned_steps,
                minimum_ratio=args.minimum_lr_ratio,
            )
            set_lr(optimizer, current_lr)
            first_image = batch["view_a"].to(device, non_blocking=True)
            second_image = batch["view_b"].to(device, non_blocking=True)
            target = batch["teacher"].to(device, non_blocking=True)
            labels = labels_from_batch(batch, device)
            batch_size = first_image.shape[0]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                fields = model(torch.cat((first_image, second_image), dim=0))
                first, second = fields.split(batch_size, dim=0)
                loss, metrics = folio_address_loss(
                    first,
                    second,
                    target,
                    target_bank,
                    labels,
                    contrastive_scale=model.contrastive_scale,
                    alignment_weight=args.alignment_weight,
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
                now = time.perf_counter()
                report: dict[str, Any] = {
                    "stage": "train",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": current_lr,
                    "elapsed_seconds": elapsed_before + now - total_started,
                    "examples_per_second": running_examples / max(1e-9, now - interval_started),
                }
                report.update({key: value / max(1, running_examples) for key, value in running.items()})
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)
                running = {}
                running_examples = 0
                interval_started = now

            if global_step % args.validate_every == 0:
                validation = validate(
                    model,
                    validation_loader,
                    target_bank=target_bank,
                    device=device,
                    precision=args.precision,
                    maximum_batches=args.validation_batches,
                    args=args,
                )
                report = {"stage": "validation", "epoch": epoch + 1, "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - total_started
                checkpoint = checkpoint_payload(
                    model,
                    optimizer,
                    transform=transform,
                    render_config=render_config,
                    teacher_cache=cache,
                    args=args,
                    epoch=epoch + 1,
                    step=global_step,
                    elapsed=elapsed,
                )
                atomic_save(checkpoint, output / f"checkpoint_step_{global_step:07d}.pt")
                atomic_save(checkpoint, output / "checkpoint_latest.pt")
        if stop:
            break

    elapsed = elapsed_before + time.perf_counter() - total_started
    validation = validate(
        model,
        validation_loader,
        target_bank=target_bank,
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
    checkpoint = checkpoint_payload(
        model,
        optimizer,
        transform=transform,
        render_config=render_config,
        teacher_cache=cache,
        args=args,
        epoch=min(args.epochs, epoch + 1),
        step=global_step,
        elapsed=elapsed,
    )
    checkpoint["final_report"] = final_report
    atomic_save(checkpoint, output / "checkpoint_latest.pt")


if __name__ == "__main__":
    main()
