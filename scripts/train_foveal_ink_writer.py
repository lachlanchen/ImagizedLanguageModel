#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa import InkJEPA, ink_jepa_config_from_payload
from ilm.visual_lm.ink_jepa_data import (
    FovealContinuationDataset,
    RetinalRenderConfig,
    extract_retinal_fovea,
    foveal_continuation_collate,
    load_visual_grammar_manifest,
)
from ilm.visual_lm.ink_writer import (
    FovealInkFlow,
    FovealWriterConfig,
    flow_training_state,
    foveal_flow_loss,
    foveal_writer_config_payload,
    retinal_foveal_prediction,
    sample_foveal_ink,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a continuous next-fovea ink writer on a frozen InkJEPA foundation."
    )
    parser.add_argument("--foundation-checkpoint", required=True)
    parser.add_argument("--manifest", default="data/visual_grammar/chinese_wikisource_public_domain.jsonl")
    parser.add_argument("--out", default="artifacts/foveal_ink_writer_chinese_mvp")
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--samples-per-epoch", type=int, default=20_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--minimum-context-cells", type=int, default=8)
    parser.add_argument("--fovea-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--condition-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.10)
    parser.add_argument("--stroke-weight", type=float, default=2.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--resume", default=None)
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


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_foundation(path: str, device: torch.device) -> tuple[InkJEPA, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "ink-jepa-retinal-predictive-field-v1":
        raise ValueError("foundation checkpoint is not an InkJEPA retinal field")
    foundation = InkJEPA(ink_jepa_config_from_payload(checkpoint["model_config"]))
    foundation.load_state_dict(checkpoint["model"])
    foundation.to(device).eval().requires_grad_(False)
    return foundation, checkpoint


def render_config(checkpoint: dict[str, Any]) -> RetinalRenderConfig:
    payload = dict(checkpoint["render_config"])
    payload["augment"] = True
    return RetinalRenderConfig(**payload)


def batch_condition(
    foundation: InkJEPA,
    batch: dict[str, Any],
    *,
    device: torch.device,
    render_config: RetinalRenderConfig,
    fovea_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    hidden_mask = batch["hidden_mask"].to(device, non_blocking=True)
    target_mask = batch["target_mask"].to(device, non_blocking=True)
    with torch.no_grad():
        condition, plan_page_logits = retinal_foveal_prediction(
            foundation,
            context,
            hidden_mask,
            target_mask,
        )
        plan = torch.stack(
            [
                extract_retinal_fovea(
                    plan_page_logits[index].sigmoid(),
                    row=int(metadata["row"]),
                    column=int(metadata["column"]),
                    config=render_config,
                    fovea_size=fovea_size,
                )
                for index, metadata in enumerate(batch["metadata"])
            ]
        )
    target = batch["target"].to(device, non_blocking=True) * 2.0 - 1.0
    return condition, plan * 2.0 - 1.0, target


def loss_for_batch(
    foundation: InkJEPA,
    writer: FovealInkFlow,
    batch: dict[str, Any],
    *,
    device: torch.device,
    render_config: RetinalRenderConfig,
    args: argparse.Namespace,
    generator: torch.Generator | None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    condition, ink_plan, target = batch_condition(
        foundation,
        batch,
        device=device,
        render_config=render_config,
        fovea_size=writer.config.fovea_size,
    )
    state, velocity, time_field, _ = flow_training_state(target, generator=generator)
    keep = (
        torch.rand(target.shape[0], device=device, generator=generator) >= writer.config.condition_dropout
    ).to(target.dtype)
    prediction = writer(state, time_field, condition, ink_plan, condition_present=keep)
    loss, metrics = foveal_flow_loss(
        prediction,
        velocity,
        state,
        target,
        time_field,
        endpoint_weight=args.endpoint_weight,
        stroke_weight=args.stroke_weight,
    )
    return loss, metrics, condition, ink_plan, target


@torch.no_grad()
def validate(
    foundation: InkJEPA,
    writer: FovealInkFlow,
    loader: DataLoader,
    *,
    device: torch.device,
    render_config: RetinalRenderConfig,
    args: argparse.Namespace,
    step: int,
    sample_root: Path,
) -> dict[str, float]:
    writer.eval()
    totals: dict[str, float] = {}
    examples = 0
    first_condition: torch.Tensor | None = None
    first_plan: torch.Tensor | None = None
    first_target: torch.Tensor | None = None
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 99991)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        with autocast_context(device, args.precision):
            loss, metrics, condition, ink_plan, target = loss_for_batch(
                foundation,
                writer,
                batch,
                device=device,
                render_config=render_config,
                args=args,
                generator=generator,
            )
        if first_condition is None:
            first_condition = condition[: args.sample_count]
            first_plan = ink_plan[: args.sample_count]
            first_target = target[: args.sample_count]
        batch_size = len(batch["metadata"])
        for key, value in {"loss": loss.detach(), **metrics}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        examples += batch_size

    if first_condition is not None and first_plan is not None and first_target is not None:
        sampled = sample_foveal_ink(
            writer,
            first_condition,
            first_plan,
            steps=args.sample_steps,
            generator=generator,
        )
        from PIL import Image

        sample_root.mkdir(parents=True, exist_ok=True)
        target_images = ((first_target[:, 0].float().cpu().clamp(-1, 1) + 1.0) * 127.5).byte().numpy()
        output_images = ((sampled[:, 0].float().cpu().clamp(-1, 1) + 1.0) * 127.5).byte().numpy()
        for index, (target_image, output_image) in enumerate(zip(target_images, output_images)):
            Image.fromarray(255 - target_image, "L").save(sample_root / f"step_{step:07d}_{index:02d}_target.png")
            Image.fromarray(255 - output_image, "L").save(sample_root / f"step_{step:07d}_{index:02d}_sample.png")
    writer.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report["examples"] = float(examples)
    return report


def checkpoint_payload(
    writer: FovealInkFlow,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    foundation_checkpoint: dict[str, Any],
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": "foveal-continuous-ink-flow-v2",
        "writer_config": foveal_writer_config_payload(writer.config),
        "writer": writer.state_dict(),
        "optimizer": optimizer.state_dict(),
        "foundation_checkpoint": args.foundation_checkpoint,
        "foundation_architecture": foundation_checkpoint["architecture"],
        "foundation_step": int(foundation_checkpoint.get("global_step", 0)),
        "foundation_model_config": foundation_checkpoint["model_config"],
        "render_config": foundation_checkpoint["render_config"],
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "deployment_contract": {
            "input": "continuous retinal prediction field plus continuous coarse ink plan",
            "output": "continuous ink fovea",
            "forbidden": ["token_ids", "unicode_ids", "character_labels", "ocr_strings", "external_model_calls"],
        },
    }


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
    foundation, foundation_checkpoint = load_foundation(args.foundation_checkpoint, device)
    retinal_render = render_config(foundation_checkpoint)
    records = load_visual_grammar_manifest(args.manifest)
    train_dataset = FovealContinuationDataset(
        records,
        render_config=retinal_render,
        patch_size=foundation.config.patch_size,
        fovea_size=args.fovea_size,
        split="train",
        validation_fraction=args.validation_fraction,
        length=args.samples_per_epoch,
        minimum_context_cells=args.minimum_context_cells,
        seed=args.seed,
    )
    validation_dataset = FovealContinuationDataset(
        records,
        render_config=retinal_render,
        patch_size=foundation.config.patch_size,
        fovea_size=args.fovea_size,
        split="validation",
        validation_fraction=args.validation_fraction,
        length=args.validation_samples,
        minimum_context_cells=args.minimum_context_cells,
        seed=args.seed + 1_000_003,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "collate_fn": foveal_continuation_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)

    condition_dimension = foundation.config.representation_dim * 2
    writer = FovealInkFlow(
        FovealWriterConfig(
            fovea_size=args.fovea_size,
            condition_dim=condition_dimension,
            base_channels=args.base_channels,
            context_dim=args.context_dim,
            condition_dropout=args.condition_dropout,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        writer.parameters(),
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
        if checkpoint.get("architecture") != "foveal-continuous-ink-flow-v2":
            raise ValueError("resume checkpoint is not a foveal ink writer")
        if checkpoint.get("writer_config") != foveal_writer_config_payload(writer.config):
            raise ValueError("resume writer configuration differs")
        writer.load_state_dict(checkpoint["writer"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    startup = {
        "stage": "startup",
        "architecture": "foveal-continuous-ink-flow-v2",
        "parameters": sum(parameter.numel() for parameter in writer.parameters()),
        "frozen_foundation_parameters": sum(parameter.numel() for parameter in foundation.parameters()),
        "foundation_step": int(foundation_checkpoint.get("global_step", 0)),
        "records": len(records),
        "planned_steps": planned_steps,
        "device": str(device),
    }
    print(json.dumps(startup), flush=True)
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
    generator = torch.Generator(device=device).manual_seed(args.seed + global_step * 31)
    epoch = start_epoch
    writer.train()
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
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss, metrics, _, _, _ = loss_for_batch(
                    foundation,
                    writer,
                    batch,
                    device=device,
                    render_config=retinal_render,
                    args=args,
                    generator=generator,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(writer.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = len(batch["metadata"])
            for key, value in {"loss": loss.detach(), **metrics}.items():
                running[key] = running.get(key, 0.0) + float(value) * batch_size
            running_examples += batch_size

            if global_step % args.log_every == 0:
                now = time.perf_counter()
                report = {
                    "stage": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
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
                    foundation,
                    writer,
                    validation_loader,
                    device=device,
                    render_config=retinal_render,
                    args=args,
                    step=global_step,
                    sample_root=output / "samples",
                )
                report = {"stage": "validation", "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                payload = checkpoint_payload(
                    writer,
                    optimizer,
                    args=args,
                    foundation_checkpoint=foundation_checkpoint,
                    epoch=epoch,
                    step=global_step,
                    elapsed_seconds=elapsed,
                )
                atomic_save(payload, output / f"checkpoint_step_{global_step:07d}.pt")
                atomic_save(payload, output / "checkpoint_latest.pt")
        if stop_requested or (args.maximum_steps is not None and global_step >= args.maximum_steps):
            break

    elapsed = elapsed_before + time.perf_counter() - started
    payload = checkpoint_payload(
        writer,
        optimizer,
        args=args,
        foundation_checkpoint=foundation_checkpoint,
        epoch=epoch,
        step=global_step,
        elapsed_seconds=elapsed,
    )
    atomic_save(payload, output / "checkpoint_latest.pt")
    final = {
        "stage": "stopped" if stop_requested else "complete",
        "step": global_step,
        "elapsed_seconds": elapsed,
        "checkpoint": str(output / "checkpoint_latest.pt"),
    }
    print(json.dumps(final), flush=True)
    append_jsonl(log_path, final)


if __name__ == "__main__":
    main()
