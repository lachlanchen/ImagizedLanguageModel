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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig, load_visual_grammar_manifest
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.saccade_lm import (
    VisualSaccadeConfig,
    VisualSaccadeLM,
    visual_saccade_config_payload,
    visual_saccade_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train causal language dynamics over continuous visual fixations."
    )
    parser.add_argument("--manifest", default="data/visual_grammar/chinese_wikisource_public_domain.jsonl")
    parser.add_argument("--out", default="artifacts/visual_saccade_chinese_mvp")
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--fovea-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=30_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--visual-dim", type=int, default=192)
    parser.add_argument("--state-dim", type=int, default=384)
    parser.add_argument("--state-layers", type=int, default=3)
    parser.add_argument("--retina-base-channels", type=int, default=64)
    parser.add_argument("--ink-base-channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--visual-hypotheses", type=int, default=8)
    parser.add_argument("--maximum-contrastive", type=int, default=768)
    parser.add_argument("--visual-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.50)
    parser.add_argument("--ink-weight", type=float, default=0.45)
    parser.add_argument("--invariance-weight", type=float, default=0.20)
    parser.add_argument("--retina-contrastive-weight", type=float, default=0.30)
    parser.add_argument("--retina-variance-weight", type=float, default=0.10)
    parser.add_argument("--variance-weight", type=float, default=0.20)
    parser.add_argument("--hypothesis-diversity-weight", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-start", type=float, default=0.99)
    parser.add_argument("--ema-end", type=float, default=0.99995)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume")
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


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    progress = min(1.0, step / max(1, total))
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def architecture_name(model: VisualSaccadeLM) -> str:
    if model.config.visual_hypotheses > 1:
        return "visual-saccade-language-model-v2"
    return "visual-saccade-language-model-v1"


def initialize_compatible_state(model: VisualSaccadeLM, path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in {
        "visual-saccade-language-model-v1",
        "visual-saccade-language-model-v2",
    }:
        raise ValueError("initialization checkpoint is not a visual saccade language model")
    source_config = dict(checkpoint["model_config"])
    target_config = visual_saccade_config_payload(model.config)
    source_config.setdefault("visual_hypotheses", 1)
    for key, value in target_config.items():
        if key == "visual_hypotheses":
            continue
        if source_config.get(key) != value:
            raise ValueError(f"initialization model differs at {key}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith("visual_distribution.")
    ]
    invalid_unexpected = [
        key for key in incompatible.unexpected_keys if not key.startswith("next_visual.")
    ]
    if invalid_missing or invalid_unexpected:
        raise ValueError(
            f"incompatible initialization state: missing={invalid_missing}, unexpected={invalid_unexpected}"
        )
    return {
        "path": path,
        "architecture": checkpoint["architecture"],
        "global_step": int(checkpoint.get("global_step", 0)),
        "missing_new_parameters": len(incompatible.missing_keys),
        "discarded_old_parameters": len(incompatible.unexpected_keys),
    }


def loss_for_batch(
    model: VisualSaccadeLM,
    batch: dict[str, Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    target_ink = batch["target_ink"].to(device, non_blocking=True)
    current_reference = batch["current_reference"].to(device, non_blocking=True)
    target_reference = batch["target_reference"].to(device, non_blocking=True)
    outputs = model(context, target_reference, current_reference)
    loss, metrics = visual_saccade_loss(
        outputs,
        target_ink,
        contrastive_scale=model.contrastive_scale,
        maximum_contrastive=args.maximum_contrastive,
        visual_weight=args.visual_weight,
        contrastive_weight=args.contrastive_weight,
        ink_weight=args.ink_weight,
        invariance_weight=args.invariance_weight,
        retina_contrastive_weight=args.retina_contrastive_weight,
        retina_variance_weight=args.retina_variance_weight,
        variance_weight=args.variance_weight,
        hypothesis_diversity_weight=args.hypothesis_diversity_weight,
        generator=generator,
    )
    return loss, metrics, outputs, target_ink


def diagonal_retrieval(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = F.normalize(predicted.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    labels = torch.arange(predicted.shape[0], device=predicted.device)
    return (predicted @ target.transpose(0, 1)).argmax(dim=1).eq(labels).float().mean()


def distribution_retrieval(
    model: VisualSaccadeLM,
    prediction: dict[str, torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    scores = model.score_visual_candidates(prediction, target, position=-1)
    labels = torch.arange(scores.shape[0], device=scores.device)
    return scores.argmax(dim=1).eq(labels).float().mean()


def save_validation_samples(
    context: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    root: Path,
    step: int,
    count: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    count = min(count, context.shape[0])
    for index in range(count):
        recent = context[index, -8:, 0].float().cpu().clamp(0, 1)
        expected = target[index, -1, 0].float().cpu().clamp(0, 1)
        generated = prediction[index, -1, 0].float().cpu().sigmoid()
        cells = [*recent, expected, generated]
        canvas = torch.zeros(cells[0].shape[0], len(cells) * cells[0].shape[1])
        for cell_index, cell in enumerate(cells):
            left = cell_index * cell.shape[1]
            canvas[:, left : left + cell.shape[1]] = cell
        image = (255.0 * (1.0 - canvas)).round().byte().numpy()
        Image.fromarray(image, mode="L").save(
            root / f"step_{step:07d}_{index:02d}_context-target-prediction.png",
            optimize=True,
        )


@torch.no_grad()
def validate(
    model: VisualSaccadeLM,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    sample_root: Path,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    first_sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 99991)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        with autocast_context(device, args.precision):
            loss, metrics, outputs, target_ink = loss_for_batch(
                model,
                batch,
                device=device,
                args=args,
                generator=generator,
            )
            context = batch["context"].to(device, non_blocking=True)
            target_reference = batch["target_reference"].to(device, non_blocking=True)
            last_only = model(context[:, -1:], target_reference[:, -1:])
            full_prediction = outputs["predicted_visual"][:, -1]
            last_prediction = last_only["predicted_visual"][:, -1]
            target_visual = outputs["target_visual"][:, -1]
            full_cosine = F.cosine_similarity(full_prediction.float(), target_visual.float(), dim=-1).mean()
            last_cosine = F.cosine_similarity(last_prediction.float(), target_visual.float(), dim=-1).mean()
            ablation = {
                "full_context_last_cosine": full_cosine,
                "last_fixation_only_cosine": last_cosine,
                "context_cosine_gain": full_cosine - last_cosine,
                "full_context_batch_top1": distribution_retrieval(model, outputs, target_visual),
                "last_fixation_batch_top1": distribution_retrieval(model, last_only, target_visual),
                "context_prediction_change": (
                    1.0
                    - F.cosine_similarity(full_prediction.float(), last_prediction.float(), dim=-1).mean()
                ),
            }
        if first_sample is None:
            first_sample = (
                context[: args.sample_count].detach().cpu(),
                target_ink[: args.sample_count].detach().cpu(),
                outputs["predicted_ink_logits"][: args.sample_count].detach().cpu(),
            )
        batch_size = len(batch["metadata"])
        for key, value in {"loss": loss.detach(), **metrics, **ablation}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        examples += batch_size
    if first_sample is not None:
        save_validation_samples(
            *first_sample,
            root=sample_root,
            step=step,
            count=args.sample_count,
        )
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report["examples"] = float(examples)
    return report


def checkpoint_payload(
    model: VisualSaccadeLM,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    render_config: RetinalRenderConfig,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": architecture_name(model),
        "model_config": visual_saccade_config_payload(model.config),
        "render_config": render_config.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "student_contract": {
            "input": "ordered continuous ink foveas",
            "output": "next continuous visual state and next continuous ink fovea",
            "forbidden": ["token_ids", "unicode_ids", "character_labels", "ocr_strings", "codebooks"],
        },
    }


def main() -> None:
    args = parse_args()
    if args.resume and args.initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    records = load_visual_grammar_manifest(args.manifest)
    render_config = RetinalRenderConfig(augment=True)
    sequence_spec = SaccadeSequenceSpec(
        sequence_length=args.sequence_length,
        fovea_size=args.fovea_size,
    )
    train_dataset = VisualSaccadeDataset(
        records,
        render_config=render_config,
        spec=sequence_spec,
        split="train",
        validation_fraction=args.validation_fraction,
        length=args.samples_per_epoch,
        seed=args.seed,
    )
    validation_dataset = VisualSaccadeDataset(
        records,
        render_config=render_config,
        spec=sequence_spec,
        split="validation",
        validation_fraction=args.validation_fraction,
        length=args.validation_samples,
        seed=args.seed + 1_000_003,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "collate_fn": visual_saccade_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)
    model = VisualSaccadeLM(
        VisualSaccadeConfig(
            fovea_size=args.fovea_size,
            visual_dim=args.visual_dim,
            state_dim=args.state_dim,
            state_layers=args.state_layers,
            retina_base_channels=args.retina_base_channels,
            ink_base_channels=args.ink_base_channels,
            dropout=args.dropout,
            visual_hypotheses=args.visual_hypotheses,
        )
    ).to(device)
    initialization = (
        initialize_compatible_state(model, args.initialize_from)
        if args.initialize_from
        else None
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
        if checkpoint.get("architecture") != architecture_name(model):
            raise ValueError("resume checkpoint is not a visual saccade language model")
        if checkpoint.get("model_config") != visual_saccade_config_payload(model.config):
            raise ValueError("resume model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    startup = {
        "stage": "startup",
        "architecture": architecture_name(model),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "records": len(records),
        "train_records": len(train_dataset.records),
        "validation_records": len(validation_dataset.records),
        "sequence_length": args.sequence_length,
        "visual_hypotheses": args.visual_hypotheses,
        "initialization": initialization,
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
    model.train()
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
                loss, metrics, _, _ = loss_for_batch(
                    model,
                    batch,
                    device=device,
                    args=args,
                    generator=generator,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            momentum = ema_momentum(global_step, planned_steps, args.ema_start, args.ema_end)
            model.update_target(momentum)
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
                    sample_root=output / "samples",
                )
                report = {"stage": "validation", "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                payload = checkpoint_payload(
                    model,
                    optimizer,
                    args=args,
                    render_config=render_config,
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
        model,
        optimizer,
        args=args,
        render_config=render_config,
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
