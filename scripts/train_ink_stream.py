#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ilm.visual_lm import (
    InkRibbonConfig,
    InkStreamConfig,
    InkStreamDataset,
    InkStreamLM,
    load_alpaca_records,
)
from ilm.visual_lm.ink_stream import ink_stream_config_payload, ink_stream_loss
from ilm.visual_lm.ink_stream_data import (
    ink_stream_collate,
    render_prompt_stream,
    render_qa_stream,
    strips_to_image,
)
from ilm.visual_lm.instruction_data import VisualInstructionRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal language model over continuous columns of rendered ink."
    )
    data = parser.add_argument_group("raster corpus")
    data.add_argument("--zh-data", default="data/raw/alpaca_zh.json")
    data.add_argument("--en-data", default="data/raw/alpaca_en.json")
    data.add_argument("--disable-zh", action="store_true")
    data.add_argument("--disable-en", action="store_true")
    data.add_argument("--max-records-per-language", type=int, default=10_000)
    data.add_argument("--max-prompt-chars", type=int, default=64)
    data.add_argument("--max-response-chars", type=int, default=96)
    data.add_argument("--validation-fraction", type=float, default=0.02)
    data.add_argument("--dataset-seed", type=int, default=419)
    data.add_argument("--ribbon-height", type=int, default=48)
    data.add_argument("--strip-width", type=int, default=8)
    data.add_argument("--maximum-strips", type=int, default=256)
    data.add_argument("--font-size", type=int, default=30)
    data.add_argument("--minimum-font-size", type=int, default=17)
    data.add_argument("--prefix-loss-weight", type=float, default=0.15)

    model = parser.add_argument_group("continuous visual dynamics")
    model.add_argument("--model-dim", type=int, default=256)
    model.add_argument("--layers", type=int, default=8)
    model.add_argument("--heads", type=int, default=8)
    model.add_argument("--mlp-ratio", type=float, default=3.0)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--local-motor-gain", type=float, default=0.15)

    train = parser.add_argument_group("optimization")
    train.add_argument("--out", default="artifacts/ink_stream")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=12)
    train.add_argument("--gradient-accumulation", type=int, default=1)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--warmup-steps", type=int, default=100)
    train.add_argument("--ink-weight", type=float, default=4.0)
    train.add_argument("--edge-weight", type=float, default=0.15)
    train.add_argument("--density-weight", type=float, default=0.50)
    train.add_argument("--self-feedback-probability", type=float, default=0.45)
    train.add_argument("--self-feedback-warmup", type=int, default=300)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=907)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--sample-strips", type=int, default=64)
    train.add_argument("--resume", default=None)
    train.add_argument("--maximum-steps", type=int, default=None)
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


def stable_fraction(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def split_records(
    records: Sequence[VisualInstructionRecord],
    fraction: float,
) -> tuple[list[VisualInstructionRecord], list[VisualInstructionRecord]]:
    train = [record for record in records if stable_fraction(record.identifier) >= fraction]
    validation = [record for record in records if stable_fraction(record.identifier) < fraction]
    if not validation and train:
        validation.append(train.pop())
    if not train:
        raise ValueError("training split is empty")
    return train, validation


def load_records(args: argparse.Namespace) -> tuple[list[VisualInstructionRecord], dict[str, Any]]:
    records: list[VisualInstructionRecord] = []
    sources = []
    if not args.disable_zh:
        sources.append((args.zh_data, "zh", "GPT-4-LLM alpaca_gpt4_data_zh", "CC-BY-NC-4.0"))
    if not args.disable_en:
        sources.append((args.en_data, "en", "Stanford Alpaca", "CC-BY-NC-4.0"))
    provenance = []
    for path_value, language, source, license_name in sources:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/download_alpaca.py")
        selected = load_alpaca_records(
            path,
            language=language,
            source=source,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            limit=args.max_records_per_language,
        )
        records.extend(selected)
        provenance.append(
            {
                "path": str(path),
                "source": source,
                "language": language,
                "license": license_name,
                "records": len(selected),
            }
        )
    if not records:
        raise ValueError("no instruction records selected")
    return records, {"sources": provenance}


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def infinite(loader: DataLoader, dataset: InkStreamDataset) -> Iterator[dict[str, Any]]:
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        yield from loader
        epoch += 1


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def learning_rate(step: int, total: int, warmup: int, maximum: float) -> float:
    if step < warmup:
        return maximum * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return maximum * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


@torch.no_grad()
def evaluate(
    model: InkStreamLM,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    ink_weight: float,
    edge_weight: float,
    density_weight: float,
    maximum_batches: int = 8,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "bce": 0.0,
        "edge": 0.0,
        "density": 0.0,
        "ink_f1": 0.0,
    }
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        weights = batch["weight"].to(device)
        with autocast_context(device, precision):
            logits = model(inputs)
            loss, parts = ink_stream_loss(
                logits,
                targets,
                weights,
                ink_weight=ink_weight,
                edge_weight=edge_weight,
                density_weight=density_weight,
            )
        totals["loss"] += float(loss)
        totals["bce"] += float(parts["bce"])
        totals["edge"] += float(parts["edge"])
        totals["density"] += float(parts["density"])
        totals["ink_f1"] += float(parts["ink_f1"])
        batches += 1
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def labeled_ribbons(images: Sequence[tuple[str, Image.Image]]) -> Image.Image:
    width = max(image.width for _, image in images)
    row_height = max(image.height for _, image in images) + 30
    canvas = Image.new("RGB", (width, row_height * len(images)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        y = index * row_height
        draw.text((8, y + 6), label, fill="#111827")
        canvas.paste(image.convert("RGB"), (0, y + 30))
    return canvas


@torch.no_grad()
def save_sample(
    model: InkStreamLM,
    record: VisualInstructionRecord,
    ribbon_config: InkRibbonConfig,
    *,
    path: Path,
    device: torch.device,
    precision: str,
    generation_strips: int,
    variant: int,
) -> None:
    model.eval()
    complete, _, _ = render_qa_stream(
        record.prompt,
        record.response,
        config=InkRibbonConfig(**{**ribbon_config.__dict__, "augment": False}),
        variant=variant,
    )
    prefix = render_prompt_stream(
        record.prompt,
        config=InkRibbonConfig(**{**ribbon_config.__dict__, "augment": False}),
        variant=variant,
    ).unsqueeze(0).to(device)
    with autocast_context(device, precision):
        generated = model.generate(
            prefix,
            maximum_new_strips=generation_strips,
            threshold=0.5,
            stochastic=False,
            feedback_mode="soft",
        )
    prefix_length = prefix.shape[1]
    answer_only = generated[:, prefix_length:]
    display = labeled_ribbons(
        (
            ("prompt image field", strips_to_image(prefix)),
            ("free-running model output field", strips_to_image(answer_only)),
            ("ground-truth prompt + answer field", strips_to_image(complete)),
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    display.save(path, optimize=True)
    model.train()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    records, provenance = load_records(args)
    train_records, validation_records = split_records(records, args.validation_fraction)
    ribbon_config = InkRibbonConfig(
        height=args.ribbon_height,
        strip_width=args.strip_width,
        maximum_strips=args.maximum_strips,
        font_size=args.font_size,
        minimum_font_size=args.minimum_font_size,
        prefix_loss_weight=args.prefix_loss_weight,
        augment=True,
    )
    validation_ribbon_config = InkRibbonConfig(**{**ribbon_config.__dict__, "augment": False})
    train_dataset = InkStreamDataset(train_records, config=ribbon_config, seed=args.dataset_seed)
    validation_dataset = InkStreamDataset(
        validation_records,
        config=validation_ribbon_config,
        seed=args.dataset_seed + 50_000,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": ink_stream_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=len(train_dataset) >= args.batch_size, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)

    model_config = InkStreamConfig(
        ribbon_height=args.ribbon_height,
        strip_width=args.strip_width,
        maximum_strips=args.maximum_strips,
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        local_motor_gain=args.local_motor_gain,
    )
    model = InkStreamLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    natural_steps = args.epochs * max(1, len(train_loader))
    total_steps = args.maximum_steps or natural_steps
    global_step = 0
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))

    manifest = {
        "architecture": "ink-stream-v1",
        "model_config": ink_stream_config_payload(model_config),
        "ribbon_config": ribbon_config.__dict__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "provenance": provenance,
        "student_boundary": "continuous grayscale image strips only",
        "forbidden_student_inputs": ["text", "token_ids", "unicode_ids", "ocr_strings", "visual_code_ids"],
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)

    started = time.monotonic()
    iterator = infinite(train_loader, train_dataset)
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    while global_step < total_steps:
        epoch += 1
        for _ in range(len(train_loader)):
            if global_step >= total_steps:
                break
            batch = next(iterator)
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            weights = batch["weight"].to(device, non_blocking=True)
            lr = learning_rate(global_step, total_steps, args.warmup_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast_context(device, args.precision):
                feedback_probability = args.self_feedback_probability * min(
                    1.0,
                    global_step / max(1, args.self_feedback_warmup),
                )
                if feedback_probability > 0:
                    with torch.no_grad():
                        teacher_prediction = model(inputs).sigmoid().reshape_as(inputs)
                    replacement = torch.rand(
                        inputs.shape[:2],
                        device=inputs.device,
                    ) < feedback_probability
                    replacement[:, 0] = False
                    replacement &= weights > 0
                    mixed_inputs = inputs.clone()
                    previous_prediction = teacher_prediction[:, :-1].detach()
                    mixed_inputs[:, 1:] = torch.where(
                        replacement[:, 1:, None, None, None],
                        previous_prediction,
                        inputs[:, 1:],
                    )
                else:
                    mixed_inputs = inputs
                logits = model(mixed_inputs)
                loss, parts = ink_stream_loss(
                    logits,
                    targets,
                    weights,
                    ink_weight=args.ink_weight,
                    edge_weight=args.edge_weight,
                    density_weight=args.density_weight,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (global_step + 1) % args.gradient_accumulation == 0
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                gradient_norm = torch.zeros(())
            global_step += 1

            if global_step % args.log_every == 0 or global_step == 1:
                event = {
                    "stage": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "lr": lr,
                    "gradient_norm": float(gradient_norm),
                    "self_feedback_probability": feedback_probability,
                    "elapsed_seconds": time.monotonic() - started,
                    **{key: float(value) for key, value in parts.items()},
                }
                append_jsonl(metrics_path, event)
                print(json.dumps(event), flush=True)

            if global_step % args.save_every == 0 or global_step == total_steps:
                validation = evaluate(
                    model,
                    validation_loader,
                    device=device,
                    precision=args.precision,
                    ink_weight=args.ink_weight,
                    edge_weight=args.edge_weight,
                    density_weight=args.density_weight,
                )
                event = {
                    "stage": "validation",
                    "epoch": epoch,
                    "step": global_step,
                    "elapsed_seconds": time.monotonic() - started,
                    **validation,
                }
                append_jsonl(metrics_path, event)
                print(json.dumps(event), flush=True)
                payload = {
                    "architecture": "ink-stream-v1",
                    "model_config": ink_stream_config_payload(model_config),
                    "ribbon_config": ribbon_config.__dict__,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "arguments": vars(args),
                    "provenance": provenance,
                }
                atomic_save(payload, output / "checkpoint_latest.pt")
                atomic_save(payload, output / f"checkpoint_step_{global_step:07d}.pt")
                save_sample(
                    model,
                    validation_records[0],
                    validation_ribbon_config,
                    path=output / "samples" / f"step_{global_step:07d}.png",
                    device=device,
                    precision=args.precision,
                    generation_strips=args.sample_strips,
                    variant=args.seed + global_step,
                )

    summary = {
        **manifest,
        "global_step": global_step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint": str(output / "checkpoint_latest.pt"),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
