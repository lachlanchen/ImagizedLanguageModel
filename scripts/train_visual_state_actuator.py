#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    VisualGrammarRecord,
    load_visual_grammar_manifest,
)
from ilm.visual_lm.predictive_visual_field import (
    PredictiveVisualField,
    predictive_visual_field_config_from_payload,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.visual_actuator import (
    ContinuousVisualActuator,
    VisualActuatorConfig,
    evaluate_visual_actuator_batch,
    visual_actuator_retrieval_metrics,
    visual_actuator_config_from_payload,
    visual_actuator_config_payload,
    visual_actuator_loss,
)


ARCHITECTURE = "continuous-visual-state-actuator-v1"
PVF_ARCHITECTURE = "predictive-visual-field-state-flow-v1"
SELECTION_RULE = {
    "primary": "maximize development correct_identity_top1",
    "requirements": [
        "correct_identity_top1 > shuffled_identity_top1",
        "correct_target_cosine > shuffled_target_cosine",
        "target_cosine_gain > 0.02",
        "correct_target_cosine > 0.30",
        "correct_pixel_f1 > 0.40",
    ],
    "frozen_partition_read_during_selection": False,
}
FROZEN_ACCEPTANCE_RULE = {
    "correct_identity_top1_above_shuffled": True,
    "target_cosine_gain_minimum": 0.05,
    "correct_target_cosine_minimum": 0.50,
    "correct_pixel_f1_minimum": 0.50,
    "correct_target_cosine_above_style_copy_by": 0.05,
    "human_readability_review_required": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image-only actuator from continuous retinal plans and style images "
            "to continuous ink pixels."
        )
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_state_actuator_v17")
    parser.add_argument("--holdout-fraction", type=float, default=0.06)
    parser.add_argument("--development-share", type=float, default=0.50)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=30_000)
    parser.add_argument("--development-samples", type=int, default=512)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--style-base-channels", type=int, default=32)
    parser.add_argument("--flow-base-channels", type=int, default=64)
    parser.add_argument("--flow-context-dim", type=int, default=256)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--endpoint-weight", type=float, default=0.10)
    parser.add_argument("--stroke-weight", type=float, default=2.0)
    parser.add_argument("--identity-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.25)
    parser.add_argument("--sampled-identity-weight", type=float, default=0.50)
    parser.add_argument("--sampled-pixel-weight", type=float, default=0.10)
    parser.add_argument("--sampled-batch-size", type=int, default=16)
    parser.add_argument("--sampled-steps", type=int, default=2)
    parser.add_argument("--duplicate-similarity", type=float, default=0.90)
    parser.add_argument("--logit-scale", type=float, default=12.5)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_partition_value(identifier: str) -> float:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def partition_records(
    records: Sequence[VisualGrammarRecord],
    *,
    holdout_fraction: float,
    development_share: float,
) -> tuple[list[VisualGrammarRecord], list[VisualGrammarRecord], list[VisualGrammarRecord]]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    if not 0.0 < development_share < 1.0:
        raise ValueError("development_share must be in (0, 1)")
    development_cutoff = holdout_fraction * development_share
    train: list[VisualGrammarRecord] = []
    development: list[VisualGrammarRecord] = []
    frozen: list[VisualGrammarRecord] = []
    for record in records:
        value = record_partition_value(record.identifier)
        if value >= holdout_fraction:
            train.append(record)
        elif value < development_cutoff:
            development.append(record)
        else:
            frozen.append(record)
    if not train or not development or not frozen:
        raise ValueError("visual actuator partition produced an empty split")
    return train, development, frozen


def partition_receipt(
    train: Sequence[VisualGrammarRecord],
    development: Sequence[VisualGrammarRecord],
    frozen: Sequence[VisualGrammarRecord],
    *,
    holdout_fraction: float,
    development_share: float,
) -> dict[str, Any]:
    payload = {
        "algorithm": "sha256(identifier) first 64 bits",
        "holdout_fraction": holdout_fraction,
        "development_share_of_holdout": development_share,
        "train_records": len(train),
        "development_records": len(development),
        "frozen_records": len(frozen),
        "frozen_identifiers_sha256": hashlib.sha256(
            "\n".join(sorted(record.identifier for record in frozen)).encode("utf-8")
        ).hexdigest(),
        "frozen_read_during_training": False,
    }
    return payload


def scheduled_lr(
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
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_pvf(
    path: str,
    device: torch.device,
) -> tuple[PredictiveVisualField, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != PVF_ARCHITECTURE:
        raise ValueError("visual actuator requires a Predictive Visual Field checkpoint")
    config = predictive_visual_field_config_from_payload(checkpoint["model_config"])
    model = PredictiveVisualField(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint


def select_examples(
    batch: dict[str, Any],
    *,
    positions_per_sequence: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    target = batch["target_ink"].to(device, non_blocking=True)
    semantic = batch["target_reference"].to(device, non_blocking=True)
    batch_size, length = context.shape[:2]
    if not 1 <= positions_per_sequence <= length:
        raise ValueError("positions_per_sequence must fit the visual sequence")
    scores = torch.rand(batch_size, length, device=device, generator=generator)
    positions = scores.topk(positions_per_sequence, dim=1).indices
    rows = torch.arange(batch_size, device=device)[:, None].expand_as(positions)

    def gather(images: torch.Tensor) -> torch.Tensor:
        return images[rows, positions].reshape(-1, *images.shape[2:])

    return gather(target), gather(semantic), gather(context)


def _display_tile(image: torch.Tensor, size: int) -> Image.Image:
    ink = image[0].float().cpu().clamp(0, 1).numpy()
    pixels = (255.0 * (1.0 - ink)).round().astype(np.uint8)
    return Image.fromarray(pixels, "L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def save_sample_grid(trace: dict[str, torch.Tensor], path: Path, count: int) -> None:
    rows = [
        ("target", trace["target_ink"]),
        ("semantic view", trace["semantic_reference"]),
        ("style exemplar", trace["style_reference"]),
        ("correct state", trace["correct_ink"]),
        ("shuffled state", trace["shuffled_ink"]),
    ]
    count = min(count, rows[0][1].shape[0])
    tile = 112
    label_width = 150
    margin = 16
    canvas = Image.new(
        "RGB",
        (label_width + count * tile + 2 * margin, len(rows) * tile + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, (label, images) in enumerate(rows):
        y = margin + row * tile
        draw.text((margin, y + tile // 2 - 8), label, fill="#102a43")
        for column in range(count):
            canvas.paste(
                _display_tile(images[column], tile),
                (margin + label_width + column * tile, y),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def validate(
    pvf: PredictiveVisualField,
    actuator: ContinuousVisualActuator,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    sample_root: Path,
) -> dict[str, float]:
    actuator.eval()
    totals: dict[str, float] = {}
    examples = 0
    first_trace: dict[str, torch.Tensor] | None = None
    retrieval_states: dict[str, list[torch.Tensor]] = {
        "correct_visual": [],
        "shuffled_output_visual": [],
        "intended_visual": [],
    }
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 100_003)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        target, semantic, style = select_examples(
            batch,
            positions_per_sequence=args.positions_per_sequence,
            device=device,
            generator=generator,
        )
        with autocast_context(device, args.precision):
            loss, training_metrics, _ = visual_actuator_loss(
                actuator,
                pvf.retina,
                target,
                semantic,
                style,
                endpoint_weight=args.endpoint_weight,
                stroke_weight=args.stroke_weight,
                identity_weight=args.identity_weight,
                contrastive_weight=args.contrastive_weight,
                sampled_identity_weight=args.sampled_identity_weight,
                sampled_pixel_weight=args.sampled_pixel_weight,
                sampled_batch_size=0,
                sampled_steps=args.sampled_steps,
                duplicate_similarity=args.duplicate_similarity,
                logit_scale=args.logit_scale,
                generator=generator,
            )
            causal_metrics, trace = evaluate_visual_actuator_batch(
                actuator,
                pvf.retina,
                target,
                semantic,
                style,
                steps=args.eval_steps,
                guidance_scale=args.guidance_scale,
                duplicate_similarity=args.duplicate_similarity,
                logit_scale=args.logit_scale,
                generator=generator,
            )
        if first_trace is None:
            first_trace = {key: value.detach().cpu() for key, value in trace.items()}
        for key in retrieval_states:
            retrieval_states[key].append(trace[key].detach())
        batch_examples = target.shape[0]
        metrics = {
            "loss": loss.detach(),
            **{f"endpoint_{key}": value for key, value in training_metrics.items()},
            **causal_metrics,
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_examples
        examples += batch_examples
    if first_trace is not None:
        save_sample_grid(
            first_trace,
            sample_root / f"step_{step:07d}.png",
            args.sample_count,
        )
    actuator.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    if retrieval_states["correct_visual"]:
        global_retrieval = visual_actuator_retrieval_metrics(
            torch.cat(retrieval_states["correct_visual"]),
            torch.cat(retrieval_states["shuffled_output_visual"]),
            torch.cat(retrieval_states["intended_visual"]),
            duplicate_similarity=args.duplicate_similarity,
            logit_scale=args.logit_scale,
        )
        report.update({key: float(value) for key, value in global_retrieval.items()})
    report["examples"] = float(examples)
    report["retrieval_candidates"] = float(
        sum(value.shape[0] for value in retrieval_states["intended_visual"])
    )
    return report


def checkpoint_payload(
    actuator: ContinuousVisualActuator,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    pvf_checkpoint: dict[str, Any],
    partition: dict[str, Any],
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "actuator_config": visual_actuator_config_payload(actuator.config),
        "actuator": actuator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": file_sha256(args.pvf_checkpoint),
        "pvf_architecture": pvf_checkpoint["architecture"],
        "pvf_step": int(pvf_checkpoint.get("global_step", 0)),
        "pvf_model_config": pvf_checkpoint["model_config"],
        "render_config": pvf_checkpoint["render_config"],
        "partition": partition,
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "selection_rule": SELECTION_RULE,
        "frozen_acceptance_rule": FROZEN_ACCEPTANCE_RULE,
        "student_contract": {
            "input": "continuous intended retinal state plus continuous style image",
            "output": "continuous ink pixels",
            "target_spatial_pixels_enter_condition": False,
            "student_received_token_ids": False,
            "student_received_unicode_ids": False,
            "student_received_ocr": False,
            "student_received_character_labels": False,
            "student_used_visual_codebook": False,
            "student_used_candidate_classifier": False,
            "student_used_external_language_model": False,
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

    pvf, pvf_checkpoint = load_pvf(args.pvf_checkpoint, device)
    render_config = RetinalRenderConfig(**pvf_checkpoint["render_config"])
    records = load_visual_grammar_manifest(args.manifest)
    train_records, development_records, frozen_records = partition_records(
        records,
        holdout_fraction=args.holdout_fraction,
        development_share=args.development_share,
    )
    partition = partition_receipt(
        train_records,
        development_records,
        frozen_records,
        holdout_fraction=args.holdout_fraction,
        development_share=args.development_share,
    )
    (output / "partition.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spec = SaccadeSequenceSpec(
        sequence_length=args.sequence_length,
        fovea_size=pvf.config.fovea_size,
    )
    train_dataset = VisualSaccadeDataset(
        train_records,
        render_config=render_config,
        spec=spec,
        split="all",
        length=args.samples_per_epoch,
        seed=args.seed,
    )
    development_dataset = VisualSaccadeDataset(
        development_records,
        render_config=render_config,
        spec=spec,
        split="all",
        length=args.development_samples,
        seed=args.seed + 1_000_003,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
        "collate_fn": visual_saccade_collate,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    development_loader = DataLoader(
        development_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    config = VisualActuatorConfig(
        fovea_size=pvf.config.fovea_size,
        visual_dim=pvf.config.visual_dim,
        style_dim=args.style_dim,
        style_base_channels=args.style_base_channels,
        flow_base_channels=args.flow_base_channels,
        flow_context_dim=args.flow_context_dim,
        condition_dropout=args.condition_dropout,
    )
    actuator = ContinuousVisualActuator(config).to(device)
    optimizer = torch.optim.AdamW(
        actuator.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    global_step = 0
    start_epoch = 0
    elapsed_before = 0.0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("architecture") != ARCHITECTURE:
            raise ValueError("resume checkpoint is not a continuous visual actuator")
        resume_config = visual_actuator_config_from_payload(resume["actuator_config"])
        if resume_config != config:
            raise ValueError("resume actuator configuration differs")
        if resume.get("pvf_sha256") != file_sha256(args.pvf_checkpoint):
            raise ValueError("resume uses a different Predictive Visual Field")
        actuator.load_state_dict(resume["actuator"])
        optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("global_step", 0))
        start_epoch = int(resume.get("epoch", 0))
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    startup = {
        "stage": "startup",
        "architecture": ARCHITECTURE,
        "parameters": sum(parameter.numel() for parameter in actuator.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in actuator.parameters() if parameter.requires_grad
        ),
        "frozen_pvf_parameters": sum(parameter.numel() for parameter in pvf.parameters()),
        "frozen_retina_parameters": sum(parameter.numel() for parameter in pvf.retina.parameters()),
        "classifier_parameters": 0,
        "token_embedding_parameters": 0,
        "device": str(device),
        "planned_steps": planned_steps,
        "sequence_length": args.sequence_length,
        "positions_per_sequence": args.positions_per_sequence,
        "partition": partition,
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": file_sha256(args.pvf_checkpoint),
        "selection_rule": SELECTION_RULE,
        "frozen_acceptance_rule": FROZEN_ACCEPTANCE_RULE,
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(log_path, startup)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop_requested = False

    def request_stop(signum: int, frame: object) -> None:
        del frame
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
    actuator.train()
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            if stop_requested or global_step >= planned_steps:
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
            target, semantic, style = select_examples(
                batch,
                positions_per_sequence=args.positions_per_sequence,
                device=device,
                generator=generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss, metrics, _ = visual_actuator_loss(
                    actuator,
                    pvf.retina,
                    target,
                    semantic,
                    style,
                    endpoint_weight=args.endpoint_weight,
                    stroke_weight=args.stroke_weight,
                    identity_weight=args.identity_weight,
                    contrastive_weight=args.contrastive_weight,
                    sampled_identity_weight=args.sampled_identity_weight,
                    sampled_pixel_weight=args.sampled_pixel_weight,
                    sampled_batch_size=args.sampled_batch_size,
                    sampled_steps=args.sampled_steps,
                    duplicate_similarity=args.duplicate_similarity,
                    logit_scale=args.logit_scale,
                    generator=generator,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actuator.parameters(),
                args.gradient_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            examples = target.shape[0]
            for key, value in {"loss": loss.detach(), **metrics}.items():
                running[key] = running.get(key, 0.0) + float(value) * examples
            running_examples += examples

            if global_step % args.log_every == 0:
                now = time.perf_counter()
                report = {
                    "stage": "train",
                    "step": global_step,
                    "epoch": epoch,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "examples_per_second": running_examples / max(1e-6, now - interval_started),
                    **{
                        key: value / max(1, running_examples)
                        for key, value in running.items()
                    },
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
                    pvf,
                    actuator,
                    development_loader,
                    device=device,
                    args=args,
                    step=global_step,
                    sample_root=output / "development_samples",
                )
                report = {"stage": "validation", "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                payload = checkpoint_payload(
                    actuator,
                    optimizer,
                    args=args,
                    pvf_checkpoint=pvf_checkpoint,
                    partition=partition,
                    epoch=epoch,
                    step=global_step,
                    elapsed_seconds=elapsed,
                )
                atomic_save(payload, output / f"checkpoint_step_{global_step:07d}.pt")
                atomic_save(payload, output / "checkpoint_latest.pt")
        if stop_requested or global_step >= planned_steps:
            break

    elapsed = elapsed_before + time.perf_counter() - started
    payload = checkpoint_payload(
        actuator,
        optimizer,
        args=args,
        pvf_checkpoint=pvf_checkpoint,
        partition=partition,
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
