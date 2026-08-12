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

from ilm.visual_lm.ink_jepa_data import (
    RETINAL_CJK_AVAILABLE_FONTS,
    RetinalRenderConfig,
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.retinal_flow_lm import (
    RetinalFlowConfig,
    RetinalFlowLanguageModel,
    retinal_flow_config_payload,
    retinal_flow_loss,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)


ARCHITECTURE = "retinal-flow-language-model-v1"
SACCADE_ARCHITECTURES = {
    "visual-saccade-language-model-v1",
    "visual-saccade-language-model-v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recurrent image-state dynamics with a direct pixel-space flow writer."
    )
    parser.add_argument("--manifest", default="data/visual_grammar/chinese_wikisource_public_domain.jsonl")
    parser.add_argument("--out", default="artifacts/retinal_flow_chinese_mvp")
    parser.add_argument("--initialize-from")
    parser.add_argument("--resume")
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--fovea-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=30_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--visual-dim", type=int, default=192)
    parser.add_argument("--state-dim", type=int, default=384)
    parser.add_argument("--state-layers", type=int, default=3)
    parser.add_argument("--retina-base-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--flow-base-channels", type=int, default=64)
    parser.add_argument("--flow-context-dim", type=int, default=256)
    parser.add_argument("--energy-dim", type=int, default=256)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--duplicate-similarity", type=float, default=0.90)
    parser.add_argument("--energy-positions-per-sequence", type=int, default=8)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--energy-weight", type=float, default=0.60)
    parser.add_argument("--invariance-weight", type=float, default=0.20)
    parser.add_argument("--retina-contrastive-weight", type=float, default=0.25)
    parser.add_argument("--retina-variance-weight", type=float, default=0.10)
    parser.add_argument("--candidate-invariance-weight", type=float, default=0.10)
    parser.add_argument("--writer-cycle-weight", type=float, default=0.35)
    parser.add_argument("--context-advantage-weight", type=float, default=0.25)
    parser.add_argument("--context-advantage-margin", type=float, default=0.25)
    parser.add_argument("--sampled-identity-weight", type=float, default=0.30)
    parser.add_argument("--sampled-identity-batch-size", type=int, default=8)
    parser.add_argument("--sampled-identity-steps", type=int, default=2)
    parser.add_argument("--sampled-identity-guidance-scale", type=float, default=1.5)
    parser.add_argument("--context-identity-start-step", type=int, default=800)
    parser.add_argument("--context-identity-ramp-steps", type=int, default=400)
    parser.add_argument("--rollout-batch-size", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2)
    parser.add_argument("--rollout-candidates", type=int, default=2)
    parser.add_argument("--rollout-sample-steps", type=int, default=2)
    parser.add_argument("--rollout-guidance-scale", type=float, default=1.5)
    parser.add_argument("--rollout-min-prefix", type=int, default=8)
    parser.add_argument("--rollout-state-weight", type=float, default=0.15)
    parser.add_argument("--rollout-energy-weight", type=float, default=0.35)
    parser.add_argument("--rollout-recovery-flow-weight", type=float, default=0.30)
    parser.add_argument("--rollout-start-step", type=int, default=800)
    parser.add_argument("--rollout-ramp-steps", type=int, default=400)
    parser.add_argument("--endpoint-weight", type=float, default=0.10)
    parser.add_argument("--stroke-weight", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-cycle-start-step", type=int, default=0)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-start", type=float, default=0.995)
    parser.add_argument("--ema-end", type=float, default=0.99995)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--samples-per-context", type=int, default=3)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
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
    return base * (
        minimum_ratio
        + (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def ema_momentum(step: int, total: int, start: float, end: float) -> float:
    progress = min(1.0, step / max(1, total))
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))


def rollout_weight_scale(step: int, *, start: int, ramp: int) -> float:
    if start < 0 or ramp < 0:
        raise ValueError("rollout start and ramp steps must be non-negative")
    if step < start:
        return 0.0
    if ramp == 0:
        return 1.0
    return min(1.0, (step - start + 1) / ramp)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def initialize_from_saccade(
    model: RetinalFlowLanguageModel,
    path: str,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") not in SACCADE_ARCHITECTURES:
        raise ValueError("initialization checkpoint is not a visual saccade language model")
    source_config = dict(checkpoint["model_config"])
    expected = {
        "fovea_size": model.config.fovea_size,
        "visual_dim": model.config.visual_dim,
        "state_dim": model.config.state_dim,
        "state_layers": model.config.state_layers,
        "retina_base_channels": model.config.retina_base_channels,
        "dropout": model.config.dropout,
    }
    for key, value in expected.items():
        if source_config.get(key) != value:
            raise ValueError(f"initialization model differs at {key}")
    state = checkpoint["model"]

    def component(prefix: str) -> dict[str, torch.Tensor]:
        marker = prefix + "."
        return {key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)}

    model.online_retina.load_state_dict(component("online_retina"))
    model.target_retina.load_state_dict(component("target_retina"))
    model.dynamics.load_state_dict(component("dynamics"))
    return {
        "path": path,
        "architecture": checkpoint["architecture"],
        "global_step": int(checkpoint.get("global_step", 0)),
        "copied": ["online_retina", "target_retina", "dynamics"],
        "discarded": ["next-state point/mixture head", "deterministic ink head"],
    }


def loss_for_batch(
    model: RetinalFlowLanguageModel,
    batch: dict[str, Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    generator: torch.Generator | None,
    rollout_scale: float,
    context_identity_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    target_ink = batch["target_ink"].to(device, non_blocking=True)
    current_reference = batch["current_reference"].to(device, non_blocking=True)
    target_reference = batch["target_reference"].to(device, non_blocking=True)
    outputs = model(context, target_reference, current_reference)
    loss, metrics, selected = retinal_flow_loss(
        model,
        outputs,
        context,
        target_ink,
        energy_positions_per_sequence=args.energy_positions_per_sequence,
        duplicate_similarity=args.duplicate_similarity,
        flow_weight=args.flow_weight,
        energy_weight=args.energy_weight,
        invariance_weight=args.invariance_weight,
        retina_contrastive_weight=args.retina_contrastive_weight,
        retina_variance_weight=args.retina_variance_weight,
        candidate_invariance_weight=args.candidate_invariance_weight,
        writer_cycle_weight=args.writer_cycle_weight,
        context_advantage_weight=args.context_advantage_weight,
        context_advantage_margin=args.context_advantage_margin,
        sampled_identity_weight=args.sampled_identity_weight,
        sampled_identity_batch_size=args.sampled_identity_batch_size,
        sampled_identity_steps=args.sampled_identity_steps,
        sampled_identity_guidance_scale=args.sampled_identity_guidance_scale,
        context_identity_weight_scale=context_identity_scale,
        rollout_batch_size=args.rollout_batch_size,
        rollout_steps=args.rollout_steps,
        rollout_candidates=args.rollout_candidates,
        rollout_sample_steps=args.rollout_sample_steps,
        rollout_guidance_scale=args.rollout_guidance_scale,
        rollout_min_prefix=args.rollout_min_prefix,
        rollout_state_weight=args.rollout_state_weight,
        rollout_energy_weight=args.rollout_energy_weight,
        rollout_recovery_flow_weight=args.rollout_recovery_flow_weight,
        rollout_weight_scale=rollout_scale,
        endpoint_weight=args.endpoint_weight,
        stroke_weight=args.stroke_weight,
        generator=generator,
    )
    return loss, metrics, outputs, context, target_ink


def positive_top1(scores: torch.Tensor, candidates: torch.Tensor, threshold: float) -> torch.Tensor:
    candidates = F.normalize(candidates.float(), dim=-1)
    positive = candidates @ candidates.transpose(0, 1) >= threshold
    positive.fill_diagonal_(True)
    return positive.gather(1, scores.argmax(dim=1, keepdim=True)).float().mean()


def save_samples(
    context: torch.Tensor,
    target: torch.Tensor,
    sampled: torch.Tensor,
    *,
    root: Path,
    step: int,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(context.shape[0]):
        cells = [
            *context[index, -8:, 0].float().cpu().clamp(0, 1),
            target[index, -1, 0].float().cpu().clamp(0, 1),
            *sampled[index, :, 0].float().cpu().clamp(0, 1),
        ]
        size = cells[0].shape[-1]
        canvas = torch.zeros(size, size * len(cells))
        for cell_index, cell in enumerate(cells):
            canvas[:, cell_index * size : (cell_index + 1) * size] = cell
        image = (255.0 * (1.0 - canvas)).round().byte().numpy()
        Image.fromarray(image, mode="L").save(
            root / f"step_{step:07d}_{index:02d}_context-target-samples.png",
            optimize=True,
        )


@torch.no_grad()
def validate(
    model: RetinalFlowLanguageModel,
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
    first: tuple[torch.Tensor, torch.Tensor] | None = None
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 99_991)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        with autocast_context(device, args.precision):
            loss, metrics, outputs, context, target_ink = loss_for_batch(
                model,
                batch,
                device=device,
                args=args,
                generator=generator,
                rollout_scale=rollout_weight_scale(
                    step,
                    start=args.rollout_start_step,
                    ramp=args.rollout_ramp_steps,
                ),
                context_identity_scale=rollout_weight_scale(
                    step,
                    start=args.context_identity_start_step,
                    ramp=args.context_identity_ramp_steps,
                ),
            )
            target_visual = outputs["target_visual"][:, -1]
            full_scores = model.score_visual_candidates(outputs, target_visual, position=-1)
            last = model.predict(context[:, -1:])
            last_scores = model.score_visual_candidates(last, target_visual, position=-1)
            full_top1 = positive_top1(full_scores, target_visual, args.duplicate_similarity)
            last_top1 = positive_top1(last_scores, target_visual, args.duplicate_similarity)
            full_target = full_scores.diagonal().mean()
            last_target = last_scores.diagonal().mean()
            ablation = {
                "full_context_batch_top1": full_top1,
                "last_fixation_batch_top1": last_top1,
                "context_batch_top1_gain": full_top1 - last_top1,
                "full_context_target_energy": full_target,
                "last_fixation_target_energy": last_target,
                "context_target_energy_gain": full_target - last_target,
            }
        if first is None:
            first = (
                context[: args.sample_count].detach(),
                target_ink[: args.sample_count].detach(),
            )
        batch_size = len(batch["metadata"])
        for key, value in {"loss": loss.detach(), **metrics, **ablation}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        examples += batch_size

    if first is not None:
        context, target = first
        with autocast_context(device, args.precision):
            sampled = model.sample_next(
                context,
                samples_per_context=args.samples_per_context,
                steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
        save_samples(context, target, sampled, root=sample_root, step=step)
        totals["sample_ink_fraction"] = float(sampled.mean()) * examples
        totals["sample_pixel_std"] = float(sampled.float().std()) * examples
        target_last = target[:, -1:, :, :, :]
        binary = sampled >= 0.5
        target_binary = target_last >= 0.5
        true_positive = (binary & target_binary).sum(dim=(2, 3, 4)).float()
        sample_f1 = 2.0 * true_positive / (
            binary.sum(dim=(2, 3, 4)) + target_binary.sum(dim=(2, 3, 4))
        ).clamp_min(1)
        totals["best_sample_ink_f1"] = float(sample_f1.amax(dim=1).mean()) * examples
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report["examples"] = float(examples)
    return report


def checkpoint_payload(
    model: RetinalFlowLanguageModel,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    render_config: RetinalRenderConfig,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "model_config": retinal_flow_config_payload(model.config),
        "render_config": render_config.__dict__,
        "retinal_fonts": retinal_font_manifest(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "student_contract": {
            "input": "ordered continuous ink foveas",
            "state": "recurrent continuous retinal field",
            "distribution": "conditional pixel-space rectified flow",
            "output": "continuous next-fovea image",
            "forbidden": [
                "token_ids",
                "unicode_ids",
                "character_labels",
                "ocr_strings",
                "glyph_codebooks",
                "external_language_models",
            ],
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
    model = RetinalFlowLanguageModel(
        RetinalFlowConfig(
            fovea_size=args.fovea_size,
            visual_dim=args.visual_dim,
            state_dim=args.state_dim,
            state_layers=args.state_layers,
            retina_base_channels=args.retina_base_channels,
            dropout=args.dropout,
            flow_base_channels=args.flow_base_channels,
            flow_context_dim=args.flow_context_dim,
            energy_dim=args.energy_dim,
            condition_dropout=args.condition_dropout,
        )
    ).to(device)
    initialization = (
        initialize_from_saccade(model, args.initialize_from)
        if args.initialize_from
        else None
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("resume checkpoint is not a retinal flow language model")
        if checkpoint.get("model_config") != retinal_flow_config_payload(model.config):
            raise ValueError("resume model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    if not 0 <= args.lr_cycle_start_step < planned_steps:
        raise ValueError("lr-cycle-start-step must be within the planned training interval")
    lr_cycle_steps = planned_steps - args.lr_cycle_start_step
    startup = {
        "stage": "startup",
        "architecture": ARCHITECTURE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "records": len(records),
        "train_records": len(train_dataset.records),
        "validation_records": len(validation_dataset.records),
        "sequence_length": args.sequence_length,
        "energy_positions_per_sequence": args.energy_positions_per_sequence,
        "visual_rollout": {
            "batch_size": args.rollout_batch_size,
            "steps": args.rollout_steps,
            "candidates": args.rollout_candidates,
            "sample_steps": args.rollout_sample_steps,
            "guidance_scale": args.rollout_guidance_scale,
            "min_prefix": args.rollout_min_prefix,
            "start_step": args.rollout_start_step,
            "ramp_steps": args.rollout_ramp_steps,
        },
        "context_and_sampled_identity": {
            "context_advantage_weight": args.context_advantage_weight,
            "context_advantage_margin": args.context_advantage_margin,
            "sampled_identity_weight": args.sampled_identity_weight,
            "sampled_identity_batch_size": args.sampled_identity_batch_size,
            "sampled_identity_steps": args.sampled_identity_steps,
            "sampled_identity_guidance_scale": args.sampled_identity_guidance_scale,
            "start_step": args.context_identity_start_step,
            "ramp_steps": args.context_identity_ramp_steps,
        },
        "retinal_fonts": list(RETINAL_CJK_AVAILABLE_FONTS),
        "initialization": initialization,
        "planned_steps": planned_steps,
        "lr_cycle_start_step": args.lr_cycle_start_step,
        "lr_cycle_steps": lr_cycle_steps,
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
        print(
            json.dumps({"stage": "signal", "signal": signum, "action": "save_then_stop"}),
            flush=True,
        )

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
            current_rollout_scale = rollout_weight_scale(
                global_step,
                start=args.rollout_start_step,
                ramp=args.rollout_ramp_steps,
            )
            current_context_identity_scale = rollout_weight_scale(
                global_step,
                start=args.context_identity_start_step,
                ramp=args.context_identity_ramp_steps,
            )
            learning_rate = scheduled_lr(
                max(1, global_step - args.lr_cycle_start_step),
                base=args.lr,
                warmup=args.warmup_steps,
                total=lr_cycle_steps,
                minimum_ratio=args.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss, metrics, _, _, _ = loss_for_batch(
                    model,
                    batch,
                    device=device,
                    args=args,
                    generator=generator,
                    rollout_scale=current_rollout_scale,
                    context_identity_scale=current_context_identity_scale,
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
                    "rollout_scale": current_rollout_scale,
                    "context_identity_scale": current_context_identity_scale,
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
