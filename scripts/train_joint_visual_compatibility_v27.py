#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.joint_visual_compatibility import (
    JointVisualCompatibilityConfig,
    JointVisualCompatibilityModel,
    exact_image_identity_mask,
    joint_visual_compatibility_boundary_receipt,
    joint_visual_compatibility_config_from_payload,
    joint_visual_compatibility_config_payload,
    multi_positive_nce,
    paired_assignment_loss,
    vicreg_loss,
)
from ilm.visual_lm.joint_visual_compatibility_data import (
    JointVisualNaturalDataset,
    JointVisualPairDataset,
    JointVisualRenderConfig,
    JointVisualSuffixPair,
    build_joint_suffix_pairs,
    joint_visual_data_boundary_receipt,
    joint_visual_natural_collate,
    joint_visual_natural_student_batch,
    joint_visual_pair_collate,
    joint_visual_pair_student_batch,
    joint_visual_render_config_payload,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import build_visual_character_statistics
from scripts.eval_joint_visual_compatibility_v27 import run_development_audit
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    scheduled_lr,
    seed_everything,
)


ARCHITECTURE = "joint-visual-compatibility-v27"
PROTOCOL_DOCUMENT = "references/joint_visual_compatibility_v27_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/joint_visual_compatibility.py",
    "ilm/visual_lm/joint_visual_compatibility_data.py",
    "scripts/eval_joint_visual_compatibility_v27.py",
    "scripts/train_joint_visual_compatibility_v27.py",
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_RETINA_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
FIXED_MODEL_CONFIG = JointVisualCompatibilityConfig()
FIXED_RENDER_CONFIG = JointVisualRenderConfig()
FIXED_OPTIMIZATION = {
    "steps": 8_000,
    "batch_size": 8,
    "pair_batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 3e-4,
    "retina_learning_rate": 3e-5,
    "minimum_lr_ratio": 0.10,
    "warmup": 400,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_momentum": 0.996,
    "training_pairs": 16_384,
    "context_noise_maximum": 0.05,
    "seed": 20260913,
    "dataset_seed": 20260914,
    "pair_seed": 20260914,
}
FIXED_EVIDENCE = {
    "precision": "bf16",
    "audit_windows": 2_048,
    "audit_pair_windows": 512,
    "audit_bank_size": 1_024,
    "audit_batch_size": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V27 image-only compatibility model."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--retina-checkpoint", default=DEFAULT_RETINA_CHECKPOINT)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--out", default="artifacts/joint_visual_compatibility_v27_evidence"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--retina-learning-rate", type=float, default=3e-5)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--training-pairs", type=int, default=16_384)
    parser.add_argument("--context-noise-maximum", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--dataset-seed", type=int, default=20260914)
    parser.add_argument("--pair-seed", type=int, default=20260914)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--audit-windows", type=int, default=2_048)
    parser.add_argument("--audit-pair-windows", type=int, default=512)
    parser.add_argument("--audit-bank-size", type=int, default=1_024)
    parser.add_argument("--audit-batch-size", type=int, default=32)
    parser.add_argument("--from-scratch-retina", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    return parser.parse_args()


def _require_fixed_evidence_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if args.from_scratch_retina:
        raise ValueError("V27 evidence requires the fixed V16 retina initialization")
    for name, expected in FIXED_OPTIMIZATION.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V27 evidence requires --{name.replace('_', '-')}={expected}"
            )
    for name, expected in FIXED_EVIDENCE.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V27 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    if not args.smoke:
        return output
    if not 1 <= args.smoke_steps <= 20:
        raise ValueError("V27 smoke mode permits 1--20 updates")
    output.update(
        {
            "steps": args.smoke_steps,
            "training_pairs": 32,
            "audit_windows": 8,
            "audit_pair_windows": 4,
            "audit_bank_size": 32,
            "audit_batch_size": 4,
        }
    )
    return output


def _trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _parameter_shapes(module: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]


def load_v16_retina(
    model: JointVisualCompatibilityModel,
    path: str | Path,
    *,
    require_expected_hash: bool,
) -> dict[str, Any]:
    digest = file_sha256(path)
    if require_expected_hash and digest != EXPECTED_RETINA_SHA256:
        raise ValueError(
            f"V27 evidence requires V16 retina {EXPECTED_RETINA_SHA256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "predictive-visual-field-state-flow-v1":
        raise ValueError("V27 retina source has the wrong architecture")
    state = {
        name.removeprefix("retina."): value
        for name, value in checkpoint["model"].items()
        if name.startswith("retina.")
    }
    if not state:
        raise ValueError("V16 checkpoint contains no retinal state")
    model.retina.load_state_dict(state, strict=True)
    model.target_retina.load_state_dict(state, strict=True)
    return {
        "route": "v16-online-and-ema-image-retina",
        "checkpoint": str(path),
        "sha256": digest,
        "source_architecture": checkpoint["architecture"],
        "source_step": checkpoint.get("global_step"),
        "retina_parameters": sum(
            parameter.numel() for parameter in model.retina.parameters()
        ),
    }


def _set_trainable(model: JointVisualCompatibilityModel) -> None:
    model.requires_grad_(True)
    model.target_retina.requires_grad_(False)
    model.target_candidate_projector.requires_grad_(False)
    model.train()
    model.target_retina.eval()
    model.target_candidate_projector.eval()


def _pair_receipt(pairs: Sequence[JointVisualSuffixPair]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(
            "\0".join(
                (
                    pair.suffix,
                    pair.identifier_a,
                    pair.identifier_b,
                    pair.target_a,
                    pair.target_b,
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return {
        "count": len(pairs),
        "sha256": digest.hexdigest(),
        "suffix_cells": 4,
        "all_identifiers_differ": all(
            pair.identifier_a != pair.identifier_b for pair in pairs
        ),
        "all_targets_differ": all(pair.target_a != pair.target_b for pair in pairs),
        "candidate_order_randomized_per_view": True,
        "student_receives_strings": False,
        "student_assignment_labels_are_positions": True,
    }


def _natural_loader(
    records: Sequence[Any],
    *,
    render_config: JointVisualRenderConfig,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * accumulation * batch_size
    consumed = start_step * accumulation * batch_size
    dataset = JointVisualNaturalDataset(
        records,
        split="train",
        render_config=render_config,
        seed=seed,
        length=total,
    )
    return DataLoader(
        Subset(dataset, range(consumed, total)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_natural_collate,
    )


def _pair_loader(
    pairs: Sequence[JointVisualSuffixPair],
    *,
    render_config: JointVisualRenderConfig,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * accumulation * batch_size
    consumed = start_step * accumulation * batch_size
    dataset = JointVisualPairDataset(
        pairs,
        split="train",
        render_config=render_config,
        seed=seed,
        length=total,
    )
    return DataLoader(
        Subset(dataset, range(consumed, total)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_pair_collate,
    )


def _to_device_natural(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in joint_visual_natural_student_batch(raw).items()
    }


def _to_device_pairs(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in joint_visual_pair_student_batch(raw).items()
    }


def _corrupt_context(context: torch.Tensor, *, maximum: float) -> torch.Tensor:
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("V27 context corruption maximum must be in [0,1]")
    amount = torch.rand(
        context.shape[:2], device=context.device, dtype=context.dtype
    ) * maximum
    noise = torch.randn_like(context)
    return (
        (1.0 - amount[:, :, None, None, None]) * context
        + amount[:, :, None, None, None] * noise
    )


def _natural_objective(
    model: JointVisualCompatibilityModel,
    batch: Mapping[str, torch.Tensor],
    *,
    context_noise_maximum: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    contexts = torch.cat((batch["context"], batch["reference_context"]))
    candidates = torch.cat(
        (batch["reference_target"][:, 0], batch["target"][:, 0])
    )
    canonical = torch.cat(
        (batch["canonical_target"][:, 0], batch["canonical_target"][:, 0])
    )
    contexts = _corrupt_context(contexts, maximum=context_noise_maximum)
    query = model.encode_context(contexts)
    with torch.no_grad():
        candidate_keys = model.encode_candidates(candidates, target=True)
    logits = model.compatibility_scale.float() * (
        query.float() @ candidate_keys.float().transpose(0, 1)
    )
    positives = exact_image_identity_mask(canonical)
    natural_loss, natural_metrics = multi_positive_nce(logits, positives)

    first_images = batch["target"][:, 0]
    second_images = batch["reference_target"][:, 0]
    first_online = model.encode_candidates(first_images, target=False)
    second_online = model.encode_candidates(second_images, target=False)
    with torch.no_grad():
        first_target = model.encode_candidates(first_images, target=True)
        second_target = model.encode_candidates(second_images, target=True)
    identity_positives = exact_image_identity_mask(batch["canonical_target"][:, 0])
    first_logits = model.compatibility_scale.float() * (
        first_online.float() @ second_target.float().transpose(0, 1)
    )
    second_logits = model.compatibility_scale.float() * (
        second_online.float() @ first_target.float().transpose(0, 1)
    )
    first_identity, first_identity_metrics = multi_positive_nce(
        first_logits, identity_positives
    )
    second_identity, second_identity_metrics = multi_positive_nce(
        second_logits, identity_positives
    )
    identity_loss = 0.5 * (first_identity + second_identity)

    first_latent = model.encode_candidates(
        first_images, target=False, normalize=False
    )
    second_latent = model.encode_candidates(
        second_images, target=False, normalize=False
    )
    visual_covariance, covariance_metrics = vicreg_loss(
        first_latent, second_latent
    )
    metrics = {
        "natural_loss": natural_loss.detach(),
        "natural_top1": natural_metrics["multi_positive_top1"],
        "identity_loss": identity_loss.detach(),
        "identity_top1": 0.5
        * (
            first_identity_metrics["multi_positive_top1"]
            + second_identity_metrics["multi_positive_top1"]
        ),
        **covariance_metrics,
    }
    return natural_loss, identity_loss, visual_covariance, metrics


def _pair_objective(
    model: JointVisualCompatibilityModel,
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: list[torch.Tensor] = []
    view_metrics: list[dict[str, torch.Tensor]] = []
    for context_key, candidate_key, assignment_key in (
        ("contexts", "candidates", "assignment"),
        (
            "reference_contexts",
            "reference_candidates",
            "reference_assignment",
        ),
    ):
        logits = model.score_paired_candidates(
            batch[context_key], batch[candidate_key]
        )
        loss, metrics = paired_assignment_loss(logits, batch[assignment_key])
        losses.append(loss)
        view_metrics.append(metrics)
    pair_loss = torch.stack(losses).mean()
    metrics = {
        name: torch.stack([view[name] for view in view_metrics]).mean()
        for name in view_metrics[0]
    }
    metrics["pair_loss"] = pair_loss.detach()
    return pair_loss, metrics


def training_microstep(
    model: JointVisualCompatibilityModel,
    natural: Mapping[str, torch.Tensor],
    pairs: Mapping[str, torch.Tensor],
    *,
    context_noise_maximum: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    natural_loss, identity_loss, visual_covariance, natural_metrics = (
        _natural_objective(
            model,
            natural,
            context_noise_maximum=context_noise_maximum,
        )
    )
    pair_loss, pair_metrics = _pair_objective(model, pairs)
    total = (
        natural_loss
        + 2.0 * pair_loss
        + 0.5 * identity_loss
        + 0.05 * visual_covariance
    )
    return total, {
        "loss": total.detach(),
        "compatibility_scale": model.compatibility_scale.detach(),
        **natural_metrics,
        **pair_metrics,
    }


def build_optimizer(
    model: JointVisualCompatibilityModel,
    *,
    learning_rate: float,
    retina_learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    retina_parameters = [
        parameter for parameter in model.retina.parameters() if parameter.requires_grad
    ]
    retina_ids = {id(parameter) for parameter in retina_parameters}
    main_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in retina_ids
    ]
    if not retina_parameters or not main_parameters:
        raise ValueError("V27 optimizer requires retina and compatibility parameters")
    return torch.optim.AdamW(
        [
            {
                "params": main_parameters,
                "lr": learning_rate,
                "schedule_base": learning_rate,
                "name": "compatibility",
            },
            {
                "params": retina_parameters,
                "lr": retina_learning_rate,
                "schedule_base": retina_learning_rate,
                "name": "retina",
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )


def _protocol_receipt(
    *,
    arguments: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "manifest_sha256": manifest_receipt["sha256"],
        "expected_manifest_sha256": V25_MANIFEST_SHA256,
        "expected_retina_sha256": EXPECTED_RETINA_SHA256,
        "fixed_model_config": joint_visual_compatibility_config_payload(
            FIXED_MODEL_CONFIG
        ),
        "fixed_render_config": joint_visual_render_config_payload(
            FIXED_RENDER_CONFIG
        ),
        "fixed_optimization": FIXED_OPTIMIZATION,
        "fixed_evidence": FIXED_EVIDENCE,
        "effective_arguments": dict(arguments),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    model: JointVisualCompatibilityModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    final: bool,
    smoke_only: bool,
    exploratory: bool,
    initialization: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
    partition: Mapping[str, Any],
    pair_receipt: Mapping[str, Any],
    arguments: Mapping[str, Any],
    peak_vram_gib: float,
    training_metrics: Mapping[str, float],
    training_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "step": step,
        "smoke_only": smoke_only,
        "exploratory": exploratory,
        "model_config": joint_visual_compatibility_config_payload(model.config),
        "render_config": joint_visual_render_config_payload(FIXED_RENDER_CONFIG),
        "model": model.state_dict(),
        "optimizer": None if final else optimizer.state_dict(),
        "rng_state": None if final else _rng_state(),
        "resumable": not final,
        "deployed_state_includes_optimizer": False,
        "deployed_state_includes_training_identity_images": False,
        "initialization": dict(initialization),
        "manifest": dict(manifest_receipt),
        "partition": dict(partition),
        "training_pairs": dict(pair_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": joint_visual_data_boundary_receipt(),
        "model_boundary": joint_visual_compatibility_boundary_receipt(model.config),
        "protocol": _protocol_receipt(
            arguments=arguments, manifest_receipt=manifest_receipt
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": _trainable_parameters(model),
        "trainable_parameter_shapes": _parameter_shapes(model),
        "effective_natural_view_contexts": (
            arguments["batch_size"] * 2 * arguments["gradient_accumulation"]
        ),
        "effective_pair_view_contexts": (
            arguments["pair_batch_size"]
            * 4
            * arguments["gradient_accumulation"]
        ),
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(training_metrics),
        "training_seconds": training_seconds,
        "frozen_images_instantiated": False,
    }


def _save_checkpoint(output: Path, payload: dict[str, Any], *, final: bool) -> Path:
    suffix = "final" if final else f"step_{payload['step']:07d}"
    path = output / f"checkpoint_{suffix}.pt"
    atomic_save(payload, path)
    return path


def _aggregate(total: dict[str, float], metrics: Mapping[str, torch.Tensor]) -> None:
    for name, value in metrics.items():
        total[name] = total.get(name, 0.0) + float(value)


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: JointVisualCompatibilityModel,
    *,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
) -> float:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(norm)


def _validate_resume(
    checkpoint: Mapping[str, Any],
    *,
    arguments: Mapping[str, Any],
    manifest_receipt: Mapping[str, Any],
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("resume checkpoint is not V27")
    if checkpoint.get("manifest", {}).get("sha256") != manifest_receipt["sha256"]:
        raise ValueError("V27 resume uses a different corpus")
    previous = checkpoint.get("protocol", {}).get("effective_arguments", {})
    for name in (
        "batch_size",
        "pair_batch_size",
        "gradient_accumulation",
        "precision",
        "learning_rate",
        "retina_learning_rate",
    ):
        if previous.get(name) != arguments[name]:
            raise ValueError(f"V27 resume differs for {name}")
    if not checkpoint.get("resumable"):
        raise ValueError("final V27 deployment checkpoints are not resumable")
    if checkpoint.get("optimizer") is None or checkpoint.get("rng_state") is None:
        raise ValueError("V27 resume checkpoint lacks optimizer or RNG state")


def train(
    model: JointVisualCompatibilityModel,
    records: Sequence[Any],
    pairs: Sequence[JointVisualSuffixPair],
    *,
    device: torch.device,
    precision: str,
    arguments: Mapping[str, Any],
    start_step: int,
    optimizer_state: Mapping[str, Any] | None,
    rng_state: Mapping[str, Any] | None,
    output: Path,
    checkpoint_context: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    steps = int(arguments["steps"])
    if not 0 <= start_step < steps:
        raise ValueError("V27 start step must precede the final step")
    _set_trainable(model)
    if _trainable_parameters(model) > 20_000_000:
        raise ValueError("V27 exceeds the preregistered trainable parameter cap")
    optimizer = build_optimizer(
        model,
        learning_rate=float(arguments["learning_rate"]),
        retina_learning_rate=float(arguments["retina_learning_rate"]),
        weight_decay=float(arguments["weight_decay"]),
        device=device,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))
    if rng_state is not None:
        _restore_rng_state(rng_state)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    natural_loader = _natural_loader(
        records,
        render_config=FIXED_RENDER_CONFIG,
        seed=int(arguments["dataset_seed"]),
        steps=steps,
        start_step=start_step,
        batch_size=int(arguments["batch_size"]),
        accumulation=int(arguments["gradient_accumulation"]),
        num_workers=int(arguments["num_workers"]),
    )
    pair_loader = _pair_loader(
        pairs,
        render_config=FIXED_RENDER_CONFIG,
        seed=int(arguments["pair_seed"]),
        steps=steps,
        start_step=start_step,
        batch_size=int(arguments["pair_batch_size"]),
        accumulation=int(arguments["gradient_accumulation"]),
        num_workers=int(arguments["num_workers"]),
    )
    natural_iterator = iter(natural_loader)
    pair_iterator = iter(pair_loader)
    log_path = output / "train.jsonl"
    final_metrics: dict[str, float] = {}
    stop_requested = False
    started = time.monotonic()

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for step in range(start_step + 1, steps + 1):
            main_lr = scheduled_lr(
                step,
                base=float(arguments["learning_rate"]),
                warmup=int(arguments["warmup"]),
                total=steps,
                minimum_ratio=float(arguments["minimum_lr_ratio"]),
            )
            retina_lr = scheduled_lr(
                step,
                base=float(arguments["retina_learning_rate"]),
                warmup=int(arguments["warmup"]),
                total=steps,
                minimum_ratio=float(arguments["minimum_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = (
                    retina_lr if group["name"] == "retina" else main_lr
                )
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            accumulation = int(arguments["gradient_accumulation"])
            for _ in range(accumulation):
                natural = _to_device_natural(next(natural_iterator), device)
                pair_batch = _to_device_pairs(next(pair_iterator), device)
                with autocast_context(device, precision):
                    loss, metrics = training_microstep(
                        model,
                        natural,
                        pair_batch,
                        context_noise_maximum=float(
                            arguments["context_noise_maximum"]
                        ),
                    )
                if scaler.is_enabled():
                    scaler.scale(loss / accumulation).backward()
                else:
                    (loss / accumulation).backward()
                _aggregate(totals, metrics)
            gradient_norm = _optimizer_step(
                optimizer,
                model,
                scaler=scaler,
                gradient_clip=float(arguments["gradient_clip"]),
            )
            model.update_target(float(arguments["ema_momentum"]))
            elapsed = time.monotonic() - started
            final_metrics = {
                name: value / accumulation for name, value in totals.items()
            }
            final_metrics.update(
                {
                    "step": float(step),
                    "learning_rate": main_lr,
                    "retina_learning_rate": retina_lr,
                    "gradient_norm": gradient_norm,
                    "updates_per_second": (step - start_step) / max(elapsed, 1e-9),
                }
            )
            if (
                step == 1
                or step % int(arguments["log_every"]) == 0
                or step == steps
            ):
                append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            final = step == steps
            if final or step % int(arguments["save_every"]) == 0 or stop_requested:
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    step=step,
                    final=final,
                    peak_vram_gib=peak,
                    training_metrics=final_metrics,
                    training_seconds=elapsed,
                    **checkpoint_context,
                )
                _save_checkpoint(output, payload, final=final)
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V27 stopped after checkpointing update {step}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    elapsed = time.monotonic() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    final_payload = _checkpoint_payload(
        model,
        optimizer,
        step=steps,
        final=True,
        peak_vram_gib=peak,
        training_metrics=final_metrics,
        training_seconds=elapsed,
        **checkpoint_context,
    )
    return final_metrics, final_payload


def main() -> None:
    args = parse_args()
    _require_fixed_evidence_arguments(args)
    arguments = _effective_arguments(args)
    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(arguments["seed"]))
    device = choose_device(str(arguments["device"]))
    exploratory = bool(arguments["exploratory"] or arguments["smoke"])
    if not exploratory and device.type != "cuda":
        raise ValueError("V27 evidence requires CUDA BF16 execution")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"], strict=not exploratory
    )
    records = load_v25_records(
        arguments["manifest"], strict_manifest=not exploratory
    )
    partition = visual_cell_partition_receipt(records)
    statistics = build_visual_character_statistics(records, bank_size=1_024)
    pairs = build_joint_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=int(arguments["training_pairs"]),
        seed=int(arguments["pair_seed"]),
        require_different_identifiers=True,
        allowed_targets=set(statistics.characters),
    )
    pair_receipt = _pair_receipt(pairs)

    start_step = 0
    optimizer_state = None
    rng_state = None
    if arguments["resume"]:
        checkpoint = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        _validate_resume(
            checkpoint,
            arguments=arguments,
            manifest_receipt=manifest_receipt,
        )
        model = JointVisualCompatibilityModel(
            joint_visual_compatibility_config_from_payload(
                checkpoint["model_config"]
            )
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = checkpoint["initialization"]
        start_step = int(checkpoint["step"])
        optimizer_state = checkpoint["optimizer"]
        rng_state = checkpoint["rng_state"]
    else:
        model = JointVisualCompatibilityModel(FIXED_MODEL_CONFIG)
        if arguments["from_scratch_retina"]:
            initialization = {
                "route": "from-scratch-online-and-ema-retina",
                "checkpoint": None,
                "sha256": None,
            }
        else:
            initialization = load_v16_retina(
                model,
                arguments["retina_checkpoint"],
                require_expected_hash=not exploratory,
            )
    model = model.to(device)
    checkpoint_context = {
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": exploratory,
        "initialization": initialization,
        "manifest_receipt": manifest_receipt,
        "partition": partition,
        "pair_receipt": pair_receipt,
        "arguments": arguments,
    }
    wall_started = time.monotonic()
    training_metrics, final_payload = train(
        model,
        records,
        pairs,
        device=device,
        precision=str(arguments["precision"]),
        arguments=arguments,
        start_step=start_step,
        optimizer_state=optimizer_state,
        rng_state=rng_state,
        output=output,
        checkpoint_context=checkpoint_context,
    )
    final_path = output / "checkpoint_final.pt"
    if not final_path.exists():
        _save_checkpoint(output, final_payload, final=True)
    report = run_development_audit(
        model.eval(),
        final_payload,
        manifest=str(arguments["manifest"]),
        retina_checkpoint=str(arguments["retina_checkpoint"]),
        device=device,
        precision=str(arguments["precision"]),
        batch_size=int(arguments["audit_batch_size"]),
        num_workers=int(arguments["num_workers"]),
        windows=int(arguments["audit_windows"]),
        pair_windows=int(arguments["audit_pair_windows"]),
        bank_size=int(arguments["audit_bank_size"]),
    )
    report.update(
        {
            "checkpoint": str(final_path),
            "checkpoint_sha256": file_sha256(final_path),
            "training_and_audit_seconds": time.monotonic() - wall_started,
            "training_metrics": training_metrics,
            "device": str(device),
        }
    )
    (output / "development_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
