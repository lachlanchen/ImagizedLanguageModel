#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.factorized_visual_context import (
    FactorizedVisualContextConfig,
    FactorizedVisualContextModel,
    factorized_visual_context_boundary_receipt,
    factorized_visual_context_config_from_payload,
    factorized_visual_context_config_payload,
    multi_positive_particle_contrastive_loss,
    particle_energy_score,
    suffix_pair_ranking_loss,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualNaturalDataset,
    FactorizedVisualPairDataset,
    FactorizedVisualRenderConfig,
    FactorizedVisualSuffixPair,
    build_factorized_suffix_pairs,
    factorized_visual_data_boundary_receipt,
    factorized_visual_natural_collate,
    factorized_visual_natural_student_batch,
    factorized_visual_pair_collate,
    factorized_visual_pair_student_batch,
    factorized_visual_render_config_payload,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import build_visual_character_statistics
from scripts.eval_factorized_visual_context_v26 import run_development_audit
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    scheduled_lr,
    seed_everything,
)


ARCHITECTURE = "factorized-visual-context-v26"
PROTOCOL_DOCUMENT = "references/factorized_visual_context_v26_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/factorized_visual_context.py",
    "ilm/visual_lm/factorized_visual_context_data.py",
    "scripts/eval_factorized_visual_context_v26.py",
    "scripts/train_factorized_visual_context_v26.py",
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_RETINA_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
FIXED_MODEL_CONFIG = FactorizedVisualContextConfig()
FIXED_RENDER_CONFIG = FactorizedVisualRenderConfig()
FIXED_OPTIMIZATION = {
    "steps": 8_000,
    "batch_size": 8,
    "pair_batch_size": 4,
    "gradient_accumulation": 4,
    "learning_rate": 3e-4,
    "minimum_lr_ratio": 0.10,
    "warmup": 400,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "queue_size": 8_192,
    "training_pairs": 16_384,
    "context_noise_maximum": 0.15,
    "seed": 20260909,
    "dataset_seed": 20260910,
    "pair_seed": 20260910,
}
FIXED_EVIDENCE = {
    "precision": "bf16",
    "audit_windows": 2_048,
    "audit_pair_windows": 512,
    "audit_bank_size": 1_024,
    "audit_batch_size": 32,
}


class VisualTargetQueue:
    """FIFO of detached image-retina observations used only during training."""

    def __init__(
        self,
        capacity: int,
        dimension: int,
        *,
        device: torch.device,
    ) -> None:
        if capacity < 1 or dimension < 1:
            raise ValueError("V26 queue dimensions must be positive")
        self.capacity = int(capacity)
        self.dimension = int(dimension)
        self.storage = torch.zeros(
            capacity, dimension, device=device, dtype=torch.float32
        )
        self.count = 0
        self.pointer = 0

    def candidates(self, current: torch.Tensor) -> torch.Tensor:
        if current.ndim != 2 or current.shape[1] != self.dimension:
            raise ValueError("V26 current targets do not match the queue")
        if not self.count:
            return current.detach()
        return torch.cat((current.detach(), self.storage[: self.count]), dim=0)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("V26 queue update has the wrong shape")
        values = values.detach().float()
        if values.shape[0] >= self.capacity:
            self.storage.copy_(values[-self.capacity :])
            self.count = self.capacity
            self.pointer = 0
            return
        first = min(values.shape[0], self.capacity - self.pointer)
        self.storage[self.pointer : self.pointer + first].copy_(values[:first])
        remaining = values.shape[0] - first
        if remaining:
            self.storage[:remaining].copy_(values[first:])
        self.pointer = (self.pointer + values.shape[0]) % self.capacity
        self.count = min(self.capacity, self.count + values.shape[0])

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "dimension": self.dimension,
            "storage": self.storage[: self.count].cpu(),
            "count": self.count,
            "pointer": self.pointer,
        }

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("V26 resume queue capacity differs")
        if int(state["dimension"]) != self.dimension:
            raise ValueError("V26 resume queue dimension differs")
        count = int(state["count"])
        pointer = int(state["pointer"])
        values = state["storage"]
        if not 0 <= count <= self.capacity or not 0 <= pointer < self.capacity:
            raise ValueError("V26 resume queue state is invalid")
        if tuple(values.shape) != (count, self.dimension):
            raise ValueError("V26 resume queue storage is invalid")
        self.storage.zero_()
        self.storage[:count].copy_(values.to(self.storage.device))
        self.count = count
        self.pointer = pointer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V26 factorized image-only model."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--retina-checkpoint", default=DEFAULT_RETINA_CHECKPOINT)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--out", default="artifacts/factorized_visual_context_v26_evidence"
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
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--queue-size", type=int, default=8_192)
    parser.add_argument("--training-pairs", type=int, default=16_384)
    parser.add_argument("--context-noise-maximum", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--dataset-seed", type=int, default=20260910)
    parser.add_argument("--pair-seed", type=int, default=20260910)
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
        raise ValueError("V26 evidence requires the frozen V16 image retina")
    for name, expected in FIXED_OPTIMIZATION.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V26 evidence requires --{name.replace('_', '-')}={expected}"
            )
    for name, expected in FIXED_EVIDENCE.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V26 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    if not args.smoke:
        return output
    if not 1 <= args.smoke_steps <= 20:
        raise ValueError("V26 smoke mode permits 1--20 updates")
    output.update(
        {
            "steps": args.smoke_steps,
            "training_pairs": 32,
            "queue_size": 64,
            "audit_windows": 8,
            "audit_pair_windows": 4,
            "audit_bank_size": 32,
            "audit_batch_size": 4,
        }
    )
    return output


def _trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _parameter_shapes(module: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]


def load_v16_retina(
    model: FactorizedVisualContextModel,
    path: str | Path,
    *,
    require_expected_hash: bool,
) -> dict[str, Any]:
    digest = file_sha256(path)
    if require_expected_hash and digest != EXPECTED_RETINA_SHA256:
        raise ValueError(
            f"V26 evidence requires V16 retina {EXPECTED_RETINA_SHA256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "predictive-visual-field-state-flow-v1":
        raise ValueError("V26 retina source has the wrong architecture")
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
        "route": "v16-frozen-image-retina",
        "checkpoint": str(path),
        "sha256": digest,
        "source_architecture": checkpoint["architecture"],
        "source_step": checkpoint.get("global_step"),
        "retina_parameters": sum(
            parameter.numel() for parameter in model.retina.parameters()
        ),
    }


def _set_trainable(model: FactorizedVisualContextModel) -> None:
    model.requires_grad_(True)
    model.retina.requires_grad_(not model.freeze_retina)
    model.target_retina.requires_grad_(False)
    model.train()
    model.target_retina.eval()
    if model.freeze_retina:
        model.retina.eval()


def _pair_receipt(pairs: Sequence[FactorizedVisualSuffixPair]) -> dict[str, Any]:
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
        "student_receives_strings": False,
    }


def _natural_loader(
    records: Sequence[Any],
    *,
    render_config: FactorizedVisualRenderConfig,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * accumulation * batch_size
    consumed = start_step * accumulation * batch_size
    dataset = FactorizedVisualNaturalDataset(
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
        collate_fn=factorized_visual_natural_collate,
    )


def _pair_loader(
    pairs: Sequence[FactorizedVisualSuffixPair],
    *,
    render_config: FactorizedVisualRenderConfig,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * accumulation * batch_size
    consumed = start_step * accumulation * batch_size
    dataset = FactorizedVisualPairDataset(
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
        collate_fn=factorized_visual_pair_collate,
    )


def _to_device_images(
    student: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=True) for name, value in student.items()
    }


def _bidirectional_natural(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = _to_device_images(
        factorized_visual_natural_student_batch(raw), device
    )
    return {
        "context": torch.cat((batch["context"], batch["reference_context"])),
        "independent_future": torch.cat(
            (batch["reference_future"], batch["future"])
        ),
    }


def _bidirectional_pairs(
    raw: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    batch = _to_device_images(factorized_visual_pair_student_batch(raw), device)
    return {
        "context_a": torch.cat(
            (batch["context_a"], batch["reference_context_a"])
        ),
        "target_a": torch.cat(
            (batch["reference_target_a"], batch["target_a"])
        ),
        "context_b": torch.cat(
            (batch["context_b"], batch["reference_context_b"])
        ),
        "target_b": torch.cat(
            (batch["reference_target_b"], batch["target_b"])
        ),
    }


def _corrupt_context(
    context: torch.Tensor,
    *,
    maximum: float,
) -> torch.Tensor:
    if not 0.0 <= maximum <= 1.0:
        raise ValueError("V26 context noise maximum must be in [0,1]")
    time_value = torch.rand(
        context.shape[:2], device=context.device, dtype=context.dtype
    ) * maximum
    noise = torch.randn_like(context)
    return (
        (1.0 - time_value[:, :, None, None, None]) * context
        + time_value[:, :, None, None, None] * noise
    )


def training_microstep(
    model: FactorizedVisualContextModel,
    natural: Mapping[str, torch.Tensor],
    pairs: Mapping[str, torch.Tensor],
    queue: VisualTargetQueue,
    *,
    context_noise_maximum: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    corrupted = _corrupt_context(
        natural["context"], maximum=context_noise_maximum
    )
    natural_output = model.language(corrupted)
    horizon_indices = (0, 1, 3, 7)
    with torch.no_grad():
        target_visual = model.encode_cells(
            natural["independent_future"][:, horizon_indices], target=True
        )
    energy = particle_energy_score(natural_output["particles"], target_visual)
    next_energy = energy[:, 0].mean()
    future_energy = energy[:, 1:].mean()
    current_target = target_visual[:, 0]
    candidates = queue.candidates(current_target)
    contrastive, contrastive_metrics = multi_positive_particle_contrastive_loss(
        natural_output["particles"][:, 0],
        current_target,
        candidates,
        scale=model.contrastive_scale,
    )

    pair_count = pairs["context_a"].shape[0]
    common_noise = torch.randn(
        pair_count,
        1,
        model.config.particle_count,
        model.config.particle_noise_dim,
        device=pairs["context_a"].device,
        dtype=pairs["context_a"].dtype,
    )
    output_a = model.language(
        pairs["context_a"], horizons=(1,), particle_noise=common_noise
    )
    output_b = model.language(
        pairs["context_b"], horizons=(1,), particle_noise=common_noise
    )
    with torch.no_grad():
        target_a = model.encode_cells(pairs["target_a"], target=True)[:, 0]
        target_b = model.encode_cells(pairs["target_b"], target=True)[:, 0]
    pair_loss, pair_metrics = suffix_pair_ranking_loss(
        output_a["particles"][:, 0],
        output_b["particles"][:, 0],
        target_a,
        target_b,
        margin=0.10,
    )
    total = next_energy + 0.5 * future_energy + contrastive + pair_loss
    nearest_cosine = torch.einsum(
        "bkd,bd->bk",
        natural_output["particles"][:, 0].float(),
        current_target.float(),
    ).amax(dim=1).mean()
    metrics = {
        "loss": total.detach(),
        "next_energy_score": next_energy.detach(),
        "future_energy_score": future_energy.detach(),
        "nearest_target_cosine": nearest_cosine.detach(),
        **contrastive_metrics,
        **pair_metrics,
    }
    return total, metrics, current_target.detach()


def _optimizer(
    model: FactorizedVisualContextModel,
    *,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("V26 has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
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
        "fixed_model_config": factorized_visual_context_config_payload(
            FIXED_MODEL_CONFIG
        ),
        "fixed_render_config": factorized_visual_render_config_payload(
            FIXED_RENDER_CONFIG
        ),
        "fixed_optimization": FIXED_OPTIMIZATION,
        "fixed_evidence": FIXED_EVIDENCE,
        "effective_arguments": dict(arguments),
    }


def _checkpoint_payload(
    model: FactorizedVisualContextModel,
    optimizer: torch.optim.Optimizer,
    queue: VisualTargetQueue,
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
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "step": step,
        "smoke_only": smoke_only,
        "exploratory": exploratory,
        "model_config": factorized_visual_context_config_payload(model.config),
        "render_config": factorized_visual_render_config_payload(
            FIXED_RENDER_CONFIG
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_queue": None if final else queue.state_dict(),
        "training_queue_present": not final,
        "deployed_state_includes_training_queue": False,
        "initialization": dict(initialization),
        "manifest": dict(manifest_receipt),
        "partition": dict(partition),
        "training_pairs": dict(pair_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": factorized_visual_data_boundary_receipt(),
        "model_boundary": factorized_visual_context_boundary_receipt(model.config),
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
    model: FactorizedVisualContextModel,
    *,
    scaler: torch.amp.GradScaler,
    gradient_clip: float,
) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
        raise ValueError("resume checkpoint is not V26")
    if checkpoint.get("manifest", {}).get("sha256") != manifest_receipt["sha256"]:
        raise ValueError("V26 resume uses a different corpus")
    previous = checkpoint.get("protocol", {}).get("effective_arguments", {})
    for name in (
        "batch_size",
        "pair_batch_size",
        "gradient_accumulation",
        "precision",
        "queue_size",
    ):
        if previous.get(name) != arguments[name]:
            raise ValueError(f"V26 resume differs for {name}")
    if checkpoint.get("training_queue") is None:
        raise ValueError("final deployment checkpoints are not resumable")


def train(
    model: FactorizedVisualContextModel,
    records: Sequence[Any],
    pairs: Sequence[FactorizedVisualSuffixPair],
    *,
    device: torch.device,
    precision: str,
    arguments: Mapping[str, Any],
    start_step: int,
    optimizer_state: Mapping[str, Any] | None,
    queue_state: Mapping[str, Any] | None,
    output: Path,
    checkpoint_context: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    steps = int(arguments["steps"])
    if not 0 <= start_step < steps:
        raise ValueError("V26 start step must precede the final step")
    _set_trainable(model)
    if _trainable_parameters(model) > 20_000_000:
        raise ValueError("V26 exceeds the preregistered trainable parameter cap")
    optimizer = _optimizer(
        model,
        learning_rate=float(arguments["learning_rate"]),
        weight_decay=float(arguments["weight_decay"]),
        device=device,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))
    queue = VisualTargetQueue(
        int(arguments["queue_size"]), model.config.visual_dim, device=device
    )
    if queue_state is not None:
        queue.load_state_dict(queue_state)
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

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for step in range(start_step + 1, steps + 1):
            lr = scheduled_lr(
                step,
                base=float(arguments["learning_rate"]),
                warmup=int(arguments["warmup"]),
                total=steps,
                minimum_ratio=float(arguments["minimum_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            accumulation = int(arguments["gradient_accumulation"])
            for _ in range(accumulation):
                natural = _bidirectional_natural(next(natural_iterator), device)
                pair_batch = _bidirectional_pairs(next(pair_iterator), device)
                with autocast_context(device, precision):
                    loss, metrics, queue_values = training_microstep(
                        model,
                        natural,
                        pair_batch,
                        queue,
                        context_noise_maximum=float(
                            arguments["context_noise_maximum"]
                        ),
                    )
                if scaler.is_enabled():
                    scaler.scale(loss / accumulation).backward()
                else:
                    (loss / accumulation).backward()
                queue.update(queue_values)
                _aggregate(totals, metrics)
            gradient_norm = _optimizer_step(
                optimizer,
                model,
                scaler=scaler,
                gradient_clip=float(arguments["gradient_clip"]),
            )
            if not model.freeze_retina:
                model.update_target_retina(momentum=0.996)
            final_metrics = {
                name: value / accumulation for name, value in totals.items()
            }
            final_metrics.update(
                {
                    "step": float(step),
                    "learning_rate": lr,
                    "gradient_norm": gradient_norm,
                    "queue_count": float(queue.count),
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
                    queue,
                    step=step,
                    final=final,
                    peak_vram_gib=peak,
                    training_metrics=final_metrics,
                    **checkpoint_context,
                )
                _save_checkpoint(output, payload, final=final)
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V26 stopped after checkpointing update {step}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0.0
    )
    final_payload = _checkpoint_payload(
        model,
        optimizer,
        queue,
        step=steps,
        final=True,
        peak_vram_gib=peak,
        training_metrics=final_metrics,
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
        raise ValueError("V26 evidence requires CUDA BF16 execution")
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
    pairs = build_factorized_suffix_pairs(
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
    queue_state = None
    if arguments["resume"]:
        checkpoint = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        _validate_resume(
            checkpoint,
            arguments=arguments,
            manifest_receipt=manifest_receipt,
        )
        model = FactorizedVisualContextModel(
            factorized_visual_context_config_from_payload(
                checkpoint["model_config"]
            ),
            freeze_retina=not bool(arguments["from_scratch_retina"]),
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = checkpoint["initialization"]
        start_step = int(checkpoint["step"])
        optimizer_state = checkpoint["optimizer"]
        queue_state = checkpoint["training_queue"]
    else:
        model = FactorizedVisualContextModel(
            FIXED_MODEL_CONFIG,
            freeze_retina=not bool(arguments["from_scratch_retina"]),
        )
        if arguments["from_scratch_retina"]:
            initialization = {
                "route": "from-scratch-ema-retina",
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
    started = time.monotonic()
    training_metrics, final_payload = train(
        model,
        records,
        pairs,
        device=device,
        precision=str(arguments["precision"]),
        arguments=arguments,
        start_step=start_step,
        optimizer_state=optimizer_state,
        queue_state=queue_state,
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
            "training_seconds": time.monotonic() - started,
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
