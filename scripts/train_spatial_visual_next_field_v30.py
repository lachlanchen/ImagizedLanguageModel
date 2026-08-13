#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.spatial_visual_next_field import (
    V30_ARCHITECTURE,
    V30_GLOBAL_ROUTE,
    V30_ROUTES,
    V30_SPATIAL_ROUTE,
    SpatialVisualNextFieldConfig,
    SpatialVisualNextFieldModel,
    model_state_sha256,
    spatial_visual_next_field_boundary_receipt,
    spatial_visual_next_field_config_from_payload,
    spatial_visual_next_field_config_payload,
)
from ilm.visual_lm.spatial_visual_next_field_data import (
    ConditionalVisualCandidateBank,
    ConditionalVisualNaturalDataset,
    ConditionalVisualRenderConfig,
    build_v30_candidate_bank,
    build_v30_candidate_statistics,
    canonical_target_indices,
    conditional_visual_candidate_bank_receipt,
    spatial_visual_data_boundary_receipt,
    conditional_visual_natural_collate,
    conditional_visual_natural_student_batch,
    conditional_visual_render_config_payload,
)
from ilm.visual_lm.spatial_visual_next_field_training import (
    V30_ORDER_MARGIN,
    spatial_visual_training_microstep,
)
from ilm.visual_lm.joint_visual_compatibility_data import (
    JointVisualPairDataset,
    JointVisualRenderConfig,
    JointVisualSuffixPair,
    build_joint_suffix_pairs,
    joint_visual_pair_collate,
    joint_visual_pair_student_batch,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    file_sha256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)

ARCHITECTURE = V30_ARCHITECTURE
PROTOCOL_DOCUMENT = "references/spatial_visual_next_field_v30_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "81d2b2af1eb3a305b4acd1028c004ddddc607e826eea1d50b6d137d32ed180a5"
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_V29_CHECKPOINT = (
    "artifacts/conditional_visual_density_ratio_v29_evidence/checkpoint_final.pt"
)
EXPECTED_V29_SHA256 = "a8ec991968b577518d801090f5953406de13c688552107f26ac400fc2d508b8a"
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
SOURCE_FILES = (
    "ilm/visual_lm/spatial_visual_next_field.py",
    "ilm/visual_lm/spatial_visual_next_field_data.py",
    "ilm/visual_lm/spatial_visual_next_field_training.py",
    "scripts/train_spatial_visual_next_field_v30.py",
    "scripts/eval_spatial_visual_next_field_v30.py",
)
FIXED_RENDER_CONFIG = ConditionalVisualRenderConfig()
FIXED_PAIR_RENDER_CONFIG = JointVisualRenderConfig()
FIXED_OPTIMIZATION: dict[str, Any] = {
    "steps": 8_000,
    "batch_size": 8,
    "pair_batch_size": 8,
    "gradient_accumulation": 2,
    "decoder_learning_rate": 3e-4,
    "context_learning_rate": 6e-5,
    "minimum_lr_ratio": 0.10,
    "warmup": 400,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "seed": 20261010,
    "dataset_seed": 20261011,
    "pair_seed": 20261012,
    "candidate_bank_seed": 20261013,
    "precision": "bf16",
}
FIXED_EVIDENCE: dict[str, Any] = {
    "training_pairs": 32_768,
    "candidate_bank_size": 1_024,
    "audit_windows": 2_048,
    "audit_pair_windows": 512,
    "audit_bank_size": 1_024,
    "audit_batch_size": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one preregistered V30 next-field route."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--v29-checkpoint", default=DEFAULT_V29_CHECKPOINT)
    parser.add_argument(
        "--route",
        choices=V30_ROUTES,
        default=V30_SPATIAL_ROUTE,
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--decoder-learning-rate", type=float, default=3e-4)
    parser.add_argument("--context-learning-rate", type=float, default=6e-5)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--training-pairs", type=int, default=32_768)
    parser.add_argument("--candidate-bank-size", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--dataset-seed", type=int, default=20261011)
    parser.add_argument("--pair-seed", type=int, default=20261012)
    parser.add_argument("--candidate-bank-seed", type=int, default=20261013)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--audit-windows", type=int, default=2_048)
    parser.add_argument("--audit-pair-windows", type=int, default=512)
    parser.add_argument("--audit-bank-size", type=int, default=1_024)
    parser.add_argument("--audit-batch-size", type=int, default=16)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    return parser.parse_args()


def _require_fixed_evidence_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    for name, expected in FIXED_OPTIMIZATION.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V30 evidence requires --{name.replace('_', '-')}={expected}"
            )
    for name, expected in FIXED_EVIDENCE.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V30 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    output = vars(args).copy()
    suffix = (
        "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence"
    )
    default_output = {
        V30_SPATIAL_ROUTE: f"artifacts/spatial_visual_next_field_v30_spatial_{suffix}",
        V30_GLOBAL_ROUTE: (
            f"artifacts/spatial_visual_next_field_v30_global_control_{suffix}"
        ),
    }
    output["out"] = args.out or default_output[args.route]
    if not args.smoke:
        return output
    if not 1 <= args.smoke_steps <= 20:
        raise ValueError("V30 smoke mode permits 1-20 updates")
    output.update(
        {
            "steps": args.smoke_steps,
            "batch_size": 2,
            "pair_batch_size": 2,
            "gradient_accumulation": 1,
            "training_pairs": 32,
            "candidate_bank_size": 32,
            "audit_windows": 8,
            "audit_pair_windows": 4,
            "audit_bank_size": 32,
            "audit_batch_size": 4,
            "num_workers": min(args.num_workers, 2),
        }
    )
    return output


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
    return torch.autocast(device_type="cuda", dtype=dtype)


def scheduled_lr(
    step: int,
    *,
    base: float,
    warmup: int,
    total: int,
    minimum_ratio: float,
) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


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


def load_v29_initialization(
    model: SpatialVisualNextFieldModel,
    path: str | Path,
    *,
    require_expected_hash: bool,
) -> dict[str, Any]:
    digest = file_sha256(path)
    if require_expected_hash and digest != EXPECTED_V29_SHA256:
        raise ValueError(
            f"V30 requires V29 checkpoint {EXPECTED_V29_SHA256}, got {digest}"
        )
    source = torch.load(path, map_location="cpu", weights_only=False)
    if source.get("architecture") != "conditional-visual-density-ratio-v29":
        raise ValueError("V30 initialization source is not V29")
    source_config = source.get("model_config", {})
    for name in (
        "cell_size",
        "maximum_cells",
        "visual_dim",
        "semantic_dim",
        "model_dim",
        "layers",
        "heads",
        "mlp_ratio",
        "dropout",
        "retina_base_channels",
        "semantic_hidden_dim",
        "semantic_residual_scale",
    ):
        if source_config.get(name) != getattr(model.config, name):
            raise ValueError(f"V30 V29 source differs for {name}")
    retina_source = source.get("initialization", {}).get("retina_sha256")
    if require_expected_hash and retina_source != EXPECTED_RETINA_SHA256:
        raise ValueError("V30 V29 source has the wrong retina receipt")
    loaded = model.load_v29_backbone_state(source["model"])
    return {
        "route": "v29-spatial-backbone-transfer",
        "checkpoint": str(path),
        "sha256": digest,
        "expected_sha256": EXPECTED_V29_SHA256,
        "retina_sha256": retina_source,
        "initialized_model_state_sha256": model_state_sha256(model.state_dict()),
        **loaded,
    }


def _pair_receipt(pairs: Sequence[JointVisualSuffixPair]) -> dict[str, Any]:
    payload = [
        {
            "suffix_cells": pair.suffix_cells,
            "identifier_a": pair.identifier_a,
            "identifier_b": pair.identifier_b,
            "script_view_a": pair.script_view_a,
            "script_view_b": pair.script_view_b,
            "context_a": pair.context_a,
            "context_b": pair.context_b,
            "target_a": pair.target_a,
            "target_b": pair.target_b,
        }
        for pair in pairs
    ]
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "count": len(pairs),
        "suffix_cells": 4,
        "require_different_identifiers": True,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "student_receives_strings": False,
    }


def _candidate_manifest(
    bank: ConditionalVisualCandidateBank,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    full = conditional_visual_candidate_bank_receipt(bank, include_host_forms=True)
    path = output / "candidate_bank.json"
    atomic_write_json(full, path)
    checkpoint_receipt = {
        "manifest": "candidate_bank.json",
        "manifest_sha256": file_sha256(path),
        "bank_size": bank.size,
        "views": 2,
        "images_sha256": full["images_sha256"],
        "canonical_sha256": full["canonical_sha256"],
        "view_mapping": {
            "first_context": 1,
            "second_context": 0,
        },
        "images_in_checkpoint": False,
        "forms_in_checkpoint": False,
        "inference_requires_bank": False,
    }
    return full, checkpoint_receipt


@torch.no_grad()
def encode_training_bank(
    model: SpatialVisualNextFieldModel,
    bank: ConditionalVisualCandidateBank,
    *,
    device: torch.device,
    precision: str,
    chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    views: list[torch.Tensor] = []
    for images in bank.images:
        field_parts: list[torch.Tensor] = []
        for start in range(0, images.shape[0], chunk_size):
            batch = images[start : start + chunk_size].to(device)
            with autocast_context(device, precision):
                fields = model.encode_route_candidates(batch)
            field_parts.append(fields.detach())
        views.append(torch.cat(field_parts))
    if len(views) != 2:
        raise ValueError("V30 training bank requires two views")
    return views[0], views[1]


def _natural_loader(
    records: Sequence[Any],
    *,
    allowed_targets: set[str],
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * batch_size * accumulation
    start = start_step * batch_size * accumulation
    dataset = ConditionalVisualNaturalDataset(
        records,
        allowed_targets=allowed_targets,
        split="train",
        render_config=FIXED_RENDER_CONFIG,
        seed=seed,
        length=total,
    )
    return DataLoader(
        Subset(dataset, range(start, total)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=conditional_visual_natural_collate,
    )


def _pair_loader(
    pairs: Sequence[JointVisualSuffixPair],
    *,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total = steps * batch_size * accumulation
    start = start_step * batch_size * accumulation
    dataset = JointVisualPairDataset(
        pairs,
        split="train",
        render_config=FIXED_PAIR_RENDER_CONFIG,
        seed=seed,
        length=total,
    )
    return DataLoader(
        Subset(dataset, range(start, total)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=joint_visual_pair_collate,
    )


def _to_device_natural(
    raw: Mapping[str, Any],
    bank: ConditionalVisualCandidateBank,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    targets = canonical_target_indices(raw["canonical_target"], bank.canonical).to(
        device, non_blocking=True
    )
    student = conditional_visual_natural_student_batch(raw)
    return {
        key: value.to(device, non_blocking=True) for key, value in student.items()
    }, targets


def _to_device_pairs(
    raw: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    student = joint_visual_pair_student_batch(raw)
    return {key: value.to(device, non_blocking=True) for key, value in student.items()}


def build_optimizer(
    model: SpatialVisualNextFieldModel,
    *,
    decoder_learning_rate: float,
    context_learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    context_names = ("context_input.", "context_blocks.", "context_norm.")
    context: list[nn.Parameter] = []
    decoder: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(context_names):
            context.append(parameter)
        else:
            decoder.append(parameter)
    if not context or not decoder:
        raise ValueError("V30 optimizer requires context and decoder parameters")
    return torch.optim.AdamW(
        [
            {
                "params": context,
                "lr": context_learning_rate,
                "base_lr": context_learning_rate,
                "group_name": "v29_context",
            },
            {
                "params": decoder,
                "lr": decoder_learning_rate,
                "base_lr": decoder_learning_rate,
                "group_name": "next_field_decoder",
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
    model_config: SpatialVisualNextFieldConfig,
) -> dict[str, Any]:
    return {
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "expected_protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_files_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "manifest_sha256": manifest_receipt["sha256"],
        "expected_manifest_sha256": V25_MANIFEST_SHA256,
        "expected_v29_sha256": EXPECTED_V29_SHA256,
        "expected_retina_sha256": EXPECTED_RETINA_SHA256,
        "fixed_model_config": spatial_visual_next_field_config_payload(model_config),
        "fixed_render_config": conditional_visual_render_config_payload(
            FIXED_RENDER_CONFIG
        ),
        "fixed_pair_render_config": {
            "cell_size": FIXED_PAIR_RENDER_CONFIG.cell_size,
            "minimum_font_size": FIXED_PAIR_RENDER_CONFIG.minimum_font_size,
            "maximum_font_size": FIXED_PAIR_RENDER_CONFIG.maximum_font_size,
            "augment": FIXED_PAIR_RENDER_CONFIG.augment,
            "script_views": FIXED_PAIR_RENDER_CONFIG.script_views,
        },
        "fixed_optimization": FIXED_OPTIMIZATION,
        "fixed_evidence": FIXED_EVIDENCE,
        "fixed_order_margin": V30_ORDER_MARGIN,
        "effective_arguments": dict(arguments),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _checkpoint_payload(
    model: SpatialVisualNextFieldModel,
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
    candidate_bank_receipt: Mapping[str, Any],
    arguments: Mapping[str, Any],
    peak_vram_gib: float,
    training_metrics: Mapping[str, float],
    training_seconds: float,
) -> dict[str, Any]:
    model_state = model.state_dict()
    if any("bank" in name.lower() for name in model_state):
        raise RuntimeError("V30 model state unexpectedly contains a bank")
    return {
        "architecture": ARCHITECTURE,
        "route_mode": model.config.route_mode,
        "step": step,
        "finite_updates_verified": step,
        "smoke_only": smoke_only,
        "exploratory": exploratory,
        "model_config": spatial_visual_next_field_config_payload(model.config),
        "render_config": conditional_visual_render_config_payload(FIXED_RENDER_CONFIG),
        "model": model_state,
        "optimizer": None if final else optimizer.state_dict(),
        "rng_state": None if final else _rng_state(),
        "resumable": not final,
        "deployed_state_includes_optimizer": False,
        "deployed_state_includes_training_candidate_images": False,
        "deployed_state_includes_training_form_labels": False,
        "initialization": dict(initialization),
        "manifest": dict(manifest_receipt),
        "partition": dict(partition),
        "training_pairs": dict(pair_receipt),
        "candidate_bank_receipt": dict(candidate_bank_receipt),
        "fonts": visual_cell_font_manifest(),
        "data_boundary": spatial_visual_data_boundary_receipt(),
        "model_boundary": spatial_visual_next_field_boundary_receipt(model.config),
        "protocol": _protocol_receipt(
            arguments=arguments,
            manifest_receipt=manifest_receipt,
            model_config=model.config,
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": _trainable_parameters(model),
        "trainable_parameter_shapes": _parameter_shapes(model),
        "effective_natural_contexts": (
            arguments["batch_size"] * 2 * arguments["gradient_accumulation"]
        ),
        "effective_pair_contexts": (
            arguments["pair_batch_size"] * 4 * arguments["gradient_accumulation"]
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
    model: SpatialVisualNextFieldModel,
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
    candidate_bank_receipt: Mapping[str, Any],
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("resume checkpoint is not V30")
    if checkpoint.get("route_mode") != arguments["route"]:
        raise ValueError("V30 resume uses a different route")
    if checkpoint.get("manifest", {}).get("sha256") != manifest_receipt["sha256"]:
        raise ValueError("V30 resume uses a different corpus")
    if (
        checkpoint.get("candidate_bank_receipt", {}).get("manifest_sha256")
        != candidate_bank_receipt["manifest_sha256"]
    ):
        raise ValueError("V30 resume uses a different candidate bank")
    previous = checkpoint.get("protocol", {}).get("effective_arguments", {})
    for name in (
        "batch_size",
        "pair_batch_size",
        "gradient_accumulation",
        "precision",
        "decoder_learning_rate",
        "context_learning_rate",
        "route",
    ):
        if previous.get(name) != arguments[name]:
            raise ValueError(f"V30 resume differs for {name}")
    if not checkpoint.get("resumable"):
        raise ValueError("final V30 checkpoints are not resumable")
    if checkpoint.get("optimizer") is None or checkpoint.get("rng_state") is None:
        raise ValueError("V30 resume lacks optimizer or RNG state")
    if checkpoint.get("finite_updates_verified") != checkpoint.get("step"):
        raise ValueError("V30 resume lacks a complete finite-update receipt")


def train(
    model: SpatialVisualNextFieldModel,
    records: Sequence[Any],
    pairs: Sequence[JointVisualSuffixPair],
    bank: ConditionalVisualCandidateBank,
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
        raise ValueError("V30 start step must precede the final step")
    model.train()
    if _trainable_parameters(model) > 18_500_000:
        raise ValueError("V30 exceeds the preregistered trainable parameter cap")
    if sum(parameter.numel() for parameter in model.parameters()) > 20_000_000:
        raise ValueError("V30 exceeds the preregistered total parameter cap")
    optimizer = build_optimizer(
        model,
        decoder_learning_rate=float(arguments["decoder_learning_rate"]),
        context_learning_rate=float(arguments["context_learning_rate"]),
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
    candidate_features = encode_training_bank(
        model, bank, device=device, precision=precision
    )
    natural_loader = _natural_loader(
        records,
        allowed_targets=set(bank.forms),
        seed=int(arguments["dataset_seed"]),
        steps=steps,
        start_step=start_step,
        batch_size=int(arguments["batch_size"]),
        accumulation=int(arguments["gradient_accumulation"]),
        num_workers=int(arguments["num_workers"]),
    )
    pair_loader = _pair_loader(
        pairs,
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
            context_lr = scheduled_lr(
                step,
                base=float(arguments["context_learning_rate"]),
                warmup=int(arguments["warmup"]),
                total=steps,
                minimum_ratio=float(arguments["minimum_lr_ratio"]),
            )
            decoder_lr = scheduled_lr(
                step,
                base=float(arguments["decoder_learning_rate"]),
                warmup=int(arguments["warmup"]),
                total=steps,
                minimum_ratio=float(arguments["minimum_lr_ratio"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = (
                    context_lr if group["group_name"] == "v29_context" else decoder_lr
                )
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            accumulation = int(arguments["gradient_accumulation"])
            for _ in range(accumulation):
                natural, targets = _to_device_natural(
                    next(natural_iterator), bank, device
                )
                pair_batch = _to_device_pairs(next(pair_iterator), device)
                with autocast_context(device, precision):
                    loss, metrics = spatial_visual_training_microstep(
                        model,
                        natural,
                        targets,
                        pair_batch,
                        candidate_features,
                    )
                if not bool(torch.isfinite(loss)) or not all(
                    bool(torch.isfinite(value).all()) for value in metrics.values()
                ):
                    raise FloatingPointError(
                        f"V30 encountered a non-finite loss at update {step}"
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
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"V30 encountered a non-finite gradient at update {step}"
                )
            elapsed = time.monotonic() - started
            final_metrics = {
                name: value / accumulation for name, value in totals.items()
            }
            final_metrics.update(
                {
                    "step": float(step),
                    "context_learning_rate": context_lr,
                    "decoder_learning_rate": decoder_lr,
                    "gradient_norm": gradient_norm,
                    "updates_per_second": (step - start_step) / max(elapsed, 1e-9),
                }
            )
            if step == 1 or step % int(arguments["log_every"]) == 0 or step == steps:
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
                    f"V30 stopped after checkpointing update {step}"
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
    protocol_sha256 = file_sha256(PROTOCOL_DOCUMENT)
    if not exploratory and protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise ValueError(
            "V30 protocol changed after preregistration: "
            f"expected {EXPECTED_PROTOCOL_SHA256}, got {protocol_sha256}"
        )
    if not exploratory and (device.type != "cuda" or arguments["precision"] != "bf16"):
        raise ValueError("V30 evidence requires CUDA BF16 execution")
    if not exploratory and device.index not in (None, 0):
        raise ValueError("V30 evidence requires CUDA device 0")
    expected_output = {
        V30_SPATIAL_ROUTE: ("artifacts/spatial_visual_next_field_v30_spatial_evidence"),
        V30_GLOBAL_ROUTE: (
            "artifacts/spatial_visual_next_field_v30_global_control_evidence"
        ),
    }
    if not exploratory and Path(arguments["out"]) != Path(
        expected_output[str(arguments["route"])]
    ):
        raise ValueError(
            "V30 evidence output path differs from the preregistered route path"
        )
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(0, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"], strict=not exploratory
    )
    records = load_v25_records(arguments["manifest"], strict_manifest=not exploratory)
    partition = visual_cell_partition_receipt(records)
    statistics = build_v30_candidate_statistics(
        records, bank_size=int(arguments["candidate_bank_size"])
    )
    bank = build_v30_candidate_bank(
        statistics, seed=int(arguments["candidate_bank_seed"])
    )
    _, candidate_bank_receipt = _candidate_manifest(bank, output)
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
            candidate_bank_receipt=candidate_bank_receipt,
        )
        model = SpatialVisualNextFieldModel(
            spatial_visual_next_field_config_from_payload(checkpoint["model_config"])
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = checkpoint["initialization"]
        start_step = int(checkpoint["step"])
        optimizer_state = checkpoint["optimizer"]
        rng_state = checkpoint["rng_state"]
    else:
        model = SpatialVisualNextFieldModel(
            SpatialVisualNextFieldConfig(route_mode=str(arguments["route"]))
        )
        initialization = load_v29_initialization(
            model,
            arguments["v29_checkpoint"],
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
        "candidate_bank_receipt": candidate_bank_receipt,
        "arguments": arguments,
    }
    training_metrics, final_payload = train(
        model,
        records,
        pairs,
        bank,
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
    _save_checkpoint(output, final_payload, final=True)
    summary = {
        "architecture": ARCHITECTURE,
        "route_mode": model.config.route_mode,
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
        "initialized_model_state_sha256": initialization[
            "initialized_model_state_sha256"
        ],
        "total_parameters": final_payload["total_parameters"],
        "trainable_parameters": final_payload["trainable_parameters"],
        "peak_allocated_vram_gib": final_payload["peak_allocated_vram_gib"],
        "training_seconds": final_payload["training_seconds"],
        "training_metrics": training_metrics,
        "joint_audit_required": True,
    }
    atomic_write_json(summary, output / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
