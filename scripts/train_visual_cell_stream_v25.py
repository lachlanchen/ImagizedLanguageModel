#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    VisualCellRenderConfig,
    VisualCellStreamDataset,
    load_v25_records,
    student_visual_cell_batch,
    verify_v25_manifest,
    visual_cell_boundary_receipt,
    visual_cell_collate,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
    visual_cell_render_config_payload,
)
from ilm.visual_lm.visual_cell_stream import (
    VisualCellStreamConfig,
    VisualCellStreamModel,
    visual_cell_flow_loss,
    visual_cell_language_loss,
    visual_cell_model_boundary_receipt,
    visual_cell_model_config_from_payload,
    visual_cell_model_config_payload,
)
from scripts.eval_visual_cell_stream_development_v25 import run_development_audit
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    scheduled_lr,
    seed_everything,
)


ARCHITECTURE = "visual-cell-stream-v25"
PROTOCOL_DOCUMENT = "references/visual_cell_stream_v25_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/visual_cell_data.py",
    "ilm/visual_lm/visual_cell_eval_data.py",
    "ilm/visual_lm/visual_cell_stream.py",
    "scripts/eval_visual_cell_stream_development_v25.py",
    "scripts/train_visual_cell_stream_v25.py",
)
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_RETINA_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
EXPECTED_RETINA_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
LANGUAGE_STAGE = "language"
WRITER_STAGE = "writer"
FIXED_MODEL_CONFIG = VisualCellStreamConfig()
FIXED_RENDER_CONFIG = VisualCellRenderConfig()
FIXED_OPTIMIZATION = {
    "language_steps": 2_400,
    "writer_steps": 1_200,
    "batch_size": 3,
    "gradient_accumulation": 4,
    "language_lr": 3e-4,
    "writer_lr": 2e-4,
    "minimum_lr_ratio": 0.10,
    "language_warmup": 120,
    "writer_warmup": 60,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "writer_positions": 4,
    "seed": 20260830,
    "dataset_seed": 20260831,
}
FIXED_EVIDENCE = {
    "precision": "bf16",
    "audit_windows": 2_048,
    "audit_bank_size": 1_024,
    "audit_batch_size": 16,
    "writer_audit_samples": 256,
    "autonomous_audit_samples": 16,
    "inference_flow_steps": 12,
    "inference_candidates": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V25 image-only Chinese cell stream."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--retina-checkpoint", default=DEFAULT_RETINA_CHECKPOINT)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default="artifacts/visual_cell_stream_v25_evidence")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--language-steps", type=int, default=2_400)
    parser.add_argument("--writer-steps", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--language-lr", type=float, default=3e-4)
    parser.add_argument("--writer-lr", type=float, default=2e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--language-warmup", type=int, default=120)
    parser.add_argument("--writer-warmup", type=int, default=60)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--writer-positions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--dataset-seed", type=int, default=20260831)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--audit-windows", type=int, default=2_048)
    parser.add_argument("--audit-bank-size", type=int, default=1_024)
    parser.add_argument("--audit-batch-size", type=int, default=16)
    parser.add_argument("--writer-audit-samples", type=int, default=256)
    parser.add_argument("--autonomous-audit-samples", type=int, default=16)
    parser.add_argument("--inference-flow-steps", type=int, default=12)
    parser.add_argument("--inference-candidates", type=int, default=4)
    parser.add_argument("--from-scratch-retina", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--continue-writer-after-failed-language", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=2)
    return parser.parse_args()


def _require_fixed_evidence_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if args.from_scratch_retina:
        raise ValueError("V25 evidence requires the frozen V16 image retina")
    if args.continue_writer_after_failed_language:
        raise ValueError("V25 evidence cannot bypass the language selection gate")
    for name, expected in FIXED_OPTIMIZATION.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V25 evidence requires --{name.replace('_', '-')}={expected}"
            )
    for name, expected in FIXED_EVIDENCE.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V25 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    if not args.smoke:
        return vars(args).copy()
    if not 1 <= args.smoke_steps <= 20:
        raise ValueError("V25 smoke mode allows 1--20 updates per stage")
    output = vars(args).copy()
    output.update(
        {
            "language_steps": args.smoke_steps,
            "writer_steps": args.smoke_steps,
            "audit_windows": 16,
            "audit_bank_size": 32,
            "audit_batch_size": 4,
            "writer_audit_samples": 4,
            "autonomous_audit_samples": 1,
            "inference_flow_steps": 2,
            "inference_candidates": 1,
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
    model: VisualCellStreamModel,
    path: str | Path,
    *,
    require_expected_hash: bool,
) -> dict[str, Any]:
    digest = file_sha256(path)
    if require_expected_hash and digest != EXPECTED_RETINA_SHA256:
        raise ValueError(
            f"V25 evidence requires V16 retina {EXPECTED_RETINA_SHA256}, got {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != "predictive-visual-field-state-flow-v1":
        raise ValueError("V25 retina source is not the expected predictive visual field")
    state = {
        name.removeprefix("retina."): value
        for name, value in checkpoint["model"].items()
        if name.startswith("retina.")
    }
    if not state:
        raise ValueError("V16 checkpoint contains no retinal state")
    model.online_retina.load_state_dict(state, strict=True)
    model.target_retina.load_state_dict(state, strict=True)
    return {
        "route": "v16-frozen-image-retina",
        "checkpoint": str(path),
        "sha256": digest,
        "source_architecture": checkpoint["architecture"],
        "source_step": checkpoint.get("global_step"),
        "retina_parameters": sum(
            parameter.numel() for parameter in model.online_retina.parameters()
        ),
    }


def _set_training_stage(
    model: VisualCellStreamModel,
    stage: str,
    *,
    train_retina: bool,
) -> None:
    model.requires_grad_(False)
    if stage == LANGUAGE_STAGE:
        for module in (
            model.context_time,
            model.visual_input,
            model.blocks,
            model.output_norm,
            model.proposal,
        ):
            module.requires_grad_(True)
        model.logit_scale.requires_grad_(True)
        if train_retina:
            model.online_retina.requires_grad_(True)
        model.train()
        model.writer.eval()
        model.target_retina.eval()
        if not train_retina:
            model.online_retina.eval()
        return
    if stage == WRITER_STAGE:
        model.writer.requires_grad_(True)
        model.eval()
        model.writer.train()
        return
    raise ValueError(f"unknown V25 training stage {stage!r}")


def _bidirectional_views(
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "context": torch.cat((batch["context"], batch["reference_context"])),
        "pixel_target": torch.cat((batch["target"], batch["reference_target"])),
        "independent_target": torch.cat(
            (batch["reference_target"], batch["target"])
        ),
    }


def _device_student_batch(
    raw: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    student = student_visual_cell_batch(raw)
    return {
        name: value.to(device, non_blocking=True) for name, value in student.items()
    }


def _build_loader(
    records: list[Any],
    *,
    render_config: VisualCellRenderConfig,
    seed: int,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
) -> DataLoader:
    total_examples = steps * accumulation * batch_size
    consumed_examples = start_step * accumulation * batch_size
    dataset = VisualCellStreamDataset(
        records,
        split="train",
        render_config=render_config,
        seed=seed,
        length=total_examples,
    )
    remaining = Subset(dataset, range(consumed_examples, total_examples))
    return DataLoader(
        remaining,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=visual_cell_collate,
    )


def _optimizer(
    model: VisualCellStreamModel,
    *,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("V25 stage has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=device.type == "cuda",
    )


def _protocol_receipt(
    *,
    arguments: dict[str, Any],
    manifest_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "manifest_sha256": manifest_receipt["sha256"],
        "expected_manifest_sha256": V25_MANIFEST_SHA256,
        "expected_retina_sha256": EXPECTED_RETINA_SHA256,
        "fixed_model_config": visual_cell_model_config_payload(FIXED_MODEL_CONFIG),
        "fixed_render_config": visual_cell_render_config_payload(FIXED_RENDER_CONFIG),
        "fixed_optimization": FIXED_OPTIMIZATION,
        "fixed_evidence": FIXED_EVIDENCE,
        "effective_arguments": arguments,
    }


def _checkpoint_payload(
    model: VisualCellStreamModel,
    optimizer: torch.optim.Optimizer | None,
    *,
    stage: str,
    step: int,
    smoke_only: bool,
    exploratory: bool,
    initialization: dict[str, Any],
    manifest_receipt: dict[str, Any],
    partition: dict[str, Any],
    render_config: VisualCellRenderConfig,
    arguments: dict[str, Any],
    peak_vram_gib: float,
    training_metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "stage": stage,
        "step": step,
        "smoke_only": smoke_only,
        "exploratory": exploratory,
        "model_config": visual_cell_model_config_payload(model.config),
        "render_config": visual_cell_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "initialization": initialization,
        "manifest": manifest_receipt,
        "partition": partition,
        "fonts": visual_cell_font_manifest(),
        "data_boundary": visual_cell_boundary_receipt(),
        "model_boundary": visual_cell_model_boundary_receipt(model.config),
        "protocol": _protocol_receipt(
            arguments=arguments,
            manifest_receipt=manifest_receipt,
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": _trainable_parameters(model),
        "trainable_parameter_shapes": _parameter_shapes(model),
        "effective_stream_batch": (
            arguments["batch_size"] * 2 * arguments["gradient_accumulation"]
        ),
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": training_metrics,
        "frozen_images_instantiated": False,
    }


def _save_checkpoint(
    output: Path,
    payload: dict[str, Any],
    *,
    final: bool,
) -> Path:
    suffix = "final" if final else f"step_{payload['step']:07d}"
    path = output / f"checkpoint_{payload['stage']}_{suffix}.pt"
    atomic_save(payload, path)
    return path


def _aggregate_metrics(
    total: dict[str, float], metrics: dict[str, torch.Tensor]
) -> None:
    for name, value in metrics.items():
        total[name] = total.get(name, 0.0) + float(value)


def _mean_metrics(total: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(1, count) for name, value in total.items()}


def _backward_step(
    loss: torch.Tensor,
    *,
    scaler: torch.amp.GradScaler,
) -> None:
    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: VisualCellStreamModel,
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


def train_stage(
    model: VisualCellStreamModel,
    records: list[Any],
    *,
    stage: str,
    device: torch.device,
    precision: str,
    render_config: VisualCellRenderConfig,
    steps: int,
    start_step: int,
    batch_size: int,
    accumulation: int,
    num_workers: int,
    base_lr: float,
    warmup: int,
    minimum_lr_ratio: float,
    weight_decay: float,
    gradient_clip: float,
    writer_positions: int,
    dataset_seed: int,
    train_retina: bool,
    output: Path,
    log_every: int,
    save_every: int,
    checkpoint_context: dict[str, Any],
    optimizer_state: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not 0 <= start_step < steps:
        raise ValueError("stage start step must be before its final step")
    _set_training_stage(model, stage, train_retina=train_retina)
    optimizer = _optimizer(
        model,
        lr=base_lr,
        weight_decay=weight_decay,
        device=device,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    loader = _build_loader(
        records,
        render_config=render_config,
        seed=dataset_seed + (0 if stage == LANGUAGE_STAGE else 10_000_019),
        steps=steps,
        start_step=start_step,
        batch_size=batch_size,
        accumulation=accumulation,
        num_workers=num_workers,
    )
    iterator = iter(loader)
    log_path = output / f"train_{stage}.jsonl"
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
                base=base_lr,
                warmup=warmup,
                total=steps,
                minimum_ratio=minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            totals: dict[str, float] = {}
            for _ in range(accumulation):
                raw = next(iterator)
                batch = _bidirectional_views(_device_student_batch(raw, device))
                if stage == LANGUAGE_STAGE:
                    noise_time = torch.rand(
                        batch["context"].shape[:2],
                        device=device,
                        dtype=batch["context"].dtype,
                    ) * model.config.context_noise_maximum
                    noise = torch.randn_like(batch["context"])
                    corrupted = (
                        (1.0 - noise_time[:, :, None, None, None]) * batch["context"]
                        + noise_time[:, :, None, None, None] * noise
                    )
                    with autocast_context(device, precision):
                        prediction = model.forward_language(
                            corrupted,
                            batch["independent_target"],
                            context_noise_time=noise_time,
                        )
                        loss, metrics = visual_cell_language_loss(
                            prediction,
                            contrastive_scale=model.contrastive_scale,
                        )
                else:
                    with torch.no_grad(), autocast_context(device, precision):
                        language = model.language(batch["context"])
                    count, length = batch["context"].shape[:2]
                    positions = torch.randint(
                        length,
                        (count, writer_positions),
                        device=device,
                    )
                    rows = torch.arange(count, device=device)[:, None]
                    context_state = language["context_state"][rows, positions].flatten(0, 1)
                    proposal = language["proposed_visual"][rows, positions].flatten(0, 1)
                    target = batch["pixel_target"][rows, positions].flatten(0, 1)
                    reference = batch["independent_target"][rows, positions].flatten(0, 1)
                    with autocast_context(device, precision):
                        loss, metrics = visual_cell_flow_loss(
                            model,
                            context_state,
                            proposal,
                            target,
                            reference,
                        )
                _backward_step(loss / accumulation, scaler=scaler)
                _aggregate_metrics(totals, metrics)
            gradient_norm = _optimizer_step(
                optimizer,
                model,
                scaler=scaler,
                gradient_clip=gradient_clip,
            )
            if stage == LANGUAGE_STAGE and train_retina:
                model.update_target_retina(momentum=0.996)
            final_metrics = _mean_metrics(totals, accumulation)
            final_metrics.update(
                {
                    "stage": stage,
                    "step": float(step),
                    "learning_rate": lr,
                    "gradient_norm": gradient_norm,
                }
            )
            if step == 1 or step % log_every == 0 or step == steps:
                append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            final = step == steps
            if final or step % save_every == 0 or stop_requested:
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint_payload(
                    model,
                    optimizer,
                    stage=stage,
                    step=step,
                    peak_vram_gib=peak,
                    training_metrics=final_metrics,
                    **checkpoint_context,
                )
                _save_checkpoint(output, payload, final=final)
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V25 {stage} stopped after checkpointing step {step}"
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
        stage=stage,
        step=steps,
        peak_vram_gib=peak,
        training_metrics=final_metrics,
        **checkpoint_context,
    )
    return final_metrics, final_payload


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    arguments: dict[str, Any],
    manifest_receipt: dict[str, Any],
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("resume checkpoint is not a V25 visual-cell stream")
    if checkpoint.get("manifest", {}).get("sha256") != manifest_receipt["sha256"]:
        raise ValueError("resume checkpoint uses a different corpus manifest")
    protocol = checkpoint.get("protocol", {})
    for name in ("batch_size", "gradient_accumulation", "precision"):
        if protocol.get("effective_arguments", {}).get(name) != arguments[name]:
            raise ValueError(f"resume checkpoint differs for {name}")


def main() -> None:
    args = parse_args()
    _require_fixed_evidence_arguments(args)
    arguments = _effective_arguments(args)
    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(arguments["seed"])
    device = choose_device(arguments["device"])
    if not (args.smoke or args.exploratory) and device.type != "cuda":
        raise ValueError("V25 evidence requires CUDA BF16 execution")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"], strict=not args.exploratory
    )
    records = load_v25_records(
        arguments["manifest"], strict_manifest=not args.exploratory
    )
    partition = visual_cell_partition_receipt(records)
    render_config = FIXED_RENDER_CONFIG
    smoke_only = bool(args.smoke)
    exploratory = bool(args.exploratory or args.smoke)

    resume_checkpoint = None
    start_stage = LANGUAGE_STAGE
    start_step = 0
    optimizer_state = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _validate_resume(
            resume_checkpoint,
            arguments=arguments,
            manifest_receipt=manifest_receipt,
        )
        config = visual_cell_model_config_from_payload(resume_checkpoint["model_config"])
        model = VisualCellStreamModel(
            config,
            freeze_retina=not arguments["from_scratch_retina"],
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        initialization = resume_checkpoint["initialization"]
        start_stage = resume_checkpoint["stage"]
        start_step = int(resume_checkpoint["step"])
        optimizer_state = resume_checkpoint.get("optimizer")
    else:
        model = VisualCellStreamModel(
            FIXED_MODEL_CONFIG,
            freeze_retina=not arguments["from_scratch_retina"],
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
        "smoke_only": smoke_only,
        "exploratory": exploratory,
        "initialization": initialization,
        "manifest_receipt": manifest_receipt,
        "partition": partition,
        "render_config": render_config,
        "arguments": arguments,
    }

    language_payload: dict[str, Any]
    if start_stage == LANGUAGE_STAGE and start_step < arguments["language_steps"]:
        _, language_payload = train_stage(
            model,
            records,
            stage=LANGUAGE_STAGE,
            device=device,
            precision=arguments["precision"],
            render_config=render_config,
            steps=arguments["language_steps"],
            start_step=start_step,
            batch_size=arguments["batch_size"],
            accumulation=arguments["gradient_accumulation"],
            num_workers=arguments["num_workers"],
            base_lr=arguments["language_lr"],
            warmup=min(arguments["language_warmup"], arguments["language_steps"]),
            minimum_lr_ratio=arguments["minimum_lr_ratio"],
            weight_decay=arguments["weight_decay"],
            gradient_clip=arguments["gradient_clip"],
            writer_positions=arguments["writer_positions"],
            dataset_seed=arguments["dataset_seed"],
            train_retina=arguments["from_scratch_retina"],
            output=output,
            log_every=arguments["log_every"],
            save_every=arguments["save_every"],
            checkpoint_context=checkpoint_context,
            optimizer_state=optimizer_state,
        )
        language_path = _save_checkpoint(output, language_payload, final=True)
    elif resume_checkpoint is not None:
        language_payload = resume_checkpoint
        language_path = Path(args.resume)
    else:
        raise ValueError("invalid V25 language-stage resume state")

    language_payload["checkpoint_path"] = str(language_path)
    language_report, _ = run_development_audit(
        model,
        language_payload,
        manifest=arguments["manifest"],
        device=device,
        precision=arguments["precision"],
        batch_size=arguments["audit_batch_size"],
        num_workers=arguments["num_workers"],
        windows=arguments["audit_windows"],
        bank_size=arguments["audit_bank_size"],
        writer_samples=arguments["writer_audit_samples"],
        autonomous_samples=arguments["autonomous_audit_samples"],
        candidates=arguments["inference_candidates"],
        flow_steps=arguments["inference_flow_steps"],
        evaluate_pixels=False,
    )
    (output / "development_language.json").write_text(
        json.dumps(language_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    language_selected = bool(language_report["language_selected"])
    may_train_writer = (
        language_selected
        or smoke_only
        or (
            exploratory
            and arguments["continue_writer_after_failed_language"]
        )
    )
    if not may_train_writer:
        print("V25 language gates failed; writer remains untrained.", flush=True)
        return

    writer_start = 0
    writer_optimizer_state = None
    if resume_checkpoint is not None and start_stage == WRITER_STAGE:
        writer_start = start_step
        writer_optimizer_state = optimizer_state
    if (
        resume_checkpoint is not None
        and start_stage == WRITER_STAGE
        and writer_start >= arguments["writer_steps"]
    ):
        writer_payload = resume_checkpoint
        writer_path = Path(args.resume)
    else:
        _, writer_payload = train_stage(
            model,
            records,
            stage=WRITER_STAGE,
            device=device,
            precision=arguments["precision"],
            render_config=render_config,
            steps=arguments["writer_steps"],
            start_step=writer_start,
            batch_size=arguments["batch_size"],
            accumulation=arguments["gradient_accumulation"],
            num_workers=arguments["num_workers"],
            base_lr=arguments["writer_lr"],
            warmup=min(arguments["writer_warmup"], arguments["writer_steps"]),
            minimum_lr_ratio=arguments["minimum_lr_ratio"],
            weight_decay=arguments["weight_decay"],
            gradient_clip=arguments["gradient_clip"],
            writer_positions=arguments["writer_positions"],
            dataset_seed=arguments["dataset_seed"],
            train_retina=False,
            output=output,
            log_every=arguments["log_every"],
            save_every=arguments["save_every"],
            checkpoint_context=checkpoint_context,
            optimizer_state=writer_optimizer_state,
        )
        writer_path = _save_checkpoint(output, writer_payload, final=True)
    writer_payload["checkpoint_path"] = str(writer_path)
    writer_report, sample = run_development_audit(
        model,
        writer_payload,
        manifest=arguments["manifest"],
        device=device,
        precision=arguments["precision"],
        batch_size=arguments["audit_batch_size"],
        num_workers=arguments["num_workers"],
        windows=arguments["audit_windows"],
        bank_size=arguments["audit_bank_size"],
        writer_samples=arguments["writer_audit_samples"],
        autonomous_samples=arguments["autonomous_audit_samples"],
        candidates=arguments["inference_candidates"],
        flow_steps=arguments["inference_flow_steps"],
        evaluate_pixels=True,
    )
    writer_report["writer_trained_after_language_selected"] = language_selected
    (output / "development_writer.json").write_text(
        json.dumps(writer_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if sample is not None:
        rows = [sample["context"], sample["reference"], sample["generated"]]
        if "autonomous" in sample:
            rows.append(sample["autonomous"])
        width = max(row.shape[0] for row in rows)
        padded = [
            torch.cat((row, torch.zeros(width - row.shape[0], 1, 32, 32)))
            if row.shape[0] < width
            else row
            for row in rows
        ]
        from ilm.visual_lm.visual_cell_data import pack_visual_cells

        pack_visual_cells(torch.cat(padded), columns=width, gutter=1).save(
            output / "writer_sample.png"
        )
    print(json.dumps(writer_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
