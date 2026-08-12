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
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    RETINAL_CJK_AVAILABLE_FONTS,
    RetinalRenderConfig,
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.predictive_visual_field import (
    PredictiveVisualField,
    PredictiveVisualFieldConfig,
    initialize_from_retinal_flow_checkpoint,
    predictive_visual_field_config_from_payload,
    predictive_visual_field_config_payload,
    predictive_visual_field_loss,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    render_glyph_fovea,
    visual_saccade_collate,
)


ARCHITECTURE = "predictive-visual-field-state-flow-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an image-only conditional flow over next retinal states; "
            "no pixel writer or character classifier is instantiated."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--initialize-from", help="RFLM checkpoint providing retina and GRU")
    source.add_argument("--resume", help="Predictive Visual Field checkpoint")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/predictive_visual_field_state_flow_v8")
    parser.add_argument("--validation-fraction", type=float, default=0.03)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--samples-per-epoch", type=int, default=30_000)
    parser.add_argument("--validation-samples", type=int, default=512)
    parser.add_argument("--flow-hidden-dim", type=int, default=512)
    parser.add_argument("--flow-blocks", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--condition-dropout", type=float, default=0.10)
    parser.add_argument("--sample-temperature", type=float, default=0.08)
    parser.add_argument(
        "--flow-geometry",
        choices=("euclidean", "hypersphere"),
        default="hypersphere",
        help="transport geometry for normalized retinal states",
    )
    parser.add_argument("--flow-positions-per-sequence", type=int, default=8)
    parser.add_argument("--duplicate-similarity", type=float, default=0.90)
    parser.add_argument("--flow-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=0.25)
    parser.add_argument("--sampled-identity-weight", type=float, default=0.50)
    parser.add_argument("--sampled-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--proposal-geodesic-weight", type=float, default=0.0)
    parser.add_argument("--proposal-identity-weight", type=float, default=0.0)
    parser.add_argument("--proposal-context-weight", type=float, default=0.0)
    parser.add_argument("--proposal-anchor-identity-weight", type=float, default=0.0)
    parser.add_argument("--proposal-anchor-context-weight", type=float, default=0.0)
    parser.add_argument("--sampled-positions-per-sequence", type=int, default=1)
    parser.add_argument("--context-advantage-weight", type=float, default=0.50)
    parser.add_argument("--context-advantage-margin", type=float, default=0.10)
    parser.add_argument("--visual-anchor-bank-size", type=int, default=512)
    parser.add_argument("--visual-anchor-views", type=int, default=4)
    parser.add_argument("--visual-anchor-positive-similarity", type=float, default=0.85)
    parser.add_argument("--visual-anchor-identity-weight", type=float, default=0.50)
    parser.add_argument("--visual-anchor-context-weight", type=float, default=0.50)
    parser.add_argument("--visual-anchor-context-margin", type=float, default=0.10)
    parser.add_argument("--visual-anchor-seed-offset", type=int, default=17_000_009)
    parser.add_argument("--samples-per-context", type=int, default=2)
    parser.add_argument("--sample-steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--dynamics-lr-ratio", type=float, default=0.25)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=200)
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
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)


def is_han_character(character: str) -> bool:
    value = ord(character)
    return any(
        lower <= value <= upper
        for lower, upper in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2FA1F),
            (0x30000, 0x323AF),
        )
    )


def build_visual_anchor_images(
    records: list[Any],
    *,
    bank_size: int,
    views: int,
    render_config: RetinalRenderConfig,
    fovea_size: int,
    seed: int,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if bank_size < 0 or views < 1:
        raise ValueError("visual anchor bank size must be non-negative and views positive")
    if bank_size == 0:
        return None, {"enabled": False, "objects": 0, "views": views}
    counts = Counter(
        character
        for record in records
        for character in record.text
        if is_han_character(character)
    )
    characters = [character for character, _ in counts.most_common(bank_size)]
    if len(characters) != bank_size:
        raise ValueError(
            f"requested {bank_size} visual anchors, corpus supplied {len(characters)}"
        )
    images = torch.stack(
        [
            render_glyph_fovea(
                character,
                render_config=render_config,
                fovea_size=fovea_size,
                variant=seed + owner * 10_007 + view * 1_000_003,
            )
            for owner, character in enumerate(characters)
            for view in range(views)
        ]
    ).reshape(bank_size, views, 1, fovea_size, fovea_size)
    return images, {
        "enabled": True,
        "objects": bank_size,
        "views": views,
        "image_candidates": bank_size * views,
        "render_seed": seed,
        "selection": "most frequent Han writing objects in the training corpus",
        "sha256": hashlib.sha256("".join(characters).encode("utf-8")).hexdigest(),
        "labels_enter_student": False,
        "target_indices_enter_student": False,
        "deployed_with_model": False,
    }


@torch.no_grad()
def encode_visual_anchor_images(
    model: PredictiveVisualField,
    images: torch.Tensor | None,
    *,
    device: torch.device,
    precision: str,
    batch_size: int = 256,
) -> torch.Tensor | None:
    if images is None:
        return None
    objects, views = images.shape[:2]
    flat = images.flatten(0, 1)
    encoded: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], batch_size):
        batch = flat[start : start + batch_size].to(device, non_blocking=True)
        with autocast_context(device, precision):
            encoded.append(model.encode_images(batch).float())
    return F.normalize(torch.cat(encoded), dim=-1).reshape(objects, views, -1).detach()


def model_config_from_source(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> PredictiveVisualFieldConfig:
    source = checkpoint["model_config"]
    return PredictiveVisualFieldConfig(
        fovea_size=int(source["fovea_size"]),
        visual_dim=int(source["visual_dim"]),
        state_dim=int(source["state_dim"]),
        state_layers=int(source["state_layers"]),
        retina_base_channels=int(source["retina_base_channels"]),
        dropout=float(source["dropout"]),
        flow_hidden_dim=args.flow_hidden_dim,
        flow_blocks=args.flow_blocks,
        time_dim=args.time_dim,
        condition_dropout=args.condition_dropout,
        sample_temperature=args.sample_temperature,
        flow_geometry=args.flow_geometry,
    )


def loss_for_batch(
    model: PredictiveVisualField,
    batch: dict[str, Any],
    *,
    device: torch.device,
    args: argparse.Namespace,
    generator: torch.Generator | None,
    visual_anchor_candidates: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    context = batch["context"].to(device, non_blocking=True)
    target_ink = batch["target_ink"].to(device, non_blocking=True)
    target_reference = batch["target_reference"].to(device, non_blocking=True)
    outputs = model(context, target_reference)
    loss, metrics = predictive_visual_field_loss(
        model,
        outputs,
        target_ink,
        flow_positions_per_sequence=args.flow_positions_per_sequence,
        duplicate_similarity=args.duplicate_similarity,
        flow_weight=args.flow_weight,
        endpoint_weight=args.endpoint_weight,
        sampled_identity_weight=args.sampled_identity_weight,
        sampled_endpoint_weight=args.sampled_endpoint_weight,
        proposal_geodesic_weight=args.proposal_geodesic_weight,
        proposal_identity_weight=args.proposal_identity_weight,
        proposal_context_weight=args.proposal_context_weight,
        proposal_anchor_identity_weight=args.proposal_anchor_identity_weight,
        proposal_anchor_context_weight=args.proposal_anchor_context_weight,
        sampled_positions_per_sequence=args.sampled_positions_per_sequence,
        context_advantage_weight=args.context_advantage_weight,
        context_advantage_margin=args.context_advantage_margin,
        visual_anchor_candidates=visual_anchor_candidates,
        visual_anchor_positive_similarity=args.visual_anchor_positive_similarity,
        visual_anchor_identity_weight=args.visual_anchor_identity_weight,
        visual_anchor_context_weight=args.visual_anchor_context_weight,
        visual_anchor_context_margin=args.visual_anchor_context_margin,
        samples_per_context=args.samples_per_context,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    return loss, metrics, context, target_reference


@torch.no_grad()
def validate(
    model: PredictiveVisualField,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    probe_root: Path,
    visual_anchor_candidates: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    first_probe: dict[str, torch.Tensor] | None = None
    generator = torch.Generator(device=device).manual_seed(args.seed + step * 99_991)
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.validation_batches:
            break
        with autocast_context(device, args.precision):
            loss, metrics, context, target_reference = loss_for_batch(
                model,
                batch,
                device=device,
                args=args,
                generator=generator,
                visual_anchor_candidates=visual_anchor_candidates,
            )
        batch_size = len(batch["metadata"])
        for key, value in {"loss": loss, **metrics}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        examples += batch_size
        if first_probe is None:
            prediction = model.predict(context[:8])
            states = model.sample_states(
                prediction["condition"][:, -1],
                samples_per_context=max(4, args.samples_per_context),
                steps=max(4, args.sample_steps),
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            first_probe = {
                "context_pixels": context[:8].float().cpu(),
                "target_pixels": target_reference[:8, -1].float().cpu(),
                "sampled_states": states.float().cpu(),
                "target_states": model.encode_images(target_reference[:8, -1]).float().cpu(),
            }
    if first_probe is not None:
        probe_root.mkdir(parents=True, exist_ok=True)
        torch.save(first_probe, probe_root / f"state_probe_step_{step:07d}.pt")
    model.train()
    report = {key: value / max(1, examples) for key, value in totals.items()}
    report["examples"] = float(examples)
    return report


def checkpoint_payload(
    model: PredictiveVisualField,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    render_config: RetinalRenderConfig,
    initialization: dict[str, Any],
    visual_anchor_manifest: dict[str, Any],
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "model_config": predictive_visual_field_config_payload(model.config),
        "render_config": render_config.__dict__,
        "retinal_fonts": retinal_font_manifest(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "initialization": initialization,
        "training_visual_anchor_bank": visual_anchor_manifest,
        "student_contract": {
            "input": "ordered continuous writing-image foveas",
            "state": "causal continuous visual field",
            "distribution": "conditional rectified flow over image-derived retinal states",
            "flow_geometry": model.config.flow_geometry,
            "output_for_this_stage": "continuous sampled retinal state",
            "deterministic_output": "continuous visual proposal from causal history",
            "pixel_actuator_instantiated": False,
            "candidate_classifier_instantiated": False,
            "forbidden": [
                "token_ids",
                "unicode_ids",
                "character_labels",
                "ocr_strings",
                "glyph_codebooks",
                "nearest_character_lookup",
                "external_language_models",
            ],
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

    resume_checkpoint: dict[str, Any] | None = None
    resume_optimizer_state = True
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("resume checkpoint is not a Predictive Visual Field")
        model_config = predictive_visual_field_config_from_payload(
            resume_checkpoint["model_config"]
        )
        render_config = RetinalRenderConfig(**resume_checkpoint["render_config"])
        initialization = dict(resume_checkpoint.get("initialization", {}))
    else:
        source_checkpoint = torch.load(
            args.initialize_from,
            map_location="cpu",
            weights_only=False,
        )
        model_config = model_config_from_source(source_checkpoint, args)
        render_config = RetinalRenderConfig(**source_checkpoint["render_config"])
        model = PredictiveVisualField(model_config)
        initialization = initialize_from_retinal_flow_checkpoint(model, source_checkpoint)
        initialization.update(
            {
                "source_checkpoint": args.initialize_from,
                "source_sha256": file_sha256(args.initialize_from),
            }
        )
        del source_checkpoint

    records = load_visual_grammar_manifest(args.manifest)
    visual_anchor_images, visual_anchor_manifest = build_visual_anchor_images(
        records,
        bank_size=args.visual_anchor_bank_size,
        views=args.visual_anchor_views,
        render_config=render_config,
        fovea_size=model_config.fovea_size,
        seed=args.seed + args.visual_anchor_seed_offset,
    )
    (output / "training_visual_anchor_bank.json").write_text(
        json.dumps(visual_anchor_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sequence_spec = SaccadeSequenceSpec(
        sequence_length=args.sequence_length,
        fovea_size=model_config.fovea_size,
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
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )
    if args.resume:
        model = PredictiveVisualField(model_config)
        if resume_checkpoint is None:
            raise RuntimeError("resume checkpoint was not loaded")
        incompatible = model.load_state_dict(resume_checkpoint["model"], strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = list(incompatible.missing_keys)
        if unexpected:
            raise ValueError(f"unexpected resume parameters: {unexpected}")
        if any(not key.startswith("visual_proposal.") for key in missing):
            raise ValueError(f"unsupported missing resume parameters: {missing}")
        if missing:
            resume_optimizer_state = False
            initialization["visual_proposal_initialization"] = {
                "source": "random",
                "missing_parameters": len(missing),
                "optimizer_reset": True,
            }
    model = model.to(device)
    visual_anchor_candidates = encode_visual_anchor_images(
        model,
        visual_anchor_images,
        device=device,
        precision=args.precision,
    )
    del visual_anchor_images

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    list(model.state_flow.parameters())
                    + list(model.visual_proposal.parameters())
                    + [model.logit_scale]
                ),
                "lr_ratio": 1.0,
            },
            {
                "params": model.dynamics.parameters(),
                "lr_ratio": args.dynamics_lr_ratio,
            },
        ],
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
    if resume_checkpoint is not None:
        if resume_optimizer_state:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        global_step = int(resume_checkpoint.get("global_step", 0))
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        elapsed_before = float(resume_checkpoint.get("elapsed_seconds", 0.0))
        del resume_checkpoint

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    startup = {
        "stage": "startup",
        "architecture": ARCHITECTURE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "retina_parameters": sum(parameter.numel() for parameter in model.retina.parameters()),
        "pixel_actuator_parameters": 0,
        "classifier_parameters": 0,
        "visual_proposal_parameters": sum(
            parameter.numel() for parameter in model.visual_proposal.parameters()
        ),
        "flow_geometry": model.config.flow_geometry,
        "records": len(records),
        "train_records": len(train_dataset.records),
        "validation_records": len(validation_dataset.records),
        "sequence_length": args.sequence_length,
        "planned_steps": planned_steps,
        "initialization": initialization,
        "training_visual_anchor_bank": visual_anchor_manifest,
        "retinal_fonts": list(RETINAL_CJK_AVAILABLE_FONTS),
        "device": str(device),
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
            if stop_requested or global_step >= planned_steps:
                break
            global_step += 1
            base_lr = scheduled_lr(
                global_step,
                base=args.lr,
                warmup=args.warmup_steps,
                total=planned_steps,
                minimum_ratio=args.minimum_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = base_lr * float(group["lr_ratio"])
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                loss, metrics, _, _ = loss_for_batch(
                    model,
                    batch,
                    device=device,
                    args=args,
                    generator=generator,
                    visual_anchor_candidates=visual_anchor_candidates,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
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
                    "learning_rate": base_lr,
                    "dynamics_learning_rate": base_lr * args.dynamics_lr_ratio,
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
                    model,
                    validation_loader,
                    device=device,
                    args=args,
                    step=global_step,
                    probe_root=output / "state_probes",
                    visual_anchor_candidates=visual_anchor_candidates,
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
                    initialization=initialization,
                    visual_anchor_manifest=visual_anchor_manifest,
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
        model,
        optimizer,
        args=args,
        render_config=render_config,
        initialization=initialization,
        visual_anchor_manifest=visual_anchor_manifest,
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
