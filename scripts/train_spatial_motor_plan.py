#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    load_visual_grammar_manifest,
)
from ilm.visual_lm.predictive_visual_field import PredictiveVisualField
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.spatial_motor_plan import (
    SpatialMotorPlanConfig,
    SpatialRetinalMotorPlan,
    evaluate_spatial_motor_plan_batch,
    spatial_motor_plan_config_from_payload,
    spatial_motor_plan_config_payload,
    spatial_motor_plan_loss,
    summarize_spatial_motor_plan_trace,
)
from ilm.visual_lm.visual_motor_plan import visual_motor_plan_config_from_payload
from scripts.train_visual_motor_plan import (
    partition_receipt,
    partition_records,
    selection_eligible as global_selection_eligible,
)
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    scheduled_lr,
    select_examples,
)


ARCHITECTURE = "spatial-retinal-motor-plan-v1"
GLOBAL_ARCHITECTURE = "continuous-visual-motor-plan-v1"
PARTITION_SALT = "spatial-retinal-motor-plan-v19"
SELECTION_RULE = {
    "primary": "maximize development dense correct pixel F1",
    "tie_breakers": [
        "maximize development overall correct pixel F1",
        "prefer the earlier step",
    ],
    "requirements": [
        "correct_pixel_f1 > 0.68",
        "correct_pixel_f1_dense > 0.58",
        "correct_pixel_f1_dense - spatial_shuffled_pixel_f1_dense > 0.12",
        "correct_pixel_f1_dense - zero_field_pixel_f1_dense > 0.03",
        "correct_identity_top1 > 0.75",
        "correct_identity_top1 > both_shuffled_identity_top1",
        "correct_target_cosine > 0.84",
        "correct_target_cosine > both_shuffled_target_cosine",
        "condition_pixel_l1 > 0.08",
        "semantic_target_pixel_l1 > 0.05",
        "frozen_images_instantiated == 0",
    ],
    "blinded_readability_review_required_before_frozen_evaluation": True,
    "frozen_partition_read_during_selection": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the V19 continuous spatial-retinal residual on a frozen, "
            "clean V19 global motor-plan baseline."
        )
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--global-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Permit an unselected clean V19 global checkpoint for at most 20 "
            "steps. Smoke outputs are non-evidentiary."
        ),
    )
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/spatial_motor_plan_v19")
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument("--holdout-fraction", type=float, default=0.06)
    parser.add_argument("--development-share", type=float, default=0.50)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=100_000)
    parser.add_argument("--development-samples", type=int, default=128)
    parser.add_argument("--spatial-hidden-channels", type=int, default=128)
    parser.add_argument("--spatial-blocks", type=int, default=2)
    parser.add_argument("--spatial-gate-init", type=float, default=-2.0)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pixel-l1-weight", type=float, default=0.50)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--spatial-margin-weight", type=float, default=0.10)
    parser.add_argument("--spatial-margin", type=float, default=0.03)
    parser.add_argument("--zero-margin-weight", type=float, default=0.10)
    parser.add_argument("--zero-margin", type=float, default=0.01)
    parser.add_argument("--duplicate-similarity", type=float, default=0.90)
    parser.add_argument("--logit-scale", type=float, default=12.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spatial_selection_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    dense = metrics["correct_pixel_f1_dense"]
    return {
        "overall_pixel_f1": metrics["correct_pixel_f1"] > 0.68,
        "dense_pixel_f1": dense > 0.58,
        "dense_spatial_shuffle_margin": (
            dense - metrics["spatial_shuffled_pixel_f1_dense"] > 0.12
        ),
        "dense_zero_field_margin": (
            dense - metrics["zero_field_pixel_f1_dense"] > 0.03
        ),
        "identity_top1": (
            metrics["correct_identity_top1"] > 0.75
            and metrics["correct_identity_top1"]
            > metrics["both_shuffled_identity_top1"]
        ),
        "target_cosine": (
            metrics["correct_target_cosine"] > 0.84
            and metrics["correct_target_cosine"]
            > metrics["both_shuffled_target_cosine"]
        ),
        "condition_pixel_l1": metrics["condition_pixel_l1"] > 0.08,
        "semantic_target_pixel_l1": metrics["semantic_target_pixel_l1"] > 0.05,
        "frozen_bank_sealed": (metrics.get("frozen_images_instantiated", 0.0) == 0.0),
    }


def spatial_selection_eligible(metrics: dict[str, float]) -> bool:
    return all(spatial_selection_gate_report(metrics).values())


def better_development_candidate(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["correct_pixel_f1_dense"]),
        float(candidate["correct_pixel_f1"]),
        -int(candidate["step"]),
    )
    incumbent_key = (
        float(incumbent["correct_pixel_f1_dense"]),
        float(incumbent["correct_pixel_f1"]),
        -int(incumbent["step"]),
    )
    return candidate_key > incumbent_key


def development_snapshot(metrics: dict[str, float], step: int) -> dict[str, Any]:
    keys = (
        "correct_pixel_f1",
        "correct_pixel_f1_simple",
        "correct_pixel_f1_medium",
        "correct_pixel_f1_dense",
        "spatial_shuffled_pixel_f1_dense",
        "global_shuffled_pixel_f1_dense",
        "both_shuffled_pixel_f1_dense",
        "zero_field_pixel_f1_dense",
        "correct_identity_top1",
        "both_shuffled_identity_top1",
        "correct_target_cosine",
        "both_shuffled_target_cosine",
        "condition_pixel_l1",
        "spatial_condition_pixel_l1",
        "zero_field_condition_pixel_l1",
        "semantic_target_pixel_l1",
        "spatial_gate",
        "spatial_residual_rms",
    )
    return {"step": step, **{key: float(metrics[key]) for key in keys}}


def validate_global_baseline_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_partition: dict[str, Any],
    expected_pvf_sha256: str,
    allow_unselected: bool,
) -> None:
    if checkpoint.get("architecture") != GLOBAL_ARCHITECTURE:
        raise ValueError("V19 global baseline has the wrong architecture")
    if checkpoint.get("partition") != expected_partition:
        raise ValueError("V19 global baseline uses a different partition")
    arguments = checkpoint.get("arguments", {})
    if arguments.get("partition_salt") != PARTITION_SALT:
        raise ValueError("V19 global baseline does not use the fixed V19 salt")
    if (
        checkpoint.get("style_warmstart") is not None
        or arguments.get("warmstart_v17") is not None
    ):
        raise ValueError("V19 global baseline is contaminated by a V17 warm-start")
    if checkpoint.get("pvf_sha256") != expected_pvf_sha256:
        raise ValueError("V19 global baseline uses a different Predictive Visual Field")
    contract = checkpoint.get("student_contract", {})
    forbidden_true = (
        "target_spatial_pixels_enter_condition",
        "student_received_token_ids",
        "student_received_unicode_ids",
        "student_received_ocr",
        "student_received_character_labels",
        "student_used_visual_codebook",
        "student_used_candidate_classifier",
        "student_used_external_language_model",
    )
    if any(bool(contract.get(key, False)) for key in forbidden_true):
        raise ValueError("V19 global baseline violates the image-only student contract")

    selected = checkpoint.get("best_development")
    if selected is None:
        if not allow_unselected:
            raise ValueError(
                "evidence training requires a selected V19 global baseline"
            )
        return
    if not global_selection_eligible(selected):
        raise ValueError("V19 global baseline does not pass the fixed V18 gates")
    if int(checkpoint.get("global_step", -1)) != int(selected["step"]):
        raise ValueError("V19 global baseline checkpoint is not the selected step")


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    expected_partition: dict[str, Any],
    expected_pvf_sha256: str,
    expected_global_sha256: str,
    expected_config: SpatialMotorPlanConfig,
    smoke: bool,
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("resume is not a V19 spatial motor-plan checkpoint")
    if checkpoint.get("partition") != expected_partition:
        raise ValueError("resume uses a different V19 partition")
    if checkpoint.get("pvf_sha256") != expected_pvf_sha256:
        raise ValueError("resume uses a different Predictive Visual Field")
    if checkpoint.get("global_baseline_sha256") != expected_global_sha256:
        raise ValueError("resume uses a different V19 global baseline")
    restored = spatial_motor_plan_config_from_payload(checkpoint["planner_config"])
    if restored != expected_config:
        raise ValueError("resume uses a different spatial motor-plan configuration")
    if bool(checkpoint.get("smoke_only", False)) != smoke:
        raise ValueError("resume cannot cross the smoke/evidence boundary")


def spatial_config_from_checkpoints(
    global_checkpoint: dict[str, Any],
    pvf_checkpoint: dict[str, Any],
    *,
    spatial_hidden_channels: int,
    spatial_blocks: int,
    spatial_gate_init: float,
) -> SpatialMotorPlanConfig:
    global_config = visual_motor_plan_config_from_payload(
        global_checkpoint["planner_config"]
    )
    pvf_config = pvf_checkpoint["model_config"]
    if global_config.fovea_size != int(pvf_config["fovea_size"]):
        raise ValueError("global baseline and retina use different fovea sizes")
    if global_config.visual_dim != int(pvf_config["visual_dim"]):
        raise ValueError("global baseline and retina use different visual dimensions")
    return SpatialMotorPlanConfig(
        **global_checkpoint["planner_config"],
        spatial_channels=int(pvf_config["retina_base_channels"]) * 3,
        spatial_hidden_channels=spatial_hidden_channels,
        spatial_blocks=spatial_blocks,
        spatial_gate_init=spatial_gate_init,
    )


def _display_tile(image: torch.Tensor, size: int) -> Image.Image:
    ink = image[0].float().cpu().clamp(0, 1).numpy()
    pixels = (255.0 * (1.0 - ink)).round().astype(np.uint8)
    return (
        Image.fromarray(pixels, "L")
        .resize((size, size), Image.Resampling.NEAREST)
        .convert("RGB")
    )


def save_spatial_sample_grid(
    trace: dict[str, torch.Tensor],
    path: Path,
    count: int,
) -> None:
    rows = (
        ("target", trace["target_ink"]),
        ("semantic view", trace["semantic_reference"]),
        ("style exemplar", trace["style_reference"]),
        ("correct global + field", trace["correct_ink"]),
        ("shuffled field", trace["spatial_shuffled_ink"]),
        ("shuffled global", trace["global_shuffled_ink"]),
        ("zero field", trace["zero_field_ink"]),
        ("both shuffled", trace["both_shuffled_ink"]),
    )
    count = min(count, rows[0][1].shape[0])
    tile = 112
    label_width = 190
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
    planner: SpatialRetinalMotorPlan,
    loader: DataLoader,
    *,
    device: torch.device,
    args: argparse.Namespace,
    step: int,
    sample_root: Path,
) -> dict[str, float]:
    planner.eval()
    totals: dict[str, float] = {}
    examples = 0
    traces: dict[str, list[torch.Tensor]] = {}
    first_trace: dict[str, torch.Tensor] | None = None
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
            loss, training_metrics, _ = spatial_motor_plan_loss(
                planner,
                pvf.retina,
                target,
                semantic,
                style,
                stroke_weight=args.stroke_weight,
                dice_weight=args.dice_weight,
                pixel_l1_weight=args.pixel_l1_weight,
                edge_weight=args.edge_weight,
                identity_weight=args.identity_weight,
                contrastive_weight=args.contrastive_weight,
                spatial_margin_weight=args.spatial_margin_weight,
                spatial_margin=args.spatial_margin,
                zero_margin_weight=args.zero_margin_weight,
                zero_margin=args.zero_margin,
                duplicate_similarity=args.duplicate_similarity,
                logit_scale=args.logit_scale,
            )
            _, trace = evaluate_spatial_motor_plan_batch(
                planner,
                pvf.retina,
                target,
                semantic,
                style,
                duplicate_similarity=args.duplicate_similarity,
                logit_scale=args.logit_scale,
            )
        if first_trace is None:
            first_trace = {key: value.detach().cpu() for key, value in trace.items()}
        for key, value in trace.items():
            traces.setdefault(key, []).append(value.detach())
        batch_examples = target.shape[0]
        for key, value in {"loss": loss.detach(), **training_metrics}.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_examples
        examples += batch_examples

    if examples == 0:
        raise ValueError("development loader yielded no V19 examples")
    aggregate_trace = {key: torch.cat(values) for key, values in traces.items()}
    causal_metrics = summarize_spatial_motor_plan_trace(
        aggregate_trace,
        duplicate_similarity=args.duplicate_similarity,
        logit_scale=args.logit_scale,
    )
    if first_trace is not None:
        save_spatial_sample_grid(
            first_trace,
            sample_root / f"step_{step:07d}.png",
            args.sample_count,
        )
    planner.train()
    report = {
        **{f"plan_{key}": value / examples for key, value in totals.items()},
        **{key: float(value) for key, value in causal_metrics.items()},
        "examples": float(examples),
        "retrieval_candidates": float(aggregate_trace["intended_visual"].shape[0]),
        "frozen_images_instantiated": 0.0,
    }
    report["selection_eligible"] = float(spatial_selection_eligible(report))
    return report


def checkpoint_payload(
    planner: SpatialRetinalMotorPlan,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    pvf_checkpoint: dict[str, Any],
    pvf_sha256: str,
    global_checkpoint: dict[str, Any],
    global_sha256: str,
    partition: dict[str, Any],
    best_development: dict[str, Any] | None,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "planner_config": spatial_motor_plan_config_payload(planner.config),
        "planner": planner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": pvf_sha256,
        "pvf_architecture": pvf_checkpoint["architecture"],
        "pvf_step": int(pvf_checkpoint.get("global_step", 0)),
        "pvf_model_config": pvf_checkpoint["model_config"],
        "render_config": pvf_checkpoint["render_config"],
        "global_baseline_checkpoint": args.global_checkpoint,
        "global_baseline_sha256": global_sha256,
        "global_baseline_step": int(global_checkpoint.get("global_step", 0)),
        "global_baseline_architecture": global_checkpoint["architecture"],
        "global_baseline_style_warmstart": global_checkpoint.get("style_warmstart"),
        "partition": partition,
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "selection_rule": SELECTION_RULE,
        "best_development": best_development,
        "smoke_only": args.smoke,
        "frozen_images_instantiated_during_training": False,
        "student_contract": {
            "input": (
                "continuous global retinal state, continuous 4x4 retinal field, "
                "and a separate continuous style image"
            ),
            "output": "continuous spatial ink plan",
            "target_spatial_pixels_enter_condition": False,
            "student_received_token_ids": False,
            "student_received_unicode_ids": False,
            "student_received_ocr": False,
            "student_received_character_labels": False,
            "student_used_visual_codebook": False,
            "student_used_candidate_classifier": False,
            "student_used_external_language_model": False,
            "global_baseline_trainable": False,
            "retina_trainable": False,
        },
    }


def main() -> None:
    args = parse_args()
    if args.partition_salt != PARTITION_SALT:
        raise ValueError(f"V19 requires partition salt {PARTITION_SALT!r}")
    if args.smoke and args.maximum_steps > 20:
        raise ValueError("V19 smoke mode is limited to 20 optimization steps")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    pvf, pvf_checkpoint = load_pvf(args.pvf_checkpoint, device)
    pvf_sha256 = file_sha256(args.pvf_checkpoint)
    global_checkpoint = torch.load(
        args.global_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    global_sha256 = file_sha256(args.global_checkpoint)

    render_config = RetinalRenderConfig(**pvf_checkpoint["render_config"])
    records = load_visual_grammar_manifest(args.manifest)
    train_records, development_records, frozen_records = partition_records(
        records,
        salt=args.partition_salt,
        holdout_fraction=args.holdout_fraction,
        development_share=args.development_share,
    )
    partition = partition_receipt(
        train_records,
        development_records,
        frozen_records,
        salt=args.partition_salt,
        holdout_fraction=args.holdout_fraction,
        development_share=args.development_share,
    )
    validate_global_baseline_checkpoint(
        global_checkpoint,
        expected_partition=partition,
        expected_pvf_sha256=pvf_sha256,
        allow_unselected=args.smoke,
    )
    config = spatial_config_from_checkpoints(
        global_checkpoint,
        pvf_checkpoint,
        spatial_hidden_channels=args.spatial_hidden_channels,
        spatial_blocks=args.spatial_blocks,
        spatial_gate_init=args.spatial_gate_init,
    )

    protocol = {
        "architecture": ARCHITECTURE,
        "experiment": "V19 spatial retinal motor plan",
        "protocol_document": "references/spatial_retinal_motor_plan_v19_protocol.md",
        "partition": partition,
        "global_baseline_sha256": global_sha256,
        "pvf_sha256": pvf_sha256,
        "selection_rule": SELECTION_RULE,
        "spatial_complexity": {
            "formula": (
                "ink_mean + 0.5*(horizontal_transition_mean + "
                "vertical_transition_mean) + 0.1*pooled_occupied_fraction"
            ),
            "simple": "score < 0.24",
            "medium": "0.24 <= score < 0.35",
            "dense": "score >= 0.35",
            "training_weights": {"simple": 1.0, "medium": 1.25, "dense": 2.0},
        },
        "causal_branches": [
            "correct",
            "spatial_shuffled",
            "global_shuffled",
            "both_shuffled",
            "zero_field",
        ],
        "smoke_only": args.smoke,
        "frozen_evaluation_status": (
            "forbidden until automatic development and blinded readability gates pass"
        ),
    }
    protocol_path = output / "preregistered_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("existing preregistered V19 protocol differs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "partition.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    planner = SpatialRetinalMotorPlan(config)
    planner.load_global_plan(global_checkpoint["planner"])
    planner.to(device)
    trainable = [
        parameter for parameter in planner.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    global_step = 0
    start_epoch = 0
    elapsed_before = 0.0
    best_development: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(
            resume,
            expected_partition=partition,
            expected_pvf_sha256=pvf_sha256,
            expected_global_sha256=global_sha256,
            expected_config=config,
            smoke=args.smoke,
        )
        planner.load_state_dict(resume["planner"], strict=True)
        planner.to(device)
        optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("global_step", 0))
        start_epoch = int(resume.get("epoch", 0))
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        best_development = resume.get("best_development")

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

    planned_steps = args.maximum_steps or args.epochs * max(1, len(train_loader))
    startup = {
        "stage": "startup",
        "architecture": ARCHITECTURE,
        "parameters": sum(parameter.numel() for parameter in planner.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "frozen_global_parameters": sum(
            parameter.numel() for parameter in planner.global_plan.parameters()
        ),
        "frozen_pvf_parameters": sum(
            parameter.numel() for parameter in pvf.parameters()
        ),
        "classifier_parameters": 0,
        "token_embedding_parameters": 0,
        "device": str(device),
        "planned_steps": planned_steps,
        "sequence_length": args.sequence_length,
        "positions_per_sequence": args.positions_per_sequence,
        "partition": partition,
        "global_baseline_sha256": global_sha256,
        "selection_rule": SELECTION_RULE,
        "smoke_only": args.smoke,
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
            json.dumps(
                {"stage": "signal", "signal": signum, "action": "save_then_stop"}
            ),
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
    planner.train()
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
                loss, metrics, _ = spatial_motor_plan_loss(
                    planner,
                    pvf.retina,
                    target,
                    semantic,
                    style,
                    stroke_weight=args.stroke_weight,
                    dice_weight=args.dice_weight,
                    pixel_l1_weight=args.pixel_l1_weight,
                    edge_weight=args.edge_weight,
                    identity_weight=args.identity_weight,
                    contrastive_weight=args.contrastive_weight,
                    spatial_margin_weight=args.spatial_margin_weight,
                    spatial_margin=args.spatial_margin,
                    zero_margin_weight=args.zero_margin_weight,
                    zero_margin=args.zero_margin,
                    duplicate_similarity=args.duplicate_similarity,
                    logit_scale=args.logit_scale,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
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
                    "examples_per_second": running_examples
                    / max(1e-6, now - interval_started),
                    **{
                        key: value / max(1, running_examples)
                        for key, value in running.items()
                    },
                }
                if device.type == "cuda":
                    report["peak_cuda_gib"] = (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    )
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)
                running.clear()
                running_examples = 0
                interval_started = now

            if global_step % args.validate_every == 0:
                validation = validate(
                    pvf,
                    planner,
                    development_loader,
                    device=device,
                    args=args,
                    step=global_step,
                    sample_root=output / "development_samples",
                )
                report = {"stage": "validation", "step": global_step, **validation}
                print(json.dumps(report), flush=True)
                append_jsonl(log_path, report)
                if spatial_selection_eligible(validation):
                    candidate = development_snapshot(validation, global_step)
                    if better_development_candidate(candidate, best_development):
                        best_development = candidate
                        elapsed = elapsed_before + time.perf_counter() - started
                        payload = checkpoint_payload(
                            planner,
                            optimizer,
                            args=args,
                            pvf_checkpoint=pvf_checkpoint,
                            pvf_sha256=pvf_sha256,
                            global_checkpoint=global_checkpoint,
                            global_sha256=global_sha256,
                            partition=partition,
                            best_development=best_development,
                            epoch=epoch,
                            step=global_step,
                            elapsed_seconds=elapsed,
                        )
                        atomic_save(
                            payload,
                            output / "checkpoint_selected_development.pt",
                        )
                        (output / "development_selection.json").write_text(
                            json.dumps(
                                {
                                    "selection_rule": SELECTION_RULE,
                                    "selected": best_development,
                                    "automatic_development_gate_passed": True,
                                    "blinded_readability_gate_passed": False,
                                    "frozen_evaluation_permitted": False,
                                    "reason": (
                                        "the fixed blinded readability review remains"
                                    ),
                                },
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                            encoding="utf-8",
                        )

            if global_step % args.save_every == 0:
                elapsed = elapsed_before + time.perf_counter() - started
                payload = checkpoint_payload(
                    planner,
                    optimizer,
                    args=args,
                    pvf_checkpoint=pvf_checkpoint,
                    pvf_sha256=pvf_sha256,
                    global_checkpoint=global_checkpoint,
                    global_sha256=global_sha256,
                    partition=partition,
                    best_development=best_development,
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
        planner,
        optimizer,
        args=args,
        pvf_checkpoint=pvf_checkpoint,
        pvf_sha256=pvf_sha256,
        global_checkpoint=global_checkpoint,
        global_sha256=global_sha256,
        partition=partition,
        best_development=best_development,
        epoch=epoch,
        step=global_step,
        elapsed_seconds=elapsed,
    )
    atomic_save(payload, output / "checkpoint_latest.pt")
    final = {
        "stage": "stopped" if stop_requested else "complete",
        "step": global_step,
        "elapsed_seconds": elapsed,
        "best_development": best_development,
        "checkpoint": str(output / "checkpoint_latest.pt"),
        "smoke_only": args.smoke,
        "frozen_evaluation_permitted": False,
    }
    print(json.dumps(final), flush=True)
    append_jsonl(log_path, final)


if __name__ == "__main__":
    main()
