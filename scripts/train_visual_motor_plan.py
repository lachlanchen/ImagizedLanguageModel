#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import signal
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    VisualGrammarRecord,
    load_visual_grammar_manifest,
)
from ilm.visual_lm.predictive_visual_field import PredictiveVisualField
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.visual_actuator import visual_actuator_retrieval_metrics
from ilm.visual_lm.visual_motor_plan import (
    ContinuousVisualMotorPlan,
    VisualMotorPlanConfig,
    evaluate_visual_motor_plan_batch,
    visual_motor_plan_config_from_payload,
    visual_motor_plan_config_payload,
    visual_motor_plan_loss,
)
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    save_sample_grid,
    scheduled_lr,
    select_examples,
)


ARCHITECTURE = "continuous-visual-motor-plan-v1"
V17_ARCHITECTURE = "continuous-visual-state-actuator-v1"
SELECTION_RULE = {
    "primary": "maximize development correct_pixel_f1",
    "requirements": [
        "correct_pixel_f1 > 0.60",
        "correct_pixel_f1 - shuffled_pixel_f1 > 0.15",
        "correct_identity_top1 > shuffled_identity_top1",
        "correct_target_cosine > 0.60",
        "correct_target_cosine > shuffled_target_cosine",
        "condition_pixel_l1 > 0.05",
    ],
    "human_readability_review_required_before_frozen_evaluation": True,
    "frozen_partition_read_during_selection": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a topology-first image actuator from continuous retinal intent "
            "and a separate style image, without IDs or target pixels in the condition."
        )
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--warmstart-v17", default=None)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_motor_plan_v18")
    parser.add_argument("--partition-salt", default="visual-motor-plan-v18")
    parser.add_argument("--holdout-fraction", type=float, default=0.06)
    parser.add_argument("--development-share", type=float, default=0.50)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--positions-per-sequence", type=int, default=4)
    parser.add_argument("--samples-per-epoch", type=int, default=30_000)
    parser.add_argument("--development-samples", type=int, default=128)
    parser.add_argument("--style-dim", type=int, default=64)
    parser.add_argument("--style-base-channels", type=int, default=32)
    parser.add_argument("--plan-base-channels", type=int, default=128)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--pixel-l1-weight", type=float, default=0.50)
    parser.add_argument("--edge-weight", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-weight", type=float, default=0.05)
    parser.add_argument("--state-margin-weight", type=float, default=0.10)
    parser.add_argument("--state-margin", type=float, default=0.03)
    parser.add_argument("--duplicate-similarity", type=float, default=0.90)
    parser.add_argument("--logit-scale", type=float, default=12.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--maximum-steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
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


def _partition_value(identifier: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def partition_records(
    records: Sequence[VisualGrammarRecord],
    *,
    salt: str,
    holdout_fraction: float,
    development_share: float,
) -> tuple[list[VisualGrammarRecord], list[VisualGrammarRecord], list[VisualGrammarRecord]]:
    if not salt:
        raise ValueError("visual motor-plan partition salt must not be empty")
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be in (0, 0.5)")
    if not 0.0 < development_share < 1.0:
        raise ValueError("development_share must be in (0, 1)")
    development_cutoff = holdout_fraction * development_share
    train: list[VisualGrammarRecord] = []
    development: list[VisualGrammarRecord] = []
    frozen: list[VisualGrammarRecord] = []
    for record in records:
        value = _partition_value(record.identifier, salt)
        if value >= holdout_fraction:
            train.append(record)
        elif value < development_cutoff:
            development.append(record)
        else:
            frozen.append(record)
    if not train or not development or not frozen:
        raise ValueError("visual motor-plan partition produced an empty split")
    return train, development, frozen


def partition_receipt(
    train: Sequence[VisualGrammarRecord],
    development: Sequence[VisualGrammarRecord],
    frozen: Sequence[VisualGrammarRecord],
    *,
    salt: str,
    holdout_fraction: float,
    development_share: float,
) -> dict[str, Any]:
    return {
        "algorithm": "sha256(salt + NUL + identifier) first 64 bits",
        "salt": salt,
        "holdout_fraction": holdout_fraction,
        "development_share_of_holdout": development_share,
        "train_records": len(train),
        "development_records": len(development),
        "frozen_records": len(frozen),
        "frozen_identifiers_sha256": hashlib.sha256(
            "\n".join(sorted(record.identifier for record in frozen)).encode("utf-8")
        ).hexdigest(),
        "frozen_images_instantiated_during_training": False,
        "frozen_evaluator_permitted_during_development": False,
    }


def selection_eligible(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["correct_pixel_f1"] > 0.60
        and metrics["correct_pixel_f1"] - metrics["shuffled_pixel_f1"] > 0.15
        and metrics["correct_identity_top1"] > metrics["shuffled_identity_top1"]
        and metrics["correct_target_cosine"] > 0.60
        and metrics["correct_target_cosine"] > metrics["shuffled_target_cosine"]
        and metrics["condition_pixel_l1"] > 0.05
    )


def load_v17_style(planner: ContinuousVisualMotorPlan, path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != V17_ARCHITECTURE:
        raise ValueError("style warm-start must be a V17 continuous actuator")
    state = {
        key.removeprefix("style_encoder."): value
        for key, value in checkpoint["actuator"].items()
        if key.startswith("style_encoder.")
    }
    planner.style_encoder.load_state_dict(state, strict=True)
    return {
        "path": path,
        "sha256": file_sha256(path),
        "step": int(checkpoint.get("global_step", 0)),
        "loaded_component": "style_encoder_only",
    }


@torch.no_grad()
def validate(
    pvf: PredictiveVisualField,
    planner: ContinuousVisualMotorPlan,
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
            loss, training_metrics, _ = visual_motor_plan_loss(
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
                state_margin_weight=args.state_margin_weight,
                state_margin=args.state_margin,
                duplicate_similarity=args.duplicate_similarity,
                logit_scale=args.logit_scale,
            )
            causal_metrics, trace = evaluate_visual_motor_plan_batch(
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
        for key in retrieval_states:
            retrieval_states[key].append(trace[key].detach())
        batch_examples = target.shape[0]
        metrics = {
            "loss": loss.detach(),
            **{f"plan_{key}": value for key, value in training_metrics.items()},
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
    planner.train()
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
    report["selection_eligible"] = float(selection_eligible(report))
    return report


def checkpoint_payload(
    planner: ContinuousVisualMotorPlan,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    pvf_checkpoint: dict[str, Any],
    partition: dict[str, Any],
    warmstart: dict[str, Any] | None,
    best_development: dict[str, Any] | None,
    epoch: int,
    step: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "planner_config": visual_motor_plan_config_payload(planner.config),
        "planner": planner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": file_sha256(args.pvf_checkpoint),
        "pvf_architecture": pvf_checkpoint["architecture"],
        "pvf_step": int(pvf_checkpoint.get("global_step", 0)),
        "pvf_model_config": pvf_checkpoint["model_config"],
        "render_config": pvf_checkpoint["render_config"],
        "style_warmstart": warmstart,
        "partition": partition,
        "epoch": epoch,
        "global_step": step,
        "elapsed_seconds": elapsed_seconds,
        "arguments": vars(args),
        "selection_rule": SELECTION_RULE,
        "best_development": best_development,
        "student_contract": {
            "input": "continuous intended retinal state plus continuous style image",
            "output": "continuous spatial ink plan",
            "target_spatial_pixels_enter_condition": False,
            "student_received_token_ids": False,
            "student_received_unicode_ids": False,
            "student_received_ocr": False,
            "student_received_character_labels": False,
            "student_used_visual_codebook": False,
            "student_used_candidate_classifier": False,
            "student_used_external_language_model": False,
            "stochastic_pixel_sampler_required": False,
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
    protocol = {
        "architecture": ARCHITECTURE,
        "selection_rule": SELECTION_RULE,
        "partition": partition,
        "frozen_evaluation_status": "forbidden until development plus human review pass",
    }
    protocol_path = output / "preregistered_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("existing preregistered motor-plan protocol differs")
    else:
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "partition.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    planner = ContinuousVisualMotorPlan(
        VisualMotorPlanConfig(
            fovea_size=pvf.config.fovea_size,
            visual_dim=pvf.config.visual_dim,
            style_dim=args.style_dim,
            style_base_channels=args.style_base_channels,
            plan_base_channels=args.plan_base_channels,
            context_dim=args.context_dim,
            dropout=args.dropout,
        )
    )
    warmstart = load_v17_style(planner, args.warmstart_v17) if args.warmstart_v17 else None
    planner.to(device)
    optimizer = torch.optim.AdamW(
        planner.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    global_step = 0
    start_epoch = 0
    elapsed_before = 0.0
    best_development: dict[str, Any] | None = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume.get("architecture") != ARCHITECTURE:
            raise ValueError("resume is not a visual motor-plan checkpoint")
        if resume["partition"] != partition:
            raise ValueError("resume uses a different visual motor-plan partition")
        if resume.get("pvf_sha256") != file_sha256(args.pvf_checkpoint):
            raise ValueError("resume uses a different Predictive Visual Field")
        planner = ContinuousVisualMotorPlan(
            visual_motor_plan_config_from_payload(resume["planner_config"])
        )
        planner.load_state_dict(resume["planner"])
        planner.to(device)
        optimizer = torch.optim.AdamW(
            planner.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        optimizer.load_state_dict(resume["optimizer"])
        global_step = int(resume.get("global_step", 0))
        start_epoch = int(resume.get("epoch", 0))
        elapsed_before = float(resume.get("elapsed_seconds", 0.0))
        best_development = resume.get("best_development")
        warmstart = resume.get("style_warmstart")

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
        "trainable_parameters": sum(
            parameter.numel() for parameter in planner.parameters() if parameter.requires_grad
        ),
        "frozen_pvf_parameters": sum(parameter.numel() for parameter in pvf.parameters()),
        "classifier_parameters": 0,
        "token_embedding_parameters": 0,
        "device": str(device),
        "planned_steps": planned_steps,
        "sequence_length": args.sequence_length,
        "positions_per_sequence": args.positions_per_sequence,
        "partition": partition,
        "style_warmstart": warmstart,
        "selection_rule": SELECTION_RULE,
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
                loss, metrics, _ = visual_motor_plan_loss(
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
                    state_margin_weight=args.state_margin_weight,
                    state_margin=args.state_margin,
                    duplicate_similarity=args.duplicate_similarity,
                    logit_scale=args.logit_scale,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                planner.parameters(),
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

            validation: dict[str, float] | None = None
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
                eligible = selection_eligible(validation)
                if eligible and (
                    best_development is None
                    or validation["correct_pixel_f1"]
                    > float(best_development["correct_pixel_f1"])
                ):
                    best_development = {
                        "step": global_step,
                        "correct_pixel_f1": validation["correct_pixel_f1"],
                        "correct_identity_top1": validation["correct_identity_top1"],
                        "correct_target_cosine": validation["correct_target_cosine"],
                        "shuffled_pixel_f1": validation["shuffled_pixel_f1"],
                        "shuffled_identity_top1": validation["shuffled_identity_top1"],
                        "shuffled_target_cosine": validation["shuffled_target_cosine"],
                        "condition_pixel_l1": validation["condition_pixel_l1"],
                    }
                    elapsed = elapsed_before + time.perf_counter() - started
                    payload = checkpoint_payload(
                        planner,
                        optimizer,
                        args=args,
                        pvf_checkpoint=pvf_checkpoint,
                        partition=partition,
                        warmstart=warmstart,
                        best_development=best_development,
                        epoch=epoch,
                        step=global_step,
                        elapsed_seconds=elapsed,
                    )
                    atomic_save(payload, output / "checkpoint_selected_development.pt")
                    (output / "development_selection.json").write_text(
                        json.dumps(
                            {
                                "selection_rule": SELECTION_RULE,
                                "selected": best_development,
                                "frozen_evaluation_permitted": False,
                                "reason": "human readability review still required",
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
                    partition=partition,
                    warmstart=warmstart,
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
        partition=partition,
        warmstart=warmstart,
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
    }
    print(json.dumps(final), flush=True)
    append_jsonl(log_path, final)


if __name__ == "__main__":
    main()
