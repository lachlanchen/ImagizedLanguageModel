#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    RetinalRenderConfig,
    load_visual_grammar_manifest,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.spatial_motor_plan import (
    SpatialRetinalMotorPlan,
    evaluate_spatial_motor_plan_batch,
    spatial_motor_plan_config_from_payload,
    summarize_spatial_motor_plan_trace,
)
from scripts.train_spatial_motor_plan import (
    ARCHITECTURE,
    save_spatial_sample_grid,
    spatial_selection_gate_report,
)
from scripts.train_visual_motor_plan import partition_receipt, partition_records
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    select_examples,
)


AUDIT_ARCHITECTURE = "spatial-retinal-motor-plan-development-audit-v1"
AUDIT_SEED_OFFSET = 1_900_019


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one V19 spatial motor plan on fresh development renderings. "
            "This command cannot instantiate the frozen split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pvf-checkpoint", default=None)
    parser.add_argument("--global-checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-columns", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def save_review_pages(
    trace: dict[str, torch.Tensor],
    output: Path,
    *,
    stem: str,
    count: int,
    columns: int,
) -> list[str]:
    if columns < 1:
        raise ValueError("sample columns must be positive")
    available = min(count, trace["target_ink"].shape[0])
    names: list[str] = []
    for page_index, start in enumerate(range(0, available, columns), start=1):
        stop = min(start + columns, available)
        page = {
            key: value[start:stop]
            for key, value in trace.items()
            if value.ndim >= 1 and value.shape[0] >= stop
        }
        name = f"{stem}_{page_index:02d}.png"
        save_spatial_sample_grid(page, output / name, stop - start)
        names.append(name)
    return names


def _threshold_trace(trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(trace)
    for key in (
        "correct_ink",
        "spatial_shuffled_ink",
        "global_shuffled_ink",
        "both_shuffled_ink",
        "zero_field_ink",
    ):
        result[key] = (trace[key] >= 0.5).float()
    return result


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sample_count < 1 or args.sample_columns < 1:
        raise ValueError("development audit sample counts must be positive")
    output = Path(args.out)
    evaluation_path = output / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite V19 development audit: {evaluation_path}"
        )

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a V19 spatial motor plan")
    if checkpoint.get("smoke_only", False):
        raise ValueError("a smoke-only V19 checkpoint cannot produce evidence")
    if checkpoint.get("frozen_images_instantiated_during_training", True):
        raise ValueError("V19 checkpoint does not preserve the sealed-frozen contract")

    pvf_path = args.pvf_checkpoint or checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != checkpoint["pvf_sha256"]:
        raise ValueError("Predictive Visual Field bytes differ from the V19 receipt")
    global_path = args.global_checkpoint or checkpoint["global_baseline_checkpoint"]
    if file_sha256(global_path) != checkpoint["global_baseline_sha256"]:
        raise ValueError("global baseline bytes differ from the V19 receipt")
    pvf, pvf_checkpoint = load_pvf(pvf_path, device)
    planner = SpatialRetinalMotorPlan(
        spatial_motor_plan_config_from_payload(checkpoint["planner_config"])
    )
    planner.load_state_dict(checkpoint["planner"], strict=True)
    planner.to(device).eval().requires_grad_(False)

    training_args = checkpoint["arguments"]
    manifest_path = args.manifest or training_args["manifest"]
    records = load_visual_grammar_manifest(manifest_path)
    train_records, development_records, frozen_records = partition_records(
        records,
        salt=training_args["partition_salt"],
        holdout_fraction=float(training_args["holdout_fraction"]),
        development_share=float(training_args["development_share"]),
    )
    partition = partition_receipt(
        train_records,
        development_records,
        frozen_records,
        salt=training_args["partition_salt"],
        holdout_fraction=float(training_args["holdout_fraction"]),
        development_share=float(training_args["development_share"]),
    )
    if partition != checkpoint["partition"]:
        raise ValueError("development audit partition differs from the V19 receipt")

    audit_seed = int(training_args["seed"]) + AUDIT_SEED_OFFSET
    dataset = VisualSaccadeDataset(
        development_records,
        render_config=RetinalRenderConfig(**pvf_checkpoint["render_config"]),
        spec=SaccadeSequenceSpec(
            sequence_length=int(training_args["sequence_length"]),
            fovea_size=planner.config.fovea_size,
        ),
        split="all",
        length=args.samples,
        seed=audit_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=visual_saccade_collate,
    )
    generator = torch.Generator(device=device).manual_seed(
        audit_seed + int(checkpoint["global_step"]) * 1_000_003
    )
    traces: dict[str, list[torch.Tensor]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            target, semantic, style = select_examples(
                batch,
                positions_per_sequence=int(training_args["positions_per_sequence"]),
                device=device,
                generator=generator,
            )
            with autocast_context(device, args.precision):
                _, trace = evaluate_spatial_motor_plan_batch(
                    planner,
                    pvf.retina,
                    target,
                    semantic,
                    style,
                    duplicate_similarity=float(training_args["duplicate_similarity"]),
                    logit_scale=float(training_args["logit_scale"]),
                )
            for key, value in trace.items():
                traces.setdefault(key, []).append(value.detach())
    elapsed_seconds = time.perf_counter() - started
    aggregate = {key: torch.cat(values) for key, values in traces.items()}
    metrics = {
        key: float(value)
        for key, value in summarize_spatial_motor_plan_trace(
            aggregate,
            duplicate_similarity=float(training_args["duplicate_similarity"]),
            logit_scale=float(training_args["logit_scale"]),
        ).items()
    }
    metrics["frozen_images_instantiated"] = 0.0
    gate_report = spatial_selection_gate_report(metrics)

    sample_trace = {
        key: value[: args.sample_count].cpu()
        for key, value in aggregate.items()
        if value.ndim >= 1 and value.shape[0] == aggregate["target_ink"].shape[0]
    }
    threshold_trace = _threshold_trace(sample_trace)
    output.mkdir(parents=True, exist_ok=True)
    continuous_sheet = "development_samples_continuous.png"
    thresholded_sheet = "development_samples_thresholded.png"
    save_spatial_sample_grid(sample_trace, output / continuous_sheet, args.sample_count)
    save_spatial_sample_grid(
        threshold_trace,
        output / thresholded_sheet,
        args.sample_count,
    )
    sample_artifacts = {
        "continuous_sheet": continuous_sheet,
        "continuous_pages": save_review_pages(
            sample_trace,
            output,
            stem="development_samples_continuous_page",
            count=args.sample_count,
            columns=args.sample_columns,
        ),
        "thresholded_sheet": thresholded_sheet,
        "thresholded_pages": save_review_pages(
            threshold_trace,
            output,
            stem="development_samples_thresholded_page",
            count=args.sample_count,
            columns=args.sample_columns,
        ),
    }
    examples = int(aggregate["target_ink"].shape[0])
    payload: dict[str, Any] = {
        "architecture": AUDIT_ARCHITECTURE,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "pvf_checkpoint": pvf_path,
        "pvf_sha256": checkpoint["pvf_sha256"],
        "global_baseline_checkpoint": global_path,
        "global_baseline_sha256": checkpoint["global_baseline_sha256"],
        "partition": "development",
        "partition_receipt": partition,
        "frozen_images_instantiated": False,
        "frozen_evaluation_permitted": False,
        "audit_seed": audit_seed,
        "source_records": args.samples,
        "positions_per_sequence": int(training_args["positions_per_sequence"]),
        "retrieval_candidates": examples,
        "metrics": metrics,
        "automatic_gate_report": gate_report,
        "automatic_development_gate_passed": all(gate_report.values()),
        "sample_artifacts": sample_artifacts,
        "human_readability_review": "not authorized after automatic-gate failure",
        "student_contract": checkpoint["student_contract"],
        "elapsed_seconds": elapsed_seconds,
        "generated_examples_per_second": examples / elapsed_seconds,
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
