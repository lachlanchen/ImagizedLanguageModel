#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import RetinalRenderConfig, load_visual_grammar_manifest
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from ilm.visual_lm.visual_actuator import visual_actuator_retrieval_metrics
from ilm.visual_lm.visual_motor_plan import (
    ContinuousVisualMotorPlan,
    evaluate_visual_motor_plan_batch,
    visual_motor_plan_config_from_payload,
)
from scripts.train_visual_motor_plan import (
    ARCHITECTURE,
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    partition_receipt,
    partition_records,
    save_sample_grid,
    select_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit one selected visual motor plan on fresh development renderings. "
            "This evaluator cannot access the frozen split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument(
        "--sample-columns",
        type=int,
        default=8,
        help="Examples per human-review page; the full-width contact sheet is also saved.",
    )
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
        page = {key: value[start:stop] for key, value in trace.items()}
        name = f"{stem}_{page_index:02d}.png"
        save_sample_grid(page, output / name, stop - start)
        names.append(name)
    return names


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sample_count < 1 or args.sample_columns < 1:
        raise ValueError("development audit sample counts must be positive")
    output = Path(args.out)
    evaluation_path = output / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"refusing to overwrite motor-plan development audit: {evaluation_path}"
        )
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a continuous visual motor plan")
    pvf_path = checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != checkpoint["pvf_sha256"]:
        raise ValueError("Predictive Visual Field bytes differ from the training receipt")
    pvf, pvf_checkpoint = load_pvf(pvf_path, device)
    planner = ContinuousVisualMotorPlan(
        visual_motor_plan_config_from_payload(checkpoint["planner_config"])
    )
    planner.load_state_dict(checkpoint["planner"])
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
        raise ValueError("development audit partition differs from checkpoint receipt")

    render_config = RetinalRenderConfig(**pvf_checkpoint["render_config"])
    audit_seed = int(training_args["seed"]) + 1_500_007
    dataset = VisualSaccadeDataset(
        development_records,
        render_config=render_config,
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
    totals: dict[str, float] = {}
    examples = 0
    traces: list[dict[str, torch.Tensor]] = []
    retrieval: dict[str, list[torch.Tensor]] = {
        "correct_visual": [],
        "shuffled_output_visual": [],
        "intended_visual": [],
    }
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
                metrics, trace = evaluate_visual_motor_plan_batch(
                    planner,
                    pvf.retina,
                    target,
                    semantic,
                    style,
                    duplicate_similarity=float(training_args["duplicate_similarity"]),
                    logit_scale=float(training_args["logit_scale"]),
                )
            count = target.shape[0]
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * count
            examples += count
            for key in retrieval:
                retrieval[key].append(trace[key].detach())
            if sum(item["target_ink"].shape[0] for item in traces) < args.sample_count:
                traces.append(
                    {
                        key: value.detach().cpu()
                        for key, value in trace.items()
                        if value.ndim >= 1 and value.shape[0] == count
                    }
                )
    elapsed_seconds = time.perf_counter() - started
    metrics = {key: value / max(1, examples) for key, value in totals.items()}
    global_retrieval = visual_actuator_retrieval_metrics(
        torch.cat(retrieval["correct_visual"]),
        torch.cat(retrieval["shuffled_output_visual"]),
        torch.cat(retrieval["intended_visual"]),
        duplicate_similarity=float(training_args["duplicate_similarity"]),
        logit_scale=float(training_args["logit_scale"]),
    )
    metrics.update({key: float(value) for key, value in global_retrieval.items()})

    sample_trace = {
        key: torch.cat([trace[key] for trace in traces])[: args.sample_count]
        for key in (
            "target_ink",
            "semantic_reference",
            "style_reference",
            "correct_ink",
            "shuffled_ink",
        )
    }
    threshold_trace = dict(sample_trace)
    threshold_trace["correct_ink"] = (sample_trace["correct_ink"] >= 0.5).float()
    threshold_trace["shuffled_ink"] = (sample_trace["shuffled_ink"] >= 0.5).float()
    output.mkdir(parents=True, exist_ok=True)
    continuous_sheet = "development_samples_continuous.png"
    thresholded_sheet = "development_samples_thresholded.png"
    save_sample_grid(sample_trace, output / continuous_sheet, args.sample_count)
    save_sample_grid(threshold_trace, output / thresholded_sheet, args.sample_count)
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

    payload: dict[str, Any] = {
        "architecture": "continuous-visual-motor-plan-development-audit-v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "pvf_checkpoint": pvf_path,
        "pvf_sha256": checkpoint["pvf_sha256"],
        "partition": "development",
        "partition_receipt": partition,
        "frozen_images_instantiated": False,
        "audit_seed": audit_seed,
        "samples": args.samples,
        "positions_per_sequence": int(training_args["positions_per_sequence"]),
        "retrieval_candidates": examples,
        "metrics": metrics,
        "sample_artifacts": sample_artifacts,
        "human_readability_review": "pending",
        "student_contract": checkpoint["student_contract"],
        "elapsed_seconds": elapsed_seconds,
        "generated_examples_per_second": examples / elapsed_seconds if elapsed_seconds else None,
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
