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
from ilm.visual_lm.visual_actuator import (
    ContinuousVisualActuator,
    evaluate_visual_actuator_batch,
    visual_actuator_config_from_payload,
    visual_actuator_retrieval_metrics,
)
from scripts.train_visual_state_actuator import (
    ARCHITECTURE,
    FROZEN_ACCEPTANCE_RULE,
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
            "Evaluate one selected visual actuator once on its untouched frozen split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("frozen evaluation samples must be positive")
    output = Path(args.out)
    evaluation_path = output / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(
            f"refusing to reread frozen actuator evaluation: {evaluation_path} exists"
        )
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("checkpoint is not a continuous visual-state actuator")
    pvf_path = checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != checkpoint["pvf_sha256"]:
        raise ValueError("Predictive Visual Field bytes differ from the training receipt")
    pvf, pvf_checkpoint = load_pvf(pvf_path, device)
    actuator = ContinuousVisualActuator(
        visual_actuator_config_from_payload(checkpoint["actuator_config"])
    )
    actuator.load_state_dict(checkpoint["actuator"])
    actuator.to(device).eval().requires_grad_(False)

    training_args = checkpoint["arguments"]
    manifest_path = args.manifest or training_args["manifest"]
    records = load_visual_grammar_manifest(manifest_path)
    train_records, development_records, frozen_records = partition_records(
        records,
        holdout_fraction=float(training_args["holdout_fraction"]),
        development_share=float(training_args["development_share"]),
    )
    partition = partition_receipt(
        train_records,
        development_records,
        frozen_records,
        holdout_fraction=float(training_args["holdout_fraction"]),
        development_share=float(training_args["development_share"]),
    )
    expected_partition = checkpoint["partition"]
    for key in (
        "train_records",
        "development_records",
        "frozen_records",
        "frozen_identifiers_sha256",
    ):
        if partition[key] != expected_partition[key]:
            raise ValueError(f"frozen actuator partition changed at {key}")

    render_config = RetinalRenderConfig(**pvf_checkpoint["render_config"])
    dataset = VisualSaccadeDataset(
        frozen_records,
        render_config=render_config,
        spec=SaccadeSequenceSpec(
            sequence_length=int(training_args["sequence_length"]),
            fovea_size=actuator.config.fovea_size,
        ),
        split="all",
        length=args.samples,
        seed=int(training_args["seed"]) + 2_000_003,
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
    seed = int(training_args["seed"]) + int(checkpoint["global_step"]) * 1_000_003
    generator = torch.Generator(device=device).manual_seed(seed)
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
                metrics, trace = evaluate_visual_actuator_batch(
                    actuator,
                    pvf.retina,
                    target,
                    semantic,
                    style,
                    steps=int(training_args["eval_steps"]),
                    guidance_scale=float(training_args["guidance_scale"]),
                    duplicate_similarity=float(training_args["duplicate_similarity"]),
                    logit_scale=float(training_args["logit_scale"]),
                    generator=generator,
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
    output.mkdir(parents=True, exist_ok=True)
    save_sample_grid(sample_trace, output / "frozen_samples.png", args.sample_count)

    gates = {
        "correct_identity_top1_above_shuffled": (
            metrics["correct_identity_top1"] > metrics["shuffled_identity_top1"]
        ),
        "target_cosine_gain": (
            metrics["target_cosine_gain"]
            >= FROZEN_ACCEPTANCE_RULE["target_cosine_gain_minimum"]
        ),
        "correct_target_cosine": (
            metrics["correct_target_cosine"]
            >= FROZEN_ACCEPTANCE_RULE["correct_target_cosine_minimum"]
        ),
        "correct_pixel_f1": (
            metrics["correct_pixel_f1"]
            >= FROZEN_ACCEPTANCE_RULE["correct_pixel_f1_minimum"]
        ),
        "target_above_style_copy": (
            metrics["correct_target_cosine"] - metrics["style_copy_cosine"]
            >= FROZEN_ACCEPTANCE_RULE["correct_target_cosine_above_style_copy_by"]
        ),
    }
    payload: dict[str, Any] = {
        "architecture": "continuous-visual-state-actuator-frozen-evaluation-v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_step": int(checkpoint["global_step"]),
        "pvf_checkpoint": pvf_path,
        "pvf_sha256": checkpoint["pvf_sha256"],
        "partition": partition,
        "samples": args.samples,
        "positions_per_sequence": int(training_args["positions_per_sequence"]),
        "retrieval_candidates": examples,
        "seed": seed,
        "sample_steps": int(training_args["eval_steps"]),
        "same_noise_for_correct_and_shuffled": True,
        "metrics": metrics,
        "automatic_gates": gates,
        "automatic_gates_accepted": all(gates.values()),
        "human_readability_review": "pending",
        "actuator_accepted": False,
        "acceptance_rule": FROZEN_ACCEPTANCE_RULE,
        "student_contract": checkpoint["student_contract"],
        "evaluator_used_labels": False,
        "elapsed_seconds": elapsed_seconds,
        "generated_examples_per_second": (
            2.0 * examples / elapsed_seconds if elapsed_seconds > 0 else None
        ),
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
