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
from ilm.visual_lm.field_complete_writer import (
    FIELD_COMPLETE_ROUTE,
    TILED_GLOBAL_CONTROL_ROUTE,
    FieldCompleteWriter,
    FieldCompleteWriterConfig,
    evaluate_field_complete_writer_batch,
    field_complete_writer_config_from_payload,
    summarize_field_complete_writer_trace,
)
from ilm.visual_lm.saccade_data import (
    SaccadeSequenceSpec,
    VisualSaccadeDataset,
    visual_saccade_collate,
)
from scripts.train_field_complete_writer import (
    ARCHITECTURE,
    EXPECTED_PARAMETERS,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    paired_gate_report,
    save_sample_grid,
    selection_gate_report,
    selection_rule,
)
from scripts.train_visual_motor_plan import partition_receipt, partition_records
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    select_examples,
)


AUDIT_ARCHITECTURE = "field-complete-writer-paired-development-audit-v1"
AUDIT_SEED_OFFSET = 2_000_003
PAIRED_ARGUMENT_NAMES = tuple(
    sorted(
        {
            *FIXED_MODEL_ARGUMENTS,
            *FIXED_LOSS_ARGUMENTS,
            *FIXED_OPTIMIZATION_ARGUMENTS,
            *FIXED_EVIDENCE_ARGUMENTS,
            "manifest",
            "partition_salt",
        }
    )
)
FORBIDDEN_STUDENT_FLAGS = (
    "learned_or_fixed_position_input",
    "spatial_cell_mixing",
    "target_spatial_pixels_enter_condition",
    "student_received_token_ids",
    "student_received_unicode_ids",
    "student_received_ocr",
    "student_received_character_labels",
    "student_used_visual_codebook",
    "student_used_candidate_classifier",
    "student_used_external_language_model",
    "retina_trainable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit selected V21 field and matched-control checkpoints on the "
            "same fresh development renderings. Frozen data remain inaccessible."
        )
    )
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pvf-checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--sample-columns", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    return parser.parse_args()


def _validate_student_contract(
    checkpoint: dict[str, Any],
    route_mode: str,
) -> None:
    contract = checkpoint.get("student_contract", {})
    if contract.get("route_mode") != route_mode:
        raise ValueError("V21 checkpoint has an inconsistent student route")
    for flag in FORBIDDEN_STUDENT_FLAGS:
        if contract.get(flag) is not False:
            raise ValueError("V21 checkpoint violates the image-only student contract")
    if contract.get("global_state_enters_uniform_modulation") is not True:
        raise ValueError("V21 checkpoint omits the shared global modulation")
    global_source = contract.get("global_state_enters_spatial_source")
    if global_source is not (route_mode == TILED_GLOBAL_CONTROL_ROUTE):
        raise ValueError("V21 checkpoint misstates its local-source condition")


def validate_pair_metadata(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> tuple[FieldCompleteWriterConfig, FieldCompleteWriterConfig]:
    for name, checkpoint, route_mode in (
        ("candidate", candidate, FIELD_COMPLETE_ROUTE),
        ("control", control, TILED_GLOBAL_CONTROL_ROUTE),
    ):
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError(f"{name} is not a V21 field-complete-writer checkpoint")
        if checkpoint.get("route_mode") != route_mode:
            raise ValueError(f"{name} has the wrong V21 route mode")
        if checkpoint.get("smoke_only", False):
            raise ValueError("smoke-only V21 checkpoints cannot produce evidence")
        if checkpoint.get("best_development") is None:
            raise ValueError(f"{name} did not pass its arm-specific development gate")
        if checkpoint.get("frozen_images_instantiated_during_training", True):
            raise ValueError(f"{name} does not preserve the sealed-frozen contract")
        if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
            raise ValueError(f"{name} does not use the preregistered V16 PVF")
        if checkpoint.get("selection_rule") != selection_rule(route_mode):
            raise ValueError(f"{name} selection rule differs from V21")
        basis = checkpoint.get("patch_basis", {})
        if (
            basis.get("name") != "walsh-hadamard-zero-dc"
            or basis.get("shape") != [64, 63]
            or basis.get("trainable") is not False
            or basis.get("dc_leakage_max") != 0.0
        ):
            raise ValueError(f"{name} has an invalid V21 patch-basis receipt")
        _validate_student_contract(checkpoint, route_mode)

    if candidate.get("partition") != control.get("partition"):
        raise ValueError("V21 arms use different partitions")
    if candidate.get("pvf_sha256") != control.get("pvf_sha256"):
        raise ValueError("V21 arms use different Predictive Visual Fields")
    for argument in PAIRED_ARGUMENT_NAMES:
        candidate_value = candidate.get("arguments", {}).get(argument)
        control_value = control.get("arguments", {}).get(argument)
        if candidate_value != control_value:
            raise ValueError(f"V21 arms differ on fixed argument {argument!r}")

    candidate_config = field_complete_writer_config_from_payload(
        candidate["writer_config"]
    )
    control_config = field_complete_writer_config_from_payload(
        control["writer_config"]
    )
    candidate_base = dict(candidate["writer_config"])
    control_base = dict(control["writer_config"])
    candidate_base.pop("route_mode", None)
    control_base.pop("route_mode", None)
    if candidate_base != control_base:
        raise ValueError("V21 arms do not have equal architecture shapes")
    if candidate_config.route_mode != FIELD_COMPLETE_ROUTE:
        raise ValueError("candidate writer configuration is not field-primary")
    if control_config.route_mode != TILED_GLOBAL_CONTROL_ROUTE:
        raise ValueError("control writer configuration is not tiled-global")
    return candidate_config, control_config


def _load_writer(
    checkpoint: dict[str, Any],
    config: FieldCompleteWriterConfig,
    device: torch.device,
) -> FieldCompleteWriter:
    writer = FieldCompleteWriter(config)
    writer.load_state_dict(checkpoint["writer"], strict=True)
    return writer.to(device).eval().requires_grad_(False)


def _parameter_count(writer: FieldCompleteWriter) -> int:
    return sum(parameter.numel() for parameter in writer.parameters())


def _threshold_trace(trace: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(trace)
    for key in (
        "correct_ink",
        "field_shuffled_ink",
        "zero_field_ink",
        "global_shuffled_ink",
        "both_shuffled_ink",
        "occluded_q0_ink",
    ):
        result[key] = (trace[key] >= 0.5).float()
    return result


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
        save_sample_grid(page, output / name, stop - start)
        names.append(name)
    return names


def _save_arm_artifacts(
    trace: dict[str, torch.Tensor],
    output: Path,
    *,
    prefix: str,
    count: int,
    columns: int,
) -> dict[str, Any]:
    sample = {
        key: value[:count].cpu()
        for key, value in trace.items()
        if value.ndim >= 1 and value.shape[0] == trace["target_ink"].shape[0]
    }
    threshold = _threshold_trace(sample)
    continuous_sheet = f"{prefix}_continuous.png"
    thresholded_sheet = f"{prefix}_thresholded.png"
    save_sample_grid(sample, output / continuous_sheet, count)
    save_sample_grid(threshold, output / thresholded_sheet, count)
    return {
        "continuous_sheet": continuous_sheet,
        "continuous_pages": save_review_pages(
            sample,
            output,
            stem=f"{prefix}_continuous_page",
            count=count,
            columns=columns,
        ),
        "thresholded_sheet": thresholded_sheet,
        "thresholded_pages": save_review_pages(
            threshold,
            output,
            stem=f"{prefix}_thresholded_page",
            count=count,
            columns=columns,
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sample_count < 1 or args.sample_columns < 1:
        raise ValueError("paired development audit counts must be positive")
    output = Path(args.out)
    evaluation_path = output / "evaluation.json"
    if evaluation_path.exists():
        raise FileExistsError(f"refusing to overwrite V21 audit: {evaluation_path}")

    device = choose_device(args.device)
    candidate_checkpoint = torch.load(
        args.candidate_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    control_checkpoint = torch.load(
        args.control_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    candidate_config, control_config = validate_pair_metadata(
        candidate_checkpoint,
        control_checkpoint,
    )
    candidate = _load_writer(candidate_checkpoint, candidate_config, device)
    control = _load_writer(control_checkpoint, control_config, device)
    candidate_parameters = _parameter_count(candidate)
    control_parameters = _parameter_count(control)
    if candidate_parameters != control_parameters:
        raise ValueError("V21 candidate and control parameter counts differ")
    if candidate_parameters != EXPECTED_PARAMETERS:
        raise ValueError("V21 arms do not have the preregistered parameter count")

    pvf_path = args.pvf_checkpoint or candidate_checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != EXPECTED_PVF_SHA256:
        raise ValueError("Predictive Visual Field bytes differ from the V21 receipt")
    pvf, pvf_checkpoint = load_pvf(pvf_path, device)
    training_args = candidate_checkpoint["arguments"]
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
    if partition != candidate_checkpoint["partition"]:
        raise ValueError("paired audit partition differs from the V21 receipt")

    audit_seed = int(training_args["seed"]) + AUDIT_SEED_OFFSET
    dataset = VisualSaccadeDataset(
        development_records,
        render_config=RetinalRenderConfig(**pvf_checkpoint["render_config"]),
        spec=SaccadeSequenceSpec(
            sequence_length=int(training_args["sequence_length"]),
            fovea_size=candidate.config.fovea_size,
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
    generator = torch.Generator(device=device).manual_seed(audit_seed + 97)
    traces: dict[str, dict[str, list[torch.Tensor]]] = {
        FIELD_COMPLETE_ROUTE: {},
        TILED_GLOBAL_CONTROL_ROUTE: {},
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
            for route_mode, writer in (
                (FIELD_COMPLETE_ROUTE, candidate),
                (TILED_GLOBAL_CONTROL_ROUTE, control),
            ):
                with autocast_context(device, args.precision):
                    _, trace = evaluate_field_complete_writer_batch(
                        writer,
                        pvf.retina,
                        target,
                        semantic,
                        style,
                        duplicate_similarity=float(
                            training_args["duplicate_similarity"]
                        ),
                        logit_scale=float(training_args["logit_scale"]),
                    )
                for key, value in trace.items():
                    if key != "detail_basis":
                        traces[route_mode].setdefault(key, []).append(value.detach())
    elapsed_seconds = time.perf_counter() - started
    aggregates = {
        route_mode: {key: torch.cat(values) for key, values in trace.items()}
        for route_mode, trace in traces.items()
    }
    aggregates[FIELD_COMPLETE_ROUTE]["detail_basis"] = candidate.detail_basis.detach()
    aggregates[TILED_GLOBAL_CONTROL_ROUTE]["detail_basis"] = (
        control.detail_basis.detach()
    )
    metrics: dict[str, dict[str, float]] = {}
    for route_mode, aggregate in aggregates.items():
        metrics[route_mode] = {
            key: float(value)
            for key, value in summarize_field_complete_writer_trace(
                aggregate,
                field_size=candidate.config.field_size,
                duplicate_similarity=float(training_args["duplicate_similarity"]),
                logit_scale=float(training_args["logit_scale"]),
            ).items()
        }
        metrics[route_mode]["frozen_images_instantiated"] = 0.0

    arm_gate_reports = {
        route_mode: selection_gate_report(route_metrics, route_mode)
        for route_mode, route_metrics in metrics.items()
    }
    paired_report = paired_gate_report(
        metrics[FIELD_COMPLETE_ROUTE],
        metrics[TILED_GLOBAL_CONTROL_ROUTE],
        candidate_parameters=candidate_parameters,
        control_parameters=control_parameters,
    )
    candidate_passed = all(arm_gate_reports[FIELD_COMPLETE_ROUTE].values())
    control_passed = all(arm_gate_reports[TILED_GLOBAL_CONTROL_ROUTE].values())
    paired_passed = candidate_passed and control_passed and all(paired_report.values())

    output.mkdir(parents=True, exist_ok=True)
    sample_count = min(
        args.sample_count,
        int(aggregates[FIELD_COMPLETE_ROUTE]["target_ink"].shape[0]),
    )
    sample_artifacts = {
        route_mode: _save_arm_artifacts(
            aggregates[route_mode],
            output,
            prefix=route_mode,
            count=sample_count,
            columns=args.sample_columns,
        )
        for route_mode in (FIELD_COMPLETE_ROUTE, TILED_GLOBAL_CONTROL_ROUTE)
    }
    examples = int(aggregates[FIELD_COMPLETE_ROUTE]["target_ink"].shape[0])
    payload: dict[str, Any] = {
        "architecture": AUDIT_ARCHITECTURE,
        "candidate_checkpoint": args.candidate_checkpoint,
        "candidate_checkpoint_sha256": file_sha256(args.candidate_checkpoint),
        "candidate_checkpoint_step": int(candidate_checkpoint["global_step"]),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": file_sha256(args.control_checkpoint),
        "control_checkpoint_step": int(control_checkpoint["global_step"]),
        "candidate_parameters": candidate_parameters,
        "control_parameters": control_parameters,
        "pvf_checkpoint": pvf_path,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "partition": "development",
        "partition_receipt": partition,
        "frozen_images_instantiated": False,
        "frozen_evaluation_permitted": False,
        "audit_seed": audit_seed,
        "source_records": args.samples,
        "positions_per_sequence": int(training_args["positions_per_sequence"]),
        "retrieval_candidates": examples,
        "metrics": metrics,
        "arm_gate_reports": arm_gate_reports,
        "arm_automatic_development_gate_passed": {
            FIELD_COMPLETE_ROUTE: candidate_passed,
            TILED_GLOBAL_CONTROL_ROUTE: control_passed,
        },
        "paired_gate_report": paired_report,
        "paired_automatic_development_gate_passed": paired_passed,
        "sample_artifacts": sample_artifacts,
        "human_readability_review": (
            "authorized but not completed"
            if paired_passed
            else "not authorized after automatic-gate failure"
        ),
        "student_contract": candidate_checkpoint["student_contract"],
        "elapsed_seconds": elapsed_seconds,
        "generated_examples_per_second": (
            2 * examples / elapsed_seconds if elapsed_seconds else None
        ),
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
