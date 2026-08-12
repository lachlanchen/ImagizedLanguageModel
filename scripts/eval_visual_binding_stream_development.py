#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.visual_binding_data import (
    PARTITION_SALT,
    VisualBindingEpisodeConfig,
    VisualBindingEpisodeDataset,
    binding_partition_receipt,
    build_binding_character_bank,
    split_binding_characters,
    visual_binding_collate,
)
from ilm.visual_lm.visual_binding_stream import (
    QUERY_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    VisualBindingStream,
    encode_identity_bank,
    visual_binding_config_from_payload,
)
from scripts.train_visual_binding_stream import (
    ARCHITECTURE,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    PROTOCOL_DOCUMENT,
    _development_bank_images,
    _parameter_shapes,
    _trainable_parameters,
    candidate_selection_gate_report,
    evaluate_development,
    load_student_state,
    paired_gate_report,
    save_sample_sheet,
    student_boundary_is_clean,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


AUDIT_ARCHITECTURE = "visual-binding-stream-paired-development-audit-v1"
AUDIT_DATASET_SEED_OFFSET = 2_000_003
AUDIT_IDENTITY_SEED_OFFSET = 3_000_017
AUDIT_SAMPLES = 1_024
AUDIT_BATCH_SIZE = 64
AUDIT_IDENTITY_VIEWS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit selected V22 query-aware and query-blind checkpoints on one "
            "fresh paired development render set. Frozen images stay sealed."
        )
    )
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pvf-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--sample-count", type=int, default=16)
    return parser.parse_args()


def _load_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_arm(
    checkpoint: dict[str, Any],
    *,
    name: str,
    route_mode: str,
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{name} is not a V22 visual binding checkpoint")
    if checkpoint.get("route_mode") != route_mode:
        raise ValueError(f"{name} has the wrong V22 route mode")
    if checkpoint.get("smoke_only", False):
        raise ValueError("smoke-only V22 checkpoints cannot produce evidence")
    if checkpoint.get("best_development") is None:
        raise ValueError(f"{name} has no selected development checkpoint")
    if int(checkpoint.get("step", -1)) != int(
        checkpoint["best_development"]["step"]
    ):
        raise ValueError(f"{name} is not its selected development checkpoint")
    if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
        raise ValueError(f"{name} does not use the preregistered PVF")
    if checkpoint.get("trainable_parameters") != EXPECTED_PARAMETERS:
        raise ValueError(f"{name} has the wrong trainable parameter count")
    if tuple(checkpoint.get("retinal_fonts", ())) != retinal_font_manifest():
        raise ValueError(f"{name} retinal font manifest differs")
    if not student_boundary_is_clean(checkpoint.get("boundary_receipt", {}), route_mode):
        raise ValueError(f"{name} violates the image-only student boundary")
    if checkpoint.get("protocol", {}).get("protocol_sha256") != file_sha256(
        PROTOCOL_DOCUMENT
    ):
        raise ValueError(f"{name} protocol document hash differs")
    fixed_groups = (
        ("fixed_model_arguments", FIXED_MODEL_ARGUMENTS),
        ("fixed_loss_arguments", FIXED_LOSS_ARGUMENTS),
        ("fixed_optimization_arguments", FIXED_OPTIMIZATION_ARGUMENTS),
        ("fixed_evidence_arguments", FIXED_EVIDENCE_ARGUMENTS),
    )
    for receipt_key, expected_arguments in fixed_groups:
        if checkpoint.get("protocol", {}).get(receipt_key) != expected_arguments:
            raise ValueError(f"{name} {receipt_key} receipt differs from V22")
        for argument, expected in expected_arguments.items():
            if checkpoint.get("args", {}).get(argument) != expected:
                raise ValueError(
                    f"{name} actual argument {argument!r} differs from V22"
                )
    for key, expected in EXPECTED_PARTITION.items():
        if checkpoint.get("partition", {}).get(key) != expected:
            raise ValueError(f"{name} partition differs at {key!r}")


def validate_pair_metadata(
    candidate: dict[str, Any],
    control: dict[str, Any],
) -> None:
    _validate_arm(candidate, name="candidate", route_mode=QUERY_AWARE_ROUTE)
    _validate_arm(control, name="control", route_mode=QUERY_BLIND_ROUTE)
    if candidate["partition"] != control["partition"]:
        raise ValueError("V22 candidate and control use different partitions")
    if candidate["pvf_sha256"] != control["pvf_sha256"]:
        raise ValueError("V22 candidate and control use different PVF checkpoints")
    if candidate.get("manifest_sha256") != control.get("manifest_sha256"):
        raise ValueError("V22 candidate and control use different manifests")
    paired_arguments = {
        *FIXED_MODEL_ARGUMENTS,
        *FIXED_LOSS_ARGUMENTS,
        *FIXED_OPTIMIZATION_ARGUMENTS,
        *FIXED_EVIDENCE_ARGUMENTS,
        "manifest",
        "partition_salt",
        "pvf_checkpoint",
    }
    for argument in paired_arguments:
        if candidate.get("args", {}).get(argument) != control.get("args", {}).get(
            argument
        ):
            raise ValueError(f"V22 arms differ on argument {argument!r}")
    candidate_config = dict(candidate["model_config"])
    control_config = dict(control["model_config"])
    candidate_config.pop("route_mode", None)
    control_config.pop("route_mode", None)
    if candidate_config != control_config:
        raise ValueError("V22 candidate and control architecture shapes differ")
    if candidate.get("trainable_parameter_shapes") != control.get(
        "trainable_parameter_shapes"
    ):
        raise ValueError("V22 candidate and control parameter shapes differ")


def _load_model(
    checkpoint: dict[str, Any],
    retina: torch.nn.Module,
    device: torch.device,
) -> VisualBindingStream:
    config = visual_binding_config_from_payload(checkpoint["model_config"])
    model = VisualBindingStream(config, retina)
    load_student_state(model, checkpoint["student"])
    return model.to(device).eval()


def _encode_bank(
    model: VisualBindingStream,
    images: torch.Tensor,
    *,
    device: torch.device,
    precision: str,
) -> torch.Tensor:
    with torch.no_grad(), autocast_context(device, precision):
        return encode_identity_bank(model, images.to(device))


def _evaluate_arm(
    model: VisualBindingStream,
    loader: DataLoader,
    *,
    bank_visual: torch.Tensor,
    bank_characters: Sequence[str],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    return evaluate_development(
        model,
        loader,
        bank_visual=bank_visual,
        bank_characters=bank_characters,
        device=device,
        precision=precision,
    )


def main() -> None:
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("V22 audit sample count must be positive")
    output = Path(args.out)
    evaluation_path = output / "evaluation.json"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite V22 audit: {output}")

    candidate_checkpoint = _load_checkpoint(args.candidate_checkpoint)
    control_checkpoint = _load_checkpoint(args.control_checkpoint)
    validate_pair_metadata(candidate_checkpoint, control_checkpoint)

    manifest_path = args.manifest or candidate_checkpoint["args"]["manifest"]
    if file_sha256(manifest_path) != candidate_checkpoint["manifest_sha256"]:
        raise ValueError("V22 audit manifest bytes differ from training")
    pvf_path = args.pvf_checkpoint or candidate_checkpoint["pvf_checkpoint"]
    if file_sha256(pvf_path) != EXPECTED_PVF_SHA256:
        raise ValueError("V22 audit PVF bytes differ from preregistration")

    seed = int(candidate_checkpoint["args"]["seed"])
    dataset_seed = int(candidate_checkpoint["args"]["dataset_seed"])
    seed_everything(seed + AUDIT_DATASET_SEED_OFFSET)
    device = choose_device(args.device)
    pvf, _ = load_pvf(pvf_path, device)
    candidate = _load_model(candidate_checkpoint, pvf.retina, device)
    control = _load_model(control_checkpoint, pvf.retina, device)

    candidate_parameters = _trainable_parameters(candidate)
    control_parameters = _trainable_parameters(control)
    candidate_shapes = _parameter_shapes(candidate)
    control_shapes = _parameter_shapes(control)
    if candidate_parameters != EXPECTED_PARAMETERS:
        raise ValueError("V22 candidate parameter count changed during load")
    if control_parameters != EXPECTED_PARAMETERS:
        raise ValueError("V22 control parameter count changed during load")
    if candidate_shapes != control_shapes:
        raise ValueError("V22 loaded arm parameter shapes differ")

    records = load_visual_grammar_manifest(manifest_path)
    bank = build_binding_character_bank(
        records,
        bank_size=int(candidate_checkpoint["args"]["bank_size"]),
    )
    partitions = split_binding_characters(bank, salt=PARTITION_SALT)
    partition = binding_partition_receipt(partitions, salt=PARTITION_SALT)
    if partition != candidate_checkpoint["partition"]:
        raise ValueError("V22 fresh audit partition differs from training receipt")

    episode_config = VisualBindingEpisodeConfig()
    audit_seed = dataset_seed + AUDIT_DATASET_SEED_OFFSET
    dataset = VisualBindingEpisodeDataset(
        partitions["development"],
        split="development",
        length=AUDIT_SAMPLES,
        config=episode_config,
        seed=audit_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=AUDIT_BATCH_SIZE,
        shuffle=False,
        num_workers=int(FIXED_EVIDENCE_ARGUMENTS["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=visual_binding_collate,
    )
    bank_images = _development_bank_images(
        partitions["development"],
        views=AUDIT_IDENTITY_VIEWS,
        config=episode_config,
        seed=dataset_seed + AUDIT_IDENTITY_SEED_OFFSET,
    )
    candidate_bank = _encode_bank(
        candidate,
        bank_images,
        device=device,
        precision=args.precision,
    )
    control_bank = _encode_bank(
        control,
        bank_images,
        device=device,
        precision=args.precision,
    )
    del bank_images

    started = time.perf_counter()
    candidate_metrics = _evaluate_arm(
        candidate,
        loader,
        bank_visual=candidate_bank,
        bank_characters=partitions["development"],
        device=device,
        precision=args.precision,
    )
    control_metrics = _evaluate_arm(
        control,
        loader,
        bank_visual=control_bank,
        bank_characters=partitions["development"],
        device=device,
        precision=args.precision,
    )
    elapsed_seconds = time.perf_counter() - started
    candidate_gates = candidate_selection_gate_report(candidate_metrics)
    paired_gates = paired_gate_report(
        candidate_metrics,
        control_metrics,
        candidate_parameters=candidate_parameters,
        control_parameters=control_parameters,
        parameter_shapes_equal=candidate_shapes == control_shapes,
    )
    paired_passed = all(candidate_gates.values()) and all(paired_gates.values())

    output.mkdir(parents=True, exist_ok=False)
    save_sample_sheet(
        candidate,
        loader,
        path=output / "candidate_counterfactual_samples.png",
        device=device,
        precision=args.precision,
        sample_count=args.sample_count,
    )
    save_sample_sheet(
        control,
        loader,
        path=output / "control_counterfactual_samples.png",
        device=device,
        precision=args.precision,
        sample_count=args.sample_count,
    )
    payload: dict[str, Any] = {
        "architecture": AUDIT_ARCHITECTURE,
        "candidate_checkpoint": args.candidate_checkpoint,
        "candidate_checkpoint_sha256": file_sha256(args.candidate_checkpoint),
        "candidate_checkpoint_step": int(candidate_checkpoint["step"]),
        "control_checkpoint": args.control_checkpoint,
        "control_checkpoint_sha256": file_sha256(args.control_checkpoint),
        "control_checkpoint_step": int(control_checkpoint["step"]),
        "pvf_checkpoint": pvf_path,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "manifest": manifest_path,
        "manifest_sha256": candidate_checkpoint["manifest_sha256"],
        "partition": "development",
        "partition_receipt": partition,
        "audit_seed": audit_seed,
        "identity_bank_seed": dataset_seed + AUDIT_IDENTITY_SEED_OFFSET,
        "paired_episodes": AUDIT_SAMPLES,
        "identity_bank_views": AUDIT_IDENTITY_VIEWS,
        "candidate_parameters": candidate_parameters,
        "control_parameters": control_parameters,
        "parameter_shapes_equal": candidate_shapes == control_shapes,
        "metrics": {
            QUERY_AWARE_ROUTE: candidate_metrics,
            QUERY_BLIND_ROUTE: control_metrics,
        },
        "candidate_gate_report": candidate_gates,
        "paired_gate_report": paired_gates,
        "paired_automatic_development_gate_passed": paired_passed,
        "frozen_images_instantiated": False,
        "frozen_evaluation_permitted": False,
        "human_readability_review": (
            "authorized but not completed"
            if paired_passed
            else "not authorized after automatic-gate failure"
        ),
        "elapsed_seconds": elapsed_seconds,
        "sample_artifacts": {
            QUERY_AWARE_ROUTE: "candidate_counterfactual_samples.png",
            QUERY_BLIND_ROUTE: "control_counterfactual_samples.png",
        },
    }
    evaluation_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
