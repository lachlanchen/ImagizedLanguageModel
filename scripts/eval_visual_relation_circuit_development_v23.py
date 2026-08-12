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
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.visual_relation_circuit import (
    OPERATION_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    RELATION_AWARE_ROUTE,
    VisualCanonicalizer,
    VisualRelationCircuit,
    relation_circuit_config_from_payload,
)
from ilm.visual_lm.visual_relation_data import (
    PARTITION_SALT,
    VisualRelationEpisodeConfig,
    VisualRelationEpisodeDataset,
    build_relation_character_bank,
    relation_partition_receipt,
    split_relation_characters,
    visual_relation_collate,
)
from scripts.train_visual_relation_circuit_v23 import (
    ARCHITECTURE,
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    PROTOCOL_DOCUMENT,
    SOURCE_FILES,
    _development_bank_images,
    _parameter_shapes,
    _trainable_parameters,
    candidate_selection_gate_report,
    control_selection_gate_report,
    encode_identity_bank,
    evaluate_development,
    load_relation_state,
    paired_gate_report,
    student_boundary_is_clean,
    validate_canonicalizer_checkpoint,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


AUDIT_ARCHITECTURE = "visual-relation-circuit-v23-paired-audit"
AUDIT_EPISODES = 1_024
AUDIT_BATCH_SIZE = 64
AUDIT_IDENTITY_BANK_VIEWS = 4
AUDIT_SEED = FIXED_OPTIMIZATION_ARGUMENTS["dataset_seed"] + 2_000_003
EXPECTED_ROUTES = (
    RELATION_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sealed fresh V23 candidate/control development audit."
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--query-blind", required=True)
    parser.add_argument("--operation-blind", required=True)
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--canonicalizer-checkpoint", required=True)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument(
        "--out",
        default="artifacts/visual_relation_circuit_v23_paired_audit",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _selected_gate_report(
    metrics: dict[str, float],
    route_mode: str,
) -> dict[str, bool]:
    if route_mode == RELATION_AWARE_ROUTE:
        return candidate_selection_gate_report(metrics)
    return control_selection_gate_report(metrics, route_mode)


def validate_selected_arm(
    checkpoint: dict[str, Any],
    *,
    route_mode: str,
) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{route_mode} checkpoint has the wrong architecture")
    if checkpoint.get("route_mode") != route_mode:
        raise ValueError(f"{route_mode} checkpoint has the wrong route mode")
    if checkpoint.get("smoke_only"):
        raise ValueError(f"{route_mode} checkpoint is smoke-only")
    best = checkpoint.get("best_development")
    if not isinstance(best, dict):
        raise ValueError(f"{route_mode} checkpoint has no selected development result")
    if int(checkpoint.get("step", -1)) != int(best.get("step", -2)):
        raise ValueError(f"{route_mode} checkpoint is not its selected endpoint")
    gates = _selected_gate_report(best, route_mode)
    if not all(gates.values()):
        failed = sorted(key for key, passed in gates.items() if not passed)
        raise ValueError(f"{route_mode} selected gates fail: {failed}")
    if checkpoint.get("trainable_parameters") != EXPECTED_PARAMETERS:
        raise ValueError(f"{route_mode} parameter count differs")
    if not student_boundary_is_clean(checkpoint["boundary_receipt"], route_mode):
        raise ValueError(f"{route_mode} student boundary differs")
    if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
        raise ValueError(f"{route_mode} PVF receipt differs")
    if checkpoint.get("canonicalizer_sha256") != EXPECTED_CANONICALIZER_SHA256:
        raise ValueError(f"{route_mode} canonicalizer receipt differs")
    if checkpoint.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"{route_mode} manifest receipt differs")
    for key, expected in EXPECTED_PARTITION.items():
        if checkpoint.get("partition", {}).get(key) != expected:
            raise ValueError(f"{route_mode} partition differs for {key}")
    protocol = checkpoint.get("protocol", {})
    expected_protocol = {
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
        "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
        "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
        "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{route_mode} protocol differs for {key}")
    for path in SOURCE_FILES:
        if protocol.get("source_files_sha256", {}).get(path) != file_sha256(path):
            raise ValueError(f"{route_mode} source hash differs for {path}")


def validate_pair_metadata(checkpoints: dict[str, dict[str, Any]]) -> None:
    if tuple(checkpoints) != EXPECTED_ROUTES:
        raise ValueError("V23 paired audit requires candidate, query, and operation arms")
    for route_mode, checkpoint in checkpoints.items():
        validate_selected_arm(checkpoint, route_mode=route_mode)
    reference = checkpoints[RELATION_AWARE_ROUTE]
    reference_shapes = reference["trainable_parameter_shapes"]
    for route_mode, checkpoint in checkpoints.items():
        if checkpoint["trainable_parameter_shapes"] != reference_shapes:
            raise ValueError(f"{route_mode} trainable parameter shapes differ")
        if checkpoint["retinal_fonts"] != reference["retinal_fonts"]:
            raise ValueError(f"{route_mode} retinal font receipt differs")
        if checkpoint["protocol"]["canonicalizer_protocol"] != reference[
            "protocol"
        ]["canonicalizer_protocol"]:
            raise ValueError(f"{route_mode} Stage A protocol receipt differs")
        if checkpoint["partition"] != reference["partition"]:
            raise ValueError(f"{route_mode} full partition receipt differs")
    configs = {
        route_mode: relation_circuit_config_from_payload(
            checkpoint["model_config"]
        )
        for route_mode, checkpoint in checkpoints.items()
    }
    reference_config = configs[RELATION_AWARE_ROUTE]
    for route_mode, config in configs.items():
        if config.route_mode != route_mode:
            raise ValueError(f"{route_mode} serialized config has wrong route")
        if config.__class__(
            **{**config.__dict__, "route_mode": RELATION_AWARE_ROUTE}
        ) != reference_config:
            raise ValueError(f"{route_mode} model configuration differs")


def _load_model(
    checkpoint: dict[str, Any],
    *,
    retina: torch.nn.Module,
    canonicalizer_state: dict[str, torch.Tensor],
    device: torch.device,
) -> VisualRelationCircuit:
    canonicalizer = VisualCanonicalizer()
    canonicalizer.load_state_dict(canonicalizer_state)
    config = relation_circuit_config_from_payload(checkpoint["model_config"])
    model = VisualRelationCircuit(config, retina, canonicalizer).to(device).eval()
    load_relation_state(model, checkpoint["relation"])
    if _trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError(f"loaded {config.route_mode} parameter count differs")
    if _parameter_shapes(model) != checkpoint["trainable_parameter_shapes"]:
        raise ValueError(f"loaded {config.route_mode} parameter shapes differ")
    return model


def main() -> None:
    args = parse_args()
    if args.precision != "bf16":
        raise ValueError("V23 paired evidence requires BF16")
    if args.num_workers != 8:
        raise ValueError("V23 paired evidence requires 8 data workers")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty V23 audit output: {output_dir}")

    checkpoint_paths = {
        RELATION_AWARE_ROUTE: args.candidate,
        QUERY_BLIND_ROUTE: args.query_blind,
        OPERATION_BLIND_ROUTE: args.operation_blind,
    }
    checkpoints = {
        route_mode: torch.load(path, map_location="cpu", weights_only=False)
        for route_mode, path in checkpoint_paths.items()
    }

    # This validation must finish before manifest loading or image construction.
    validate_pair_metadata(checkpoints)
    pvf_sha256 = file_sha256(args.pvf_checkpoint)
    if pvf_sha256 != EXPECTED_PVF_SHA256:
        raise ValueError("paired audit PVF file hash differs")
    manifest_sha256 = file_sha256(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("paired audit manifest file hash differs")
    canonicalizer_sha256 = file_sha256(args.canonicalizer_checkpoint)
    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=canonicalizer_sha256,
    )
    if retinal_font_manifest() != checkpoints[RELATION_AWARE_ROUTE][
        "retinal_fonts"
    ]:
        raise ValueError("paired audit retinal font files differ")

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)
    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    retina = pvf.retina
    del pvf
    models = {
        route_mode: _load_model(
            checkpoint,
            retina=retina,
            canonicalizer_state=canonicalizer_checkpoint["canonicalizer"],
            device=device,
        )
        for route_mode, checkpoint in checkpoints.items()
    }

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_relation_character_bank(
        records,
        bank_size=FIXED_MODEL_ARGUMENTS["bank_size"],
    )
    partitions = split_relation_characters(bank, salt=PARTITION_SALT)
    partition = relation_partition_receipt(partitions, salt=PARTITION_SALT)
    if partition != checkpoints[RELATION_AWARE_ROUTE]["partition"]:
        raise ValueError("fresh audit partition differs after construction")
    episode_config = VisualRelationEpisodeConfig()
    dataset = VisualRelationEpisodeDataset(
        partitions["development"],
        split="development",
        length=AUDIT_EPISODES,
        config=episode_config,
        seed=AUDIT_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=AUDIT_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_relation_collate,
    )
    bank_images = _development_bank_images(
        partitions["development"],
        views=AUDIT_IDENTITY_BANK_VIEWS,
        config=episode_config,
        seed=AUDIT_SEED + 100_000,
    ).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(
            models[RELATION_AWARE_ROUTE], bank_images
        )
    del bank_images

    start = time.perf_counter()
    metrics = {
        route_mode: evaluate_development(
            model,
            loader,
            bank_visual=bank_visual,
            bank_characters=partitions["development"],
            device=device,
            precision=args.precision,
        )
        for route_mode, model in models.items()
    }
    parameter_shapes_equal = len(
        {
            json.dumps(
                checkpoints[route_mode]["trainable_parameter_shapes"],
                sort_keys=True,
            )
            for route_mode in EXPECTED_ROUTES
        }
    ) == 1
    gates = paired_gate_report(
        metrics[RELATION_AWARE_ROUTE],
        metrics[QUERY_BLIND_ROUTE],
        metrics[OPERATION_BLIND_ROUTE],
        candidate_parameters=checkpoints[RELATION_AWARE_ROUTE][
            "trainable_parameters"
        ],
        query_blind_parameters=checkpoints[QUERY_BLIND_ROUTE][
            "trainable_parameters"
        ],
        operation_blind_parameters=checkpoints[OPERATION_BLIND_ROUTE][
            "trainable_parameters"
        ],
        parameter_shapes_equal=parameter_shapes_equal,
    )
    passed = all(gates.values())
    report = {
        "architecture": AUDIT_ARCHITECTURE,
        "stage": "complete",
        "audit_seed": AUDIT_SEED,
        "audit_episodes": AUDIT_EPISODES,
        "batch_size": AUDIT_BATCH_SIZE,
        "identity_bank_views": AUDIT_IDENTITY_BANK_VIEWS,
        "device": str(device),
        "precision": args.precision,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint_sha256": {
            route_mode: file_sha256(path)
            for route_mode, path in checkpoint_paths.items()
        },
        "pvf_sha256": pvf_sha256,
        "canonicalizer_sha256": canonicalizer_sha256,
        "manifest_sha256": manifest_sha256,
        "partition": partition,
        "metrics": metrics,
        "paired_gates": gates,
        "paired_gate_passed": passed,
        "blinded_review_permitted": passed,
        "blinded_review_passed": False,
        "frozen_evaluation_permitted": False,
        "frozen_images_instantiated": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "paired_development_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
