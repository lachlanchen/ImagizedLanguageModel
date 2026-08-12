#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.visual_packet_data import (
    PARTITION_SALT,
    VisualPacketEpisodeConfig,
    VisualPacketEpisodeDataset,
    build_packet_character_bank,
    packet_partition_receipt,
    split_packet_characters,
    visual_packet_collate,
)
from ilm.visual_lm.visual_packet_stream import (
    HEADER_BLIND_ROUTE,
    HISTORY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
    PACKET_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    VisualPacketRereadStream,
    visual_packet_stream_config_from_payload,
)
from ilm.visual_lm.visual_relation_circuit import VisualCanonicalizer
from scripts.train_visual_packet_stream_v24 import (
    ARCHITECTURE,
    DEFAULT_CANONICALIZER_CHECKPOINT,
    DEFAULT_PVF_CHECKPOINT,
    DEFAULT_RELATION_CHECKPOINT,
    EXPECTED_CANONICALIZER_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_PARTITION,
    EXPECTED_PVF_SHA256,
    EXPECTED_RELATION_SHA256,
    FIXED_EVIDENCE_ARGUMENTS,
    FIXED_LOSS_ARGUMENTS,
    FIXED_MODEL_ARGUMENTS,
    FIXED_OPTIMIZATION_ARGUMENTS,
    GATE_EPSILON,
    PROTOCOL_DOCUMENT,
    SOURCE_FILES,
    _development_bank_images,
    _label_bank_images,
    _load_v23_relation_parts,
    _parameter_shapes,
    _trainable_parameters,
    candidate_selection_gate_report,
    control_selection_gate_report,
    encode_identity_bank,
    evaluate_development,
    load_packet_state,
    student_boundary_is_clean,
    validate_canonicalizer_checkpoint,
    validate_relation_checkpoint,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    seed_everything,
)


AUDIT_ARCHITECTURE = "visual-packet-reread-stream-v24-paired-audit"
AUDIT_EPISODES = 1_024
AUDIT_BATCH_SIZE = 64
AUDIT_IDENTITY_BANK_VIEWS = 4
AUDIT_SEED = FIXED_OPTIMIZATION_ARGUMENTS["dataset_seed"] + 2_000_003
EXPECTED_ROUTES = (
    PACKET_AWARE_ROUTE,
    HEADER_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    OPERATION_BLIND_ROUTE,
    HISTORY_BLIND_ROUTE,
)
DEFAULT_EVIDENCE_ROOT = Path("artifacts/visual_packet_stream_v24_evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sealed fresh V24 candidate/control development audit."
    )
    parser.add_argument(
        "--candidate",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / PACKET_AWARE_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument(
        "--header-blind",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / HEADER_BLIND_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument(
        "--query-blind",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / QUERY_BLIND_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument(
        "--operation-blind",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / OPERATION_BLIND_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument(
        "--history-blind",
        default=str(
            DEFAULT_EVIDENCE_ROOT
            / HISTORY_BLIND_ROUTE
            / "checkpoint_selected_development.pt"
        ),
    )
    parser.add_argument("--pvf-checkpoint", default=DEFAULT_PVF_CHECKPOINT)
    parser.add_argument(
        "--canonicalizer-checkpoint", default=DEFAULT_CANONICALIZER_CHECKPOINT
    )
    parser.add_argument("--relation-checkpoint", default=DEFAULT_RELATION_CHECKPOINT)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument(
        "--out",
        default="artifacts/visual_packet_stream_v24_paired_audit",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def _minimum_localization(metrics: dict[str, float]) -> float:
    return min(
        metrics["query_header_localization_accuracy"],
        metrics["operation_header_localization_accuracy"],
        metrics["pair_header_localization_accuracy"],
    )


def paired_gate_report(
    metrics: dict[str, dict[str, float]],
    *,
    arm_parameters: dict[str, int],
    parameter_shapes_equal: bool,
    metadata_validated: bool,
) -> dict[str, bool]:
    if tuple(metrics) != EXPECTED_ROUTES:
        raise ValueError("V24 paired metrics must contain all arms in fixed order")
    if tuple(arm_parameters) != EXPECTED_ROUTES:
        raise ValueError("V24 paired parameter counts must contain all arms")
    candidate = metrics[PACKET_AWARE_ROUTE]
    header_blind = metrics[HEADER_BLIND_ROUTE]
    query_blind = metrics[QUERY_BLIND_ROUTE]
    operation_blind = metrics[OPERATION_BLIND_ROUTE]
    history_blind = metrics[HISTORY_BLIND_ROUTE]
    return {
        "candidate_query_switch_gain": _strictly_above(
            candidate["query_switch_accuracy"] - query_blind["query_switch_accuracy"],
            0.40,
        ),
        "candidate_operation_switch_gain": _strictly_above(
            candidate["operation_switch_accuracy"]
            - operation_blind["operation_switch_accuracy"],
            0.40,
        ),
        "candidate_history_switch_gain": _strictly_above(
            candidate["history_switch_accuracy"]
            - history_blind["history_switch_accuracy"],
            0.40,
        ),
        "candidate_role_localization_gain": _strictly_above(
            _minimum_localization(candidate) - _minimum_localization(header_blind),
            0.40,
        ),
        "candidate_frame1_identity_gain_over_query_blind": _strictly_above(
            candidate["frame1_identity_top1"] - query_blind["frame1_identity_top1"],
            0.30,
        ),
        "candidate_frame1_identity_gain_over_operation_blind": _strictly_above(
            candidate["frame1_identity_top1"] - operation_blind["frame1_identity_top1"],
            0.30,
        ),
        "candidate_frame1_identity_gain_over_header_blind": _strictly_above(
            candidate["frame1_identity_top1"] - header_blind["frame1_identity_top1"],
            0.30,
        ),
        "candidate_frame2_label_gain_over_history_blind": _strictly_above(
            candidate["frame2_label_top1"] - history_blind["frame2_label_top1"],
            0.30,
        ),
        "candidate_frame2_label_gain_over_header_blind": _strictly_above(
            candidate["frame2_label_top1"] - header_blind["frame2_label_top1"],
            0.30,
        ),
        "query_blind_intervention_invariant": _strictly_below(
            query_blind["query_output_pixel_l1"], 1e-7
        ),
        "operation_blind_intervention_invariant": _strictly_below(
            operation_blind["operation_output_pixel_l1"], 1e-7
        ),
        "history_blind_intervention_invariant": _strictly_below(
            history_blind["history_output_pixel_l1"], 1e-7
        ),
        "header_blind_intervention_invariant": _strictly_below(
            header_blind["header_output_pixel_l1"], 1e-7
        ),
        "candidate_arm_gates": all(candidate_selection_gate_report(candidate).values()),
        "header_blind_arm_gates": all(
            control_selection_gate_report(header_blind, HEADER_BLIND_ROUTE).values()
        ),
        "query_blind_arm_gates": all(
            control_selection_gate_report(query_blind, QUERY_BLIND_ROUTE).values()
        ),
        "operation_blind_arm_gates": all(
            control_selection_gate_report(
                operation_blind, OPERATION_BLIND_ROUTE
            ).values()
        ),
        "history_blind_arm_gates": all(
            control_selection_gate_report(history_blind, HISTORY_BLIND_ROUTE).values()
        ),
        "metadata_validated": metadata_validated,
        "parameter_count_equal": (
            len(set(arm_parameters.values())) == 1
            and next(iter(arm_parameters.values())) == EXPECTED_PARAMETERS
        ),
        "parameter_shapes_equal": parameter_shapes_equal,
    }


def _selected_gate_report(
    metrics: dict[str, float], route_mode: str
) -> dict[str, bool]:
    if route_mode == PACKET_AWARE_ROUTE:
        return candidate_selection_gate_report(metrics)
    return control_selection_gate_report(metrics, route_mode)


def validate_selected_arm(checkpoint: dict[str, Any], *, route_mode: str) -> None:
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{route_mode} checkpoint has the wrong architecture")
    if checkpoint.get("route_mode") != route_mode:
        raise ValueError(f"{route_mode} checkpoint has the wrong route mode")
    if checkpoint.get("smoke_only"):
        raise ValueError(f"{route_mode} checkpoint is smoke-only")
    best = checkpoint.get("best_development")
    if not isinstance(best, dict):
        raise ValueError(f"{route_mode} checkpoint has no selected result")
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
    expected_hashes = {
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "relation_sha256": EXPECTED_RELATION_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    for key, expected in expected_hashes.items():
        if checkpoint.get(key) != expected:
            raise ValueError(f"{route_mode} receipt differs for {key}")
    for key, expected in EXPECTED_PARTITION.items():
        if checkpoint.get("partition", {}).get(key) != expected:
            raise ValueError(f"{route_mode} partition differs for {key}")
    protocol = checkpoint.get("protocol", {})
    expected_protocol = {
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "route_mode": route_mode,
        "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
        "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
        "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
        "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "expected_pvf_sha256": EXPECTED_PVF_SHA256,
        "expected_canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "expected_relation_sha256": EXPECTED_RELATION_SHA256,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "smoke_only": False,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{route_mode} protocol differs for {key}")
    for path in SOURCE_FILES:
        if protocol.get("source_files_sha256", {}).get(path) != file_sha256(path):
            raise ValueError(f"{route_mode} source hash differs for {path}")


def validate_pair_metadata(checkpoints: dict[str, dict[str, Any]]) -> None:
    if tuple(checkpoints) != EXPECTED_ROUTES:
        raise ValueError("V24 paired audit requires all five fixed arms")
    for route_mode, checkpoint in checkpoints.items():
        validate_selected_arm(checkpoint, route_mode=route_mode)
    reference = checkpoints[PACKET_AWARE_ROUTE]
    for route_mode, checkpoint in checkpoints.items():
        if (
            checkpoint["trainable_parameter_shapes"]
            != reference["trainable_parameter_shapes"]
        ):
            raise ValueError(f"{route_mode} trainable parameter shapes differ")
        if checkpoint["retinal_fonts"] != reference["retinal_fonts"]:
            raise ValueError(f"{route_mode} retinal font receipt differs")
        if checkpoint["partition"] != reference["partition"]:
            raise ValueError(f"{route_mode} full partition receipt differs")
        state_signature = {
            name: list(value.shape)
            for name, value in checkpoint["packet_stream"].items()
        }
        reference_signature = {
            name: list(value.shape)
            for name, value in reference["packet_stream"].items()
        }
        if state_signature != reference_signature:
            raise ValueError(f"{route_mode} serialized state shapes differ")
    configs = {
        route_mode: visual_packet_stream_config_from_payload(checkpoint["model_config"])
        for route_mode, checkpoint in checkpoints.items()
    }
    reference_config = configs[PACKET_AWARE_ROUTE]
    for route_mode, config in configs.items():
        if config.route_mode != route_mode:
            raise ValueError(f"{route_mode} serialized config has wrong route")
        if replace(config, route_mode=PACKET_AWARE_ROUTE) != reference_config:
            raise ValueError(f"{route_mode} model configuration differs")


def _load_model(
    checkpoint: dict[str, Any],
    *,
    retina: torch.nn.Module,
    canonicalizer_state: dict[str, torch.Tensor],
    relation_checkpoint: dict[str, Any],
    device: torch.device,
) -> VisualPacketRereadStream:
    canonicalizer = VisualCanonicalizer()
    canonicalizer.load_state_dict(canonicalizer_state)
    operation_reader, match_temperature = _load_v23_relation_parts(relation_checkpoint)
    config = visual_packet_stream_config_from_payload(checkpoint["model_config"])
    model = (
        VisualPacketRereadStream(
            config,
            retina,
            canonicalizer,
            operation_reader,
            match_temperature=match_temperature,
        )
        .to(device)
        .eval()
    )
    load_packet_state(model, checkpoint["packet_stream"])
    if _trainable_parameters(model) != EXPECTED_PARAMETERS:
        raise ValueError(f"loaded {config.route_mode} parameter count differs")
    if _parameter_shapes(model) != checkpoint["trainable_parameter_shapes"]:
        raise ValueError(f"loaded {config.route_mode} parameter shapes differ")
    if model.boundary_receipt() != checkpoint["boundary_receipt"]:
        raise ValueError(f"loaded {config.route_mode} boundary receipt differs")
    return model


def main() -> None:
    args = parse_args()
    if args.precision != "bf16":
        raise ValueError("V24 paired evidence requires BF16")
    if args.num_workers != 8:
        raise ValueError("V24 paired evidence requires 8 data workers")
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty V24 audit output: {output_dir}")

    checkpoint_paths = {
        PACKET_AWARE_ROUTE: args.candidate,
        HEADER_BLIND_ROUTE: args.header_blind,
        QUERY_BLIND_ROUTE: args.query_blind,
        OPERATION_BLIND_ROUTE: args.operation_blind,
        HISTORY_BLIND_ROUTE: args.history_blind,
    }
    checkpoints = {
        route_mode: torch.load(path, map_location="cpu", weights_only=False)
        for route_mode, path in checkpoint_paths.items()
    }

    # Complete every receipt check before manifest loading or image construction.
    validate_pair_metadata(checkpoints)
    input_hashes = {
        "pvf": file_sha256(args.pvf_checkpoint),
        "canonicalizer": file_sha256(args.canonicalizer_checkpoint),
        "relation": file_sha256(args.relation_checkpoint),
        "manifest": file_sha256(args.manifest),
    }
    expected_hashes = {
        "pvf": EXPECTED_PVF_SHA256,
        "canonicalizer": EXPECTED_CANONICALIZER_SHA256,
        "relation": EXPECTED_RELATION_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
    }
    for name, expected in expected_hashes.items():
        if input_hashes[name] != expected:
            raise ValueError(f"paired audit {name} file hash differs")

    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint, map_location="cpu", weights_only=False
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=input_hashes["canonicalizer"],
    )
    relation_checkpoint = torch.load(
        args.relation_checkpoint, map_location="cpu", weights_only=False
    )
    validate_relation_checkpoint(
        relation_checkpoint, checkpoint_sha256=input_hashes["relation"]
    )
    if retinal_font_manifest() != checkpoints[PACKET_AWARE_ROUTE]["retinal_fonts"]:
        raise ValueError("paired audit retinal font files differ")

    seed_everything(FIXED_OPTIMIZATION_ARGUMENTS["seed"])
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
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
            relation_checkpoint=relation_checkpoint,
            device=device,
        )
        for route_mode, checkpoint in checkpoints.items()
    }

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_packet_character_bank(
        records, bank_size=FIXED_MODEL_ARGUMENTS["bank_size"]
    )
    partitions = split_packet_characters(bank, salt=PARTITION_SALT)
    partition = packet_partition_receipt(partitions, salt=PARTITION_SALT)
    if partition != checkpoints[PACKET_AWARE_ROUTE]["partition"]:
        raise ValueError("fresh audit partition differs after construction")
    episode_config = VisualPacketEpisodeConfig()
    dataset = VisualPacketEpisodeDataset(
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
        collate_fn=visual_packet_collate,
    )
    bank_images = _development_bank_images(
        partitions["development"],
        views=AUDIT_IDENTITY_BANK_VIEWS,
        config=episode_config,
        seed=AUDIT_SEED + 100_000,
    ).to(device)
    label_images = _label_bank_images(episode_config).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(models[PACKET_AWARE_ROUTE], bank_images)
        label_visual = models[PACKET_AWARE_ROUTE].encode_images(label_images)
    del bank_images, label_images

    start = time.perf_counter()
    metrics = {
        route_mode: evaluate_development(
            model,
            loader,
            bank_visual=bank_visual,
            bank_characters=partitions["development"],
            label_visual=label_visual,
            device=device,
            precision=args.precision,
        )
        for route_mode, model in models.items()
    }
    arm_parameters = {
        route_mode: int(checkpoint["trainable_parameters"])
        for route_mode, checkpoint in checkpoints.items()
    }
    parameter_shapes_equal = (
        len(
            {
                json.dumps(checkpoint["trainable_parameter_shapes"], sort_keys=True)
                for checkpoint in checkpoints.values()
            }
        )
        == 1
    )
    gates = paired_gate_report(
        metrics,
        arm_parameters=arm_parameters,
        parameter_shapes_equal=parameter_shapes_equal,
        metadata_validated=True,
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
        "pvf_sha256": input_hashes["pvf"],
        "canonicalizer_sha256": input_hashes["canonicalizer"],
        "relation_sha256": input_hashes["relation"],
        "manifest_sha256": input_hashes["manifest"],
        "partition": partition,
        "arm_parameters": arm_parameters,
        "metrics": metrics,
        "paired_gates": gates,
        "paired_gate_passed": passed,
        "opaque_review_permitted": passed,
        "opaque_review_passed": False,
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
