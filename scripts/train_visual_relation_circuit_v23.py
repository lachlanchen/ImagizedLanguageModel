#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import signal
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.saccade_data import render_glyph_fovea
from ilm.visual_lm.visual_binding_data import noncanonical_variant
from ilm.visual_lm.visual_relation_circuit import (
    OPERATION_BLIND_ROUTE,
    QUERY_BLIND_ROUTE,
    RELATION_AWARE_ROUTE,
    ROUTE_MODES,
    VisualCanonicalizer,
    VisualRelationCircuit,
    VisualRelationCircuitConfig,
    pixel_f1_rows,
    relation_circuit_config_from_payload,
    relation_circuit_config_payload,
    topology_loss,
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
from scripts.train_visual_canonicalizer_v23 import (
    ARCHITECTURE as CANONICALIZER_ARCHITECTURE,
    EXPECTED_PARAMETERS as EXPECTED_CANONICALIZER_PARAMETERS,
    canonicalizer_boundary_is_clean,
    canonicalizer_selection_gate_report,
)
from scripts.train_visual_state_actuator import (
    append_jsonl,
    atomic_save,
    autocast_context,
    choose_device,
    file_sha256,
    load_pvf,
    scheduled_lr,
    seed_everything,
)


ARCHITECTURE = "visual-relation-circuit-v23"
PROTOCOL_DOCUMENT = "references/visual_relation_circuit_v23_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/visual_relation_data.py",
    "ilm/visual_lm/visual_relation_circuit.py",
    "scripts/train_visual_relation_circuit_v23.py",
)
EXPECTED_PVF_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
EXPECTED_MANIFEST_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_CANONICALIZER_SHA256 = (
    "26cf1bab490abe867e7055a679eff6a9e26e81ad78e6cd9694afd3e425c06135"
)
EXPECTED_PARAMETERS = 25_602
EXPECTED_PARTITION = {
    "train_identities": 817,
    "development_identities": 109,
    "frozen_identities": 98,
    "development_identifiers_sha256": (
        "6e89f898a17028125a060deec8249bbf35b4d02f898f716f2f519a29cd314170"
    ),
    "frozen_identifiers_sha256": (
        "206efd6fa2a0e640368a178c61f2f82ee737260afcaed6e97226bfef1f366d0c"
    ),
}
FIXED_MODEL_ARGUMENTS = {
    "bank_size": 1_024,
}
FIXED_LOSS_ARGUMENTS = {
    "stroke_weight": 4.0,
    "generated_visual_weight": 0.10,
}
FIXED_OPTIMIZATION_ARGUMENTS = {
    "lr": 1e-3,
    "minimum_lr_ratio": 0.10,
    "warmup_steps": 25,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "seed": 20260826,
    "dataset_seed": 20260827,
}
FIXED_EVIDENCE_ARGUMENTS = {
    "maximum_steps": 600,
    "batch_size": 64,
    "num_workers": 8,
    "precision": "bf16",
    "development_samples": 512,
    "identity_bank_views": 4,
    "validate_every": 100,
    "save_every": 100,
}
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one preregistered V23 image-only visual relation arm."
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--canonicalizer-checkpoint", required=True)
    parser.add_argument("--route-mode", choices=ROUTE_MODES, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_relation_circuit_v23")
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--generated-visual-weight", type=float, default=0.10)
    parser.add_argument("--maximum-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--dataset-seed", type=int, default=20260827)
    parser.add_argument("--development-samples", type=int, default=512)
    parser.add_argument("--identity-bank-views", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def _require_fixed_arguments(args: argparse.Namespace) -> None:
    if args.partition_salt != PARTITION_SALT:
        raise ValueError(f"V23 requires partition salt {PARTITION_SALT!r}")
    for group in (
        FIXED_MODEL_ARGUMENTS,
        FIXED_LOSS_ARGUMENTS,
        FIXED_OPTIMIZATION_ARGUMENTS,
    ):
        for name, expected in group.items():
            if getattr(args, name) != expected:
                raise ValueError(f"V23 requires --{name.replace('_', '-')}={expected}")
    if args.smoke:
        if not 1 <= args.maximum_steps <= 20:
            raise ValueError("V23 smoke mode is limited to 1--20 optimization steps")
        return
    for name, expected in FIXED_EVIDENCE_ARGUMENTS.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V23 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def candidate_selection_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "binary_choice_accuracy": _strictly_above(
            metrics["binary_choice_accuracy"], 0.95
        ),
        "query_switch_accuracy": _strictly_above(
            metrics["query_switch_accuracy"], 0.90
        ),
        "operation_switch_accuracy": _strictly_above(
            metrics["operation_switch_accuracy"], 0.90
        ),
        "heldout_combination_minimum_switch_accuracy": _strictly_above(
            metrics["heldout_combination_minimum_switch_accuracy"], 0.85
        ),
        "pair_swap_identity_consistency": _strictly_above(
            metrics["pair_swap_identity_consistency"], 0.99
        ),
        "pair_swap_output_pixel_l1": _strictly_below(
            metrics["pair_swap_output_pixel_l1"], 1e-6
        ),
        "identity_top1": _strictly_above(metrics["identity_top1"], 0.75),
        "identity_bank_size": metrics["identity_bank_identities"] >= 96.0,
        "pixel_f1": _strictly_above(metrics["pixel_f1"], 0.68),
        "target_cosine": _strictly_above(metrics["target_cosine"], 0.82),
        "query_output_pixel_l1": _strictly_above(
            metrics["query_output_pixel_l1"], 0.12
        ),
        "operation_output_pixel_l1": _strictly_above(
            metrics["operation_output_pixel_l1"], 0.12
        ),
        "query_label_match_accuracy": _strictly_above(
            metrics["query_label_match_accuracy"], 0.98
        ),
        "operation_gate_accuracy": _strictly_above(
            metrics["operation_gate_accuracy"], 0.98
        ),
        "operation_gate_separation": _strictly_above(
            metrics["operation_gate_separation"], 0.80
        ),
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
    }


def control_selection_gate_report(
    metrics: dict[str, float],
    route_mode: str,
) -> dict[str, bool]:
    if route_mode == QUERY_BLIND_ROUTE:
        intervention = _strictly_below(metrics["query_output_pixel_l1"], 1e-7)
        name = "query_blind_invariant"
    elif route_mode == OPERATION_BLIND_ROUTE:
        intervention = _strictly_below(
            metrics["operation_output_pixel_l1"], 1e-7
        )
        name = "operation_blind_invariant"
    else:
        raise ValueError(f"V23 control gate cannot evaluate {route_mode!r}")
    return {
        name: intervention,
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
    }


def selection_gate_report(
    metrics: dict[str, float],
    route_mode: str,
) -> dict[str, bool]:
    if route_mode == RELATION_AWARE_ROUTE:
        return candidate_selection_gate_report(metrics)
    return control_selection_gate_report(metrics, route_mode)


def selection_rank(metrics: dict[str, float], route_mode: str) -> tuple[float, ...]:
    if route_mode == RELATION_AWARE_ROUTE:
        return (
            min(
                metrics["query_switch_accuracy"],
                metrics["operation_switch_accuracy"],
            ),
            metrics["identity_top1"],
            metrics["pixel_f1"],
            -metrics["step"],
        )
    if route_mode in {QUERY_BLIND_ROUTE, OPERATION_BLIND_ROUTE}:
        return (metrics["pixel_f1"], -metrics["step"])
    raise ValueError(f"unknown V23 route mode {route_mode!r}")


def paired_gate_report(
    candidate: dict[str, float],
    query_blind: dict[str, float],
    operation_blind: dict[str, float],
    *,
    candidate_parameters: int,
    query_blind_parameters: int,
    operation_blind_parameters: int,
    parameter_shapes_equal: bool,
) -> dict[str, bool]:
    return {
        "candidate_query_switch_gain": _strictly_above(
            candidate["query_switch_accuracy"]
            - query_blind["query_switch_accuracy"],
            0.40,
        ),
        "candidate_operation_switch_gain": _strictly_above(
            candidate["operation_switch_accuracy"]
            - operation_blind["operation_switch_accuracy"],
            0.40,
        ),
        "candidate_identity_gain_over_query_blind": _strictly_above(
            candidate["identity_top1"] - query_blind["identity_top1"], 0.30
        ),
        "candidate_identity_gain_over_operation_blind": _strictly_above(
            candidate["identity_top1"]
            - operation_blind["identity_top1"],
            0.30,
        ),
        "candidate_query_output_gain": _strictly_above(
            candidate["query_output_pixel_l1"]
            - query_blind["query_output_pixel_l1"],
            0.10,
        ),
        "candidate_operation_output_gain": _strictly_above(
            candidate["operation_output_pixel_l1"]
            - operation_blind["operation_output_pixel_l1"],
            0.10,
        ),
        "candidate_arm_gates": all(
            candidate_selection_gate_report(candidate).values()
        ),
        "query_blind_arm_gates": all(
            control_selection_gate_report(
                query_blind, QUERY_BLIND_ROUTE
            ).values()
        ),
        "operation_blind_arm_gates": all(
            control_selection_gate_report(
                operation_blind, OPERATION_BLIND_ROUTE
            ).values()
        ),
        "parameter_count_equal": (
            candidate_parameters
            == query_blind_parameters
            == operation_blind_parameters
        ),
        "parameter_shapes_equal": parameter_shapes_equal,
    }


def student_boundary_is_clean(
    receipt: dict[str, bool | str],
    route_mode: str,
) -> bool:
    required_true = {
        "input_is_continuous_image",
        "output_is_continuous_image",
        "uses_frame_positions",
    }
    required_false = {
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_ocr",
        "uses_character_labels",
        "uses_operation_ids",
        "uses_slot_indices",
        "uses_target_indices",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "retina_trainable",
        "canonicalizer_trainable",
    }
    return (
        receipt.get("architecture") == ARCHITECTURE
        and receipt.get("route_mode") == route_mode
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def _trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _parameter_shapes(model: torch.nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def relation_state_dict(model: VisualRelationCircuit) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(("retina.", "canonicalizer."))
    }


def load_relation_state(
    model: VisualRelationCircuit,
    state: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"V23 relation state has unexpected keys: {incompatible.unexpected_keys}"
        )
    if any(
        not key.startswith(("retina.", "canonicalizer."))
        for key in incompatible.missing_keys
    ):
        raise ValueError(
            f"V23 relation state is incomplete: {incompatible.missing_keys}"
        )


def _device_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _visual_variants(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts = torch.cat(
        (
            batch["prompt"],
            batch["query_counterfactual_prompt"],
            batch["operation_counterfactual_prompt"],
            batch["pair_swapped_prompt"],
        )
    )
    targets = torch.cat(
        (
            batch["target"],
            batch["query_counterfactual_target"],
            batch["operation_counterfactual_target"],
            batch["pair_swapped_target"],
        )
    )
    distractors = torch.cat(
        (
            batch["distractor_target"],
            batch["target"],
            batch["target"],
            batch["distractor_target"],
        )
    )
    if not torch.is_floating_point(prompts):
        raise TypeError("V23 model input must remain a continuous image tensor")
    return prompts, targets, distractors


def relation_loss(
    model: VisualRelationCircuit,
    batch: dict[str, Any],
    *,
    stroke_weight: float,
    generated_visual_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prompts, targets, _ = _visual_variants(batch)
    logits_stream, trace = model.logits_with_trace(prompts)
    logits = logits_stream[:, 0]
    topology, topology_metrics = topology_loss(
        logits,
        targets,
        stroke_weight=stroke_weight,
    )
    generated = logits.sigmoid()
    with torch.no_grad():
        target_visual = model.encode_images(targets)
    generated_visual = model.encode_images(generated)
    generated_visual_loss = (
        1.0 - (generated_visual * target_visual).sum(dim=1)
    ).mean()
    total = topology + generated_visual_weight * generated_visual_loss
    return total, {
        "loss": total.detach(),
        **topology_metrics,
        "generated_visual_loss": generated_visual_loss.detach(),
        "temperature": model.temperature().detach(),
        "match_weight_max": trace["match_weights"].max(dim=1).values.mean().detach(),
        "same_gate_mean": trace["same_gate"].mean().detach(),
    }


def _development_bank_images(
    characters: Sequence[str],
    *,
    views: int,
    config: VisualRelationEpisodeConfig,
    seed: int,
) -> torch.Tensor:
    render_config = config.source_render_config()
    return torch.stack(
        [
            render_glyph_fovea(
                character,
                render_config=render_config,
                fovea_size=config.fovea_size,
                variant=noncanonical_variant(
                    random.Random(seed + owner * 10_007 + view * 1_000_003),
                    config.canonical_target_variant,
                ),
            )
            for owner, character in enumerate(characters)
            for view in range(views)
        ]
    ).reshape(len(characters), views, 1, config.fovea_size, config.fovea_size)


@torch.no_grad()
def encode_identity_bank(
    model: VisualRelationCircuit,
    images: torch.Tensor,
) -> torch.Tensor:
    identities, views = images.shape[:2]
    visual = model.encode_images(images.flatten(0, 1))
    return visual.reshape(identities, views, -1)


@torch.no_grad()
def evaluate_development(
    model: VisualRelationCircuit,
    loader: DataLoader,
    *,
    bank_visual: torch.Tensor,
    bank_characters: Sequence[str],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    bank_visual = F.normalize(bank_visual.float(), dim=-1)
    bank_index = {
        character: index for index, character in enumerate(bank_characters)
    }
    totals: dict[str, float] = {
        "examples": 0.0,
        "pairs": 0.0,
        "heldout_pairs": 0.0,
        "binary_choice_correct": 0.0,
        "query_switch_correct": 0.0,
        "operation_switch_correct": 0.0,
        "heldout_query_switch_correct": 0.0,
        "heldout_operation_switch_correct": 0.0,
        "pair_swap_identity_consistent": 0.0,
        "pair_swap_output_pixel_l1_sum": 0.0,
        "identity_correct": 0.0,
        "pixel_f1_sum": 0.0,
        "target_cosine_sum": 0.0,
        "query_output_pixel_l1_sum": 0.0,
        "operation_output_pixel_l1_sum": 0.0,
        "query_label_match_correct": 0.0,
        "operation_gate_correct": 0.0,
        "same_gate_sum": 0.0,
        "same_gate_count": 0.0,
        "other_gate_sum": 0.0,
        "other_gate_count": 0.0,
        "route_finite": 1.0,
        "route_sum_error_max": 0.0,
    }
    for raw_batch in loader:
        batch = _device_batch(raw_batch, device)
        prompts, targets, distractors = _visual_variants(batch)
        batch_size = batch["prompt"].shape[0]
        with autocast_context(device, precision):
            logits_stream, trace = model.logits_with_trace(prompts)
            generated = logits_stream[:, 0].sigmoid()
            generated_visual = model.encode_images(generated)
            target_visual = model.encode_images(targets)
            distractor_visual = model.encode_images(distractors)

        target_cosine = (generated_visual * target_visual).sum(dim=1)
        distractor_cosine = (generated_visual * distractor_visual).sum(dim=1)
        choice = target_cosine > distractor_cosine
        choice_original, choice_query, choice_operation, _ = choice.split(batch_size)
        query_pair_correct = choice_original & choice_query
        operation_pair_correct = choice_original & choice_operation

        identity_scores = torch.einsum(
            "bd,nvd->bnv", generated_visual, bank_visual
        ).amax(dim=2)
        predicted_identity = identity_scores.argmax(dim=1)
        expected_identity = torch.tensor(
            [bank_index[item["target_character"]] for item in batch["metadata"]]
            + [
                bank_index[item["counterfactual_target_character"]]
                for item in batch["metadata"]
            ]
            + [
                bank_index[item["counterfactual_target_character"]]
                for item in batch["metadata"]
            ]
            + [bank_index[item["target_character"]] for item in batch["metadata"]],
            device=device,
        )
        original_identity, _, _, swapped_identity = predicted_identity.split(
            batch_size
        )

        query_expected = torch.tensor(
            [int(item["query_index"]) for item in batch["metadata"]]
            + [1 - int(item["query_index"]) for item in batch["metadata"]]
            + [int(item["query_index"]) for item in batch["metadata"]]
            + [1 - int(item["query_index"]) for item in batch["metadata"]],
            device=device,
        )
        match_choice = trace["match_weights"].argmax(dim=1)

        original_same = torch.tensor(
            [item["operation"] == "同" for item in batch["metadata"]],
            device=device,
            dtype=torch.bool,
        )
        operation_same = torch.cat(
            (original_same, original_same, ~original_same, original_same)
        )
        same_gate = trace["same_gate"][:, 0].float()
        operation_choice = same_gate > 0.5
        heldout = torch.tensor(
            [bool(item["heldout_combination"]) for item in batch["metadata"]],
            device=device,
        )

        generated_original, generated_query, generated_operation, generated_swap = (
            generated.split(batch_size)
        )
        pair_l1 = (
            generated_original - generated_swap
        ).abs().flatten(1).mean(dim=1)
        query_l1 = (
            generated_original - generated_query
        ).abs().flatten(1).mean(dim=1)
        operation_l1 = (
            generated_original - generated_operation
        ).abs().flatten(1).mean(dim=1)

        totals["examples"] += float(generated.shape[0])
        totals["pairs"] += float(batch_size)
        totals["heldout_pairs"] += float(heldout.sum())
        totals["binary_choice_correct"] += float(choice.sum())
        totals["query_switch_correct"] += float(query_pair_correct.sum())
        totals["operation_switch_correct"] += float(operation_pair_correct.sum())
        totals["heldout_query_switch_correct"] += float(
            query_pair_correct[heldout].sum()
        )
        totals["heldout_operation_switch_correct"] += float(
            operation_pair_correct[heldout].sum()
        )
        totals["pair_swap_identity_consistent"] += float(
            (original_identity == swapped_identity).sum()
        )
        totals["pair_swap_output_pixel_l1_sum"] += float(pair_l1.sum())
        totals["identity_correct"] += float(
            (predicted_identity == expected_identity).sum()
        )
        totals["pixel_f1_sum"] += float(pixel_f1_rows(generated, targets).sum())
        totals["target_cosine_sum"] += float(target_cosine.sum())
        totals["query_output_pixel_l1_sum"] += float(query_l1.sum())
        totals["operation_output_pixel_l1_sum"] += float(operation_l1.sum())
        totals["query_label_match_correct"] += float(
            (match_choice == query_expected).sum()
        )
        totals["operation_gate_correct"] += float(
            (operation_choice == operation_same).sum()
        )
        totals["same_gate_sum"] += float(same_gate[operation_same].sum())
        totals["same_gate_count"] += float(operation_same.sum())
        totals["other_gate_sum"] += float(same_gate[~operation_same].sum())
        totals["other_gate_count"] += float((~operation_same).sum())
        totals["route_finite"] *= float(
            torch.isfinite(trace["route_weights"]).all()
        )
        totals["route_sum_error_max"] = max(
            totals["route_sum_error_max"],
            float((trace["route_weights"].sum(dim=1) - 1.0).abs().max()),
        )

    examples = totals["examples"]
    pairs = totals["pairs"]
    heldout_pairs = totals["heldout_pairs"]
    if examples < 1 or pairs < 1 or heldout_pairs < 1:
        raise ValueError("V23 relation development loader lacks required examples")
    heldout_query = totals["heldout_query_switch_correct"] / heldout_pairs
    heldout_operation = totals["heldout_operation_switch_correct"] / heldout_pairs
    same_mean = totals["same_gate_sum"] / max(1.0, totals["same_gate_count"])
    other_mean = totals["other_gate_sum"] / max(1.0, totals["other_gate_count"])
    return {
        "examples": examples,
        "pairs": pairs,
        "heldout_pairs": heldout_pairs,
        "binary_choice_accuracy": totals["binary_choice_correct"] / examples,
        "query_switch_accuracy": totals["query_switch_correct"] / pairs,
        "operation_switch_accuracy": totals["operation_switch_correct"] / pairs,
        "heldout_query_switch_accuracy": heldout_query,
        "heldout_operation_switch_accuracy": heldout_operation,
        "heldout_combination_minimum_switch_accuracy": min(
            heldout_query, heldout_operation
        ),
        "pair_swap_identity_consistency": (
            totals["pair_swap_identity_consistent"] / pairs
        ),
        "pair_swap_output_pixel_l1": (
            totals["pair_swap_output_pixel_l1_sum"] / pairs
        ),
        "identity_top1": totals["identity_correct"] / examples,
        "identity_bank_identities": float(len(bank_characters)),
        "pixel_f1": totals["pixel_f1_sum"] / examples,
        "target_cosine": totals["target_cosine_sum"] / examples,
        "query_output_pixel_l1": (
            totals["query_output_pixel_l1_sum"] / pairs
        ),
        "operation_output_pixel_l1": (
            totals["operation_output_pixel_l1_sum"] / pairs
        ),
        "query_label_match_accuracy": (
            totals["query_label_match_correct"] / examples
        ),
        "operation_gate_accuracy": totals["operation_gate_correct"] / examples,
        "same_operation_gate_mean": same_mean,
        "other_operation_gate_mean": other_mean,
        "operation_gate_separation": same_mean - other_mean,
        "temperature": float(model.temperature()),
        "route_weights_finite": totals["route_finite"],
        "route_weight_sum_error_max": totals["route_sum_error_max"],
        "student_boundary_clean": float(
            student_boundary_is_clean(
                model.boundary_receipt(), model.config.route_mode
            )
        ),
        "frozen_images_instantiated": 0.0,
    }


def _ink_image(tensor: torch.Tensor, scale: int = 2) -> Image.Image:
    array = (
        (1.0 - tensor.detach().float().cpu().clamp(0, 1)[0]).numpy() * 255.0
    ).round().astype(np.uint8)
    image = Image.fromarray(array).convert("RGB")
    if scale != 1:
        image = image.resize(
            (image.width * scale, image.height * scale),
            Image.Resampling.NEAREST,
        )
    return image


@torch.no_grad()
def save_sample_sheet(
    model: VisualRelationCircuit,
    loader: DataLoader,
    *,
    path: Path,
    device: torch.device,
    precision: str,
    sample_count: int,
) -> None:
    model.eval()
    batch = _device_batch(next(iter(loader)), device)
    count = min(sample_count, batch["prompt"].shape[0])
    prompts = torch.cat(
        (
            batch["prompt"][:count],
            batch["query_counterfactual_prompt"][:count],
            batch["operation_counterfactual_prompt"][:count],
            batch["pair_swapped_prompt"][:count],
        )
    )
    with autocast_context(device, precision):
        generated = model(prompts)[:, 0]
    generated_variants = generated.split(count)
    targets = (
        batch["target"][:count],
        batch["query_counterfactual_target"][:count],
        batch["operation_counterfactual_target"][:count],
        batch["pair_swapped_target"][:count],
    )
    tile = 64
    card_width = 6 * tile + 16
    card_height = 3 * tile + 42
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * card_width + 16, rows * card_height + 16),
        "#e9eff1",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    variant_names = ("base", "query cf", "operation cf", "pair swap")
    for index in range(count):
        x = 8 + (index % columns) * card_width
        y = 8 + (index // columns) * card_height
        draw.rectangle(
            (x, y, x + card_width - 8, y + card_height - 8),
            fill="white",
            outline="#91a4ab",
        )
        draw.text(
            (x + 4, y + 4),
            "heldout" if batch["metadata"][index]["heldout_combination"] else "seen",
            fill="#24373f",
            font=font,
        )
        for frame in range(6):
            sheet.paste(
                _ink_image(batch["prompt"][index, frame]),
                (x + frame * tile, y + 18),
            )
        output_y = y + 92
        for variant, name in enumerate(variant_names):
            answer_x = x + variant * 94
            draw.text((answer_x, output_y - 12), name, fill="#50636b", font=font)
            sheet.paste(
                _ink_image(generated_variants[variant][index]),
                (answer_x, output_y),
            )
            sheet.paste(
                _ink_image(targets[variant][index]),
                (answer_x, output_y + tile),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def validate_canonicalizer_checkpoint(
    checkpoint: dict[str, Any],
    *,
    checkpoint_sha256: str,
) -> None:
    if checkpoint_sha256 != EXPECTED_CANONICALIZER_SHA256:
        raise ValueError(
            "V23 requires the selected Stage A checkpoint SHA-256 "
            f"{EXPECTED_CANONICALIZER_SHA256}, got {checkpoint_sha256}"
        )
    if checkpoint.get("architecture") != CANONICALIZER_ARCHITECTURE:
        raise ValueError("Stage B requires a V23 canonicalizer checkpoint")
    if checkpoint.get("smoke_only"):
        raise ValueError("Stage B refuses a smoke-only canonicalizer")
    best = checkpoint.get("best_development")
    if not isinstance(best, dict) or not all(
        canonicalizer_selection_gate_report(best).values()
    ):
        raise ValueError("Stage B requires a gate-selected canonicalizer")
    if int(checkpoint.get("step", -1)) != int(best.get("step", -2)):
        raise ValueError("Stage B requires the exact selected canonicalizer step")
    if checkpoint.get("trainable_parameters") != EXPECTED_CANONICALIZER_PARAMETERS:
        raise ValueError("Stage A canonicalizer parameter count differs")
    if not canonicalizer_boundary_is_clean(checkpoint["boundary_receipt"]):
        raise ValueError("Stage A canonicalizer boundary is not image-only")
    if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
        raise ValueError("Stage A PVF receipt differs")
    if checkpoint.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Stage A manifest receipt differs")
    for key, expected in EXPECTED_PARTITION.items():
        if checkpoint.get("partition", {}).get(key) != expected:
            raise ValueError(f"Stage A partition receipt differs for {key}")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("protocol_sha256") != file_sha256(PROTOCOL_DOCUMENT):
        raise ValueError("Stage A protocol document hash differs")
    for path, expected_hash in protocol.get("source_files_sha256", {}).items():
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Stage A source hash differs for {path}")


def _protocol_payload(
    args: argparse.Namespace,
    partition: dict[str, Any],
    canonicalizer_checkpoint: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "stage": "relation",
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "route_mode": args.route_mode,
        "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
        "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
        "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
        "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "expected_pvf_sha256": EXPECTED_PVF_SHA256,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256": file_sha256(args.manifest),
        "canonicalizer_checkpoint": args.canonicalizer_checkpoint,
        "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "canonicalizer_step": canonicalizer_checkpoint["step"],
        "canonicalizer_best_development": canonicalizer_checkpoint[
            "best_development"
        ],
        "canonicalizer_protocol": canonicalizer_checkpoint["protocol"],
        "partition": partition,
        "smoke_only": bool(args.smoke),
    }


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: VisualRelationCircuit,
    optimizer: torch.optim.Optimizer,
    step: int,
    partition: dict[str, Any],
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    best_development: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "stage": "relation",
        "route_mode": args.route_mode,
        "model_config": relation_circuit_config_payload(model.config),
        "relation": relation_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": protocol["expected_pvf_sha256"],
        "canonicalizer_checkpoint": args.canonicalizer_checkpoint,
        "canonicalizer_sha256": protocol["canonicalizer_sha256"],
        "manifest_sha256": protocol["manifest_sha256"],
        "retinal_fonts": retinal_font_manifest(),
        "step": step,
        "smoke_only": bool(args.smoke),
        "args": vars(args),
        "partition": partition,
        "protocol": protocol,
        "boundary_receipt": model.boundary_receipt(),
        "trainable_parameters": _trainable_parameters(model),
        "trainable_parameter_shapes": _parameter_shapes(model),
        "metrics": metrics,
        "best_development": best_development,
    }


def main() -> None:
    args = parse_args()
    _require_fixed_arguments(args)
    if args.batch_size < 2 or args.development_samples < 2:
        raise ValueError("V23 requires batch size and development samples at least two")
    if args.identity_bank_views < 1:
        raise ValueError("V23 requires at least one identity-bank view")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    pvf_sha256 = file_sha256(args.pvf_checkpoint)
    if pvf_sha256 != EXPECTED_PVF_SHA256:
        raise ValueError(
            f"V23 requires PVF SHA-256 {EXPECTED_PVF_SHA256}, got {pvf_sha256}"
        )
    manifest_sha256 = file_sha256(args.manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "V23 manifest differs from the preregistered corpus: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {manifest_sha256}"
        )
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
    canonicalizer = VisualCanonicalizer()
    canonicalizer.load_state_dict(canonicalizer_checkpoint["canonicalizer"])

    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    retina = pvf.retina
    del pvf
    records = load_visual_grammar_manifest(args.manifest)
    bank = build_relation_character_bank(records, bank_size=args.bank_size)
    partitions = split_relation_characters(bank, salt=args.partition_salt)
    partition = relation_partition_receipt(partitions, salt=args.partition_salt)
    for key, expected in EXPECTED_PARTITION.items():
        if partition.get(key) != expected:
            raise ValueError(
                f"V23 partition receipt {key!r} differs: "
                f"expected {expected!r}, got {partition.get(key)!r}"
            )

    episode_config = VisualRelationEpisodeConfig()
    total_training_examples = args.maximum_steps * args.batch_size
    train_dataset = VisualRelationEpisodeDataset(
        partitions["train"],
        split="train",
        length=total_training_examples,
        config=episode_config,
        seed=args.dataset_seed,
    )
    development_dataset = VisualRelationEpisodeDataset(
        partitions["development"],
        split="development",
        length=args.development_samples,
        config=episode_config,
        seed=args.dataset_seed + 50_000,
    )
    config = VisualRelationCircuitConfig(
        visual_dim=retina.config.visual_dim,
        route_mode=args.route_mode,
    )
    model = VisualRelationCircuit(config, retina, canonicalizer).to(device)
    trainable = _trainable_parameters(model)
    if trainable != EXPECTED_PARAMETERS:
        raise ValueError(
            f"V23 requires {EXPECTED_PARAMETERS:,} relation parameters, "
            f"got {trainable:,}"
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    step = 0
    best_development: dict[str, Any] | None = None
    output_dir = Path(args.out) / args.route_mode
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to append a new V23 run to nonempty output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"
    protocol = _protocol_payload(args, partition, canonicalizer_checkpoint)
    (output_dir / "partition.json").write_text(
        json.dumps(partition, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "preregistered_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("resume checkpoint is not a V23 relation circuit")
        if checkpoint.get("route_mode") != args.route_mode:
            raise ValueError("resume route mode differs from requested V23 arm")
        if checkpoint.get("smoke_only") and not args.smoke:
            raise ValueError("a V23 smoke checkpoint cannot resume into evidence")
        if checkpoint.get("protocol") != protocol:
            raise ValueError("resume protocol differs from the current sealed V23 run")
        expected_config = relation_circuit_config_from_payload(
            checkpoint["model_config"]
        )
        if expected_config != model.config:
            raise ValueError("resume model configuration differs from V23 protocol")
        load_relation_state(model, checkpoint["relation"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        best_development = checkpoint.get("best_development")
        if step >= args.maximum_steps:
            raise ValueError("resume checkpoint already reached maximum steps")

    train_loader = DataLoader(
        Subset(
            train_dataset,
            range(step * args.batch_size, total_training_examples),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=visual_relation_collate,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_relation_collate,
    )
    bank_images = _development_bank_images(
        partitions["development"],
        views=args.identity_bank_views,
        config=episode_config,
        seed=args.dataset_seed + 100_000,
    ).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(model, bank_images)
    del bank_images

    run_receipt = {
        "stage": "start",
        "architecture": ARCHITECTURE,
        "route_mode": args.route_mode,
        "device": str(device),
        "precision": args.precision,
        "trainable_parameters": trainable,
        "pvf_sha256": pvf_sha256,
        "canonicalizer_sha256": canonicalizer_sha256,
        "manifest_sha256": manifest_sha256,
        "partition": partition,
        "boundary_receipt": model.boundary_receipt(),
        "smoke_only": bool(args.smoke),
    }
    print(json.dumps(run_receipt, ensure_ascii=False, sort_keys=True))
    append_jsonl(log_path, run_receipt)

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start_time = time.perf_counter()
    last_metrics: dict[str, Any] = {}

    for raw_batch in train_loader:
        if stop_requested or step >= args.maximum_steps:
            break
        step += 1
        model.train()
        batch = _device_batch(raw_batch, device)
        learning_rate = scheduled_lr(
            step,
            base=args.lr,
            warmup=args.warmup_steps,
            total=args.maximum_steps,
            minimum_ratio=args.minimum_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.precision):
            loss, train_metrics = relation_loss(
                model,
                batch,
                stroke_weight=args.stroke_weight,
                generated_visual_weight=args.generated_visual_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            args.gradient_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            row = {
                "stage": "train",
                "step": step,
                "route_mode": args.route_mode,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - start_time,
                **{key: float(value) for key, value in train_metrics.items()},
            }
            if device.type == "cuda":
                row["peak_cuda_gib"] = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                )
            print(json.dumps(row, sort_keys=True))
            append_jsonl(log_path, row)

        if step % args.validate_every == 0 or step == args.maximum_steps:
            metrics = evaluate_development(
                model,
                development_loader,
                bank_visual=bank_visual,
                bank_characters=partitions["development"],
                device=device,
                precision=args.precision,
            )
            metrics.update(
                {
                    "stage": "validation",
                    "step": float(step),
                    "route_mode": args.route_mode,
                }
            )
            gates = selection_gate_report(metrics, args.route_mode)
            eligible = all(gates.values()) and not args.smoke
            metrics["selection_eligible"] = float(eligible)
            metrics["selection_gates"] = gates
            print(json.dumps(metrics, sort_keys=True))
            append_jsonl(log_path, metrics)
            save_sample_sheet(
                model,
                development_loader,
                path=(
                    output_dir
                    / "development_samples"
                    / f"step_{step:07d}.png"
                ),
                device=device,
                precision=args.precision,
                sample_count=args.sample_count,
            )
            last_metrics = metrics
            if eligible and (
                best_development is None
                or selection_rank(metrics, args.route_mode)
                > selection_rank(best_development, args.route_mode)
            ):
                best_development = dict(metrics)

        if step % args.save_every == 0 or step == args.maximum_steps:
            payload = _checkpoint_payload(
                args=args,
                model=model,
                optimizer=optimizer,
                step=step,
                partition=partition,
                protocol=protocol,
                metrics=last_metrics,
                best_development=best_development,
            )
            checkpoint_path = output_dir / f"checkpoint_step_{step:07d}.pt"
            atomic_save(payload, checkpoint_path)
            atomic_save(payload, output_dir / "checkpoint_latest.pt")
            if best_development is not None and int(best_development["step"]) == step:
                atomic_save(
                    payload,
                    output_dir / "checkpoint_selected_development.pt",
                )

    if stop_requested and step < args.maximum_steps:
        payload = _checkpoint_payload(
            args=args,
            model=model,
            optimizer=optimizer,
            step=step,
            partition=partition,
            protocol=protocol,
            metrics=last_metrics,
            best_development=best_development,
        )
        interrupted = output_dir / "checkpoint_interrupted.pt"
        atomic_save(payload, interrupted)
        append_jsonl(
            log_path,
            {
                "stage": "interrupted",
                "step": step,
                "route_mode": args.route_mode,
                "checkpoint": str(interrupted),
            },
        )
        return

    selected_path = output_dir / "checkpoint_selected_development.pt"
    complete = {
        "stage": "complete",
        "step": step,
        "route_mode": args.route_mode,
        "smoke_only": bool(args.smoke),
        "elapsed_seconds": time.perf_counter() - start_time,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "best_development": best_development,
        "selected_checkpoint": str(selected_path) if selected_path.exists() else None,
        "paired_control_gate_passed": False,
        "blinded_readability_gate_passed": False,
        "frozen_evaluation_permitted": False,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
    }
    print(json.dumps(complete, sort_keys=True))
    append_jsonl(log_path, complete)


if __name__ == "__main__":
    main()
