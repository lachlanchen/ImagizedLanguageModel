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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.saccade_data import render_glyph_fovea
from ilm.visual_lm.visual_binding_data import LABEL_PAIRS, noncanonical_variant
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
    ROUTE_MODES,
    VisualPacketRereadStream,
    VisualPacketStreamConfig,
    visual_packet_stream_config_from_payload,
    visual_packet_stream_config_payload,
)
from ilm.visual_lm.visual_relation_circuit import (
    RELATION_AWARE_ROUTE,
    VisualCanonicalizer,
    pixel_f1_rows,
    relation_circuit_config_from_payload,
    topology_loss,
)
from scripts.train_visual_relation_circuit_v23 import (
    ARCHITECTURE as V23_RELATION_ARCHITECTURE,
    candidate_selection_gate_report as v23_candidate_gate_report,
    validate_canonicalizer_checkpoint,
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


ARCHITECTURE = "visual-packet-reread-stream-v24"
PROTOCOL_DOCUMENT = "references/visual_packet_reread_stream_v24_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/visual_packet_data.py",
    "ilm/visual_lm/visual_packet_stream.py",
    "scripts/train_visual_packet_stream_v24.py",
)
DEFAULT_PVF_CHECKPOINT = (
    "artifacts/predictive_visual_field_v16_memory_pilot/"
    "checkpoint_step_0002200.pt"
)
DEFAULT_CANONICALIZER_CHECKPOINT = (
    "artifacts/visual_canonicalizer_v23_evidence/"
    "checkpoint_selected_development.pt"
)
DEFAULT_RELATION_CHECKPOINT = (
    "artifacts/visual_relation_circuit_v23_evidence/relation_aware/"
    "checkpoint_selected_development.pt"
)
EXPECTED_PVF_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
EXPECTED_CANONICALIZER_SHA256 = (
    "26cf1bab490abe867e7055a679eff6a9e26e81ad78e6cd9694afd3e425c06135"
)
EXPECTED_RELATION_SHA256 = (
    "69c5cb06a5f02b5bed26b8687042534e9481fec96bea6ab41e2e00df7c23df43"
)
EXPECTED_MANIFEST_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_PARAMETERS = 1_347
EXPECTED_PARTITION = {
    "train_identities": 829,
    "development_identities": 88,
    "frozen_identities": 107,
    "development_identifiers_sha256": (
        "2b611e66778061319bb2502ad850c635b5d89e81e9eab7f9c8ef23a09514e892"
    ),
    "frozen_identifiers_sha256": (
        "d3f6d51ef6c0cb0eeeab664d89e8a2c467bc35ea7482ae55f873f4b28b85c2ab"
    ),
}
FIXED_MODEL_ARGUMENTS = {"bank_size": 1_024}
FIXED_LOSS_ARGUMENTS = {
    "stroke_weight": 4.0,
    "generated_visual_weight": 0.10,
    "localization_weight": 0.25,
}
FIXED_OPTIMIZATION_ARGUMENTS = {
    "lr": 2e-3,
    "minimum_lr_ratio": 0.10,
    "warmup_steps": 25,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "seed": 20260828,
    "dataset_seed": 20260829,
}
FIXED_EVIDENCE_ARGUMENTS = {
    "maximum_steps": 800,
    "batch_size": 64,
    "num_workers": 8,
    "precision": "bf16",
    "development_samples": 512,
    "identity_bank_views": 4,
    "validate_every": 100,
    "save_every": 100,
}
GATE_EPSILON = 1e-12
LABEL_CHARACTERS = tuple(character for pair in LABEL_PAIRS for character in pair)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one preregistered V24 image-only packet-stream arm."
    )
    parser.add_argument("--pvf-checkpoint", default=DEFAULT_PVF_CHECKPOINT)
    parser.add_argument(
        "--canonicalizer-checkpoint", default=DEFAULT_CANONICALIZER_CHECKPOINT
    )
    parser.add_argument("--relation-checkpoint", default=DEFAULT_RELATION_CHECKPOINT)
    parser.add_argument("--route-mode", choices=ROUTE_MODES, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_packet_stream_v24")
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--generated-visual-weight", type=float, default=0.10)
    parser.add_argument("--localization-weight", type=float, default=0.25)
    parser.add_argument("--maximum-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--dataset-seed", type=int, default=20260829)
    parser.add_argument("--development-samples", type=int, default=512)
    parser.add_argument("--identity-bank-views", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def _require_fixed_arguments(args: argparse.Namespace) -> None:
    if args.partition_salt != PARTITION_SALT:
        raise ValueError(f"V24 requires partition salt {PARTITION_SALT!r}")
    for group in (
        FIXED_MODEL_ARGUMENTS,
        FIXED_LOSS_ARGUMENTS,
        FIXED_OPTIMIZATION_ARGUMENTS,
    ):
        for name, expected in group.items():
            if getattr(args, name) != expected:
                raise ValueError(f"V24 requires --{name.replace('_', '-')}={expected}")
    if args.smoke:
        if not 1 <= args.maximum_steps <= 20:
            raise ValueError("V24 smoke mode is limited to 1--20 optimization steps")
        return
    for name, expected in FIXED_EVIDENCE_ARGUMENTS.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V24 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def candidate_selection_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "frame1_binary_choice_accuracy": _strictly_above(
            metrics["frame1_binary_choice_accuracy"], 0.95
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
        "frame1_identity_top1": _strictly_above(
            metrics["frame1_identity_top1"], 0.75
        ),
        "identity_bank_complete": metrics["identity_bank_identities"] == 88.0,
        "frame1_pixel_f1": _strictly_above(metrics["frame1_pixel_f1"], 0.68),
        "frame1_target_cosine": _strictly_above(
            metrics["frame1_target_cosine"], 0.82
        ),
        "frame2_label_top1": _strictly_above(
            metrics["frame2_label_top1"], 0.95
        ),
        "frame2_pixel_f1": _strictly_above(metrics["frame2_pixel_f1"], 0.58),
        "frame2_target_cosine": _strictly_above(
            metrics["frame2_target_cosine"], 0.80
        ),
        "teacher_forced_label_agreement": _strictly_above(
            metrics["teacher_forced_label_agreement"], 0.95
        ),
        "frame2_generated_history_consistency": _strictly_above(
            metrics["frame2_generated_history_consistency"], 0.92
        ),
        "history_switch_accuracy": _strictly_above(
            metrics["history_switch_accuracy"], 0.90
        ),
        "history_output_pixel_l1": _strictly_above(
            metrics["history_output_pixel_l1"], 0.10
        ),
        "query_header_localization_accuracy": _strictly_above(
            metrics["query_header_localization_accuracy"], 0.99
        ),
        "operation_header_localization_accuracy": _strictly_above(
            metrics["operation_header_localization_accuracy"], 0.99
        ),
        "pair_header_localization_accuracy": _strictly_above(
            metrics["pair_header_localization_accuracy"], 0.99
        ),
        "packet_permutation_identity_consistency": (
            _strictly_above(
                metrics["packet_permutation_frame1_identity_consistency"], 0.99
            )
            and _strictly_above(
                metrics["packet_permutation_frame2_identity_consistency"], 0.99
            )
        ),
        "packet_permutation_output_pixel_l1": (
            _strictly_below(
                metrics["packet_permutation_frame1_output_pixel_l1"], 1e-6
            )
            and _strictly_below(
                metrics["packet_permutation_frame2_output_pixel_l1"], 1e-6
            )
        ),
        "distractor_identity_consistency": (
            _strictly_above(
                metrics["distractor_frame1_identity_consistency"], 0.95
            )
            and _strictly_above(
                metrics["distractor_frame2_identity_consistency"], 0.95
            )
        ),
        "heldout_length_quality": (
            _strictly_above(metrics["heldout_length_frame1_identity_top1"], 0.90)
            and _strictly_above(metrics["heldout_length_frame2_label_top1"], 0.90)
        ),
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
    }


def control_selection_gate_report(
    metrics: dict[str, float], route_mode: str
) -> dict[str, bool]:
    intervention_name = {
        HEADER_BLIND_ROUTE: "header_output_pixel_l1",
        QUERY_BLIND_ROUTE: "query_output_pixel_l1",
        OPERATION_BLIND_ROUTE: "operation_output_pixel_l1",
        HISTORY_BLIND_ROUTE: "history_output_pixel_l1",
    }.get(route_mode)
    if intervention_name is None:
        raise ValueError(f"V24 control gate cannot evaluate {route_mode!r}")
    return {
        f"{route_mode}_invariant": _strictly_below(
            metrics[intervention_name], 1e-7
        ),
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
    }


def selection_gate_report(
    metrics: dict[str, float], route_mode: str
) -> dict[str, bool]:
    if route_mode == PACKET_AWARE_ROUTE:
        return candidate_selection_gate_report(metrics)
    return control_selection_gate_report(metrics, route_mode)


def selection_rank(metrics: dict[str, float], route_mode: str) -> tuple[float, ...]:
    if route_mode == PACKET_AWARE_ROUTE:
        return (
            min(
                metrics["query_switch_accuracy"],
                metrics["operation_switch_accuracy"],
                metrics["history_switch_accuracy"],
            ),
            metrics["frame1_identity_top1"],
            metrics["frame2_label_top1"],
            0.5 * (metrics["frame1_pixel_f1"] + metrics["frame2_pixel_f1"]),
            -metrics["step"],
        )
    return (
        min(
            metrics["query_header_localization_accuracy"],
            metrics["operation_header_localization_accuracy"],
            metrics["pair_header_localization_accuracy"],
        ),
        0.5 * (metrics["frame1_pixel_f1"] + metrics["frame2_pixel_f1"]),
        -metrics["step"],
    )


def student_boundary_is_clean(
    receipt: dict[str, bool | str], route_mode: str
) -> bool:
    required_true = {
        "input_is_continuous_image_stream",
        "output_is_continuous_image_stream",
        "uses_relative_packet_offsets",
        "uses_visible_packet_headers",
        "rereads_generated_pixels",
    }
    required_false = {
        "uses_absolute_frame_roles",
        "uses_padding_mask",
        "uses_active_lengths",
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_ocr",
        "uses_character_labels",
        "uses_operation_ids",
        "uses_role_ids_as_input",
        "uses_packet_indices_as_input",
        "uses_target_indices",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "retina_trainable",
        "canonicalizer_trainable",
        "operation_reader_trainable",
    }
    return (
        receipt.get("architecture") == ARCHITECTURE
        and receipt.get("route_mode") == route_mode
        and all(receipt.get(key) is True for key in required_true)
        and all(receipt.get(key) is False for key in required_false)
    )


def _trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _parameter_shapes(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def packet_state_dict(model: VisualPacketRereadStream) -> dict[str, torch.Tensor]:
    frozen_prefixes = ("retina.", "canonicalizer.", "operation_reader.")
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith(frozen_prefixes) and name != "match_temperature"
    }


def load_packet_state(
    model: VisualPacketRereadStream, state: dict[str, torch.Tensor]
) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"V24 packet state has unexpected keys: {incompatible.unexpected_keys}"
        )
    allowed = (
        "retina.",
        "canonicalizer.",
        "operation_reader.",
        "match_temperature",
    )
    if any(not key.startswith(allowed) for key in incompatible.missing_keys):
        raise ValueError(f"V24 packet state is incomplete: {incompatible.missing_keys}")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _visual_variants(
    batch: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_keys = (
        "prompt",
        "query_counterfactual_prompt",
        "operation_counterfactual_prompt",
        "target_stream",
        "query_counterfactual_target_stream",
        "operation_counterfactual_target_stream",
        "localization_target",
        "query_counterfactual_localization_target",
        "operation_counterfactual_localization_target",
    )
    if any(not torch.is_floating_point(batch[key]) for key in image_keys):
        raise TypeError("V24 inputs and targets must remain continuous image tensors")
    prompts = torch.cat(
        (
            batch["prompt"],
            batch["query_counterfactual_prompt"],
            batch["operation_counterfactual_prompt"],
        )
    )
    targets = torch.cat(
        (
            batch["target_stream"],
            batch["query_counterfactual_target_stream"],
            batch["operation_counterfactual_target_stream"],
        )
    )
    localization = torch.cat(
        (
            batch["localization_target"],
            batch["query_counterfactual_localization_target"],
            batch["operation_counterfactual_localization_target"],
        )
    )
    return prompts, targets, localization


def packet_stream_loss(
    model: VisualPacketRereadStream,
    batch: dict[str, Any],
    *,
    stroke_weight: float,
    generated_visual_weight: float,
    localization_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prompts, targets, localization_targets = _visual_variants(batch)
    logits, trace = model.logits_with_trace(prompts)
    topology, topology_metrics = topology_loss(
        logits.flatten(0, 1),
        targets.flatten(0, 1),
        stroke_weight=stroke_weight,
    )
    generated = logits.sigmoid()
    with torch.no_grad():
        target_visual = model.encode_images(targets.flatten(0, 1))
    generated_visual = model.encode_images(generated.flatten(0, 1))
    generated_visual_loss = (
        1.0 - (generated_visual * target_visual).sum(dim=1)
    ).mean()
    routed_localization = torch.stack(
        (
            trace["routed_query_image"],
            trace["routed_operation_image"],
            trace["routed_pair_label_mean"],
            trace["routed_pair_glyph_mean"],
        ),
        dim=1,
    )
    localization_loss = (routed_localization - localization_targets).abs().mean()
    total = (
        topology
        + generated_visual_weight * generated_visual_loss
        + localization_weight * localization_loss
    )
    return total, {
        "loss": total.detach(),
        **topology_metrics,
        "frame1_pixel_f1": pixel_f1_rows(
            generated[:, 0], targets[:, 0]
        ).mean().detach(),
        "frame2_pixel_f1": pixel_f1_rows(
            generated[:, 1], targets[:, 1]
        ).mean().detach(),
        "generated_visual_loss": generated_visual_loss.detach(),
        "localization_loss": localization_loss.detach(),
        "role_temperature_pair": model.role_temperatures()[0].detach(),
        "role_temperature_operation": model.role_temperatures()[1].detach(),
        "role_temperature_query": model.role_temperatures()[2].detach(),
    }


def _development_bank_images(
    characters: Sequence[str],
    *,
    views: int,
    config: VisualPacketEpisodeConfig,
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
    ).reshape(len(characters), views, 1, 32, 32)


def _label_bank_images(config: VisualPacketEpisodeConfig) -> torch.Tensor:
    return torch.stack(
        [
            render_glyph_fovea(
                character,
                render_config=config.target_render_config(),
                fovea_size=config.fovea_size,
                variant=config.canonical_target_variant,
            )
            for character in LABEL_CHARACTERS
        ]
    )


@torch.no_grad()
def encode_identity_bank(
    model: VisualPacketRereadStream, images: torch.Tensor
) -> torch.Tensor:
    identities, views = images.shape[:2]
    visual = model.encode_images(images.flatten(0, 1))
    return visual.reshape(identities, views, -1)


def _predict_identity(
    visual: torch.Tensor, bank_visual: torch.Tensor
) -> torch.Tensor:
    return torch.einsum("bd,nvd->bnv", visual, bank_visual).amax(dim=2).argmax(dim=1)


def _predict_label(visual: torch.Tensor, label_visual: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bd,nd->bn", visual, label_visual).argmax(dim=1)


def _mean_image_l1(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first - second).abs().flatten(1).mean(dim=1)


@torch.no_grad()
def evaluate_development(
    model: VisualPacketRereadStream,
    loader: DataLoader,
    *,
    bank_visual: torch.Tensor,
    bank_characters: Sequence[str],
    label_visual: torch.Tensor,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    bank_visual = F.normalize(bank_visual.float(), dim=-1)
    label_visual = F.normalize(label_visual.float(), dim=-1)
    bank_index = {
        character: index for index, character in enumerate(bank_characters)
    }
    label_index = {character: index for index, character in enumerate(LABEL_CHARACTERS)}
    totals: dict[str, float] = {
        "episodes": 0.0,
        "quality_examples": 0.0,
        "heldout_combinations": 0.0,
        "heldout_lengths": 0.0,
        "binary_choice_correct": 0.0,
        "binary_choice_examples": 0.0,
        "query_switch_correct": 0.0,
        "operation_switch_correct": 0.0,
        "heldout_query_switch_correct": 0.0,
        "heldout_operation_switch_correct": 0.0,
        "frame1_identity_correct": 0.0,
        "frame1_f1_sum": 0.0,
        "frame1_cosine_sum": 0.0,
        "frame2_label_correct": 0.0,
        "frame2_f1_sum": 0.0,
        "frame2_cosine_sum": 0.0,
        "teacher_agreement": 0.0,
        "history_consistency": 0.0,
        "history_switch_correct": 0.0,
        "history_l1_sum": 0.0,
        "query_l1_sum": 0.0,
        "operation_l1_sum": 0.0,
        "header_l1_sum": 0.0,
        "query_header_correct": 0.0,
        "operation_header_correct": 0.0,
        "pair_header_correct": 0.0,
        "permutation_frame1_identity": 0.0,
        "permutation_frame2_identity": 0.0,
        "permutation_frame1_l1_sum": 0.0,
        "permutation_frame2_l1_sum": 0.0,
        "distractor_frame1_identity": 0.0,
        "distractor_frame2_identity": 0.0,
        "heldout_length_frame1_correct": 0.0,
        "heldout_length_frame2_correct": 0.0,
    }
    for raw_batch in loader:
        batch = _device_batch(raw_batch, device)
        batch_size = batch["prompt"].shape[0]
        header_zero = batch["prompt"].clone()
        header_zero[:, 0::3] = 0.0
        with autocast_context(device, precision):
            base_logits, base_trace = model.logits_with_trace(batch["prompt"])
            query_logits, _ = model.logits_with_trace(
                batch["query_counterfactual_prompt"]
            )
            operation_logits, _ = model.logits_with_trace(
                batch["operation_counterfactual_prompt"]
            )
            permutation_logits, _ = model.logits_with_trace(
                batch["permuted_prompt"]
            )
            distractor_logits, _ = model.logits_with_trace(
                batch["distractor_counterfactual_prompt"]
            )
            history_logits, _ = model.logits_with_trace(
                batch["prompt"],
                first_frame_override=batch["history_override_frame"],
            )
            teacher_logits, _ = model.logits_with_trace(
                batch["prompt"],
                first_frame_override=batch["teacher_forced_frame"],
            )
            header_logits, _ = model.logits_with_trace(header_zero)

        outputs = [
            value.sigmoid()
            for value in (
                base_logits,
                query_logits,
                operation_logits,
                permutation_logits,
                distractor_logits,
            )
        ]
        history_output = history_logits.sigmoid()
        teacher_output = teacher_logits.sigmoid()
        header_output = header_logits.sigmoid()
        targets = [
            batch["target_stream"],
            batch["query_counterfactual_target_stream"],
            batch["operation_counterfactual_target_stream"],
            batch["permuted_target_stream"],
            batch["distractor_counterfactual_target_stream"],
        ]
        all_output = torch.cat(outputs)
        all_target = torch.cat(targets)
        with autocast_context(device, precision):
            all_frame1_visual = model.encode_images(all_output[:, 0])
            all_frame2_visual = model.encode_images(all_output[:, 1])
            target_frame1_visual = model.encode_images(all_target[:, 0])
            target_frame2_visual = model.encode_images(all_target[:, 1])
            history_label_visual = model.encode_images(history_output[:, 1])
            teacher_label_visual = model.encode_images(teacher_output[:, 1])

        frame1_prediction = _predict_identity(all_frame1_visual, bank_visual)
        frame2_prediction = _predict_label(all_frame2_visual, label_visual)
        history_label_prediction = _predict_label(
            history_label_visual, label_visual
        )
        teacher_label_prediction = _predict_label(
            teacher_label_visual, label_visual
        )
        frame1_predictions = frame1_prediction.split(batch_size)
        frame2_predictions = frame2_prediction.split(batch_size)
        base_frame1 = frame1_predictions[0]
        query_frame1 = frame1_predictions[1]
        operation_frame1 = frame1_predictions[2]
        base_frame2 = frame2_predictions[0]

        metadata = batch["metadata"]
        expected_characters = (
            [item["target_character"] for item in metadata]
            + [item["counterfactual_target_character"] for item in metadata]
            + [item["counterfactual_target_character"] for item in metadata]
            + [item["target_character"] for item in metadata]
            + [item["target_character"] for item in metadata]
        )
        expected_labels = (
            [item["target_label"] for item in metadata]
            + [item["counterfactual_target_label"] for item in metadata]
            + [item["counterfactual_target_label"] for item in metadata]
            + [item["target_label"] for item in metadata]
            + [item["target_label"] for item in metadata]
        )
        expected_frame1 = torch.tensor(
            [bank_index[value] for value in expected_characters], device=device
        )
        expected_frame2 = torch.tensor(
            [label_index[value] for value in expected_labels], device=device
        )
        expected_base_frame1, expected_query_frame1, expected_operation_frame1, _, _ = (
            expected_frame1.split(batch_size)
        )
        expected_base_frame2 = expected_frame2[:batch_size]
        expected_history_frame2 = torch.tensor(
            [label_index[item["counterfactual_target_label"]] for item in metadata],
            device=device,
        )

        base_frame1_visual, query_frame1_visual, operation_frame1_visual, _, _ = (
            all_frame1_visual.split(batch_size)
        )
        base_target_visual = model.encode_images(batch["target_stream"][:, 0])
        counterfactual_target_visual = model.encode_images(
            batch["query_counterfactual_target_stream"][:, 0]
        )
        choice_target = torch.cat(
            (base_target_visual, counterfactual_target_visual, counterfactual_target_visual)
        )
        choice_other = torch.cat(
            (counterfactual_target_visual, base_target_visual, base_target_visual)
        )
        choice_generated = torch.cat(
            (base_frame1_visual, query_frame1_visual, operation_frame1_visual)
        )
        choice_correct = (
            (choice_generated * choice_target).sum(dim=1)
            > (choice_generated * choice_other).sum(dim=1)
        )
        base_choice, query_choice, operation_choice = choice_correct.split(batch_size)
        query_switch = (
            base_choice
            & query_choice
            & (base_frame1 == expected_base_frame1)
            & (query_frame1 == expected_query_frame1)
            & (base_frame1 != query_frame1)
        )
        operation_switch = (
            base_choice
            & operation_choice
            & (base_frame1 == expected_base_frame1)
            & (operation_frame1 == expected_operation_frame1)
            & (base_frame1 != operation_frame1)
        )

        generated_character_to_label: list[int] = []
        for row, item in enumerate(metadata):
            predicted_character = bank_characters[int(base_frame1[row])]
            if predicted_character == item["glyphs"][0]:
                generated_character_to_label.append(label_index[item["labels"][0]])
            elif predicted_character == item["glyphs"][1]:
                generated_character_to_label.append(label_index[item["labels"][1]])
            else:
                generated_character_to_label.append(-1)
        generated_character_label = torch.tensor(
            generated_character_to_label, device=device
        )

        expected_query_packet = torch.tensor(
            [item["packet_kinds"].index("query") for item in metadata],
            device=device,
        )
        expected_operation_packet = torch.tensor(
            [item["packet_kinds"].index("operation") for item in metadata],
            device=device,
        )
        query_header_correct = (
            base_trace["query_indices"][:, 0] == expected_query_packet
        )
        operation_header_correct = (
            base_trace["operation_indices"][:, 0] == expected_operation_packet
        )
        pair_header_correct = []
        for row, item in enumerate(metadata):
            expected = {
                index
                for index, kind in enumerate(item["packet_kinds"])
                if kind == "pair"
            }
            actual = set(base_trace["pair_indices"][row].tolist())
            pair_header_correct.append(actual == expected)
        pair_header_correct_tensor = torch.tensor(
            pair_header_correct, device=device, dtype=torch.bool
        )

        heldout_combination = torch.tensor(
            [bool(item["heldout_combination"]) for item in metadata],
            device=device,
        )
        heldout_length = torch.tensor(
            [bool(item["heldout_length"]) for item in metadata], device=device
        )
        quality_examples = float(all_output.shape[0])
        totals["episodes"] += float(batch_size)
        totals["quality_examples"] += quality_examples
        totals["heldout_combinations"] += float(heldout_combination.sum())
        totals["heldout_lengths"] += float(heldout_length.sum())
        totals["binary_choice_correct"] += float(choice_correct.sum())
        totals["binary_choice_examples"] += float(choice_correct.numel())
        totals["query_switch_correct"] += float(query_switch.sum())
        totals["operation_switch_correct"] += float(operation_switch.sum())
        totals["heldout_query_switch_correct"] += float(
            query_switch[heldout_combination].sum()
        )
        totals["heldout_operation_switch_correct"] += float(
            operation_switch[heldout_combination].sum()
        )
        totals["frame1_identity_correct"] += float(
            (frame1_prediction == expected_frame1).sum()
        )
        totals["frame1_f1_sum"] += float(
            pixel_f1_rows(all_output[:, 0], all_target[:, 0]).sum()
        )
        totals["frame1_cosine_sum"] += float(
            (all_frame1_visual * target_frame1_visual).sum(dim=1).sum()
        )
        totals["frame2_label_correct"] += float(
            (frame2_prediction == expected_frame2).sum()
        )
        totals["frame2_f1_sum"] += float(
            pixel_f1_rows(all_output[:, 1], all_target[:, 1]).sum()
        )
        totals["frame2_cosine_sum"] += float(
            (all_frame2_visual * target_frame2_visual).sum(dim=1).sum()
        )
        totals["teacher_agreement"] += float(
            (teacher_label_prediction == base_frame2).sum()
        )
        totals["history_consistency"] += float(
            (base_frame2 == generated_character_label).sum()
        )
        totals["history_switch_correct"] += float(
            (
                (base_frame2 == expected_base_frame2)
                & (history_label_prediction == expected_history_frame2)
                & (base_frame2 != history_label_prediction)
            ).sum()
        )
        totals["history_l1_sum"] += float(
            _mean_image_l1(outputs[0][:, 1], history_output[:, 1]).sum()
        )
        totals["query_l1_sum"] += float(
            _mean_image_l1(outputs[0], outputs[1]).sum()
        )
        totals["operation_l1_sum"] += float(
            _mean_image_l1(outputs[0], outputs[2]).sum()
        )
        totals["header_l1_sum"] += float(
            _mean_image_l1(outputs[0], header_output).sum()
        )
        totals["query_header_correct"] += float(query_header_correct.sum())
        totals["operation_header_correct"] += float(
            operation_header_correct.sum()
        )
        totals["pair_header_correct"] += float(pair_header_correct_tensor.sum())
        totals["permutation_frame1_identity"] += float(
            (frame1_predictions[0] == frame1_predictions[3]).sum()
        )
        totals["permutation_frame2_identity"] += float(
            (frame2_predictions[0] == frame2_predictions[3]).sum()
        )
        totals["permutation_frame1_l1_sum"] += float(
            _mean_image_l1(outputs[0][:, 0], outputs[3][:, 0]).sum()
        )
        totals["permutation_frame2_l1_sum"] += float(
            _mean_image_l1(outputs[0][:, 1], outputs[3][:, 1]).sum()
        )
        totals["distractor_frame1_identity"] += float(
            (frame1_predictions[0] == frame1_predictions[4]).sum()
        )
        totals["distractor_frame2_identity"] += float(
            (frame2_predictions[0] == frame2_predictions[4]).sum()
        )
        totals["heldout_length_frame1_correct"] += float(
            (base_frame1[heldout_length] == expected_base_frame1[heldout_length]).sum()
        )
        totals["heldout_length_frame2_correct"] += float(
            (base_frame2[heldout_length] == expected_base_frame2[heldout_length]).sum()
        )

    episodes = totals["episodes"]
    quality = totals["quality_examples"]
    heldout_combinations = totals["heldout_combinations"]
    heldout_lengths = totals["heldout_lengths"]
    if min(episodes, quality, heldout_combinations, heldout_lengths) < 1:
        raise ValueError("V24 development loader lacks required strata")
    heldout_query = (
        totals["heldout_query_switch_correct"] / heldout_combinations
    )
    heldout_operation = (
        totals["heldout_operation_switch_correct"] / heldout_combinations
    )
    return {
        "episodes": episodes,
        "quality_examples": quality,
        "heldout_combinations": heldout_combinations,
        "heldout_lengths": heldout_lengths,
        "frame1_binary_choice_accuracy": (
            totals["binary_choice_correct"] / totals["binary_choice_examples"]
        ),
        "query_switch_accuracy": totals["query_switch_correct"] / episodes,
        "operation_switch_accuracy": (
            totals["operation_switch_correct"] / episodes
        ),
        "heldout_query_switch_accuracy": heldout_query,
        "heldout_operation_switch_accuracy": heldout_operation,
        "heldout_combination_minimum_switch_accuracy": min(
            heldout_query, heldout_operation
        ),
        "frame1_identity_top1": totals["frame1_identity_correct"] / quality,
        "identity_bank_identities": float(len(bank_characters)),
        "frame1_pixel_f1": totals["frame1_f1_sum"] / quality,
        "frame1_target_cosine": totals["frame1_cosine_sum"] / quality,
        "frame2_label_top1": totals["frame2_label_correct"] / quality,
        "label_bank_identities": float(len(LABEL_CHARACTERS)),
        "frame2_pixel_f1": totals["frame2_f1_sum"] / quality,
        "frame2_target_cosine": totals["frame2_cosine_sum"] / quality,
        "teacher_forced_label_agreement": totals["teacher_agreement"] / episodes,
        "frame2_generated_history_consistency": (
            totals["history_consistency"] / episodes
        ),
        "history_switch_accuracy": totals["history_switch_correct"] / episodes,
        "history_output_pixel_l1": totals["history_l1_sum"] / episodes,
        "query_output_pixel_l1": totals["query_l1_sum"] / episodes,
        "operation_output_pixel_l1": totals["operation_l1_sum"] / episodes,
        "header_output_pixel_l1": totals["header_l1_sum"] / episodes,
        "query_header_localization_accuracy": (
            totals["query_header_correct"] / episodes
        ),
        "operation_header_localization_accuracy": (
            totals["operation_header_correct"] / episodes
        ),
        "pair_header_localization_accuracy": (
            totals["pair_header_correct"] / episodes
        ),
        "packet_permutation_frame1_identity_consistency": (
            totals["permutation_frame1_identity"] / episodes
        ),
        "packet_permutation_frame2_identity_consistency": (
            totals["permutation_frame2_identity"] / episodes
        ),
        "packet_permutation_frame1_output_pixel_l1": (
            totals["permutation_frame1_l1_sum"] / episodes
        ),
        "packet_permutation_frame2_output_pixel_l1": (
            totals["permutation_frame2_l1_sum"] / episodes
        ),
        "distractor_frame1_identity_consistency": (
            totals["distractor_frame1_identity"] / episodes
        ),
        "distractor_frame2_identity_consistency": (
            totals["distractor_frame2_identity"] / episodes
        ),
        "heldout_length_frame1_identity_top1": (
            totals["heldout_length_frame1_correct"] / heldout_lengths
        ),
        "heldout_length_frame2_label_top1": (
            totals["heldout_length_frame2_correct"] / heldout_lengths
        ),
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
            (image.width * scale, image.height * scale), Image.Resampling.NEAREST
        )
    return image


@torch.no_grad()
def save_sample_sheet(
    model: VisualPacketRereadStream,
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
    with autocast_context(device, precision):
        generated = model(batch["prompt"][:count])
    tile = 48
    prompt_frames = batch["prompt"].shape[1]
    card_width = prompt_frames * tile + 16
    card_height = 3 * tile + 42
    sheet = Image.new(
        "RGB", (card_width + 16, count * card_height + 16), "#e9eff1"
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row in range(count):
        x = 8
        y = 8 + row * card_height
        draw.rectangle(
            (x, y, x + card_width - 8, y + card_height - 8),
            fill="white",
            outline="#91a4ab",
        )
        metadata = batch["metadata"][row]
        label = (
            f"packets={metadata['active_packets']} "
            f"heldout_length={metadata['heldout_length']}"
        )
        draw.text((x + 4, y + 4), label, fill="#24373f", font=font)
        for frame in range(prompt_frames):
            sheet.paste(
                _ink_image(batch["prompt"][row, frame], scale=1).resize(
                    (tile, tile), Image.Resampling.NEAREST
                ),
                (x + frame * tile, y + 18),
            )
        for frame in range(2):
            sheet.paste(
                _ink_image(generated[row, frame], scale=1).resize(
                    (tile, tile), Image.Resampling.NEAREST
                ),
                (x + frame * tile, y + 18 + tile),
            )
            sheet.paste(
                _ink_image(batch["target_stream"][row, frame], scale=1).resize(
                    (tile, tile), Image.Resampling.NEAREST
                ),
                (x + frame * tile, y + 18 + 2 * tile),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def validate_relation_checkpoint(
    checkpoint: dict[str, Any], *, checkpoint_sha256: str
) -> None:
    if checkpoint_sha256 != EXPECTED_RELATION_SHA256:
        raise ValueError(
            f"V24 requires V23 relation SHA-256 {EXPECTED_RELATION_SHA256}, "
            f"got {checkpoint_sha256}"
        )
    if checkpoint.get("architecture") != V23_RELATION_ARCHITECTURE:
        raise ValueError("V24 requires a V23 relation checkpoint")
    if checkpoint.get("route_mode") != RELATION_AWARE_ROUTE:
        raise ValueError("V24 requires the V23 relation-aware candidate")
    if checkpoint.get("smoke_only"):
        raise ValueError("V24 refuses a smoke-only V23 relation checkpoint")
    best = checkpoint.get("best_development")
    if not isinstance(best, dict) or not all(v23_candidate_gate_report(best).values()):
        raise ValueError("V24 requires the gate-selected V23 relation candidate")
    if int(checkpoint.get("step", -1)) != int(best.get("step", -2)):
        raise ValueError("V24 requires the exact selected V23 relation step")
    if checkpoint.get("pvf_sha256") != EXPECTED_PVF_SHA256:
        raise ValueError("V23 relation PVF receipt differs")
    if checkpoint.get("canonicalizer_sha256") != EXPECTED_CANONICALIZER_SHA256:
        raise ValueError("V23 relation canonicalizer receipt differs")
    if checkpoint.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise ValueError("V23 relation manifest receipt differs")


def _load_v23_relation_parts(
    checkpoint: dict[str, Any]
) -> tuple[nn.Module, float]:
    config = relation_circuit_config_from_payload(checkpoint["model_config"])
    operation_reader = nn.Sequential(
        nn.LayerNorm(config.visual_dim),
        nn.Linear(config.visual_dim, config.operation_hidden_dim),
        nn.SiLU(),
        nn.Linear(config.operation_hidden_dim, 1),
    )
    prefix = "operation_reader."
    operation_state = {
        name[len(prefix) :]: value
        for name, value in checkpoint["relation"].items()
        if name.startswith(prefix)
    }
    operation_reader.load_state_dict(operation_state)
    raw_temperature = checkpoint["relation"]["raw_temperature"].float()
    span = config.maximum_temperature - config.minimum_temperature
    temperature = config.minimum_temperature + span * raw_temperature.sigmoid()
    return operation_reader, float(temperature)


def _protocol_payload(
    args: argparse.Namespace, partition: dict[str, Any]
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
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
        "expected_canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "expected_relation_sha256": EXPECTED_RELATION_SHA256,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256": file_sha256(args.manifest),
        "partition": partition,
        "smoke_only": bool(args.smoke),
    }


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: VisualPacketRereadStream,
    optimizer: torch.optim.Optimizer,
    step: int,
    partition: dict[str, Any],
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    best_development: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "route_mode": args.route_mode,
        "model_config": visual_packet_stream_config_payload(model.config),
        "packet_stream": packet_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": EXPECTED_PVF_SHA256,
        "canonicalizer_checkpoint": args.canonicalizer_checkpoint,
        "canonicalizer_sha256": EXPECTED_CANONICALIZER_SHA256,
        "relation_checkpoint": args.relation_checkpoint,
        "relation_sha256": EXPECTED_RELATION_SHA256,
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
        raise ValueError("V24 requires batch size and development samples at least two")
    if args.identity_bank_views < 1:
        raise ValueError("V24 requires at least one identity-bank view")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    hashes = {
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
        if hashes[name] != expected:
            raise ValueError(
                f"V24 {name} SHA-256 differs: expected {expected}, got {hashes[name]}"
            )

    canonicalizer_checkpoint = torch.load(
        args.canonicalizer_checkpoint, map_location="cpu", weights_only=False
    )
    validate_canonicalizer_checkpoint(
        canonicalizer_checkpoint,
        checkpoint_sha256=hashes["canonicalizer"],
    )
    canonicalizer = VisualCanonicalizer()
    canonicalizer.load_state_dict(canonicalizer_checkpoint["canonicalizer"])

    relation_checkpoint = torch.load(
        args.relation_checkpoint, map_location="cpu", weights_only=False
    )
    validate_relation_checkpoint(
        relation_checkpoint, checkpoint_sha256=hashes["relation"]
    )
    operation_reader, match_temperature = _load_v23_relation_parts(
        relation_checkpoint
    )
    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    retina = pvf.retina
    del pvf

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_packet_character_bank(records, bank_size=args.bank_size)
    partitions = split_packet_characters(bank, salt=args.partition_salt)
    partition = packet_partition_receipt(partitions, salt=args.partition_salt)
    for key, expected in EXPECTED_PARTITION.items():
        if partition.get(key) != expected:
            raise ValueError(
                f"V24 partition receipt {key!r} differs: expected {expected!r}, "
                f"got {partition.get(key)!r}"
            )

    episode_config = VisualPacketEpisodeConfig()
    total_training_examples = args.maximum_steps * args.batch_size
    train_dataset = VisualPacketEpisodeDataset(
        partitions["train"],
        split="train",
        length=total_training_examples,
        config=episode_config,
        seed=args.dataset_seed,
    )
    development_dataset = VisualPacketEpisodeDataset(
        partitions["development"],
        split="development",
        length=args.development_samples,
        config=episode_config,
        seed=args.dataset_seed + 50_000,
    )
    model = VisualPacketRereadStream(
        VisualPacketStreamConfig(
            visual_dim=retina.config.visual_dim, route_mode=args.route_mode
        ),
        retina,
        canonicalizer,
        operation_reader,
        match_temperature=match_temperature,
    ).to(device)
    trainable = _trainable_parameters(model)
    if trainable != EXPECTED_PARAMETERS:
        raise ValueError(
            f"V24 requires {EXPECTED_PARAMETERS:,} trainable parameters, "
            f"got {trainable:,}"
        )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "fp16"
    )
    step = 0
    best_development: dict[str, Any] | None = None
    output_dir = Path(args.out) / args.route_mode
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to append a new V24 run to nonempty output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.jsonl"
    protocol = _protocol_payload(args, partition)
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
            raise ValueError("resume checkpoint is not a V24 packet stream")
        if checkpoint.get("route_mode") != args.route_mode:
            raise ValueError("resume route mode differs from requested V24 arm")
        if checkpoint.get("smoke_only") and not args.smoke:
            raise ValueError("a V24 smoke checkpoint cannot resume into evidence")
        if checkpoint.get("protocol") != protocol:
            raise ValueError("resume protocol differs from current sealed V24 run")
        expected_config = visual_packet_stream_config_from_payload(
            checkpoint["model_config"]
        )
        if expected_config != model.config:
            raise ValueError("resume model configuration differs from V24 protocol")
        load_packet_state(model, checkpoint["packet_stream"])
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
        collate_fn=visual_packet_collate,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_packet_collate,
    )
    bank_images = _development_bank_images(
        partitions["development"],
        views=args.identity_bank_views,
        config=episode_config,
        seed=args.dataset_seed + 100_000,
    ).to(device)
    label_images = _label_bank_images(episode_config).to(device)
    with torch.no_grad(), autocast_context(device, args.precision):
        bank_visual = encode_identity_bank(model, bank_images)
        label_visual = model.encode_images(label_images)
    del bank_images, label_images

    start_receipt = {
        "stage": "start",
        "architecture": ARCHITECTURE,
        "route_mode": args.route_mode,
        "device": str(device),
        "precision": args.precision,
        "trainable_parameters": trainable,
        "match_temperature": match_temperature,
        **{f"{name}_sha256": value for name, value in hashes.items()},
        "partition": partition,
        "boundary_receipt": model.boundary_receipt(),
        "smoke_only": bool(args.smoke),
    }
    print(json.dumps(start_receipt, ensure_ascii=False, sort_keys=True))
    append_jsonl(log_path, start_receipt)

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
            loss, train_metrics = packet_stream_loss(
                model,
                batch,
                stroke_weight=args.stroke_weight,
                generated_visual_weight=args.generated_visual_weight,
                localization_weight=args.localization_weight,
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
                label_visual=label_visual,
                device=device,
                precision=args.precision,
            )
            metrics.update(
                {"stage": "validation", "step": float(step), "route_mode": args.route_mode}
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
                path=output_dir / "development_samples" / f"step_{step:07d}.png",
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
                atomic_save(payload, output_dir / "checkpoint_selected_development.pt")

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
        "opaque_visual_gate_passed": False,
        "frozen_evaluation_permitted": False,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
    }
    print(json.dumps(complete, sort_keys=True))
    append_jsonl(log_path, complete)


if __name__ == "__main__":
    main()
