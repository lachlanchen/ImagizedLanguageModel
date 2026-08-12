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
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.ink_jepa_data import (
    load_visual_grammar_manifest,
    retinal_font_manifest,
)
from ilm.visual_lm.saccade_data import render_glyph_fovea
from ilm.visual_lm.visual_binding_data import (
    PARTITION_SALT,
    VisualBindingEpisodeConfig,
    VisualBindingEpisodeDataset,
    binding_partition_receipt,
    build_binding_character_bank,
    noncanonical_variant,
    split_binding_characters,
    visual_binding_collate,
)
from ilm.visual_lm.visual_binding_stream import (
    QUERY_AWARE_ROUTE,
    QUERY_BLIND_ROUTE,
    ROUTE_MODES,
    VisualBindingStream,
    VisualBindingStreamConfig,
    encode_identity_bank,
    visual_binding_batch_metrics,
    visual_binding_config_from_payload,
    visual_binding_config_payload,
    visual_binding_stream_loss,
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


ARCHITECTURE = "visual-binding-stream-v1"
PROTOCOL_DOCUMENT = "references/visual_binding_stream_v22_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/visual_binding_data.py",
    "ilm/visual_lm/visual_binding_stream.py",
    "scripts/eval_visual_binding_stream_development.py",
    "scripts/train_visual_binding_stream.py",
)
EXPECTED_PVF_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
EXPECTED_PARAMETERS = 3_410_128
EXPECTED_PARTITION = {
    "train_identities": 815,
    "development_identities": 104,
    "frozen_identities": 105,
    "development_identifiers_sha256": (
        "86007f870644707c6de2379f068c2ac5666265661891e0aa1ed964ed13815047"
    ),
    "frozen_identifiers_sha256": (
        "7e144212e1b90a64cd5b7ad095ed2b95ccd6aa52095b2ae474d30cfec5a438de"
    ),
}
FIXED_MODEL_ARGUMENTS = {
    "bank_size": 1_024,
    "prompt_length": 6,
    "answer_length": 1,
    "model_dim": 256,
    "transformer_blocks": 4,
    "attention_heads": 8,
    "feedforward_dim": 768,
    "writer_hidden_channels": 128,
    "writer_context_dim": 128,
    "writer_blocks": 3,
    "writer_patch_size": 12,
    "writer_stride": 8,
    "writer_padding": 2,
    "dropout": 0.05,
}
FIXED_LOSS_ARGUMENTS = {
    "stroke_weight": 4.0,
    "oracle_weight": 0.50,
    "visual_state_weight": 0.20,
    "field_state_weight": 0.15,
    "reread_visual_weight": 0.10,
    "reread_field_weight": 0.10,
    "attention_entropy_weight": 0.05,
}
FIXED_OPTIMIZATION_ARGUMENTS = {
    "lr": 3e-4,
    "minimum_lr_ratio": 0.10,
    "warmup_steps": 100,
    "weight_decay": 0.03,
    "gradient_clip": 1.0,
    "seed": 20260822,
    "dataset_seed": 20260823,
}
FIXED_EVIDENCE_ARGUMENTS = {
    "maximum_steps": 1_600,
    "batch_size": 64,
    "num_workers": 8,
    "precision": "bf16",
    "development_samples": 512,
    "identity_bank_views": 4,
    "validate_every": 200,
    "save_every": 200,
}
GATE_EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one preregistered V22 image-only visual binding stream arm."
        )
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--route-mode", choices=ROUTE_MODES, required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_binding_stream_v22")
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--prompt-length", type=int, default=6)
    parser.add_argument("--answer-length", type=int, default=1)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--transformer-blocks", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--feedforward-dim", type=int, default=768)
    parser.add_argument("--writer-hidden-channels", type=int, default=128)
    parser.add_argument("--writer-context-dim", type=int, default=128)
    parser.add_argument("--writer-blocks", type=int, default=3)
    parser.add_argument("--writer-patch-size", type=int, default=12)
    parser.add_argument("--writer-stride", type=int, default=8)
    parser.add_argument("--writer-padding", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--oracle-weight", type=float, default=0.50)
    parser.add_argument("--visual-state-weight", type=float, default=0.20)
    parser.add_argument("--field-state-weight", type=float, default=0.15)
    parser.add_argument("--reread-visual-weight", type=float, default=0.10)
    parser.add_argument("--reread-field-weight", type=float, default=0.10)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.05)
    parser.add_argument("--maximum-steps", type=int, default=1_600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--dataset-seed", type=int, default=20260823)
    parser.add_argument("--development-samples", type=int, default=512)
    parser.add_argument("--identity-bank-views", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--sample-count", type=int, default=8)
    return parser.parse_args()


def _require_fixed_arguments(args: argparse.Namespace) -> None:
    if args.partition_salt != PARTITION_SALT:
        raise ValueError(f"V22 requires partition salt {PARTITION_SALT!r}")
    for group in (
        FIXED_MODEL_ARGUMENTS,
        FIXED_LOSS_ARGUMENTS,
        FIXED_OPTIMIZATION_ARGUMENTS,
    ):
        for name, expected in group.items():
            if getattr(args, name) != expected:
                raise ValueError(f"V22 requires --{name.replace('_', '-')}={expected}")
    if args.smoke:
        if args.maximum_steps > 20:
            raise ValueError("V22 smoke mode is limited to 20 optimization steps")
        return
    for name, expected in FIXED_EVIDENCE_ARGUMENTS.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"V22 evidence requires --{name.replace('_', '-')}={expected}"
            )


def _strictly_above(value: float, threshold: float) -> bool:
    return value - threshold > GATE_EPSILON


def _strictly_below(value: float, threshold: float) -> bool:
    return threshold - value > GATE_EPSILON


def candidate_selection_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "binary_choice_accuracy": _strictly_above(
            metrics["binary_choice_accuracy"], 0.85
        ),
        "counterfactual_switch_accuracy": _strictly_above(
            metrics["counterfactual_switch_accuracy"], 0.80
        ),
        "heldout_combination_switch_accuracy": _strictly_above(
            metrics["heldout_combination_switch_accuracy"], 0.75
        ),
        "identity_top1": _strictly_above(metrics["identity_top1"], 0.45),
        "identity_bank_size": metrics["identity_bank_identities"] >= 96.0,
        "identity_query_gain": _strictly_above(
            metrics["identity_top1"]
            - metrics["query_shuffled_identity_top1"],
            0.20,
        ),
        "target_cosine": _strictly_above(metrics["target_cosine"], 0.78),
        "pixel_f1": _strictly_above(metrics["pixel_f1"], 0.58),
        "oracle_pixel_f1": _strictly_above(metrics["oracle_pixel_f1"], 0.64),
        "paired_output_pixel_l1": _strictly_above(
            metrics["paired_output_pixel_l1"], 0.08
        ),
        "not_operation_copy": _strictly_above(
            metrics["target_margin_over_operation"], 0.15
        ),
        "not_query_label_copy": _strictly_above(
            metrics["target_margin_over_query_label"], 0.15
        ),
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
    }


def control_selection_gate_report(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "query_blind_invariant": _strictly_below(
            metrics["paired_output_pixel_l1"], 1e-7
        ),
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
    }


def selection_gate_report(
    metrics: dict[str, float],
    route_mode: str,
) -> dict[str, bool]:
    if route_mode == QUERY_AWARE_ROUTE:
        return candidate_selection_gate_report(metrics)
    if route_mode == QUERY_BLIND_ROUTE:
        return control_selection_gate_report(metrics)
    raise ValueError(f"unknown route mode {route_mode!r}")


def selection_rank(metrics: dict[str, float], route_mode: str) -> tuple[float, ...]:
    if route_mode == QUERY_AWARE_ROUTE:
        return (
            metrics["counterfactual_switch_accuracy"],
            metrics["identity_top1"],
            metrics["pixel_f1"],
            -metrics["step"],
        )
    if route_mode == QUERY_BLIND_ROUTE:
        return (
            metrics["binary_choice_accuracy"],
            metrics["pixel_f1"],
            -metrics["step"],
        )
    raise ValueError(f"unknown route mode {route_mode!r}")


def paired_gate_report(
    candidate: dict[str, float],
    control: dict[str, float],
    *,
    candidate_parameters: int,
    control_parameters: int,
    parameter_shapes_equal: bool,
) -> dict[str, bool]:
    return {
        "candidate_switch_gain": _strictly_above(
            candidate["counterfactual_switch_accuracy"]
            - control["counterfactual_switch_accuracy"],
            0.25,
        ),
        "candidate_identity_gain": _strictly_above(
            candidate["identity_top1"] - control["identity_top1"],
            0.20,
        ),
        "candidate_paired_output_gain": _strictly_above(
            candidate["paired_output_pixel_l1"]
            - control["paired_output_pixel_l1"],
            0.06,
        ),
        "candidate_arm_gates": all(
            candidate_selection_gate_report(candidate).values()
        ),
        "control_arm_gates": all(
            control_selection_gate_report(control).values()
        ),
        "parameter_count_equal": candidate_parameters == control_parameters,
        "parameter_shapes_equal": parameter_shapes_equal,
    }


def student_boundary_is_clean(
    receipt: dict[str, bool | str],
    route_mode: str,
) -> bool:
    required_true = {
        "input_is_continuous_image",
        "output_is_continuous_image",
    }
    required_false = {
        "uses_strings",
        "uses_token_ids",
        "uses_unicode_ids",
        "uses_ocr",
        "uses_character_labels",
        "uses_operation_ids",
        "uses_slot_indices",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
        "retina_trainable",
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


def student_state_dict(model: VisualBindingStream) -> dict[str, torch.Tensor]:
    return {
        name: value
        for name, value in model.state_dict().items()
        if not name.startswith("retina.")
    }


def load_student_state(
    model: VisualBindingStream,
    state: dict[str, torch.Tensor],
) -> None:
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "V22 student state has unexpected keys: "
            f"{incompatible.unexpected_keys}"
        )
    if any(not key.startswith("retina.") for key in incompatible.missing_keys):
        raise ValueError(
            "V22 student state is incomplete: "
            f"{incompatible.missing_keys}"
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


def _development_bank_images(
    characters: Sequence[str],
    *,
    views: int,
    config: VisualBindingEpisodeConfig,
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


_EXAMPLE_WEIGHTED_METRICS = frozenset(
    {
        "binary_choice_accuracy",
        "target_cosine",
        "distractor_cosine",
        "pixel_f1",
        "pixel_l1",
        "oracle_pixel_f1",
        "oracle_pixel_l1",
        "target_attention_mass",
        "attention_max",
        "identity_top1",
    }
)
_PAIR_WEIGHTED_METRICS = frozenset(
    {
        "counterfactual_switch_accuracy",
        "paired_output_pixel_l1",
        "query_shuffled_output_pixel_l1",
        "target_margin_over_operation",
        "target_margin_over_query_label",
        "query_shuffled_identity_top1",
    }
)


@torch.no_grad()
def evaluate_development(
    model: VisualBindingStream,
    loader: DataLoader,
    *,
    bank_visual: torch.Tensor,
    bank_characters: Sequence[str],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for raw_batch in loader:
        batch = _device_batch(raw_batch, device)
        with autocast_context(device, precision):
            metrics = visual_binding_batch_metrics(
                model,
                batch,
                bank_visual=bank_visual,
                bank_characters=bank_characters,
            )
        examples = float(metrics["examples"])
        pairs = float(metrics["pairs"])
        heldout_pairs = float(metrics["heldout_pairs"])
        seen_pairs = float(metrics["seen_pairs"])
        for key, value in metrics.items():
            scalar = float(value)
            if key in {"examples", "pairs", "heldout_pairs", "seen_pairs"}:
                totals[key] = totals.get(key, 0.0) + scalar
                continue
            if not np.isfinite(scalar):
                continue
            if key in _EXAMPLE_WEIGHTED_METRICS:
                weight = examples
            elif key in _PAIR_WEIGHTED_METRICS:
                weight = pairs
            elif key == "heldout_combination_switch_accuracy":
                weight = heldout_pairs
            elif key == "seen_combination_switch_accuracy":
                weight = seen_pairs
            else:
                weight = pairs
            totals[key] = totals.get(key, 0.0) + scalar * weight
            denominators[key] = denominators.get(key, 0.0) + weight
    for key, denominator in denominators.items():
        totals[key] /= max(1.0, denominator)
    totals["frozen_images_instantiated"] = 0.0
    totals["identity_bank_identities"] = float(len(bank_characters))
    totals["student_boundary_clean"] = float(
        student_boundary_is_clean(model.boundary_receipt(), model.config.route_mode)
    )
    return totals


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
    model: VisualBindingStream,
    loader: DataLoader,
    *,
    path: Path,
    device: torch.device,
    precision: str,
    sample_count: int,
) -> None:
    model.eval()
    raw_batch = next(iter(loader))
    batch = _device_batch(raw_batch, device)
    count = min(sample_count, batch["prompt"].shape[0])
    with autocast_context(device, precision):
        generated = model(batch["prompt"][:count])[:, 0]
        counterfactual = model(batch["counterfactual_prompt"][:count])[:, 0]
    tile_width = 6 * 64 + 8
    tile_height = 64 * 3 + 54
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * tile_width + 16, rows * tile_height + 16),
        "#eef2f3",
    )
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()
    for index in range(count):
        column = index % columns
        row = index // columns
        x = 8 + column * tile_width
        y = 8 + row * tile_height
        metadata = batch["metadata"][index]
        draw.rectangle(
            (x, y, x + tile_width - 8, y + tile_height - 8),
            fill="white",
            outline="#9cabb1",
        )
        draw.text(
            (x + 4, y + 4),
            "heldout" if metadata["heldout_combination"] else "seen",
            font=label_font,
            fill="#24373f",
        )
        frame_y = y + 18
        for frame_index in range(6):
            sheet.paste(
                _ink_image(batch["prompt"][index, frame_index]),
                (x + frame_index * 64, frame_y),
            )
        answer_y = frame_y + 72
        answer_images = (
            generated[index],
            batch["target"][index],
            counterfactual[index],
            batch["counterfactual_target"][index],
        )
        labels = ("pred", "target", "cf pred", "cf target")
        for answer_index, (image, label) in enumerate(zip(answer_images, labels)):
            answer_x = x + answer_index * 92
            draw.text(
                (answer_x, answer_y - 11),
                label,
                font=label_font,
                fill="#50636b",
            )
            sheet.paste(_ink_image(image), (answer_x, answer_y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: VisualBindingStream,
    optimizer: torch.optim.Optimizer,
    step: int,
    partition: dict[str, Any],
    protocol: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "route_mode": args.route_mode,
        "model_config": visual_binding_config_payload(model.config),
        "student": student_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "pvf_checkpoint": args.pvf_checkpoint,
        "pvf_sha256": protocol["expected_pvf_sha256"],
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
    }


def _protocol_payload(
    args: argparse.Namespace,
    partition: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": {
            path: file_sha256(path)
            for path in SOURCE_FILES
        },
        "route_mode": args.route_mode,
        "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
        "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
        "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
        "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "expected_pvf_sha256": EXPECTED_PVF_SHA256,
        "manifest_sha256": file_sha256(args.manifest),
        "partition": partition,
        "smoke_only": bool(args.smoke),
    }


def main() -> None:
    args = parse_args()
    _require_fixed_arguments(args)
    if args.maximum_steps < 1 or args.batch_size < 2:
        raise ValueError("V22 requires positive steps and batch size at least two")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats(device)

    pvf_sha256 = file_sha256(args.pvf_checkpoint)
    if pvf_sha256 != EXPECTED_PVF_SHA256:
        raise ValueError(
            f"V22 requires PVF SHA-256 {EXPECTED_PVF_SHA256}, got {pvf_sha256}"
        )
    pvf, _ = load_pvf(args.pvf_checkpoint, device)
    retina = pvf.retina
    del pvf

    records = load_visual_grammar_manifest(args.manifest)
    bank = build_binding_character_bank(records, bank_size=args.bank_size)
    partitions = split_binding_characters(bank, salt=args.partition_salt)
    partition = binding_partition_receipt(
        partitions,
        salt=args.partition_salt,
    )
    for key, expected in EXPECTED_PARTITION.items():
        if partition.get(key) != expected:
            raise ValueError(
                f"V22 partition receipt {key!r} differs: "
                f"expected {expected!r}, got {partition.get(key)!r}"
            )
    episode_config = VisualBindingEpisodeConfig()
    total_training_examples = args.maximum_steps * args.batch_size
    train_dataset = VisualBindingEpisodeDataset(
        partitions["train"],
        split="train",
        length=total_training_examples,
        config=episode_config,
        seed=args.dataset_seed,
    )
    development_dataset = VisualBindingEpisodeDataset(
        partitions["development"],
        split="development",
        length=args.development_samples,
        config=episode_config,
        seed=args.dataset_seed + 50_000,
    )

    config = VisualBindingStreamConfig(
        prompt_length=args.prompt_length,
        answer_length=args.answer_length,
        visual_dim=retina.config.visual_dim,
        spatial_channels=retina.config.retina_base_channels * 3,
        model_dim=args.model_dim,
        transformer_blocks=args.transformer_blocks,
        attention_heads=args.attention_heads,
        feedforward_dim=args.feedforward_dim,
        writer_hidden_channels=args.writer_hidden_channels,
        writer_context_dim=args.writer_context_dim,
        writer_blocks=args.writer_blocks,
        writer_patch_size=args.writer_patch_size,
        writer_stride=args.writer_stride,
        writer_padding=args.writer_padding,
        dropout=args.dropout,
        route_mode=args.route_mode,
    )
    model = VisualBindingStream(config, retina).to(device)
    trainable = _trainable_parameters(model)
    if trainable != EXPECTED_PARAMETERS:
        raise ValueError(
            f"V22 requires {EXPECTED_PARAMETERS:,} trainable parameters, got "
            f"{trainable:,}"
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
    output_dir = Path(args.out)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"refusing to append a new V22 run to nonempty output: {output_dir}"
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
            raise ValueError("resume checkpoint is not a V22 visual binding stream")
        if checkpoint.get("route_mode") != args.route_mode:
            raise ValueError("resume route mode differs from requested route")
        if checkpoint.get("smoke_only") and not args.smoke:
            raise ValueError("a V22 smoke checkpoint cannot resume into evidence")
        expected_config = visual_binding_config_from_payload(
            checkpoint["model_config"]
        )
        if expected_config != model.config:
            raise ValueError("resume model configuration differs from V22 protocol")
        load_student_state(model, checkpoint["student"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        best_development = checkpoint.get("best_development")
        if step >= args.maximum_steps:
            raise ValueError("resume checkpoint already reached maximum steps")

    remaining_indices = range(step * args.batch_size, total_training_examples)
    train_subset = Subset(train_dataset, remaining_indices)
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        collate_fn=visual_binding_collate,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=visual_binding_collate,
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
            loss, train_metrics, _ = visual_binding_stream_loss(
                model,
                batch["prompt"],
                batch["target"],
                batch["counterfactual_prompt"],
                batch["counterfactual_target"],
                batch["oracle_reference"],
                batch["counterfactual_oracle_reference"],
                **FIXED_LOSS_ARGUMENTS,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
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
                row["peak_cuda_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            print(json.dumps(row, sort_keys=True))
            append_jsonl(log_path, row)

        should_validate = (
            step % args.validate_every == 0 or step == args.maximum_steps
        )
        if should_validate:
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
            eligible = all(gates.values())
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

        should_save = step % args.save_every == 0 or step == args.maximum_steps
        if should_save:
            payload = _checkpoint_payload(
                args=args,
                model=model,
                optimizer=optimizer,
                step=step,
                partition=partition,
                protocol=protocol,
                metrics=last_metrics,
            )
            payload["best_development"] = best_development
            checkpoint_path = output_dir / f"checkpoint_step_{step:07d}.pt"
            atomic_save(payload, checkpoint_path)
            atomic_save(payload, output_dir / "checkpoint_latest.pt")
            if (
                best_development is not None
                and int(best_development["step"]) == step
            ):
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
        )
        payload["best_development"] = best_development
        atomic_save(payload, output_dir / "checkpoint_interrupted.pt")
        append_jsonl(
            log_path,
            {
                "stage": "interrupted",
                "step": step,
                "route_mode": args.route_mode,
                "checkpoint": str(output_dir / "checkpoint_interrupted.pt"),
            },
        )
        return

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
        "paired_control_gate_passed": False,
        "blinded_readability_gate_passed": False,
        "frozen_evaluation_permitted": False,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
    }
    print(json.dumps(complete, sort_keys=True))
    append_jsonl(log_path, complete)


if __name__ == "__main__":
    main()
