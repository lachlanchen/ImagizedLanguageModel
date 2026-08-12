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
    VisualCanonicalizer,
    canonicalizer_loss,
    pixel_f1_rows,
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


ARCHITECTURE = "visual-canonicalizer-v23"
PROTOCOL_DOCUMENT = "references/visual_relation_circuit_v23_protocol.md"
SOURCE_FILES = (
    "ilm/visual_lm/visual_relation_data.py",
    "ilm/visual_lm/visual_relation_circuit.py",
    "scripts/train_visual_canonicalizer_v23.py",
)
EXPECTED_PVF_SHA256 = (
    "90791001203640f0de66316cf2e30b3e2c588480fef0e3d9d4f6283ba043ecbe"
)
EXPECTED_MANIFEST_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_PARAMETERS = 1_122_081
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
}
FIXED_OPTIMIZATION_ARGUMENTS = {
    "lr": 5e-4,
    "minimum_lr_ratio": 0.10,
    "warmup_steps": 50,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "seed": 20260824,
    "dataset_seed": 20260825,
}
FIXED_EVIDENCE_ARGUMENTS = {
    "maximum_steps": 1_200,
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
        description="Train the preregistered V23 image-only visual canonicalizer."
    )
    parser.add_argument("--pvf-checkpoint", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out", default="artifacts/visual_canonicalizer_v23")
    parser.add_argument("--partition-salt", default=PARTITION_SALT)
    parser.add_argument("--bank-size", type=int, default=1_024)
    parser.add_argument("--stroke-weight", type=float, default=4.0)
    parser.add_argument("--maximum-steps", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--dataset-seed", type=int, default=20260825)
    parser.add_argument("--development-samples", type=int, default=512)
    parser.add_argument("--identity-bank-views", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=200)
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


def canonicalizer_selection_gate_report(
    metrics: dict[str, float],
) -> dict[str, bool]:
    return {
        "pixel_f1": _strictly_above(metrics["pixel_f1"], 0.72),
        "identity_top1": _strictly_above(metrics["identity_top1"], 0.80),
        "target_cosine": _strictly_above(metrics["target_cosine"], 0.85),
        "raw_source_pixel_f1_gain": _strictly_above(
            metrics["pixel_f1"] - metrics["raw_source_pixel_f1"], 0.12
        ),
        "source_shuffled_pixel_f1_gain": _strictly_above(
            metrics["pixel_f1"] - metrics["source_shuffled_pixel_f1"], 0.25
        ),
        "source_shuffled_identity_gain": _strictly_above(
            metrics["identity_top1"]
            - metrics["source_shuffled_identity_top1"],
            0.70,
        ),
        "ink_fraction_lower": metrics["ink_fraction"] >= 0.03,
        "ink_fraction_upper": metrics["ink_fraction"] <= 0.50,
        "identity_bank_complete": metrics["identity_bank_identities"] == 109.0,
        "student_boundary_clean": metrics["student_boundary_clean"] == 1.0,
        "frozen_bank_sealed": metrics["frozen_images_instantiated"] == 0.0,
    }


def canonicalizer_selection_rank(metrics: dict[str, float]) -> tuple[float, ...]:
    return (
        metrics["pixel_f1"],
        metrics["identity_top1"],
        metrics["target_cosine"],
        -metrics["step"],
    )


def canonicalizer_boundary_is_clean(receipt: dict[str, bool | str]) -> bool:
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
        "uses_font_ids",
        "uses_visual_codebook",
        "uses_glyph_lookup",
        "uses_external_language_model",
    }
    return (
        receipt.get("architecture") == ARCHITECTURE
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


def _canonicalizer_pair(
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    sources = torch.cat(
        (batch["oracle_reference"], batch["counterfactual_oracle_reference"])
    )
    targets = torch.cat((batch["target"], batch["query_counterfactual_target"]))
    characters = [
        metadata["target_character"] for metadata in batch["metadata"]
    ] + [
        metadata["counterfactual_target_character"]
        for metadata in batch["metadata"]
    ]
    return sources, targets, characters


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


def _encode_images(retina: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    visual, _ = retina.forward_with_field(images.float())
    return F.normalize(visual.float(), dim=-1)


@torch.no_grad()
def encode_identity_bank(
    retina: torch.nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    identities, views = images.shape[:2]
    visual = _encode_images(retina, images.flatten(0, 1))
    return F.normalize(visual.reshape(identities, views, -1).mean(dim=1), dim=-1)


@torch.no_grad()
def evaluate_development(
    model: VisualCanonicalizer,
    retina: torch.nn.Module,
    loader: DataLoader,
    *,
    bank_visual: torch.Tensor,
    bank_characters: Sequence[str],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    character_to_index = {
        character: index for index, character in enumerate(bank_characters)
    }
    totals = {
        "development_loss": 0.0,
        "pixel_f1": 0.0,
        "pixel_l1": 0.0,
        "raw_source_pixel_f1": 0.0,
        "source_shuffled_pixel_f1": 0.0,
        "identity_top1": 0.0,
        "source_shuffled_identity_top1": 0.0,
        "target_cosine": 0.0,
        "ink_fraction": 0.0,
    }
    examples = 0
    for raw_batch in loader:
        batch = _device_batch(raw_batch, device)
        source, target, characters = _canonicalizer_pair(batch)
        expected = torch.tensor(
            [character_to_index[character] for character in characters],
            device=device,
        )
        permutation = torch.arange(source.shape[0], device=device).roll(1)
        with autocast_context(device, precision):
            logits, _ = model.logits_with_trace(source)
            generated = logits.sigmoid()
            shuffled_generated = model(source[permutation])
            loss, _ = topology_loss(
                logits,
                target,
                stroke_weight=FIXED_LOSS_ARGUMENTS["stroke_weight"],
            )
            generated_visual = _encode_images(retina, generated)
            shuffled_visual = _encode_images(retina, shuffled_generated)
            target_visual = _encode_images(retina, target)

        rows = source.shape[0]
        totals["development_loss"] += float(loss) * rows
        totals["pixel_f1"] += float(pixel_f1_rows(generated, target).sum())
        totals["pixel_l1"] += float(
            (generated - target).abs().flatten(1).mean(dim=1).sum()
        )
        totals["raw_source_pixel_f1"] += float(
            pixel_f1_rows(source, target).sum()
        )
        totals["source_shuffled_pixel_f1"] += float(
            pixel_f1_rows(shuffled_generated, target).sum()
        )
        totals["identity_top1"] += float(
            ((generated_visual @ bank_visual.T).argmax(dim=1) == expected).sum()
        )
        totals["source_shuffled_identity_top1"] += float(
            ((shuffled_visual @ bank_visual.T).argmax(dim=1) == expected).sum()
        )
        totals["target_cosine"] += float(
            (generated_visual * target_visual).sum(dim=1).sum()
        )
        totals["ink_fraction"] += float(
            generated.flatten(1).mean(dim=1).sum()
        )
        examples += rows
    if examples < 1:
        raise ValueError("V23 canonicalizer development loader is empty")
    metrics = {key: value / examples for key, value in totals.items()}
    metrics.update(
        {
            "examples": float(examples),
            "identity_bank_identities": float(len(bank_characters)),
            "frozen_images_instantiated": 0.0,
            "student_boundary_clean": float(
                canonicalizer_boundary_is_clean(model.boundary_receipt())
            ),
        }
    )
    return metrics


def _ink_image(tensor: torch.Tensor, scale: int = 3) -> Image.Image:
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
    model: VisualCanonicalizer,
    loader: DataLoader,
    *,
    path: Path,
    device: torch.device,
    precision: str,
    sample_count: int,
) -> None:
    model.eval()
    batch = _device_batch(next(iter(loader)), device)
    source, target, _ = _canonicalizer_pair(batch)
    count = min(sample_count, source.shape[0])
    permutation = torch.arange(count, device=device).roll(1)
    with autocast_context(device, precision):
        generated = model(source[:count])
        shuffled = model(source[:count][permutation])
    rows = (
        ("source", source[:count]),
        ("target", target[:count]),
        ("generated", generated),
        ("shuffled source", shuffled),
    )
    tile = 96
    label_width = 128
    margin = 12
    sheet = Image.new(
        "RGB",
        (label_width + count * tile + 2 * margin, len(rows) * tile + 2 * margin),
        "#eef2f3",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, (label, images) in enumerate(rows):
        y = margin + row_index * tile
        draw.text((margin, y + tile // 2 - 6), label, fill="#24373f", font=font)
        for column in range(count):
            sheet.paste(
                _ink_image(images[column]),
                (margin + label_width + column * tile, y),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _protocol_payload(
    args: argparse.Namespace,
    partition: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "stage": "canonicalizer",
        "protocol_document": PROTOCOL_DOCUMENT,
        "protocol_sha256": file_sha256(PROTOCOL_DOCUMENT),
        "source_files_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "fixed_model_arguments": FIXED_MODEL_ARGUMENTS,
        "fixed_loss_arguments": FIXED_LOSS_ARGUMENTS,
        "fixed_optimization_arguments": FIXED_OPTIMIZATION_ARGUMENTS,
        "fixed_evidence_arguments": FIXED_EVIDENCE_ARGUMENTS,
        "selection_gates": {
            "pixel_f1": ">0.72",
            "identity_top1": ">0.80",
            "target_cosine": ">0.85",
            "raw_source_pixel_f1_gain": ">0.12",
            "source_shuffled_pixel_f1_gain": ">0.25",
            "source_shuffled_identity_gain": ">0.70",
            "ink_fraction": "[0.03,0.50]",
            "student_boundary_clean": True,
            "frozen_images_instantiated": 0,
        },
        "expected_trainable_parameters": EXPECTED_PARAMETERS,
        "expected_pvf_sha256": EXPECTED_PVF_SHA256,
        "expected_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256": file_sha256(args.manifest),
        "partition": partition,
        "smoke_only": bool(args.smoke),
    }


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: VisualCanonicalizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    partition: dict[str, Any],
    protocol: dict[str, Any],
    metrics: dict[str, Any],
    best_development: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "stage": "canonicalizer",
        "canonicalizer": model.state_dict(),
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

    model = VisualCanonicalizer().to(device)
    trainable = _trainable_parameters(model)
    if trainable != EXPECTED_PARAMETERS:
        raise ValueError(
            f"V23 requires {EXPECTED_PARAMETERS:,} canonicalizer parameters, "
            f"got {trainable:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
            f"refusing to append a new V23 run to nonempty output: {output_dir}"
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
            raise ValueError("resume checkpoint is not a V23 canonicalizer")
        if checkpoint.get("smoke_only") and not args.smoke:
            raise ValueError("a V23 smoke checkpoint cannot resume into evidence")
        if checkpoint.get("protocol") != protocol:
            raise ValueError("resume protocol differs from the current sealed V23 run")
        model.load_state_dict(checkpoint["canonicalizer"])
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
        bank_visual = encode_identity_bank(retina, bank_images)
    del bank_images

    run_receipt = {
        "stage": "start",
        "architecture": ARCHITECTURE,
        "device": str(device),
        "precision": args.precision,
        "trainable_parameters": trainable,
        "pvf_sha256": pvf_sha256,
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
        source, target, _ = _canonicalizer_pair(batch)
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
            loss, train_metrics = canonicalizer_loss(
                model,
                source,
                target,
                stroke_weight=args.stroke_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            row = {
                "stage": "train",
                "step": step,
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
                retina,
                development_loader,
                bank_visual=bank_visual,
                bank_characters=partitions["development"],
                device=device,
                precision=args.precision,
            )
            metrics.update({"stage": "validation", "step": float(step)})
            gates = canonicalizer_selection_gate_report(metrics)
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
                or canonicalizer_selection_rank(metrics)
                > canonicalizer_selection_rank(best_development)
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
            {"stage": "interrupted", "step": step, "checkpoint": str(interrupted)},
        )
        return

    selected_path = output_dir / "checkpoint_selected_development.pt"
    complete = {
        "stage": "complete",
        "step": step,
        "smoke_only": bool(args.smoke),
        "elapsed_seconds": time.perf_counter() - start_time,
        "peak_cuda_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "best_development": best_development,
        "selected_checkpoint": str(selected_path) if selected_path.exists() else None,
        "stage_b_permitted": bool(best_development is not None and not args.smoke),
        "frozen_evaluation_permitted": False,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
    }
    print(json.dumps(complete, sort_keys=True))
    append_jsonl(log_path, complete)


if __name__ == "__main__":
    main()
