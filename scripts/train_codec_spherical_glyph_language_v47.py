#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
    render_canonical_character_bank,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47 import (
    V47_ARCHITECTURE,
    V47_PROTOCOL,
    CodecSphericalGlyphLanguageModelV47,
    CodecSphericalGlyphLanguageV47Config,
    codec_spherical_glyph_language_v47_boundary_receipt,
    codec_spherical_glyph_language_v47_config_from_payload,
    codec_spherical_glyph_language_v47_config_payload,
    load_verified_v34_codec,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_data import (
    V47_PAIR_COUNT,
    V47_PAIR_SEED,
    CodecSphericalPairTrainingDatasetV47,
    canonical_glyph_pair_student_batch,
    canonical_glyph_pair_training_collate,
    codec_spherical_glyph_language_v47_data_boundary_receipt,
    validate_frozen_pair_sequence_v47,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_evaluation import (
    V47_TOTAL_PARAMETER_LIMIT,
    V47_TRAINABLE_PARAMETER_LIMIT,
    codec_spherical_field_preflight_v47,
)
from ilm.visual_lm.codec_spherical_glyph_language_v47_training import (
    V47_LOSS_WEIGHTS,
    codec_spherical_glyph_language_v47_loss,
    codec_spherical_glyph_language_v47_pair_loss,
)
from ilm.visual_lm.factorized_visual_context_data import (
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.visual_cell_data import (
    V25_DEVELOPMENT_FONTS,
    V25_MANIFEST_SHA256,
    load_v25_records,
    render_visual_cell_stream,
    verify_v25_manifest,
    visual_cell_font_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_character_statistics,
)
from scripts.train_canonical_glyph_language_v42 import (
    _append_jsonl,
    _atomic_json,
    _atomic_save,
    _scheduled_lr,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_CODEC = "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
DEFAULT_OUTPUT = "artifacts/codec_spherical_glyph_language_v47_20260815"
SOURCE_FILES = (
    "ilm/visual_lm/codec_spherical_glyph_language_v47.py",
    "ilm/visual_lm/codec_spherical_glyph_language_v47_data.py",
    "ilm/visual_lm/codec_spherical_glyph_language_v47_training.py",
    "ilm/visual_lm/codec_spherical_glyph_language_v47_evaluation.py",
    "scripts/train_codec_spherical_glyph_language_v47.py",
    "scripts/eval_codec_spherical_glyph_language_v47.py",
)
FIXED_OPTIMIZATION = {
    "steps": 10_000,
    "batch_size": 8,
    "gradient_accumulation": 2,
    "pair_batch_size": 8,
    "pair_count": V47_PAIR_COUNT,
    "learning_rate": 3e-4,
    "warmup_steps": 500,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "maximum_contrastive_positions": 512,
    "maximum_energy_positions": 128,
    "energy_samples": 4,
    "model_seed": 20264700,
    "dataset_seed": 20264701,
    "pair_seed": V47_PAIR_SEED,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen V47 codec-spherical Chinese language core."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--codec", default=DEFAULT_CODEC)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--batch-size",
        type=int,
        default=FIXED_OPTIMIZATION["batch_size"],
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=FIXED_OPTIMIZATION["gradient_accumulation"],
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=FIXED_OPTIMIZATION["pair_batch_size"],
    )
    parser.add_argument(
        "--pair-count",
        type=int,
        default=FIXED_OPTIMIZATION["pair_count"],
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=FIXED_OPTIMIZATION["learning_rate"],
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=FIXED_OPTIMIZATION["warmup_steps"],
    )
    parser.add_argument(
        "--minimum-lr-ratio",
        type=float,
        default=FIXED_OPTIMIZATION["minimum_lr_ratio"],
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=FIXED_OPTIMIZATION["weight_decay"],
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=FIXED_OPTIMIZATION["gradient_clip"],
    )
    parser.add_argument(
        "--maximum-contrastive-positions",
        type=int,
        default=FIXED_OPTIMIZATION["maximum_contrastive_positions"],
    )
    parser.add_argument(
        "--maximum-energy-positions",
        type=int,
        default=FIXED_OPTIMIZATION["maximum_energy_positions"],
    )
    parser.add_argument(
        "--energy-samples",
        type=int,
        default=FIXED_OPTIMIZATION["energy_samples"],
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=FIXED_OPTIMIZATION["model_seed"],
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=FIXED_OPTIMIZATION["dataset_seed"],
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=FIXED_OPTIMIZATION["pair_seed"],
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if args.smoke:
        steps = min(args.steps, 2)
        pair_batch_size = min(args.pair_batch_size, 2)
        payload.update(
            {
                "steps": steps,
                "batch_size": min(args.batch_size, 2),
                "gradient_accumulation": 1,
                "pair_batch_size": pair_batch_size,
                "pair_count": steps * pair_batch_size,
                "num_workers": 0,
                "warmup_steps": 1,
                "maximum_contrastive_positions": min(
                    args.maximum_contrastive_positions,
                    64,
                ),
                "maximum_energy_positions": min(
                    args.maximum_energy_positions,
                    4,
                ),
                "energy_samples": 2,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return payload


def _assert_production_arguments(arguments: Mapping[str, Any]) -> None:
    if arguments["smoke"] or arguments["exploratory"]:
        return
    if arguments["precision"] != "bf16":
        raise ValueError("V47 production fixes BF16 precision")
    for key, expected in FIXED_OPTIMIZATION.items():
        if arguments[key] != expected:
            raise ValueError(
                f"V47 production fixes {key}={expected!r}, "
                f"got {arguments[key]!r}"
            )


def _resolve_device(value: str) -> torch.device:
    device = choose_device(value)
    if device.type != "cuda":
        return device
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _protocol_receipt(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "document": V47_PROTOCOL,
        "sha256": file_sha256(V47_PROTOCOL),
        "source_files_sha256": {
            path: file_sha256(path)
            for path in SOURCE_FILES
            if Path(path).exists()
        },
        "fixed_optimization": FIXED_OPTIMIZATION,
        "effective_arguments": dict(arguments),
    }


def _render_held_font_banks(
    characters: tuple[str, ...],
    render_config: CanonicalGlyphRenderConfig,
) -> dict[str, torch.Tensor]:
    writing = "".join(characters)
    return {
        "noto_sans_cjk_bold": render_visual_cell_stream(
            writing,
            config=render_config.visual_cell_config(),
            font_path=V25_DEVELOPMENT_FONTS[0],
            variant=0,
        ),
        "noto_serif_cjk_medium": render_visual_cell_stream(
            writing,
            config=render_config.visual_cell_config(),
            font_path=V25_DEVELOPMENT_FONTS[1],
            variant=0,
        ),
    }


def _checkpoint(
    model: CodecSphericalGlyphLanguageModelV47,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    codec_receipt: Mapping[str, Any],
    pair_receipt: Mapping[str, Any],
    preflight: Mapping[str, Any],
    training_generator: torch.Generator,
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "experiment": V47_ARCHITECTURE,
        "architecture": V47_ARCHITECTURE,
        "model_config": codec_spherical_glyph_language_v47_config_payload(
            model.config
        ),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "update": update,
        "pair_rows_consumed": update * int(arguments["pair_batch_size"]),
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "fonts": visual_cell_font_manifest(),
        "codec": dict(codec_receipt),
        "pair_sequence": dict(pair_receipt),
        "field_preflight": dict(preflight),
        "data_boundary": {
            "natural": {
                **canonical_glyph_data_boundary_receipt(),
                "architecture": V47_ARCHITECTURE,
            },
            "pair": codec_spherical_glyph_language_v47_data_boundary_receipt(),
        },
        "model_boundary": codec_spherical_glyph_language_v47_boundary_receipt(
            model
        ),
        "protocol": _protocol_receipt(arguments),
        "loss_weights": V47_LOSS_WEIGHTS.__dict__,
        "training_elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": bool(arguments["exploratory"]),
    }


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    _assert_production_arguments(arguments)
    positive_sizes = (
        "steps",
        "batch_size",
        "gradient_accumulation",
        "pair_batch_size",
        "pair_count",
        "maximum_contrastive_positions",
        "maximum_energy_positions",
        "energy_samples",
    )
    if any(int(arguments[key]) < 1 for key in positive_sizes):
        raise ValueError("V47 training sizes must be positive")
    if arguments["energy_samples"] < 2:
        raise ValueError("V47 energy score requires at least two samples")
    if arguments["pair_count"] != (
        arguments["steps"] * arguments["pair_batch_size"]
    ):
        raise ValueError("V47 must consume the pair sequence exactly once")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    if not arguments["smoke"] and not arguments["exploratory"]:
        if device.type != "cuda" or device.index != 0:
            raise ValueError("V47 production is frozen to CUDA device 0")
    seed_everything(arguments["model_seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    codec, codec_receipt = load_verified_v34_codec(
        arguments["codec"],
        strict_digest=True,
    )
    manifest_receipt = verify_v25_manifest(
        arguments["manifest"],
        strict=not arguments["smoke"] and not arguments["exploratory"],
    )
    if (
        not arguments["smoke"]
        and not arguments["exploratory"]
        and manifest_receipt["sha256"] != V25_MANIFEST_SHA256
    ):
        raise ValueError("V47 production requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"],
        strict_manifest=not arguments["smoke"] and not arguments["exploratory"],
    )
    render_config = CanonicalGlyphRenderConfig()
    partition_receipt = visual_cell_partition_receipt(records)

    resume_payload: dict[str, Any] | None = None
    model_config = CodecSphericalGlyphLanguageV47Config()
    start_update = 0
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"],
            map_location="cpu",
            weights_only=False,
        )
        if resume_payload.get("architecture") != V47_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V47")
        if resume_payload.get("codec") != codec_receipt:
            raise ValueError("resume checkpoint has a different V34 codec")
        model_config = codec_spherical_glyph_language_v47_config_from_payload(
            resume_payload["model_config"]
        )
        start_update = int(resume_payload["update"])
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested V47 updates")

    model = CodecSphericalGlyphLanguageModelV47(
        model_config,
        codec,
        codec_checkpoint_sha256=codec_receipt["checkpoint_sha256"],
        codec_state_sha256=codec_receipt["ema_tensor_state_sha256"],
    )
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"], strict=True)
    model.to(device)
    boundary = codec_spherical_glyph_language_v47_boundary_receipt(model)
    if (
        boundary["total_parameters"] >= V47_TOTAL_PARAMETER_LIMIT
        or boundary["trainable_parameters"] >= V47_TRAINABLE_PARAMETER_LIMIT
        or boundary["field_trainable_parameters"] != 0
    ):
        raise ValueError("V47 instantiated model violates its parameter budget")

    statistics = build_visual_character_statistics(
        records,
        bank_size=1_024,
        script_views_mode=render_config.script_views,
    )
    canonical_bank = render_canonical_character_bank(
        statistics,
        render_config=render_config,
    )
    held_banks = _render_held_font_banks(statistics.characters, render_config)
    preflight = codec_spherical_field_preflight_v47(
        model,
        canonical_bank,
        held_banks,
        device=device,
    )
    _atomic_json(preflight, output / "field_preflight.json")
    if not preflight["pass"]:
        raise RuntimeError("V47 frozen codec-sphere failed mandatory preflight")

    pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=arguments["pair_count"],
        seed=arguments["pair_seed"],
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    pair_receipt = validate_frozen_pair_sequence_v47(
        pairs,
        strict=not arguments["smoke"] and not arguments["exploratory"],
    )
    if resume_payload is not None and resume_payload.get("pair_sequence") != pair_receipt:
        raise ValueError("resume checkpoint has a different V47 pair sequence")

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    total_natural_examples = (
        arguments["steps"]
        * arguments["gradient_accumulation"]
        * arguments["batch_size"]
    )
    consumed_natural_examples = (
        start_update
        * arguments["gradient_accumulation"]
        * arguments["batch_size"]
    )
    consumed_pair_examples = start_update * arguments["pair_batch_size"]
    natural_dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_natural_examples,
    )
    pair_dataset = CodecSphericalPairTrainingDatasetV47(
        pairs,
        render_config=render_config,
        seed=arguments["pair_seed"],
    )
    natural_loader = DataLoader(
        Subset(
            natural_dataset,
            range(consumed_natural_examples, total_natural_examples),
        ),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_collate,
    )
    pair_loader = DataLoader(
        Subset(pair_dataset, range(consumed_pair_examples, len(pair_dataset))),
        batch_size=arguments["pair_batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_pair_training_collate,
    )
    natural_iterator = iter(natural_loader)
    pair_iterator = iter(pair_loader)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and arguments["precision"] == "fp16",
    )
    training_generator = torch.Generator(device=device)
    if resume_payload is not None and isinstance(
        resume_payload.get("training_generator_state"),
        torch.Tensor,
    ):
        training_generator.set_state(resume_payload["training_generator_state"])
    else:
        training_generator.manual_seed(arguments["model_seed"])

    prior_elapsed = (
        float(resume_payload.get("training_elapsed_seconds", 0.0))
        if resume_payload is not None
        else 0.0
    )
    log_path = output / "training_metrics.jsonl"
    started = time.perf_counter()
    final_metrics: dict[str, float] = {}
    stop_requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for update in range(start_update + 1, arguments["steps"] + 1):
            model.train()
            learning_rate = _scheduled_lr(
                update,
                base=arguments["learning_rate"],
                warmup=arguments["warmup_steps"],
                total=arguments["steps"],
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            natural_totals: dict[str, float] = {}
            for _ in range(arguments["gradient_accumulation"]):
                natural = canonical_glyph_student_batch(next(natural_iterator))
                natural = {
                    key: value.to(device, non_blocking=True)
                    for key, value in natural.items()
                }
                with autocast_context(device, arguments["precision"]):
                    prediction = model(natural["context"])
                    natural_loss = codec_spherical_glyph_language_v47_loss(
                        model,
                        prediction,
                        natural["target"],
                        generator=training_generator,
                        maximum_contrastive_positions=arguments[
                            "maximum_contrastive_positions"
                        ],
                        maximum_energy_positions=arguments[
                            "maximum_energy_positions"
                        ],
                        energy_samples=arguments["energy_samples"],
                    )
                    scaled_natural = (
                        natural_loss.loss / arguments["gradient_accumulation"]
                    )
                if scaler.is_enabled():
                    scaler.scale(scaled_natural).backward()
                else:
                    scaled_natural.backward()
                for key, value in natural_loss.detached_metrics().items():
                    natural_totals[key] = natural_totals.get(key, 0.0) + value

            paired = canonical_glyph_pair_student_batch(next(pair_iterator))
            paired = {
                key: value.to(device, non_blocking=True)
                for key, value in paired.items()
            }
            with autocast_context(device, arguments["precision"]):
                pair_loss = codec_spherical_glyph_language_v47_pair_loss(
                    model,
                    paired["contexts"],
                    paired["candidates"],
                    paired["assignment"],
                )
            if scaler.is_enabled():
                scaler.scale(pair_loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                pair_loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                arguments["gradient_clip"],
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if any(parameter.grad is not None for parameter in model.field.parameters()):
                raise RuntimeError("V47 frozen codec unexpectedly received gradients")

            final_metrics = {
                f"natural_{key}": value / arguments["gradient_accumulation"]
                for key, value in natural_totals.items()
            }
            final_metrics.update(pair_loss.detached_metrics())
            final_metrics.update(
                {
                    "objective": final_metrics["natural_loss"]
                    + final_metrics["pair"],
                    "update": float(update),
                    "pair_rows_consumed": float(
                        update * arguments["pair_batch_size"]
                    ),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": (
                        prior_elapsed + time.perf_counter() - started
                    ),
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            should_save = (
                update == arguments["steps"]
                or update % arguments["save_every"] == 0
                or stop_requested
            )
            if should_save:
                elapsed = prior_elapsed + time.perf_counter() - started
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint(
                    model,
                    optimizer,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    codec_receipt=codec_receipt,
                    pair_receipt=pair_receipt,
                    preflight=preflight,
                    training_generator=training_generator,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=final_metrics,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V47_ARCHITECTURE,
                            "update": update,
                            "pair_rows_consumed": payload["pair_rows_consumed"],
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "total_parameters": payload["model_boundary"][
                                "total_parameters"
                            ],
                            "trainable_parameters": payload["model_boundary"][
                                "trainable_parameters"
                            ],
                            "codec": codec_receipt,
                            "pair_sequence": pair_receipt,
                            "field_preflight": preflight,
                            "training_metrics": final_metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V47 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
