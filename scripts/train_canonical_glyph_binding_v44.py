#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_binding_v44 import (
    CanonicalGlyphBindingV44,
    CanonicalGlyphBindingV44Config,
    V44_ARCHITECTURE,
    canonical_glyph_binding_v44_boundary_receipt,
    canonical_glyph_binding_v44_config_payload,
)
from ilm.visual_lm.canonical_glyph_binding_v44_training import (
    canonical_glyph_binding_v44_loss,
    shuffle_v44_pair_prefixes,
)
from ilm.visual_lm.canonical_glyph_flow_v43_data import (
    CanonicalGlyphPairTrainingDataset,
    canonical_glyph_pair_student_batch,
    canonical_glyph_pair_training_collate,
)
from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
    canonical_glyph_language_config_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.factorized_visual_context_data import (
    FactorizedVisualSuffixPair,
    build_factorized_suffix_pairs,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    choose_device,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/canonical_glyph_binding_v44_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_BASE = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_OUTPUT = "artifacts/canonical_glyph_binding_v44_20260814"
PINNED_V42_SHA256 = "a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870"
SOURCE_FILES = (
    "ilm/visual_lm/canonical_glyph_binding_v44.py",
    "ilm/visual_lm/canonical_glyph_binding_v44_evaluation.py",
    "ilm/visual_lm/canonical_glyph_binding_v44_training.py",
    "scripts/train_canonical_glyph_binding_v44.py",
    "scripts/eval_canonical_glyph_binding_v44.py",
)
FIXED_OPTIMIZATION = {
    "steps": 3_000,
    "batch_size": 8,
    "train_pairs": 24_000,
    "holdout_pairs": 1_024,
    "learning_rate": 2e-4,
    "warmup_steps": 150,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "seed": 20264400,
    "dataset_seed": 20264401,
    "pair_seed": 20264402,
    "shuffle_seed": 20264403,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V44 frozen-base visual binding residual."
    )
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--resume")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--batch-size", type=int, default=FIXED_OPTIMIZATION["batch_size"]
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-pairs", type=int, default=FIXED_OPTIMIZATION["train_pairs"]
    )
    parser.add_argument(
        "--holdout-pairs", type=int, default=FIXED_OPTIMIZATION["holdout_pairs"]
    )
    parser.add_argument(
        "--learning-rate", type=float, default=FIXED_OPTIMIZATION["learning_rate"]
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=FIXED_OPTIMIZATION["warmup_steps"]
    )
    parser.add_argument(
        "--minimum-lr-ratio",
        type=float,
        default=FIXED_OPTIMIZATION["minimum_lr_ratio"],
    )
    parser.add_argument(
        "--weight-decay", type=float, default=FIXED_OPTIMIZATION["weight_decay"]
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=FIXED_OPTIMIZATION["gradient_clip"]
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument(
        "--pair-seed", type=int, default=FIXED_OPTIMIZATION["pair_seed"]
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=FIXED_OPTIMIZATION["shuffle_seed"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _scheduled_lr(
    step: int,
    *,
    base: float,
    warmup: int,
    total: int,
    minimum_ratio: float,
) -> float:
    if warmup > 0 and step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    payload = vars(args).copy()
    if args.smoke:
        payload.update(
            {
                "steps": min(args.steps, 2),
                "batch_size": min(args.batch_size, 2),
                "holdout_pairs": min(args.holdout_pairs, 4),
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
            }
        )
        payload["train_pairs"] = payload["steps"] * payload["batch_size"]
    return payload


def _validate_production_arguments(arguments: Mapping[str, Any]) -> None:
    if arguments["smoke"]:
        return
    for key, expected in FIXED_OPTIMIZATION.items():
        if arguments[key] != expected:
            raise ValueError(
                f"V44 production fixes {key}={expected!r}, got {arguments[key]!r}"
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


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pair_sequence_receipt(
    pairs: Sequence[FactorizedVisualSuffixPair],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for pair in pairs:
        for value in (
            pair.identifier_a,
            pair.script_view_a,
            pair.context_a,
            pair.target_a,
            pair.identifier_b,
            pair.script_view_b,
            pair.context_b,
            pair.target_b,
        ):
            digest.update(value.encode("utf-8") + b"\0")
    return {
        "count": len(pairs),
        "unique_suffixes": len({pair.suffix for pair in pairs}),
        "sha256": digest.hexdigest(),
    }


def _base_matches(
    model: CanonicalGlyphBindingV44,
    expected: Mapping[str, torch.Tensor],
) -> bool:
    actual = model.base.state_dict()
    return actual.keys() == expected.keys() and all(
        torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu())
        for name in actual
    )


def _checkpoint(
    model: CanonicalGlyphBindingV44,
    optimizer: torch.optim.Optimizer,
    shuffle_generator: torch.Generator,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    training_pairs_receipt: Mapping[str, Any],
    holdout_pairs_receipt: Mapping[str, Any],
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
    base_checkpoint_sha256: str,
    base_state_sha256: str,
    base_state_exact: bool,
) -> dict[str, Any]:
    source_hashes = {
        path: file_sha256(path) for path in SOURCE_FILES if Path(path).exists()
    }
    return {
        "experiment": "canonical-glyph-binding-v44",
        "architecture": V44_ARCHITECTURE,
        "language_config": canonical_glyph_language_config_payload(
            model.language_config
        ),
        "v44_config": canonical_glyph_binding_v44_config_payload(model.config),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "shuffle_generator_state": shuffle_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(model.adapter.output.weight.device)
            if model.adapter.output.weight.device.type == "cuda"
            else None
        ),
        "update": update,
        "base_v42_checkpoint_sha256": base_checkpoint_sha256,
        "base_v42_state_sha256": base_state_sha256,
        "base_v42_state_exact": base_state_exact,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "pair_pool": {
            "seed": int(arguments["pair_seed"]),
            "suffix_cells": 4,
            "training": dict(training_pairs_receipt),
            "holdout": dict(holdout_pairs_receipt),
            "one_pass": True,
            "candidate_columns_permuted": True,
        },
        "data_boundary": {
            "student_keys": ["context", "target", "contexts", "candidates", "assignment"],
            "metadata_excluded_from_student": True,
            "uses_strings": False,
            "uses_token_ids": False,
            "uses_unicode_ids": False,
            "uses_character_ids": False,
            "uses_ocr": False,
            "uses_visual_codebook": False,
            "uses_glyph_lookup": False,
            "candidate_bank_deployed": False,
        },
        "model_boundary": canonical_glyph_binding_v44_boundary_receipt(model),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": source_hashes,
            "fixed_optimization": FIXED_OPTIMIZATION,
            "effective_arguments": dict(arguments),
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
    }


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    _validate_production_arguments(arguments)
    if min(
        arguments["steps"],
        arguments["batch_size"],
        arguments["train_pairs"],
        arguments["holdout_pairs"],
    ) < 1:
        raise ValueError("V44 training sizes must be positive")
    if arguments["steps"] * arguments["batch_size"] != arguments["train_pairs"]:
        raise ValueError("V44 must consume every training pair exactly once")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    base_checkpoint_sha256 = file_sha256(arguments["base_checkpoint"])
    if base_checkpoint_sha256 != PINNED_V42_SHA256:
        raise ValueError(
            "V44 requires the pinned V42 checkpoint, got "
            f"{base_checkpoint_sha256}"
        )
    base_payload = torch.load(
        arguments["base_checkpoint"], map_location="cpu", weights_only=False
    )
    expected_base_state = base_payload["model"]
    base_state_sha256 = tensor_state_sha256(expected_base_state)
    language_config = canonical_glyph_language_config_from_payload(
        base_payload["model_config"]
    )
    model = CanonicalGlyphBindingV44(
        language_config,
        CanonicalGlyphBindingV44Config(),
    )
    model.base.load_state_dict(expected_base_state, strict=True)
    model.freeze_base()

    start_update = 0
    resume_payload: dict[str, Any] | None = None
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("architecture") != V44_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V44")
        if resume_payload["base_v42_checkpoint_sha256"] != base_checkpoint_sha256:
            raise ValueError("V44 resume base checkpoint differs")
        model.load_state_dict(resume_payload["model"], strict=True)
        model.freeze_base()
        start_update = int(resume_payload["update"])
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested updates")
    if not _base_matches(model, expected_base_state):
        raise RuntimeError("V44 base changed before optimization")
    model.to(device)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter_parameters = sum(parameter.numel() for parameter in trainable)
    if adapter_parameters >= 2_000_000:
        raise ValueError(f"V44 adapter is too large: {adapter_parameters:,}")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    shuffle_generator = torch.Generator(device=device).manual_seed(
        arguments["shuffle_seed"]
    )
    if resume_payload is not None:
        shuffle_generator.set_state(resume_payload["shuffle_generator_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"])
        if device.type == "cuda" and resume_payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume_payload["cuda_rng_state"], device)

    manifest_receipt = verify_v25_manifest(
        arguments["manifest"], strict=not arguments["smoke"]
    )
    if not arguments["smoke"] and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V44 requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"], strict_manifest=not arguments["smoke"]
    )
    render_config = CanonicalGlyphRenderConfig()
    all_pairs = build_factorized_suffix_pairs(
        records,
        split="train",
        suffix_cells=4,
        count=arguments["train_pairs"] + arguments["holdout_pairs"],
        seed=arguments["pair_seed"],
        require_different_identifiers=True,
        script_views_mode=render_config.script_views,
    )
    training_pairs = all_pairs[: arguments["train_pairs"]]
    holdout_pairs = all_pairs[arguments["train_pairs"] :]
    if set(pair.suffix for pair in training_pairs) & set(
        pair.suffix for pair in holdout_pairs
    ):
        raise RuntimeError("V44 train and holdout suffixes overlap")
    training_pairs_receipt = pair_sequence_receipt(training_pairs)
    holdout_pairs_receipt = pair_sequence_receipt(holdout_pairs)

    total_examples = arguments["train_pairs"]
    consumed_examples = start_update * arguments["batch_size"]
    natural_dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_examples,
    )
    pair_dataset = CanonicalGlyphPairTrainingDataset(
        training_pairs,
        render_config=render_config,
        seed=arguments["pair_seed"],
        length=total_examples,
    )
    natural_loader = DataLoader(
        Subset(natural_dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_collate,
    )
    pair_loader = DataLoader(
        Subset(pair_dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
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
    started = time.perf_counter()
    metrics: dict[str, float] = {}
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
            natural = canonical_glyph_student_batch(next(natural_iterator))
            paired = canonical_glyph_pair_student_batch(next(pair_iterator))
            natural = {
                key: value.to(device, non_blocking=True)
                for key, value in natural.items()
            }
            paired = {
                key: value.to(device, non_blocking=True)
                for key, value in paired.items()
            }
            shuffled = shuffle_v44_pair_prefixes(
                paired["contexts"],
                suffix_cells=model.config.suffix_cells,
                generator=shuffle_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                natural_output = model.language(natural["context"])
                loss = canonical_glyph_binding_v44_loss(
                    model,
                    natural_output,
                    natural["target"][:, -1],
                    paired["contexts"],
                    paired["candidates"],
                    paired["assignment"],
                    shuffled,
                )
            if scaler.is_enabled():
                scaler.scale(loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable, arguments["gradient_clip"]
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            metrics = loss.detached_metrics()
            metrics.update(
                {
                    "update": float(update),
                    "pairs_consumed": float(update * arguments["batch_size"]),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(output / "training_metrics.jsonl", metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
            should_save = (
                update == arguments["steps"]
                or update % arguments["save_every"] == 0
                or stop_requested
            )
            if should_save:
                elapsed = time.perf_counter() - started
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                base_state_exact = _base_matches(model, expected_base_state)
                if not base_state_exact:
                    raise RuntimeError("V44 optimizer modified the frozen V42 base")
                payload = _checkpoint(
                    model,
                    optimizer,
                    shuffle_generator,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=visual_cell_partition_receipt(records),
                    training_pairs_receipt=training_pairs_receipt,
                    holdout_pairs_receipt=holdout_pairs_receipt,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=metrics,
                    base_checkpoint_sha256=base_checkpoint_sha256,
                    base_state_sha256=base_state_sha256,
                    base_state_exact=base_state_exact,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V44_ARCHITECTURE,
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "total_parameters": sum(
                                parameter.numel() for parameter in model.parameters()
                            ),
                            "adapter_parameters": adapter_parameters,
                            "base_v42_state_exact": base_state_exact,
                            "unique_training_pairs_consumed": arguments["train_pairs"],
                            "training_pairs_receipt": training_pairs_receipt,
                            "holdout_pairs_receipt": holdout_pairs_receipt,
                            "training_metrics": metrics,
                            "checkpoint_sha256": file_sha256(
                                output / "checkpoint_final.pt"
                            ),
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                raise KeyboardInterrupt(
                    f"V44 stopped after checkpointing update {update}"
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
