#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ilm.visual_lm.visual_semantic_raster_data import (
    V32_ARCHITECTURE,
    VisualRasterContinuationDataset,
    VisualRasterInstructionDataset,
    VisualRasterRenderConfig,
    VisualRasterWarmupDataset,
    load_visual_raster_instructions,
    load_visual_text_records,
    visual_raster_partition,
    visual_semantic_raster_collate,
    visual_semantic_raster_data_boundary_receipt,
    visual_semantic_raster_student_batch,
)
from ilm.visual_lm.visual_semantic_raster_training import (
    SelectiveExponentialMovingAverage,
    raster_warmup_microstep,
    set_visual_semantic_raster_learning_rates,
    stage_cosine_learning_rate,
    visual_semantic_raster_optimizer_groups,
    visual_semantic_raster_optimizer_receipt,
    visual_semantic_raster_training_microstep,
)
from ilm.visual_lm.visual_semantic_raster_transducer import (
    VisualSemanticRasterConfig,
    VisualSemanticRasterTransducer,
    file_sha256,
    load_pixel_m4_reader,
    resolve_pixel_m4_checkpoint,
    visual_semantic_raster_boundary_receipt,
)


PROTOCOL_DOCUMENT = "references/visual_semantic_raster_transducer_v32_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "226f7c8e7619dec8ba3c297bdd9405b90a2f16a85fd3244a0d729bed8a3808a3"
)
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SOURCE_FILES = (
    "ilm/visual_lm/visual_semantic_raster_data.py",
    "ilm/visual_lm/visual_semantic_raster_transducer.py",
    "ilm/visual_lm/visual_semantic_raster_training.py",
    "scripts/train_visual_semantic_raster_v32.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int


FIXED_STAGES = (
    TrainingStage("raster-warmup", 2_000),
    TrainingStage("continuation", 4_000),
    TrainingStage("instruction", 6_000),
)
FIXED_OPTIMIZATION: dict[str, Any] = {
    "batch_size": 4,
    "gradient_accumulation": 16,
    "writer_learning_rate": 3e-4,
    "reader_learning_rate": 2e-5,
    "minimum_learning_rate_ratio": 0.10,
    "stage_warmup": 500,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "reader_final_blocks": 2,
    "reader_unfreeze_after": 3_000,
    "seed": 20_263_200,
    "precision": "bf16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V32 visual-semantic raster transducer."
    )
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--reader-checkpoint", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--raster-warmup-updates", type=int, default=2_000)
    parser.add_argument("--continuation-updates", type=int, default=4_000)
    parser.add_argument("--instruction-updates", type=int, default=6_000)
    parser.add_argument("--writer-learning-rate", type=float, default=3e-4)
    parser.add_argument("--reader-learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--stage-warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--reader-final-blocks", type=int, default=2)
    parser.add_argument("--reader-unfreeze-after", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=20_263_200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    suffix = "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence"
    values["out"] = args.out or f"artifacts/visual_semantic_raster_v32_{suffix}"
    if args.smoke:
        values.update(
            {
                "num_workers": 0,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "raster_warmup_updates": 1,
                "continuation_updates": 1,
                "instruction_updates": 1,
                "stage_warmup": 0,
                "reader_unfreeze_after": 0,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return argparse.Namespace(**values)


def training_stages(args: argparse.Namespace) -> tuple[TrainingStage, ...]:
    stages = (
        TrainingStage("raster-warmup", args.raster_warmup_updates),
        TrainingStage("continuation", args.continuation_updates),
        TrainingStage("instruction", args.instruction_updates),
    )
    if any(stage.updates < 1 for stage in stages):
        raise ValueError("V32 stages must contain at least one update")
    return stages


def require_preregistered_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        if args.tiny_model and not (args.smoke or args.exploratory):
            raise ValueError("V32 tiny model is not an evidence configuration")
        return
    expected_stages = {f"{stage.name.replace('-', '_')}_updates": stage.updates for stage in FIXED_STAGES}
    for name, expected in {**FIXED_OPTIMIZATION, **expected_stages}.items():
        if getattr(args, name) != expected:
            raise ValueError(f"V32 evidence requires --{name.replace('_', '-')}={expected}")
    if args.tiny_model:
        raise ValueError("V32 evidence requires the production model")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V32 requested CUDA but CUDA is unavailable")
    return device


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def instruction_microbatches_before(total: int) -> int:
    if total < 0:
        raise ValueError("V32 microbatch count cannot be negative")
    return total - total // 4


def stage_stream(stage: str, microbatch: int) -> str:
    if stage == "raster-warmup":
        return "warmup"
    if stage == "continuation":
        return "continuation"
    if stage == "instruction":
        return "continuation" if microbatch % 4 == 3 else "instruction"
    raise ValueError(f"unknown V32 stage: {stage}")


def _loader(
    dataset: Dataset,
    *,
    consumed_examples: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Iterator[dict[str, Any]]:
    if not 0 <= consumed_examples < len(dataset):
        raise ValueError("V32 loader has no remaining examples")
    subset = Subset(dataset, range(consumed_examples, len(dataset)))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        collate_fn=visual_semantic_raster_collate,
    )
    return iter(loader)


def build_stage_loaders(
    stage: TrainingStage,
    *,
    completed_updates: int,
    args: argparse.Namespace,
    render_config: VisualRasterRenderConfig,
    public_records: list[Any],
    instruction_records: list[Any],
    pin_memory: bool,
) -> dict[str, Iterator[dict[str, Any]]]:
    accumulated = args.gradient_accumulation
    batch_size = args.batch_size
    total_microbatches = stage.updates * accumulated
    consumed_microbatches = completed_updates * accumulated
    common = {
        "split": "train",
        "render_config": render_config,
    }
    if stage.name == "raster-warmup":
        dataset = VisualRasterWarmupDataset(
            public_records,
            seed=args.seed + 101,
            length=total_microbatches * batch_size,
            **common,
        )
        return {
            "warmup": _loader(
                dataset,
                consumed_examples=consumed_microbatches * batch_size,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
        }
    if stage.name == "continuation":
        dataset = VisualRasterContinuationDataset(
            public_records,
            seed=args.seed + 211,
            length=total_microbatches * batch_size,
            maximum_prompt_cells=32 if args.tiny_model else 160,
            **common,
        )
        return {
            "continuation": _loader(
                dataset,
                consumed_examples=consumed_microbatches * batch_size,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
        }
    if stage.name != "instruction":
        raise ValueError(f"unknown V32 stage: {stage.name}")

    instruction_total = instruction_microbatches_before(total_microbatches)
    instruction_consumed = instruction_microbatches_before(consumed_microbatches)
    continuation_total = total_microbatches - instruction_total
    continuation_consumed = consumed_microbatches - instruction_consumed
    loaders: dict[str, Iterator[dict[str, Any]]] = {}
    if instruction_total > instruction_consumed:
        dataset = VisualRasterInstructionDataset(
            instruction_records,
            seed=args.seed + 307,
            length=instruction_total * batch_size,
            **common,
        )
        loaders["instruction"] = _loader(
            dataset,
            consumed_examples=instruction_consumed * batch_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    if continuation_total > continuation_consumed:
        dataset = VisualRasterContinuationDataset(
            public_records,
            seed=args.seed + 401,
            length=continuation_total * batch_size,
            maximum_prompt_cells=32 if args.tiny_model else 160,
            **common,
        )
        loaders["continuation"] = _loader(
            dataset,
            consumed_examples=continuation_consumed * batch_size,
            batch_size=batch_size,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def _move_student_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    student = visual_semantic_raster_student_batch(batch)
    return {
        name: value.to(device, non_blocking=True) for name, value in student.items()
    }


def _data_receipt(
    public_records: list[Any],
    instruction_records: list[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "public_manifest": args.public_manifest,
        "public_sha256": file_sha256(args.public_manifest),
        "public_records": len(public_records),
        "public_partition": {
            split: sum(
                visual_raster_partition(record.identifier, stream="public-domain") == split
                for record in public_records
            )
            for split in ("train", "development", "sealed")
        },
        "instruction_manifest": args.instruction_manifest,
        "instruction_sha256": file_sha256(args.instruction_manifest),
        "instruction_records": len(instruction_records),
        "instruction_partition": {
            split: sum(
                visual_raster_partition(record.identifier, stream="instruction") == split
                for record in instruction_records
            )
            for split in ("train", "development", "sealed")
        },
        "student_boundary": visual_semantic_raster_data_boundary_receipt(),
    }


def _model_config(args: argparse.Namespace) -> VisualSemanticRasterConfig:
    if not args.tiny_model:
        return VisualSemanticRasterConfig()
    return VisualSemanticRasterConfig(
        maximum_prompt_patches=32,
        maximum_answer_cells=8,
        reader_hidden_size=96,
        reader_layers=3,
        reader_heads=4,
        reader_intermediate_size=192,
        reader_dropout=0.0,
        planner_dim=128,
        planner_layers=2,
        planner_heads=4,
        planner_mlp_dim=256,
        planner_dropout=0.0,
        cell_retina_channels=16,
        target_width=96,
        target_blocks=2,
        latent_dim=16,
        decoder_width=96,
        decoder_layers=2,
        decoder_heads=4,
        decoder_mlp_dim=192,
        decoder_dropout=0.0,
    )


def _source_receipt() -> dict[str, str]:
    return {path: file_sha256(path) for path in SOURCE_FILES if Path(path).is_file()}


def _capture_rng(feedback_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "feedback": feedback_generator.get_state(),
    }


def _restore_rng(state: Mapping[str, Any], feedback_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    feedback_generator.set_state(state["feedback"])


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: VisualSemanticRasterTransducer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: SelectiveExponentialMovingAverage,
    feedback_generator: torch.Generator,
    global_update: int,
    initialization: Mapping[str, Any],
    run_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": V32_ARCHITECTURE,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "global_update": global_update,
        "arguments": vars(args),
        "initialization": dict(initialization),
        "run_receipt": dict(run_receipt),
        "rng": _capture_rng(feedback_generator),
    }


def _stage_progress(global_update: int, stages: tuple[TrainingStage, ...]) -> list[tuple[TrainingStage, int]]:
    output = []
    prior = 0
    for stage in stages:
        completed = min(stage.updates, max(0, global_update - prior))
        output.append((stage, completed))
        prior += stage.updates
    return output


def main() -> None:
    raw_args = parse_args()
    require_preregistered_arguments(raw_args)
    args = effective_arguments(raw_args)
    stages = training_stages(args)
    if not args.smoke and not args.exploratory:
        if args.batch_size * args.gradient_accumulation < 64:
            raise ValueError("V32 evidence requires effective batch size >=64")
    if file_sha256(PROTOCOL_DOCUMENT) != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V32 protocol changed after preregistration")
    if not args.exploratory:
        if file_sha256(args.public_manifest) != EXPECTED_PUBLIC_SHA256:
            raise ValueError("V32 public-domain manifest differs from the protocol")
        if file_sha256(args.instruction_manifest) != EXPECTED_INSTRUCTION_SHA256:
            raise ValueError("V32 instruction manifest differs from the protocol")

    device = choose_device(args.device)
    if not args.smoke and not args.exploratory and device.type != "cuda":
        raise ValueError("V32 evidence requires CUDA")
    seed_everything(args.seed)
    cuda_index = None
    if device.type == "cuda":
        cuda_index = 0 if device.index is None else device.index
        torch.cuda.set_device(cuda_index)
        torch.cuda.reset_peak_memory_stats(cuda_index)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "training_metrics.jsonl"

    config = _model_config(args)
    render_config = VisualRasterRenderConfig(
        maximum_prompt_patches=config.maximum_prompt_patches,
        maximum_answer_cells=config.maximum_answer_cells,
    )
    public_records = load_visual_text_records(args.public_manifest)
    instruction_records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=160 if not args.tiny_model else 32,
        maximum_answer_cells=config.maximum_answer_cells,
    )
    data_receipt = _data_receipt(public_records, instruction_records, args)

    model = VisualSemanticRasterTransducer(config)
    if args.tiny_model:
        initialization: dict[str, Any] = {
            "route": "random-tiny-smoke",
            "evidence_eligible": False,
        }
    else:
        reader_checkpoint = resolve_pixel_m4_checkpoint(
            args.reader_checkpoint,
            local_files_only=args.local_files_only,
        )
        initialization = {
            "route": "pinned-pixel-m4-reader",
            "evidence_eligible": True,
            **load_pixel_m4_reader(model, reader_checkpoint),
        }
    model.freeze_reader()
    model.to(device)
    groups = visual_semantic_raster_optimizer_groups(
        model,
        writer_learning_rate=args.writer_learning_rate,
        reader_learning_rate=args.reader_learning_rate,
        weight_decay=args.weight_decay,
        reader_final_blocks=args.reader_final_blocks,
    )
    optimizer_receipt = visual_semantic_raster_optimizer_receipt(model, groups)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    ema = SelectiveExponentialMovingAverage(
        model,
        optimizer_receipt["optimized_parameter_names"],
        decay=args.ema_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and args.precision == "fp16",
    )
    feedback_generator = torch.Generator(device=device)
    feedback_generator.manual_seed(args.seed + 503)
    global_update = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != V32_ARCHITECTURE:
            raise ValueError("V32 resume checkpoint has the wrong architecture")
        if checkpoint.get("model_config") != asdict(config):
            raise ValueError("V32 resume checkpoint has a different model configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        ema.load_state_dict(checkpoint["ema"])
        global_update = int(checkpoint["global_update"])
        _restore_rng(checkpoint["rng"], feedback_generator)

    run_receipt = {
        "architecture": V32_ARCHITECTURE,
        "protocol": PROTOCOL_DOCUMENT,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "arguments": vars(args),
        "stages": [asdict(stage) for stage in stages],
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None,
        "model_boundary": visual_semantic_raster_boundary_receipt(model),
        "optimizer": optimizer_receipt,
        "initialization": initialization,
        "data": data_receipt,
        "sources": _source_receipt(),
        "non_commercial_research_checkpoint": True,
    }
    atomic_write_json(run_receipt, output_dir / "run_receipt.json")
    print(json.dumps(run_receipt, ensure_ascii=False), flush=True)

    stop_requested = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested[0] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    total_updates = sum(stage.updates for stage in stages)

    for stage, completed_updates in _stage_progress(global_update, stages):
        if completed_updates >= stage.updates:
            continue
        if stage.name != "instruction" or completed_updates < args.reader_unfreeze_after:
            model.freeze_reader()
        else:
            model.unfreeze_reader_final_blocks(args.reader_final_blocks)
        model.train()
        loaders = build_stage_loaders(
            stage,
            completed_updates=completed_updates,
            args=args,
            render_config=render_config,
            public_records=public_records,
            instruction_records=instruction_records,
            pin_memory=device.type == "cuda",
        )
        for stage_update in range(completed_updates, stage.updates):
            if (
                stage.name == "instruction"
                and stage_update == args.reader_unfreeze_after
            ):
                model.unfreeze_reader_final_blocks(args.reader_final_blocks)
                model.train()
            writer_lr = stage_cosine_learning_rate(
                stage_update + 1,
                peak=args.writer_learning_rate,
                warmup=args.stage_warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            reader_lr = stage_cosine_learning_rate(
                stage_update + 1,
                peak=args.reader_learning_rate,
                warmup=args.stage_warmup,
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            set_visual_semantic_raster_learning_rates(
                optimizer,
                writer=writer_lr,
                reader=reader_lr,
            )
            optimizer.zero_grad(set_to_none=True)
            aggregate: dict[str, float] = {}
            for accumulation_index in range(args.gradient_accumulation):
                microbatch = stage_update * args.gradient_accumulation + accumulation_index
                stream = stage_stream(stage.name, microbatch)
                batch = _move_student_batch(next(loaders[stream]), device)
                with autocast_context(device, args.precision):
                    if stage.name == "raster-warmup":
                        loss, metrics = raster_warmup_microstep(
                            model,
                            batch,
                            generator=feedback_generator,
                        )
                    else:
                        loss, metrics = visual_semantic_raster_training_microstep(
                            model,
                            batch,
                            feedback_mode="decoded",
                            generator=feedback_generator,
                        )
                    scaled_loss = loss / args.gradient_accumulation
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("V32 training loss became non-finite")
                scaler.scale(scaled_loss).backward()
                for name, value in metrics.items():
                    aggregate[name] = aggregate.get(name, 0.0) + (
                        float(value) / args.gradient_accumulation
                    )
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V32 gradient norm became non-finite")
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            global_update += 1

            row: dict[str, Any] = {
                "stage": stage.name,
                "stage_update": stage_update + 1,
                "global_update": global_update,
                "total_updates": total_updates,
                "writer_learning_rate": writer_lr,
                "reader_learning_rate": reader_lr,
                "reader_trainable_parameters": model.reader_trainable_parameter_count(),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
                **aggregate,
            }
            if device.type == "cuda":
                row["peak_vram_bytes"] = torch.cuda.max_memory_allocated(cuda_index)
            if global_update == 1 or global_update % args.log_every == 0:
                print(json.dumps(row), flush=True)
            append_jsonl(metrics_path, row)

            should_save = (
                global_update % args.save_every == 0
                or stage_update + 1 == stage.updates
                or stop_requested[0]
            )
            if should_save:
                payload = checkpoint_payload(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    ema=ema,
                    feedback_generator=feedback_generator,
                    global_update=global_update,
                    initialization=initialization,
                    run_receipt=run_receipt,
                )
                atomic_torch_save(payload, output_dir / "checkpoint_latest.pt")
            if stop_requested[0]:
                break
        if stop_requested[0]:
            break

    final = {
        "architecture": V32_ARCHITECTURE,
        "global_update": global_update,
        "planned_updates": total_updates,
        "complete": global_update == total_updates,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(cuda_index)
            if cuda_index is not None
            else 0
        ),
        "stopped_by_signal": stop_requested[0],
    }
    atomic_write_json(final, output_dir / "training_summary.json")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
