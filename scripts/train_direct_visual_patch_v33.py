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
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchInstructionDataset,
    DirectPatchRenderConfig,
    direct_patch_collate,
    direct_patch_data_boundary_receipt,
    direct_patch_student_batch,
)
from ilm.visual_lm.direct_visual_patch_evaluation import evaluate_visual_calibration
from ilm.visual_lm.direct_visual_patch_lm import (
    V33_ARCHITECTURE,
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
    direct_visual_patch_boundary_receipt,
    file_sha256,
    load_pixar_initialization,
)
from ilm.visual_lm.direct_visual_patch_training import (
    ExponentialMovingAverage,
    direct_visual_patch_loss,
    direct_visual_patch_optimizer_groups,
    module_state_sha256,
    optimizer_receipt,
    set_core_trainable,
    set_optimizer_learning_rates,
    stage_cosine_learning_rate,
)
from ilm.visual_lm.visual_semantic_raster_data import (
    load_visual_raster_instructions,
    load_visual_text_records,
)


PROTOCOL_DOCUMENT = "references/direct_visual_patch_lm_v33_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "45d40ba96276aa43d2b796a3d5e0c39a225b650644c349d7315802b19784b97c"
)
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_INSTRUCTION_MANIFEST = "data/raw/alpaca_zh.json"
DEFAULT_PIXAR_CHECKPOINT = "artifacts/upstream/pixar"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
EXPECTED_INSTRUCTION_SHA256 = (
    "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
)
SOURCE_FILES = (
    "ilm/visual_lm/direct_visual_patch_data.py",
    "ilm/visual_lm/direct_visual_patch_lm.py",
    "ilm/visual_lm/direct_visual_patch_training.py",
    "ilm/visual_lm/direct_visual_patch_evaluation.py",
    "scripts/train_direct_visual_patch_v33.py",
)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    updates: int
    adapter_learning_rate: float
    core_learning_rate: float


FIXED_STAGES = (
    TrainingStage("visual-calibration", 2_000, 3e-4, 0.0),
    TrainingStage("public-continuation", 8_000, 1e-4, 1e-5),
    TrainingStage("instruction-continuation", 12_000, 8e-5, 8e-6),
)
FIXED_OPTIMIZATION: dict[str, Any] = {
    "batch_size": 8,
    "gradient_accumulation": 8,
    "stage_warmup": 500,
    "minimum_learning_rate_ratio": 0.10,
    "weight_decay": 0.05,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": 20_263_300,
    "precision": "bf16",
    "gate_minimum_patches": 2_048,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the preregistered V33 direct visual patch language model."
    )
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--instruction-manifest", default=DEFAULT_INSTRUCTION_MANIFEST)
    parser.add_argument("--pixar-checkpoint", default=DEFAULT_PIXAR_CHECKPOINT)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--calibration-updates", type=int, default=2_000)
    parser.add_argument("--continuation-updates", type=int, default=8_000)
    parser.add_argument("--instruction-updates", type=int, default=12_000)
    parser.add_argument("--calibration-adapter-lr", type=float, default=3e-4)
    parser.add_argument("--continuation-adapter-lr", type=float, default=1e-4)
    parser.add_argument("--continuation-core-lr", type=float, default=1e-5)
    parser.add_argument("--instruction-adapter-lr", type=float, default=8e-5)
    parser.add_argument("--instruction-core-lr", type=float, default=8e-6)
    parser.add_argument("--stage-warmup", type=int, default=500)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20_263_300)
    parser.add_argument("--gate-minimum-patches", type=int, default=2_048)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--allow-failed-calibration", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    return parser.parse_args()


def effective_arguments(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    label = "smoke" if args.smoke else "exploratory" if args.exploratory else "evidence"
    values["out"] = args.out or f"artifacts/direct_visual_patch_v33_{label}"
    if args.smoke:
        values.update(
            {
                "num_workers": 0,
                "batch_size": 1,
                "gradient_accumulation": 1,
                "calibration_updates": 1,
                "continuation_updates": 1,
                "instruction_updates": 1,
                "stage_warmup": 0,
                "gate_minimum_patches": 8,
                "log_every": 1,
                "save_every": 1,
                "allow_failed_calibration": True,
            }
        )
    return argparse.Namespace(**values)


def training_stages(args: argparse.Namespace) -> tuple[TrainingStage, ...]:
    stages = (
        TrainingStage(
            "visual-calibration",
            args.calibration_updates,
            args.calibration_adapter_lr,
            0.0,
        ),
        TrainingStage(
            "public-continuation",
            args.continuation_updates,
            args.continuation_adapter_lr,
            args.continuation_core_lr,
        ),
        TrainingStage(
            "instruction-continuation",
            args.instruction_updates,
            args.instruction_adapter_lr,
            args.instruction_core_lr,
        ),
    )
    if any(stage.updates < 0 for stage in stages):
        raise ValueError("V33 stage updates cannot be negative")
    if stages[0].updates < 1:
        raise ValueError("V33 requires visual calibration")
    return tuple(stage for stage in stages if stage.updates)


def require_preregistered_arguments(args: argparse.Namespace) -> None:
    if args.smoke or args.exploratory:
        return
    if args.random_init or args.allow_failed_calibration or args.calibration_only:
        raise ValueError("V33 evidence requires the selected complete route")
    expected = {
        **FIXED_OPTIMIZATION,
        "calibration_updates": FIXED_STAGES[0].updates,
        "continuation_updates": FIXED_STAGES[1].updates,
        "instruction_updates": FIXED_STAGES[2].updates,
        "calibration_adapter_lr": FIXED_STAGES[0].adapter_learning_rate,
        "continuation_adapter_lr": FIXED_STAGES[1].adapter_learning_rate,
        "continuation_core_lr": FIXED_STAGES[1].core_learning_rate,
        "instruction_adapter_lr": FIXED_STAGES[2].adapter_learning_rate,
        "instruction_core_lr": FIXED_STAGES[2].core_learning_rate,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"V33 evidence requires --{name.replace('_', '-')}={value}")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("V33 requested CUDA but CUDA is unavailable")
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
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
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


def _trim_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    visible = int(batch["patch_mask"].sum(dim=1).max().item())
    result = dict(batch)
    result["pixels"] = batch["pixels"][..., : visible * 32]
    for key in (
        "patch_mask",
        "next_patch_mask",
        "reconstruction_mask",
        "stop_targets",
        "stop_mask",
    ):
        result[key] = batch[key][:, :visible]
    return result


def _loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Iterator[dict[str, Any]]:
    return iter(
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            collate_fn=direct_patch_collate,
        )
    )


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def main() -> None:
    args = effective_arguments(parse_args())
    require_preregistered_arguments(args)
    stages = training_stages(args)
    device = choose_device(args.device)
    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "training_metrics.jsonl"
    protocol_hash = file_sha256(PROTOCOL_DOCUMENT)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V33 protocol changed after preregistration")
    data_hashes = {
        "public": file_sha256(args.public_manifest),
        "instruction": file_sha256(args.instruction_manifest),
    }
    if data_hashes != {
        "public": EXPECTED_PUBLIC_SHA256,
        "instruction": EXPECTED_INSTRUCTION_SHA256,
    }:
        raise RuntimeError("V33 input manifests differ from preregistration")

    render_config = DirectPatchRenderConfig()
    public_records = load_visual_text_records(args.public_manifest)
    instruction_records = load_visual_raster_instructions(
        args.instruction_manifest,
        maximum_prompt_characters=64,
        maximum_answer_cells=32,
    )
    calibration_dataset = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=render_config,
        variants_per_record=max(1, args.calibration_updates * args.batch_size * args.gradient_accumulation // len(public_records) + 2),
        seed=args.seed,
    )
    continuation_dataset = DirectPatchContinuationDataset(
        public_records,
        split="train",
        config=render_config,
        variants_per_record=max(1, args.continuation_updates * args.batch_size * args.gradient_accumulation // len(public_records) + 2),
        seed=args.seed + 1_000_000,
    )
    instruction_dataset = DirectPatchInstructionDataset(
        instruction_records,
        split="train",
        config=render_config,
        variants_per_record=max(1, args.instruction_updates * args.batch_size * args.gradient_accumulation // len(instruction_records) + 2),
        seed=args.seed + 2_000_000,
    )
    development_calibration = DirectPatchContinuationDataset(
        public_records,
        split="development",
        config=render_config,
        variants_per_record=2,
        seed=args.seed + 3_000_000,
    )

    model = DirectVisualPatchLM(DirectVisualPatchConfig())
    initialization = (
        {"route": "random", "seed": args.seed}
        if args.random_init
        else load_pixar_initialization(model, args.pixar_checkpoint)
    )
    initial_core_hash = module_state_sha256(model.backbone)
    set_core_trainable(model, False)
    model.to(device)
    groups = direct_visual_patch_optimizer_groups(
        model,
        adapter_learning_rate=args.calibration_adapter_lr,
        core_learning_rate=0.0,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    sample_batch = direct_patch_collate([calibration_dataset[0]])
    receipt = {
        "architecture": V33_ARCHITECTURE,
        "arguments": vars(args),
        "protocol": {"path": PROTOCOL_DOCUMENT, "sha256": protocol_hash},
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "data_sha256": data_hashes,
        "data_counts": {
            "public_records": len(public_records),
            "instruction_records": len(instruction_records),
            "calibration_examples": len(calibration_dataset),
            "continuation_examples": len(continuation_dataset),
            "instruction_examples": len(instruction_dataset),
        },
        "initialization": initialization,
        "initial_core_sha256": initial_core_hash,
        "model_boundary": direct_visual_patch_boundary_receipt(model),
        "data_boundary": direct_patch_data_boundary_receipt(sample_batch),
        "optimizer": optimizer_receipt(model, groups),
        "device": str(device),
        "torch": torch.__version__,
    }
    atomic_write_json(receipt, out / "run_receipt.json")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"V33 received signal {signum}; stopping after current update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    datasets = {
        "visual-calibration": calibration_dataset,
        "public-continuation": continuation_dataset,
        "instruction-continuation": instruction_dataset,
    }
    global_update = 0
    started = time.monotonic()
    stage_summaries: list[dict[str, Any]] = []
    calibration_report: dict[str, Any] | None = None

    for stage in stages:
        if args.calibration_only and stage.name != "visual-calibration":
            break
        set_core_trainable(model, stage.name != "visual-calibration")
        loader = _loader(
            datasets[stage.name],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        mode = "calibration" if stage.name == "visual-calibration" else "causal"
        stage_started = time.monotonic()
        for stage_update in range(1, stage.updates + 1):
            adapter_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.adapter_learning_rate,
                warmup=min(args.stage_warmup, max(0, stage.updates - 1)),
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            core_lr = stage_cosine_learning_rate(
                stage_update,
                peak=stage.core_learning_rate,
                warmup=min(args.stage_warmup, max(0, stage.updates - 1)),
                total=stage.updates,
                minimum_ratio=args.minimum_learning_rate_ratio,
            )
            set_optimizer_learning_rates(optimizer, adapter=adapter_lr, core=core_lr)
            optimizer.zero_grad(set_to_none=True)
            accumulated: dict[str, float] = {}
            for _ in range(args.gradient_accumulation):
                raw_batch = next(loader)
                student = {
                    key: value.to(device, non_blocking=True)
                    for key, value in direct_patch_student_batch(raw_batch).items()
                }
                student = _trim_batch(student)
                with autocast_context(device, args.precision):
                    output = model(student["pixels"], student["patch_mask"])
                    losses = direct_visual_patch_loss(output, student, mode=mode)
                    scaled_loss = losses.loss / args.gradient_accumulation
                if not bool(torch.isfinite(scaled_loss)):
                    raise FloatingPointError("V33 encountered a non-finite loss")
                scaled_loss.backward()
                for key, value in losses.detached_metrics().items():
                    accumulated[key] = accumulated.get(key, 0.0) + value / args.gradient_accumulation
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            ema.update(model)
            global_update += 1
            row = {
                "global_update": global_update,
                "stage": stage.name,
                "stage_update": stage_update,
                "adapter_lr": adapter_lr,
                "core_lr": core_lr,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started,
                **accumulated,
            }
            append_jsonl(metrics_path, row)
            if stage_update == 1 or stage_update % args.log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if stop_requested:
                break

        stage_summaries.append(
            {
                "name": stage.name,
                "updates_completed": stage_update,
                "updates_planned": stage.updates,
                "elapsed_seconds": time.monotonic() - stage_started,
            }
        )
        checkpoint = {
            "architecture": V33_ARCHITECTURE,
            "model_config": asdict(model.config),
            "model": _cpu_model_state(model),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_update": global_update,
            "stage": stage.name,
            "stage_update": stage_update,
            "initialization": initialization,
            "receipt": receipt,
        }
        atomic_torch_save(checkpoint, out / "checkpoint_latest.pt")
        if stage.name == "visual-calibration":
            final_core_hash = module_state_sha256(model.backbone)
            calibration_report = evaluate_visual_calibration(
                model,
                development_calibration,
                device=device,
                precision=args.precision,
                minimum_patches=args.gate_minimum_patches,
                batch_size=args.batch_size,
                num_workers=0,
                gallery_path=out / "calibration_gallery.png",
            )
            calibration_report["core_sha256_before"] = initial_core_hash
            calibration_report["core_sha256_after"] = final_core_hash
            calibration_report["core_unchanged"] = final_core_hash == initial_core_hash
            calibration_report["gates"]["core_unchanged"] = calibration_report[
                "core_unchanged"
            ]
            calibration_report["pass"] = all(calibration_report["gates"].values())
            atomic_write_json(calibration_report, out / "calibration_report.json")
            print(json.dumps(calibration_report, ensure_ascii=False, indent=2), flush=True)
            if not calibration_report["pass"] and not args.allow_failed_calibration:
                stop_requested = True
        del checkpoint
        if stop_requested:
            break

    summary = {
        "architecture": V33_ARCHITECTURE,
        "global_update": global_update,
        "planned_updates": sum(stage.updates for stage in stages),
        "complete": not stop_requested and global_update == sum(stage.updates for stage in stages),
        "elapsed_seconds": time.monotonic() - started,
        "stages": stage_summaries,
        "calibration": calibration_report,
        "checkpoint": str(out / "checkpoint_latest.pt"),
        "checkpoint_sha256": file_sha256(out / "checkpoint_latest.pt"),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "stopped_by_signal_or_gate": stop_requested,
    }
    atomic_write_json(summary, out / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

