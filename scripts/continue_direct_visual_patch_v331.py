#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchRenderConfig,
    direct_patch_collate,
    direct_patch_data_boundary_receipt,
    direct_patch_student_batch,
)
from ilm.visual_lm.direct_visual_patch_evaluation import (
    evaluate_visual_calibration_v331,
)
from ilm.visual_lm.direct_visual_patch_lm import (
    V33_ARCHITECTURE,
    DirectVisualPatchConfig,
    DirectVisualPatchLM,
    direct_visual_patch_boundary_receipt,
    file_sha256,
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
from ilm.visual_lm.visual_semantic_raster_data import load_visual_text_records


EXPERIMENT = "direct-visual-patch-lm-v33.1-calibration"
PROTOCOL_DOCUMENT = "references/direct_visual_patch_lm_v331_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "9ed5873293d4249cb120204eb7b13074371fbdd0785b13e94ed309512a0767ab"
)
DEFAULT_SOURCE_CHECKPOINT = (
    "artifacts/direct_visual_patch_v33_calibration_20260813/checkpoint_latest.pt"
)
EXPECTED_SOURCE_SHA256 = (
    "cacb0215634a23bd7801fd7544c4b3a59e68274d41e39f2d44a68fb909a39696"
)
EXPECTED_SOURCE_CORE_SHA256 = (
    "2f420abd2d75950278d2104e18f739bec2657f7241f5f7ee80729febe8d88293"
)
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
SOURCE_GLOBAL_UPDATE = 2_000
SOURCE_EXAMPLES_CONSUMED = 128_000
ADDITIONAL_UPDATES = 6_000
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 8
ADAPTER_LEARNING_RATE = 3e-5
MINIMUM_LEARNING_RATE_RATIO = 0.10
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0
SEED = 20_263_300
EMA_DECAY = 0.999
GATE_MINIMUM_PATCHES = 2_048
CHECKPOINT_INTERVAL = 1_000
SOURCE_FILES = (
    "ilm/visual_lm/direct_visual_patch_data.py",
    "ilm/visual_lm/direct_visual_patch_lm.py",
    "ilm/visual_lm/direct_visual_patch_training.py",
    "ilm/visual_lm/direct_visual_patch_evaluation.py",
    "scripts/continue_direct_visual_patch_v331.py",
)


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], *, start: int, count: int) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V33.1 dataset window is outside the rendered stream")
        self.dataset = dataset
        self.start = start
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.dataset[self.start + index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue the frozen V33.1 direct-raster calibration experiment."
    )
    parser.add_argument("--source-checkpoint", default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument(
        "--out",
        default="artifacts/direct_visual_patch_v331_calibration_20260813",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str, *, smoke: bool) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        if smoke and device.type == "cpu":
            return device
        raise RuntimeError("V33.1 evidence requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if not smoke and "4090" not in name:
        raise RuntimeError(f"V33.1 evidence requires an RTX 4090, found {name!r}")
    if not smoke and not torch.cuda.is_bf16_supported():
        raise RuntimeError("V33.1 evidence requires CUDA BF16 support")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


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


def cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def trim_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def build_training_window(
    records: list[Any],
    *,
    completed_updates: int,
    planned_updates: int,
    batch_size: int,
    gradient_accumulation: int,
) -> tuple[DatasetWindow, dict[str, int]]:
    probe = DirectPatchContinuationDataset(
        records,
        split="train",
        config=DirectPatchRenderConfig(),
        variants_per_record=1,
        seed=SEED,
    )
    examples_per_update = batch_size * gradient_accumulation
    stream_end = SOURCE_EXAMPLES_CONSUMED + planned_updates * examples_per_update
    variants = math.ceil(stream_end / len(probe))
    stream = DirectPatchContinuationDataset(
        records,
        split="train",
        config=DirectPatchRenderConfig(),
        variants_per_record=variants,
        seed=SEED,
    )
    start = SOURCE_EXAMPLES_CONSUMED + completed_updates * examples_per_update
    remaining = (planned_updates - completed_updates) * examples_per_update
    window = DatasetWindow(stream, start=start, count=remaining)
    return window, {
        "training_records": len(probe),
        "variants_per_record": variants,
        "source_examples_consumed": SOURCE_EXAMPLES_CONSUMED,
        "window_start": start,
        "window_count": remaining,
        "stream_end": stream_end,
    }


def make_loader(
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


def validate_source_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("architecture") != V33_ARCHITECTURE:
        raise ValueError("V33.1 source architecture does not match V33")
    if checkpoint.get("stage") != "visual-calibration":
        raise ValueError("V33.1 source is not a visual-calibration checkpoint")
    if int(checkpoint.get("global_update", -1)) != SOURCE_GLOBAL_UPDATE:
        raise ValueError("V33.1 source checkpoint is not at global update 2000")
    if int(checkpoint.get("stage_update", -1)) != SOURCE_GLOBAL_UPDATE:
        raise ValueError("V33.1 source checkpoint is not at stage update 2000")


def validate_resume_checkpoint(checkpoint: Mapping[str, Any]) -> int:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError("V33.1 resume checkpoint has the wrong experiment marker")
    protocol = checkpoint.get("protocol", {})
    if protocol.get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V33.1 resume checkpoint has the wrong protocol hash")
    if checkpoint.get("source_checkpoint_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("V33.1 resume checkpoint has the wrong source hash")
    completed = int(checkpoint.get("extension_update", -1))
    if not 0 <= completed < ADDITIONAL_UPDATES:
        raise ValueError("V33.1 resume update is outside the extension")
    expected_examples = SOURCE_EXAMPLES_CONSUMED + (
        completed * BATCH_SIZE * GRADIENT_ACCUMULATION
    )
    if int(checkpoint.get("data_examples_consumed", -1)) != expected_examples:
        raise ValueError("V33.1 resume data cursor is inconsistent")
    return completed


def main() -> None:
    args = parse_args()
    if args.num_workers < 0 or args.log_every < 1:
        raise ValueError("V33.1 worker and logging settings must be non-negative")
    device = choose_device(args.device, smoke=args.smoke)
    seed_everything(SEED)
    protocol_hash = file_sha256(PROTOCOL_DOCUMENT)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V33.1 protocol changed after preregistration")
    if file_sha256(args.public_manifest) != EXPECTED_PUBLIC_SHA256:
        raise RuntimeError("V33.1 public manifest differs from preregistration")

    out = Path(args.out)
    checkpoint_path = out / "checkpoint_latest.pt"
    metrics_path = out / "training_metrics.jsonl"
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("V33.1 resume checkpoint does not exist")
        load_path = checkpoint_path
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("V33.1 output directory is nonempty; use --resume")
        out.mkdir(parents=True, exist_ok=True)
        load_path = Path(args.source_checkpoint)
        if file_sha256(load_path) != EXPECTED_SOURCE_SHA256:
            raise RuntimeError("V33.1 source checkpoint hash differs from protocol")

    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("V33.1 checkpoint must be a mapping")
    if args.resume:
        extension_completed = validate_resume_checkpoint(checkpoint)
    else:
        validate_source_checkpoint(checkpoint)
        extension_completed = 0

    planned_updates = 1 if args.smoke else ADDITIONAL_UPDATES
    batch_size = 1 if args.smoke else BATCH_SIZE
    gradient_accumulation = 1 if args.smoke else GRADIENT_ACCUMULATION
    gate_minimum_patches = 8 if args.smoke else GATE_MINIMUM_PATCHES
    checkpoint_interval = 1 if args.smoke else CHECKPOINT_INTERVAL
    if extension_completed >= planned_updates:
        raise ValueError("V33.1 checkpoint already completed this run")

    model = DirectVisualPatchLM(
        DirectVisualPatchConfig(**dict(checkpoint["model_config"]))
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    source_core_hash = module_state_sha256(model.backbone)
    if source_core_hash != EXPECTED_SOURCE_CORE_SHA256:
        raise RuntimeError("V33.1 transformer core differs from the frozen source")
    set_core_trainable(model, False)
    model.to(device)
    groups = direct_visual_patch_optimizer_groups(
        model,
        adapter_learning_rate=ADAPTER_LEARNING_RATE,
        core_learning_rate=0.0,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer = torch.optim.AdamW(
        groups,
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema = ExponentialMovingAverage(model, decay=EMA_DECAY)
    ema.load_state_dict(checkpoint["ema"])
    del checkpoint

    records = load_visual_text_records(args.public_manifest)
    training_window, data_cursor = build_training_window(
        records,
        completed_updates=extension_completed,
        planned_updates=planned_updates,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
    )
    development_dataset = DirectPatchContinuationDataset(
        records,
        split="development",
        config=DirectPatchRenderConfig(),
        variants_per_record=2,
        seed=SEED + 3_000_000,
    )
    loader = make_loader(
        training_window,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}
    if args.resume:
        receipt = json.loads((out / "run_receipt.json").read_text(encoding="utf-8"))
        if receipt["source_sha256"] != source_hashes:
            raise RuntimeError("V33.1 source changed since the interrupted run")
    else:
        sample_batch = direct_patch_collate([training_window[0]])
        receipt = {
            "experiment": EXPERIMENT,
            "label": "smoke" if args.smoke else "evidence",
            "protocol": {"path": PROTOCOL_DOCUMENT, "sha256": protocol_hash},
            "source_checkpoint": str(Path(args.source_checkpoint)),
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "source_core_sha256": source_core_hash,
            "public_manifest": args.public_manifest,
            "public_manifest_sha256": EXPECTED_PUBLIC_SHA256,
            "source_sha256": source_hashes,
            "data_cursor": data_cursor,
            "model_boundary": direct_visual_patch_boundary_receipt(model),
            "data_boundary": direct_patch_data_boundary_receipt(sample_batch),
            "optimizer": optimizer_receipt(model, groups),
            "fixed_optimization": {
                "source_global_update": SOURCE_GLOBAL_UPDATE,
                "additional_updates": planned_updates,
                "batch_size": batch_size,
                "gradient_accumulation": gradient_accumulation,
                "effective_batch_size": batch_size * gradient_accumulation,
                "adapter_learning_rate": ADAPTER_LEARNING_RATE,
                "minimum_learning_rate_ratio": MINIMUM_LEARNING_RATE_RATIO,
                "core_learning_rate": 0.0,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip": GRADIENT_CLIP,
                "ema_decay": EMA_DECAY,
                "seed": SEED,
                "precision": "bf16" if device.type == "cuda" else "fp32",
            },
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "torch": torch.__version__,
        }
        atomic_write_json(receipt, out / "run_receipt.json")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"V33.1 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.monotonic()
    current_update = extension_completed

    def save_checkpoint() -> None:
        payload = {
            "architecture": V33_ARCHITECTURE,
            "experiment": EXPERIMENT,
            "protocol": {"path": PROTOCOL_DOCUMENT, "sha256": protocol_hash},
            "source_checkpoint_sha256": EXPECTED_SOURCE_SHA256,
            "model_config": asdict(model.config),
            "model": cpu_model_state(model),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_update": SOURCE_GLOBAL_UPDATE + current_update,
            "stage": "visual-calibration-v33.1",
            "extension_update": current_update,
            "data_examples_consumed": SOURCE_EXAMPLES_CONSUMED
            + current_update * batch_size * gradient_accumulation,
            "receipt": receipt,
        }
        atomic_torch_save(payload, checkpoint_path)

    for extension_update in range(extension_completed + 1, planned_updates + 1):
        learning_rate = stage_cosine_learning_rate(
            extension_update,
            peak=ADAPTER_LEARNING_RATE,
            warmup=0,
            total=planned_updates,
            minimum_ratio=MINIMUM_LEARNING_RATE_RATIO,
        )
        set_optimizer_learning_rates(optimizer, adapter=learning_rate, core=0.0)
        optimizer.zero_grad(set_to_none=True)
        accumulated: dict[str, float] = {}
        for _ in range(gradient_accumulation):
            raw_batch = next(loader)
            student = {
                key: value.to(device, non_blocking=True)
                for key, value in direct_patch_student_batch(raw_batch).items()
            }
            student = trim_batch(student)
            with autocast_context(device):
                output = model(student["pixels"], student["patch_mask"])
                losses = direct_visual_patch_loss(output, student, mode="calibration")
                scaled_loss = losses.loss / gradient_accumulation
            if not bool(torch.isfinite(scaled_loss)):
                raise FloatingPointError("V33.1 encountered a non-finite loss")
            scaled_loss.backward()
            for key, value in losses.detached_metrics().items():
                accumulated[key] = (
                    accumulated.get(key, 0.0) + value / gradient_accumulation
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        ema.update(model)
        current_update = extension_update
        row = {
            "extension_update": extension_update,
            "global_update": SOURCE_GLOBAL_UPDATE + extension_update,
            "adapter_lr": learning_rate,
            "core_lr": 0.0,
            "gradient_norm": float(gradient_norm),
            "data_examples_consumed": SOURCE_EXAMPLES_CONSUMED
            + extension_update * batch_size * gradient_accumulation,
            "elapsed_seconds": time.monotonic() - started,
            **accumulated,
        }
        append_jsonl(metrics_path, row)
        if extension_update == 1 or extension_update % args.log_every == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if extension_update % checkpoint_interval == 0 or stop_requested:
            save_checkpoint()
        if stop_requested:
            break

    updates_complete = current_update == planned_updates and not stop_requested
    if not checkpoint_path.is_file() or current_update % checkpoint_interval:
        save_checkpoint()

    raw_report: dict[str, Any] | None = None
    ema_report: dict[str, Any] | None = None
    final_core_hash = module_state_sha256(model.backbone)
    core_unchanged = final_core_hash == EXPECTED_SOURCE_CORE_SHA256
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    if updates_complete:
        raw_report = evaluate_visual_calibration_v331(
            model,
            development_dataset,
            device=device,
            precision="bf16" if device.type == "cuda" else "fp32",
            minimum_patches=gate_minimum_patches,
            batch_size=batch_size,
            num_workers=0,
            gallery_path=out / "calibration_gallery_raw.png",
        )
        raw_report["core_unchanged"] = core_unchanged
        raw_report["updates_complete"] = updates_complete
        raw_report["peak_vram_bytes"] = peak_vram
        raw_report["gates"].update(
            {
                "core_unchanged": core_unchanged,
                "updates_complete": updates_complete,
                "peak_vram_below_20_gib": peak_vram < 20 * 1024**3,
            }
        )
        raw_report["pass"] = all(raw_report["gates"].values())

        raw_state = cpu_model_state(model)
        ema.copy_to(model)
        ema_report = evaluate_visual_calibration_v331(
            model,
            development_dataset,
            device=device,
            precision="bf16" if device.type == "cuda" else "fp32",
            minimum_patches=gate_minimum_patches,
            batch_size=batch_size,
            num_workers=0,
            gallery_path=out / "calibration_gallery_ema.png",
        )
        model.load_state_dict(raw_state, strict=True)
        del raw_state
        atomic_write_json(raw_report, out / "calibration_report_raw.json")
        atomic_write_json(ema_report, out / "calibration_report_ema.json")
        print(json.dumps(raw_report, ensure_ascii=False, indent=2), flush=True)

    if not updates_complete:
        decision = "invalid-run"
    elif raw_report is not None and raw_report["pass"]:
        decision = "raster-interface-qualified"
    elif raw_report is not None and min(
        raw_report["ink_pixel_f1"], raw_report["edge_f1"]
    ) >= 0.80:
        decision = "readable-below-gate"
    else:
        decision = "raster-interface-failure"
    summary = {
        "experiment": EXPERIMENT,
        "label": "smoke" if args.smoke else "evidence",
        "complete": updates_complete,
        "decision": decision,
        "source_global_update": SOURCE_GLOBAL_UPDATE,
        "extension_updates_completed": current_update,
        "extension_updates_planned": planned_updates,
        "global_update": SOURCE_GLOBAL_UPDATE + current_update,
        "data_examples_consumed": SOURCE_EXAMPLES_CONSUMED
        + current_update * batch_size * gradient_accumulation,
        "elapsed_seconds": time.monotonic() - started,
        "raw_calibration": raw_report,
        "ema_calibration": ema_report,
        "source_core_sha256": EXPECTED_SOURCE_CORE_SHA256,
        "final_core_sha256": final_core_hash,
        "core_unchanged": core_unchanged,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "peak_vram_bytes": peak_vram,
        "stopped_by_signal": stop_requested,
    }
    atomic_write_json(summary, out / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
