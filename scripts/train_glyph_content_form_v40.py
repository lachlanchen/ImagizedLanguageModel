#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm.continuous_glyph_codec_data import (
    file_sha256,
    load_historic_glyph_records,
)
from ilm.visual_lm.glyph_content_form import (
    V40_ARCHITECTURE,
    GlyphContentFormConfig,
    GlyphContentFormModel,
    glyph_content_form_boundary_receipt,
    load_v40_v34_codec,
)
from ilm.visual_lm.glyph_content_form_data import (
    CrossEraContentFormDataset,
    glyph_content_form_collate,
    glyph_content_form_data_boundary_receipt,
    glyph_content_form_stage_ids,
    glyph_content_form_student_batch,
)
from ilm.visual_lm.glyph_content_form_evaluation import (
    evaluate_glyph_content_form,
)
from ilm.visual_lm.glyph_content_form_training import (
    WarmStartTrainableEMA,
    glyph_content_form_loss,
)


EXPERIMENT = "continuous-glyph-content-form-v40"
DEFAULT_V34_CHECKPOINT = (
    "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
)
DEFAULT_DATABASE = (
    "/home/lachlan/ProjectsLFS/incoder/data/historic/etymology.sqlite3"
)
EXPECTED_DATABASE_SHA256 = (
    "c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738"
)
DEFAULT_CACHE = "artifacts/cache/v34_historic_glyph_rasters_32.pt"
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c"
)
DEFAULT_OUT = "artifacts/glyph_content_form_v40_20260814"
UPDATES = 3_000
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WARMUP_UPDATES = 100
MINIMUM_LEARNING_RATE_RATIO = 0.10
WEIGHT_DECAY = 0.02
GRADIENT_CLIP = 1.0
EMA_DECAY = 0.999
SEED = 20_264_000
SOURCE_FILES = (
    "ilm/visual_lm/glyph_content_form.py",
    "ilm/visual_lm/glyph_content_form_data.py",
    "ilm/visual_lm/glyph_content_form_training.py",
    "ilm/visual_lm/glyph_content_form_evaluation.py",
    "scripts/train_glyph_content_form_v40.py",
)


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], *, start: int, count: int) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V40 dataset window lies outside its deterministic stream")
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
        description="Train the V40 image-only cross-era glyph content/form factorizer."
    )
    parser.add_argument("--v34-checkpoint", default=DEFAULT_V34_CHECKPOINT)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup-updates", type=int, default=WARMUP_UPDATES)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--evaluate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--maximum-dev-families", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value: str, *, smoke: bool) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V40 requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(device)
        if not smoke and "4090" not in torch.cuda.get_device_name(device):
            raise RuntimeError("V40 evidence runs require an RTX 4090")
    elif not smoke:
        raise RuntimeError("V40 non-smoke training requires CUDA")
    return device


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


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
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")


@torch.no_grad()
def module_state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@torch.no_grad()
def load_trainable_model_state(
    model: torch.nn.Module,
    state: Mapping[str, Any],
) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(state) != set(parameters):
        raise ValueError("V40 trainable checkpoint parameter names changed")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape:
            raise ValueError(f"V40 trainable checkpoint tensor differs for {name}")
        parameters[name].copy_(value.to(parameters[name]))


def cosine_learning_rate(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    peak: float,
) -> float:
    if update < 1 or total_updates < 1 or not 0 <= warmup_updates < total_updates:
        raise ValueError("V40 learning-rate schedule is invalid")
    if update <= warmup_updates and warmup_updates:
        return peak * update / warmup_updates
    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak * (MINIMUM_LEARNING_RATE_RATIO + (1.0 - MINIMUM_LEARNING_RATE_RATIO) * cosine)


@contextmanager
def use_ema_parameters(
    model: GlyphContentFormModel,
    ema: WarmStartTrainableEMA,
):
    parameters = dict(model.named_parameters())
    backup = {
        name: parameters[name].detach().clone()
        for name in ema.shadow
    }
    ema.copy_to(model)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, value in backup.items():
                parameters[name].copy_(value)


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=glyph_content_form_collate,
    )


def load_data(
    *,
    database_path: Path,
    cache_path: Path,
    verified: bool,
) -> tuple[list[Any], torch.Tensor, dict[str, Any]]:
    database_hash = file_sha256(database_path)
    if verified and database_hash != EXPECTED_DATABASE_SHA256:
        raise RuntimeError("V40 historical database hash changed")
    records = load_historic_glyph_records(database_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache, Mapping) or not isinstance(cache.get("pixels"), torch.Tensor):
        raise TypeError("V40 historical raster cache is incomplete")
    expected = {
        "database_sha256": database_hash,
        "manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA256,
    }
    changed = [key for key, value in expected.items() if cache.get(key) != value]
    if verified and changed:
        raise RuntimeError(f"V40 historical raster cache provenance changed: {changed}")
    pixels = cache["pixels"]
    if len(pixels) != len(records):
        raise ValueError("V40 historical raster cache and records do not align")
    return records, pixels, {
        "database": str(database_path.resolve()),
        "database_sha256": database_hash,
        "cache": str(cache_path.resolve()),
        "cache_database_sha256": cache.get("database_sha256"),
        "cache_manifest_sha256": cache.get("manifest_sha256"),
        "records": len(records),
    }


def main() -> None:
    args = parse_args()
    if min(args.updates, args.batch_size, args.log_every) < 1:
        raise ValueError("V40 training dimensions must be positive")
    if args.num_workers < 0 or args.maximum_dev_families < 0:
        raise ValueError("V40 worker and evaluation limits cannot be negative")
    if min(args.evaluate_every, args.checkpoint_every) < 1:
        raise ValueError("V40 intervals must be positive")
    if args.smoke:
        args.updates = min(args.updates, 2)
        args.batch_size = min(args.batch_size, 4)
        args.warmup_updates = min(args.warmup_updates, max(0, args.updates - 1))
        args.evaluate_every = 1
        args.checkpoint_every = 1
    if args.learning_rate <= 0 or not 0 <= args.warmup_updates < args.updates:
        raise ValueError("V40 optimizer schedule is invalid")

    seed_everything(SEED)
    device = choose_device(args.device, smoke=args.smoke)
    verified = not args.allow_unverified
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    metrics_path = output_dir / "metrics.jsonl"
    if not args.resume and (checkpoint_path.is_file() or metrics_path.is_file()):
        raise FileExistsError(
            "V40 output already contains a run; use --resume or another --out"
        )
    records, pixels, data_receipt = load_data(
        database_path=Path(args.database),
        cache_path=Path(args.cache),
        verified=verified,
    )
    config = GlyphContentFormConfig()
    model = GlyphContentFormModel(config)
    codec_receipt = load_v40_v34_codec(
        model,
        args.v34_checkpoint,
        verify_hash=verified,
    )
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    ema = WarmStartTrainableEMA(model, decay=EMA_DECAY)
    completed_updates = 0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("architecture") != V40_ARCHITECTURE:
            raise ValueError("V40 resume checkpoint has the wrong architecture")
        if checkpoint.get("config") != asdict(config):
            raise ValueError("V40 resume checkpoint configuration changed")
        completed_updates = int(checkpoint.get("update", -1))
        if not 0 <= completed_updates < args.updates:
            raise ValueError("V40 resume update is outside the requested run")
        load_trainable_model_state(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])

    stream = CrossEraContentFormDataset(
        records,
        pixels,
        split="train",
        length=args.updates * args.batch_size,
        seed=SEED,
    )
    train_window = DatasetWindow(
        stream,
        start=completed_updates * args.batch_size,
        count=(args.updates - completed_updates) * args.batch_size,
    )
    development_probe = CrossEraContentFormDataset(
        records,
        pixels,
        split="development",
        length=1,
        seed=SEED + 1,
    )
    development_count = len(development_probe.families)
    if args.maximum_dev_families:
        development_count = min(development_count, args.maximum_dev_families)
    development = CrossEraContentFormDataset(
        records,
        pixels,
        split="development",
        length=development_count,
        seed=SEED + 1,
    )
    train_loader = make_loader(
        train_window,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    development_loader = make_loader(
        development,
        batch_size=min(args.batch_size, development_count),
        num_workers=args.num_workers,
        drop_last=False,
    )
    probe_batch = next(iter(development_loader))
    run_receipt = {
        "experiment": EXPERIMENT,
        "architecture": V40_ARCHITECTURE,
        "config": asdict(config),
        "boundary": glyph_content_form_boundary_receipt(model),
        "data_boundary": glyph_content_form_data_boundary_receipt(probe_batch),
        "data": data_receipt,
        "codec": codec_receipt,
        "source_sha256": {path: file_sha256(path) for path in SOURCE_FILES},
        "seed": SEED,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_families": len(stream.families),
        "development_families": development_count,
        "verified": verified,
        "smoke": args.smoke,
    }
    atomic_write_json(run_receipt, output_dir / "run_receipt.json")
    if completed_updates == 0:
        baseline = evaluate_glyph_content_form(
            model,
            development_loader,
            device=device,
        )
        atomic_write_json(baseline, output_dir / "development_zero_trained.json")

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"V40 received signal {signum}; checkpointing after this update", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    model.train()
    iterator: Iterator[dict[str, Any]] = iter(train_loader)
    last_metrics: dict[str, Any] = {}

    for update in range(completed_updates + 1, args.updates + 1):
        batch = next(iterator)
        student = {
            key: value.to(device=device, dtype=torch.float32, non_blocking=True)
            for key, value in glyph_content_form_student_batch(batch).items()
        }
        stage_ids, _ = glyph_content_form_stage_ids(batch["metadata"])
        stage_ids = stage_ids.to(device, non_blocking=True)
        learning_rate = cosine_learning_rate(
            update,
            total_updates=args.updates,
            warmup_updates=args.warmup_updates,
            peak=args.learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            output = model(**student)
            losses = glyph_content_form_loss(
                model,
                output,
                student,
                stage_ids=stage_ids,
            )
        losses.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("V40 gradient norm is not finite")
        optimizer.step()
        effective_ema_decay = ema.update(model)
        last_metrics = {
            "update": update,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm),
            "effective_ema_decay": effective_ema_decay,
            "elapsed_seconds": time.perf_counter() - started,
            **losses.detached_metrics(),
        }
        append_jsonl(metrics_path, last_metrics)
        if update == 1 or update % args.log_every == 0:
            print(json.dumps(last_metrics, sort_keys=True), flush=True)

        should_evaluate = update % args.evaluate_every == 0 or update == args.updates
        development_report: dict[str, Any] | None = None
        if should_evaluate:
            with use_ema_parameters(model, ema):
                development_report = evaluate_glyph_content_form(
                    model,
                    development_loader,
                    device=device,
                )
            development_report = {
                "update": update,
                "route": "warm-start-ema",
                **development_report,
            }
            atomic_write_json(
                development_report,
                output_dir / f"development_{update:06d}.json",
            )
            atomic_write_json(
                development_report,
                output_dir / "development_latest.json",
            )

        should_checkpoint = (
            update % args.checkpoint_every == 0
            or update == args.updates
            or stop_requested
        )
        if should_checkpoint:
            checkpoint = {
                "experiment": EXPERIMENT,
                "architecture": V40_ARCHITECTURE,
                "config": asdict(config),
                "update": update,
                "model": trainable_model_state(model),
                "optimizer": optimizer.state_dict(),
                "ema": ema.state_dict(),
                "last_metrics": last_metrics,
                "development": development_report,
                "run_receipt": run_receipt,
            }
            previous_checkpoint = output_dir / "checkpoint_previous.pt"
            if checkpoint_path.is_file():
                os.replace(checkpoint_path, previous_checkpoint)
            atomic_torch_save(checkpoint, checkpoint_path)
        if stop_requested:
            break

    with use_ema_parameters(model, ema):
        selected_sha256 = module_state_sha256(model)
    final = {
        **run_receipt,
        "completed_updates": int(last_metrics.get("update", completed_updates)),
        "last_metrics": last_metrics,
        "selected_route": "warm-start-ema",
        "selected_model_sha256": selected_sha256,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "stopped_early": stop_requested,
    }
    atomic_write_json(final, output_dir / "final_report.json")
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
