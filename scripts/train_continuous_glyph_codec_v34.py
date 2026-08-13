#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
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

from ilm.visual_lm.continuous_glyph_codec import (
    V34_ARCHITECTURE,
    ContinuousGlyphCodec,
    ContinuousGlyphCodecConfig,
    ContinuousGlyphCodecOutput,
    continuous_glyph_codec_boundary_receipt,
)
from ilm.visual_lm.continuous_glyph_codec_data import (
    HistoricGlyphRasterDataset,
    active_rendered_patches,
    file_sha256,
    historic_glyph_collate,
    historic_svg_manifest_sha256,
    load_historic_glyph_records,
    load_or_build_historic_raster_cache,
    v34_codec_data_boundary_receipt,
    v34_historic_split_counts,
)
from ilm.visual_lm.continuous_glyph_codec_evaluation import (
    evaluate_continuous_glyph_codec,
    v34_development_gate,
    v34_sealed_transfer_gate,
    write_v34_evaluation,
)
from ilm.visual_lm.continuous_glyph_codec_training import (
    continuous_glyph_codec_loss,
    training_latent_noise,
)
from ilm.visual_lm.direct_visual_patch_data import (
    DirectPatchContinuationDataset,
    DirectPatchRenderConfig,
    direct_patch_collate,
)
from ilm.visual_lm.direct_visual_patch_training import (
    ExponentialMovingAverage,
    module_state_sha256,
    stage_cosine_learning_rate,
)
from ilm.visual_lm.visual_semantic_raster_data import load_visual_text_records


EXPERIMENT = "continuous-glyph-representation-codec-v34"
PROTOCOL_DOCUMENT = "references/continuous_glyph_codec_v34_protocol.md"
EXPECTED_PROTOCOL_SHA256 = (
    "c2370374f202714e217236f7634f464eb98bed6a0f8afe898b9658614df7ce51"
)
DEFAULT_PUBLIC_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
EXPECTED_PUBLIC_SHA256 = (
    "76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03"
)
DEFAULT_HISTORIC_ROOT = "/home/lachlan/ProjectsLFS/incoder"
DEFAULT_HISTORIC_DATABASE = (
    "/home/lachlan/ProjectsLFS/incoder/data/historic/etymology.sqlite3"
)
EXPECTED_HISTORIC_DATABASE_SHA256 = (
    "c563e8587d7dcacf73704c0fb7816f6d830db11122e0a3da62678b3a7119f738"
)
EXPECTED_HISTORIC_MANIFEST_SHA256 = (
    "3c4064441563c88dffe0c36d42cce0c381bf8b401b764b87484edfb4aa7db99c"
)
UPDATES = 6_000
RENDERED_BATCH_SIZE = 8
HISTORIC_BATCH_SIZE = 128
PEAK_LEARNING_RATE = 2e-4
WARMUP_UPDATES = 250
MINIMUM_LEARNING_RATE_RATIO = 0.10
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
EMA_DECAY = 0.999
SEED = 20_263_400
CHECKPOINT_INTERVAL = 1_000
MAXIMUM_VRAM_BYTES = 12 * 1024**3
SOURCE_FILES = (
    "ilm/visual_lm/continuous_glyph_codec.py",
    "ilm/visual_lm/continuous_glyph_codec_data.py",
    "ilm/visual_lm/continuous_glyph_codec_training.py",
    "ilm/visual_lm/continuous_glyph_codec_evaluation.py",
    "scripts/train_continuous_glyph_codec_v34.py",
)


class DatasetWindow(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], *, start: int, count: int) -> None:
        if start < 0 or count < 1 or start + count > len(dataset):
            raise ValueError("V34 dataset window lies outside its deterministic stream")
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
        description="Train the preregistered V34 continuous visual glyph codec."
    )
    parser.add_argument("--public-manifest", default=DEFAULT_PUBLIC_MANIFEST)
    parser.add_argument("--historic-root", default=DEFAULT_HISTORIC_ROOT)
    parser.add_argument("--historic-database", default=DEFAULT_HISTORIC_DATABASE)
    parser.add_argument(
        "--historic-cache",
        default="artifacts/cache/v34_historic_glyph_rasters_32.pt",
    )
    parser.add_argument(
        "--out",
        default="artifacts/continuous_glyph_codec_v34_20260814",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def choose_device(value: str, *, smoke: bool) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type != "cuda" or not torch.cuda.is_available():
        if smoke and device.type == "cpu":
            return device
        raise RuntimeError("V34 evidence requires CUDA")
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if not smoke and "4090" not in name:
        raise RuntimeError(f"V34 evidence requires an RTX 4090, found {name!r}")
    if not smoke and not torch.cuda.is_bf16_supported():
        raise RuntimeError("V34 evidence requires CUDA BF16 support")
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


def tensors_are_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(tensors_are_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensors_are_finite(item) for item in value)
    return True


def build_rendered_stream(
    records: list[Any],
    *,
    completed_updates: int,
    planned_updates: int,
    rendered_batch_size: int,
) -> tuple[DatasetWindow, dict[str, int]]:
    probe = DirectPatchContinuationDataset(
        records,
        split="train",
        config=DirectPatchRenderConfig(),
        variants_per_record=1,
        seed=SEED,
    )
    stream_examples = planned_updates * rendered_batch_size
    variants = math.ceil(stream_examples / len(probe))
    stream = DirectPatchContinuationDataset(
        records,
        split="train",
        config=DirectPatchRenderConfig(),
        variants_per_record=variants,
        seed=SEED,
    )
    start = completed_updates * rendered_batch_size
    count = (planned_updates - completed_updates) * rendered_batch_size
    return DatasetWindow(stream, start=start, count=count), {
        "records": len(probe),
        "variants_per_record": variants,
        "stream_examples": stream_examples,
        "window_start": start,
        "window_count": count,
    }


def build_historic_stream(
    records: list[Any],
    pixels: torch.Tensor,
    *,
    completed_updates: int,
    planned_updates: int,
    historic_batch_size: int,
) -> tuple[DatasetWindow, dict[str, int]]:
    examples = planned_updates * historic_batch_size
    stream = HistoricGlyphRasterDataset(
        records,
        pixels,
        split="train",
        example_count=examples,
    )
    start = completed_updates * historic_batch_size
    count = (planned_updates - completed_updates) * historic_batch_size
    return DatasetWindow(stream, start=start, count=count), {
        "unique_train_glyphs": len(stream.indices),
        "stream_examples": examples,
        "window_start": start,
        "window_count": count,
    }


def make_loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    num_workers: int,
    collate_fn: Any,
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
            collate_fn=collate_fn,
        )
    )


def validate_resume_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    planned_updates: int,
    rendered_batch_size: int,
    historic_batch_size: int,
) -> int:
    if checkpoint.get("experiment") != EXPERIMENT:
        raise ValueError("V34 resume checkpoint has the wrong experiment marker")
    if checkpoint.get("architecture") != V34_ARCHITECTURE:
        raise ValueError("V34 resume checkpoint has the wrong architecture")
    if checkpoint.get("protocol", {}).get("sha256") != EXPECTED_PROTOCOL_SHA256:
        raise ValueError("V34 resume checkpoint has the wrong protocol hash")
    completed = int(checkpoint.get("update", -1))
    if not 0 <= completed < planned_updates:
        raise ValueError("V34 resume update lies outside the planned run")
    expected_rendered = completed * rendered_batch_size
    expected_historic = completed * historic_batch_size
    if int(checkpoint.get("rendered_examples_consumed", -1)) != expected_rendered:
        raise ValueError("V34 rendered resume cursor is inconsistent")
    if int(checkpoint.get("historic_examples_consumed", -1)) != expected_historic:
        raise ValueError("V34 historical resume cursor is inconsistent")
    return completed


def main() -> None:
    args = parse_args()
    if args.num_workers < 0 or args.cache_workers < 0 or args.log_every < 1:
        raise ValueError("V34 worker and logging settings are invalid")
    if args.smoke:
        planned_updates = min(args.updates, 2)
        rendered_batch_size = 2
        historic_batch_size = 4
        evaluation_rendered_patches = 8
        evaluation_historic_patches = 8
        checkpoint_interval = 1
    else:
        if args.updates != UPDATES:
            raise ValueError(f"V34 evidence requires --updates={UPDATES}")
        planned_updates = UPDATES
        rendered_batch_size = RENDERED_BATCH_SIZE
        historic_batch_size = HISTORIC_BATCH_SIZE
        evaluation_rendered_patches = 4_096
        evaluation_historic_patches = 4_096
        checkpoint_interval = CHECKPOINT_INTERVAL
    if planned_updates < 1:
        raise ValueError("V34 requires at least one update")

    device = choose_device(args.device, smoke=args.smoke)
    seed_everything(SEED)
    protocol_hash = file_sha256(PROTOCOL_DOCUMENT)
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("V34 protocol changed after preregistration")
    public_hash = file_sha256(args.public_manifest)
    if public_hash != EXPECTED_PUBLIC_SHA256:
        raise RuntimeError("V34 rendered manifest differs from preregistration")
    database_hash = file_sha256(args.historic_database)
    if database_hash != EXPECTED_HISTORIC_DATABASE_SHA256:
        raise RuntimeError("V34 historical database differs from preregistration")
    historic_manifest_hash, historic_svg_files = historic_svg_manifest_sha256(
        args.historic_root
    )
    if historic_manifest_hash != EXPECTED_HISTORIC_MANIFEST_SHA256:
        raise RuntimeError("V34 historical SVG manifest differs from preregistration")

    out = Path(args.out)
    checkpoint_path = out / "checkpoint_latest.pt"
    metrics_path = out / "training_metrics.jsonl"
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("V34 resume checkpoint does not exist")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("V34 checkpoint must be a mapping")
        completed_updates = validate_resume_checkpoint(
            checkpoint,
            planned_updates=planned_updates,
            rendered_batch_size=rendered_batch_size,
            historic_batch_size=historic_batch_size,
        )
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("V34 output directory is nonempty; use --resume")
        out.mkdir(parents=True, exist_ok=True)
        checkpoint = None
        completed_updates = 0

    historic_records = load_historic_glyph_records(args.historic_database)
    historic_pixels = load_or_build_historic_raster_cache(
        historic_records,
        root=args.historic_root,
        cache_path=args.historic_cache,
        database_sha256=database_hash,
        manifest_sha256=historic_manifest_hash,
        workers=args.cache_workers,
    )
    public_records = load_visual_text_records(args.public_manifest)
    rendered_stream, rendered_cursor = build_rendered_stream(
        public_records,
        completed_updates=completed_updates,
        planned_updates=planned_updates,
        rendered_batch_size=rendered_batch_size,
    )
    historic_stream, historic_cursor = build_historic_stream(
        historic_records,
        historic_pixels,
        completed_updates=completed_updates,
        planned_updates=planned_updates,
        historic_batch_size=historic_batch_size,
    )
    development_rendered = DirectPatchContinuationDataset(
        public_records,
        split="development",
        config=DirectPatchRenderConfig(),
        variants_per_record=2,
        seed=SEED + 1_000_000,
    )
    development_historic = HistoricGlyphRasterDataset(
        historic_records,
        historic_pixels,
        split="development",
    )

    model = ContinuousGlyphCodec(ContinuousGlyphCodecConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        fused=device.type == "cuda",
    )
    ema = ExponentialMovingAverage(model, decay=EMA_DECAY)
    total_patches = 0
    elapsed_before_resume = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"])
        total_patches = int(checkpoint.get("total_patches", 0))
        elapsed_before_resume = float(checkpoint.get("training_elapsed_seconds", 0.0))
        del checkpoint

    source_hashes = {path: file_sha256(path) for path in SOURCE_FILES}
    if args.resume:
        receipt = json.loads((out / "run_receipt.json").read_text(encoding="utf-8"))
        if receipt["source_sha256"] != source_hashes:
            raise RuntimeError("V34 source changed since the interrupted run")
    else:
        rendered_sample = direct_patch_collate([rendered_stream[0]])
        historic_sample = historic_glyph_collate([historic_stream[0]])
        receipt = {
            "experiment": EXPERIMENT,
            "label": "smoke" if args.smoke else "evidence",
            "architecture": V34_ARCHITECTURE,
            "protocol": {"path": PROTOCOL_DOCUMENT, "sha256": protocol_hash},
            "source_sha256": source_hashes,
            "data": {
                "public_manifest": str(args.public_manifest),
                "public_sha256": public_hash,
                "public_records": len(public_records),
                "historic_database": str(args.historic_database),
                "historic_database_sha256": database_hash,
                "historic_root": str(args.historic_root),
                "historic_svg_manifest_sha256": historic_manifest_hash,
                "historic_svg_files": historic_svg_files,
                "historic_indexed_records": len(historic_records),
                "historic_splits": v34_historic_split_counts(historic_records),
                "rendered_cursor": rendered_cursor,
                "historic_cursor": historic_cursor,
            },
            "fixed_optimization": {
                "updates": planned_updates,
                "rendered_strips_per_update": rendered_batch_size,
                "historic_glyphs_per_update": historic_batch_size,
                "optimizer": "AdamW",
                "betas": [0.9, 0.95],
                "weight_decay": WEIGHT_DECAY,
                "peak_learning_rate": PEAK_LEARNING_RATE,
                "warmup_updates": min(WARMUP_UPDATES, max(0, planned_updates - 1)),
                "minimum_learning_rate": PEAK_LEARNING_RATE
                * MINIMUM_LEARNING_RATE_RATIO,
                "gradient_clip": GRADIENT_CLIP,
                "ema_decay": EMA_DECAY,
                "seed": SEED,
                "precision": "bf16" if device.type == "cuda" else "fp32",
            },
            "model_boundary": continuous_glyph_codec_boundary_receipt(model),
            "data_boundary": v34_codec_data_boundary_receipt(
                rendered_sample,
                historic_sample,
            ),
            "initial_model_sha256": module_state_sha256(model),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
            "torch": torch.__version__,
        }
        atomic_write_json(receipt, out / "run_receipt.json")
        print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)

    rendered_loader = make_loader(
        rendered_stream,
        batch_size=rendered_batch_size,
        num_workers=args.num_workers,
        collate_fn=direct_patch_collate,
        pin_memory=device.type == "cuda",
    )
    historic_loader = make_loader(
        historic_stream,
        batch_size=historic_batch_size,
        num_workers=args.num_workers,
        collate_fn=historic_glyph_collate,
        pin_memory=device.type == "cuda",
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"V34 received signal {signum}; saving after this update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    current_update = completed_updates
    current_elapsed = elapsed_before_resume
    started = time.monotonic()

    def save_checkpoint() -> None:
        payload = {
            "experiment": EXPERIMENT,
            "architecture": V34_ARCHITECTURE,
            "protocol": {"path": PROTOCOL_DOCUMENT, "sha256": protocol_hash},
            "model_config": asdict(model.config),
            "model": cpu_model_state(model),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "update": current_update,
            "rendered_examples_consumed": current_update * rendered_batch_size,
            "historic_examples_consumed": current_update * historic_batch_size,
            "total_patches": total_patches,
            "training_elapsed_seconds": current_elapsed,
            "receipt": receipt,
        }
        atomic_torch_save(payload, checkpoint_path)

    for update in range(completed_updates + 1, planned_updates + 1):
        learning_rate = stage_cosine_learning_rate(
            update,
            peak=PEAK_LEARNING_RATE,
            warmup=min(WARMUP_UPDATES, max(0, planned_updates - 1)),
            total=planned_updates,
            minimum_ratio=MINIMUM_LEARNING_RATE_RATIO,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        rendered_batch = next(rendered_loader)
        historic_batch = next(historic_loader)
        rendered_patches = active_rendered_patches(rendered_batch)
        patches = torch.cat((rendered_patches, historic_batch["pixels"])).to(
            device,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            latents = model.encode(patches)
            latent_noise = training_latent_noise(
                latents,
                seed=SEED,
                update=update,
            )
            logits = model.decode(latents + latent_noise)
            output = ContinuousGlyphCodecOutput(
                logits=logits,
                latents=latents,
                decoder_latents=latents + latent_noise,
            )
            losses = continuous_glyph_codec_loss(output, patches)
        if not bool(torch.isfinite(losses.loss)):
            raise FloatingPointError("V34 encountered a non-finite loss")
        losses.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("V34 encountered a non-finite gradient")
        optimizer.step()
        ema.update(model)
        current_update = update
        total_patches += patches.shape[0]
        current_elapsed = elapsed_before_resume + time.monotonic() - started
        row = {
            "update": update,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm),
            "rendered_patches": rendered_patches.shape[0],
            "historic_patches": historic_batch_size,
            "total_patches": total_patches,
            "elapsed_seconds": current_elapsed,
            **losses.detached_metrics(),
        }
        append_jsonl(metrics_path, row)
        if update == 1 or update % args.log_every == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if update % checkpoint_interval == 0 or stop_requested:
            save_checkpoint()
        if stop_requested:
            break

    updates_complete = current_update == planned_updates and not stop_requested
    if not checkpoint_path.is_file() or current_update % checkpoint_interval:
        save_checkpoint()
    if not updates_complete:
        partial = {
            "decision": "interrupted-resumable",
            "updates_completed": current_update,
            "updates_planned": planned_updates,
            "checkpoint": str(checkpoint_path),
        }
        atomic_write_json(partial, out / "training_summary.json")
        print(json.dumps(partial, indent=2), flush=True)
        return

    del rendered_loader, historic_loader, rendered_stream, historic_stream
    gc.collect()
    saved_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_finite = isinstance(saved_checkpoint, Mapping) and tensors_are_finite(
        saved_checkpoint
    )
    del saved_checkpoint
    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    raw_development = evaluate_continuous_glyph_codec(
        model,
        development_rendered,
        development_historic,
        split="development",
        device=device,
        precision="bf16" if device.type == "cuda" else "fp32",
        rendered_minimum_patches=evaluation_rendered_patches,
        historical_minimum_patches=evaluation_historic_patches,
        rendered_batch_size=rendered_batch_size,
        historical_batch_size=512 if not args.smoke else 8,
        gallery_directory=out / "galleries" / "raw",
    )
    write_v34_evaluation(raw_development, out / "development_raw.json")

    raw_state = cpu_model_state(model)
    ema.copy_to(model)
    ema_development = evaluate_continuous_glyph_codec(
        model,
        development_rendered,
        development_historic,
        split="development",
        device=device,
        precision="bf16" if device.type == "cuda" else "fp32",
        rendered_minimum_patches=evaluation_rendered_patches,
        historical_minimum_patches=evaluation_historic_patches,
        rendered_batch_size=rendered_batch_size,
        historical_batch_size=512 if not args.smoke else 8,
        gallery_directory=out / "galleries" / "ema",
    )
    development_gate = v34_development_gate(
        ema_development,
        updates_complete=updates_complete,
        checkpoint_finite=checkpoint_finite,
        peak_vram_bytes=peak_vram,
    )
    ema_development["selection_route"] = "ema"
    ema_development["development_gate"] = development_gate
    write_v34_evaluation(ema_development, out / "development_ema.json")

    sealed_report: dict[str, Any] | None = None
    sealed_gate: dict[str, Any] | None = None
    if development_gate["pass"] and not args.smoke:
        sealed_rendered = DirectPatchContinuationDataset(
            public_records,
            split="sealed",
            config=DirectPatchRenderConfig(),
            variants_per_record=2,
            seed=SEED + 2_000_000,
        )
        sealed_historic = HistoricGlyphRasterDataset(
            historic_records,
            historic_pixels,
            split="sealed",
        )
        sealed_report = evaluate_continuous_glyph_codec(
            model,
            sealed_rendered,
            sealed_historic,
            split="sealed",
            device=device,
            precision="bf16",
            rendered_minimum_patches=4_096,
            historical_minimum_patches=4_096,
            rendered_batch_size=rendered_batch_size,
            historical_batch_size=512,
            gallery_directory=out / "galleries" / "ema",
        )
        sealed_gate = v34_sealed_transfer_gate(ema_development, sealed_report)
        sealed_report["transfer_gate"] = sealed_gate
        write_v34_evaluation(sealed_report, out / "sealed_ema.json")

    model.load_state_dict(raw_state, strict=True)
    if args.smoke:
        decision = "smoke-complete-not-evidence"
    elif not development_gate["pass"]:
        decision = "continuous-codec-failure"
    elif sealed_gate is not None and sealed_gate["pass"]:
        decision = "continuous-codec-qualified"
    else:
        decision = "development-qualified-sealed-failure"
    checkpoint_hash = file_sha256(checkpoint_path)
    summary = {
        "experiment": EXPERIMENT,
        "decision": decision,
        "updates_completed": current_update,
        "updates_planned": planned_updates,
        "training_elapsed_seconds": current_elapsed,
        "total_training_patches": total_patches,
        "training_patches_per_second": total_patches / current_elapsed,
        "peak_vram_bytes": peak_vram,
        "peak_vram_below_12_gib": peak_vram < MAXIMUM_VRAM_BYTES,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_finite": checkpoint_finite,
        "raw_model_sha256": module_state_sha256(model),
        "development_gate": development_gate,
        "sealed_gate": sealed_gate,
        "sealed_opened": sealed_report is not None,
    }
    atomic_write_json(summary, out / "training_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
