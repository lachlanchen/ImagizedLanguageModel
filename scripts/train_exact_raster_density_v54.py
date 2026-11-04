#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
)
from ilm.visual_lm.exact_raster_density_v54 import (
    V54_ARCHITECTURE,
    ExactRasterDensity,
    ExactRasterDensityConfig,
)
from ilm.visual_lm.visual_cell_data import load_v25_records, verify_v25_manifest
from scripts.train_canonical_glyph_binding_v43 import (
    _append_jsonl,
    _atomic_json,
    _atomic_save,
    _resolve_device,
    _scheduled_lr,
)
from scripts.train_causal_visual_block_diffusion_v53 import (
    DEFAULT_MANIFEST,
    DEFAULT_READER,
    load_frozen_reader,
)
from scripts.train_noisy_visual_energy_guidance_v52 import (
    REQUIRED_READER_SHA256,
    _selected_positions,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)

DEFAULT_OUT = "artifacts/exact_raster_density_v54_20261104"
PROTOCOL = "references/exact_raster_density_v54_protocol.md"
SOURCES = (
    "ilm/visual_lm/exact_raster_density_v54.py",
    "scripts/train_exact_raster_density_v54.py",
)


def args_parser():
    parser = argparse.ArgumentParser(
        description="Train exact next-raster probability with a fixed copying coordinate."
    )
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = args_parser()
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.batch_size = min(args.batch_size, 2)
        args.num_workers, args.log_every = 0, 1
    elif (args.steps, args.batch_size) != (8000, 8):
        raise ValueError("production V54 requires 8000 updates, 8 streams")
    device = _resolve_device(args.device)
    out = Path(args.out)
    if (out / "checkpoint_final.pt").exists():
        raise FileExistsError("completed V54 output already exists")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(20265410)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reader_hash = file_sha256(args.reader_checkpoint)
    if reader_hash != REQUIRED_READER_SHA256:
        raise ValueError("V54 requires the frozen V42 reader")
    provenance = {p: file_sha256(p) for p in (*SOURCES, PROTOCOL)}
    manifest = verify_v25_manifest(args.manifest, strict=True)
    reader = load_frozen_reader(args.reader_checkpoint, device)
    model = ExactRasterDensity(reader, ExactRasterDensityConfig()).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    boundary = model.boundary_receipt()
    if (
        boundary["density_parameters"] > 5_000_000
        or boundary["runtime_parameters"] > 23_000_000
    ):
        raise ValueError("V54 exceeds declared parameter budget")
    generator = torch.Generator(device=device).manual_seed(20265410)
    start, elapsed_before, peak_before = 0, 0.0, 0.0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            resume["architecture"] != V54_ARCHITECTURE
            or resume["provenance"] != provenance
        ):
            raise ValueError("resume architecture/source/protocol mismatch")
        if (
            resume["reader_checkpoint_sha256"] != reader_hash
            or resume["manifest"]["sha256"] != manifest["sha256"]
        ):
            raise ValueError("resume data/foundation mismatch")
        for key in ("steps", "batch_size", "smoke"):
            if resume["arguments"][key] != getattr(args, key):
                raise ValueError(f"resume optimization differs: {key}")
        model.load_state_dict(resume["model"])
        ema.load_state_dict(resume["ema_model"])
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["training_rng"].cpu())
        start, elapsed_before, peak_before = (
            resume["update"],
            resume["elapsed_seconds"],
            resume["peak_allocated_vram_gib"],
        )
    if start >= args.steps:
        raise ValueError("resume already reached update limit")
    records = load_v25_records(args.manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig()
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render,
        seed=20265411,
        length=args.steps * args.batch_size,
    )
    loader = DataLoader(
        Subset(dataset, range(start * args.batch_size, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=canonical_glyph_collate,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(20265411),
    )
    iterator = iter(loader)
    if resume:
        torch.set_rng_state(resume["cpu_rng"])
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"], device)
    stopped = False

    def request_stop(_sig, _frame):
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    started = time.perf_counter()
    try:
        for update in range(start + 1, args.steps + 1):
            model.train()
            lr = _scheduled_lr(
                update,
                base=2e-4,
                warmup=min(400, args.steps // 2),
                total=args.steps,
                minimum_ratio=0.1,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            raw = next(iterator)
            context = raw["context"].to(device, non_blocking=True)
            targets = raw["target"].to(device, non_blocking=True)
            positions = _selected_positions(
                len(context),
                per_stratum=1 if args.smoke else 4,
                device=device,
                generator=generator,
            )
            row = torch.arange(len(context), device=device)[:, None]
            with torch.no_grad(), autocast_context(device, "bf16"):
                language = reader.language(context)
                hidden = language["hidden_states"][row, positions].flatten(0, 1).float()
                fields = language["anchor_fields"][row, positions].flatten(0, 1).float()
                # Font rendering contains antialiasing. The registered density
                # is over binary sensor values, matching the reader threshold.
                image = (targets[row, positions].flatten(0, 1) >= 0.5).float()
                exact_fields = reader.field.encode_unit(image).float()
            copying = torch.rand(len(image), device=device, generator=generator) < 0.25
            hidden = hidden.masked_fill(copying[:, None], 0)
            fields = torch.where(copying[:, None], exact_fields, fields)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, "bf16"):
                nll = model.image_nll(image, hidden, fields)
                loss = nll.mean() / 1024
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("V54 loss is not finite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 0.001)
                for eb, b in zip(ema.buffers(), model.buffers()):
                    eb.copy_(b)
            if any(p.grad is not None for p in reader.parameters()):
                raise RuntimeError("frozen reader received gradients")
            peak = max(
                peak_before,
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0,
            )
            if peak >= 18:
                raise MemoryError("V54 crossed resource limit")
            metrics = {
                "update": update,
                "image_nll_nats": float(nll.detach().mean()),
                "density_loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "learning_rate": lr,
                "copy_fraction": float(copying.float().mean()),
                "language_image_nll": float(nll.detach()[~copying].mean())
                if bool((~copying).any())
                else None,
                "copy_image_nll": float(nll.detach()[copying].mean())
                if bool(copying.any())
                else None,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            }
            if update == 1 or update % args.log_every == 0:
                _append_jsonl(out / "training_metrics.jsonl", metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
            stopped |= args.stop_after is not None and update >= args.stop_after
            if update % args.save_every == 0 or update == args.steps or stopped:
                payload = {
                    "architecture": V54_ARCHITECTURE,
                    "model_config": asdict(model.config),
                    "reader_config": asdict(reader.config),
                    "model": model.state_dict(),
                    "ema_model": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "training_rng": generator.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "arguments": vars(args),
                    "render_config": asdict(render),
                    "manifest": manifest,
                    "provenance": provenance,
                    "reader_checkpoint_sha256": reader_hash,
                    "boundary": boundary,
                    "elapsed_seconds": metrics["elapsed_seconds"],
                    "peak_allocated_vram_gib": peak,
                    "training_metrics": metrics,
                    "smoke_only": args.smoke,
                }
                _atomic_save(payload, out / "checkpoint_latest.pt")
                if update == args.steps:
                    _atomic_save(payload, out / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            k: payload[k]
                            for k in (
                                "architecture",
                                "update",
                                "boundary",
                                "elapsed_seconds",
                                "peak_allocated_vram_gib",
                                "training_metrics",
                                "provenance",
                                "smoke_only",
                            )
                        },
                        out / "training_summary.json",
                    )
            if stopped:
                break
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
