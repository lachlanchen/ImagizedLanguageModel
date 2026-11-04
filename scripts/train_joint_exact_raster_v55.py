#!/usr/bin/env python3
"""Matched frozen/joint learning with an unchanged exact raster probability law."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
)
from ilm.visual_lm.joint_exact_raster_v55 import ARCHITECTURE, model_from_checkpoint
from ilm.visual_lm.visual_cell_data import load_v25_records, verify_v25_manifest
from scripts.train_canonical_glyph_binding_v43 import (
    _append_jsonl,
    _atomic_json,
    _atomic_save,
    _resolve_device,
    _scheduled_lr,
)
from scripts.train_noisy_visual_energy_guidance_v52 import _selected_positions
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)

DEFAULT_SOURCE = "artifacts/exact_raster_density_v54_20261104/checkpoint_final.pt"
DEFAULT_OUT = "artifacts/joint_exact_raster_v55_20261105"
SOURCE_SHA = "f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee"
PROTOCOL = "references/joint_exact_raster_v55_protocol.md"
SOURCES = (
    "ilm/visual_lm/joint_exact_raster_v55.py",
    "ilm/visual_lm/exact_raster_density_v54.py",
    "scripts/train_joint_exact_raster_v55.py",
    PROTOCOL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("frozen", "joint"), required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    parser.add_argument("--out")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.steps, args.batch_size, args.num_workers = min(args.steps, 4), 2, 0
    elif (args.steps, args.batch_size) != (8000, 8):
        raise ValueError("production V55 requires 8000 updates and eight streams")
    out = Path(args.out or f"{DEFAULT_OUT}/{args.arm}")
    if (out / "checkpoint_final.pt").exists():
        raise FileExistsError("completed arm already exists")
    out.mkdir(parents=True, exist_ok=True)
    if file_sha256(args.source) != SOURCE_SHA:
        raise ValueError("V55 requires the unchanged V54 initialization")
    provenance = {p: file_sha256(p) for p in SOURCES}
    device = _resolve_device(args.device)
    seed_everything(20265510)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    source["arm"] = args.arm
    model = model_from_checkpoint(source, device, trainable=True)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("reader.")
            ],
            "peak_lr": 2e-4,
        }
    ]
    if args.arm == "joint":
        groups.append(
            {
                "params": [p for p in model.reader.parameters() if p.requires_grad],
                "peak_lr": 2e-5,
            }
        )
    optimizer = torch.optim.AdamW(
        groups,
        lr=2e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    manifest = verify_v25_manifest(args.manifest, strict=True)
    if manifest["sha256"] != source["manifest"]["sha256"]:
        raise ValueError("initialization corpus differs")
    boundary = model.boundary_receipt()
    generator = torch.Generator(device=device).manual_seed(20265510)
    start, elapsed_before, peak_before, resume = 0, 0.0, 0.0, None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume["architecture"] != ARCHITECTURE or resume["provenance"] != provenance:
            raise ValueError("resume source/protocol mismatch")
        if (
            resume["arm"] != args.arm
            or resume["initialization_sha256"] != SOURCE_SHA
            or resume["manifest"]["sha256"] != manifest["sha256"]
        ):
            raise ValueError("resume arm/initialization/corpus mismatch")
        for key in ["steps", "batch_size", "smoke"]:
            if resume["arguments"][key] != getattr(args, key):
                raise ValueError("resume optimization mismatch")
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
        raise ValueError("resume already completed")
    records = load_v25_records(args.manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig(**source["render_config"])
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render,
        seed=20265511,
        length=args.steps * args.batch_size,
    )
    loader = DataLoader(
        Subset(dataset, range(start * args.batch_size, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=canonical_glyph_collate,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(20265511),
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
            model.train()  # The inherited method keeps reader dropout disabled in both arms.
            for group in optimizer.param_groups:
                group["lr"] = _scheduled_lr(
                    update,
                    base=group["peak_lr"],
                    warmup=min(400, args.steps // 2),
                    total=args.steps,
                    minimum_ratio=0.1,
                )
            raw = next(iterator)
            context = raw["context"].to(device, non_blocking=True)
            targets = raw["target"].to(device, non_blocking=True)
            positions = _selected_positions(
                len(context),
                per_stratum=1 if args.smoke else 4,
                device=device,
                generator=generator,
            )
            indices = torch.arange(len(context), device=device)[:, None]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, "bf16"):
                # No no_grad: the joint arm learns the reader from image NLL.
                states = model.reader.language(context)
                hidden = (
                    states["hidden_states"][indices, positions].flatten(0, 1).float()
                )
                fields = (
                    states["anchor_fields"][indices, positions].flatten(0, 1).float()
                )
                image = (targets[indices, positions].flatten(0, 1) >= 0.5).float()
                exact_fields = model.reader.field.encode_unit(image).float()
                copying = (
                    torch.rand(len(image), device=device, generator=generator) < 0.25
                )
                hidden = hidden.masked_fill(copying[:, None], 0)
                fields = torch.where(copying[:, None], exact_fields, fields)
                nll = model.image_nll(image, hidden, fields)
                loss = nll.mean() / 1024
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite V55 loss")
            loss.backward()
            reader_squares = [
                p.grad.detach().float().square().sum()
                for p in model.reader.parameters()
                if p.grad is not None
            ]
            reader_grad = (
                float(torch.stack(reader_squares).sum().sqrt())
                if reader_squares
                else 0.0
            )
            if args.arm == "frozen" and reader_grad:
                raise RuntimeError("frozen control reader received gradients")
            if args.arm == "joint" and (
                not np.isfinite(reader_grad) or reader_grad <= 0
            ):
                raise RuntimeError("joint reader has no finite likelihood gradient")
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(p, 0.001)
                for eb, b in zip(ema.buffers(), model.buffers()):
                    eb.copy_(b)
            peak = max(
                peak_before,
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0,
            )
            if peak >= 18:
                raise MemoryError("V55 exceeded resource budget")
            metrics = {
                "update": update,
                "arm": args.arm,
                "image_nll_nats": float(nll.detach().mean()),
                "natural_nll": float(nll.detach()[~copying].mean())
                if bool((~copying).any())
                else None,
                "copy_nll": float(nll.detach()[copying].mean())
                if bool(copying.any())
                else None,
                "reader_gradient_norm": reader_grad,
                "global_gradient_norm": float(norm),
                "copy_fraction": float(copying.float().mean()),
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            }
            if update == 1 or update % 100 == 0 or args.smoke:
                _append_jsonl(out / "training_metrics.jsonl", metrics)
                print(json.dumps(metrics, sort_keys=True), flush=True)
            stopped |= args.stop_after is not None and update >= args.stop_after
            if update % 1000 == 0 or update == args.steps or stopped:
                payload = {
                    "architecture": ARCHITECTURE,
                    "arm": args.arm,
                    "model_config": asdict(model.config),
                    "reader_config": asdict(model.reader.config),
                    "model": model.state_dict(),
                    "ema_model": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_rng": generator.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "arguments": vars(args),
                    "update": update,
                    "manifest": manifest,
                    "render_config": asdict(render),
                    "provenance": provenance,
                    "initialization_sha256": SOURCE_SHA,
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
                            for k in [
                                "architecture",
                                "arm",
                                "update",
                                "manifest",
                                "initialization_sha256",
                                "boundary",
                                "elapsed_seconds",
                                "peak_allocated_vram_gib",
                                "training_metrics",
                                "provenance",
                                "smoke_only",
                            ]
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
