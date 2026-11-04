#!/usr/bin/env python3
"""Train the image-only V57 copy/raster mixture or its matched no-copy arm."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
)
from ilm.visual_lm.visual_cell_data import load_v25_records, verify_v25_manifest
from ilm.visual_lm.visual_copy_language import ARCHITECTURE, model_from_foundation
from ilm.visual_lm.visual_qa_data import VisualQARenderer, pack_qa, read_questions
from scripts.train_canonical_glyph_binding_v43 import (
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

SOURCE = "artifacts/joint_exact_raster_v55_20261105/joint/checkpoint_final.pt"
SOURCE_HASH = "ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394"
DATA = "data/visual_qa/cmrc2018_short"
DATA_HASHES = {
    "train.jsonl": "742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c",
    "development.jsonl": "8bef093075ac0715d5878a1abe61aa38092893ade8753b935cdf657551ebc6a7",
    "counterfactual_pairs.jsonl": "3bcb2e0534290ed9cf3dd1a4e378bbaa5a7eb4903dce0b4b0befb8f3ce58029d",
}
SOURCES = [
    "ilm/visual_lm/visual_copy_language.py",
    "ilm/visual_lm/visual_qa_data.py",
    "scripts/train_visual_copy_language_v57.py",
    "references/visual_copy_language_v57_protocol.md",
    "ilm/visual_lm/canonical_glyph_language.py",
    "ilm/visual_lm/hierarchical_raster_language.py",
    "ilm/visual_lm/joint_exact_raster_v55.py",
    "ilm/visual_lm/exact_raster_density_v54.py",
    "ilm/visual_lm/visual_cell_stream.py",
    "ilm/visual_lm/canonical_glyph_language_data.py",
    "ilm/visual_lm/visual_cell_data.py",
]


def natural_batch(dataset, update, batch_size, device, generator):
    raw = [dataset[(update - 1) * batch_size + i] for i in range(batch_size)]
    sequence = (torch.stack([r["context"] for r in raw]).to(device) >= 0.5).float()
    targets = (torch.stack([r["target"] for r in raw]).to(device) >= 0.5).float()
    positions = _selected_positions(
        batch_size, per_stratum=4, device=device, generator=generator
    )
    b = torch.arange(batch_size, device=device)[:, None]
    targets = targets[b, positions]
    continued = torch.ones(positions.shape, dtype=torch.bool, device=device)
    motor = torch.rand(positions.shape, device=device, generator=generator) < 0.25
    return dict(
        sequence=sequence,
        prompt_lengths=torch.full((batch_size,), 64, device=device),
        query_positions=positions,
        targets=targets,
        continued=continued,
        stopped=torch.zeros_like(continued),
        motor=motor,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["copy", "no_copy"], required=True)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--data", default=DATA)
    p.add_argument(
        "--manifest",
        default="data/visual_grammar/chinese_wikisource_public_domain.jsonl",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.smoke:
        a.steps, a.batch_size = min(a.steps, 4), 2
    elif (a.steps, a.batch_size) != (4000, 4):
        raise ValueError("production schedule is locked")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint_final.pt").exists():
        raise FileExistsError("completed arm exists")
    if file_sha256(a.source) != SOURCE_HASH:
        raise ValueError("wrong foundation")
    for name, digest in DATA_HASHES.items():
        if file_sha256(Path(a.data) / name) != digest:
            raise ValueError("changed prepared data")
    provenance = {name: file_sha256(name) for name in SOURCES}
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = _resolve_device(a.device)
    torch.set_num_threads(4)
    seed_everything(20265710)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    foundation = torch.load(a.source, map_location="cpu", weights_only=False)
    for name, digest in foundation["provenance"].items():
        if file_sha256(name) != digest:
            raise ValueError("foundation source or protocol changed")
    model = model_from_foundation(foundation, device, copy_enabled=a.arm == "copy")
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    groups = []
    new = ("copy_query.", "copy_key.", "copy_gate.", "stop_head.")
    for prefix, rate in [("reader", 3e-5), ("density", 1e-4), ("heads", 3e-4)]:
        params = [
            v
            for k, v in model.named_parameters()
            if v.requires_grad
            and (
                (
                    "reader"
                    if k.startswith("reader.")
                    else "heads"
                    if k.startswith(new)
                    else "density"
                )
                == prefix
            )
        ]
        groups.append({"params": params, "peak_lr": rate})
    opt = torch.optim.AdamW(
        groups,
        lr=1e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    parameters = [v for v in model.parameters() if v.requires_grad]
    render = CanonicalGlyphRenderConfig(**foundation["render_config"])
    renderer = VisualQARenderer(render)
    questions = read_questions(Path(a.data) / "train.jsonl")
    manifest = verify_v25_manifest(a.manifest, strict=True)
    if manifest["sha256"] != foundation["manifest"]["sha256"]:
        raise ValueError("changed natural corpus")
    natural = CanonicalGlyphLanguageDataset(
        load_v25_records(a.manifest, strict_manifest=True),
        split="train",
        render_config=render,
        seed=20265711,
        length=a.steps * a.batch_size,
    )
    data_rng = torch.Generator().manual_seed(20265711)
    train_rng = torch.Generator(device=device).manual_seed(20265710)
    start = 0
    prior_elapsed = 0.0
    prior_peak = 0.0
    if a.resume:
        r = torch.load(a.resume, map_location="cpu", weights_only=False)
        if (
            r["provenance"] != provenance
            or r["arm"] != a.arm
            or r["smoke_only"] != a.smoke
            or r["arguments"]["steps"] != a.steps
            or r["arguments"]["batch_size"] != a.batch_size
            or r["data_sha256"] != DATA_HASHES
            or r["manifest"]["sha256"] != manifest["sha256"]
        ):
            raise ValueError("resume mismatch")
        model.load_state_dict(r["model"])
        ema.load_state_dict(r["ema_model"])
        opt.load_state_dict(r["optimizer"])
        data_rng.set_state(r["data_rng"].cpu())
        train_rng.set_state(r["training_rng"].cpu())
        torch.set_rng_state(r["cpu_rng"])
        random.setstate(r["python_rng"])
        np.random.set_state(r["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(r["cuda_rng"], device)
        start = r["update"]
        prior_elapsed = r["elapsed_seconds"]
        prior_peak = r["peak_allocated_vram_gib"]
    if start >= a.steps:
        raise ValueError("resume already complete")
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    handlers = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    begun = time.perf_counter()
    try:
        for update in range(start + 1, a.steps + 1):
            model.train()
            for group in opt.param_groups:
                group["lr"] = _scheduled_lr(
                    update,
                    base=group["peak_lr"],
                    warmup=min(200, a.steps // 2),
                    total=a.steps,
                    minimum_ratio=0.1,
                )
            if update % 4:
                indices = torch.randint(
                    len(questions), (a.batch_size,), generator=data_rng
                ).tolist()
                batch = pack_qa([questions[i] for i in indices], renderer, device)
                task = "qa"
            else:
                batch = natural_batch(natural, update, a.batch_size, device, train_rng)
                task = "natural_motor"
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, "bf16"):
                condition = model.visual_condition(
                    batch["sequence"], batch["prompt_lengths"], batch["query_positions"]
                )
                nll = model.event_nll(
                    condition,
                    batch["targets"],
                    batch["continued"],
                    batch["stopped"],
                    motor=batch.get("motor"),
                )
                valid = batch["continued"] | batch["stopped"]
                loss = nll[valid].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite exact event NLL")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, 1, error_if_nonfinite=True
            )
            opt.step()
            with torch.no_grad():
                for ep, v in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(v, 0.001)
                for eb, v in zip(ema.buffers(), model.buffers()):
                    eb.copy_(v)
            elapsed = prior_elapsed + time.perf_counter() - begun
            peak = max(
                prior_peak,
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0,
            )
            metrics = {
                "update": update,
                "arm": a.arm,
                "task": task,
                "event_nll": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
                "elapsed_seconds": elapsed,
                "peak_allocated_vram_gib": peak,
            }
            if update == 1 or update % 50 == 0 or a.smoke:
                with (out / "training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
            resource_exceeded = peak >= 16 or elapsed > 4 * 3600
            stopping |= resource_exceeded or (
                a.stop_after is not None and update >= a.stop_after
            )
            if update % 500 == 0 or update == a.steps or stopping:
                payload = {
                    "architecture": ARCHITECTURE,
                    "arm": a.arm,
                    "model_config": asdict(model.config),
                    "reader_config": asdict(model.reader.config),
                    "render_config": asdict(render),
                    "model": model.state_dict(),
                    "ema_model": ema.state_dict(),
                    "optimizer": opt.state_dict(),
                    "data_rng": data_rng.get_state(),
                    "training_rng": train_rng.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "arguments": vars(a),
                    "update": update,
                    "manifest": manifest,
                    "data_sha256": DATA_HASHES,
                    "provenance": provenance,
                    "initialization_sha256": SOURCE_HASH,
                    "boundary": model.boundary_receipt(),
                    "elapsed_seconds": elapsed,
                    "peak_allocated_vram_gib": peak,
                    "training_metrics": metrics,
                    "smoke_only": a.smoke,
                    "qualification": "unqualified research checkpoint",
                }
                _atomic_save(payload, out / "checkpoint_latest.pt")
                if update == a.steps and not resource_exceeded:
                    _atomic_save(payload, out / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            k: payload[k]
                            for k in (
                                "architecture",
                                "arm",
                                "update",
                                "data_sha256",
                                "initialization_sha256",
                                "provenance",
                                "boundary",
                                "elapsed_seconds",
                                "peak_allocated_vram_gib",
                                "smoke_only",
                                "qualification",
                            )
                        },
                        out / "training_report.json",
                    )
                if resource_exceeded:
                    raise RuntimeError("resource limit reached; checkpoint saved")
                if stopping:
                    break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
