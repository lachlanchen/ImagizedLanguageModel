#!/usr/bin/env python3
"""Train matched source-image readouts of frozen visual-language features."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from ilm.visual_lm.frozen_visual_foundation import FrozenFeatureBlockHead
from ilm.visual_lm.frozen_visual_readout_data import FeatureQuestionData, sha256
from scripts.train_canonical_glyph_binding_v43 import (
    _atomic_json,
    _atomic_save,
    _resolve_device,
    _scheduled_lr,
)

ARCHITECTURE = "frozen-visual-foundation-readout-v59"
DEFAULT_CACHE = "artifacts/frozen_visual_foundation_v59_20261105/features"
SOURCES = [
    "ilm/visual_lm/frozen_visual_foundation.py",
    "ilm/visual_lm/frozen_visual_readout_data.py",
    "scripts/train_frozen_visual_readout_v59.py",
    "references/frozen_visual_foundation_v59_protocol.md",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--encoder", choices=["student", "mixar"], required=True)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--data", default="data/visual_qa/cmrc2018_short")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    steps, batch_size = (4, 2) if a.smoke else (4000, 8)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoint_final.pt").exists():
        raise FileExistsError("completed readout exists")
    cache = Path(a.cache)
    cache_report = json.loads((cache / "report.json").read_text())
    if cache_report["smoke_only"] and not a.smoke:
        raise ValueError("smoke cache excluded from production")
    for name, digest in cache_report["sources"].items():
        if sha256(name) != digest:
            raise ValueError("frozen cache source changed")
    if not a.smoke and cache_report["counts"] != {
        "train": 3211,
        "development": 1321,
        "development_blank": 1321,
    }:
        raise ValueError("incomplete production features")
    provenance = {name: sha256(name) for name in SOURCES}
    cache_sha = sha256(cache / "report.json")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    random.seed(20265910)
    np.random.seed(20265910)
    torch.manual_seed(20265910)
    device = _resolve_device(a.device)
    if device.type == "cuda":
        torch.cuda.manual_seed(20265910)
        torch.cuda.reset_peak_memory_stats(device)
    model = FrozenFeatureBlockHead().to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.05,
        fused=device.type == "cuda",
    )
    dataset = FeatureQuestionData(cache, a.data, a.encoder)
    questions = dataset.records("train")
    generator = torch.Generator().manual_seed(20265911)
    start = 0
    prior_elapsed = 0.0
    prior_peak = 0.0
    if a.resume:
        r = torch.load(a.resume, map_location="cpu", weights_only=False)
        if (
            r["architecture"] != ARCHITECTURE
            or r["encoder"] != a.encoder
            or r["provenance"] != provenance
            or r["cache_sha256"] != cache_sha
            or r["smoke_only"] != a.smoke
        ):
            raise ValueError("resume identity mismatch")
        model.load_state_dict(r["model"])
        ema.load_state_dict(r["ema_model"])
        optimizer.load_state_dict(r["optimizer"])
        generator.set_state(r["data_rng"])
        torch.set_rng_state(r["cpu_rng"])
        random.setstate(r["python_rng"])
        np.random.set_state(r["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(r["cuda_rng"], device)
        start = r["update"]
        prior_elapsed = r["elapsed_seconds"]
        prior_peak = r["peak_allocated_vram_gib"]
    if start >= steps:
        raise ValueError("resume already complete")
    stopping = False

    def stop(_sig, _frame):
        nonlocal stopping
        stopping = True

    handlers = {
        sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    begun = time.perf_counter()
    try:
        for update in range(start + 1, steps + 1):
            optimizer.param_groups[0]["lr"] = _scheduled_lr(
                update,
                base=3e-4,
                warmup=min(200, steps // 2),
                total=steps,
                minimum_ratio=0.1,
            )
            indices = torch.randint(
                len(questions), (batch_size,), generator=generator
            ).tolist()
            features, images, lengths, targets, target_lengths = dataset.batch(
                [questions[i] for i in indices], device
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logp = model(features, lengths)
            nll = model.block_nll(logp, images, lengths, targets, target_lengths)
            loss = nll.sum() / (target_lengths.sum() + batch_size)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite block likelihood")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for ep, v in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(v, 0.001)
            elapsed = prior_elapsed + time.perf_counter() - begun
            peak = max(
                prior_peak,
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0,
            )
            metrics = {
                "encoder": a.encoder,
                "update": update,
                "training_nll_per_reference_event": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
                "elapsed_seconds": elapsed,
                "peak_allocated_vram_gib": peak,
            }
            if update == 1 or update % 50 == 0 or a.smoke:
                with (out / "training_metrics.jsonl").open("a") as f:
                    f.write(json.dumps(metrics) + "\n")
                print(json.dumps(metrics), flush=True)
            exceeded = peak >= 16 or elapsed > 4 * 3600
            stopping |= exceeded or (
                a.stop_after is not None and update >= a.stop_after
            )
            if update % 500 == 0 or update == steps or stopping:
                r = {
                    "architecture": ARCHITECTURE,
                    "encoder": a.encoder,
                    "model": model.state_dict(),
                    "ema_model": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "data_rng": generator.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "provenance": provenance,
                    "cache_sha256": cache_sha,
                    "feature_cache_identity": json.loads(
                        (cache / "identity.json").read_text()
                    ),
                    "arguments": vars(a),
                    "update": update,
                    "steps": steps,
                    "batch_size": batch_size,
                    "elapsed_seconds": elapsed,
                    "peak_allocated_vram_gib": peak,
                    "smoke_only": a.smoke,
                    "trained_parameters": sum(v.numel() for v in model.parameters()),
                    "boundary": {
                        "input": "continuous features derived from prompt pixels only",
                        "output": "observed image block; no persistent glyph bank",
                        "encoder_frozen": True,
                        "tokenizer": False,
                        "unicode_ids": False,
                        "ocr": False,
                        "external_model_call": False,
                    },
                    "qualification": "frozen-foundation reading probe; general ILM remains unqualified",
                }
                _atomic_save(r, out / "checkpoint_latest.pt")
                if update == steps and not exceeded:
                    _atomic_save(r, out / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            k: r[k]
                            for k in [
                                "architecture",
                                "encoder",
                                "provenance",
                                "cache_sha256",
                                "update",
                                "steps",
                                "batch_size",
                                "elapsed_seconds",
                                "peak_allocated_vram_gib",
                                "smoke_only",
                                "trained_parameters",
                                "boundary",
                                "qualification",
                            ]
                        },
                        out / "training_report.json",
                    )
                if exceeded:
                    raise RuntimeError("resource ceiling; resumable checkpoint saved")
                if stopping:
                    break
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
