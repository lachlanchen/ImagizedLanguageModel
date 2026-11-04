#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.causal_visual_block_diffusion_v53 import (
    V53_ARCHITECTURE,
    CausalVisualBlockDiffusion,
    CausalVisualBlockDiffusionConfig,
    ConditionalStripDenoiser,
    causal_visual_block_boundary_receipt,
    causal_visual_block_config_payload,
)
from ilm.visual_lm.causal_visual_block_diffusion_v53_training import (
    stratified_future_timesteps,
    visual_block_diffusion_loss,
)
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.visual_cell_data import load_v25_records, verify_v25_manifest
from ilm.visual_lm.visual_posterior_language_v51 import (
    VisualGlyphScorePrior,
    visual_posterior_language_config_from_payload,
)
from scripts.train_canonical_glyph_binding_v43 import (
    _append_jsonl,
    _atomic_json,
    _atomic_save,
    _resolve_device,
    _scheduled_lr,
)
from scripts.train_noisy_visual_energy_guidance_v52 import (
    DEFAULT_MANIFEST,
    DEFAULT_PRIOR,
    DEFAULT_READER,
    REQUIRED_PRIOR_SHA256,
    REQUIRED_READER_SHA256,
)
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)

DEFAULT_OUTPUT = "artifacts/causal_visual_block_diffusion_v53_20261104"
PROTOCOL_FILES = (
    "references/causal_visual_block_diffusion_v53_protocol.md",
    "references/v53_pretraining_corrections_and_article_plan.md",
)
SOURCE_FILES = (
    "ilm/visual_lm/causal_visual_block_diffusion_v53.py",
    "ilm/visual_lm/causal_visual_block_diffusion_v53_training.py",
    "scripts/train_causal_visual_block_diffusion_v53.py",
)
FIXED = {"steps": 8000, "batch_size": 8, "seed": 20265310, "dataset_seed": 20265311}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train directly conditional visual block diffusion."
    )
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--prior-checkpoint", default=DEFAULT_PRIOR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=FIXED["steps"])
    parser.add_argument("--batch-size", type=int, default=FIXED["batch_size"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=FIXED["seed"])
    parser.add_argument("--dataset-seed", type=int, default=FIXED["dataset_seed"])
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument(
        "--stop-after", type=int, help="Save and stop after this absolute update."
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_frozen_reader(path: str, device: torch.device) -> CausalRasterReader:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V53 requires the V42 reader")
    reader = CausalRasterReader(
        canonical_glyph_language_config_from_payload(payload["model_config"])
    )
    reader.load_v42_state(payload["model"])
    return reader.to(device).requires_grad_(False).eval()


def _hashes(paths: tuple[str, ...]) -> dict[str, str]:
    return {path: file_sha256(path) for path in paths}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.batch_size = min(args.batch_size, 2)
        args.num_workers = 0
        args.log_every = 1
    elif any(getattr(args, key) != value for key, value in FIXED.items()):
        raise ValueError(f"production V53 requires {FIXED}")
    if min(args.steps, args.batch_size, args.log_every, args.save_every) < 1:
        raise ValueError("training sizes must be positive")
    out = Path(args.out)
    if (out / "checkpoint_final.pt").exists():
        raise FileExistsError("completed training output already exists")
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    reader_hash = file_sha256(args.reader_checkpoint)
    prior_hash = file_sha256(args.prior_checkpoint)
    if reader_hash != REQUIRED_READER_SHA256 or prior_hash != REQUIRED_PRIOR_SHA256:
        raise ValueError("V53 frozen foundation hash mismatch")
    manifest = verify_v25_manifest(args.manifest, strict=True)
    provenance = {"source": _hashes(SOURCE_FILES), "protocol": _hashes(PROTOCOL_FILES)}
    reader = load_frozen_reader(args.reader_checkpoint, device)
    denoiser = ConditionalStripDenoiser(CausalVisualBlockDiffusionConfig()).to(device)
    prior_payload = torch.load(
        args.prior_checkpoint, map_location="cpu", weights_only=False
    )
    prior = VisualGlyphScorePrior(
        visual_posterior_language_config_from_payload(prior_payload["model_config"])
    )
    prior.load_state_dict(prior_payload["ema_model"], strict=True)
    denoiser.initialize_from_prior(prior.to(device))
    del prior, prior_payload
    model = CausalVisualBlockDiffusion(reader, denoiser).to(device)
    ema = copy.deepcopy(denoiser).requires_grad_(False).eval()
    transferred_prefixes = (
        "time_embed.",
        "in_conv.",
        "downs.",
        "mid1.",
        "ups.",
        "out_conv.",
    )
    transferred, new = [], []
    for name, parameter in denoiser.named_parameters():
        (transferred if name.startswith(transferred_prefixes) else new).append(
            parameter
        )
    optimizer = torch.optim.AdamW(
        [
            {"params": transferred, "base_lr": 5e-5},
            {"params": new, "base_lr": 2e-4},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=device.type == "cuda",
    )
    boundary = causal_visual_block_boundary_receipt(model)
    if (
        boundary["denoiser_parameters"] > 8_000_000
        or boundary["combined_parameters"] > 26_000_000
    ):
        raise ValueError("V53 exceeds parameter cap")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    start, elapsed_before, peak_before = 0, 0.0, 0.0
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            resume["architecture"] != V53_ARCHITECTURE
            or resume["provenance"] != provenance
        ):
            raise ValueError("resume architecture/source/protocol mismatch")
        if (
            resume["reader_checkpoint_sha256"] != reader_hash
            or resume["prior_checkpoint_sha256"] != prior_hash
        ):
            raise ValueError("resume foundation mismatch")
        for key in (*FIXED, "smoke"):
            if resume["arguments"][key] != getattr(args, key):
                raise ValueError(f"resume argument mismatch: {key}")
        if resume["manifest"]["sha256"] != manifest["sha256"]:
            raise ValueError("resume corpus mismatch")
        denoiser.load_state_dict(resume["denoiser"], strict=True)
        ema.load_state_dict(resume["ema_denoiser"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        generator.set_state(resume["training_generator_state"].cpu())
        start = int(resume["update"])
        elapsed_before = resume["elapsed_seconds"]
        peak_before = resume["peak_allocated_vram_gib"]
    if start >= args.steps:
        raise ValueError("resume has already reached the update limit")
    records = load_v25_records(args.manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig()
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render,
        seed=args.dataset_seed,
        length=args.steps * args.batch_size,
    )
    loader = DataLoader(
        Subset(dataset, range(start * args.batch_size, len(dataset))),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=canonical_glyph_collate,
        generator=torch.Generator().manual_seed(args.dataset_seed),
    )
    iterator = iter(loader)
    if resume is not None:
        random.setstate(resume["python_rng"])
        np.random.set_state(resume["numpy_rng"])
        torch.set_rng_state(resume["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"], device)
    stopped = False

    def request_stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    started = time.perf_counter()
    final_metrics = {}
    try:
        for update in range(start + 1, args.steps + 1):
            model.train()
            for group in optimizer.param_groups:
                group["lr"] = _scheduled_lr(
                    update,
                    base=group["base_lr"],
                    warmup=min(400, args.steps // 2),
                    total=args.steps,
                    minimum_ratio=0.10,
                )
            raw = next(iterator)
            images = raw["context"].to(device, non_blocking=True)
            context, targets = images[:, :60], images[:, 60:64]
            shuffled = _shuffle_prefix(
                context, first_index=(update - 1) * args.batch_size
            )
            times = stratified_future_timesteps(
                len(context),
                device=device,
                generator=generator,
                synchronous=update % 2 == 0,
            )
            noise = torch.randn(targets.shape, device=device, generator=generator)
            drop = (
                torch.rand(len(context), device=device, generator=generator)
                < model.config.condition_dropout
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, "bf16"):
                loss = visual_block_diffusion_loss(
                    model,
                    context=context,
                    shuffled_context=shuffled,
                    target_cells=targets,
                    timesteps=times,
                    noise=noise,
                    drop_condition=drop,
                )
            if not bool(torch.isfinite(loss.loss)):
                raise FloatingPointError("V53 loss is not finite")
            loss.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                denoiser.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            with torch.no_grad():
                for target, source in zip(ema.parameters(), denoiser.parameters()):
                    target.lerp_(source, 0.001)
                for target, source in zip(ema.buffers(), denoiser.buffers()):
                    target.copy_(source)
            if any(p.grad is not None for p in reader.parameters()):
                raise RuntimeError("frozen reader received gradients")
            peak = max(
                peak_before,
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0,
            )
            if peak >= 18.0:
                raise MemoryError("V53 exceeds registered memory limit")
            final_metrics = loss.detached_metrics() | {
                "update": update,
                "gradient_norm": float(norm),
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "synchronous_noise": update % 2 == 0,
            }
            if update == 1 or update % args.log_every == 0:
                _append_jsonl(out / "training_metrics.jsonl", final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            stopped |= args.stop_after is not None and update >= args.stop_after
            if update % args.save_every == 0 or update == args.steps or stopped:
                payload = {
                    "architecture": V53_ARCHITECTURE,
                    "model_config": causal_visual_block_config_payload(model.config),
                    "denoiser": denoiser.state_dict(),
                    "ema_denoiser": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "update": update,
                    "training_generator_state": generator.get_state(),
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device)
                    if device.type == "cuda"
                    else None,
                    "arguments": vars(args),
                    "render_config": vars(render),
                    "reader_checkpoint_sha256": reader_hash,
                    "prior_checkpoint_sha256": prior_hash,
                    "manifest": manifest,
                    "provenance": provenance,
                    "boundary": boundary,
                    "elapsed_seconds": final_metrics["elapsed_seconds"],
                    "peak_allocated_vram_gib": peak,
                    "training_metrics": final_metrics,
                    "smoke_only": args.smoke,
                    "initialization": "V51 epsilon weights; fine-tuned as conditional velocity",
                }
                _atomic_save(payload, out / "checkpoint_latest.pt")
                if update == args.steps:
                    _atomic_save(payload, out / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            key: payload[key]
                            for key in (
                                "architecture",
                                "update",
                                "elapsed_seconds",
                                "peak_allocated_vram_gib",
                                "boundary",
                                "training_metrics",
                                "smoke_only",
                                "provenance",
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
