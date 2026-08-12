#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm import ConditionalVisualFlow, VisualFlowConfig, VisualPageVAE, VisualVAEConfig
from ilm.visual_lm.autoencoder import sobel_edges
from ilm.visual_lm.dataset import tensor_to_pil, visual_collate
from ilm.visual_lm.flow import sample_heun
from ilm.visual_lm.rendering import make_triptych
from scripts.train_image_native_lm import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ILM visual-flow checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="artifacts/ilm_eval")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def edge_f1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction_edge = sobel_edges(prediction.float())
    target_edge = sobel_edges(target.float())
    prediction_mask = prediction_edge > prediction_edge.flatten(1).quantile(0.80, dim=1)[:, None, None, None]
    target_mask = target_edge > target_edge.flatten(1).quantile(0.80, dim=1)[:, None, None, None]
    true_positive = (prediction_mask & target_mask).flatten(1).sum(1).float()
    precision = true_positive / prediction_mask.flatten(1).sum(1).clamp_min(1)
    recall = true_positive / target_mask.flatten(1).sum(1).clamp_min(1)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).mean())


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = argparse.Namespace(**checkpoint["args"])
    saved_args.batch_size = args.batch_size
    saved_args.num_workers = 0
    _, validation_dataset, _ = build_datasets(saved_args)
    loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=visual_collate,
    )

    vae = VisualPageVAE(VisualVAEConfig.from_dict(checkpoint["vae_config"]))
    flow = ConditionalVisualFlow(VisualFlowConfig.from_dict(checkpoint["flow_config"]))
    vae.load_state_dict(checkpoint["vae"])
    flow.load_state_dict(checkpoint.get("flow_ema", checkpoint["flow"]))
    vae.to(device).eval().requires_grad_(False)
    flow.to(device).eval().requires_grad_(False)
    output = Path(args.out)
    sample_dir = output / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    totals = {"l1": 0.0, "mse": 0.0, "edge_f1": 0.0, "vae_reconstruction_l1": 0.0}
    count = 0
    latencies: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if count >= args.samples:
            break
        prompt = batch["prompt"].to(device)
        target = batch["target"].to(device)
        remaining = args.samples - count
        prompt = prompt[:remaining]
        target = target[:remaining]
        generator = torch.Generator(device=device).manual_seed(args.seed + count)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode(), autocast_context(device, args.precision):
            condition = vae.encode(prompt, sample=False, normalize=True)
            latent = sample_heun(
                flow,
                condition,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )
            prediction = vae.decode(latent, normalized=True)
            reconstruction, _ = vae(target, sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started) / prompt.shape[0])
        batch_size = prompt.shape[0]
        totals["l1"] += float(F.l1_loss(prediction, target)) * batch_size
        totals["mse"] += float(F.mse_loss(prediction, target)) * batch_size
        totals["edge_f1"] += edge_f1(prediction, target) * batch_size
        totals["vae_reconstruction_l1"] += float(F.l1_loss(reconstruction, target)) * batch_size
        for offset in range(batch_size):
            triptych = make_triptych(
                tensor_to_pil(prompt[offset]),
                tensor_to_pil(prediction[offset]),
                tensor_to_pil(target[offset]),
            )
            triptych.save(sample_dir / f"sample_{count + offset:04d}.png")
        count += batch_size

    metrics = {key: value / max(1, count) for key, value in totals.items()}
    metrics["psnr"] = -10.0 * math.log10(max(metrics["mse"] / 4.0, 1e-12))
    metrics.update(
        {
            "samples": count,
            "sampling_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "latency_seconds_per_image_mean": sum(latencies) / max(1, len(latencies)),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "scope_warning": "Pixel metrics do not establish semantic correctness or general Qwen-8B parity.",
        }
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
