#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ilm.visual_lm import (
    ConditionalVisualFlow,
    GlyphCorpus,
    InstructionRenderConfig,
    MixedVisualDataset,
    RenderConfig,
    VisualFlowConfig,
    VisualInstructionDataset,
    VisualLanguageDataset,
    VisualPageVAE,
    VisualVAEConfig,
    load_alpaca_records,
)
from ilm.visual_lm.autoencoder import sobel_edges, visual_reconstruction_loss
from ilm.visual_lm.dataset import tensor_to_pil, visual_collate
from ilm.visual_lm.flow import (
    ExponentialMovingAverage,
    condition_keep_mask,
    flow_training_pair,
    sample_heun,
)
from ilm.visual_lm.rendering import make_triptych
from ilm.visual_lm.teacher import load_teacher_manifest


DEFAULT_HISTORICAL_CHARS = "言,中,水,日,月,人,山,火,木,口,学,車,车,王,雨,田,金"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the independent image-native language model on writing-image pairs."
    )
    data = parser.add_argument_group("visual data")
    data.add_argument("--zh-data", default="data/raw/alpaca_zh.json")
    data.add_argument("--en-data", default="data/raw/alpaca_en.json")
    data.add_argument("--disable-zh", action="store_true")
    data.add_argument("--disable-en", action="store_true")
    data.add_argument("--max-records-per-language", type=int, default=20_000)
    data.add_argument("--max-prompt-chars", type=int, default=120)
    data.add_argument("--max-response-chars", type=int, default=240)
    data.add_argument("--glyph-root", default=None)
    data.add_argument("--teacher-manifest", default="data/teacher/historical_qwen8b_v2.jsonl")
    data.add_argument("--historical-chars", default=DEFAULT_HISTORICAL_CHARS)
    data.add_argument("--historical-weight", type=float, default=0.20)
    data.add_argument("--validation-fraction", type=float, default=0.02)
    data.add_argument("--image-size", type=int, default=384)
    data.add_argument("--dataset-seed", type=int, default=43)

    model = parser.add_argument_group("model")
    model.add_argument("--latent-channels", type=int, default=8)
    model.add_argument("--vae-base-channels", type=int, default=32)
    model.add_argument("--vae-channel-multipliers", default="1,2,4,4")
    model.add_argument("--vae-blocks-per-level", type=int, default=2)
    model.add_argument("--flow-base-channels", type=int, default=64)
    model.add_argument("--flow-channel-multipliers", default="1,2,4")
    model.add_argument("--flow-blocks-per-level", type=int, default=2)
    model.add_argument("--flow-time-channels", type=int, default=256)
    model.add_argument("--condition-dropout", type=float, default=0.10)

    train = parser.add_argument_group("optimization")
    train.add_argument("--out", default="artifacts/ilm_first_proof")
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--gradient-accumulation", type=int, default=1)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--vae-steps", type=int, default=5_000)
    train.add_argument("--flow-steps", type=int, default=20_000)
    train.add_argument("--vae-lr", type=float, default=2e-4)
    train.add_argument("--flow-lr", type=float, default=2e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--edge-weight", type=float, default=0.20)
    train.add_argument("--multiscale-weight", type=float, default=0.15)
    train.add_argument("--kl-weight", type=float, default=1e-6)
    train.add_argument("--ema-decay", type=float, default=0.999)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--save-every", type=int, default=250)
    train.add_argument("--sample-every", type=int, default=250)
    train.add_argument("--sample-steps", type=int, default=12)
    train.add_argument("--guidance-scale", type=float, default=2.0)
    train.add_argument("--calibration-batches", type=int, default=32)
    train.add_argument("--resume", default=None)
    train.add_argument("--skip-vae-training", action="store_true")
    return parser.parse_args()


def integer_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or any(item < 1 for item in result):
        raise ValueError(f"invalid channel multipliers: {value!r}")
    return result


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def endless(loader: DataLoader) -> Iterator[dict[str, Any]]:
    while True:
        yield from loader


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_metrics(path: Path, values: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, ensure_ascii=False, sort_keys=True) + "\n")


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset, dict[str, Any]]:
    train_sets: list[Dataset] = []
    validation_sets: list[Dataset] = []
    train_weights: list[float] = []
    provenance: dict[str, Any] = {"instruction_sources": [], "historical": None}
    instruction_config = InstructionRenderConfig(image_size=args.image_size, augment=True)
    validation_config = InstructionRenderConfig(image_size=args.image_size, augment=False)

    sources = []
    if not args.disable_zh:
        sources.append((args.zh_data, "zh", "GPT-4-LLM alpaca_gpt4_data_zh", "CC-BY-NC-4.0"))
    if not args.disable_en:
        sources.append((args.en_data, "en", "Stanford Alpaca", "CC-BY-NC-4.0"))
    for path_value, language, source_name, license_name in sources:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: PYTHONPATH=. python scripts/download_alpaca.py --outdir data/raw"
            )
        records = load_alpaca_records(
            path,
            language=language,
            source=source_name,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            limit=args.max_records_per_language,
        )
        train_sets.append(
            VisualInstructionDataset(
                records,
                render_config=instruction_config,
                length=max(1_024, len(records)),
                seed=args.dataset_seed,
                split="train",
                validation_fraction=args.validation_fraction,
            )
        )
        validation_sets.append(
            VisualInstructionDataset(
                records,
                render_config=validation_config,
                length=max(16, min(128, len(records))),
                seed=args.dataset_seed + 50_000,
                split="validation",
                validation_fraction=args.validation_fraction,
                script_augmentation=False,
            )
        )
        train_weights.append(1.0)
        provenance["instruction_sources"].append(
            {
                "path": str(path),
                "source": source_name,
                "language": language,
                "license": license_name,
                "accepted_records": len(records),
            }
        )

    if args.historical_weight > 0:
        try:
            characters = [item.strip() for item in args.historical_chars.split(",") if item.strip()]
            seed_corpus = GlyphCorpus(args.glyph_root)
            characters = [char for char in characters if (seed_corpus.root / char).exists()]
            corpus = GlyphCorpus(seed_corpus.root, characters=characters)
            teacher_records = load_teacher_manifest(args.teacher_manifest)
            historical_train = VisualLanguageDataset(
                corpus,
                render_config=RenderConfig(image_size=args.image_size),
                length=max(1_024, len(characters) * 64),
                seed=args.dataset_seed + 100_000,
                characters=characters,
                teacher_records=teacher_records,
            )
            historical_validation = VisualLanguageDataset(
                corpus,
                render_config=RenderConfig(image_size=args.image_size),
                length=max(16, len(characters) * 2),
                seed=args.dataset_seed + 150_000,
                characters=characters,
                teacher_records=teacher_records,
            )
            train_sets.append(historical_train)
            validation_sets.append(historical_validation)
            train_weights.append(args.historical_weight)
            provenance["historical"] = {
                "root": str(corpus.root),
                "characters": characters,
                "teacher_manifest": args.teacher_manifest,
                "teacher_records": len(teacher_records),
                "license": "local research data; redistribution license not verified",
            }
        except FileNotFoundError:
            if not train_sets:
                raise
            provenance["historical"] = {"disabled": "glyph root not found"}

    if not train_sets:
        raise ValueError("No training source enabled")
    total_length = max(args.batch_size * 256, sum(len(dataset) for dataset in train_sets))
    train_dataset = MixedVisualDataset(
        train_sets,
        weights=train_weights,
        length=total_length,
        seed=args.dataset_seed,
    )
    validation_dataset = MixedVisualDataset(
        validation_sets,
        weights=[1.0] * len(validation_sets),
        length=max(32, sum(min(64, len(dataset)) for dataset in validation_sets)),
        seed=args.dataset_seed + 200_000,
    )
    return train_dataset, validation_dataset, provenance


@torch.no_grad()
def calibrate_latents(
    vae: VisualPageVAE,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    batches: int,
) -> dict[str, list[float]]:
    vae.eval()
    channel_sum = torch.zeros(vae.config.latent_channels, device=device, dtype=torch.float64)
    channel_square_sum = torch.zeros_like(channel_sum)
    count = 0
    iterator = endless(loader)
    for _ in range(max(1, batches)):
        batch = next(iterator)
        images = torch.cat((batch["prompt"], batch["target"]), dim=0).to(device)
        with autocast_context(device, precision):
            latent = vae.encode(images, sample=False, normalize=False)
        latent = latent.double()
        channel_sum += latent.sum(dim=(0, 2, 3))
        channel_square_sum += latent.square().sum(dim=(0, 2, 3))
        count += latent.shape[0] * latent.shape[2] * latent.shape[3]
    mean = channel_sum / max(1, count)
    variance = channel_square_sum / max(1, count) - mean.square()
    std = variance.clamp_min(1e-8).sqrt()
    vae.set_latent_statistics(mean.float(), std.float())
    return {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()}


@torch.no_grad()
def validation_metrics(
    vae: VisualPageVAE,
    flow: ConditionalVisualFlow | None,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    maximum_batches: int = 8,
) -> dict[str, float]:
    vae.eval()
    if flow is not None:
        flow.eval()
    totals = {"vae_l1": 0.0, "vae_edge_l1": 0.0, "flow_velocity_mse": 0.0}
    examples = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        prompt = batch["prompt"].to(device)
        target = batch["target"].to(device)
        with autocast_context(device, precision):
            reconstructed, _ = vae(target, sample=False)
            totals["vae_l1"] += float(F.l1_loss(reconstructed, target)) * target.shape[0]
            totals["vae_edge_l1"] += float(
                F.l1_loss(sobel_edges(reconstructed), sobel_edges(target))
            ) * target.shape[0]
            if flow is not None:
                condition = vae.encode(prompt, sample=False, normalize=True)
                data = vae.encode(target, sample=False, normalize=True)
                state, velocity, time_value, _ = flow_training_pair(data)
                prediction = flow(state, time_value, condition)
                totals["flow_velocity_mse"] += float(F.mse_loss(prediction, velocity)) * target.shape[0]
        examples += target.shape[0]
    return {key: value / max(1, examples) for key, value in totals.items()}


@torch.no_grad()
def save_generated_sample(
    vae: VisualPageVAE,
    flow: ConditionalVisualFlow,
    loader: DataLoader,
    *,
    output: Path,
    device: torch.device,
    precision: str,
    steps: int,
    guidance_scale: float,
    seed: int,
) -> None:
    vae.eval()
    flow.eval()
    batch = next(iter(loader))
    prompt = batch["prompt"][:1].to(device)
    target = batch["target"][:1].to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with autocast_context(device, precision):
        condition = vae.encode(prompt, sample=False, normalize=True)
        generated_latent = sample_heun(
            flow,
            condition,
            steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        generated = vae.decode(generated_latent, normalized=True)
    image = make_triptych(
        tensor_to_pil(prompt[0]),
        tensor_to_pil(generated[0]),
        tensor_to_pil(target[0]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def checkpoint_payload(
    *,
    args: argparse.Namespace,
    vae: VisualPageVAE,
    flow: ConditionalVisualFlow,
    ema: ExponentialMovingAverage,
    vae_optimizer: torch.optim.Optimizer,
    flow_optimizer: torch.optim.Optimizer,
    vae_step: int,
    flow_step: int,
    provenance: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "ilm-visual-flow-v1",
        "args": vars(args),
        "vae_config": vae.config.to_dict(),
        "flow_config": flow.config.to_dict(),
        "vae": vae.state_dict(),
        "flow": flow.state_dict(),
        "flow_ema": ema.state_dict(),
        "vae_optimizer": vae_optimizer.state_dict(),
        "flow_optimizer": flow_optimizer.state_dict(),
        "vae_step": vae_step,
        "flow_step": flow_step,
        "provenance": provenance,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("batch size and gradient accumulation must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"

    train_dataset, validation_dataset, provenance = build_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=visual_collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=visual_collate,
    )

    vae_config = VisualVAEConfig(
        latent_channels=args.latent_channels,
        base_channels=args.vae_base_channels,
        channel_multipliers=integer_tuple(args.vae_channel_multipliers),
        blocks_per_level=args.vae_blocks_per_level,
    )
    if args.image_size % vae_config.downsample_factor:
        raise ValueError(
            f"image size must be divisible by VAE factor {vae_config.downsample_factor}"
        )
    flow_config = VisualFlowConfig(
        latent_channels=args.latent_channels,
        base_channels=args.flow_base_channels,
        channel_multipliers=integer_tuple(args.flow_channel_multipliers),
        blocks_per_level=args.flow_blocks_per_level,
        time_channels=args.flow_time_channels,
        condition_dropout=args.condition_dropout,
    )
    vae = VisualPageVAE(vae_config).to(device)
    flow = ConditionalVisualFlow(flow_config).to(device)
    ema = ExponentialMovingAverage(flow, decay=args.ema_decay)
    vae_optimizer = torch.optim.AdamW(
        vae.parameters(), lr=args.vae_lr, weight_decay=args.weight_decay
    )
    flow_optimizer = torch.optim.AdamW(
        flow.parameters(), lr=args.flow_lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.precision == "fp16"
    )
    vae_step = 0
    flow_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        vae.load_state_dict(checkpoint["vae"])
        flow.load_state_dict(checkpoint["flow"])
        ema.load_state_dict(checkpoint.get("flow_ema", checkpoint["flow"]))
        if "vae_optimizer" in checkpoint:
            vae_optimizer.load_state_dict(checkpoint["vae_optimizer"])
        if "flow_optimizer" in checkpoint:
            flow_optimizer.load_state_dict(checkpoint["flow_optimizer"])
        vae_step = int(checkpoint.get("vae_step", 0))
        flow_step = int(checkpoint.get("flow_step", 0))

    run_summary = {
        "device": str(device),
        "precision": args.precision,
        "vae_parameters": parameter_count(vae),
        "flow_parameters": parameter_count(flow),
        "total_parameters": parameter_count(vae) + parameter_count(flow),
        "image_size": args.image_size,
        "latent_size": args.image_size // vae_config.downsample_factor,
        "provenance": provenance,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), **run_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, ensure_ascii=False))

    train_iterator = endless(train_loader)
    start_time = time.perf_counter()
    if not args.skip_vae_training:
        vae.train()
        while vae_step < args.vae_steps:
            vae_optimizer.zero_grad(set_to_none=True)
            aggregated: dict[str, float] = {}
            for _ in range(args.gradient_accumulation):
                batch = next(train_iterator)
                images = torch.cat((batch["prompt"], batch["target"]), dim=0).to(
                    device, non_blocking=True
                )
                with autocast_context(device, args.precision):
                    reconstruction, posterior = vae(images, sample=True)
                    loss, parts = visual_reconstruction_loss(
                        reconstruction,
                        images,
                        posterior,
                        edge_weight=args.edge_weight,
                        multiscale_weight=args.multiscale_weight,
                        kl_weight=args.kl_weight,
                    )
                    scaled_loss = loss / args.gradient_accumulation
                scaler.scale(scaled_loss).backward()
                for key, value in parts.items():
                    aggregated[key] = aggregated.get(key, 0.0) + float(value.detach()) / args.gradient_accumulation
                aggregated["loss"] = aggregated.get("loss", 0.0) + float(loss.detach()) / args.gradient_accumulation
            scaler.unscale_(vae_optimizer)
            torch.nn.utils.clip_grad_norm_(vae.parameters(), args.grad_clip)
            scaler.step(vae_optimizer)
            scaler.update()
            vae_step += 1
            if vae_step == 1 or vae_step % args.log_every == 0:
                row = {
                    "stage": "vae",
                    "step": vae_step,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    **aggregated,
                }
                print(json.dumps(row))
                append_metrics(metrics_path, row)
            if vae_step % args.save_every == 0 or vae_step == args.vae_steps:
                metrics = validation_metrics(
                    vae,
                    None,
                    validation_loader,
                    device=device,
                    precision=args.precision,
                    maximum_batches=4,
                )
                payload = checkpoint_payload(
                    args=args,
                    vae=vae,
                    flow=flow,
                    ema=ema,
                    vae_optimizer=vae_optimizer,
                    flow_optimizer=flow_optimizer,
                    vae_step=vae_step,
                    flow_step=flow_step,
                    provenance=provenance,
                    metrics=metrics,
                )
                atomic_torch_save(payload, output_dir / "checkpoint_latest.pt")
                print(json.dumps({"stage": "vae-validation", "step": vae_step, **metrics}))

    print(json.dumps({"stage": "latent-calibration", "batches": args.calibration_batches}))
    latent_statistics = calibrate_latents(
        vae,
        train_loader,
        device=device,
        precision=args.precision,
        batches=args.calibration_batches,
    )
    append_metrics(metrics_path, {"stage": "latent-calibration", **latent_statistics})

    vae.eval().requires_grad_(False)
    flow.train()
    while flow_step < args.flow_steps:
        flow_optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            batch = next(train_iterator)
            prompt = batch["prompt"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.no_grad(), autocast_context(device, args.precision):
                condition = vae.encode(prompt, sample=False, normalize=True)
                data_latent = vae.encode(target, sample=False, normalize=True)
            with autocast_context(device, args.precision):
                state, target_velocity, time_value, _ = flow_training_pair(data_latent)
                keep = condition_keep_mask(
                    prompt.shape[0],
                    flow.config.condition_dropout,
                    device=device,
                    dtype=state.dtype,
                )
                predicted_velocity = flow(
                    state,
                    time_value,
                    condition,
                    condition_present=keep,
                )
                loss = F.mse_loss(predicted_velocity, target_velocity)
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach()) / args.gradient_accumulation
        scaler.unscale_(flow_optimizer)
        torch.nn.utils.clip_grad_norm_(flow.parameters(), args.grad_clip)
        scaler.step(flow_optimizer)
        scaler.update()
        ema.update(flow)
        flow_step += 1

        if flow_step == 1 or flow_step % args.log_every == 0:
            row = {
                "stage": "flow",
                "step": flow_step,
                "loss": accumulated_loss,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            if device.type == "cuda":
                row["peak_vram_bytes"] = torch.cuda.max_memory_allocated(device)
            print(json.dumps(row))
            append_metrics(metrics_path, row)

        should_save = flow_step % args.save_every == 0 or flow_step == args.flow_steps
        if flow_step % args.sample_every == 0 or flow_step == args.flow_steps:
            save_generated_sample(
                vae,
                ema.model,
                validation_loader,
                output=output_dir / "samples" / f"flow_{flow_step:07d}.png",
                device=device,
                precision=args.precision,
                steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed + flow_step,
            )
        if should_save:
            metrics = validation_metrics(
                vae,
                ema.model,
                validation_loader,
                device=device,
                precision=args.precision,
                maximum_batches=4,
            )
            payload = checkpoint_payload(
                args=args,
                vae=vae,
                flow=flow,
                ema=ema,
                vae_optimizer=vae_optimizer,
                flow_optimizer=flow_optimizer,
                vae_step=vae_step,
                flow_step=flow_step,
                provenance=provenance,
                metrics=metrics,
            )
            atomic_torch_save(payload, output_dir / "checkpoint_latest.pt")
            if flow_step == args.flow_steps:
                atomic_torch_save(payload, output_dir / f"checkpoint_flow_{flow_step}.pt")
            print(json.dumps({"stage": "flow-validation", "step": flow_step, **metrics}))

    final = {
        **run_summary,
        "vae_step": vae_step,
        "flow_step": flow_step,
        "elapsed_seconds": time.perf_counter() - start_time,
        "checkpoint": str(output_dir / "checkpoint_latest.pt"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    main()
