#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import signal
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Subset

from ilm.visual_lm.canonical_glyph_language import (
    canonical_glyph_language_config_from_payload,
)
from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphLanguageDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_collate,
    canonical_glyph_data_boundary_receipt,
    canonical_glyph_render_config_payload,
    canonical_glyph_student_batch,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.hierarchical_raster_language import CausalRasterReader
from ilm.visual_lm.noisy_visual_energy_guidance_v52 import (
    V52_ARCHITECTURE,
    NoisyVisualEnergyConfig,
    NoisyVisualEnergyCritic,
    noisy_visual_energy_boundary_receipt,
    noisy_visual_energy_config_from_payload,
    noisy_visual_energy_config_payload,
)
from ilm.visual_lm.noisy_visual_energy_guidance_v52_training import (
    noisy_visual_energy_loss,
)
from ilm.visual_lm.visual_cell_data import (
    V25_MANIFEST_SHA256,
    load_v25_records,
    verify_v25_manifest,
    visual_cell_partition_receipt,
)
from ilm.visual_lm.visual_posterior_language_v51 import (
    V51_ARCHITECTURE,
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
from scripts.train_visual_state_actuator import (
    autocast_context,
    file_sha256,
    seed_everything,
)


PROTOCOL_DOCUMENT = "references/noisy_visual_energy_guidance_v52_protocol.md"
DEFAULT_MANIFEST = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
DEFAULT_READER = "artifacts/canonical_glyph_language_v42_20260814/checkpoint_final.pt"
DEFAULT_PRIOR = (
    "artifacts/visual_posterior_language_v51_calibration_20261104/"
    "checkpoint_final.pt"
)
DEFAULT_OUTPUT = "artifacts/noisy_visual_energy_guidance_v52_20261104"
REQUIRED_READER_SHA256 = (
    "a5ce2ff20d0fc6d336f489b1e2d29bb96a1d0666399c2e2b95e7ac33a9abe870"
)
REQUIRED_PRIOR_SHA256 = (
    "ef607cb67c49da6d9bea32a784cb821a92ea546efbe8bf2226f8065171c00629"
)
SOURCE_FILES = (
    "ilm/visual_lm/noisy_visual_energy_guidance_v52.py",
    "ilm/visual_lm/noisy_visual_energy_guidance_v52_training.py",
    "scripts/train_noisy_visual_energy_guidance_v52.py",
)
FIXED_OPTIMIZATION = {
    "steps": 4_000,
    "batch_size": 8,
    "positions_per_stratum": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 200,
    "minimum_lr_ratio": 0.10,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "ema_decay": 0.999,
    "seed": 20265210,
    "dataset_seed": 20265211,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V52 noise-aware image-language energy critic."
    )
    parser.add_argument("--reader-checkpoint", default=DEFAULT_READER)
    parser.add_argument("--prior-checkpoint", default=DEFAULT_PRIOR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--resume")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--steps", type=int, default=FIXED_OPTIMIZATION["steps"])
    parser.add_argument(
        "--batch-size", type=int, default=FIXED_OPTIMIZATION["batch_size"]
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--positions-per-stratum",
        type=int,
        default=FIXED_OPTIMIZATION["positions_per_stratum"],
    )
    parser.add_argument(
        "--learning-rate", type=float, default=FIXED_OPTIMIZATION["learning_rate"]
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=FIXED_OPTIMIZATION["warmup_steps"]
    )
    parser.add_argument(
        "--minimum-lr-ratio",
        type=float,
        default=FIXED_OPTIMIZATION["minimum_lr_ratio"],
    )
    parser.add_argument(
        "--weight-decay", type=float, default=FIXED_OPTIMIZATION["weight_decay"]
    )
    parser.add_argument(
        "--gradient-clip", type=float, default=FIXED_OPTIMIZATION["gradient_clip"]
    )
    parser.add_argument(
        "--ema-decay", type=float, default=FIXED_OPTIMIZATION["ema_decay"]
    )
    parser.add_argument("--seed", type=int, default=FIXED_OPTIMIZATION["seed"])
    parser.add_argument(
        "--dataset-seed", type=int, default=FIXED_OPTIMIZATION["dataset_seed"]
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    return parser.parse_args()


def _effective_arguments(args: argparse.Namespace) -> dict[str, Any]:
    values = vars(args).copy()
    if args.smoke:
        values.update(
            {
                "steps": min(args.steps, 2),
                "batch_size": min(args.batch_size, 2),
                "positions_per_stratum": 1,
                "num_workers": 0,
                "warmup_steps": 1,
                "log_every": 1,
                "save_every": 1,
            }
        )
    return values


def _validate_fixed_arguments(arguments: Mapping[str, Any]) -> None:
    if arguments["smoke"] or arguments["exploratory"]:
        return
    for name, expected in FIXED_OPTIMIZATION.items():
        if arguments[name] != expected:
            raise ValueError(f"production V52 requires {name}={expected!r}")


def _selected_positions(
    batch_size: int,
    *,
    per_stratum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if batch_size < 1 or not 1 <= per_stratum <= 4:
        raise ValueError("V52 stratified position settings are invalid")
    rows: list[torch.Tensor] = []
    for _ in range(batch_size):
        selected: list[torch.Tensor] = []
        for start in (0, 16, 32):
            selected.append(
                torch.randperm(16, device=device, generator=generator)[
                    :per_stratum
                ]
                + start
            )
        if per_stratum == 1:
            final = torch.tensor([63], device=device)
        else:
            final = torch.cat(
                (
                    torch.randperm(15, device=device, generator=generator)[
                        : per_stratum - 1
                    ]
                    + 48,
                    torch.tensor([63], device=device),
                )
            )
        selected.append(final)
        rows.append(torch.cat(selected))
    return torch.stack(rows)


def _training_timestep(
    update: int,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> int:
    phase = (update - 1) % 4
    if phase == 0:
        return 0
    lower, upper = ((1, 30), (31, 70), (71, 99))[phase - 1]
    return int(
        torch.randint(
            lower,
            upper + 1,
            (),
            device=device,
            generator=generator,
        )
    )


@torch.no_grad()
def _update_ema(
    ema: NoisyVisualEnergyCritic,
    critic: NoisyVisualEnergyCritic,
    *,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), critic.parameters()):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), critic.buffers()):
        ema_buffer.copy_(buffer)


def _checkpoint(
    critic: NoisyVisualEnergyCritic,
    ema: NoisyVisualEnergyCritic,
    optimizer: torch.optim.Optimizer,
    training_generator: torch.Generator,
    *,
    update: int,
    arguments: Mapping[str, Any],
    render_config: CanonicalGlyphRenderConfig,
    manifest_receipt: Mapping[str, Any],
    partition_receipt: Mapping[str, Any],
    reader_sha256: str,
    prior_sha256: str,
    elapsed_seconds: float,
    peak_vram_gib: float,
    metrics: Mapping[str, float],
    device: torch.device,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": V52_ARCHITECTURE,
        "architecture": V52_ARCHITECTURE,
        "model_config": noisy_visual_energy_config_payload(critic.config),
        "critic": critic.state_dict(),
        "ema_critic": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_generator_state": training_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "update": update,
        "reader_checkpoint_sha256": reader_sha256,
        "prior_checkpoint_sha256": prior_sha256,
        "manifest": dict(manifest_receipt),
        "partition": dict(partition_receipt),
        "render_config": canonical_glyph_render_config_payload(render_config),
        "data_boundary": canonical_glyph_data_boundary_receipt(),
        "model_boundary": noisy_visual_energy_boundary_receipt(critic),
        "protocol": {
            "document": PROTOCOL_DOCUMENT,
            "sha256": file_sha256(PROTOCOL_DOCUMENT),
            "source_files_sha256": {
                path: file_sha256(path) for path in SOURCE_FILES
            },
            "fixed_optimization": FIXED_OPTIMIZATION,
            "effective_arguments": dict(arguments),
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_allocated_vram_gib": peak_vram_gib,
        "training_metrics": dict(metrics),
        "smoke_only": bool(arguments["smoke"]),
        "exploratory": bool(arguments["exploratory"]),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    return payload


def main() -> None:
    args = parse_args()
    arguments = _effective_arguments(args)
    _validate_fixed_arguments(arguments)
    if min(
        arguments["steps"],
        arguments["batch_size"],
        arguments["positions_per_stratum"],
    ) < 1:
        raise ValueError("V52 optimization sizes must be positive")
    if not 0.0 < arguments["ema_decay"] < 1.0:
        raise ValueError("V52 EMA decay must lie in (0,1)")

    output = Path(arguments["out"])
    output.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(arguments["device"])
    seed_everything(arguments["seed"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reader_sha256 = file_sha256(arguments["reader_checkpoint"])
    prior_sha256 = file_sha256(arguments["prior_checkpoint"])
    if not arguments["smoke"] and reader_sha256 != REQUIRED_READER_SHA256:
        raise ValueError("V52 reader checkpoint differs from the frozen protocol")
    if not arguments["smoke"] and prior_sha256 != REQUIRED_PRIOR_SHA256:
        raise ValueError("V52 prior checkpoint differs from the frozen protocol")
    reader_payload = torch.load(
        arguments["reader_checkpoint"], map_location="cpu", weights_only=False
    )
    prior_payload = torch.load(
        arguments["prior_checkpoint"], map_location="cpu", weights_only=False
    )
    if reader_payload.get("architecture") != "canonical-glyph-language-v42":
        raise ValueError("V52 requires the production V42 reader")
    if prior_payload.get("architecture") != V51_ARCHITECTURE:
        raise ValueError("V52 requires the measured V51 prior")
    reader = CausalRasterReader(
        canonical_glyph_language_config_from_payload(reader_payload["model_config"])
    )
    reader.load_v42_state(reader_payload["model"])
    reader.to(device).requires_grad_(False).eval()
    prior = VisualGlyphScorePrior(
        visual_posterior_language_config_from_payload(
            prior_payload["model_config"]
        )
    )
    prior.load_state_dict(prior_payload["ema_model"], strict=True)
    prior.to(device).requires_grad_(False).eval()

    resume_payload: dict[str, Any] | None = None
    start_update = 0
    if arguments["resume"]:
        resume_payload = torch.load(
            arguments["resume"], map_location="cpu", weights_only=False
        )
        if resume_payload.get("architecture") != V52_ARCHITECTURE:
            raise ValueError("resume checkpoint is not V52")
        if resume_payload["reader_checkpoint_sha256"] != reader_sha256:
            raise ValueError("resume checkpoint uses a different reader")
        if resume_payload["prior_checkpoint_sha256"] != prior_sha256:
            raise ValueError("resume checkpoint uses a different prior")
        config = noisy_visual_energy_config_from_payload(
            resume_payload["model_config"]
        )
        start_update = int(resume_payload["update"])
    else:
        config = NoisyVisualEnergyConfig()
    if start_update >= arguments["steps"]:
        raise ValueError("resume checkpoint already reached requested updates")
    critic = NoisyVisualEnergyCritic(config)
    ema = copy.deepcopy(critic).requires_grad_(False)
    if resume_payload is not None:
        critic.load_state_dict(resume_payload["critic"], strict=True)
        ema.load_state_dict(resume_payload["ema_critic"], strict=True)
    critic.to(device)
    ema.to(device).eval()
    boundary = noisy_visual_energy_boundary_receipt(critic)
    if boundary["critic_parameters"] > 6_000_000:
        raise RuntimeError("V52 critic exceeds its parameter cap")

    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=arguments["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=arguments["weight_decay"],
        fused=device.type == "cuda",
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer"])

    strict = not arguments["exploratory"] and not arguments["smoke"]
    manifest_receipt = verify_v25_manifest(arguments["manifest"], strict=strict)
    if strict and manifest_receipt["sha256"] != V25_MANIFEST_SHA256:
        raise ValueError("V52 production requires the frozen corpus manifest")
    records = load_v25_records(
        arguments["manifest"], strict_manifest=strict
    )
    partition_receipt = visual_cell_partition_receipt(records)
    render_config = CanonicalGlyphRenderConfig()
    total_examples = arguments["steps"] * arguments["batch_size"]
    consumed_examples = start_update * arguments["batch_size"]
    dataset = CanonicalGlyphLanguageDataset(
        records,
        split="train",
        render_config=render_config,
        seed=arguments["dataset_seed"],
        length=total_examples,
    )
    loader = DataLoader(
        Subset(dataset, range(consumed_examples, total_examples)),
        batch_size=arguments["batch_size"],
        shuffle=False,
        drop_last=True,
        num_workers=arguments["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=arguments["num_workers"] > 0,
        collate_fn=canonical_glyph_collate,
    )
    iterator = iter(loader)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and arguments["precision"] == "fp16",
    )
    training_generator = torch.Generator(device=device)
    if resume_payload is None:
        training_generator.manual_seed(arguments["seed"])
    else:
        training_generator.set_state(
            resume_payload["training_generator_state"].to(device="cpu")
        )
    log_path = output / "training_metrics.jsonl"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_metrics: dict[str, float] = {}
    stop_requested = False

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        for update in range(start_update + 1, arguments["steps"] + 1):
            critic.train()
            learning_rate = _scheduled_lr(
                update,
                base=arguments["learning_rate"],
                warmup=arguments["warmup_steps"],
                total=arguments["steps"],
                minimum_ratio=arguments["minimum_lr_ratio"],
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            raw = next(iterator)
            student = canonical_glyph_student_batch(raw)
            context = student["context"].to(device, non_blocking=True)
            target = student["target"].to(device, non_blocking=True)
            shuffled = _shuffle_prefix(
                context,
                first_index=(update - 1) * arguments["batch_size"],
            )
            with torch.no_grad(), autocast_context(
                device, arguments["precision"]
            ):
                ordered_language = reader.language(context)
                shuffled_language = reader.language(shuffled)
            positions = _selected_positions(
                len(context),
                per_stratum=arguments["positions_per_stratum"],
                device=device,
                generator=training_generator,
            )
            rows = torch.arange(len(context), device=device)[:, None]
            hidden = ordered_language["hidden_states"][rows, positions].flatten(
                0, 1
            )
            anchors = ordered_language["anchor_fields"][rows, positions].flatten(
                0, 1
            )
            selected_targets = target[rows, positions].flatten(0, 1)
            timestep = _training_timestep(
                update, device=device, generator=training_generator
            )
            timesteps = torch.full(
                (len(selected_targets),),
                timestep,
                device=device,
                dtype=torch.long,
            )
            endpoint_timesteps = torch.full(
                (len(context),), timestep, device=device, dtype=torch.long
            )
            noise = torch.randn(
                selected_targets.shape,
                device=device,
                dtype=selected_targets.dtype,
                generator=training_generator,
            )
            endpoint_noise = torch.randn(
                target[:, -1].shape,
                device=device,
                dtype=target.dtype,
                generator=training_generator,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, arguments["precision"]):
                loss = noisy_visual_energy_loss(
                    critic,
                    reader,
                    prior,
                    hidden=hidden,
                    anchors=anchors,
                    clean_images=selected_targets,
                    timesteps=timesteps,
                    noise=noise,
                    ordered_endpoint_hidden=ordered_language["hidden_states"][
                        :, -1
                    ],
                    ordered_endpoint_anchor=ordered_language["anchor_fields"][
                        :, -1
                    ],
                    shuffled_endpoint_hidden=shuffled_language["hidden_states"][
                        :, -1
                    ],
                    shuffled_endpoint_anchor=shuffled_language["anchor_fields"][
                        :, -1
                    ],
                    endpoint_images=target[:, -1],
                    endpoint_noise=endpoint_noise,
                    endpoint_timesteps=endpoint_timesteps,
                )
            if scaler.is_enabled():
                scaler.scale(loss.loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(), arguments["gradient_clip"]
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("V52 critic gradient is not finite")
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            _update_ema(ema, critic, decay=arguments["ema_decay"])
            if any(parameter.grad is not None for parameter in reader.parameters()):
                raise RuntimeError("frozen V42 reader unexpectedly received gradients")
            if any(parameter.grad is not None for parameter in prior.parameters()):
                raise RuntimeError("frozen V51 prior unexpectedly received gradients")
            final_metrics = loss.detached_metrics()
            final_metrics.update(
                {
                    "update": float(update),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if update == 1 or update % arguments["log_every"] == 0:
                _append_jsonl(log_path, final_metrics)
                print(json.dumps(final_metrics, sort_keys=True), flush=True)
            should_save = (
                update == arguments["steps"]
                or update % arguments["save_every"] == 0
                or stop_requested
            )
            if should_save:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                peak = (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                )
                payload = _checkpoint(
                    critic,
                    ema,
                    optimizer,
                    training_generator,
                    update=update,
                    arguments=arguments,
                    render_config=render_config,
                    manifest_receipt=manifest_receipt,
                    partition_receipt=partition_receipt,
                    reader_sha256=reader_sha256,
                    prior_sha256=prior_sha256,
                    elapsed_seconds=elapsed,
                    peak_vram_gib=peak,
                    metrics=final_metrics,
                    device=device,
                )
                _atomic_save(payload, output / "checkpoint_latest.pt")
                if update == arguments["steps"]:
                    _atomic_save(payload, output / "checkpoint_final.pt")
                    _atomic_json(
                        {
                            "architecture": V52_ARCHITECTURE,
                            "update": update,
                            "elapsed_seconds": elapsed,
                            "peak_allocated_vram_gib": peak,
                            "critic_parameters": boundary["critic_parameters"],
                            "combined_parameters": (
                                boundary["critic_parameters"]
                                + sum(
                                    parameter.numel()
                                    for parameter in reader.parameters()
                                )
                                + sum(
                                    parameter.numel()
                                    for parameter in prior.parameters()
                                )
                            ),
                            "training_metrics": final_metrics,
                            "checkpoint": "checkpoint_final.pt",
                        },
                        output / "training_summary.json",
                    )
            if stop_requested:
                break
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
