#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ilm.visual_lm import (
    GlyphCorpus,
    RetinalFieldConfig,
    VisualAssociativeReader,
    VisualEpisodeDataset,
    VisualEpisodeSpec,
    historical_episode_specs,
    instruction_episode_specs,
    load_alpaca_records,
)
from ilm.visual_lm.dataset import pil_to_tensor
from ilm.visual_lm.retinal_memory import (
    field_variance_loss,
    retinal_config_payload,
    symmetric_info_nce,
)
from ilm.visual_lm.teacher import load_teacher_manifest
from ilm.visual_lm.visual_episodes import (
    augment_episode_image,
    render_episode_answer,
    render_episode_query,
    stable_episode_fraction,
    visual_episode_collate,
)


DEFAULT_HISTORICAL_CHARS = "言,中,水,日,月,人,山,火,木,口,学,車,车,王,雨,田,金"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the token-free retinal reader and build its image-valued visual memory."
    )
    data = parser.add_argument_group("visual episodes")
    data.add_argument("--zh-data", default="data/raw/alpaca_zh.json")
    data.add_argument("--en-data", default="data/raw/alpaca_en.json")
    data.add_argument("--disable-zh", action="store_true")
    data.add_argument("--disable-en", action="store_true")
    data.add_argument("--max-records-per-language", type=int, default=5_000)
    data.add_argument("--max-prompt-chars", type=int, default=120)
    data.add_argument("--max-response-chars", type=int, default=220)
    data.add_argument("--glyph-root", default=None)
    data.add_argument("--teacher-manifest", default="data/teacher/historical_qwen8b_v2.jsonl")
    data.add_argument("--historical-chars", default=DEFAULT_HISTORICAL_CHARS)
    data.add_argument("--encoder-holdout-fraction", type=float, default=0.10)
    data.add_argument("--memory-keys-per-episode", type=int, default=2)

    model = parser.add_argument_group("retinal field")
    model.add_argument("--image-size", type=int, default=384)
    model.add_argument("--peripheral-size", type=int, default=128)
    model.add_argument("--fovea-size", type=int, default=96)
    model.add_argument("--fovea-extent", type=int, default=112)
    model.add_argument("--fovea-count", type=int, default=8)
    model.add_argument("--saliency-grid", type=int, default=12)
    model.add_argument("--base-channels", type=int, default=32)
    model.add_argument("--field-dim", type=int, default=256)
    model.add_argument("--embedding-dim", type=int, default=256)
    model.add_argument("--read-steps", type=int, default=4)

    train = parser.add_argument_group("optimization")
    train.add_argument("--out", default="artifacts/visual_field_memory")
    train.add_argument("--epochs", type=int, default=16)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--answer-weight", type=float, default=0.55)
    train.add_argument("--variance-weight", type=float, default=0.08)
    train.add_argument("--historical-batches-per-epoch", type=int, default=6)
    train.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=711)
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--save-every-epochs", type=int, default=2)
    train.add_argument("--resume", default=None)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else value if value != "auto" else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.amp.autocast("cuda", dtype=dtype)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_episodes(args: argparse.Namespace) -> tuple[list[VisualEpisodeSpec], list[VisualEpisodeSpec], dict[str, Any]]:
    instruction_episodes: list[VisualEpisodeSpec] = []
    provenance: dict[str, Any] = {"instruction_sources": [], "historical": None}
    sources = []
    if not args.disable_zh:
        sources.append((args.zh_data, "zh", "GPT-4-LLM alpaca_gpt4_data_zh", "CC-BY-NC-4.0"))
    if not args.disable_en:
        sources.append((args.en_data, "en", "Stanford Alpaca", "CC-BY-NC-4.0"))
    for path_value, language, source, license_name in sources:
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"missing {path}; run scripts/download_alpaca.py")
        records = load_alpaca_records(
            path,
            language=language,
            source=source,
            max_prompt_chars=args.max_prompt_chars,
            max_response_chars=args.max_response_chars,
            limit=args.max_records_per_language,
        )
        instruction_episodes.extend(instruction_episode_specs(records))
        provenance["instruction_sources"].append(
            {
                "path": str(path),
                "source": source,
                "language": language,
                "license": license_name,
                "episodes": len(records),
            }
        )

    historical: list[VisualEpisodeSpec] = []
    requested = [item.strip() for item in args.historical_chars.split(",") if item.strip()]
    try:
        seed_corpus = GlyphCorpus(args.glyph_root)
        characters = [character for character in requested if (seed_corpus.root / character).exists()]
        corpus = GlyphCorpus(seed_corpus.root, characters=characters)
        teacher_records = load_teacher_manifest(args.teacher_manifest)
        historical = historical_episode_specs(corpus, characters, teacher_records=teacher_records)
        provenance["historical"] = {
            "root": str(corpus.root),
            "characters": [episode.historical_char for episode in historical],
            "teacher_manifest": args.teacher_manifest,
            "license": "local research data; redistribution license not verified",
            "evidence_policy": "copy exact source pixels; never label synthesis as attested",
        }
    except FileNotFoundError:
        provenance["historical"] = {"disabled": "glyph corpus unavailable"}

    if not instruction_episodes and not historical:
        raise ValueError("no visual episodes were loaded")
    return instruction_episodes, historical, provenance


def train_split(
    instruction_episodes: Sequence[VisualEpisodeSpec],
    historical: Sequence[VisualEpisodeSpec],
    holdout_fraction: float,
) -> tuple[list[VisualEpisodeSpec], list[VisualEpisodeSpec]]:
    if not 0.0 <= holdout_fraction < 0.5:
        raise ValueError("encoder holdout fraction must be in [0, 0.5)")
    train = [
        episode
        for episode in instruction_episodes
        if stable_episode_fraction(episode.identifier) >= holdout_fraction
    ]
    holdout = [
        episode
        for episode in instruction_episodes
        if stable_episode_fraction(episode.identifier) < holdout_fraction
    ]
    train.extend(historical)
    if not train:
        raise ValueError("encoder training split is empty")
    return train, holdout


def infinite_batches(loader: DataLoader, dataset: VisualEpisodeDataset) -> Iterator[dict[str, Any]]:
    epoch = 0
    while True:
        dataset.set_epoch(100_000 + epoch)
        yield from loader
        epoch += 1


def encode_training_fields(
    model: VisualAssociativeReader,
    query_a: torch.Tensor,
    query_b: torch.Tensor,
    answer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = query_a.shape[0]
    fields = model.retina(torch.cat((query_a, query_b, answer), dim=0))
    first, second, answer_field = fields.split(batch, dim=0)
    first = F.normalize(model.query_head(first), dim=-1)
    second = F.normalize(model.query_head(second), dim=-1)
    answer_field = F.normalize(model.answer_head(answer_field), dim=-1)
    return first, second, answer_field


def train_batch(
    model: VisualAssociativeReader,
    batch: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    precision: str,
    answer_weight: float,
    variance_weight: float,
) -> dict[str, float]:
    query_a = batch["query_a"].to(device, non_blocking=True)
    query_b = batch["query_b"].to(device, non_blocking=True)
    answer = batch["answer"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        first, second, answer_field = encode_training_fields(model, query_a, query_b, answer)
        view_loss, view_accuracy = symmetric_info_nce(first, second, model.contrastive_scale)
        answer_loss, answer_accuracy = symmetric_info_nce(first, answer_field, model.contrastive_scale)
        variance = field_variance_loss(first, second, answer_field)
        loss = view_loss + answer_weight * answer_loss + variance_weight * variance
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": float(loss.detach()),
        "view_loss": float(view_loss.detach()),
        "answer_loss": float(answer_loss.detach()),
        "variance_loss": float(variance.detach()),
        "view_batch_accuracy": float(view_accuracy.detach()),
        "answer_batch_accuracy": float(answer_accuracy.detach()),
        "gradient_norm": float(gradient_norm),
    }


@torch.no_grad()
def validation_batch_metrics(
    model: VisualAssociativeReader,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    maximum_batches: int = 8,
) -> dict[str, float]:
    model.eval()
    totals = {"view_loss": 0.0, "answer_loss": 0.0, "view_accuracy": 0.0, "answer_accuracy": 0.0}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= maximum_batches:
            break
        query_a = batch["query_a"].to(device)
        query_b = batch["query_b"].to(device)
        answer = batch["answer"].to(device)
        with autocast_context(device, precision):
            first, second, answer_field = encode_training_fields(model, query_a, query_b, answer)
            view_loss, view_accuracy = symmetric_info_nce(first, second, model.contrastive_scale)
            answer_loss, answer_accuracy = symmetric_info_nce(first, answer_field, model.contrastive_scale)
        totals["view_loss"] += float(view_loss)
        totals["answer_loss"] += float(answer_loss)
        totals["view_accuracy"] += float(view_accuracy)
        totals["answer_accuracy"] += float(answer_accuracy)
        batches += 1
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def checkpoint_payload(
    model: VisualAssociativeReader,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "architecture": "visual-field-memory-v1",
        "retinal_config": retinal_config_payload(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "arguments": vars(args),
        "provenance": provenance,
    }


def safe_stem(identifier: str) -> str:
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:20]


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


@torch.no_grad()
def build_visual_memory(
    model: VisualAssociativeReader,
    episodes: Sequence[VisualEpisodeSpec],
    *,
    output: Path,
    image_size: int,
    keys_per_episode: int,
    batch_size: int,
    device: torch.device,
    precision: str,
    holdout_identifiers: set[str],
) -> dict[str, Any]:
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    answers_dir = output / "answers"
    queries_dir = output / "queries"
    evaluations_dir = output / "evaluation_queries"
    answers_dir.mkdir(exist_ok=True)
    queries_dir.mkdir(exist_ok=True)
    evaluations_dir.mkdir(exist_ok=True)
    entries: list[dict[str, Any]] = []
    rendered_keys: list[tuple[int, torch.Tensor]] = []

    for entry_index, episode in enumerate(episodes):
        stem = safe_stem(episode.identifier)
        answer_path = answers_dir / f"{stem}.png"
        query_path = queries_dir / f"{stem}.png"
        evaluation_path = evaluations_dir / f"{stem}.png"
        answer_image = render_episode_answer(
            episode,
            image_size=image_size,
            variant=entry_index * 101 + 17,
            augment=False,
        )
        answer_image.save(answer_path, optimize=True)
        evaluation_query = (
            episode.evaluation_queries[0]
            if episode.evaluation_queries
            else episode.query_variants[0]
        )
        evaluation_image = render_episode_query(
            episode,
            query=evaluation_query,
            image_size=image_size,
            variant=entry_index * 1297 + 1_000_003,
            detailed_layout=False,
            augment=False,
        )
        evaluation_image = augment_episode_image(
            evaluation_image,
            random.Random(entry_index * 8191 + 991),
        )
        evaluation_image.save(evaluation_path, optimize=True)
        metadata = episode.metadata()
        metadata.update(
            {
                "answer_image": str(answer_path.relative_to(output)),
                "query_audit_image": str(query_path.relative_to(output)),
                "evaluation_image": str(evaluation_path.relative_to(output)),
                "evaluation_policy": "held_out_wording_and_render"
                if episode.historical_char is not None
                else "held_out_render_and_capture_damage",
                "encoder_training_status": "inserted_after_training"
                if episode.identifier in holdout_identifiers
                else "seen_during_encoder_training",
            }
        )
        entries.append(metadata)
        for view_index in range(max(1, keys_per_episode)):
            query_text = episode.query_variants[view_index % len(episode.query_variants)]
            query_image = render_episode_query(
                episode,
                query=query_text,
                image_size=image_size,
                variant=entry_index * 997 + view_index * 43 + 5,
                detailed_layout=episode.historical_char is not None and view_index % 2 == 1,
                augment=False,
            )
            if view_index == 0:
                query_image.save(query_path, optimize=True)
            rendered_keys.append((entry_index, pil_to_tensor(query_image)))

    key_tensors: list[torch.Tensor] = []
    entry_indices: list[int] = []
    for batch in chunks(rendered_keys, max(1, batch_size)):
        images = torch.stack([item[1] for item in batch]).to(device)
        with autocast_context(device, precision):
            encoded = model.encode_query(images)
        key_tensors.append(encoded.float().cpu())
        entry_indices.extend(item[0] for item in batch)

    keys = torch.cat(key_tensors, dim=0)
    atomic_torch_save(
        {"keys": keys, "entry_indices": torch.tensor(entry_indices, dtype=torch.long)},
        output / "visual_memory.pt",
    )
    with (output / "memory_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "entries": len(entries),
        "keys": int(keys.shape[0]),
        "embedding_dimensions": int(keys.shape[1]),
        "answer_images": len(entries),
        "holdout_entries_inserted_without_retraining": len(holdout_identifiers),
        "student_value_types": ["image/png", "continuous-image-key", "provenance-metadata"],
        "forbidden_student_inputs": ["text", "token_ids", "unicode_ids", "ocr_strings"],
    }
    (output / "memory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    instruction_episodes, historical, provenance = build_episodes(args)
    encoder_train, holdout = train_split(
        instruction_episodes,
        historical,
        args.encoder_holdout_fraction,
    )
    all_episodes = list(instruction_episodes) + list(historical)
    train_dataset = VisualEpisodeDataset(encoder_train, image_size=args.image_size, seed=args.seed)
    validation_source = holdout[: max(args.batch_size, min(len(holdout), args.batch_size * 8))]
    if not validation_source:
        validation_source = encoder_train[: max(1, min(len(encoder_train), args.batch_size * 4))]
    validation_dataset = VisualEpisodeDataset(validation_source, image_size=args.image_size, seed=args.seed + 50_000)
    historical_dataset = (
        VisualEpisodeDataset(historical, image_size=args.image_size, seed=args.seed + 100_000)
        if historical
        else None
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": visual_episode_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=len(train_dataset) >= args.batch_size, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, drop_last=False, **loader_options)
    historical_iterator = None
    if historical_dataset is not None:
        historical_loader = DataLoader(
            historical_dataset,
            batch_size=min(args.batch_size, len(historical_dataset)),
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=visual_episode_collate,
            drop_last=False,
        )
        historical_iterator = infinite_batches(historical_loader, historical_dataset)

    config = RetinalFieldConfig(
        image_size=args.image_size,
        peripheral_size=args.peripheral_size,
        fovea_size=args.fovea_size,
        fovea_extent=args.fovea_extent,
        fovea_count=args.fovea_count,
        saliency_grid=args.saliency_grid,
        base_channels=args.base_channels,
        field_dim=args.field_dim,
        embedding_dim=args.embedding_dim,
        read_steps=args.read_steps,
    )
    model = VisualAssociativeReader(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * (len(train_loader) + (args.historical_batches_per_epoch if historical else 0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    run_manifest = {
        "architecture": "visual-field-memory-v1",
        "retinal_config": retinal_config_payload(config),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_episodes": len(encoder_train),
        "encoder_holdout_episodes": len(holdout),
        "historical_episodes": len(historical),
        "all_memory_episodes": len(all_episodes),
        "provenance": provenance,
        "model_boundary": "floating image tensors only",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, ensure_ascii=False), flush=True)

    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        epoch_metrics: list[dict[str, float]] = []
        for batch in train_loader:
            values = train_batch(
                model,
                batch,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                precision=args.precision,
                answer_weight=args.answer_weight,
                variance_weight=args.variance_weight,
            )
            scheduler.step()
            global_step += 1
            epoch_metrics.append(values)
            if global_step % args.log_every == 0:
                event = {
                    "stage": "retinal-field",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_seconds": time.monotonic() - started,
                    **values,
                }
                append_jsonl(metrics_path, event)
                print(json.dumps(event), flush=True)

        if historical_iterator is not None:
            for _ in range(args.historical_batches_per_epoch):
                values = train_batch(
                    model,
                    next(historical_iterator),
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    precision=args.precision,
                    answer_weight=args.answer_weight,
                    variance_weight=args.variance_weight,
                )
                scheduler.step()
                global_step += 1
                epoch_metrics.append(values)

        validation_dataset.set_epoch(epoch + 1_000_000)
        validation = validation_batch_metrics(
            model,
            validation_loader,
            device=device,
            precision=args.precision,
        )
        event = {
            "stage": "epoch",
            "epoch": epoch + 1,
            "step": global_step,
            "training_loss": sum(item["loss"] for item in epoch_metrics) / max(1, len(epoch_metrics)),
            "elapsed_seconds": time.monotonic() - started,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        append_jsonl(metrics_path, event)
        print(json.dumps(event), flush=True)
        payload = checkpoint_payload(
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            provenance=provenance,
        )
        atomic_torch_save(payload, output / "checkpoint_latest.pt")
        if (epoch + 1) % args.save_every_epochs == 0:
            atomic_torch_save(payload, output / f"checkpoint_epoch_{epoch + 1:04d}.pt")

    memory_summary = build_visual_memory(
        model,
        all_episodes,
        output=output / "memory",
        image_size=args.image_size,
        keys_per_episode=args.memory_keys_per_episode,
        batch_size=args.batch_size,
        device=device,
        precision=args.precision,
        holdout_identifiers={episode.identifier for episode in holdout},
    )
    summary = {
        **run_manifest,
        "epochs": args.epochs,
        "global_step": global_step,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "checkpoint": str(output / "checkpoint_latest.pt"),
        "memory": memory_summary,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
