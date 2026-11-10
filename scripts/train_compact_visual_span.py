#!/usr/bin/env python3
"""Fit the compact visual language model on the complete visual sequence corpus."""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.compact_visual_span_training import (
    CompactVisualSpanObjective, VisualSpanTrainingExample, initialize_feedback_statistics,
)
from ilm.visual_lm.compact_visual_span_weights import load_compact_visual_span_model, open_compact_archive
from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.cache_visual_sequence_teacher_states import batches, read_sequences
from scripts.train_image_native_lm import atomic_torch_save


LOCAL = ROOT.parent / "LocalLLM"
CODEC = ROOT / "artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt"
SOURCES = (
    "scripts/train_compact_visual_span.py", "scripts/cache_visual_sequence_teacher_states.py",
    "ilm/visual_lm/compact_visual_span_model.py", "ilm/visual_lm/compact_visual_span_weights.py",
    "ilm/visual_lm/compact_visual_span_training.py", "ilm/visual_lm/visual_sequence_corpus.py",
)


def training_source(teacher_run):
    preparation_path = teacher_run / "preparation.json"
    cache_path = teacher_run / "cache/report.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Complete the teacher cache before fitting: {cache_path}")
    preparation = json.loads(preparation_path.read_text())
    cache = json.loads(cache_path.read_text())
    prep_sha = sha(preparation_path)
    corpus = VisualSequenceCorpus(preparation["data"], split="train",
                                  expected_report_sha256=preparation["data_report_sha256"])
    planned = preparation["shards"]
    completed = cache["shards"]
    if (cache["preparation_sha256"] != prep_sha or len(corpus) != preparation["records"]
            or len(corpus.shards) != len(planned) or len(planned) != len(completed)
            or cache["state_rows"] != preparation["state_rows"]):
        raise ValueError("A complete teacher cache for this training corpus is required")
    for index, (original, saved) in enumerate(zip(planned, completed)):
        if (saved["directory"] != f"shard_{index:04d}" or saved["preparation_sha256"] != prep_sha
                or any(saved[key] != original[key] for key in ("records", "first_record", "state_rows"))
                or original["first_record"] != corpus.offsets[index]
                or original["files"] != corpus.shards[index]["files"]):
            raise ValueError(f"Teacher shard {index} does not match the visual corpus")
    return corpus, preparation, completed, prep_sha, sha(cache_path)


def load_states(teacher_run, shard, records):
    path = teacher_run / "cache" / shard["directory"] / "states.pt"
    if sha(path) != shard["states_sha256"]:
        raise ValueError(f"Teacher states changed in {shard['directory']}")
    saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    offsets = np.cumsum([0] + [len(row.positions) for row in records], dtype=np.int64)
    if (saved["first_record"] != shard["first_record"]
            or saved["preparation_sha256"] != shard["preparation_sha256"]
            or saved["states"].shape != (int(offsets[-1]), 4096) or saved["states"].dtype != torch.bfloat16
            or not np.array_equal(saved["record_offsets"].numpy(), offsets)
            or not np.array_equal(saved["first_state_span"].numpy(), [row.first_state_span for row in records])
            or not np.array_equal(saved["native_positions"].numpy(), np.concatenate([row.positions for row in records]))):
        raise ValueError(f"Teacher offsets or state positions differ in {shard['directory']}")
    return saved["states"], offsets


def source_targets(archive, units):
    stored = archive.omitted["token_embd.weight"]
    lookup = torch.full((len(stored.data),), -1, dtype=torch.long)
    lookup[units] = torch.arange(len(units))
    pieces = []
    for first in range(0, len(units), 512):
        indices = units[first:first + 512]
        values = archive.gguf.dequantize(stored.data[indices], stored.tensor_type)
        pieces.append(torch.from_numpy(values).to(torch.bfloat16))
    return lookup, torch.cat(pieces)


def image_heads(model):
    return {"encoder": model.encoder.state_dict(), "writer": model.writer.state_dict()}


def checkpoint(run, model, objective, optimizer, identity, cursor, updates, history, device, milestone):
    deploy = {"identity": identity, "updates": updates, "image_heads": image_heads(model)}
    if milestone:
        atomic_torch_save(deploy, run / f"image_heads_step_{updates:07d}.pt")
    atomic_torch_save({**deploy, "cursor": cursor, "history": history,
                       "teacher_projection": objective.state_dict(), "optimizer": optimizer.state_dict(),
                       "cpu_rng": torch.get_rng_state(), "gpu_rng": torch.cuda.get_rng_state(device)},
                      run / "checkpoint_latest.pt")


def fit(args, corpus, plan, cache, identity, monitor, device):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, provenance = load_compact_visual_span_model(LOCAL, CODEC, device)
    model.train()
    model.codec.eval()
    model.context.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    objective = CompactVisualSpanObjective().to(device)
    archive = open_compact_archive(LOCAL)
    lookup, references = source_targets(archive, plan["training_native_units"])
    statistics = initialize_feedback_statistics(model, archive, plan["training_native_units"])
    del archive
    params = [p for p in [*model.parameters(), *objective.parameters()] if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.01, fused=True)
    latest = args.run / "checkpoint_latest.pt"
    cursor = updates = 0
    history = []
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Resume requires the original corpus, source and training configuration")
        model.encoder.load_state_dict(saved["image_heads"]["encoder"])
        model.writer.load_state_dict(saved["image_heads"]["writer"])
        objective.load_state_dict(saved["teacher_projection"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        torch.cuda.set_rng_state(saved["gpu_rng"], device)
        cursor, updates, history = saved["cursor"], saved["updates"], saved["history"]
        del saved
    else:
        checkpoint(args.run, model, objective, optimizer, identity, cursor, updates, history, device, True)

    shard_count = len(cache)
    total = args.epochs * shard_count
    milestones = {(total * quarter + 3) // 4 for quarter in range(1, 5)}
    order = list(range(shard_count))
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        for position, shard_index in enumerate(order):
            ordinal = epoch * shard_count + position
            if ordinal < cursor:
                continue
            monitor.position = {"updates": updates, "completed_shards": ordinal, "total_shards": total}
            monitor.status("fitting_complete_sequences", shard=shard_index)
            monitor.guard()
            shard_start = time.perf_counter()
            shard = plan["shards"][shard_index]
            records = read_sequences(corpus.directory, shard)
            states, offsets = load_states(args.teacher_run, cache[shard_index], records)
            groups = list(batches(records, args.padded_unit_budget))
            random.Random(args.seed + total + ordinal).shuffle(groups)
            sums = {}
            for group in groups:
                monitor.guard()
                examples = []
                for index in group:
                    record = records[index]
                    row = corpus[shard["first_record"] + index]
                    alignment = corpus.offline_teacher_alignment(shard["first_record"] + index)
                    selected = lookup[alignment["unit_ids"].long()]
                    if bool((selected < 0).any()) or not np.array_equal(record.units, alignment["unit_ids"].numpy()):
                        raise ValueError("A visual training example has an unprepared source unit")
                    examples.append(VisualSpanTrainingExample(
                        row["ink"].to(device), row["span_glyph_ends"], alignment["span_unit_ends"],
                        row["prompt_spans"], references[selected], states[offsets[index]:offsets[index + 1]],
                    ))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = objective(model, examples)
                    loss = torch.stack(list(losses.values())).sum()
                values = torch.stack([value.detach() for value in losses.values()]).tolist()
                if not all(np.isfinite(values)):
                    raise ValueError("Nonfinite full-sequence loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                for name, value in zip(losses, values):
                    sums[name] = sums.get(name, 0.0) + value
            result = {"epoch": epoch + 1, "shard": shard_index, "records": len(records),
                      "target_states": len(states), "optimizer_updates": updates,
                      "seconds": time.perf_counter() - shard_start,
                      "peak_allocated_vram_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                      "mean_batch_losses": {name: value / len(groups) for name, value in sums.items()}}
            history.append(result)
            checkpoint(args.run, model, objective, optimizer, identity, ordinal + 1, updates, history,
                       device, ordinal + 1 in milestones)
            print(json.dumps(result), flush=True)
    if (sum(item["records"] for item in history) != len(corpus) * args.epochs
            or sum(item["target_states"] for item in history) != plan["state_rows"] * args.epochs):
        raise ValueError("Complete-corpus exposures do not reconcile")
    final = args.run / f"image_heads_step_{updates:07d}.pt"
    return {"provenance": provenance, "feedback_initialization": statistics, "identity": identity,
            "history": history, "updates": updates, "training_records": len(corpus),
            "sequence_exposures": len(corpus) * args.epochs, "supervised_states_per_epoch": plan["state_rows"],
            "completed_shard_seconds": sum(item["seconds"] for item in history),
            "completed_shard_peak_allocated_vram_gib": max(item["peak_allocated_vram_gib"] for item in history),
            "training_parameters": sum(p.numel() for p in params), "final_image_heads_sha256": sha(final),
            "final_image_heads": final.name, "held_out_language_evaluated": False, **monitor.receipt()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, default=ROOT / "artifacts/visual_sequence_teacher_v1")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/compact_visual_span_fit_v1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--padded-unit-budget", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=202611152)
    parser.add_argument("--seconds", type=float, default=10800)
    args = parser.parse_args()
    if min(args.epochs, args.padded_unit_budget, args.learning_rate, args.seconds) <= 0:
        parser.error("Training budgets and learning rate must be positive")
    torch.set_num_threads(2)
    corpus, plan, cache, preparation_sha, cache_sha = training_source(args.teacher_run)
    if args.padded_unit_budget < plan["maximum_native_sequence"]:
        parser.error("Batch budget must fit the longest complete sequence")
    identity = {"corpus_report_sha256": corpus.report_sha256, "teacher_preparation_sha256": preparation_sha,
                "teacher_cache_sha256": cache_sha, "codec_sha256": sha(CODEC),
                "code_sha256": {name: sha(ROOT / name) for name in SOURCES}, "epochs": args.epochs,
                "padded_unit_budget": args.padded_unit_budget, "learning_rate": args.learning_rate,
                "seed": args.seed, "optimizer": "AdamW with gradient norm capped at 1.0"}
    with full_reader_gpu() as (device, admission):
        if (args.run / "report.json").exists():
            raise FileExistsError("This complete-sequence fit already finished")
        attempt = args.run / "attempts" / f"{time.time_ns()}_{os.getpid()}"
        attempt.mkdir(parents=True)
        monitor = GlyphReaderMonitor(attempt, args.seconds)
        monitor.admission = admission
        monitor.status("loading_compact_model")
        try:
            monitor.guard()
            result = fit(args, corpus, plan, cache, identity, monitor, device)
            save_json(args.run / "report.json", result)
            monitor.status("completed")
        except BaseException as error:
            monitor.status("failed", reason=repr(error))
            raise
        finally:
            status = attempt / "status.json"
            save_json(status, {**json.loads(status.read_text()), "pid": None})


if __name__ == "__main__":
    main()
