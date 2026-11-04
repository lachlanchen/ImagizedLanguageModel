#!/usr/bin/env python3
"""Fixed raw-pixel beam versus greedy audit; no training or glyph selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ilm.visual_lm.canonical_glyph_language_data import (
    CanonicalGlyphAuditDataset,
    CanonicalGlyphRenderConfig,
    canonical_glyph_audit_collate,
)
from ilm.visual_lm.canonical_glyph_language_evaluation import _shuffle_prefix
from ilm.visual_lm.joint_exact_raster_v55 import model_from_checkpoint
from ilm.visual_lm.raster_beam import raster_beam_search
from ilm.visual_lm.visual_cell_data import load_v25_records, pack_visual_cells
from ilm.visual_lm.visual_cell_eval_data import (
    build_visual_cell_audit_windows,
    build_visual_character_statistics,
)
from ilm.visual_lm.visual_posterior_language_v51_evaluation import (
    isolated_ink_fraction,
    pixel_f1,
)
from scripts.diagnose_raster_form_validity import raster_keys
from scripts.eval_exact_raster_density_v54 import evaluator_bank
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _resolve_device
from scripts.train_visual_state_actuator import file_sha256, seed_everything

CHECKPOINT = "artifacts/joint_exact_raster_v55_20261105/joint/checkpoint_final.pt"
CHECKPOINT_SHA = "ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394"
PROTOCOL = "references/raster_beam_v56_protocol.md"
DEFAULT_OUT = "artifacts/raster_beam_v56_20261105"
SEED = 20265620


def bootstrap(values, ids):
    values = np.asarray(values, dtype=float)
    ids = np.asarray(ids)
    rows = [np.flatnonzero(ids == group) for group in sorted(set(ids))]
    totals = np.array([values[row].sum() for row in rows])
    counts = np.array([len(row) for row in rows])
    draws = np.random.default_rng(20265719).integers(0, len(rows), (2000, len(rows)))
    means = totals[draws].sum(1) / counts[draws].sum(1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "source_records": len(rows),
        "windows": len(values),
        "replicates": 2000,
        "bootstrap_seed": 20265719,
    }


def stats(images, targets, known):
    b, t = images.shape[:2]
    keys = raster_keys(images)
    exact = (images == targets).flatten(2).all(-1).float()
    form = np.array([key in known for key in keys]).reshape(b, t)
    flat = images.flatten(0, 1)
    repeats = (
        (images[:, 1:] == images[:, :-1]).flatten(2).all(-1).float().mean().item()
        if t > 1
        else 0.0
    )
    per_stream = [len(set(keys[i * t : (i + 1) * t])) for i in range(b)]
    return {
        "examples": b,
        "cells": t,
        "exact_target": float(exact.mean()),
        "exact_target_by_step": exact.mean(0).tolist(),
        "exact_known_form": float(form.mean()),
        "blank_fraction": float((flat.flatten(1).sum(-1) < 2).float().mean()),
        "pixel_f1": float(pixel_f1(flat, targets.flatten(0, 1)).mean()),
        "isolated_ink_fraction": float(isolated_ink_fraction(flat).mean()),
        "ink_ratio": float(images.mean() / targets.mean()),
        "distinct_rasters": len(set(keys)),
        "mean_distinct_per_stream": float(np.mean(per_stream)),
        "adjacent_repeated_raster_fraction": repeats,
    }, exact.mean(1).numpy()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "report.json").exists():
        raise FileExistsError("completed beam audit is immutable")
    if file_sha256(args.checkpoint) != CHECKPOINT_SHA:
        raise ValueError("beam audit requires unchanged V55 joint EMA")
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for path, sha in checkpoint["provenance"].items():
        if file_sha256(path) != sha:
            raise ValueError("trained source/protocol changed")
    manifest = "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
    if file_sha256(manifest) != checkpoint["manifest"]["sha256"]:
        raise ValueError("corpus changed")
    model = model_from_checkpoint(checkpoint, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = load_v25_records(manifest, strict_manifest=True)
    render = CanonicalGlyphRenderConfig(**checkpoint["render_config"])
    distribution = build_visual_character_statistics(
        records, bank_size=1024, script_views_mode=render.script_views
    )
    count, feedback_count, motor_count = (16, 4, 8) if args.smoke else (2048, 128, 256)
    windows = build_visual_cell_audit_windows(
        records,
        distribution,
        count=count,
        continuation_cells=8,
        seed=SEED,
        script_views_mode=render.script_views,
    )
    loader = DataLoader(
        CanonicalGlyphAuditDataset(windows, distribution, render_config=render),
        batch_size=16,
        num_workers=0 if args.smoke else 4,
        collate_fn=canonical_glyph_audit_collate,
    )
    output = {name: [] for name in ["beam8", "beam8_shuffled", "greedy"]}
    self_nll = {key: [] for key in output}
    feedback = {key: [] for key in output}
    motor = []
    targets = []
    feedback_targets = []
    motor_targets = []
    ids = []
    feedback_ids = []
    score_errors = []
    width_one_equal = True
    target_nll = {"ordered": [], "shuffled": []}
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(SEED)
    for raw in loader:
        offset = len(ids)
        context = raw["context"].to(device)
        target = (raw["continuation"].to(device) >= 0.5).float()
        hidden, fields = model.read(context)
        shuffled = _shuffle_prefix(context, first_index=offset)
        sh, sf = model.read(shuffled)
        states = {"beam8": (hidden, fields), "beam8_shuffled": (sh, sf)}
        for route, (h, f) in states.items():
            result = raster_beam_search(model, h, f, beam_width=8)
            exact_nll = model.image_nll(result.images, h, f)
            score_errors.append(float((exact_nll + result.log_probability).abs().max()))
            output[route].append(result.images.cpu()[:, None])
            self_nll[route].append(exact_nll.cpu())
        greedy = model.sample_state(hidden, fields, generator=generator, greedy=True)
        output["greedy"].append(greedy.cpu()[:, None])
        self_nll["greedy"].append(model.image_nll(greedy, hidden, fields).cpu())
        if offset == 0:
            one = raster_beam_search(model, hidden, fields, beam_width=1)
            width_one_equal = torch.equal(one.images, greedy)
        target_nll["ordered"].append(
            model.image_nll(target[:, 0], hidden, fields).cpu()
        )
        target_nll["shuffled"].append(model.image_nll(target[:, 0], sh, sf).cpu())
        remaining = motor_count - offset
        if remaining > 0:
            supplied = target[: min(len(target), remaining), 0]
            exact_fields = model.reader.field.encode_unit(supplied)
            motor.append(
                raster_beam_search(
                    model,
                    torch.zeros(len(supplied), 384, device=device),
                    exact_fields,
                    beam_width=8,
                ).images.cpu()[:, None]
            )
            motor_targets.append(supplied.cpu()[:, None])
        remaining = feedback_count - offset
        if remaining > 0:
            n = min(len(context), remaining)
            for route in feedback:
                current = (shuffled if route == "beam8_shuffled" else context)[:n]
                generated = []
                for _ in range(8):
                    h, f = model.read(current)
                    image = (
                        model.sample_state(h, f, generator=generator, greedy=True)
                        if route == "greedy"
                        else raster_beam_search(model, h, f, beam_width=8).images
                    )
                    generated.append(image.cpu())
                    current = torch.cat((current, image[:, None]), dim=1)[:, -64:]
                feedback[route].append(torch.stack(generated, 1))
            feedback_targets.append(target[:n].cpu())
            feedback_ids.extend(raw["identifier"][:n])
        targets.append(target[:, :1].cpu())
        ids.extend(raw["identifier"])
        print(
            json.dumps(
                {
                    "phase": "pixel_beam_audit",
                    "windows": len(ids),
                    "total": count,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    single = {
        "outputs": {k: torch.cat(v) for k, v in output.items()},
        "targets": torch.cat(targets),
        "identifiers": ids,
    }
    stream = {
        "outputs": {k: torch.cat(v) for k, v in feedback.items()},
        "targets": torch.cat(feedback_targets),
        "identifiers": feedback_ids,
    }
    supplied = {
        "outputs": {"beam8": torch.cat(motor)},
        "targets": torch.cat(motor_targets),
    }
    # All model output pixels exist before the evaluator bank is constructed.
    bank = evaluator_bank(records, render, (single, stream, supplied))
    known = set(raster_keys(bank))
    if not all(key in known for key in raster_keys(single["targets"])):
        raise ValueError("bank lacks target coverage")
    measured = {}
    correct = {}
    for phase, data in [
        ("generation", single),
        ("feedback", stream),
        ("motor", supplied),
    ]:
        measured[phase] = {}
        correct[phase] = {}
        for route, images in data["outputs"].items():
            measured[phase][route], correct[phase][route] = stats(
                images, data["targets"], known
            )
    measured["target_stream"], _ = stats(stream["targets"], stream["targets"], known)
    comparisons = {
        "beam_minus_greedy": bootstrap(
            correct["generation"]["beam8"] - correct["generation"]["greedy"], ids
        ),
        "beam_minus_shuffled": bootstrap(
            correct["generation"]["beam8"] - correct["generation"]["beam8_shuffled"],
            ids,
        ),
        "feedback_beam_minus_greedy": bootstrap(
            correct["feedback"]["beam8"] - correct["feedback"]["greedy"], feedback_ids
        ),
    }
    peak = (
        torch.cuda.max_memory_allocated(device) / 1024**3
        if device.type == "cuda"
        else 0
    )
    primary = measured["generation"]["beam8"]
    g = measured["feedback"]["beam8"]
    greedy_stream = measured["feedback"]["greedy"]
    gates = {
        "implementation": width_one_equal and max(score_errors) < 1e-3,
        "boundary_resource": peak < 18
        and not any(
            model.boundary_receipt()[k]
            for k in [
                "text_tokenizer",
                "symbol_ids",
                "glyph_inventory",
                "codebook",
                "ocr",
                "external_runtime_model",
            ]
        ),
        "beam_correctness_gain": comparisons["beam_minus_greedy"]["mean"] >= 0.02
        and comparisons["beam_minus_greedy"]["ci95"][0] > 0,
        "earlier_history_gain": comparisons["beam_minus_shuffled"]["mean"] >= 0.01
        and comparisons["beam_minus_shuffled"]["ci95"][0] > 0,
        "surface_and_motor": primary["exact_known_form"] >= 0.95
        and primary["blank_fraction"] < 0.02
        and measured["motor"]["beam8"]["exact_target"] >= 0.995,
        "feedback": g["exact_target"] >= greedy_stream["exact_target"]
        and g["adjacent_repeated_raster_fraction"]
        <= measured["target_stream"]["adjacent_repeated_raster_fraction"] + 0.10,
    }
    self_values = {k: torch.cat(v) for k, v in self_nll.items()}
    true_values = {k: torch.cat(v) for k, v in target_nll.items()}
    report = {
        "experiment": "raw-pixel-beam-v56",
        "decision": "smoke-only"
        if args.smoke
        else "retain_decoder"
        if all(gates.values())
        else "reject_decoder",
        "model_remains_unqualified_as_general_ilm": True,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "provenance": {
            p: file_sha256(p)
            for p in [__file__, "ilm/visual_lm/raster_beam.py", PROTOCOL]
        },
        "manifest_sha256": checkpoint["manifest"]["sha256"],
        "audit_seed": SEED,
        "beam_width": 8,
        "windows": count,
        "gates": gates,
        "measurements": measured,
        "comparisons": comparisons,
        "mean_self_nll": {k: float(v.mean()) for k, v in self_values.items()},
        "mean_true_target_nll": {k: float(v.mean()) for k, v in true_values.items()},
        "max_score_consistency_error": max(score_errors),
        "width_one_equal_to_original_greedy": width_one_equal,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gib": peak,
        "evaluator_bank_size": len(bank),
        "bank_created_after_generation": True,
        "sealed_partition_opened": False,
        "training_updates": 0,
    }
    for name, data in [
        ("generation", single),
        ("feedback", stream),
        ("motor", supplied),
    ]:
        torch.save(data, out / f"{name}_image_evidence.pt")
        cells = torch.cat(
            (data["targets"][:16], *[v[:16] for v in data["outputs"].values()]), dim=1
        )
        pack_visual_cells(cells.flatten(0, 1), columns=cells.shape[1], gutter=2).save(
            out / f"{name}_gallery.png"
        )
    torch.save(
        {"self_nll": self_values, "true_target_nll": true_values, "identifiers": ids},
        out / "per_example_likelihood.pt",
    )
    _atomic_json(
        {
            phase: {key: value.tolist() for key, value in routes.items()}
            for phase, routes in correct.items()
        },
        out / "per_example_correctness.json",
    )
    _atomic_json(report, out / "report.json")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
