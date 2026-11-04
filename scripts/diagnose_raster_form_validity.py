#!/usr/bin/env python3
"""Read-only canonical-form validity of saved V54/V55 generated images."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig
from ilm.visual_lm.visual_cell_data import load_v25_records
from scripts.eval_exact_raster_density_v54 import evaluator_bank, grouped_bootstrap
from scripts.train_canonical_glyph_binding_v43 import _atomic_json
from scripts.train_visual_state_actuator import file_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/raster_form_validity_20261105"
RUNS = {
    "v54": ROOT / "artifacts/exact_raster_density_v54_20261104",
    "v55_frozen": ROOT / "artifacts/joint_exact_raster_v55_20261105/frozen",
    "v55_joint": ROOT / "artifacts/joint_exact_raster_v55_20261105/joint",
}
CHECKPOINTS = {
    "v54": "f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee",
    "v55_frozen": "0f993f484c7998629a1adf1dc94f60044c3f23f9ef95fd1459a4636c7f2b6c5a",
    "v55_joint": "ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394",
}


def raster_keys(images):
    flat = images.reshape(-1, 1024)
    if not bool(((flat == 0) | (flat == 1)).all()):
        raise ValueError("saved evidence must already be binary; no threshold repair")
    packed = np.packbits(flat.numpy().astype(np.uint8), axis=1)
    return [row.tobytes() for row in packed]


def measure(images, targets, bank, known):
    b, t = images.shape[:2]
    flat = images.flatten(0, 1).reshape(-1, 1024)
    keys = raster_keys(flat)
    exact = np.asarray([key in known for key in keys], dtype=bool).reshape(b, t)
    ink = flat.sum(-1).numpy().reshape(b, t)
    match = (images == targets).flatten(2).all(-1).numpy()
    bank_flat = bank.reshape(-1, 1024)
    bank_ink = bank_flat.sum(-1)
    f1 = []
    for chunk in flat.split(64):
        intersection = chunk @ bank_flat.T
        denominator = chunk.sum(-1, keepdim=True) + bank_ink[None]
        f1.append((2 * intersection / denominator.clamp_min(1)).max(-1).values)
    f1 = torch.cat(f1).numpy().reshape(b, t)
    counts = Counter(key for key in keys if key in known)
    return {
        "examples": b,
        "cells_per_example": t,
        "exact_known_raster_fraction": float(exact.mean()),
        "exact_nonblank_fraction": float((exact & (ink >= 2)).mean()),
        "exact_target_fraction": float(match.mean()),
        "blank_fraction": float((ink < 2).mean()),
        "distinct_exact_known_rasters": len(counts),
        "most_frequent_exact_count": max(counts.values(), default=0),
        "nearest_ink_f1_mean": float(f1.mean()),
        "nearest_ink_f1_quantiles": np.quantile(f1, [0.1, 0.5, 0.9]).tolist(),
    }, {
        "exact_form": exact.tolist(),
        "exact_target": match.tolist(),
        "nearest_ink_f1": f1.tolist(),
    }


def main():
    if (OUT / "report.json").exists():
        raise FileExistsError("completed post-result diagnostic is immutable")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    records = load_v25_records(
        "data/visual_grammar/chinese_wikisource_public_domain.jsonl",
        strict_manifest=True,
    )
    summaries, rows, fingerprints, sources = {}, {}, {}, {}
    for name, run in RUNS.items():
        report_path = run / "development/development_report.json"
        report = json.loads(report_path.read_text())
        if report["checkpoint_sha256"] != CHECKPOINTS[name]:
            raise ValueError("unexpected saved checkpoint result")
        if (
            file_sha256(
                ROOT / "data/visual_grammar/chinese_wikisource_public_domain.jsonl"
            )
            != report["manifest_sha256"]
        ):
            raise ValueError("corpus changed")
        evidence = {}
        for phase in ("generation", "feedback", "motor"):
            path = run / "development" / f"{phase}_image_evidence.pt"
            evidence[phase] = torch.load(path, map_location="cpu", weights_only=False)
            sources[str(path.relative_to(ROOT))] = file_sha256(path)
        bank = evaluator_bank(
            records, CanonicalGlyphRenderConfig(), tuple(evidence.values())
        )
        bank_keys = raster_keys(bank)
        if len(bank) != 9932 or len(set(bank_keys)) != 9932:
            raise ValueError("canonical diagnostic bank differs")
        fingerprints[name] = hashlib.sha256(b"".join(sorted(bank_keys))).hexdigest()
        known = set(bank_keys)
        summaries[name], rows[name] = {}, {}
        for phase in ("generation", "feedback"):
            data = evidence[phase]
            if not all(key in known for key in raster_keys(data["targets"])):
                raise ValueError("target coverage is incomplete")
            summaries[name][phase], rows[name][phase] = (
                {},
                {"identifiers": data["identifiers"], "routes": {}},
            )
            for route, images in data["outputs"].items():
                stats, values = measure(images, data["targets"], bank, known)
                summaries[name][phase][route] = stats
                rows[name][phase]["routes"][route] = values
    if len(set(fingerprints.values())) != 1:
        raise ValueError("cross-run evaluator bank differs")
    paired = {}
    for phase in ("generation", "feedback"):
        control, joint = rows["v55_frozen"][phase], rows["v55_joint"][phase]
        if control["identifiers"] != joint["identifiers"]:
            raise ValueError("matched diagnostic rows differ")
        paired[phase] = {}
        for key in ("exact_form", "exact_target"):
            delta = np.asarray(joint["routes"]["ordered"][key], dtype=float).mean(
                1
            ) - np.asarray(control["routes"]["ordered"][key], dtype=float).mean(1)
            paired[phase][key] = grouped_bootstrap(delta, joint["identifiers"])
    report = {
        "post_result_exploratory": True,
        "primary_decisions_unchanged": True,
        "checkpoint_sha256": CHECKPOINTS,
        "bank_size": 9932,
        "bank_fingerprints": fingerprints,
        "target_coverage": 1.0,
        "source_hashes": sources,
        "diagnostic_source_sha256": file_sha256(__file__),
        "plan_sha256": file_sha256(ROOT / "references/v55_form_validity_plan.md"),
        "summaries": summaries,
        "paired_joint_minus_frozen": paired,
        "elapsed_seconds": time.perf_counter() - started,
        "new_samples_generated": 0,
        "model_or_ocr_calls": 0,
        "sealed_partition_opened": False,
    }
    _atomic_json(rows, OUT / "per_example.json")
    _atomic_json(report, OUT / "report.json")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), layout="constrained")
    labels = ["V54", "V55 frozen", "V55 joint"]
    x = np.arange(3)
    for axis, phase in zip(axes, ("generation", "feedback")):
        form = [
            100 * summaries[n][phase]["ordered"]["exact_known_raster_fraction"]
            for n in RUNS
        ]
        target = [
            100 * summaries[n][phase]["ordered"]["exact_target_fraction"] for n in RUNS
        ]
        axis.bar(x - 0.17, form, 0.32, label="Exact known form", color="#28638b")
        axis.bar(x + 0.17, target, 0.32, label="Exact observed target", color="#ae593f")
        axis.set(
            xticks=x,
            xticklabels=labels,
            ylabel="Saved image fraction (%)",
            ylim=(0, 100),
        )
        axis.set_title(
            "First sample" if phase == "generation" else "Four-step visible feedback"
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"form_validity.{suffix}", dpi=220)
    plt.close(fig)
    print(json.dumps({"summaries": summaries, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
