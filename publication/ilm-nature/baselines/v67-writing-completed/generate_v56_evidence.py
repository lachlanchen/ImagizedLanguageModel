#!/usr/bin/env python3
"""Publish immutable form-validity and raw-pixel decoding evidence."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_evidence import HERE, ROOT, save

RUN = ROOT / "artifacts/raster_beam_v56_20261105"
FORMS = ROOT / "artifacts/raster_form_validity_20261105"


def main():
    report = json.loads((RUN / "report.json").read_text())
    forms = json.loads((FORMS / "report.json").read_text())
    if (
        report["checkpoint_sha256"]
        != "ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394"
        or report["windows"] != 2048
        or report["float32_matmul_precision"] != "highest"
    ):
        raise ValueError("unexpected V56 audit")
    m = report["measurements"]
    c = report["comparisons"]
    values = {
        "BeamAccuracy": f"{100*m['generation']['beam8']['exact_target']:.3f}",
        "BeamShuffled": f"{100*m['generation']['beam8_shuffled']['exact_target']:.3f}",
        "GreedyAccuracy": f"{100*m['generation']['greedy']['exact_target']:.3f}",
        "BeamGain": f"{100*c['beam_minus_greedy']['mean']:.3f}",
        "BeamGainLow": f"{100*c['beam_minus_greedy']['ci95'][0]:.3f}",
        "BeamGainHigh": f"{100*c['beam_minus_greedy']['ci95'][1]:.3f}",
        "BeamHistoryGain": f"{100*c['beam_minus_shuffled']['mean']:.3f}",
        "BeamHistoryLow": f"{100*c['beam_minus_shuffled']['ci95'][0]:.3f}",
        "BeamHistoryHigh": f"{100*c['beam_minus_shuffled']['ci95'][1]:.3f}",
        "BeamRepeat": f"{100*m['feedback']['beam8']['adjacent_repeated_raster_fraction']:.3f}",
        "TargetRepeat": f"{100*m['target_stream']['adjacent_repeated_raster_fraction']:.3f}",
        "BeamSeconds": f"{report['elapsed_seconds']:.1f}",
        "BeamMemory": f"{report['peak_allocated_vram_gib']:.3f}",
        "BeamDecision": report["decision"].replace("_", r"\_"),
    }
    (HERE / "beam_values.tex").write_text(
        "% Generated from unchanged V56 report.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items())
    )
    sources = [
        RUN / "report.json",
        RUN / "generation_image_evidence.pt",
        RUN / "feedback_image_evidence.pt",
        FORMS / "report.json",
        ROOT / "artifacts/raster_beam_v56_numerical_preflight/precision_report.json",
    ]
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (HERE / "evidence/v56_source_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    for path, name in [
        (RUN / "report.json", "v56_report.json"),
        (FORMS / "report.json", "v55_form_validity.json"),
        (sources[-1], "v56_precision_preflight.json"),
    ]:
        (HERE / "evidence" / name).write_text(
            path.read_text().replace(str(ROOT) + "/", "")
        )
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.7), layout="constrained")
    x = np.arange(3)
    runs = ["v54", "v55_frozen", "v55_joint"]
    for ax, phase in zip(axes, ["generation", "feedback"]):
        ax.bar(
            x - 0.17,
            [
                100
                * forms["summaries"][n][phase]["ordered"]["exact_known_raster_fraction"]
                for n in runs
            ],
            0.32,
            label="Exact known form",
            color="#28638b",
        )
        ax.bar(
            x + 0.17,
            [
                100 * forms["summaries"][n][phase]["ordered"]["exact_target_fraction"]
                for n in runs
            ],
            0.32,
            label="Exact observed target",
            color="#ae593f",
        )
        ax.set(
            xticks=x,
            xticklabels=["V54", "V55 frozen", "V55 joint"],
            ylabel="Saved image fraction (%)",
            ylim=(0, 100),
        )
        ax.set_title(
            "First sample" if phase == "generation" else "Four-step visible feedback",
            weight="bold",
            fontsize=9,
        )
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "Figure_S3_form_validity")
    raw = torch.load(
        RUN / "feedback_image_evidence.pt", map_location="cpu", weights_only=False
    )
    fig = plt.figure(figsize=(7.25, 4.5), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.1))
    ax = fig.add_subplot(grid[0, 0])
    routes = ["beam8", "beam8_shuffled", "greedy"]
    ax.bar(
        ["Beam", "Shuffled beam", "Greedy"],
        [100 * m["generation"][r]["exact_target"] for r in routes],
        color=["#28638b", "#ae593f", "#939b9f"],
    )
    ax.set(ylabel="Exact next-image match (%)", ylim=(0, 15))
    ax.set_title(
        "a  Search improves conditional choice", loc="left", weight="bold", fontsize=9
    )
    ax = fig.add_subplot(grid[0, 1])
    ax.bar(
        ["Target", "Beam", "Shuffled", "Greedy"],
        [100 * m["target_stream"]["adjacent_repeated_raster_fraction"]]
        + [100 * m["feedback"][r]["adjacent_repeated_raster_fraction"] for r in routes],
        color=["#66786c", "#28638b", "#ae593f", "#939b9f"],
    )
    ax.set(ylabel="Adjacent repeated images (%)", ylim=(0, 40))
    ax.set_title(
        "b  Repetition fails the rollout test", loc="left", weight="bold", fontsize=9
    )
    panel = grid[1, :].subgridspec(1, 4, wspace=0.08)
    for i, (label, array) in enumerate(
        [
            ("c  Observed future", raw["targets"]),
            ("Ordered beam", raw["outputs"]["beam8"]),
            ("Shuffled beam", raw["outputs"]["beam8_shuffled"]),
            ("Original greedy", raw["outputs"]["greedy"]),
        ]
    ):
        ax = fig.add_subplot(panel[0, i])
        ax.axis("off")
        rows = [
            np.concatenate([cell[0].numpy() for cell in source], axis=1)
            for source in array[:4]
        ]
        ax.imshow(
            np.concatenate(rows, axis=0),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=7.5, weight="bold")
    save(fig, "Figure_S4_pixel_beam")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
