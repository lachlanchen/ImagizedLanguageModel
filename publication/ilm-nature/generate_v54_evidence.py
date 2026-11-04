#!/usr/bin/env python3
"""Produce the exact-raster result from immutable audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, Rectangle

from generate_evidence import HERE, ROOT, save

RUN = ROOT / "artifacts/exact_raster_density_v54_20261104"
CHECKPOINT = "f2c3fab74981541243a963c8cffab6bbbd8c7f9d051512b228347db04a57a9ee"
BLUE, RED, GRAY = "#28638b", "#ae593f", "#939b9f"


def main():
    report = json.loads((RUN / "development/development_report.json").read_text())
    training = json.loads((RUN / "training_summary.json").read_text())
    if report["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("unexpected V54 checkpoint")
    if report["evaluator_bank_size"] != 9932:
        raise ValueError("unexpected identity evaluator")
    ordered = report["generation"]["ordered"]
    advantage = report["likelihood"]["ordered_nll_advantage_over_shuffled"]
    values = {
        "DensitySeconds": f"{training['elapsed_seconds']:.1f}",
        "DensityMemory": f"{training['peak_allocated_vram_gib']:.3f}",
        "DensityNll": f"{report['likelihood']['ordered']['nats_per_image']:.3f}",
        "DensityShuffledNll": f"{report['likelihood']['shuffled']['nats_per_image']:.3f}",
        "DensityAdvantage": f"{advantage['mean']:.4f}",
        "DensityLow": f"{advantage['ci95'][0]:.4f}",
        "DensityHigh": f"{advantage['ci95'][1]:.4f}",
        "DensityIdentity": f"{100 * ordered['identity_top1']:.3f}",
        "DensityFone": f"{ordered['pixel_f1']:.3f}",
        "DensityCopyFone": f"{report['motor']['copy_sample']['pixel_f1']:.5f}",
        "DensityDecision": report["decision"],
    }
    (HERE / "density_values.tex").write_text(
        "% Generated from the pinned V54 audit; no fitted presentation values.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items())
    )
    sources = [
        RUN / "training_summary.json",
        RUN / "development/development_report.json",
        RUN / "development/generation_image_evidence.pt",
        RUN / "development/motor_image_evidence.pt",
        RUN / "development/feedback_image_evidence.pt",
        RUN / "development/per_example_likelihood.pt",
    ]
    for path in sources[:2]:
        (HERE / "evidence" / ("v54_" + path.name)).write_text(
            path.read_text().replace(str(ROOT) + "/", "")
        )
    (HERE / "evidence/v54_source_hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
            indent=2,
        )
        + "\n"
    )
    rows = []
    for label, section, route in (
        ("Ordered sample", "generation", "ordered"),
        ("Shuffled sample", "generation", "shuffled"),
        ("Zero-condition sample", "generation", "zero"),
        ("Ordered greedy", "generation", "greedy"),
        ("Supplied image, sample", "motor", "copy_sample"),
        ("Supplied image, greedy", "motor", "copy_greedy"),
        ("Deranged image, greedy", "motor", "deranged_greedy"),
    ):
        item = report[section][route]
        rows.append(
            f"{label} & {100*item['identity_top1']:.3f} & "
            f"{100*item['exact_image_match']:.3f} & {item['pixel_f1']:.5f} & "
            f"{item['ink_ratio']:.3f} \\\\\n"
        )
    (HERE / "density_table.tex").write_text(
        "\\section*{Supplementary Table 2: exact raster density audit}\n"
        "\\begin{center}\\begin{tabular}{lrrrr}\\toprule\n"
        "Condition & Identity (\\%) & Exact image (\\%) & Pixel F1 & Ink ratio \\\\\n"
        "\\midrule\n" + "".join(rows) + "\\bottomrule\\end{tabular}\\end{center}\n"
        "Each row uses 256 images. The evaluator bank contains 9,932 distinct rasters "
        "and covers every target. Supplied-image rows reveal the intended image and "
        "are motor controls, not language results.\n"
    )
    images = torch.load(
        RUN / "development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    motor = torch.load(
        RUN / "development/motor_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    if not torch.equal(images["targets"], motor["targets"]):
        raise ValueError("motor and generation rows are not aligned")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.25, 6.25), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(0.95, 1.15, 1.6))
    ax = fig.add_subplot(grid[0, :])
    ax.axis("off")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_title(
        "a  Exact image probability with an aligned visual plan",
        loc="left",
        weight="bold",
        fontsize=9,
    )
    for x, w, title, subtitle in (
        (0.00, 0.24, "Visible history", "Frozen causal reader"),
        (0.34, 0.29, "Row transformer + MADE", "Earlier rows and pixels only"),
        (0.73, 0.27, "Next raster", "1,024 Bernoulli factors"),
    ):
        ax.add_patch(
            Rectangle((x, 0.42), w, 0.43, facecolor="#eef3f7", edgecolor=BLUE, lw=0.8)
        )
        ax.text(
            x + w / 2, 0.70, title, ha="center", va="center", fontsize=8, weight="bold"
        )
        ax.text(x + w / 2, 0.52, subtitle, ha="center", va="center", fontsize=7)
    for x1, x2 in ((0.245, 0.333), (0.637, 0.723)):
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.64), (x2, 0.64), arrowstyle="-|>", mutation_scale=9, color=BLUE
            )
        )
    ax.text(
        0.5,
        0.17,
        "Aligned inverse-field logits + learned causal residual; pure image NLL",
        ha="center",
        fontsize=8,
    )

    ax = fig.add_subplot(grid[1, 0])
    low, high = advantage["ci95"]
    mean = advantage["mean"]
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.errorbar(
        mean, 0, xerr=[[mean - low], [high - mean]], fmt="o", color=BLUE, capsize=4
    )
    ax.set(
        yticks=[],
        ylim=(-0.65, 0.65),
        xlim=(-0.15, 0.22),
        xlabel="Ordered NLL advantage (nats/image)",
    )
    ax.set_title(
        "b  Earlier-history effect is unresolved", loc="left", weight="bold", fontsize=9
    )
    ax.text(
        0.02,
        0.83,
        f"Ordered {values['DensityNll']}; shuffled {values['DensityShuffledNll']}",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    ax.text(
        0.02,
        0.08,
        "512 targets; 2,000 source-record bootstrap draws",
        transform=ax.transAxes,
        fontsize=7,
    )

    ax = fig.add_subplot(grid[1, 1])
    categories = ["Copy / greedy", "Copy / sample", "Next / sample"]
    f1 = [
        report["motor"]["copy_greedy"]["pixel_f1"],
        report["motor"]["copy_sample"]["pixel_f1"],
        ordered["pixel_f1"],
    ]
    identity = [1, 1, ordered["identity_top1"]]
    x = np.arange(3)
    ax.bar(x - 0.17, np.array(f1) * 100, 0.32, color=BLUE, label="Pixel F1 × 100")
    ax.bar(
        x + 0.17,
        np.array(identity) * 100,
        0.32,
        color=RED,
        label="Nearest identity (%)",
    )
    ax.set(xticks=x, xticklabels=categories, ylim=(0, 123), yticks=[0, 50, 100])
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", handlelength=1)
    ax.set_title(
        "c  Copying succeeds; continuation fails", loc="left", weight="bold", fontsize=9
    )
    gallery = grid[2, :].subgridspec(1, 5, wspace=0.18)
    arrays = [
        images["targets"],
        motor["outputs"]["copy_greedy"],
        images["outputs"]["ordered"],
        images["outputs"]["shuffled"],
        images["outputs"]["greedy"],
    ]
    labels = [
        "d  Target",
        "Supplied / greedy",
        "Ordered / sample",
        "Shuffled / sample",
        "Ordered / greedy",
    ]
    for col, (array, label) in enumerate(zip(arrays, labels)):
        ax = fig.add_subplot(gallery[0, col])
        # The first eight rows are fixed before seeing the result; no reranking.
        display = np.concatenate([cell[0, 0].numpy() for cell in array[:8]], axis=0)
        ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        ax.set_title(label, fontsize=7.5, weight="bold")
    save(fig, "Figure_3_exact_raster")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
