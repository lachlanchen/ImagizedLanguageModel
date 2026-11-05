#!/usr/bin/env python3
"""Build research figures and TeX values from the unchanged locked reports."""

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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / "artifacts/causal_visual_block_diffusion_v53_20261104"
BLUE, RED, GRAY = "#28638b", "#ae593f", "#939b9f"


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(HERE / "figures" / f"{name}.{suffix}", dpi=240, facecolor="white")
        if suffix == "svg":
            path = HERE / "figures" / f"{name}.svg"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def block_image(cells):
    return np.concatenate([cell[0].numpy() for cell in cells], axis=1)


def main():
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    report = read(RUN / "development/development_report.json")
    training = read(RUN / "training_summary.json")
    v52 = read(
        ROOT
        / "artifacts/noisy_visual_energy_guidance_v52_20261104/development/development_report.json"
    )
    v42_path = (
        ROOT / "publication/ilm-image-native/evidence/v42/development_report.json"
    )
    v42 = read(v42_path)
    images = torch.load(
        RUN / "development/generated_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        report["checkpoint_sha256"]
        != "bd8dd3e58483d4ef527ae9a0fc4db1c0df545d18ec0fa0067ff594256d7cc71b"
    ):
        raise ValueError("unexpected V53 checkpoint")
    for path in (
        RUN / "training_summary.json",
        RUN / "development/development_report.json",
        RUN / "target_supplied_motor_diagnostic/report.json",
        v42_path,
        ROOT
        / "artifacts/noisy_visual_energy_guidance_v52_20261104/development/development_report.json",
    ):
        name = (
            "v42_report.json"
            if path == v42_path
            else "v52_report.json"
            if "v52" in str(path)
            else "v53_motor_diagnostic.json"
            if "target_supplied" in str(path)
            else "v53_" + path.name
        )
        # Delivery receipts use repository-relative paths; scientific values
        # remain unchanged. Original byte hashes are retained separately.
        text = path.read_text().replace(str(ROOT) + "/", "")
        (evidence / name).write_text(text)
    sources = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            RUN / "training_summary.json",
            RUN / "development/development_report.json",
            RUN / "development/generated_image_evidence.pt",
            RUN / "target_supplied_motor_diagnostic/report.json",
            v42_path,
            ROOT
            / "artifacts/noisy_visual_energy_guidance_v52_20261104/development/development_report.json",
        )
    }
    (evidence / "source_hashes.json").write_text(json.dumps(sources, indent=2) + "\n")
    ordered = report["generation"]["ordered"]
    comparison = report["comparisons"]["ordered_minus_shuffled"]
    values = {
        "TrainSeconds": f"{training['elapsed_seconds']:.1f}",
        "TrainMemory": f"{training['peak_allocated_vram_gib']:.3f}",
        "FirstIdentity": f"{100 * ordered['identity_top1_by_position'][0]:.2f}",
        "FirstShuffled": f"{100 * report['generation']['shuffled']['identity_top1_by_position'][0]:.2f}",
        "FirstZero": f"{100 * report['generation']['zero_condition']['identity_top1_by_position'][0]:.2f}",
        "BlockIdentity": f"{100 * ordered['mean_identity_top1']:.3f}",
        "BlockGain": f"{100 * comparison['mean']:.3f}",
        "BlockLow": f"{100 * comparison['ci95'][0]:.3f}",
        "BlockHigh": f"{100 * comparison['ci95'][1]:.3f}",
        "BlockFone": f"{ordered['mean_pixel_f1']:.3f}",
        "RunDecision": report["decision"],
    }
    (HERE / "results_values.tex").write_text(
        "% Generated from pinned audit evidence.\n"
        + "".join(
            f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items()
        )
    )
    rows = []
    for name, route in (
        ("Ordered", "ordered"),
        ("Shuffled", "shuffled"),
        ("Zero condition", "zero_condition"),
    ):
        item = report["generation"][route]
        rows.append(
            f"{name} & {100*item['identity_top1_by_position'][0]:.3f} & {100*item['mean_identity_top1']:.3f} & {item['mean_pixel_f1']:.4f} & {item['isolated_ink_fraction']:.4f} \\\\\n"
        )
    table = (
        "\\section*{Supplementary Table 1: direct visual block audit}\n"
        "\\begin{center}\\begin{tabular}{lrrrr}\\toprule\n"
        "Condition & First identity (\\%) & Block identity (\\%) & Pixel F1 & Isolated ink \\\\\n"
        "\\midrule\n" + "".join(rows) + "\\bottomrule\\end{tabular}\\end{center}\n"
        "All 256 prefixes contribute one four-cell sample. Target coverage in the Han bank is "
        "60.94, 58.98, 64.45 and 68.36\\% by position. No out-of-bank target receives identity credit.\n"
    )
    (HERE / "results_table.tex").write_text(table)
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

    # Mechanism figure: all writing thumbnails come from measured image evidence.
    fig = plt.figure(figsize=(7.25, 4.8))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(
        0, 0.97, "a   Visible writing and conditional state", fontsize=10, weight="bold"
    )
    inset = fig.add_axes([0.04, 0.73, 0.25, 0.13])
    inset.axis("off")
    inset.imshow(
        block_image(images["targets"][0]),
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.text(0.12, 0.70, "Example raster cells", ha="center")

    def node(x, y, w, h, title, subtitle):
        ax.add_patch(
            Rectangle((x, y), w, h, facecolor="#f0f4f6", edgecolor=BLUE, lw=0.8)
        )
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=9,
        )
        ax.text(
            x + w / 2, y + h * 0.27, subtitle, ha="center", va="center", fontsize=7.5
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=BLUE, lw=1
            )
        )

    node(0.36, 0.74, 0.24, 0.13, "Causal visual reader", "60 images; frozen 16.28M")
    node(0.72, 0.74, 0.27, 0.13, "Continuous condition", "hidden state + visual field")
    arrow(0.27, 0.805, 0.35, 0.805)
    arrow(0.61, 0.805, 0.71, 0.805)
    ax.text(0, 0.61, "b   Joint future-image training", fontsize=10, weight="bold")
    node(0.02, 0.38, 0.24, 0.13, "Noisy future strip", "four 32 × 32 cells")
    node(0.37, 0.38, 0.29, 0.13, "Conditional U-Net", "velocity; FiLM; cell attention")
    node(0.77, 0.38, 0.22, 0.13, "Learning signal", "velocity + image losses")
    arrow(0.27, 0.445, 0.36, 0.445)
    arrow(0.67, 0.445, 0.76, 0.445)
    arrow(0.855, 0.735, 0.56, 0.52)
    ax.text(
        0.04,
        0.32,
        "Stratified cell noise alternates with synchronous noise",
        color="#47545e",
        fontsize=8,
    )
    ax.text(
        0,
        0.23,
        "c   Raster generation and visible feedback",
        fontsize=10,
        weight="bold",
    )
    ax.text(0.03, 0.12, "Gaussian noise", fontsize=9)
    node(
        0.33,
        0.06,
        0.32,
        0.13,
        "32-level reverse process",
        "fixed guidance 2; first sample",
    )
    arrow(0.20, 0.12, 0.32, 0.12)
    arrow(0.66, 0.12, 0.73, 0.12)
    inset = fig.add_axes([0.74, 0.06, 0.24, 0.12])
    inset.axis("off")
    inset.imshow(
        block_image(images["outputs"]["ordered"][0]),
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.text(0.86, 0.015, "Actual generated raster", ha="center", fontsize=7.5)
    save(fig, "Figure_1_mechanism")

    fig = plt.figure(figsize=(7.25, 6.3), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.9))
    ax = fig.add_subplot(grid[0, 0])
    reader_values = [
        100 * v42["language"][name]
        for name in ("full_top1", "shuffled_top1", "bigram_top1", "unigram_top1")
    ]
    ax.barh(
        ["Full history", "Shuffled", "Bigram", "Unigram"],
        reader_values,
        color=[BLUE, GRAY, GRAY, GRAY],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 25)
    ax.set_xlabel("Next-image top-1 (%)")
    ax.set_title(
        "a  Completed-image reader ranking", loc="left", weight="bold", fontsize=9
    )
    ax = fig.add_subplot(grid[0, 1])
    ax.bar(
        ["Clean rank", "First sample", "Selected / 4"],
        [100 * v52["clean_energy"]["ordered"]["identity_top1"], 0, 100 / 256],
        color=[BLUE, RED, RED],
    )
    ax.set_ylim(0, 25)
    ax.set_ylabel("Identity (%)")
    ax.set_title(
        "b  Prior + critic generation gap", loc="left", weight="bold", fontsize=9
    )
    ax.tick_params(axis="x", labelsize=7)
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(4)
    for offset, route, name, color in (
        (-0.22, "ordered", "Ordered", BLUE),
        (0, "shuffled", "Shuffled", GRAY),
        (0.22, "zero_condition", "Zero condition", RED),
    ):
        ax.bar(
            x + offset,
            np.array(report["generation"][route]["identity_top1_by_position"]) * 100,
            width=0.21,
            color=color,
            label=name,
        )
    ax.set(
        xticks=x,
        xticklabels=["1", "2", "3", "4"],
        xlabel="Future-cell position",
        ylabel="Identity (%)",
        ylim=(0, 2),
    )
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
    ax.set_title(
        "c  Direct block generation, paired noise",
        loc="left",
        weight="bold",
        fontsize=9,
    )
    gallery_grid = grid[2, :].subgridspec(1, 4, wspace=0.12)
    for col, (key, title) in enumerate(
        (
            (None, "Target"),
            ("ordered", "Ordered"),
            ("shuffled", "Shuffled"),
            ("zero_condition", "Zero condition"),
        )
    ):
        ax = fig.add_subplot(gallery_grid[0, col])
        ax.axis("off")
        array = images["targets"] if key is None else images["outputs"][key]
        display = np.concatenate([block_image(row) for row in array[:7]], axis=0)
        ax.imshow(display, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(("d  " if col == 0 else "") + title, fontsize=8, weight="bold")
    save(fig, "Figure_2_evidence")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
