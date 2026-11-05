#!/usr/bin/env python3
"""Measured block, frozen-foundation and input-order figures from locked audits."""

from __future__ import annotations

import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

from generate_evidence import HERE, ROOT, save
from generate_v57_evidence import pad_answer

RUNS = {
    "v58": ROOT / "artifacts/visual_answer_blocks_v58_20261105",
    "v59": ROOT / "artifacts/frozen_visual_foundation_v59_20261105",
    "v60": ROOT / "artifacts/question_first_reading_v60_20261105",
}


def score_chart(ax, report, routes, labels, *, maximum=45):
    x = np.arange(len(routes))
    m = report["measurements"]
    for delta, key, title, color in [
        (-.17, "exact_answer", "Exact answer", "#28638b"),
        (.17, "raster_f1", "Multiset raster F1", "#ae593f"),
    ]:
        values = [100 * m[r][key] for r in routes]
        ax.bar(x + delta, values, .32, label=title, color=color)
        for pos, value in zip(x + delta, values):
            ax.text(pos, value + .65, f"{value:.1f}", ha="center", fontsize=7)
    ax.set(xticks=x, xticklabels=labels, ylabel="Answer score (%)", ylim=(0, maximum))
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def mechanism(ax, *, order=False):
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    if order:
        for y, label, first, second in [
            (.75, "Context first", "Passage images X", "Question images Q"),
            (.25, "Question first", "Question images Q", "Passage images X"),
        ]:
            ax.text(0, y, label, va="center", fontsize=8, fontweight="bold")
            for x, width, txt in [(.22, .28, first), (.54, .28, second), (.86, .13, "Answer")]:
                ax.add_patch(FancyBboxPatch((x, y-.15), width, .30,
                    boxstyle="round,pad=0.006", facecolor="#edf2f5", edgecolor="#647985", lw=.7))
                ax.text(x+width/2, y, txt, ha="center", va="center", fontsize=7)
            for start, end in [(.50, .54), (.82, .86)]:
                ax.annotate("", xy=(end, y), xytext=(start, y),
                            arrowprops=dict(arrowstyle="->", lw=.8))
    else:
        boxes = [(.01, .18, "Prompt images\nquestion + passage"),
                 (.25, .19, "Causal reader\ncontinuous states"),
                 (.50, .22, "Joint start / end\nall observed blocks"),
                 (.78, .21, "Predicted answer\nactual source pixels")]
        for x, width, label in boxes:
            ax.add_patch(FancyBboxPatch((x, .32), width, .5,
                boxstyle="round,pad=0.006", facecolor="#edf2f5", edgecolor="#647985", lw=.7))
            ax.text(x+width/2, .57, label, ha="center", va="center", fontsize=7)
        for start, end in [(.19, .25), (.44, .50), (.72, .78)]:
            ax.annotate("", xy=(end, .57), xytext=(start, .57),
                        arrowprops=dict(arrowstyle="->", lw=.8))
        ax.text(.5, .06, "Equal image sequences share probability mass; end position is learned.",
                ha="center", fontsize=7)


def raw_rows(ax, report):
    import sys
    sys.path.insert(0, str(ROOT))
    from ilm.visual_lm.visual_qa_data import VisualQARenderer
    renderer = VisualQARenderer()
    records = {r["id"]: r for r in map(json.loads,
        (ROOT / "data/visual_qa/cmrc2018_short/development.jsonl").read_text().splitlines())}
    ids = report["primary_ids"][:2]
    ax.axis("off")
    ax.set(xlim=(0, 19 * 32), ylim=(5.8 * 32, 0))
    for i, identifier in enumerate(ids):
        r = records[identifier]
        q = np.concatenate([x[0].numpy() for x in renderer(r["question"])], 1)
        y = i * 3.0 * 32
        ax.imshow(q, cmap="gray_r", vmin=0, vmax=1,
                  extent=(65, 65 + q.shape[1]*.60, y+19, y), interpolation="nearest")
        ax.text(0, y+15, f"Q{i+1}", fontsize=7, fontweight="bold")
        for j, (label, images) in enumerate([
            ("Reference", renderer(r["references"][0]["text"])),
            ("Block", torch.load(RUNS["v58"] / "development/predictions/block" /
                               (identifier + ".pt"), map_location="cpu", weights_only=False)["images"]),
        ]):
            yy = y + 26 + j * 26
            ax.text(0, yy+16, label, fontsize=7)
            bitmap = pad_answer(images, 16)
            ax.imshow(bitmap, cmap="gray_r", vmin=0, vmax=1,
                      extent=(65, 65+16*19, yy+19, yy), interpolation="nearest")


def main():
    reports = {k: json.loads((v / "development/report.json").read_text()) for k, v in RUNS.items()}
    if any(len(r["primary_ids"]) != 256 for r in reports.values()):
        raise ValueError("completed production audits required")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    r = reports["v58"]
    fig = plt.figure(figsize=(7.25, 5.45), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=(.9, 1.8, 1.65))
    a, b, c = [fig.add_subplot(grid[i]) for i in range(3)]
    mechanism(a); score_chart(b, r, ["block", "block_blank_question", "glyph"],
                             ["Whole image block", "Question blanked", "Matched glyph answers"])
    raw_rows(c, r)
    for ax, label in zip((a, b, c), "abc"):
        ax.text(-.04, 1.04, label, transform=ax.transAxes, fontweight="bold")
    save(fig, "Figure_5_image_blocks")
    fig, ax = plt.subplots(figsize=(7.25, 2.9), layout="constrained")
    score_chart(ax, reports["v59"], ["student", "student_blank", "mixar", "mixar_blank"],
                ["Student", "Student / blank Q", "MIXAR", "MIXAR / blank Q"])
    save(fig, "Figure_S5_frozen_foundations")
    fig, (a, b) = plt.subplots(2, 1, figsize=(7.25, 4.1),
                              gridspec_kw={"height_ratios": (1, 2)}, layout="constrained")
    mechanism(a, order=True)
    maximum = max(45, 10 + 100 * max(x["raster_f1"] for x in reports["v60"]["measurements"].values()))
    score_chart(b, reports["v60"], ["context", "context_blank", "question", "question_blank"],
                ["Context first", "Context / blank Q", "Question first", "Question / blank Q"], maximum=maximum)
    save(fig, "Figure_S6_question_first")
    for version, run in RUNS.items():
        for source in [run / "development/report.json"] + sorted(run.glob("*/training_report.json")):
            name = f"{version}_{source.parent.name}_{source.name}"
            (HERE / "evidence" / name).write_text(source.read_text().replace(str(ROOT)+"/", ""))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for run in RUNS.values() for p in [run / "development/report.json"] + sorted(run.glob("*/training_report.json"))}
    (HERE / "evidence/reading_intervention_source_hashes.json").write_text(json.dumps(hashes, indent=2)+"\n")
    print("Generated three measured reading figures")


if __name__ == "__main__":
    main()
