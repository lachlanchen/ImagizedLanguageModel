#!/usr/bin/env python3
"""Completed assignment evidence and the separately executed successor actor."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
import torch

from generate_evidence import HERE, ROOT, save
from generate_v57_evidence import pad_answer

RUN = ROOT/"artifacts/question_assignment_v63_20261105"
PREFLIGHT = ROOT/"artifacts/grounded_image_actor_preflight_20261105/attempt03"
BLUE, RED, GRAY = "#28638b", "#ae593f", "#939b9f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_report():
    if json.loads((RUN/"pipeline_status.json").read_text())["status"] != "audit_complete":
        raise ValueError("completed V63 audit required")
    report = json.loads((RUN/"development/report.json").read_text())
    receipt = json.loads((ROOT/"references/evidence/question_assignment_v63_result.json").read_text())
    if sha(RUN/"development/report.json") != receipt["files_sha256"][str((RUN/"development/report.json").relative_to(ROOT))]:
        raise ValueError("verified primary report changed")
    for arm in ("control", "aligned"):
        row = json.loads((RUN/arm/"training_report.json").read_text())
        if row["update"] != 2000 or row["smoke_only"]:
            raise ValueError("both complete matched arms required")
    return report


def raw_images(report):
    from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
    records = {r["id"]: r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")}
    renderer = VisualQARenderer()
    folder = HERE/"evidence/v63_raw"
    folder.mkdir(exist_ok=True)
    images = []
    for row, identifier in enumerate(report["primary_ids"][:2]):
        for route in ("reference", "control", "aligned"):
            cells = renderer(records[identifier]["references"][0]["text"]) if route == "reference" else torch.load(
                RUN/"development/predictions"/route/(identifier+".pt"), map_location="cpu", weights_only=False)["images"]
            if len(cells) > 20:
                raise ValueError("display would remove generated ink")
            image = (255*(1-pad_answer(cells, 20))).astype(np.uint8)
            path = folder/f"{row}_{route}.png"
            Image.fromarray(image).save(path)
            images.append((row, route, image))
    return images


def assignment_figure(report, images):
    fig = plt.figure(figsize=(7.25, 5.9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.2), width_ratios=(1, 1.28))
    a, b, c = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    routes = ("control", "control_blank", "aligned", "aligned_blank")
    values = [100*report["measurements"][r]["raster_f1"] for r in routes]
    a.bar(range(4), values, width=.62, color=(GRAY, "#ccd3d7", BLUE, "#9fbdcf"))
    for i, value in enumerate(values):
        a.text(i, value+.12, f"{value:.3f}", ha="center", fontsize=7)
    a.set(xticks=range(4), xticklabels=("Control", "Blank Q", "Aligned", "Blank Q"),
          ylim=(0, 4.2), ylabel="Strict-raster F1 (%)", title="Autonomous writing")
    contrasts = ("aligned_minus_control_raster_f1", "control_minus_control_blank_raster_f1", "aligned_minus_aligned_blank_raster_f1")
    names = ("Aligned − control", "Control − blank Q", "Aligned − blank Q")
    for i, (key, label) in enumerate(zip(contrasts, names)):
        item = report["comparisons"][key]
        mean = 100*item["mean"]
        lo, hi = np.array(item["ci95"])*100
        b.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color=RED if i == 0 else BLUE,
            capsize=3, markersize=4, linewidth=1.2)
    b.axvline(0, color="#a8afb2", linewidth=.7, zorder=0)
    b.set(yticks=range(3), yticklabels=names, ylim=(2.65, -.65), xlim=(-1.8, 3.8),
          xlabel="Raster F1 difference (percentage points)", title="Paired effects and 95% intervals")
    b.tick_params(axis="y", labelsize=7)
    c.axis("off")
    c.set(xlim=(0, 640), ylim=(220, 0))
    c.set_title("First two fixed audit rows", loc="left", fontsize=8, weight="bold")
    for row, route, image in images:
        j = ("reference", "control", "aligned").index(route)
        y = 5+row*106+j*32
        c.text(0, y+15, f"{row+1}: {route.capitalize()}", va="center", fontsize=7)
        c.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="none",
            extent=(100, 612, y+25.6, y), aspect="equal")
    c.text(0, 218, "White display padding only. Both reading gates fail.", fontsize=7)
    for axis, label in zip((a, b, c), "abc"):
        axis.text(-.12 if axis != c else -.04, 1.05, label, transform=axis.transAxes, weight="bold")
    save(fig, "Figure_8_question_assignment")


def grounded_figure(preflight):
    fig, ax = plt.subplots(figsize=(7.25, 4.4), layout="constrained")
    ax.axis("off")
    ax.set(xlim=(0, 100), ylim=(0, 100))
    boxes = ((1, 61, 22, 17, "Observed prompt image\nCompleted blocks"),
        (28, 61, 20, 17, "27-layer vision encoder\nFrozen, FP32"),
        (54, 61, 23, 17, "36-layer contextual body\nRank-eight adapters"),
        (80, 61, 19, 17, "Two continuous\nwriting queries"),
        (54, 30, 23, 17, "Six-layer flow writer\n19.37M parameters"),
        (80, 30, 19, 17, "32 × 64\noutput pixels"))
    for x, y, w, h, label in boxes:
        trainable = x >= 54 and (x, y) != (80, 30)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#eaf2f7" if trainable else "#f2f4f5",
            edgecolor=BLUE if trainable else "#798790", linewidth=.8))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=7.5)
    for start, end in (((23, 69.5), (28, 69.5)), ((48, 69.5), (54, 69.5)),
                       ((77, 69.5), (80, 69.5)), ((89.5, 61), (89.5, 53)),
                       ((89.5, 53), (65.5, 53)), ((65.5, 53), (65.5, 47)),
                       ((77, 38.5), (80, 38.5))):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->" if end in ((28,69.5),(54,69.5),(80,69.5),(65.5,47),(80,38.5)) else "-",
            mutation_scale=8, linewidth=.85, color="#536674"))
    ax.text(12, 45, "Each completed block is an\nindependent encoder image.\nFuture blocks stay hidden.", fontsize=7.5, ha="left")
    ax.text(1, 94, "Complete image actor: execution verified", fontsize=11, weight="bold")
    ax.text(1, 85, "7.548B total parameters    25.58M trainable    17.181 GiB peak allocated", fontsize=8)
    ax.text(1, 19, r"$Y_t=(1-t)U+tY,\quad U\sim\mathcal{N}(0,I),\quad\mathcal{L}_{\rm flow}=\mathbb{E}\|v_\theta(Y_t,t,h)-(Y-U)\|^2$", fontsize=10)
    ax.text(1, 9, "Frozen contextual weights use BF16 storage and FP32 computation. A separate head predicts stopping.", fontsize=7)
    ax.text(1, 2, "No language vocabulary or external model call in the actor. Writing and language quality remain unqualified.", fontsize=7)
    save(fig, "Figure_S7_grounded_actor")


def copy_evidence():
    sources = {"v63_development_report.json": RUN/"development/report.json",
        "v63_preflight_report.json": RUN/"preflight/report.json",
        "grounded_actor_preflight.json": PREFLIGHT/"report.json",
        "inherited_context_check.json": ROOT/"artifacts/grounded_context_transfer_20261105/check/report.json",
        "grounded_writer_component.json": ROOT/"artifacts/grounded_writer_learning_v64_20261105/report.json",
        "grounded_writer_verification.json": ROOT/"references/evidence/grounded_writer_v64_result.json"}
    for arm in ("control", "aligned"):
        sources[f"v63_{arm}_training_report.json"] = RUN/arm/"training_report.json"
        sources[f"v63_{arm}_standalone_verification.json"] = RUN/"standalone"/arm/"verification.json"
    for name, path in sources.items():
        (HERE/"evidence"/name).write_text(path.read_text().replace(str(ROOT)+"/", "").replace(str(Path.home())+"/", "~/"))
    manifest = {str(path.relative_to(ROOT)): sha(path) for path in sources.values()}
    (HERE/"evidence/assignment_source_hashes.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU evidence generation only")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "image.composite_image": False})
    report = completed_report()
    preflight = json.loads((PREFLIGHT/"report.json").read_text())
    if not all(preflight["checks"].values()):
        raise ValueError("accepted complete actor execution required")
    images = raw_images(report)
    assignment_figure(report, images)
    grounded_figure(preflight)
    copy_evidence()
    print("Generated completed V63 evidence and execution-only actor schematic")


if __name__ == "__main__":
    main()
