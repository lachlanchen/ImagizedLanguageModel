#!/usr/bin/env python3
"""Measured matched joint-learning values and supplementary evidence plate."""

from __future__ import annotations

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_evidence import HERE, ROOT, save

RUN = ROOT / "artifacts/joint_exact_raster_v55_20261105"
HASHES = {
    "frozen": "0f993f484c7998629a1adf1dc94f60044c3f23f9ef95fd1459a4636c7f2b6c5a",
    "joint": "ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394",
}


def main():
    matched = json.loads((RUN / "matched_development_report.json").read_text())
    reports = {
        arm: json.loads((RUN / arm / "development/development_report.json").read_text())
        for arm in HASHES
    }
    training = {
        arm: json.loads((RUN / arm / "training_summary.json").read_text())
        for arm in HASHES
    }
    if matched["checkpoint_sha256"] != HASHES or any(
        reports[arm]["checkpoint_sha256"] != HASHES[arm] for arm in HASHES
    ):
        raise ValueError("unexpected matched checkpoint pair")
    gain = matched["joint_nll_advantage_over_matched_frozen"]
    values = {
        "JointNll": f"{reports['joint']['likelihood']['ordered']['nats_per_image']:.3f}",
        "ControlNll": f"{reports['frozen']['likelihood']['ordered']['nats_per_image']:.3f}",
        "JointNllGain": f"{gain['mean']:.3f}",
        "JointNllLow": f"{gain['ci95'][0]:.3f}",
        "JointNllHigh": f"{gain['ci95'][1]:.3f}",
        "JointTime": f"{training['joint']['elapsed_seconds']:.1f}",
        "ControlTime": f"{training['frozen']['elapsed_seconds']:.1f}",
        "JointMemory": f"{training['joint']['peak_allocated_vram_gib']:.3f}",
        "ControlMemory": f"{training['frozen']['peak_allocated_vram_gib']:.3f}",
        "JointSampleFone": f"{reports['joint']['generation']['ordered']['pixel_f1']:.3f}",
        "JointPrefixGain": f"{reports['joint']['likelihood']['ordered_nll_advantage_over_shuffled']['mean']:.3f}",
        "JointDecision": matched["decision"],
    }
    (HERE / "joint_values.tex").write_text(
        "% Generated from the locked V55 matched audit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in values.items())
    )
    sources = [
        RUN / "matched_development_report.json",
        RUN / "matched_training_verification.json",
        RUN / "smoke_verification.json",
    ]
    for arm in HASHES:
        sources.extend(
            [
                RUN / arm / "training_summary.json",
                RUN / arm / "development/development_report.json",
                RUN / arm / "development/generation_image_evidence.pt",
            ]
        )
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    (HERE / "evidence/v55_source_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n"
    )
    for path in sources:
        if path.suffix != ".json":
            continue
        label = (
            "v55_"
            + (
                path.parent.parent.name + "_"
                if path.name == "development_report.json"
                else path.parent.name + "_"
                if path.name == "training_summary.json"
                else ""
            )
            + path.name
        )
        (HERE / "evidence" / label).write_text(
            path.read_text().replace(str(ROOT) + "/", "")
        )
    rows = []
    for arm in HASHES:
        for route in ["ordered", "shuffled", "greedy"]:
            item = reports[arm]["generation"][route]
            rows.append(
                f"{arm.title()} / {route} & {100*item['identity_top1']:.3f} & {100*item['exact_image_match']:.3f} & {item['pixel_f1']:.4f} & {item['isolated_ink_fraction']:.4f} \\\\\n"
            )
    (HERE / "joint_table.tex").write_text(
        "\\section*{Supplementary Table 3: matched learning audit}\n\\begin{center}\\begin{tabular}{lrrrr}\\toprule\nArm / decoding & Identity (\\%) & Exact image (\\%) & Pixel F1 & Isolated ink \\\\\n\\midrule\n"
        + "".join(rows)
        + "\\bottomrule\\end{tabular}\\end{center}\nBoth arms use 256 paired prefixes and the same post-generation evaluator bank. No sampled output is selected or replaced.\n"
    )
    frozen = torch.load(
        RUN / "frozen/development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    joint = torch.load(
        RUN / "joint/development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    motor = torch.load(
        RUN / "joint/development/motor_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(frozen["targets"], joint["targets"]) and torch.equal(
        joint["targets"], motor["targets"]
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
    fig = plt.figure(figsize=(7.25, 4.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.6))
    ax = fig.add_subplot(grid[0, 0])
    ax.bar(
        ["Frozen continuation", "Joint learning"],
        [
            reports["frozen"]["likelihood"]["ordered"]["nats_per_image"],
            reports["joint"]["likelihood"]["ordered"]["nats_per_image"],
        ],
        color=["#939b9f", "#28638b"],
    )
    ax.set(ylabel="Exact target NLL (nats/image)", ylim=(0, 9))
    ax.set_title(
        "a  End-to-end likelihood improves", loc="left", weight="bold", fontsize=9
    )
    ax.text(
        0.5,
        0.93,
        f"Paired gain {values['JointNllGain']} [{values['JointNllLow']}, {values['JointNllHigh']}]",
        ha="center",
        transform=ax.transAxes,
        fontsize=7,
    )
    ax = fig.add_subplot(grid[0, 1])
    labels = ["Frozen / ordered", "Joint / ordered", "Joint / shuffled"]
    pairs = [("frozen", "ordered"), ("joint", "ordered"), ("joint", "shuffled")]
    ax.bar(
        labels,
        [100 * reports[a]["generation"][r]["identity_top1"] for a, r in pairs],
        color=["#939b9f", "#28638b", "#ae593f"],
    )
    ax.set(ylabel="First-sample target identity (%)", ylim=(0, 5))
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title(
        "b  Generation remains unqualified", loc="left", weight="bold", fontsize=9
    )
    panels = grid[1, :].subgridspec(1, 6, wspace=0.1)
    arrays = [
        joint["targets"],
        frozen["outputs"]["ordered"],
        joint["outputs"]["ordered"],
        joint["outputs"]["shuffled"],
        joint["outputs"]["greedy"],
        motor["outputs"]["copy_greedy"],
    ]
    labels = [
        "c  Target",
        "Frozen sample",
        "Joint sample",
        "Joint shuffled",
        "Joint greedy",
        "Supplied copy",
    ]
    for i, (label, array) in enumerate(zip(labels, arrays)):
        ax = fig.add_subplot(panels[0, i])
        ax.axis("off")
        ax.imshow(
            np.concatenate([v[0, 0].numpy() for v in array[:8]], axis=0),
            cmap="gray_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=7.2, weight="bold")
    save(fig, "Figure_S2_joint_learning")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
