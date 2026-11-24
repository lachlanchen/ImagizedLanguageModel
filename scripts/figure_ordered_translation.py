"""Render measured learning curves and replay the first failure and exact specimen translation."""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from ilm.visual_lm.portable_writer import save_grid
from ilm.visual_lm.specimen_plan import SpecimenPlanWriter
from scripts.figure_recurrent_evidence import FIG, Plate, TEAL, RUST, MUTED
from scripts.prepare_ordered_translation import MODEL, WORK, PUB, read, dump
from scripts.train_specimen_translation import materials, names, changed
from scripts.write_ordered_translation import restore


def comparison():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.55, 2.85), gridspec_kw={"width_ratios": [1, 1.05]}
    )
    colors = [TEAL, "#9B6D46"]
    for color, stride in zip(colors, (1, 4)):
        h = read(MODEL / str(stride) / "fit.json")["epochs"]
        axes[0].plot(
            [r["epoch"] for r in h],
            [r["dev"]["mean_nll"] for r in h],
            marker="o",
            ms=3,
            color=color,
            label=f"Stride {stride}",
        )
    axes[0].set(
        xlabel="Pass through training pairs",
        ylabel="Development loss (nats/glyph)",
        xticks=[1, 4, 8, 12],
        title="a   Prediction of the next pictured glyph",
    )
    axes[0].legend(frameon=False, fontsize=8)
    entries = read(PUB / "judgments.json")["groups"]
    for i, (method, lab, c) in enumerate(
        zip(
            ("stride4", "stride1", "qwen"),
            ("Stride 4", "Stride 1", "Qwen 8B Q4"),
            (colors[1], colors[0], "#577189"),
        )
    ):
        x = np.arange(2) + (i - 1) * 0.24
        values = [entries[d + "-sample"][method] for d in ("en", "zh")]
        bars = axes[1].bar(x, values, 0.22, color=c, label=lab)
        axes[1].bar_label(bars, padding=2, fontsize=8)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["Chinese to English", "English to Chinese"],
        ylabel="Meaning preserved (/32)",
        ylim=(0, 37),
        title="b   Autonomous translation",
        yticks=[0, 8, 16, 24, 32],
    )
    axes[1].legend(loc="upper left", fontsize=7, frameon=False)
    for ax in axes:
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(10)
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)
    fig.tight_layout(w_pad=1.7)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"Figure_187_translation_quality.{ext}", dpi=240)
    plt.close(fig)


def examples():
    torch.set_num_threads(2)
    palette = np.load(MODEL / "palette.npy")
    writer = SpecimenPlanWriter(
        restore("specimen", "cpu"),
        palette,
        read(MODEL / "inventories.json"),
        np.load(MODEL / "specimen/role.npy"),
    )
    alltiles, marks, labels = materials()
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    f = Plate("Figure_188_translating_specimens", assets)
    log = {}
    for i, r in enumerate(read(WORK / "test.json")):
        extent = names(r)
        if extent is None:
            continue
        mark = marks["sealed"][(2 * i) % len(marks["sealed"])]
        src, gold = changed(r, extent, mark)["cells"]
        inp = alltiles[src]
        answer, stop = writer.translate(inp, 1)
        expected = tuple(map(bytes, alltiles[gold]))
        group = "exact" if stop and answer == expected else "error"
        if group in log:
            continue
        y = 4 if group == "error" else 140
        dest = MODEL / "examples" / ("substitution-" + group)
        dest.mkdir(exist_ok=True)
        lab = (
            "a    A preserved image inside a mistranslation"
            if group == "error"
            else "b    A complete match to the human translation"
        )
        f.label(6, y, lab, 11, True)
        f.band(6, y + 20, 532, 31)
        rows = (
            ("source", tuple(map(bytes, inp)), "Pictured request"),
            ("reference", expected, "Human target"),
            ("answer", answer, "Model output"),
        )
        for j, (k, v, title) in enumerate(rows):
            baseline = y + 26 + 31 * j
            f.label(12, baseline + 4, title, 9, True, TEAL if k == "answer" else MUTED)
            save_grid(b"".join(v), dest / (k + ".png"))
            f.strip(
                group + "-" + k, dest / (k + ".png"), len(v), 126, baseline, 403, 19
            )
        log[group] = dict(
            record=i,
            credit=r["credit"],
            original=r["text"],
            specimen=labels[mark],
            stopped=stop,
            counts={k: len(v) for k, v, _ in rows},
        )
        dump(dest / "source.json", log[group])
        if len(log) == 2:
            break
    if len(log) != 2:
        raise ValueError("Missing output class")
    f.rule(129)
    dump(PUB / "plates.json", [f.render()])
    dump(PUB / "examples.json", log)
    print(json.dumps(log, ensure_ascii=False))


def news_transfer():
    groups = read(PUB / "news-judgments.json")["groups"]
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 2.7), sharey=True)
    colors = ["#829087", "#9B6D46", TEAL, "#577189"]
    labels = ["Base", "Replay", "News mix", "Qwen Q4"]
    for column, (ax, domain) in enumerate(zip(axes, ("conversation", "news"))):
        for k, (m, c) in enumerate(zip(("parent", "replay", "news", "qwen"), colors)):
            full = groups[domain][m]["stopped"]
            value = groups[domain][m]["meaning"]
            ax.bar(k, full, width=0.64, color="#e5eae7")
            ax.bar(k, value, width=0.64, color=c)
            ax.text(k, value + 0.7, str(value), ha="center", fontsize=9)
        ax.set(
            xticks=range(4), xticklabels=labels, ylim=(0, 37), yticks=[0, 8, 16, 24, 32]
        )
        ax.set_title(
            ("a   Conversation", "b   Editorial articles")[column],
            loc="left",
            fontsize=10,
            weight="bold",
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Translations (/32 per domain)", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_189_translation_transfer." + ext), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    comparison()
    examples()
    news_transfer()
