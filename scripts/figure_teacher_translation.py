"""Reproduce raw translation images and plot primary and source-only assessments."""

import numpy as np
import torch
from scripts.figure_book_breadth import FIG, plt
from scripts.figure_book_rehearsal import plate
from scripts.prepare_teacher_translation import WORK, PUB, read, dump, sha
from scripts.write_ordered_translation import restore, infer
from ilm.visual_lm.portable_writer import read_image, save_grid

ASSETS = FIG / "teacher_translation"


def main():
    torch.set_num_threads(2)
    ASSETS.mkdir(exist_ok=True)
    gen = read(WORK / "generated.json")
    networks = {
        k: restore(v, "cpu")
        for k, v in (
            ("parent", "prefix-recovery"),
            ("corpus", "teacher-corpus"),
            ("distilled", "teacher-distilled"),
        )
    }

    def make(stem, figure, cases):
        groups = []
        for i, title, methods in cases:
            r = gen["rows"][i]
            p = WORK / "pixels" / f"{i:04d}"
            raw = read_image(p / "input.png", len(r["input"]))
            files = [("Pictured source", f"request-{i}-input", p / "input.png")]
            for m in methods:
                with torch.inference_mode():
                    out, stop = infer(networks[m], raw, r["t"], "cpu")
                tmp = WORK / f"replay-{i}-{m}.png"
                save_grid(out, tmp)
                assert (
                    sha(tmp) == sha(p / (m + ".png")) == gen["answers"][i][m]["png"]
                    and gen["answers"][i][m]["stop"] == stop
                )
                files.append(
                    (
                        {
                            "parent": "Starting writer",
                            "corpus": "Corpus targets",
                            "distilled": "Teacher targets",
                        }[m],
                        f"request-{i}-{m}",
                        p / (m + ".png"),
                    )
                )
            groups.append((title, files))
        plate(stem, figure, groups, assets=ASSETS, evidence=PUB)

    make(
        "natural-rasters",
        "Figure_204_translating_from_the_ink",
        [
            (
                38,
                "a    Fitted with two sources of supervision",
                ("parent", "corpus", "distilled"),
            ),
            (55, "b    Removing a slide", ("corpus", "distilled")),
            (85, "c    An incorrect reference", ("corpus", "distilled")),
        ],
    )
    make(
        "limits-rasters",
        "Figure_206_learning_a_wording",
        [
            (0, "a    First reserved literary request", ("distilled",)),
            (1, "b    The reverse translation", ("distilled",)),
            (
                203,
                "c    A chapter heading present in training",
                ("corpus", "distilled"),
            ),
        ],
    )
    results = read(PUB / "results.json")
    other = read(PUB / "fresh-source-review.json")
    forks = read(PUB / "forks.json")["models"]
    arms = {
        "Starting writer": "parent",
        "Corpus targets": "corpus",
        "Teacher targets": "distilled",
        "Qwen Q4": "qwen",
    }
    a = dict(
        title="a    Complete meanings",
        labels=["With reference", "Source only"],
        unit="Credits out of 96 new requests",
        series={},
    )
    for name, m in arms.items():
        a["series"][name] = [
            sum(condition[d][m] for d in ("book", "talk"))
            for condition in [results["semantic"]["groups"], other["counts"]]
        ]
    b = dict(
        title="b    Exact fitted replies",
        labels=["Starting writer", "Selected pass", "Pass 12"],
        unit="Exact outputs out of 2,614",
        series={},
    )
    for name, m, k in (
        ("Corpus targets", "corpus", "corpus_exact"),
        ("Teacher targets", "distilled", "teacher_exact"),
    ):
        b["series"][name] = [forks[model][k] for model in ("parent", m, m + "-final")]
    colors = dict(zip(arms, ("#9CAAA7", "#54768B", "#006B60", "#9A622F"), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for panel, ax, limit in zip((a, b), axes, (100, 1250), strict=True):
        n = len(panel["series"])
        x = np.arange(len(panel["labels"]))
        w = 0.84 / n
        for j, (name, values) in enumerate(panel["series"].items()):
            bars = ax.bar(
                x + (j - (n - 1) / 2) * w,
                values,
                w * 0.89,
                label=name,
                color=colors[name],
            )
            ax.bar_label(bars, fontsize=7, padding=2)
        ax.set_title(panel["title"], loc="left", fontsize=11, weight="bold")
        ax.set_ylim(0, limit)
        ax.set_xticks(x, panel["labels"], fontsize=7)
        ax.set_ylabel(panel["unit"], fontsize=8)
        ax.tick_params(length=0, labelsize=7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#DEE5E3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=2,
            fontsize=7,
            frameon=False,
        )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.87, bottom=0.27, wspace=0.38)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / ("Figure_205_teaching_complete_translations." + ext), dpi=240)
    plt.close(fig)
    dump(PUB / "charts.json", dict(panels=[a, b], colors=colors))


if __name__ == "__main__":
    main()
