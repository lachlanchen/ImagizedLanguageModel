"""Plot complete pairing and writing outcomes from the same frozen models."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.export_raw_ilm import PAPER, PUB
from scripts.prepare_paired_ink import read, dump


def main():
    d = read(PUB / "scores.json")
    f = d["fits"]
    models = ("parent", "likelihood", "assignment")
    names = ("Starting writer", "Likelihood", "Assignment")
    grades = d["semantic"]["groups"]
    panels = [
        dict(
            title="a   Two supplied translations",
            labels=["Joint pairing", "Both row choices"],
            den="Correct decisions per 96",
            series={
                n: [f[m]["correct_pairings"], f[m]["both_rows"]]
                for m, n in zip(models, names)
            },
        ),
        dict(
            title="b   Freely written translations",
            labels=["Chinese to English", "English to Chinese"],
            den="Complete meanings per 96",
            series={
                n: [grades[g][m] for g in ("english", "chinese")]
                for m, n in (
                    ("likelihood", "Likelihood"),
                    ("assignment", "Assignment"),
                    ("qwen", "Qwen Q4"),
                )
            },
        ),
    ]
    colors = {
        "Starting writer": "#82978F",
        "Likelihood": "#46647D",
        "Assignment": "#006E61",
        "Qwen Q4": "#986338",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, p in zip(axes, panels):
        x = np.arange(len(p["labels"]))
        for j, (name, y) in enumerate(p["series"].items()):
            bars = ax.bar(
                x + (j - 1) * 0.23, y, width=0.21, color=colors[name], label=name
            )
            ax.bar_label(bars, padding=2, fontsize=8)
        ax.set_title(p["title"], loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 101)
        ax.set_yticks([0, 24, 48, 72, 96])
        ax.set_ylabel(p["den"], fontsize=8)
        ax.set_xticks(x, p["labels"], fontsize=8)
        ax.tick_params(axis="y", labelsize=7, length=0)
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#DDE6E2", linewidth=0.6)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            fontsize=7,
            frameon=False,
            ncol=2,
        )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.26, wspace=0.36)
    for ext in ("png", "pdf"):
        fig.savefig(PAPER / f"figures/Figure_194_paired_ink.{ext}", dpi=220)
    plt.close(fig)
    dump(PUB / "charts.json", dict(panels=panels, colors=colors))


if __name__ == "__main__":
    main()
