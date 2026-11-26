"""Publish the original replayed answer PNGs and plot the complete matched comparison."""

import numpy as np
import torch
from scripts.figure_book_breadth import FIG, plt
from scripts.figure_book_rehearsal import plate
from scripts.prepare_visual_rollins import WORK, PUB, read, dump, sha
from scripts.write_ordered_translation import restore, infer
from ilm.visual_lm.portable_writer import read_image, save_grid

ASSETS = FIG / "prefix_recovery"


def charts():
    result = read(PUB / "results.json")
    groups = result["semantic"]["groups"]
    panels = [
        dict(
            title="a    Complete translations",
            labels=["Books", "Talks", "FLORES"],
            unit="Credited requests out of 48",
            series={
                name: [groups[g][m] for g in ("book", "talk", "flores")]
                for name, m in [
                    ("Reference-prefix fit", "clean"),
                    ("Draft-prefix fit", "recover"),
                    ("Qwen3-8B Q4", "qwen"),
                ]
            },
        ),
        dict(
            title="b    Errors after the same draft",
            labels=["Starting writer", "Reference-prefix fit", "Draft-prefix fit"],
            unit="Mean edits per completed request",
            series={
                name: [
                    round(result["prefixes"][m]["draft"][k] / 143, 2)
                    for m in ("parent", "clean", "recover")
                ]
                for name, k in (
                    ("Committed minimum", "floor"),
                    ("Above minimum", "excess"),
                )
            },
        ),
    ]
    colors = dict(
        zip(
            [*panels[0]["series"], *panels[1]["series"]],
            ["#52778C", "#0A6D61", "#9A632D", "#909F9D", "#31556A"],
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.15))
    for j, (p, ax) in enumerate(zip(panels, axes, strict=True)):
        x = np.arange(3)
        for i, (name, values) in enumerate(p["series"].items()):
            n = len(p["series"])
            bars = ax.bar(
                x + (i - (n - 1) / 2) * 0.23,
                values,
                0.21,
                color=colors[name],
                label=name,
            )
            ax.bar_label(bars, fmt="%.1f" if j else "%.0f", fontsize=7, padding=2)
        ax.set_title(p["title"], loc="left", weight="bold", fontsize=10)
        ax.set_xticks(
            x,
            p["labels"] if not j else ["Starting", "Reference-prefix", "Draft-prefix"],
            fontsize=7,
        )
        ax.set_ylabel(p["unit"], fontsize=8)
        ax.set_ylim(0, [50, 35][j])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#DFE5E3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2 if not j else 1,
            fontsize=6.5,
            frameon=False,
        )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.29, wspace=0.32)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_201_learning_from_drafts." + ext), dpi=240)
    plt.close(fig)
    dump(PUB / "charts.json", dict(panels=panels, colors=colors))


def main():
    torch.set_num_threads(2)
    ASSETS.mkdir(exist_ok=True)
    gen = read(WORK / "generated.json")
    writer = restore("prefix-recovery", "cpu")
    panels = []
    for i, title in [
        (0, "a    Emma, Chinese to English"),
        (1, "b    Emma, English to Chinese"),
        (145, "c    A held passive construction"),
        (2111, "d    A sentence seen during fitting"),
    ]:
        folder = WORK / "outputs" / f"{i:04d}"
        r = gen["rows"][i]
        o = gen["answers"][i]["recover"]
        actual = read_image(folder / "input.png", len(r["input"]))
        with torch.inference_mode():
            pixels, stop = infer(writer, actual, r["t"], "cpu")
        replay = WORK / ("replay-" + str(i) + ".png")
        save_grid(pixels, replay)
        assert stop == o["stop"] and sha(replay) == o["png"] == sha(
            folder / "recover.png"
        )
        panels.append(
            (
                title,
                [
                    ("Input", f"case-{i}-input", folder / "input.png"),
                    ("Writer output", f"case-{i}-output", folder / "recover.png"),
                ],
            )
        )
    plate(
        "rasters",
        "Figure_202_after_the_first_glyphs",
        panels,
        assets=ASSETS,
        evidence=PUB,
    )
    charts()


if __name__ == "__main__":
    main()
