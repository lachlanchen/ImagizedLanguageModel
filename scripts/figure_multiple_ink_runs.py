"""Make the article plate from the held PNG replay, and plot the complete held tests."""

import matplotlib.pyplot as plt
from scripts.figure_recurrent_evidence import Plate, FIG, TEAL, MUTED
from scripts.prepare_multiple_ink_runs import PUB, MODEL, read, dump


def make_plate():
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    plate = Plate("Figure_153_visual_lists", assets)

    def strip(case, field, x, y, w, h, lo=0, hi=None):
        dest = MODEL / "examples" / case
        count = read(dest / "receipt.json")[field + "_cells"]
        plate.strip(
            f"{case}-{field}-{lo}", dest / (field + ".png"), count, x, y, w, h, lo, hi
        )

    plate.label(5, 2, "a", 13, True)
    plate.label(24, 4, "Which two people does the question ask for?", 11, True)
    plate.label(9, 25, "Pictured question", 8, True, MUTED)
    plate.label(320, 25, "Answer from the pictured page", 8, True, TEAL)
    strip("dunk", "question", 9, 39, 295, 20)
    strip("dunk", "answer", 320, 39, 210, 20)
    plate.rule(67)
    plate.label(5, 76, "b", 13, True)
    plate.label(24, 78, "Replacing one printed name with unfamiliar marks", 11, True)
    plate.label(9, 101, "Original answer", 8, True, MUTED)
    plate.label(278, 101, "Answer after editing the page", 8, True, TEAL)
    strip("song", "answer", 9, 115, 254, 21)
    strip("changed-song", "answer", 278, 115, 250, 21)
    plate.band(5, 143, 534, 43)
    plate.label(
        11,
        148,
        "The English sentence writer, given the original selection",
        8,
        True,
        MUTED,
    )
    strip("song", "sentence", 11, 163, 519, 20)
    plate.rule(193)
    plate.label(5, 203, "c", 13, True)
    plate.label(24, 205, "A partial answer about the Western Xia", 11, True)
    plate.label(9, 226, "Pictured question", 8, True, MUTED)
    plate.label(199, 226, "Emitted ink", 8, True, TEAL)
    plate.label(314, 226, "Relevant ink on the supplied page", 8, True, MUTED)
    strip("xixia", "question", 9, 240, 177, 21)
    strip("xixia", "answer", 199, 240, 74, 21)
    strip("xixia", "page", 314, 240, 215, 21, 7, 19)
    dump(PUB / "plates.json", [plate.render()])


def outcomes():
    fig, axes = plt.subplots(
        1, 2, figsize=(7.55, 2.6), gridspec_kw={"width_ratios": [1.24, 1]}
    )
    s = [
        read(PUB / f)["scores"]["count"]["original"]
        for f in (
            "scores.json",
            "survival-scores.json",
            "confirm-scores.json",
            "survival-confirm-scores.json",
        )
    ]
    ax = axes[0]
    for i, (group, arm) in enumerate(
        (("en-multi", 0), ("en-multi", 2), ("zh-multi", 0), ("zh-multi", 2))
    ):
        g = [s[j][group] for j in (arm, arm + 1)]
        y = 3 - i
        for off, obs, color in zip((-0.13, 0.13), g, (MUTED, TEAL), strict=True):
            n, k = obs["pages"], obs["exact"]
            ax.scatter(100 * k / n, y + off, s=24, color=color, zorder=3)
            ax.text(
                100 * k / n + 0.27,
                y + off,
                f"{k}/{n}",
                va="center",
                fontsize=7,
                color=color,
            )
    ax.set_yticks(
        [3, 2, 1, 0],
        [
            "English · initial",
            "English · fresh",
            "Chinese · initial",
            "Chinese · fresh",
        ],
    )
    ax.set_xlim(-0.5, 8.8)
    ax.set_ylim(-0.55, 3.55)
    ax.set_xlabel("Complete answer sets (%)")
    ax.set_title("a   Selecting every fact", loc="left", fontweight="bold")
    ax.text(
        0.015,
        -0.41,
        "Grey  cell classifier     Teal  conditional continuation",
        transform=ax.transAxes,
        fontsize=7,
        color=MUTED,
    )
    ax = axes[1]
    d = read(PUB / "composition.json")
    for y, k, color in ((1, "read", TEAL), (0, "given", MUTED)):
        v = d[k]
        ax.barh(y, 100 * v["contains_all"] / v["n"], height=0.26, color=color)
        ax.text(
            100 * v["contains_all"] / v["n"] + 1,
            y,
            f'{v["contains_all"]}/{v["n"]}',
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, 53)
    ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([1, 0], ["Reader's facts", "Human facts"])
    ax.set_xlabel("Stopped with every fact present (%)")
    ax.set_title("b   Stating the facts", loc="left", fontweight="bold")
    for ax in axes:
        ax.tick_params(length=0, labelsize=8, pad=4)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#AFBBB7")
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.18)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(8)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.32, top=0.85, wspace=0.54)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_154_multiple_facts_accuracy." + ext), dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    make_plate()
    outcomes()
