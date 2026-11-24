"""Compare paired page accuracy from original stopped answers on identical question IDs."""

from collections import defaultdict
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_competing_facts import PUB, ROOT, dump

ARMS = {
    "parent": "Original ILM · image",
    "contrast": "Contrast ILM · image",
    "qwen8-text": "Qwen3-8B · text",
    "qwen8vl-image": "Qwen3-VL-8B · image",
    "qwen8vl-text": "Qwen3-VL-8B · text",
}
COLORS = ["#718782", "#006B66", "#4A5978", "#A14B34", "#765065"]


def paged(rows, keys):
    group = defaultdict(list)
    selected = [r for r in rows if r["key"] in keys]
    if {r["key"] for r in selected} != keys or len(selected) != len(keys):
        raise ValueError("Incomplete matched question set")
    for row in selected:
        group[row["doc"]].append(row["correct"])
    if any(len(v) != 2 for v in group.values()):
        raise ValueError("Expected exactly two questions per page")
    return {doc: all(pair) for doc, pair in group.items()}


def wilson(k, n):
    z = 1.959963984540054
    den = 1 + z * z / n
    mid = (k / n + z * z / (2 * n)) / den
    radius = z * sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / den
    return 100 * (mid - radius), 100 * (mid + radius)


def main():
    rows = {
        a: [
            r
            for r in unpack(PUB / (a + "-outputs.json.gz"))
            if r.get("test", "human") == "human"
        ]
        for a in ARMS
    }
    full = {r["key"] for r in rows["contrast"]}
    subset = {r["key"] for r in rows["qwen8vl-image"]}
    assert len(full) == 512 and len(subset) == 128 and subset <= full
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.2), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.955, top=0.74, bottom=0.19, wspace=0.22)
    plt.rcParams["pdf.fonttype"] = 42
    report = {}
    for ax, (group, keys, arms, title) in zip(
        axes,
        (
            ("full", full, list(ARMS)[:3], "a   All 256 pages"),
            ("subset", subset, list(ARMS), "b   Same 64 pages"),
        ),
    ):
        report[group] = {}
        ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=12)
        for i, arm in enumerate(arms):
            outs = paged(rows[arm], keys)
            k, n = sum(outs.values()), len(outs)
            q = sum(r["correct"] for r in rows[arm] if r["key"] in keys)
            report[group][arm] = dict(both=k, pages=n, correct=q, questions=len(keys))
            y = 4 - i
            lo, hi = wilson(k, n)
            color = COLORS[list(ARMS).index(arm)]
            ax.plot([lo, hi], [y, y], color=color, lw=1.65, zorder=2)
            ax.scatter([100 * k / n], [y], color=color, s=24, zorder=3)
            ax.text(0, y + 0.21, ARMS[arm], fontsize=8, color="#182526")
            ax.text(60, y, f"{k}/{n}", ha="right", va="center", size=8, color=color)
        ax.set(xlim=(-1, 61), ylim=(-0.55, 4.9), xticks=[0, 15, 30, 45, 60], yticks=[])
        ax.set_xlabel("Pages with both answers exact (%)", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8, length=2, color="#708280")
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color("#BBC8C4")
    folder = ROOT / "publication/ilm-nature/figures"
    for ext in ("pdf", "png"):
        fig.savefig(folder / ("Figure_151_local_quantized_readers." + ext), dpi=240)
    plt.close(fig)
    dump(PUB / "benchmark.json", report)


if __name__ == "__main__":
    main()
