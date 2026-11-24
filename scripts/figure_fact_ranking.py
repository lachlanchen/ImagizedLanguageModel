"""Publish original bilingual answers and measured candidate-selection outcomes."""

from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt
from scripts.prepare_fact_ranking_confirmation import PUB, WORK
from scripts.prepare_fact_channel import MODEL as BASE, Ink, read, dump
from scripts.train_fact_ranking import MODEL
from scripts.evaluate_visual_discourse import unpack, draw
from scripts.figure_recurrent_evidence import Plate, TEAL, MUTED, FIG
from scripts.figure_phrase_region import breaks


def examples():
    rows = read(WORK / "confirmation.json")
    result = {
        m: unpack(WORK / (m + ".json.gz"))["original"] for m in ("frozen", "whole")
    }
    ink = Ink(np.load(BASE / "palette.npy"))

    def bits(s):
        return b"".join(ink.tiles[k] for k in ink.encode(s))

    selected = {}
    for i, r in enumerate(rows):
        f, w = (result[k][i] for k in ("frozen", "whole"))
        refs = {bits(y).hex() for y in r["answers"]}
        facts = refs if r["task"] == "exam" else {bits(r["facts"]["answer"]).hex()}
        if w["fact_hex"] in facts and f["fact_hex"] not in facts:
            correct = w["stop"] and w["ink_hex"] in refs
            if (r["task"] == "exam") == correct:
                selected.setdefault(r["task"], i)
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    records = []
    for label, task in (("a", "statements"), ("b", "exam")):
        i = selected[task]
        r = rows[i]
        f, w = (result[k][i] for k in ("frozen", "whole"))
        if label == "a":
            plate = Plate("Figure_171_reading_a_span_in_context", assets)
        folder = MODEL / "examples" / task
        folder.mkdir(parents=True, exist_ok=True)

        def raster(name, raw):
            arr = np.frombuffer(raw, np.uint8).reshape(-1, 128)
            draw(folder, name, arr)
            return dict(png=name + ".png", cells=len(arr))

        query = raster("question", bits(r["q"]))
        bs = [raster(f"source-{j}", bits(v)) for j, v in enumerate(r["blocks"])]
        dump(folder / "input.json", dict(question=query, regions=bs))
        ref = next(
            (s for s in r["answers"] if bits(s).hex() == w["ink_hex"]), r["answers"][0]
        )
        raster("reference", bits(ref))
        for k, v in (("frozen", f), ("whole", w)):
            raw = bytes.fromhex(v["ink_hex"])
            assert sha256(raw).hexdigest() == v["sha256"]
            raster(k, raw)
            raster(k + "-fact", bytes.fromhex(v["fact_hex"]))
        top = 0 if label == "a" else 135
        heading = (
            "A recovered name enters the wrong grammatical role"
            if label == "a"
            else "A recovered date answers a Chinese question"
        )
        plate.label(3, top + 2, label, 12, True)
        plate.label(22, top + 2, heading, 10, True)
        fields = [
            ("Question", "question", r["q"]),
            ("First choice", "frozen-fact", f["fact_text"]),
            ("Revised choice", "whole-fact", w["fact_text"]),
            ("Human reply", "reference", ref),
            ("First reply", "frozen", f["text"]),
            ("Revised reply", "whole", w["text"]),
        ]
        lines = [breaks(v) for _, _, v in fields]
        dy = min(15.8, 95 / sum(len(e) - 1 for e in lines))
        y = top + 18
        for (caption, key, text), edges in zip(fields, lines, strict=True):
            plate.label(
                6, y, caption, 7.5, True, TEAL if key.startswith("whole") else MUTED
            )
            for line, (a, b) in enumerate(zip(edges, edges[1:])):
                if b > a:
                    plate.strip(
                        f"{label}-{key}-{line}",
                        folder / (key + ".png"),
                        len(text),
                        86,
                        y + dy * line,
                        448,
                        dy,
                        a,
                        b,
                    )
            y += dy * (len(edges) - 1) + 2
        records.append(
            dict(
                task=task,
                index=i,
                question=r["q"],
                reference=ref,
                frozen=f["text"],
                frozen_fact=f["fact_text"],
                whole=w["text"],
                whole_fact=w["fact_text"],
                output=w["sha256"],
                fact_hash=w["fact_sha256"],
                stop=w["stop"],
            )
        )
    plate.rule(132)
    dump(PUB / "plates.json", [plate.render()])
    dump(PUB / "examples.json", records)


def counts():
    sc = read(PUB / "scores.json")
    rd = read(PUB / "reader.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.4), sharey=True)
    labels = ["Selected fact", "Exact reply", "Both questions", "Both edits"]
    x = np.arange(len(labels))
    for ax, t, title in zip(
        axes, ("statements", "exam"), ("English statements", "Chinese short answers")
    ):
        for j, (mode, label, color) in enumerate(
            (
                ("frozen", "Original", "#83766C"),
                ("endpoints", "Endpoints", "#5879A3"),
                ("whole", "Whole span", "#006B66"),
            )
        ):
            z = sc[mode][t]
            vals = [
                rd[mode][t]["selected"],
                z["bitmap_exact"],
                z["both_exact"],
                z["edit_pair_exact"],
            ]
            b = ax.bar(
                x + (j - 1) * 0.24, vals, 0.23, label=label, color=color, zorder=2
            )
            ax.bar_label(b, fontsize=7, padding=2)
        ax.set_xticks(x, labels, rotation=23, ha="right", fontsize=8)
        ax.set_title(title, loc="left", fontsize=11, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e7e5", linewidth=0.65, zorder=0)
    axes[0].set_ylabel("Number correct", fontsize=9)
    axes[0].set_ylim(0, 49)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.89, bottom=0.31, left=0.08, right=0.99, wspace=0.14)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_172_writing_after_span_selection." + ext), dpi=210)
    plt.close(fig)


if __name__ == "__main__":
    examples()
    counts()
