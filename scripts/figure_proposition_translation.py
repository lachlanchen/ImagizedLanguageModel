"""Produce editable, measured visual-translation figures and replay printed examples."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from ilm.visual_lm.portable_writer import save_grid, read_image
from scripts.figure_recurrent_evidence import Plate, FIG, MUTED, INK
from scripts.prepare_proposition_translation import (
    BASE,
    WORK,
    PUB,
    ROOT,
    Ink,
    read,
    dump,
)
from scripts.write_ordered_translation import restore, infer

PAPER = ROOT / "publication/ilm-nature"
SECOND = ROOT / "references/evidence/bilingual_ink_v1"
THIRD = ROOT / "references/evidence/translation_paraphrases_v1"
MODELDIR = ROOT / "models/translation_paraphrases/examples"
COLORS = ("#526F84", "#00766B", "#A96941")


def outputs():
    data = read(WORK / "graded.json")
    later = read(ROOT / "artifacts/translation_paraphrases_v1/generated.json")
    assets = PUB / "plates"
    assets.mkdir(exist_ok=True)
    ink = Ink(np.load(BASE / "palette.npy"))
    palette = np.load(BASE / "palette.npy")
    plate = Plate("Figure_190_optical_roles", assets)
    labels = ["Pictured request", "First event-trained writer", "Varied-source writer"]
    for x, text in zip((7, 249, 398), labels, strict=True):
        plate.label(x, 27, text, 8, True, MUTED)
    plate.label(6, 3, "a    Changing who invited whom", 11, True)
    plate.label(6, 136, "b    Moving the agent into a by-phrase", 11, True)
    plate.rule(125)
    models = {
        m: restore(m, "cpu") for m in ("proposition-likelihood", "paraphrase-varied")
    }
    record = []
    for i, y in zip((2, 3, 54, 55), (49, 85, 174, 213), strict=True):
        row = data["rows"][i]
        if row != later["rows"][i] or row["t"] != 1:
            raise ValueError("Example order changed")
        d = MODELDIR / ("item-" + str(i))
        d.mkdir(exist_ok=True, parents=True)
        src = ink.encode(row["source"])
        save_grid(ink.array()[src].tobytes(), d / "source.png")
        ref = ink.encode(row["ref"])
        save_grid(ink.array()[ref].tobytes(), d / "reference.png")
        plate.strip(f"{i}-source", d / "source.png", len(src), 7, y, 225, 20)
        meta = dict(
            source=row["source"],
            reference=row["ref"],
            source_cells=len(src),
            outputs={},
        )
        for x, w, tag, ans in [
            (249, 138, "proposition-likelihood", data["outputs"][i]["likelihood"]),
            (398, 140, "paraphrase-varied", later["outputs"][i]["varied"]),
        ]:
            with torch.inference_mode():
                raw, stop = infer(
                    models[tag], read_image(d / "source.png", len(src)), 1, "cpu"
                )
            if raw != palette[ans["cells"]].tobytes() or stop != ans["stop"]:
                raise ValueError(
                    "PNG replay differs from the original autonomous output"
                )
            save_grid(raw, d / (tag + ".png"))
            plate.strip(f"{i}-{tag}", d / (tag + ".png"), len(raw) // 128, x, y, w, 20)
            meta["outputs"][tag] = dict(
                text=ans["text"], stopped=stop, cells=len(raw) // 128
            )
        dump(d / "receipt.json", meta)
        record.append(meta)
    plate.label(
        7,
        249,
        "Original input and complete output pixels; intercell spacing reduced for print.",
        8,
        False,
        MUTED,
    )
    dump(PUB / "plates.json", [plate.render()])
    return record


def charts(name, panels):
    fig, axes = plt.subplots(1, 2, figsize=(7.55, 3.42), gridspec_kw={"wspace": 0.45})
    for ax, p in zip(axes, panels, strict=True):
        x = np.arange(len(p["labels"]))
        w = 0.72 / len(p["series"])
        for j, (label, values) in enumerate(p["series"].items()):
            bars = ax.bar(
                x + (j - (len(p["series"]) - 1) / 2) * w,
                values,
                width=w * 0.93,
                color=COLORS[j],
                label=label,
            )
            ax.bar_label(bars, fontsize=7, padding=2)
        ax.set_xticks(x, p["labels"], fontsize=8)
        ax.set_ylim(0, p["top"])
        ax.set_yticks(p["ticks"])
        ax.tick_params(axis="y", labelsize=7, length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#C7D1CF")
        ax.grid(axis="y", color="#E4EBE8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(
            p["title"], loc="left", weight="bold", color=INK, fontsize=10, pad=12
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.04, -0.18),
            frameon=False,
            fontsize=7,
            ncols=2,
            handlelength=1,
            columnspacing=0.8,
        )
    fig.subplots_adjust(top=0.85, bottom=0.30, left=0.065, right=0.99)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / (name + "." + ext), dpi=200)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    outputs()
    broad = read(SECOND / "result.json")
    sem = broad["semantic"]["groups"]
    recall = broad["retrieval"]
    varied = read(THIRD / "result.json")["semantic"]["groups"]
    voice = read(THIRD / "voice.json")["scores"]
    panels = [
        (
            "Figure_191_recognition_and_writing",
            [
                dict(
                    title="a   Translation retrieved (of 128)",
                    labels=["EN to ZH", "ZH to EN"],
                    top=138,
                    ticks=[0, 32, 64, 96, 128],
                    series={
                        label: [recall[k][t] for t in ("en_to_zh", "zh_to_en")]
                        for label, k in [
                            ("Likelihood", "likelihood/raw"),
                            ("Aligned", "aligned/raw"),
                            ("Projected", "aligned/projected"),
                        ]
                    },
                ),
                dict(
                    title="b   Complete meaning preserved",
                    labels=["Dialogue\n(/32)", "News\n(/32)", "OPUS\n(/64)"],
                    top=47,
                    ticks=[0, 10, 20, 30, 40],
                    series={
                        label: [sem[g][k] for g in ("conversation", "news", "opus")]
                        for label, k in [
                            ("Aligned", "aligned"),
                            ("Round trip", "roundtrip"),
                            ("Qwen Q4", "qwen"),
                        ]
                    },
                ),
            ],
        ),
        (
            "Figure_192_paraphrase_transfer",
            [
                dict(
                    title="a   Exact Chinese outputs (of 472)",
                    labels=["English active", "English passive"],
                    top=535,
                    ticks=[0, 118, 236, 354, 472],
                    series={
                        label: [voice[k][g] for g in ("active", "passive")]
                        for label, k in (
                            ("Canonical", "canonical"),
                            ("Varied", "varied"),
                        )
                    },
                ),
                dict(
                    title="b   Natural sentences",
                    labels=[
                        "Dialogue\n(/32)",
                        "News\n(/32)",
                        "OPUS\n(/64)",
                        "PAWS-X\n(/64)",
                    ],
                    top=74,
                    ticks=[0, 16, 32, 48, 64],
                    series={
                        label: [
                            varied[g][k]
                            for g in ("conversation", "news", "opus", "paws")
                        ]
                        for label, k in [
                            ("Canonical", "canonical"),
                            ("Varied", "varied"),
                            ("Qwen Q4", "qwen"),
                        ]
                    },
                ),
            ],
        ),
    ]
    for name, p in panels:
        charts(name, p)
    dump(PUB / "chart_data.json", dict(panels))


if __name__ == "__main__":
    main()
