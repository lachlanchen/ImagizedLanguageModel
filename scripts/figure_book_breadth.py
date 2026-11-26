"""Build book-learning figures from retained checkpoints, raw PNGs and grading receipts."""

from shutil import copyfile
import numpy as np
import torch
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fitz
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas

from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_book_breadth import (
    ROOT,
    PUB,
    MODEL,
    WORK,
    DOMAINS,
    read,
    dump,
    sha,
)
from scripts.write_ordered_translation import restore, infer

PAPER = ROOT / "publication/ilm-nature"
FIG = PAPER / "figures"
ASSETS = FIG / "book_breadth"
COLORS = {
    "Starting writer": "#9AA8A2",
    "Narrow": "#54718B",
    "Broad": "#006C60",
    "Qwen Q4": "#A16D43",
}


def chart_data():
    fits = {m: read(MODEL / m / "fit.json") for m in ("narrow", "broad")}
    p = dict(
        title="a    Likelihood on unseen passages",
        labels=list(map(str, range(7))),
        unit="Nats per target glyph",
        series={},
    )
    for m, f in fits.items():
        losses = [f["initial_dev_nll"]] + [e["dev_nll"] for e in f["epochs"]]
        p["series"][m.title()] = [round(float(np.mean(list(d.values()))), 9) for d in losses]
    group = read(PUB / "result.json")["semantic"]["groups"]
    q = dict(
        title="b    Freely written complete meanings",
        labels=["Literature", "Talks", "FLORES"],
        unit="Accepted replies per 48",
        series={},
    )
    for m, k in (
        ("parent", "Starting writer"),
        ("narrow", "Narrow"),
        ("broad", "Broad"),
        ("qwen", "Qwen Q4"),
    ):
        q["series"][k] = [group[d][m] for d in DOMAINS]
    return [p, q]


def charts():
    panels = chart_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.05))
    for i, (panel, ax) in enumerate(zip(panels, axes, strict=True)):
        x = np.arange(len(panel["labels"]))
        for k, (name, s) in enumerate(panel["series"].items()):
            if i == 0:
                ax.plot(x, s, "o-", color=COLORS[name], markersize=3, label=name)
            else:
                bars = ax.bar(
                    x + (k - 1.5) * 0.21, s, 0.20, color=COLORS[name], label=name
                )
                ax.bar_label(bars, fontsize=7, padding=2)
        ax.set_title(panel["title"], loc="left", size=10, weight="bold")
        ax.set_xticks(x, panel["labels"], fontsize=8)
        ax.set_ylabel(panel["unit"], size=8)
        if i == 0:
            ax.set_xlabel("Completed training passes", fontsize=8)
        else:
            ax.set_ylim(0, 51)
            ax.set_yticks([0, 12, 24, 36, 48])
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.tick_params(length=0, labelsize=8)
        ax.grid(axis="y", color="#DEE5E3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            fontsize=7,
            frameon=False,
            ncol=2,
        )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.33, wspace=0.38)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_195_book_learning." + ext), dpi=220)
    plt.close(fig)
    dump(PUB / "charts.json", dict(panels=panels, colors=COLORS))


def rasters():
    torch.set_num_threads(2)
    gen = read(WORK / "generated.json")
    rows, answers = gen["rows"], gen["answers"]
    # Fixed first English input from each named public-domain work in the sealed ordering.
    chosen = [
        next(i for i, r in enumerate(rows) if r["doc"] == name and r["t"] == 1)
        for name in ("huckleberry-finn", "jane-eyre", "tale-of-two-cities")
    ]
    build_plate(
        rows, answers, chosen, "Figure_196_original_book_replies", "rasters", True
    )
    grades = read(WORK / "graded.json")["grades"]
    accepted = [
        i
        for i in range(len(grades["votes"]) // 2)
        if answers[i]["broad"]["stop"]
        and all(grades["votes"][2 * i + j]["broad"] for j in (0, 1))
    ]
    build_plate(
        rows, answers, accepted, "Figure_197_credited_book_replies", "credited", False
    )


def build_plate(rows, answers, chosen, figure, manifest, show_control):
    ASSETS.mkdir(exist_ok=True)
    width, cell = 530, 13
    used = []
    y = 4
    for i in chosen:
        r, o = rows[i], answers[i]
        used.append(
            dict(
                type="text",
                x=3,
                y=y,
                text=(
                    r["doc"].replace("-", " ").title()
                    if show_control
                    else (
                        "Jane Eyre"
                        if r["group"] == "book"
                        else "TED talk " + r["doc"].split("-")[-1].split(".")[0]
                    )
                    + "  /  "
                    + (
                        "Exact reference image"
                        if o["broad"]["raw_exact"]
                        else "Alternative wording"
                    )
                ),
                size=11,
                bold=True,
            )
        )
        y += 17
        for name, caption in (
            ("input", "English input" if r["t"] == 1 else "Chinese input"),
            ("narrow", "Narrow writer"),
            ("broad", "Broad writer"),
        ):
            if name == "narrow" and not show_control:
                continue
            src = WORK / "pixels" / f"{i:04}" / (name + ".png")
            stem = r["doc"] if show_control else f"accepted-{i}"
            path = ASSETS / (stem + "-" + name + ".png")
            copyfile(src, path)
            if name != "input":
                assert sha(src) == o[name]["png"]
                ink, done = infer(
                    restore("book-" + name, "cpu"),
                    read_image(
                        WORK / "pixels" / f"{i:04}" / "input.png", len(r["input"])
                    ),
                    r["t"],
                    "cpu",
                )
                assert (
                    done == o[name]["stop"]
                    and ink == read_image(path, len(o[name]["cells"])).tobytes()
                )
            with Image.open(path) as im:
                w, h = im.size
            used.append(
                dict(type="text", x=4, y=y + 1, text=caption, size=8, bold=False)
            )
            used.append(
                dict(
                    type="image",
                    x=93,
                    y=y,
                    w=w * cell / 32,
                    h=h * cell / 32,
                    name=path.name,
                    sha256=sha(src),
                    row=i,
                    model=name,
                )
            )
            y += max(15, h * cell / 32) + 3
        y += 9
    height = y
    pdf = FIG / (figure + ".pdf")
    canvas = Canvas(str(pdf), pagesize=(width, height), pageCompression=1)
    for it in used:
        if it["type"] == "image":
            canvas.drawImage(
                str(ASSETS / it["name"]),
                it["x"],
                height - it["y"] - it["h"],
                it["w"],
                it["h"],
            )
        else:
            canvas.setFont("Helvetica-Bold" if it["bold"] else "Helvetica", it["size"])
            canvas.setFillColor(HexColor("#213634"))
            canvas.drawString(it["x"], height - it["y"] - it["size"], it["text"])
    canvas.showPage()
    canvas.save()
    with fitz.open(pdf) as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            pdf.with_suffix(".png")
        )
    dump(
        PUB / (manifest + ".json"),
        dict(width=width, height=height, rows=chosen, items=used),
    )


if __name__ == "__main__":
    charts()
    rasters()
