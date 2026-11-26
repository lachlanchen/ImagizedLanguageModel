"""Make plates from original PNGs and charts from complete assessment counts."""

from pathlib import Path
from shutil import copyfile
import fitz
import numpy as np
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.colors import HexColor
import torch
from scripts.figure_book_breadth import plt, FIG
from scripts.prepare_book_rehearsal import BASE, WORK, PUB, read, dump, sha
from scripts.write_ordered_translation import restore, infer
from ilm.visual_lm.portable_writer import read_image, save_grid
from ilm.visual_lm.glyph_typeface_reader import load_typeface_reader
from scripts.train_rehearsal_typeface import MODEL

ASSETS = FIG / "book_rehearsal"


def plate(stem, figure, groups):
    items = []
    y = 4
    units = 12.5
    for heading, files in groups:
        items.append(dict(type="text", text=heading, x=4, y=y, size=11, bold=True))
        y += 20
        for label, name, src in files:
            dst = ASSETS / (name + ".png")
            copyfile(src, dst)
            w, h = Image.open(dst).size
            items.append(
                dict(type="text", x=4, y=y + 2, text=label, size=8, bold=False)
            )
            items.append(
                dict(
                    type="image",
                    x=105,
                    y=y,
                    w=w * units / 32,
                    h=h * units / 32,
                    name=dst.name,
                    sha256=sha(src),
                )
            )
            y += max(18, h * units / 32 + 4)
        y += 12
    width = 515
    file = FIG / (figure + ".pdf")
    canvas = Canvas(str(file), pagesize=(width, y), pageCompression=1)
    for item in items:
        if item["type"] == "text":
            canvas.setFillColor(HexColor("#213634"))
            canvas.setFont(
                "Helvetica-Bold" if item["bold"] else "Helvetica", item["size"]
            )
            canvas.drawString(item["x"], y - item["y"] - item["size"], item["text"])
        else:
            canvas.drawImage(
                str(ASSETS / item["name"]),
                item["x"],
                y - item["y"] - item["h"],
                item["w"],
                item["h"],
            )
    canvas.showPage()
    canvas.save()
    with fitz.open(file) as pdf:
        pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(
            file.with_suffix(".png")
        )
    dump(PUB / (stem + ".json"), dict(width=width, height=y, items=items))


def panels():
    result = read(PUB / "result.json")
    sem = result["semantic"]["groups"]
    prior = read(PUB / "typography.json")["conditions"]
    new = read(PUB / "typeface-reader.json")["conditions"]
    x = dict(
        title="a    Complete role families",
        labels=["Original", "4 flips", "Serif"],
        unit="All eight replies exact, out of 118",
        series={},
    )
    for name, variants in (
        ("Unaided", [prior["original"], prior["four_flips"], prior["serif"]]),
        (
            "Hamming rule",
            [prior["original"], prior["four_flips_guarded"], prior["serif_guarded"]],
        ),
        (
            "Learned reader",
            [new["original_learned"], new["four_flips_learned"], new["serif_learned"]],
        ),
    ):
        x["series"][name] = [v["whole_role_families"] for v in variants]
    y = dict(
        title="b    Natural translation",
        labels=["Earlier panel", "Fresh panel"],
        unit="Credited complete meanings (%)",
        series={},
    )
    for m, name in (
        ("broad", "Broad writer"),
        ("rehearsal", "Rehearsal"),
        ("qwen", "Qwen Q4"),
    ):
        y["series"][name] = [
            round(100 * sum(sem[d][m] for d in ds) / sum(sem[d]["n"] for d in ds), 4)
            for ds in (("book", "talk", "flores"), ("new-book", "new-flores"))
        ]
    colors = dict(
        zip(
            [*x["series"], *y["series"]],
            ["#9AA8A2", "#54718B", "#006C60", "#54718B", "#006C60", "#9A6133"],
            strict=True,
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.25))
    for p, ax, max_y in zip((x, y), axes, (125, 100), strict=True):
        pos = np.arange(len(p["labels"]))
        for j, (k, vals) in enumerate(p["series"].items()):
            bars = ax.bar(pos + (j - 1) * 0.25, vals, 0.23, color=colors[k], label=k)
            ax.bar_label(
                bars,
                labels=[f"{v:.0f}" if p is x else f"{v:.1f}" for v in vals],
                fontsize=7,
                padding=2,
            )
        ax.set_title(p["title"], loc="left", fontsize=11, weight="bold")
        ax.set_ylabel(p["unit"], fontsize=8)
        ax.set_xticks(pos, p["labels"], fontsize=8)
        ax.set_ylim(0, max_y)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", color="#DEE5E3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=8)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            frameon=False,
            fontsize=7,
        )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.85, bottom=0.27, wspace=0.40)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / ("Figure_198_rehearsal_and_transfer." + ext), dpi=220)
    plt.close(fig)
    dump(PUB / "charts.json", dict(panels=[x, y], colors=colors))


def main():
    torch.set_num_threads(2)
    ASSETS.mkdir(exist_ok=True)
    gen = read(WORK / "generated.json")
    net = restore("book-rehearsal", "cpu")
    control = restore("book-broad", "cpu")
    groups = []
    for i, title in (
        (241, "a    Mary helps Tom"),
        (245, "b    Tom helps Mary"),
        (144, "c    A house and a reception in Gulliver's Travels"),
    ):
        folder = WORK / "pixels" / f"{i:04}"
        r = gen["rows"][i]
        raw = read_image(folder / "input.png", len(r["input"]))
        files = [("Pictured input", f"case-{i}-input", folder / "input.png")]
        for m, n, label in (
            ("broad", control, "Broad writer"),
            ("rehearsal", net, "With rehearsal"),
        ):
            o = gen["answers"][i][m]
            packed, stop = infer(n, raw, r["t"], "cpu")
            assert stop == o["stop"] and sha(folder / (m + ".png")) == o["png"]
            assert (
                packed == read_image(folder / (m + ".png"), len(o["cells"])).tobytes()
            )
            files.append((label, f"case-{i}-{m}", folder / (m + ".png")))
        groups.append((title, files))
    plate("rasters", "Figure_199_who_helped_whom", groups)
    palette = np.load(BASE / "palette.npy")[2:]
    adapter = load_typeface_reader(MODEL, palette, "cpu")
    folder = WORK / "typography"
    groups = []
    for prefix, label in (
        ("serif", "a    Noto Serif CJK Regular"),
        ("condensed", "b    DejaVu Serif Condensed"),
    ):
        path = folder / (prefix + "-input.png")
        cells = read_image(path, len(gen["rows"][241]["input"]))
        files = [("Pictured input", prefix + "-input", path)]
        for tag, obs, caption in (
            (prefix, cells, "Original reader"),
            (prefix + "_learned", adapter.transport(cells, palette), "Learned reader"),
        ):
            packed, stop = infer(net, obs, 1, "cpu")
            replay = WORK / (tag + "-replay.png")
            save_grid(packed, replay)
            assert sha(replay) == sha(folder / (tag + "-output.png"))
            files.append((caption, tag + "-output", folder / (tag + "-output.png")))
        groups.append((label, files))
    plate("typefaces", "Figure_200_optical_transfer", groups)
    panels()


if __name__ == "__main__":
    main()
