"""Lay out measured glyph strips at print size, with vector type and editable scene data."""

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ilm.visual_lm.portable_writer import read_image
from scripts.prepare_recurrent_evidence import ROOT, MODEL, PUBLIC, read, dump
from scripts.prepare_paired_evidence import MODEL as PREV

FIG = ROOT / "publication/ilm-nature/figures"
ASSETS = PUBLIC / "plates"
INK = "#182526"
MUTED = "#536566"
TEAL = "#006B66"
RUST = "#A14B34"
PALE = "#F0F4F1"
WIDTH = 544
HEIGHT = 269
plt.rcParams.update(
    {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}
)


class Plate:
    def __init__(self, name):
        self.name = name
        self.items = []

    def label(self, x, y, value, size=9, bold=False, color=INK):
        self.items.append(
            dict(type="text", x=x, y=y, value=value, size=size, bold=bold, color=color)
        )

    def rule(self, y, x=5, w=534):
        self.items.append(dict(type="rule", x=x, y=y, w=w, h=0.55, color="#B8C4C0"))

    def band(self, x, y, w, h):
        self.items.append(dict(type="rect", x=x, y=y, w=w, h=h, color=PALE))

    def strip(self, tag, src, count, x, y, maxwidth, height, lo=0, hi=None):
        cells = read_image(src, count)[lo:hi]
        parts = []
        for cell in cells:
            mask = np.unpackbits(cell).reshape(32, 32)
            cols = np.flatnonzero(mask.any(0))
            glyph = mask[:, cols[0] : cols[-1] + 1] if len(cols) else mask[:, :8]
            parts.extend((glyph, np.zeros((32, 2), np.uint8)))
        im = Image.fromarray(255 * (1 - np.concatenate(parts, axis=1)))
        f = ASSETS / (self.name + "-" + tag + ".png")
        im.save(f)
        scale = min(maxwidth / im.width, height / 32)
        self.items.append(
            dict(type="image", x=x, y=y, w=im.width * scale, h=32 * scale, path=f.name)
        )

    def render(self):
        fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
        ax.set_axis_off()
        for z, r in enumerate(self.items, 1):
            x, y = r["x"], r["y"]
            t = r["type"]
            if t == "text":
                ax.text(
                    x,
                    y,
                    r["value"],
                    va="top",
                    ha="left",
                    fontsize=r["size"],
                    weight="bold" if r["bold"] else "normal",
                    color=r["color"],
                    zorder=z,
                )
            elif t in ("rule", "rect"):
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        r["w"],
                        r["h"],
                        facecolor=r["color"],
                        edgecolor="none",
                        zorder=z,
                    )
                )
            else:
                ax.imshow(
                    np.asarray(Image.open(ASSETS / r["path"])),
                    cmap="gray",
                    vmin=0,
                    vmax=255,
                    interpolation="none",
                    extent=(x, x + r["w"], y + r["h"], y),
                    aspect="auto",
                    zorder=z,
                )
        fig.savefig(FIG / (self.name + ".pdf"), dpi=300)
        fig.savefig(FIG / (self.name + ".png"), dpi=200)
        plt.close(fig)
        return dict(name=self.name, width=WIDTH, height=HEIGHT, items=self.items)


def reading():
    f = Plate("Figure_146_reading_visible_facts")
    old = PREV / "examples/cmrc"
    new = MODEL / "examples/dynamic/paired/cmrc"
    f.label(5, 3, "a", 12, True)
    f.label(23, 4, "A fish described on a Chinese page", 10, True)
    f.band(5, 22, 534, 29)
    f.strip("fish-source", old / "both/0/passage.png", 343, 11, 28, 520, 17, 0, 34)
    f.label(7, 58, "Question image", 8.4, True, MUTED)
    f.label(227, 58, "Interval writer", 8.4, True, MUTED)
    f.label(390, 58, "Recurrent writer", 8.4, True, TEAL)
    for i, y in enumerate((79, 108)):
        p = old / f"both/{i}"
        r = read(p / "readout.json")
        v = read(new / f"both/{i}/readout.json")
        f.strip("fish-q" + str(i), p / "query.png", r["query_cells"], 7, y, 209, 15)
        f.strip(
            "fish-old" + str(i), p / "answer.png", len(r["glyphs"]), 227, y, 154, 15
        )
        f.strip(
            "fish-new" + str(i),
            new / f"both/{i}/answer.png",
            v["written"],
            390,
            y,
            148,
            15,
        )
    f.label(
        7,
        133,
        "Both writers name the fish. Only the interval writer gives its classification.",
        8.7,
    )
    f.rule(151)
    f.label(5, 161, "b", 12, True)
    f.label(23, 162, "The location of a fossil discovery", 10, True)
    p = old / "miss/1"
    r = read(p / "readout.json")
    v = read(new / "miss/1/readout.json")
    f.band(5, 180, 534, 28)
    f.strip(
        "fossil-source", p / "passage.png", r["passage_cells"], 11, 185, 518, 17, 46, 76
    )
    f.strip("fossil-q", p / "query.png", r["query_cells"], 7, 220, 207, 15)
    f.strip("fossil-old", p / "answer.png", len(r["glyphs"]), 227, 220, 149, 15)
    f.strip("fossil-new", new / "miss/1/answer.png", v["written"], 390, 220, 146, 15)
    f.label(7, 246, "Where was the fossil found?", 8.4, False, MUTED)
    f.label(227, 246, "Fragment", 8.4, False, RUST)
    f.label(390, 246, "Niger", 8.4, False, TEAL)
    return f.render()


def limits():
    f = Plate("Figure_147_reading_and_composition")
    p = MODEL / "examples/dynamic/dolly_13380"
    r = read(p / "readout.json")
    held = read(ROOT / "artifacts/recurrent_evidence_v1/held.json")
    src = next(v["source"] for v in held if v["id"] == "dolly_13380")
    pos = src.index("first public or state school")
    f.label(5, 2, "a", 12, True)
    f.label(23, 3, "A date read from a pictured article", 10, True)
    f.band(5, 21, 535, 24)
    f.strip(
        "school-page",
        p / "passage.png",
        r["passage_cells"],
        10,
        25,
        523,
        15,
        pos,
        pos + 56,
    )
    f.strip("school-q", p / "question.png", r["question_cells"], 7, 53, 402, 16)
    f.strip("school-a", p / "answer.png", r["written"], 437, 53, 96, 18)
    f.rule(82)
    f.label(5, 92, "b", 12, True)
    f.label(23, 93, "A question that needs two facts", 10, True)
    p = MODEL / "examples/dynamic/museum-zh"
    r = read(p / "readout.json")
    f.band(5, 110, 535, 25)
    f.strip("museum-page", p / "passage.png", r["passage_cells"], 10, 115, 520, 15)
    f.strip("museum-q", p / "question.png", r["question_cells"], 7, 142, 420, 16)
    f.strip("museum-a", p / "answer.png", r["written"], 437, 142, 95, 18)
    f.label(
        7,
        166,
        "The source gives 17 and 9 bronzes. The writer stops after the first count.",
        8.6,
    )
    f.rule(185)
    f.label(5, 194, "c", 12, True)
    f.label(
        23, 195, "An attested oracle-bone form of the character for speech", 10, True
    )
    p = MODEL / "examples/dynamic/yan/zh/oracle"
    r = read(p / "readout.json")
    f.label(7, 216, "Evidence on the page", 8, True, MUTED)
    f.label(334, 216, "Reference", 8, True, MUTED)
    f.label(413, 216, "Actual output", 8, True, TEAL)
    f.strip("yan-page", p / "passage.png", r["passage_cells"], 7, 235, 310, 20)
    f.strip("yan-reference", p / "reference.png", 1, 334, 231, 53, 30)
    f.strip("yan-output", p / "answer.png", r["written"], 413, 234, 119, 24)
    return f.render()


def statements():
    from scripts.prepare_visual_statements import MODEL as ST, WORK as SW

    f = Plate("Figure_148_writing_visual_statements")
    group = ST / "examples/intervention"

    def strip(tag, case, image, x, y, w, h, lo=0, hi=None):
        p = group / case
        r = read(p / "readout.json")
        k = {
            "answer": "written",
            "passage": "passage_cells",
            "fact": "fact_cells",
            "question": "question_cells",
        }[image]
        f.strip(tag, p / (image + ".png"), r[k], x, y, w, h, lo, hi)

    f.label(5, 1, "a", 12, True)
    f.label(24, 2, "Change one name in the page", 11, True)
    f.label(7, 23, "Question", 8.5, True, MUTED)
    strip("q", "locomotive-original", "question", 105, 18, 388, 19)
    src = read(SW / "fresh.json")[12]["source"]
    pos = src.index("locomotive Salamanca")
    f.band(5, 42, 534, 48)
    for j, (case, name) in enumerate(
        (("original", "Matthew Murray"), ("edited", "Daniel Harper"))
    ):
        y = 47 + j * 21
        f.label(11, y + 2, ("Original page", "Edited page")[j], 8.5, True, MUTED)
        strip(
            "src-" + case,
            "locomotive-" + case,
            "passage",
            105,
            y - 1,
            427,
            18,
            pos,
            pos + len("locomotive Salamanca by " + name),
        )
    f.label(7, 101, "Fact the model read", 8.5, True, MUTED)
    f.label(198, 101, "Sentence the model wrote", 8.5, True, TEAL)
    for j, k in enumerate(("original", "edited")):
        strip("read-" + k, "locomotive-" + k, "fact", 7, 120 + 22 * j, 180, 18)
        strip("out-" + k, "locomotive-" + k, "answer", 198, 118 + 22 * j, 338, 21)
    f.rule(171)
    f.label(5, 181, "b", 12, True)
    f.label(24, 182, "A fact lost during writing", 11, True)
    f.label(7, 204, "Question", 8.5, True, MUTED)
    strip("school-q", "school-original", "question", 105, 199, 429, 18)
    f.label(7, 226, "Fact the model read", 8.5, True, MUTED)
    f.label(198, 226, "Sentence the model wrote", 8.5, True, TEAL)
    strip("school-fact", "school-original", "fact", 7, 243, 179, 19)
    strip("school-out", "school-original", "answer", 198, 242, 339, 20)
    return f.render()


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    dump(PUBLIC / "reading-plates.json", [statements(), reading(), limits()])


if __name__ == "__main__":
    main()
