"""Draw a one-page vector companion from checked Lean source and measured results."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOFS = ROOT / "formal/span_list"
PIN = "https://github.com/lachlanchen/ImagizedLanguageModel/blob/ff85222f/"
W, H = 720, 932
INK = "#162B35"
TEAL = "#0D6660"
AMBER = "#995329"
GRAY = "#4D6267"
LINE = "#C9D5D1"


def build(lean, out):
    for file in ("VisualClassTransfer.lean", "EvidencePointerMass.lean"):
        subprocess.run(
            [lean, str(PROOFS / file)], check=True, stdout=subprocess.DEVNULL
        )
    meter = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                str(ROOT / "scripts/check_matched_reading_cost.py"),
                "--lean",
                lean,
            ]
        )
    )
    mass = json.loads(
        (ROOT / "references/evidence/dialogue_teacher_v1/mass.json").read_text()
    )
    files = {
        "Sans": "DejaVuSans",
        "Bold": "DejaVuSans-Bold",
        "Serif": "DejaVuSerif",
        "Italic": "DejaVuSerif-Italic",
        "Mono": "DejaVuSansMono",
    }
    for name, file in files.items():
        pdfmetrics.registerFont(
            TTFont(name, "/usr/share/fonts/truetype/dejavu/" + file + ".ttf")
        )
    c = Canvas(str(out), pagesize=(W, H), invariant=1, pageCompression=1)
    c.setTitle("Reading ink, checking the proof")
    c.setAuthor("Imagized Language Model")

    def box(x, y, w, h, color):
        c.setFillColor(HexColor(color))
        c.rect(x, H - y - h, w, h, stroke=0, fill=1)

    def line(x, y, x2, y2, color=LINE, width=1):
        c.setStrokeColor(HexColor(color))
        c.setLineWidth(width)
        c.line(x, H - y, x2, H - y2)

    def txt(x, y, s, font="Sans", size=11, color=INK):
        c.setFillColor(HexColor(color))
        c.setFont(font, size)
        c.drawString(x, H - y, s)

    def link(x, y, label, file):
        txt(x, y, label, "Mono", 10, TEAL)
        tw = pdfmetrics.stringWidth(label, "Mono", 10)
        c.linkURL(
            PIN + "formal/span_list/" + file,
            (x, H - y - 3, x + tw, H - y + 10),
            relative=0,
            thickness=0,
        )

    def arrow(x, y, u, v):
        from math import atan2, cos, sin

        line(x, y, u, v, TEAL, 1.2)
        angle = atan2(v - y, u - x)
        for step in (-0.55, 0.55):
            line(u, v, u - 7 * cos(angle + step), v - 7 * sin(angle + step), TEAL, 1.2)

    def section(y, n, title):
        line(38, y, 682, y)
        txt(39, y + 29, n, "Mono", 13, AMBER)
        txt(77, y + 30, title, "Serif", 20)

    box(0, 0, W, H, "#FCFBF6")
    box(0, 0, 8, H, TEAL)
    txt(39, 35, "IMAGIZED LANGUAGE MODEL", "Bold", 10, TEAL)
    txt(490, 35, "FORMAL METHODS   /   LEAN 4", "Mono", 9, GRAY)
    txt(38, 91, "Reading ink, checking the proof", "Serif", 33)
    txt(
        40,
        119,
        "Three claims about a visual language learner, from a shared shape to a measured cost.",
        "Sans",
        12,
    )
    section(143, "01", "When can one example teach two prompts?")
    txt(41, 198, "Two pictured prompts", "Sans", 10, GRAY)
    for y, font, label in [
        (217, "Serif", "same question"),
        (251, "Italic", "same question"),
    ]:
        box(41, y - 3, 148, 29, "#ECEEE7")
        txt(54, y + 15, label, font, 13)
    arrow(197, 228, 253, 245)
    arrow(197, 261, 253, 245)
    box(261, 228, 142, 36, "#DDEEE8")
    txt(281, 251, "shared visual key", "Bold", 11, TEAL)
    arrow(411, 246, 473, 246)
    box(482, 228, 168, 36, "#EDE9DA")
    txt(502, 251, "one learned answer", "Bold", 11)
    txt(
        41,
        301,
        "The key must preserve the answer in context. If it does, one labelled example from",
        "Sans",
        12,
    )
    txt(
        41,
        320,
        "the class suffices. Learning such a key is an empirical problem.",
        "Sans",
        12,
    )
    link(41, 346, "VisualClassTransfer.transfer", "VisualClassTransfer.lean")
    section(369, "02", "When does a highlight help?")
    txt(42, 425, "POINTER PROBABILITY MASS", "Bold", 9, TEAL)
    for x, label in [(150, "Correct word"), (245, "Other words")]:
        txt(x, 447, label, "Sans", 9, GRAY)
    for y, label, a, b in [(456, "Inside quote", "a", "b"), (487, "Outside", "c", "d")]:
        txt(43, y + 19, label, "Sans", 11)
        for x, s in [(147, a), (244, b)]:
            box(x, y, 86, 27, "#DBECE6" if y == 456 else "#EBEEE9")
            txt(x + 37, y + 19, s, "Italic", 16, INK)
    proof = (PROOFS / "EvidencePointerMass.lean").read_text()
    predicate = next(
        s.split("↔", 1)[1].split(":=", 1)[0].strip()
        for s in proof.splitlines()
        if "↔ b*c < a*d :=" in s
    )
    txt(365, 431, "Exact condition from Lean", "Sans", 10, GRAY)
    box(364, 445, 275, 43, INK)
    txt(381, 474, predicate, "Mono", 18, "#E5F7E9")
    txt(365, 511, "Increasing the quote weight helps when", "Sans", 10)
    txt(365, 527, "correct words are enriched inside it.", "Sans", 10)
    a = mass["counts"]
    txt(
        42,
        556,
        f"On {a['reference_has_evidence_and_copy']:,} copyable positions, quadrupling the human-marked interval corrected",
        "Sans",
        12,
    )
    txt(
        42,
        576,
        f"{a['fixed']} choices and spoiled {a['lost']}. The model received the correct preceding words.",
        "Sans",
        12,
    )
    link(42, 601, "EvidencePointerMass.boostIff", "EvidencePointerMass.lean")
    section(625, "03", "Does the image reader learn more cheaply?")
    txt(
        42,
        682,
        "First recorded checkpoint above 20% exact accuracy on both held splits",
        "Sans",
        11,
    )
    for y, mode, label, color in [
        (704, "ink", "Glyph images", TEAL),
        (742, "byte", "ASCII bytes", "#738D99"),
    ]:
        kj = meter[mode]["20"] / 1e6
        txt(42, y + 17, label, "Sans", 12)
        box(157, y, 54 * kj, 23, color)
        txt(167 + 54 * kj, y + 17, f"{kj:.2f} kJ", "Bold", 12)
    txt(
        42,
        800,
        "Lean checks the minimum over all qualifying recorded checkpoints. Both readers",
        "Sans",
        12,
    )
    txt(
        42,
        820,
        "qualified at 2,400 updates. These GPU costs include preparation and training.",
        "Sans",
        12,
    )
    link(42, 847, "MatchedReadingCost.minimumSpec", "MatchedReadingCost.lean")
    link(355, 847, "MatchedReadingCost.factorIff", "MatchedReadingCost.lean")
    line(38, 872, 682, 872)
    txt(
        40,
        895,
        "Proof names link to the checked source. Measurements are from one RTX 4090 D.",
        "Sans",
        10,
        GRAY,
    )
    txt(
        40,
        914,
        "Companion to Figure 130 • Conditional language generation in glyph image space",
        "Italic",
        10,
        GRAY,
    )
    c.showPage()
    c.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lean", default="lean")
    p.add_argument("--output", type=Path, default=HERE / "lean_reading_guide.pdf")
    a = p.parse_args()
    build(a.lean, a.output)
