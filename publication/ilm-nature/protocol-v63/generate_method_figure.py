#!/usr/bin/env python3
"""Draw the registered training computation using exact observed image regions."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt
import torch


HERE = Path(__file__).resolve().parent
PUB = HERE.parent
ROOT = PUB.parents[1]
RUN = ROOT/"artifacts/question_assignment_v63_20261105"
INK, BLUE, GREEN, GREY = "1D2B34", "28638B", "32714C", "677B87"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def px(value):
    return round(value*9525)


def main():
    preflight = json.loads((RUN/"preflight/report.json").read_text())
    if not all(preflight["checks"].values()):
        raise ValueError("verified implementation preflight required")
    for name, expected in preflight["provenance"].items():
        if digest(ROOT/name) != expected:
            raise ValueError("registered training source differs")
    pair_report = json.loads((RUN/"data/report.json").read_text())
    if digest(RUN/"data/report.json") != preflight["data_sha256"]:
        raise ValueError("training pair report differs from completed preflight")
    if digest(RUN/"data/pairs.jsonl") != pair_report["manifest_sha256"]:
        raise ValueError("training pair manifest differs")
    pair = json.loads((RUN/"data/pairs.jsonl").read_text().splitlines()[0])
    source = ROOT/"data/visual_qa/cmrc2018_short/train.jsonl"
    if digest(source) != pair_report["source_sha256"]:
        raise ValueError("training source differs")
    records = {row["id"]: row for row in map(json.loads, source.read_text().splitlines())}
    cache = ROOT/pair_report["image_cache"]
    if digest(cache/"report.json") != pair_report["image_report_sha256"]:
        raise ValueError("physical image cache identity differs")
    cache_report = json.loads((cache/"report.json").read_text())
    (HERE/"images").mkdir(exist_ok=True)
    evidence, pictures, input_hashes = [], {}, {}
    for i, identifier in enumerate(pair["record_ids"]):
        row = records[identifier]
        if row["context_hash"] != pair["context_hash"]:
            raise ValueError("shared paragraph identity differs")
        if row["prompt"] != "资料："+row["context"]+"问题："+row["question"]+"回答：":
            raise ValueError("source prompt layout differs")
        relative = "train/"+identifier+".pt"
        path = cache/relative
        if digest(path) != cache_report["files_sha256"][relative]:
            raise ValueError("physical training image changed")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        white = np.unpackbits(raw["bits"].numpy(), count=32*raw["width"]).reshape(32, raw["width"])*255
        start = 32*(len("资料：")+len(row["context"])+len("问题："))
        stop = start+32*len(row["question"])
        if not np.all(white[:, -128:] == 255):
            raise ValueError("physical termination differs")
        arrays = dict(question=white[:, start:stop], answer=white[:, raw["prompt_blocks"]*64:-128])
        for part, pixels in arrays.items():
            path = HERE/"images"/f"{part}_{i}.png"
            Image.fromarray(pixels.astype(np.uint8), mode="L").save(path)
            pictures[f"{part}_{i}"] = path
            evidence.append(dict(name=f"{part}_{i}", source_id=identifier,
                source_role="observed training question region" if part == "question" else "observed padded answer; appended termination omitted from display",
                width=pixels.shape[1], height=32, png_sha256=digest(path),
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
        input_hashes[str((cache/relative).relative_to(ROOT))] = digest(cache/relative)
    template = PUB/"baselines/v62-primary/ILM_Figures_editable.pptx"
    prs = Presentation(template)
    if (prs.slide_width, prs.slide_height) != (15240000, 8572500):
        raise ValueError("preserved publication slide geometry differs")
    for item in list(prs.slides._sldIdLst):
        prs.part.drop_rel(item.rId)
        prs.slides._sldIdLst.remove(item)
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    for shape in list(slide.shapes):
        if shape.is_placeholder:
            shape._element.getparent().remove(shape._element)
    plt.rcParams.update({"font.family": "Liberation Sans", "pdf.fonttype": 42,
        "svg.fonttype": "none", "image.composite_image": False})
    fig = plt.figure(figsize=(7.25, 7.25*9/16))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1600), ylim=(900, 0)); ax.axis("off")
    objects = []

    def text(words, x, y, w, h, size=21, bold=False, color=INK, center=False):
        frame = slide.shapes.add_textbox(px(x), px(y), px(w), px(h)).text_frame
        frame.word_wrap = True
        frame.auto_size = MSO_AUTO_SIZE.NONE
        frame.vertical_anchor = MSO_ANCHOR.TOP
        frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
        for index, line in enumerate(words.split("\n")):
            p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
            p.text = line; p.font.name = "Arial"; p.font.size = Pt(size); p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
            p.alignment = PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
            p.space_after = Pt(0)
        ax.text(x+w/2 if center else x, y, words, va="top", ha="center" if center else "left",
            fontsize=size*.435, weight="bold" if bold else "normal", color="#"+color, linespacing=1.25)
        objects.append(dict(kind="text", text=words, x=x, y=y, width=w, height=h))

    def box(x, y, w, h, fill="F2F5F6", border=GREY):
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, px(x), px(y), px(w), px(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor.from_string(fill)
        shape.line.color.rgb = RGBColor.from_string(border); shape.line.width = Pt(.8)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#"+fill, edgecolor="#"+border, linewidth=.6))
        objects.append(dict(kind="box", x=x, y=y, width=w, height=h))

    def arrow(x1, y, x2):
        shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x1), px(y), px(x2), px(y))
        shape.line.color.rgb = RGBColor.from_string(GREY); shape.line.width = Pt(1.5)
        end = OxmlElement("a:tailEnd"); end.set("type", "triangle")
        shape.line._get_or_add_ln().append(end)
        ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="->", mutation_scale=7,
            color="#"+GREY, linewidth=.7))

    def picture(name, x, y, maximum_width, scale=1.):
        image = Image.open(pictures[name]).convert("L")
        factor = min(scale, maximum_width/image.width)
        w, h = factor*image.width, factor*image.height
        slide.shapes.add_picture(str(pictures[name]), px(x), px(y), width=px(w), height=px(h))
        ax.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255, interpolation="none",
            extent=(x, x+w, y+h, y), aspect="auto")
        objects.append(dict(kind="image", name=name, x=x, y=y, width=w, height=h))

    text("Question-conditioned image-language training", 45, 25, 1510, 60, size=32, bold=True)
    text("a   Observed training images: two questions about one paragraph", 65, 115, 1470, 40, size=23, bold=True)
    for i, x in enumerate((80, 850)):
        text(f"Question region Q{i}", x, 170, 600, 32, size=19, bold=True)
        picture(f"question_{i}", x, 212, 670)
        text(f"Recorded answer Y{i}", x, 275, 300, 34, size=19)
        picture(f"answer_{i}", x+320, 270, 330, scale=1.25)
    text("Each full prompt Xi also contains the same paragraph image. Appended white termination blocks are not drawn.",
        80, 332, 1450, 30, size=16, color=GREY)

    text("b   Four causal likelihoods under the same parameters", 65, 390, 1470, 40, size=23, bold=True)
    box(80, 465, 355, 115)
    text("(X0, Y0)   (X0, Y1)\n(X1, Y0)   (X1, Y1)", 100, 485, 315, 80, size=21, center=True)
    arrow(450, 522, 520)
    box(535, 465, 410, 115)
    text("Shared image generator\n490M parameters, 8 experts", 550, 485, 380, 80, size=22, center=True)
    arrow(960, 522, 1030)
    for j, label in enumerate(("Y0", "Y1")):
        text(label, 1110+j*180, 435, 180, 30, size=20, center=True)
    for i in range(2):
        text(f"X{i}", 1050, 477+i*62, 55, 35, size=20, center=True)
        for j in range(2):
            x, y = 1110+j*180, 465+i*62
            box(x, y, 180, 62, fill="E8F1EB" if i == j else "F5EBE6", border=GREEN if i == j else "AE593F")
            text(f"e{chr(0x2080+i)}{chr(0x2080+j)}", x, y+10, 180, 44, size=28, center=True)
    text("eij = mean negative log likelihood of the complete answer suffix, including termination", 80, 605, 1440, 32,
        size=18, color=GREY)

    text("c   Positive image likelihood with paired question assignment", 65, 670, 1470, 40, size=23, bold=True)
    text("D = e₀₁ + e₁₀ − e₀₀ − e₁₁", 80, 725, 670, 46, size=27)
    text("L = (e₀₀ + e₁₁) / 2 + λ · 0.01 · softplus(−D / 0.01)", 80, 778, 1450, 46, size=27)
    text("λ = 0: control;  λ = 1: assignment", 820, 730, 670, 32, size=20)
    text("Observed targets, not model outputs. Both arms start from the same V62 EMA; autonomous answers are evaluated separately.",
        65, 850, 1470, 30, size=15, color=GREY)

    slide.notes_slide.notes_text_frame.text = (
        "Training-method schematic only. Source: the first row of the frozen V63 training-pair manifest; "
        "CMRC 2018 training images; registered references/question_assignment_v63_protocol.md and actual preflight. "
        "Question images are exact regions of the full cached prompts; the shared paragraph is omitted from display. "
        "Answer bitmaps are observed targets with native padding; only appended termination is omitted from display. "
        "No generated model answer, benchmark score or language qualification is shown. All diagram objects and text are native/editable.")
    prs.save(HERE/"question_assignment_method_editable.pptx")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(HERE/f"question_assignment_method.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    for name in ("references/question_assignment_v63_protocol.md", "ilm/visual_lm/question_assignment.py",
                 "data/visual_qa/cmrc2018_short/train.jsonl",
                 "artifacts/native_pixel_instruction_v61_20261105/data/report.json",
                 "artifacts/question_assignment_v63_20261105/data/report.json",
                 "artifacts/question_assignment_v63_20261105/data/pairs.jsonl",
                 "artifacts/question_assignment_v63_20261105/preflight/report.json"):
        input_hashes[name] = digest(ROOT/name)
    manifest = dict(scope="Training method and observed training targets only", model_generation_calls=0,
        first_manifest_pair=pair, protocol_sha256=input_hashes["references/question_assignment_v63_protocol.md"],
        inputs_sha256=input_hashes, displayed_images=evidence, native_objects=objects,
        template_sha256=digest(template), generator_sha256=digest(__file__), language_quality_claim=False,
        artifacts={name: digest(HERE/name) for name in ("question_assignment_method.pdf", "question_assignment_method.svg",
            "question_assignment_method.png", "question_assignment_method_editable.pptx")})
    (HERE/"figure_sources.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(pair=pair["record_ids"], images=len(evidence), objects=len(objects), language_quality_claim=False)))


if __name__ == "__main__":
    main()
