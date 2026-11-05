#!/usr/bin/env python3
"""Append three editable measured reading-intervention slides."""

from __future__ import annotations

import json

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
import torch

import generate_v57_editable_figures
from generate_evidence import HERE
from generate_reading_interventions import RUNS
from generate_v55_editable_figures import px, text
from generate_v56_editable_figures import chart
from generate_v57_editable_figures import picture
from generate_v57_evidence import pad_answer


def box(slide, label, x, y, w, h):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, px(x), px(y), px(w), px(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor.from_string("EDF2F5")
    shape.line.color.rgb = RGBColor.from_string("647985")
    text(slide, label, x+8, y+8, w-16, h-16, size=19, center=True)


def score(slide, report, routes, labels, height=300):
    chart(slide, labels, [
        ("Exact answer (%)", [100*report["measurements"][r]["exact_answer"] for r in routes]),
        ("Multiset raster F1 (%)", [100*report["measurements"][r]["raster_f1"] for r in routes]),
    ], maximum=max(45, 10+100*max(report["measurements"][r]["raster_f1"] for r in routes)), height=height)


def main():
    generate_v57_editable_figures.main()
    path = HERE / "ILM_Figures_editable.pptx"
    prs = Presentation(path)
    reports = {k: json.loads((v / "development/report.json").read_text()) for k, v in RUNS.items()}
    r = reports["v58"]
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(slide, "Whole observed image answers: better F1, unresolved exact-match gain", 45, 25, 1510, 75, size=30, bold=True)
    score(slide, r, ["block", "block_blank_question", "glyph"],
          ["Whole image block", "Question blanked", "Matched glyph answers"])
    text(slide, "25/256 exact answers; matched glyph control 21/256. F1 gain 4.893 points [0.469, 9.047].\nOnly 4/393 paragraph pairs have both answers correct. Six of seven gates pass; reading rejected.",
         65, 455, 1470, 90, size=21)
    for x, label in [(50, "Prompt pixels"), (435, "Causal reader"), (820, "Start / end law"), (1205, "Source image answer")]:
        box(slide, label, x, 570, 330, 75)
        if x < 1200:
            text(slide, "→", x+335, 580, 45, 45, size=26, center=True)
    for i, identifier in enumerate(r["primary_ids"][:2]):
        from generate_evidence import ROOT
        from ilm.visual_lm.visual_qa_data import VisualQARenderer
        records = {x["id"]: x for x in map(json.loads,
            (ROOT / "data/visual_qa/cmrc2018_short/development.jsonl").read_text().splitlines())}
        ref = VisualQARenderer()(records[identifier]["references"][0]["text"])
        raw = torch.load(RUNS["v58"] / "development/predictions/block" / (identifier+".pt"),
                         map_location="cpu", weights_only=False)["images"]
        y = 700+i*65
        text(slide, f"Audit row {i+1}", 50, y, 180, 40, size=18)
        picture(slide, pad_answer(ref, 16), 250, y, 480, 30)
        picture(slide, pad_answer(raw, 16), 900, y, 480, 30)
    text(slide, "Retained reference", 250, 660, 500, 35, size=17, bold=True)
    text(slide, "Actual block answer", 900, 660, 500, 35, size=17, bold=True)
    text(slide, "First two fixed audit rows. Copying preserves selected pixels; it does not certify the meaning of the selected text.",
         45, 855, 1510, 35, size=14)
    r = reports["v59"]
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(slide, "Frozen visual foundation probe: larger pretrained features do not help here", 45, 25, 1510, 75, size=30, bold=True)
    score(slide, r, ["student", "student_blank", "mixar", "mixar_blank"],
          ["Student", "Student / blank Q", "MIXAR", "MIXAR / blank Q"], height=400)
    text(slide, "Identical 789,520-parameter heads; 4,000 matched updates each.\nStudent 24/256 exact; MIXAR 6/256. MIXAR-minus-student F1 −13.752 points [−18.247, −9.094].",
         65, 590, 1470, 100, size=23)
    text(slide, "MIXAR has different pretraining and native rendering; this probe uses our Noto CJK grid.\nExternal four-H200 pretraining is not one-4090 work. Frozen readout comparison rejected; not a general encoder ranking.",
         65, 720, 1470, 100, size=21)
    text(slide, "Continuous prompt-image features only. Shared cached release weights; no external inference call or answer annotation enters feature preparation.",
         45, 855, 1510, 35, size=14)
    r = reports["v60"]
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(slide, "Does reading the question first improve evidence selection?", 45, 25, 1510, 75, size=30, bold=True)
    score(slide, r, ["context", "context_blank", "question", "question_blank"],
          ["Context first", "Context / blank Q", "Question first", "Question / blank Q"])
    for y, name, first, second in [(520, "Context first", "Passage images X", "Question images Q"),
                                   (635, "Question first", "Question images Q", "Passage images X")]:
        text(slide, name, 55, y+15, 260, 50, size=23, bold=True)
        for x, label in [(340, first), (790, second), (1240, "Answer")]:
            box(slide, label, x, y, 310, 75)
            if x < 1200:
                text(slide, "→", x+340, y+10, 60, 45, size=27, center=True)
    delta = r["comparisons"]["question_minus_context_raster_f1"]
    text(slide, f"Question-first minus context-first F1: {100*delta['mean']:.3f} points [{100*delta['ci95'][0]:.3f}, {100*delta['ci95'][1]:.3f}].\n{r['gates_passed']}/{len(r['gates'])} registered gates pass; decision: {r['decision']}.",
         65, 750, 1470, 80, size=21)
    text(slide, "Same checkpoint, data, parameters and training budget. Source pixels only; reused development, not a sealed final test or general language qualification.",
         45, 855, 1510, 35, size=14)
    prs.save(path)
    print(f"Saved {len(prs.slides)} editable slides")


if __name__ == "__main__":
    main()
