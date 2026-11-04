#!/usr/bin/env python3
"""Append measured reading scores and unselected image answers to the deck."""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image
from pptx import Presentation
import torch

import generate_v56_editable_figures
from generate_evidence import HERE
from generate_v55_editable_figures import px, text
from generate_v56_editable_figures import chart
from generate_v57_evidence import RUN, ROUTES, pad_answer


def picture(slide, bitmap, x, y, w, h):
    buffer = io.BytesIO()
    Image.fromarray(np.round(255 * (1 - bitmap)).astype(np.uint8)).save(
        buffer, format="PNG"
    )
    buffer.seek(0)
    slide.shapes.add_picture(buffer, px(x), px(y), width=px(w), height=px(h))


def main():
    generate_v56_editable_figures.main()
    path = HERE / "ILM_Figures_editable.pptx"
    prs = Presentation(path)
    report = json.loads((RUN / "development/report.json").read_text())
    if report["smoke_only"] or len(report["primary_ids"]) != 256:
        raise ValueError("production audit required")
    m = report["measurements"]
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(
        slide,
        "Natural Chinese questions: some binding, inaccurate complete answers",
        45,
        25,
        1510,
        75,
        size=30,
        bold=True,
    )
    chart(
        slide,
        ["Copy / full question", "Copy / blank question", "Matched no-copy"],
        [
            ("Exact stopped answer (%)", [100 * m[r]["exact_answer"] for r in ROUTES]),
            ("Multiset raster F1 (%)", [100 * m[r]["raster_f1"] for r in ROUTES]),
        ],
        maximum=25,
        height=290,
    )
    gain = report["comparisons"]["copy_minus_copy_blank_question_raster_f1"]
    text(
        slide,
        f"Question-dependent F1 gain: {100*gain['mean']:.3f} points [{100*gain['ci95'][0]:.3f}, {100*gain['ci95'][1]:.3f}]\nExact answers: 10/256; both questions correct: 1/393 pairs. Four of seven gates pass; model rejected.",
        65,
        450,
        1470,
        110,
        size=22,
    )
    evidence = torch.load(
        RUN / "development/publication_first_four.pt",
        map_location="cpu",
        weights_only=False,
    )
    for i in range(2):
        text(slide, f"Q{i+1}", 45, 585 + i * 45, 70, 35, size=18, bold=True)
        q = np.concatenate([im[0].numpy() for im in evidence["questions"][i]], axis=1)
        picture(slide, q, 160, 590 + i * 45, q.shape[1] / 32 * 20, 20)
    for x, (label, key) in zip(
        [45, 430, 815, 1200],
        [
            ("Reference 1", "references"),
            ("Copy", "copy"),
            ("Blank question", "copy_blank_question"),
            ("No-copy", "no_copy"),
        ],
    ):
        text(slide, label, x, 695, 355, 40, size=20, bold=True, center=True)
        bitmap = np.concatenate(
            [pad_answer(evidence[key][i]) for i in range(2)], axis=0
        )
        picture(slide, bitmap, x, 755, 355, 355 * 64 / 544)
    text(
        slide,
        "First two fixed audit rows; actual output images. White means zero emitted cells. All primary copy-arm answer cells use observed prompt images.",
        45,
        855,
        1510,
        35,
        size=14,
    )
    prs.save(path)
    print(f"Saved {len(prs.slides)} editable slides")


if __name__ == "__main__":
    main()
