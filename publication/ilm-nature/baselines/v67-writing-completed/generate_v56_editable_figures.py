#!/usr/bin/env python3
"""Add native form/search charts and untouched feedback rasters to the deck."""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Pt
import torch

import generate_v55_editable_figures
from generate_evidence import HERE
from generate_v55_editable_figures import px, text
from generate_v56_evidence import FORMS, RUN


def chart(slide, categories, series, *, maximum, height):
    data = CategoryChartData()
    data.categories = categories
    for label, values in series:
        data.add_series(label, values)
    c = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, px(90), px(140), px(1400), px(height), data
    ).chart
    c.has_legend = True
    c.legend.position = XL_LEGEND_POSITION.BOTTOM
    c.legend.include_in_layout = False
    c.legend.font.name = "Arial"
    c.legend.font.size = Pt(14)
    c.value_axis.minimum_scale = 0
    c.value_axis.maximum_scale = maximum
    c.value_axis.tick_labels.font.size = Pt(14)
    c.category_axis.tick_labels.font.size = Pt(14)
    for s, color in zip(c.series, ["28638B", "AE593F"]):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = RGBColor.from_string(color)
        s.format.line.fill.background()


def main():
    generate_v55_editable_figures.main()
    path = HERE / "ILM_Figures_editable.pptx"
    prs = Presentation(path)
    forms = json.loads((FORMS / "report.json").read_text())
    report = json.loads((RUN / "report.json").read_text())
    if report["windows"] != 2048 or report["decision"] != "reject_decoder":
        raise ValueError("unexpected fixed audit")
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(slide, "Clear glyphs and correct language are separate tests", 45, 25,
         1510, 75, size=30, bold=True)
    names = ["v54", "v55_frozen", "v55_joint"]
    chart(slide, ["V54", "V55 frozen", "V55 joint"], [
        ("Exact known form (%)", [100 * forms["summaries"][n]["generation"]["ordered"]["exact_known_raster_fraction"] for n in names]),
        ("Exact observed target (%)", [100 * forms["summaries"][n]["generation"]["ordered"]["exact_target_fraction"] for n in names]),
    ], maximum=60, height=420)
    text(slide, "Exact known-form gain, joint minus frozen: 10.156 percentage points\n95% source-record interval [4.219, 16.240]; 256 saved first samples",
         65, 610, 1470, 120, size=24)
    text(slide, "The fixed 9,932-image bank is used only for evaluation. No image is repaired or replaced.",
         65, 760, 1470, 65, size=22)
    text(slide, "Post-result diagnostic; all original qualification decisions remain unchanged. Native chart includes its editable workbook.",
         45, 855, 1510, 35, size=14)

    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(slide, "Raw-pixel beam search: better immediate choice, repetitive output",
         45, 25, 1510, 75, size=30, bold=True)
    routes = ["beam8", "beam8_shuffled", "greedy"]
    m = report["measurements"]
    chart(slide, ["Ordered beam", "Shuffled beam", "Greedy"], [
        ("Exact next-image match (%)", [100 * m["generation"][r]["exact_target"] for r in routes]),
        ("Adjacent feedback repetition (%)", [100 * m["feedback"][r]["adjacent_repeated_raster_fraction"] for r in routes]),
    ], maximum=40, height=330)
    text(slide, "2,048 prefixes: beam gain over greedy 3.369 points [2.351, 4.412]; over shuffled 1.758 [0.966, 2.552]\nObserved-stream repetition: 0.112%; beam: 25.446%. Rollout gate fails; decoder rejected.",
         65, 490, 1470, 115, size=21)
    raw = torch.load(RUN / "feedback_image_evidence.pt", map_location="cpu", weights_only=False)
    panels = [("Observed future", raw["targets"]),
              ("Ordered beam", raw["outputs"]["beam8"]),
              ("Shuffled beam", raw["outputs"]["beam8_shuffled"]),
              ("Original greedy", raw["outputs"]["greedy"])]
    for x, (label, array) in zip([45, 430, 815, 1200], panels):
        text(slide, label, x, 635, 355, 45, size=20, bold=True, center=True)
        rows = [np.concatenate([cell[0].numpy() for cell in source], axis=1) for source in array[:2]]
        bitmap = Image.fromarray(np.round(255 * (1 - np.concatenate(rows, axis=0))).astype(np.uint8))
        buffer = io.BytesIO()
        bitmap.save(buffer, format="PNG")
        buffer.seek(0)
        slide.shapes.add_picture(buffer, px(x), px(695), width=px(355), height=px(88.75))
    text(slide, "Unselected first two eight-step audit streams. Unchanged V55 weights; no vocabulary or repair. Both controls use highest FP32 precision.",
         45, 855, 1510, 35, size=14)
    prs.save(path)
    print(f"Saved {len(prs.slides)} editable slides")


if __name__ == "__main__":
    main()
