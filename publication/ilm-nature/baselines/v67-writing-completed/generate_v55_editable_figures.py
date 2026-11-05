#!/usr/bin/env python3
"""Append native matched-learning charts and raw image evidence to the deck."""

from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt
import torch

import generate_v54_editable_figures
from generate_evidence import HERE
from generate_v55_evidence import HASHES, RUN


def px(n):
    return round(n * 9525)


def text(slide, words, x, y, w, h, *, size=24, bold=False, center=False):
    frame = slide.shapes.add_textbox(px(x), px(y), px(w), px(h)).text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = px(5)
    frame.margin_top = frame.margin_bottom = px(3)
    for i, line in enumerate(words.split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = "Arial"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string("1D2B34")
        if center:
            p.alignment = PP_ALIGN.CENTER


def main():
    generate_v54_editable_figures.main()
    path = HERE / "ILM_Figures_editable.pptx"
    prs = Presentation(path)
    matched = json.loads((RUN / "matched_development_report.json").read_text())
    if matched["checkpoint_sha256"] != HASHES:
        raise ValueError("unexpected matched pair")
    reports = {
        arm: json.loads((RUN / arm / "development/development_report.json").read_text())
        for arm in HASHES
    }
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(
        slide,
        "Matched joint learning: better likelihood, incomplete generation",
        45,
        25,
        1510,
        75,
        size=30,
        bold=True,
    )
    data = CategoryChartData()
    data.categories = ["Frozen / ordered", "Joint / ordered", "Joint / shuffled"]
    pairs = [("frozen", "ordered"), ("joint", "ordered"), ("joint", "shuffled")]
    data.add_series(
        "Pixel F1 x 100",
        [100 * reports[a]["generation"][r]["pixel_f1"] for a, r in pairs],
    )
    data.add_series(
        "Target identity (%)",
        [100 * reports[a]["generation"][r]["identity_top1"] for a, r in pairs],
    )
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, px(90), px(145), px(1400), px(420), data
    ).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.name = "Arial"
    chart.legend.font.size = Pt(14)
    chart.value_axis.minimum_scale = 0
    chart.value_axis.maximum_scale = 45
    chart.value_axis.tick_labels.font.size = Pt(14)
    chart.category_axis.tick_labels.font.size = Pt(14)
    for series, color in zip(chart.series, ["28638B", "AE593F"]):
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = RGBColor.from_string(color)
        series.format.line.fill.background()
    gain = matched["joint_nll_advantage_over_matched_frozen"]
    nll_f = reports["frozen"]["likelihood"]["ordered"]["nats_per_image"]
    nll_j = reports["joint"]["likelihood"]["ordered"]["nats_per_image"]
    text(
        slide,
        f'Exact target NLL: joint {nll_j:.3f}; matched frozen {nll_f:.3f} nats/image\nPaired gain {gain["mean"]:.3f}; 95% source-record interval [{gain["ci95"][0]:.3f}, {gain["ci95"][1]:.3f}]',
        65,
        620,
        1470,
        115,
        size=24,
    )
    text(
        slide,
        "Both arms retain exact supplied-image copying. Joint model passes 6/8 gates; generation remains unqualified.",
        65,
        755,
        1470,
        65,
        size=22,
    )
    text(
        slide,
        "V55 locked matched audit; 512 NLL targets, 256 single-sample prefixes. Native chart includes its editable workbook.",
        45,
        855,
        1510,
        35,
        size=14,
    )

    frozen = torch.load(
        RUN / "frozen/development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    joint = torch.load(
        RUN / "joint/development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    motor = torch.load(
        RUN / "joint/development/motor_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    text(
        slide,
        "Raw matched output: exact copying and remaining language errors",
        45,
        25,
        1510,
        75,
        size=29,
        bold=True,
    )
    arrays = [
        joint["targets"],
        frozen["outputs"]["ordered"],
        joint["outputs"]["ordered"],
        joint["outputs"]["shuffled"],
        joint["outputs"]["greedy"],
        motor["outputs"]["copy_greedy"],
    ]
    labels = [
        "Target",
        "Frozen sample",
        "Joint sample",
        "Joint shuffled",
        "Joint greedy",
        "Supplied copy",
    ]
    for i, (label, array) in enumerate(zip(labels, arrays)):
        text(slide, label, 50 + i * 250, 125, 245, 60, size=20, center=True)
        pixels = np.concatenate([v[0, 0].numpy() for v in array[:8]], axis=0)
        picture = Image.fromarray(np.uint8((1 - pixels) * 255))
        stream = io.BytesIO()
        picture.save(stream, format="PNG")
        stream.seek(0)
        slide.shapes.add_picture(
            stream, px(135 + i * 250), px(210), width=px(74), height=px(592)
        )
    text(
        slide,
        "First eight fixed audit rows, no selection. Pixels are actual output; no OCR, evaluator-bank replacement or external inference.",
        45,
        855,
        1510,
        35,
        size=14,
    )
    prs.save(path)
    print("Eight-slide deck with native matched evidence chart")


if __name__ == "__main__":
    main()
