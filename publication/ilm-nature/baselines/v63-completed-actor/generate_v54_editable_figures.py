#!/usr/bin/env python3
"""Retain the V53 plates and append editable V54 mechanism and evidence."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import openpyxl
from PIL import Image
import torch

import generate_editable_figures as base
from generate_v54_evidence import RUN, CHECKPOINT


def main():
    base.main()
    path = base.HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    report = json.loads((RUN / "development/development_report.json").read_text())
    if report["checkpoint_sha256"] != CHECKPOINT:
        raise ValueError("unexpected density checkpoint")
    sh = base.shape
    body = sh(
        2,
        "Exact raster likelihood preserves a visible copy path",
        45,
        25,
        1510,
        75,
        size=32,
        bold=True,
    )
    for ident, x, w, title in (
        (3, 50, 400, "64 writing images\nFrozen 16.28M reader"),
        (4, 580, 440, "Continuous visual state\nHidden384 + image field1024"),
        (5, 1130, 410, "Fixed inverse field\nAligned signed pixel plan"),
    ):
        body += sh(ident, title, x, 180, w, 125, fill="EEF3F7")
    body += base.arrow(6, 475, 235, 80) + base.arrow(7, 1040, 235, 70)
    body += sh(
        8, "Training: exact Bernoulli image NLL", 45, 365, 1500, 55, size=26, bold=True
    )
    body += sh(
        9,
        "Earlier target rows\nFour-layer row transformer",
        50,
        450,
        400,
        120,
        fill="EEF3F7",
    )
    body += base.arrow(10, 475, 498, 80)
    body += sh(
        11,
        "Earlier pixels within row\nStrictly masked MADE",
        580,
        450,
        440,
        120,
        fill="EEF3F7",
    )
    body += base.arrow(12, 1040, 498, 70)
    body += sh(
        13,
        "Aligned logit + residual\nNormalized next-image law",
        1130,
        450,
        410,
        120,
        fill="EEF3F7",
    )
    body += sh(
        14,
        "25% supplied-image motor exercises; 75% natural predicted conditions. No glyph inventory.",
        50,
        605,
        1480,
        55,
        size=22,
    )
    body += sh(
        15,
        "Inference: sample 1,024 causal pixels; append the visible raster to history",
        50,
        700,
        1480,
        65,
        size=26,
        bold=True,
    )
    body += sh(
        16,
        "2.43M new density parameters; 18.71M total. Supplied-image copying is architectural, not language evidence.",
        45,
        845,
        1510,
        40,
        size=15,
    )
    files["ppt/slides/slide4.xml"] = base.slide(body)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Density"
    ws.append(["Condition", "Pixel F1 x 100", "Nearest identity (%)"])
    conditions = [
        ("Copy / greedy", "motor", "copy_greedy"),
        ("Copy / sample", "motor", "copy_sample"),
        ("Next / sample", "generation", "ordered"),
    ]
    for label, section, route in conditions:
        item = report[section][route]
        ws.append([label, 100 * item["pixel_f1"], 100 * item["identity_top1"]])
    ws.append([])
    ws.append(["Exact target NLL (nats/image)", "Ordered", "Shuffled"])
    ws.append(
        [
            "512 targets",
            report["likelihood"]["ordered"]["nats_per_image"],
            report["likelihood"]["shuffled"]["nats_per_image"],
        ]
    )
    stream = io.BytesIO()
    wb.save(stream)
    files["ppt/embeddings/density.xlsx"] = stream.getvalue()
    template = ET.fromstring(files["ppt/charts/chart1.xml"])
    ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart", "a": base.A}
    c = ns["c"]
    bar = template.find(".//c:barChart", ns)
    prototypes = bar.findall("c:ser", ns)
    for series in prototypes:
        bar.remove(series)
    for idx, (title, color) in enumerate(
        (("Pixel F1 x 100", "28638B"), ("Nearest identity (%)", "AE593F"))
    ):
        series = ET.fromstring(ET.tostring(prototypes[idx]))
        series.find("c:tx/c:v", ns).text = title
        series.find("c:spPr/a:solidFill/a:srgbClr", ns).set("val", color)
        series.find("c:cat/c:strRef/c:f", ns).text = "Density!$A$2:$A$4"
        series.find(
            "c:val/c:numRef/c:f", ns
        ).text = f"Density!${chr(66+idx)}$2:${chr(66+idx)}$4"
        for cache_tag, values in (
            ("c:cat/c:strRef/c:strCache", [x[0] for x in conditions]),
            (
                "c:val/c:numRef/c:numCache",
                [ws.cell(row, idx + 2).value for row in range(2, 5)],
            ),
        ):
            cache = series.find(cache_tag, ns)
            for point in cache.findall("c:pt", ns):
                cache.remove(point)
            cache.find("c:ptCount", ns).set("val", "3")
            for i, value in enumerate(values):
                point = ET.SubElement(cache, f"{{{c}}}pt", {"idx": str(i)})
                ET.SubElement(point, f"{{{c}}}v").text = str(value)
        bar.insert(idx + 2, series)
    template.find(".//c:valAx/c:scaling/c:max", ns).set("val", "110")
    for prefix, uri in (("c", c), ("a", base.A), ("r", base.R)):
        ET.register_namespace(prefix, uri)
    files["ppt/charts/chart2.xml"] = ET.tostring(
        template, encoding="utf-8", xml_declaration=True
    )
    files["ppt/charts/_rels/chart2.xml.rels"] = base.relations(
        [("rId1", "package", "../embeddings/density.xlsx")]
    )
    frame = f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="3" name="Exact raster motor and language chart"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="{90*base.EM}" y="{145*base.EM}"/><a:ext cx="{1400*base.EM}" cy="{440*base.EM}"/></p:xfrm><a:graphic><a:graphicData uri="{c}"><c:chart xmlns:c="{c}" r:id="rId2"/></a:graphicData></a:graphic></p:graphicFrame>'
    body = (
        sh(
            2,
            "Copying succeeds; autonomous language remains unresolved",
            45,
            25,
            1510,
            75,
            size=30,
            bold=True,
        )
        + frame
    )
    body += sh(
        4,
        "Exact-image copying: 256/256 greedy. Autonomous target identity: 4/256 ordered, 3/256 shuffled.",
        65,
        620,
        1470,
        60,
        size=23,
    )
    body += sh(
        5,
        "Target NLL: 15.219 ordered; 15.254 shuffled nats/image\nPaired advantage 0.0355 nat; source-record bootstrap 95% CI [-0.1014, 0.1693]",
        65,
        700,
        1470,
        110,
        size=23,
    )
    body += sh(
        6,
        "V54 locked development audit. Native chart and embedded workbook. Supplied-image controls reveal the answer.",
        45,
        850,
        1510,
        35,
        size=14,
    )
    files["ppt/slides/slide5.xml"] = base.slide(body)

    gen = torch.load(
        RUN / "development/generation_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    motor = torch.load(
        RUN / "development/motor_image_evidence.pt",
        map_location="cpu",
        weights_only=False,
    )
    arrays = [
        gen["targets"],
        motor["outputs"]["copy_greedy"],
        gen["outputs"]["ordered"],
        gen["outputs"]["shuffled"],
        gen["outputs"]["greedy"],
    ]
    body = sh(
        2,
        "Actual rasters: exact motor copying and failed continuation",
        45,
        25,
        1510,
        75,
        size=30,
        bold=True,
    )
    for i, (label, array) in enumerate(
        zip(
            [
                "Target",
                "Supplied / greedy",
                "Ordered / sample",
                "Shuffled / sample",
                "Ordered / greedy",
            ],
            arrays,
        )
    ):
        pixels = np.concatenate([cell[0, 0].numpy() for cell in array[:8]], axis=0)
        im = Image.fromarray(np.uint8((1 - pixels) * 255))
        stream = io.BytesIO()
        im.save(stream, format="PNG")
        files[f"ppt/media/v54_{i}.png"] = stream.getvalue()
        body += sh(10 + i, label, 50 + i * 305, 125, 295, 60, size=20, align="ctr")
        body += base.picture(20 + i, f"rId{2+i}", 155 + i * 305, 210, 74, 592)
    body += sh(
        30,
        "First eight fixed audit rows; nearest-neighbour enlargement. No OCR, image replacement or selected examples.",
        45,
        850,
        1510,
        35,
        size=14,
    )
    files["ppt/slides/slide6.xml"] = base.slide(body)
    for index in range(4, 7):
        rels = [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]
        if index == 5:
            rels.append(("rId2", "chart", "../charts/chart2.xml"))
        if index == 6:
            rels.extend(
                (f"rId{2+i}", "image", f"../media/v54_{i}.png") for i in range(5)
            )
        files[f"ppt/slides/_rels/slide{index}.xml.rels"] = base.relations(rels)
    pres = ET.fromstring(files["ppt/presentation.xml"])
    ids = pres.find(f"{{{base.P}}}sldIdLst")
    for index in range(4, 7):
        ET.SubElement(
            ids,
            f"{{{base.P}}}sldId",
            {"id": str(255 + index), f"{{{base.R}}}id": f"rId{index+1}"},
        )
    ET.register_namespace("p", base.P)
    files["ppt/presentation.xml"] = ET.tostring(
        pres, encoding="utf-8", xml_declaration=True
    )
    rels = ET.fromstring(files["ppt/_rels/presentation.xml.rels"])
    for index in range(4, 7):
        ET.SubElement(
            rels,
            f"{{{base.REL}}}Relationship",
            {
                "Id": f"rId{index+1}",
                "Type": base.R + "/slide",
                "Target": f"slides/slide{index}.xml",
            },
        )
    ET.register_namespace("", base.REL)
    files["ppt/_rels/presentation.xml.rels"] = ET.tostring(
        rels, encoding="utf-8", xml_declaration=True
    )
    types = ET.fromstring(files["[Content_Types].xml"])
    ct = "http://schemas.openxmlformats.org/package/2006/content-types"
    for index in range(4, 7):
        ET.SubElement(
            types,
            f"{{{ct}}}Override",
            {
                "PartName": f"/ppt/slides/slide{index}.xml",
                "ContentType": "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
            },
        )
    ET.SubElement(
        types,
        f"{{{ct}}}Override",
        {
            "PartName": "/ppt/charts/chart2.xml",
            "ContentType": "application/vnd.openxmlformats-officedocument.drawingml.chart+xml",
        },
    )
    ET.register_namespace("", ct)
    files["[Content_Types].xml"] = ET.tostring(
        types, encoding="utf-8", xml_declaration=True
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for name, value in files.items():
            out.writestr(name, value)
    print("Appended three editable V54 evidence slides")


if __name__ == "__main__":
    main()
