#!/usr/bin/env python3
"""Create a portable OOXML figure deck without a slide-authoring dependency.

The local Artifact Tool runtime is absent. Diagram objects and the measured
bar chart remain native PowerPoint elements; raw image evidence stays raster.
"""

from __future__ import annotations

import io
import json
import zipfile
from html import escape
from pathlib import Path

import openpyxl
from PIL import Image

HERE = Path(__file__).resolve().parent
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
EM = 9525


def relations(items):
    return (
        f'<Relationships xmlns="{REL}">'
        + "".join(
            f'<Relationship Id="{rid}" Type="{R}/{kind}" Target="{target}"/>'
            for rid, kind, target in items
        )
        + "</Relationships>"
    )


def group():
    return '<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'


def transform(x, y, w, h):
    return f'<a:xfrm><a:off x="{int(x*EM)}" y="{int(y*EM)}"/><a:ext cx="{int(w*EM)}" cy="{int(h*EM)}"/></a:xfrm>'


def shape(
    ident,
    text,
    x,
    y,
    w,
    h,
    *,
    size=24,
    fill=None,
    bold=False,
    color="1D2B34",
    align="l",
):
    background = (
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        if fill
        else "<a:noFill/>"
    )
    paragraphs = "".join(
        f'<a:p><a:pPr algn="{align}"/><a:r><a:rPr lang="en-US" sz="{size*100}" b="{int(bold)}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Arial"/></a:rPr><a:t>{escape(line)}</a:t></a:r><a:endParaRPr lang="en-US"/></a:p>'
        for line in text.split("\n")
    )
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{ident}" name="{escape(text[:45])}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>{transform(x,y,w,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{background}<a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr wrap="square" lIns="90000" rIns="90000" tIns="60000" bIns="60000"/><a:lstStyle/>{paragraphs}</p:txBody></p:sp>'


def arrow(ident, x, y, w):
    return f'<p:sp><p:nvSpPr><p:cNvPr id="{ident}" name="Flow arrow"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>{transform(x,y,w,14)}<a:prstGeom prst="rightArrow"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="28638B"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>'


def picture(ident, rid, x, y, w, h):
    return f'<p:pic><p:nvPicPr><p:cNvPr id="{ident}" name="Raw measured image evidence"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr>{transform(x,y,w,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'


def slide(body):
    return f'<p:sld xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>{group()}{body}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'


def main():
    data = json.loads((HERE / "evidence/v53_development_report.json").read_text())
    output = HERE / "ILM_Figures_editable.pptx"
    files = {}
    categories = ["Future 1", "Future 2", "Future 3", "Future 4"]
    routes = [
        ("ordered", "Ordered", "28638B"),
        ("shuffled", "Shuffled", "939B9F"),
        ("zero_condition", "Zero condition", "AE593F"),
    ]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evidence"
    ws.append(["Position"] + [name for _, name, _ in routes])
    for pos in range(4):
        ws.append(
            [categories[pos]]
            + [
                100 * data["generation"][r]["identity_top1_by_position"][pos]
                for r, _, _ in routes
            ]
        )
    stream = io.BytesIO()
    wb.save(stream)
    files["ppt/embeddings/evidence.xlsx"] = stream.getvalue()
    series = []
    for idx, (route, name, color) in enumerate(routes):
        vals = [100 * v for v in data["generation"][route]["identity_top1_by_position"]]
        column = chr(66 + idx)
        cats = "".join(
            f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(categories)
        )
        numbers = "".join(
            f'<c:pt idx="{i}"><c:v>{v}</c:v></c:pt>' for i, v in enumerate(vals)
        )
        series.append(
            f'<c:ser><c:idx val="{idx}"/><c:order val="{idx}"/><c:tx><c:v>{name}</c:v></c:tx><c:spPr><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></c:spPr><c:cat><c:strRef><c:f>Evidence!$A$2:$A$5</c:f><c:strCache><c:ptCount val="4"/>{cats}</c:strCache></c:strRef></c:cat><c:val><c:numRef><c:f>Evidence!${column}$2:${column}$5</c:f><c:numCache><c:formatCode>0.00</c:formatCode><c:ptCount val="4"/>{numbers}</c:numCache></c:numRef></c:val></c:ser>'
        )
    files["ppt/charts/chart1.xml"] = (
        f'<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:a="{A}" xmlns:r="{R}"><c:lang val="en-US"/><c:chart><c:autoTitleDeleted val="1"/><c:plotArea><c:layout/><c:barChart><c:barDir val="col"/><c:grouping val="clustered"/>{"".join(series)}<c:gapWidth val="120"/><c:axId val="1"/><c:axId val="2"/></c:barChart><c:catAx><c:axId val="1"/><c:scaling><c:orientation val="minMax"/></c:scaling><c:axPos val="b"/><c:tickLblPos val="nextTo"/><c:crossAx val="2"/><c:crosses val="autoZero"/></c:catAx><c:valAx><c:axId val="2"/><c:scaling><c:orientation val="minMax"/><c:max val="2"/><c:min val="0"/></c:scaling><c:axPos val="l"/><c:majorGridlines/><c:numFmt formatCode="0.0" sourceLinked="0"/><c:tickLblPos val="nextTo"/><c:crossAx val="1"/><c:crosses val="autoZero"/></c:valAx></c:plotArea><c:legend><c:legendPos val="b"/><c:layout/><c:overlay val="0"/></c:legend><c:plotVisOnly val="1"/></c:chart><c:externalData r:id="rId1"><c:autoUpdate val="0"/></c:externalData></c:chartSpace>'
    )
    files["ppt/charts/_rels/chart1.xml.rels"] = relations(
        [("rId1", "package", "../embeddings/evidence.xlsx")]
    )
    body = shape(
        2,
        "Conditional generation of visible language",
        45,
        25,
        1500,
        70,
        size=34,
        bold=True,
    )
    body += shape(3, "a   Training", 45, 125, 500, 55, size=25, bold=True)
    body += shape(
        4,
        "60 visible glyph images\nFrozen causal reader",
        50,
        215,
        410,
        120,
        fill="EEF3F7",
    )
    body += arrow(5, 480, 260, 80)
    body += shape(
        6,
        "Continuous visual history\n384 state + 1024 image field",
        580,
        215,
        440,
        120,
        fill="EEF3F7",
    )
    body += shape(
        7,
        "Four noisy future cells\nIndependent or shared noise levels",
        50,
        415,
        410,
        120,
        fill="EEF3F7",
    )
    body += arrow(8, 480, 460, 80)
    body += shape(
        9,
        "Conditional strip denoiser\nFiLM and cell attention",
        580,
        415,
        440,
        120,
        fill="EEF3F7",
    )
    body += arrow(10, 1040, 460, 70)
    body += shape(
        11,
        "Velocity and visual losses\n5.90M trainable parameters",
        1130,
        415,
        410,
        120,
        fill="EEF3F7",
    )
    body += shape(
        12, "History conditions every residual scale", 590, 345, 600, 45, size=18
    )
    body += shape(13, "b   Inference", 45, 610, 500, 55, size=25, bold=True)
    body += shape(14, "Noise and visible history", 50, 705, 410, 80, fill="EEF3F7")
    body += arrow(15, 480, 737, 80)
    body += shape(16, "32-level reverse process", 580, 705, 440, 80, fill="EEF3F7")
    body += arrow(17, 1040, 737, 70)
    body += shape(
        18,
        "Four output images\nAppend pixels to next context",
        1130,
        705,
        410,
        125,
        fill="EEF3F7",
    )
    body += shape(
        19,
        "Source: V53 frozen protocol, implementation and locked audit. Schematic objects are editable.",
        45,
        850,
        1500,
        35,
        size=14,
    )
    files["ppt/slides/slide1.xml"] = slide(body)
    chart_frame = f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="3" name="Measured identity chart"/><p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr><p:xfrm><a:off x="{100*EM}" y="{150*EM}"/><a:ext cx="{1380*EM}" cy="{540*EM}"/></p:xfrm><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:chart xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" r:id="rId2"/></a:graphicData></a:graphic></p:graphicFrame>'
    body = (
        shape(
            2,
            "Direct block generation under paired visual histories",
            45,
            25,
            1500,
            75,
            size=32,
            bold=True,
        )
        + chart_frame
    )
    body += shape(4, "Identity (%)", 45, 100, 400, 40, size=20)
    body += shape(
        5,
        "256 prefixes, one primary sample each, four output cells\nOrdered block identity 0.586%, shuffled 0.488%, zero condition 0.195%",
        70,
        725,
        1450,
        85,
        size=23,
    )
    body += shape(
        6,
        "Source: locked V53 development report. Native chart includes its editable evidence workbook.",
        45,
        850,
        1500,
        35,
        size=14,
    )
    files["ppt/slides/slide2.xml"] = slide(body)
    raw = (
        HERE.parents[1]
        / "artifacts/causal_visual_block_diffusion_v53_20261104/development/target_ordered_shuffled_zero_plan.png"
    )
    image = Image.open(raw)
    # Retain eight full rows, including all 17 original audit columns.
    image = image.crop((0, 0, image.width, 8 * 34))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    files["ppt/media/raw.png"] = stream.getvalue()
    body = shape(
        2,
        "Unselected output images expose the language error",
        45,
        25,
        1500,
        70,
        size=32,
        bold=True,
    )
    displayed_height = 590
    displayed_width = displayed_height * image.width / image.height
    left = (1600 - displayed_width) / 2
    cell_width = displayed_width / 17
    for index, title in enumerate(
        ("Target (4)", "Ordered (4)", "Shuffled (4)", "Zero condition (4)", "Plan")
    ):
        width = cell_width * (1 if index == 4 else 4)
        body += shape(
            20 + index,
            title,
            left + index * 4 * cell_width,
            130,
            width,
            60,
            size=20,
            align="ctr",
        )
    body += picture(
        4, "rId2", (1600 - displayed_width) / 2, 215, displayed_width, displayed_height
    )
    body += shape(
        5,
        "Fixed first eight audit rows. Pixels are model output, with nearest-neighbour enlargement. No glyph replacement.",
        45,
        850,
        1500,
        35,
        size=14,
    )
    files["ppt/slides/slide3.xml"] = slide(body)
    for idx in range(1, 4):
        rels = [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]
        if idx == 2:
            rels.append(("rId2", "chart", "../charts/chart1.xml"))
        if idx == 3:
            rels.append(("rId2", "image", "../media/raw.png"))
        files[f"ppt/slides/_rels/slide{idx}.xml.rels"] = relations(rels)
    files["ppt/presentation.xml"] = (
        f'<p:presentation xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId4"/></p:sldMasterIdLst><p:sldIdLst><p:sldId id="256" r:id="rId1"/><p:sldId id="257" r:id="rId2"/><p:sldId id="258" r:id="rId3"/></p:sldIdLst><p:sldSz cx="{1600*EM}" cy="{900*EM}" type="custom"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>'
    )
    files["ppt/_rels/presentation.xml.rels"] = relations(
        [
            *(("rId" + str(i), "slide", f"slides/slide{i}.xml") for i in range(1, 4)),
            ("rId4", "slideMaster", "slideMasters/slideMaster1.xml"),
        ]
    )
    files["ppt/slideMasters/slideMaster1.xml"] = (
        f'<p:sldMaster xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}"><p:cSld><p:spTree>{group()}</p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" folHlink="folHlink" hlink="hlink" tx1="dk1" tx2="dk2"/><p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    )
    files["ppt/slideMasters/_rels/slideMaster1.xml.rels"] = relations(
        [("rId1", "slideLayout", "../slideLayouts/slideLayout1.xml")]
    )
    files["ppt/slideLayouts/slideLayout1.xml"] = (
        f'<p:sldLayout xmlns:p="{P}" xmlns:a="{A}" xmlns:r="{R}" type="blank"><p:cSld name="Blank"><p:spTree>{group()}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    )
    files["ppt/slideLayouts/_rels/slideLayout1.xml.rels"] = relations(
        [("rId1", "slideMaster", "../slideMasters/slideMaster1.xml")]
    )
    files["_rels/.rels"] = relations(
        [("rId1", "officeDocument", "ppt/presentation.xml")]
    )
    types = [
        ("/ppt/presentation.xml", "presentationml.presentation.main"),
        ("/ppt/slideMasters/slideMaster1.xml", "presentationml.slideMaster"),
        ("/ppt/slideLayouts/slideLayout1.xml", "presentationml.slideLayout"),
    ]
    types.extend(
        (f"/ppt/slides/slide{i}.xml", "presentationml.slide") for i in range(1, 4)
    )
    overrides = "".join(
        f'<Override PartName="{path}" ContentType="application/vnd.openxmlformats-officedocument.{kind}+xml"/>'
        for path, kind in types
    )
    files["[Content_Types].xml"] = (
        f'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Default Extension="xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/>{overrides}<Override PartName="/ppt/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/></Types>'
    )
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(
                name, value if isinstance(value, bytes) else value.encode()
            )
    print(output)


if __name__ == "__main__":
    main()
