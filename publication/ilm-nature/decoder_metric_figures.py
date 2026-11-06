#!/usr/bin/env python3
"""Append the completed decoder-loss comparison without changing prior slides."""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "af9571c"
QA = ROOT / "artifacts/decoder_metric_publication_20261107"
FIGURE = "Figure_S16_decoder_metric"
RESULT = ROOT / "references/evidence/decoder_metric_v82_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    value = json.loads(RESULT.read_text())
    verified = value["verification"]
    if (not verified["all_64_final_outputs_independently_scored"]
            or verified["training_fit_repair_pass"] or any(verified["gates"].values())
            or not value["raw_png_bytes_unchanged"] or not value["training_only"]
            or value["language_quality_claim"]):
        raise ValueError("completed failed matched comparison required")
    for name, digest in value["raw_png_sha256"].items():
        if sha(RESULT.parent / name) != digest:
            raise ValueError("source PNG changed")
    summaries = verified["summaries"]
    for arm in ("control", "metric"):
        if summaries[arm]["free_exact"] or summaries[arm]["assisted_exact_pixels"]:
            raise ValueError("figure labels no longer describe the source result")
    return value, [[100 * summaries[a][key] for a in ("control", "metric")]
                   for key in ("free_f1", "assisted_f1")]


def generate():
    value, values = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old:
        original = {name: old.read(name) for name in old.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    size = presentation.find("p:sldSz", NS)
    if len(slides) != 30 or (int(size.get("cx")), int(size.get("cy"))) != (15240000, 8572500):
        raise ValueError("the current 30-slide template and geometry must be preserved")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("new part collides with preserved content")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    specifications = []
    for number, numbers in zip((30, 31), values):
        chart, workbook = chart_part(parts["ppt/charts/chart12.xml"], ["Control", "Metric"], numbers,
                                      "Foreground F1 (%)", 0, 100, "296486")
        for parent in chart.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        for node in chart.findall(".//a:defRPr", NS):
            node.set("sz", "1900")
        for axis in chart.findall(".//c:valAx", NS):
            unit = axis.find("c:majorUnit", NS)
            if unit is None:
                unit = element("c:majorUnit", axis)
            unit.set("val", "25")
            formatting = axis.find("c:numFmt", NS)
            if formatting is None:
                formatting = element("c:numFmt", formatCode="0", sourceLinked="0")
                following = {q("c:" + n) for n in ("majorTickMark", "minorTickMark", "tickLblPos", "spPr", "txPr", "crossAx")}
                axis.insert(next(i for i, n in enumerate(axis) if n.tag in following), formatting)
            formatting.set("formatCode", "0")
            formatting.set("sourceLinked", "0")
        for node in chart.findall(".//c:numFmt", NS):
            node.set("formatCode", "0")
            node.set("sourceLinked", "0")
        bar = chart.find(".//c:barChart", NS)
        for node in bar.findall("c:dLbls", NS):
            bar.remove(node)
        labels = element("c:dLbls")
        element("c:numFmt", labels, formatCode="0.000", sourceLinked="0")
        element("c:dLblPos", labels, val="outEnd")
        for tag, show in (("showLegendKey", 0), ("showVal", 1), ("showCatName", 0),
                          ("showSerName", 0), ("showPercent", 0), ("showBubbleSize", 0)):
            element("c:" + tag, labels, val=show)
        bar.insert(list(bar).index(bar.find("c:axId", NS)), labels)
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/decoder_metric_chart{number}.xlsx"
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        specifications.append(dict(part=name, workbook=book, values=numbers, categories=["Control", "Metric"]))
    source_part = "ppt/slides/slide30.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Decoder-sensitive loss reduces answer-image fidelity", 45, 26, 1510, 58, 31, True)
    slide.shape("a  Autonomous writing / F1 (%)", 65, 110, 730, 48, 23, True)
    slide.chart(slide.relation("chart", "../charts/chart30.xml"), 45, 154, 730, 290)
    slide.shape("b  Correct preceding history / F1 (%)", 845, 110, 710, 48, 23, True)
    slide.chart(slide.relation("chart", "../charts/chart31.xml"), 825, 154, 730, 290)
    slide.shape("Exact complete images: 0/32 in both arms", 80, 447, 700, 38, 19)
    slide.shape("Exact complete images: 0/32 in both arms", 865, 447, 700, 38, 19)
    slide.shape("c  First four fixed questions / complete raw images", 65, 500, 1460, 45, 23, True)
    pictures = []
    for x, title in ((180, "Reference"), (680, "Control"), (1180, "Decoder metric")):
        slide.shape(title, x, 550, 370, 36, 20, True)
    for i in range(4):
        y = 598 + 66 * i
        slide.shape(f"{i:02}", 65, y + 8, 85, 40, 19)
        for x, arm, kind in ((180, "control", "reference"), (680, "control", "answer"), (1180, "metric", "answer")):
            source = RESULT.parent / "raw" / f"{arm}_after_{i:02}_{kind}.png"
            digest = value["raw_png_sha256"]["raw/" + source.name]
            with Image.open(source) as image:
                width, height = image.size
            if height != 32 or x + width * 1.5 > 1570:
                raise ValueError("complete raw sample does not fit the declared figure geometry")
            part = f"ppt/media/decoder_metric_{i:02}_{arm}_{kind}.png"
            if part in parts:
                raise ValueError("raw picture part collides")
            parts[part] = source.read_bytes()
            slide.picture(slide.relation("image", "../media/" + Path(part).name),
                          f"Unmodified case {i:02} {arm} {kind}", x, y, width * 1.5, height * 1.5)
            pictures.append(dict(part=part, source=str(source.relative_to(ROOT)), sha256=digest))
    slide.shape("32 training questions; 2,000 updates/arm. All four repair gates fail. Supplied-code F1: 99.9953% (28/32 exact).", 45, 867, 1510, 32, 17)
    slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/decoder_metric_notes31.xml"
    slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
    part = "ppt/slides/slide31.xml"
    override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S16. Source: references/evidence/decoder_metric_v82_result/result.json. "
                   "Same 32 seen training questions, same warm V81 adapted weights, 2000 updates/arm, B4, 8000 draws. "
                   f"Free F1 control/metric={values[0]}; correct-history F1={values[1]}. "
                   "All complete exact scores 0/32. Supplied-code F1 99.9953159%, 28/32 exact. "
                   "All four original metric repair gates fail. 7,583,225,971 complete parameters, "
                   "53,096,961 local trainables; external pretraining retained. 18.8928375 GiB allocated peak per arm. "
                   "First four fixed cases, all complete PNG bytes unchanged at 1.5x placement. "
                   "All cases remain in the source export. Image agreement is not semantic accuracy. "
                   "Source interruption adds compute excluded from the retained invocation timing. No language-success claim.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            for i, node in enumerate(shape.findall(".//a:t", NS)):
                node.text = description if i == 0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target", "../slides/slide31.xml")
    override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note], parts[relpath(note)] = xml(notes), xml(rels)
    used = {r.get("Id") for r in relations}
    rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
    element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target="slides/slide31.xml")
    element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation),
                       ("ppt/_rels/presentation.xml.rels", relations)):
        parts[name] = xml(node)
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=30, slides=31,
                  charts=specifications, pictures=pictures, builder_sha256=sha(__file__), source_result_sha256=sha(RESULT),
                  language_quality_claim=False, general_ilm_goal_complete=False)
    (HERE / "evidence/decoder_metric_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    _, numbers = evidence()
    record = json.loads((HERE / "evidence/decoder_metric_deck_build.json").read_text())
    if (record["pptx_sha256"] != sha(HERE / "ILM_Figures_editable.pptx")
            or record["builder_sha256"] != sha(__file__) or record["source_result_sha256"] != sha(RESULT)):
        raise ValueError("current deck/source binding changed")
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old, zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError("old native part changed: " + name)
        for name in new.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 31:
            raise ValueError("incorrect slide count")
        for spec, expected in zip(record["charts"], numbers):
            chart = ET.fromstring(new.read(spec["part"]))
            cached = [float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
            categories = [n.text for n in chart.findall(".//c:ser/c:cat/c:strRef/c:strCache/c:pt/c:v", NS)]
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])), data_only=True)
            actual = [book.active.cell(i + 2, 2).value for i in range(2)]
            labels = [book.active.cell(i + 2, 1).value for i in range(2)]
            axes = chart.findall(".//c:valAx/c:scaling", NS)
            if (cached != expected or any(abs(a - b) > 1e-12 for a, b in zip(actual, expected))
                    or labels != spec["categories"] or categories != spec["categories"]
                    or not axes or any(float(a.find("c:min", NS).get("val")) != 0
                                       or float(a.find("c:max", NS).get("val")) != 100 for a in axes)):
                raise ValueError("native chart/workbook differs from full-precision data")
        for pic in record["pictures"]:
            if hashlib.sha256(new.read(pic["part"])).hexdigest() != pic["sha256"] or sha(ROOT / pic["source"]) != pic["sha256"]:
                raise ValueError("raw embedded evidence changed")
    return record


def render():
    record = verify()
    QA.mkdir(parents=True, exist_ok=True)
    memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
    if memory["MemAvailable"] < 24 * 1024**2 or 1 - memory["SwapFree"] / memory["SwapTotal"] > .75:
        raise RuntimeError("shared workstation memory policy prevents rendering")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source = QA / "baseline.pptx"
        source.write_bytes(baseline("ILM_Figures_editable.pptx"))
        output = QA / "render"
        output.mkdir(exist_ok=True)
        command = ["soffice", "-env:UserInstallation=" + (QA / "libreoffice-profile").as_uri(),
                   "--headless", "--convert-to", "pdf", "--outdir", str(output), str(source), str(HERE / "ILM_Figures_editable.pptx")]
        with (QA / "libreoffice.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if process.wait(timeout=180):
                    raise RuntimeError("owned converter failed")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
    prior = fitz.open(output / "baseline.pdf")
    current = fitz.open(output / "ILM_Figures_editable.pdf")
    prefix = fitz.open(stream=baseline("ILM_Figures_review.pdf"), filetype="pdf")
    if (len(prior), len(current), len(prefix)) != (30, 31, 30):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(30):
        before, after = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if before.samples != after.samples:
            raise ValueError("older slide appearance changed")
        preserved.append(hashlib.sha256(after.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=30, to_page=30)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=30, to_page=30)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    current[30].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[30].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
                  preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
                  new_slide=31, visual_review_required=True, artifact_marker_available=False)
    (HERE / "evidence/decoder_metric_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide31 and preserved all30 previous native slide appearances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = parser.parse_args()
    {"generate": generate, "verify": verify, "render": render}[args.stage]()
