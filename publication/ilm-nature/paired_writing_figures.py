#!/usr/bin/env python3
"""Append the completed paired-copy comparison without changing prior slides."""
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
BASE = "70fc934"
QA = ROOT / "artifacts/paired_writing_publication_20261107"
FIGURE = "Figure_S17_paired_writing"
RESULT = ROOT / "references/evidence/paired_padding_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    value = json.loads(RESULT.read_text())
    if (value["summary"] != {"train": {"cases": 32, "ocr_exact": 6}, "validation": {"cases": 32, "ocr_exact": 10}}
            or not value["all_72_answer_pngs_match_postprocessed_tensors"]
            or not value["all_160_source_pixel_restorations_exact"]
            or not value["ids_and_sigma_match_all_512_updates"]
            or value["complete_copy_reliability_pass"] or value["language_quality_qualified"]):
        raise ValueError("completed failed copying result required")
    for name, digest in value["files_sha256"].items():
        if sha(RESULT.parent / name) != digest:
            raise ValueError("source evidence changed: " + name)
    costs = value["cost"]
    values = [[costs[a]["elapsed_seconds"] for a in ("square_train", "strip_train")], [2, 3], [6, 10, 27, 28]]
    return value, values


def generate():
    value, values = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    size = presentation.find("p:sldSz", NS)
    if len(slides) != 31 or (int(size.get("cx")), int(size.get("cy"))) != (15240000, 8572500):
        raise ValueError("31-slide template required")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    specs = []
    settings = [(["Square", "Strip"], 900, 300, "Seconds", "0.0"),
                (["Square", "Strip"], 8, 2, "Exact / 8", "0"),
                (["Train", "Val", "Ref train", "Ref val"], 32, 8, "Exact / 32", "0")]
    for number, numbers, (categories, maximum, unit, title, formatting) in zip((32, 33, 34), values, settings):
        chart, workbook = chart_part(parts["ppt/charts/chart12.xml"], categories, numbers, title, 0, maximum, "296486")
        for parent in chart.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        for node in chart.findall(".//a:defRPr", NS):
            node.set("sz", "1700")
        for axis in chart.findall(".//c:valAx", NS):
            major = axis.find("c:majorUnit", NS)
            if major is None:
                major = element("c:majorUnit", axis)
            major.set("val", str(unit))
            number_format = axis.find("c:numFmt", NS)
            if number_format is None:
                number_format = element("c:numFmt", formatCode="0", sourceLinked="0")
                following = {q("c:" + n) for n in ("majorTickMark", "minorTickMark", "tickLblPos", "spPr", "txPr", "crossAx")}
                axis.insert(next(i for i, n in enumerate(axis) if n.tag in following), number_format)
            number_format.set("formatCode", "0")
            number_format.set("sourceLinked", "0")
        for node in chart.findall(".//c:numFmt", NS):
            node.set("formatCode", "0"); node.set("sourceLinked", "0")
        for node in chart.findall(".//c:dLbls/c:numFmt", NS):
            node.set("formatCode", formatting)
        bar = chart.find(".//c:barChart", NS)
        for node in bar.findall("c:dLbls", NS):
            bar.remove(node)
        labels = element("c:dLbls")
        element("c:numFmt", labels, formatCode=formatting, sourceLinked="0")
        element("c:dLblPos", labels, val="outEnd")
        for tag, show in (("showLegendKey", 0), ("showVal", 1), ("showCatName", 0),
                          ("showSerName", 0), ("showPercent", 0), ("showBubbleSize", 0)):
            element("c:" + tag, labels, val=show)
        bar.insert(list(bar).index(bar.find("c:axId", NS)), labels)
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/paired_writing_chart{number}.xlsx"
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        specs.append(dict(part=name, workbook=book, values=numbers, categories=categories, maximum=maximum))
    source_part = "ppt/slides/slide31.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Paired image learning remains unreliable", 45, 26, 1510, 58, 31, True)
    for x, title, number in ((45, "a  Training invocation (s)", 32), (565, "b  Fixed screen: OCR exact / 8", 33), (1085, "c  Full strip: OCR exact / 32", 34)):
        slide.shape(title, x, 105, 490, 48, 22, True)
        slide.chart(slide.relation("chart", f"../charts/chart{number}.xml"), x, 154, 475, 245)
    slide.shape("d  First four validation cases / full reference and generated canvases", 45, 402, 1500, 40, 23, True)
    pictures = []
    for i in range(4):
        case = value["cases"][32 + i]
        x = 60 + i * 385
        slide.shape(f"Validation {i:02}", x, 445, 330, 32, 20, True)
        for key, title, label_y, y in (("target", "Reference", 478, 505), ("answer", "Generated", 673, 700)):
            slide.shape(title, x, label_y, 330, 28, 18)
            source = RESULT.parent / case[key]
            with Image.open(source) as im:
                if im.size != (512, 256):
                    raise ValueError("raw evidence geometry changed")
            part = f"ppt/media/paired_writing_{i:02}_{key}.png"
            if part in parts:
                raise ValueError("picture collision")
            parts[part] = source.read_bytes()
            slide.picture(slide.relation("image", "../media/" + Path(part).name), f"Complete raw validation {i:02} {key}", x, y, 330, 165)
            pictures.append(dict(part=part, source=str(source.relative_to(ROOT)), sha256=sha(source)))
    slide.shape("512 updates/arm on 32 copies; reference OCR: 27/32 train, 28/32 validation. No QA training or language-efficiency claim.", 45, 868, 1510, 30, 16)
    slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/paired_writing_notes32.xml"
    slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
    part = "ppt/slides/slide32.xml"
    override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S17. Source: references/evidence/paired_padding_result/result.json and paired_image_writing_result/. "
        "Fresh rank16/alpha16 adapters from identical FLUX.2-klein-base-4B, 512 updates/arm, B1, 32 copies. "
        "Same training IDs and sigmas. White padding removal preserves all160 source RGB pages exactly; VAE input features change. "
        "Single training invocations795.6240408755839/287.18171550612897s, peak9.501317977905273/8.437482833862305GiB. "
        "Square/strip matched screen OCR2/8 and3/8; full strip6/32train10/32val; clean refs27/32,28/32. "
        "64logicalcases reuse8originalcalls and add56calls. Raw first4validation canvases, no crop or repair. All64public. "
        "30/32 legible exact strings-alone criterion fails visibly. No QA optimized. 3,899,903,488 deployed transformer "
        "+84,046,115VAE;24,358,912localtrainables. External pretraining and hybrid constant-condition provenance apply. "
        "Faster copying invocation is not an advantage at useful language quality.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            for i, node in enumerate(shape.findall(".//a:t", NS)):
                node.text = description if i == 0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target", "../slides/slide32.xml")
    override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note], parts[relpath(note)] = xml(notes), xml(rels)
    used = {r.get("Id") for r in relations}
    rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
    element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target="slides/slide32.xml")
    element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relations)):
        parts[name] = xml(node)
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=31, slides=32,
        charts=specs, pictures=pictures, builder_sha256=sha(__file__), source_result_sha256=sha(RESULT),
        artifact_tool_available=False, language_quality_claim=False, general_ilm_goal_complete=False)
    (HERE / "evidence/paired_writing_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    _, numbers = evidence()
    record = json.loads((HERE / "evidence/paired_writing_deck_build.json").read_text())
    if record["pptx_sha256"] != sha(HERE / "ILM_Figures_editable.pptx") or record["builder_sha256"] != sha(__file__) or record["source_result_sha256"] != sha(RESULT):
        raise ValueError("source/deck binding changed")
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old, zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError("old native part changed: " + name)
        for name in new.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 32:
            raise ValueError("incorrect slide count")
        for spec, expected in zip(record["charts"], numbers):
            chart = ET.fromstring(new.read(spec["part"]))
            cached = [float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
            categories = [n.text for n in chart.findall(".//c:ser/c:cat/c:strRef/c:strCache/c:pt/c:v", NS)]
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])), data_only=True)
            actual = [book.active.cell(i + 2, 2).value for i in range(len(expected))]
            labels = [book.active.cell(i + 2, 1).value for i in range(len(expected))]
            axes = chart.findall(".//c:valAx/c:scaling", NS)
            if (cached != expected or any(abs(a-b)>1e-9 for a,b in zip(actual,expected))
                    or categories != spec["categories"] or labels != spec["categories"] or not axes
                    or any(float(a.find("c:min", NS).get("val")) != 0 or float(a.find("c:max", NS).get("val")) != spec["maximum"] for a in axes)):
                raise ValueError("native chart/workbook data changed")
        for pic in record["pictures"]:
            if hashlib.sha256(new.read(pic["part"])).hexdigest() != pic["sha256"] or sha(ROOT/pic["source"]) != pic["sha256"]:
                raise ValueError("raw picture changed")
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
    if (len(prior), len(current), len(prefix)) != (31, 32, 31):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(31):
        before, after = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if before.samples != after.samples:
            raise ValueError("older slide appearance changed")
        preserved.append(hashlib.sha256(after.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=31, to_page=31)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=31, to_page=31)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    current[31].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[31].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
                  preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
                  new_slide=32, visual_review_required=True, artifact_marker_available=True)
    (HERE / "evidence/paired_writing_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide32 and preserved all31 previous native slide appearances.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = parser.parse_args()
    {"generate": generate, "verify": verify, "render": render}[args.stage]()
