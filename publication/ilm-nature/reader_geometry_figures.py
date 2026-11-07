#!/usr/bin/env python3
"""Append the measured geometry comparison, preserving 33 native slides."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import openpyxl
from PIL import Image

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "353ae8f"
QA = ROOT / "artifacts/native_reader_geometry_publication_20261107"
FIGURE = "Figure_S19_reader_geometry"
RESULT = ROOT / "references/evidence/native_visual_alignment_result/result.json"
SOURCE = ROOT / "references/evidence/native_reader_geometry_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    result, source = [json.loads(path.read_text()) for path in (RESULT, SOURCE)]
    for path, value in ((RESULT, result), (SOURCE, source)):
        if value["general_ilm_goal_complete"]:
            raise ValueError("no completed ILM claim admitted")
        for name, digest in value["files_sha256"].items():
            if sha(path.parent / name) != digest:
                raise ValueError("changed experiment evidence: " + name)
    if result["training"]["updates"] != 2560:
        raise ValueError("complete curriculum required")
    copy = result["cohorts"]["full_validation"]["groups"]["validation/copy"]["exact_stopped"]
    qa = result["cohorts"]["full_validation"]["groups"]["validation/qa"]["exact_stopped"]
    values = [[27, copy, 0, qa], [source["models"][key]["summary"]["normal_exact_stopped"]
              for key in ("source_1024", "source_432", "predecessor_432", "dense_432")]]
    if values != [[27, 29, 0, 0], [0, 11, 5, 5]]:
        raise ValueError("completed figure values changed")
    return result, source, values


def generate():
    result, source, values = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 33:
        raise ValueError("current 33-slide baseline required")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("native part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    specs = []
    settings = [(["Copy\nbefore", "Copy\ndense", "Chinese QA\nbefore", "Chinese QA\ndense"], 32, 8),
                (["Source\n1024", "Source\n432", "Adapters\n432", "Dense\n432"], 32, 8)]
    for number, numbers, (categories, maximum, unit) in zip((38, 39), values, settings):
        chart, workbook = chart_part(parts["ppt/charts/chart12.xml"], categories, numbers, "Exact stopped", 0, maximum, "296486")
        for parent in chart.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        for node in chart.findall(".//a:defRPr", NS):
            node.set("sz", "1550")
        for axis in chart.findall(".//c:valAx", NS):
            major = axis.find("c:majorUnit", NS)
            if major is None:
                major = element("c:majorUnit", axis)
            major.set("val", str(unit))
            formatting = axis.find("c:numFmt", NS)
            if formatting is None:
                formatting = element("c:numFmt", formatCode="0", sourceLinked="0")
                following = {q("c:" + name) for name in ("majorTickMark", "minorTickMark", "tickLblPos", "spPr", "txPr", "crossAx")}
                axis.insert(next(i for i, node in enumerate(axis) if node.tag in following), formatting)
        for node in chart.findall(".//c:numFmt", NS):
            node.set("formatCode", "0")
            node.set("sourceLinked", "0")
        bar = chart.find(".//c:barChart", NS)
        for node in bar.findall("c:dLbls", NS):
            bar.remove(node)
        labels = element("c:dLbls")
        element("c:numFmt", labels, formatCode="0", sourceLinked="0")
        element("c:dLblPos", labels, val="outEnd")
        for tag, show in (("showLegendKey", 0), ("showVal", 1), ("showCatName", 0),
                          ("showSerName", 0), ("showPercent", 0), ("showBubbleSize", 0)):
            element("c:" + tag, labels, val=show)
        bar.insert(list(bar).index(bar.find("c:axId", NS)), labels)
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/reader_geometry_chart{number}.xlsx"
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        specs.append(dict(part=name, workbook=book, values=numbers, categories=categories))
    source_part = "ppt/slides/slide33.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Native image geometry recovers source reading", 45, 26, 1510, 58, 31, True)
    for x, title, number in ((45, "a  Original geometry: exact answers / 32", 38),
                              (835, "b  English geometry check: exact / 32", 39)):
        slide.shape(title, x, 105, 720, 48, 22, True)
        slide.chart(slide.relation("chart", f"../charts/chart{number}.xml"), x, 154, 700, 245)
    slide.shape("c  Same complete page at both input scales", 45, 420, 750, 45, 23, True)
    pictures = []
    sources = [ROOT / "references/evidence/native_english_transfer_result/inputs/00_0.png",
               SOURCE.parent / "inputs/00_0.png"]
    for i, (path, size, label) in enumerate(zip(sources, (1024, 432),
            ("1024 square: 4096 visual positions", "432 square: 729 visual positions"))):
        x = 60 + i * 365
        slide.shape(label, x, 474, 340, 55, 18)
        with Image.open(path) as image:
            if image.size != (size, size):
                raise ValueError("complete square source required")
        part = f"ppt/media/reader_geometry_input_{i:02}.png"
        parts[part] = path.read_bytes()
        slide.picture(slide.relation("image", "../media/" + Path(part).name), "Complete original input", x, 540, 290, 290)
        pictures.append(dict(part=part, source=str(path.relative_to(ROOT)), sha256=sha(path)))
    slide.shape("d  First four source answers at 432: text readouts", 835, 420, 720, 45, 22, True)
    slide.shape("Primary reference", 845, 474, 255, 35, 18, True)
    slide.shape("Source answer", 1120, 474, 430, 35, 18, True)
    for i, case in enumerate(source["models"]["source_432"]["cases"][:4]):
        y = 523 + i * 82
        slide.shape(case["alternatives"][0], 845, y, 255, 75, 17)
        slide.shape(case["normal"]["text"], 1120, y, 430, 75, 17)
    slide.shape("Auxiliary Unicode readouts. Exact scores accept all annotated alternatives and require EOS. No trained image-answer result.", 45, 868, 1510, 30, 16)
    slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/reader_geometry_notes34.xml"
    slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
    part = "ppt/slides/slide34.xml"
    override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S19. Completed curriculum: 2560 additional updates, original geometry; copying 27 to 29/32, Chinese QA 0/32. "
        "Source geometry comparison: same 32 SQuAD questions and matched question/passage erasures, all 80 calls/model, fixed seeds, no optimizer. "
        "Source 1024 0/32, source 432 11/32; prior and dense adapters at 432 5/32 each. Nine of 11 correct source answers lost without passage. "
        "Whole-square bicubic resize changes pixels and position count jointly; unique mechanism not isolated. "
        "Complete first page shown at both scales to document intervention. First four source readouts retained in fixed order, not all exact. "
        "SQuAD CC-BY-SA-4.0, Rajpurkar et al. 2016; source Show-o2 NUS Show Lab Apache-2.0. "
        "Auxiliary Unicode head; no image answers or independent ILM claimed. Sources: references/evidence/native_visual_alignment_result, "
        "native_english_transfer_result and native_reader_geometry_result. " + json.dumps(dict(chart_values=values), ensure_ascii=False))
    for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            for i, node in enumerate(shape.findall(".//a:t", NS)):
                node.text = description if i == 0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target", "../slides/slide34.xml")
    override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note], parts[relpath(note)] = xml(notes), xml(rels)
    used = {r.get("Id") for r in relations}
    rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
    element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target="slides/slide34.xml")
    element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relations)):
        parts[name] = xml(node)
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=33, slides=34,
        charts=specs, pictures=pictures, builder_sha256=sha(__file__), source_result_sha256=sha(RESULT),
        source_control_sha256=sha(SOURCE), artifact_tool_available=False, general_ilm_goal_complete=False)
    (HERE / "evidence/native_reader_geometry_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    _, _, values = evidence()
    record = json.loads((HERE / "evidence/native_reader_geometry_deck_build.json").read_text())
    if record["pptx_sha256"] != sha(HERE / "ILM_Figures_editable.pptx") or record["builder_sha256"] != sha(__file__) or record["source_result_sha256"] != sha(RESULT):
        raise ValueError("source/deck binding changed")
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old, zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError("earlier native part changed: " + name)
        for name in new.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 34:
            raise ValueError("current deck must contain 34 slides")
        for spec, expected in zip(record["charts"], values):
            chart = ET.fromstring(new.read(spec["part"]))
            actual = [float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])), data_only=True)
            if actual != expected or [book.active.cell(i + 2, 2).value for i in range(len(expected))] != expected:
                raise ValueError("chart or embedded workbook differs")
        for picture in record["pictures"]:
            if hashlib.sha256(new.read(picture["part"])).hexdigest() != picture["sha256"]:
                raise ValueError("original input-page image changed")
    return record


def render():
    import fitz
    from build_paired_writing_publication import build_slot
    record = verify()
    QA.mkdir(parents=True, exist_ok=True)
    with build_slot():
        source = QA / "baseline.pptx"
        source.write_bytes(baseline("ILM_Figures_editable.pptx"))
        output = QA / "render"
        output.mkdir(exist_ok=True)
        command = ["soffice", "-env:UserInstallation=" + (QA / "libreoffice-profile").as_uri(),
            "--headless", "--convert-to", 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            "--outdir", str(output), str(source), str(HERE / "ILM_Figures_editable.pptx")]
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
    if (len(prior), len(current), len(prefix)) != (33, 34, 33):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(33):
        a, b = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=33, to_page=33)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=33, to_page=33)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    embedded = [Image.open(io.BytesIO(figure.extract_image(item[0])["image"])).convert("RGB")
                for item in figure[0].get_images(full=True)]
    for picture in record["pictures"]:
        with Image.open(ROOT / picture["source"]) as image:
            image = image.convert("RGB")
            if not any(candidate.size == image.size and candidate.tobytes() == image.tobytes() for candidate in embedded):
                raise ValueError("standalone PDF changed input-page pixels")
    current[33].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[33].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
        new_slide=34, visual_review_required=True, artifact_tool_available=False)
    result["both_pdf_input_images_pixel_identical"] = True
    (HERE / "evidence/native_reader_geometry_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide 34 and preserved all 33 earlier native slide appearances.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args()
    {"generate": generate, "verify": verify, "render": render}[args.stage]()
