#!/usr/bin/env python3
"""Append one measured visual-access plate, preserving the previous deck."""
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
BASE = "25e6a7b"
QA = ROOT / "artifacts/native_visual_access_publication_20261107"
FIGURE = "Figure_S18_visual_access"
RESULT = ROOT / "references/evidence/native_visual_access_result/result.json"
SOURCE = ROOT / "references/evidence/native_unified_source_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    result = json.loads(RESULT.read_text())
    source = json.loads(SOURCE.read_text())
    if result["complete_finite_updates"] != 4096 or result["validation_ids_optimized"] != 0 or result["general_ilm_goal_complete"]:
        raise ValueError("complete bounded observation required")
    for path, value in ((RESULT, result), (SOURCE, source)):
        for name, digest in value["files_sha256"].items():
            if sha(path.parent / name) != digest:
                raise ValueError("source evidence changed: " + name)
    cohorts = result["cohorts"]
    screen = [cohorts[stage]["groups"][group]["exact_stopped"] for group in ("validation/copy", "validation/qa") for stage in ("before_v1", "after_v1")]
    full = [cohorts["full_validation"]["groups"]["validation/" + task]["exact_stopped"] for task in ("copy", "qa")]
    controls = cohorts["paired_question_controls"]
    values = [screen, full, [controls[k] for k in ("both_questions_exact", "normal_outputs_differ", "erased_matches_either_reference")]]
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
    if len(slides) != 32:
        raise ValueError("current 32-slide baseline required")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("native part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    specs = []
    settings = [(["Copy\nbefore", "Copy\nafter", "QA\nbefore", "QA\nafter"], 4, 1),
                (["Copy", "QA"], 32, 8),
                (["Both correct", "Different", "Erased match"], 16, 4)]
    for number, numbers, (categories, maximum, unit) in zip((35, 36, 37), values, settings):
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
        book = f"ppt/embeddings/visual_access_chart{number}.xlsx"
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        specs.append(dict(part=name, workbook=book, values=numbers, categories=categories))
    source_part = "ppt/slides/slide32.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Visual reading after adapting the image frontend", 45, 26, 1510, 58, 31, True)
    for x, title, number in ((45, "a  Held-out screen: exact / 4", 35),
                              (565, "b  Full validation: exact / 32", 36),
                              (1085, "c  Passage-pair controls / 16", 37)):
        slide.shape(title, x, 105, 490, 48, 22, True)
        slide.chart(slide.relation("chart", f"../charts/chart{number}.xml"), x, 154, 475, 245)
    slide.shape("d  Source writing: supplied-string controls", 45, 420, 750, 45, 23, True)
    pictures = []
    for i in range(2):
        case = source["cases"][i]
        x = 60 + i * 360
        slide.shape(case["expected"], x, 470, 330, 65, 18)
        path = SOURCE.parent / case["assisted_writing"]
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise ValueError("complete source-writing image required")
        part = f"ppt/media/visual_access_source_writer_{i:02}.png"
        parts[part] = path.read_bytes()
        slide.picture(slide.relation("image", "../media/" + Path(part).name), "Complete assisted source output", x, 535, 300, 300)
        pictures.append(dict(part=part, source=str(path.relative_to(ROOT)), sha256=sha(path)))
    slide.shape("e  First four held-out copies: auxiliary text readouts", 820, 420, 740, 45, 22, True)
    slide.shape("Requested", 830, 470, 340, 35, 18, True)
    slide.shape("After learning", 1190, 470, 360, 35, 18, True)
    shown = result["cohorts"]["full_validation"]["cases"][:4]
    for i, case in enumerate(shown):
        if case["id"] != f"copy/validation/{i:03}" or max(len(case["expected"]), len(case["text"])) > 80:
            raise ValueError("first-case order changed or full response needs a revised layout")
        for x, words in ((830, case["expected"]), (1190, case["text"])):
            shape = slide.shape(words, x, 515 + 78 * i, 350, 75, 18)
            for run in shape.findall(".//a:rPr", NS):
                element("a:ea", run, typeface="Noto Sans CJK SC")
    slide.shape("4,096 updates; 384 training examples; only visual adapters learned. Text readouts are not image answers. General ILM remains unachieved.", 45, 868, 1510, 30, 16)
    slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/visual_access_notes33.xml"
    slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
    part = "ppt/slides/slide33.xml"
    override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S18. Sources: references/evidence/native_visual_access_result/result.json and native_unified_source_result/result.json. "
        "Show-o2-1.5B-HQ, NUS Show Lab, Apache-2.0. Gated rank16 visual-only adaptation, 8,489,984 trainables, 384 train/64 held out. "
        "Native language body/head frozen; auxiliary Unicode supervision, no independent image-answer result. "
        "Screen and complete-cohort repeats not pooled. Exact requires NFC/declared whitespace canonicalization and EOS; case/quotes retained. "
        "Controls: sixteen distinct-answer passage pairs; two normal calls and one shared erased page per passage, identical seeds. "
        "Different outputs are not necessarily correct. Erased match means matching either reference, sixteen independent calls. "
        "First two complete original supplied-string source-writing controls are assisted, not trained image answers. "
        "First four held-out copies are displayed as native text-head outputs. All observations retained. " + json.dumps(dict(chart_values=values), ensure_ascii=False))
    for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            for i, node in enumerate(shape.findall(".//a:t", NS)):
                node.text = description if i == 0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target", "../slides/slide33.xml")
    override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note], parts[relpath(note)] = xml(notes), xml(rels)
    used = {r.get("Id") for r in relations}
    rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
    element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target="slides/slide33.xml")
    element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relations)):
        parts[name] = xml(node)
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=32, slides=33,
        charts=specs, pictures=pictures, builder_sha256=sha(__file__), source_result_sha256=sha(RESULT),
        source_control_sha256=sha(SOURCE), artifact_tool_available=False, general_ilm_goal_complete=False)
    (HERE / "evidence/native_visual_access_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    _, _, values = evidence()
    record = json.loads((HERE / "evidence/native_visual_access_deck_build.json").read_text())
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
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 33:
            raise ValueError("current deck must contain 33 slides")
        for spec, expected in zip(record["charts"], values):
            chart = ET.fromstring(new.read(spec["part"]))
            actual = [float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])), data_only=True)
            if actual != expected or [book.active.cell(i + 2, 2).value for i in range(len(expected))] != expected:
                raise ValueError("chart or embedded workbook differs")
        for picture in record["pictures"]:
            if hashlib.sha256(new.read(picture["part"])).hexdigest() != picture["sha256"]:
                raise ValueError("original source-writing image changed")
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
    if (len(prior), len(current), len(prefix)) != (32, 33, 32):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(32):
        a, b = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=32, to_page=32)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=32, to_page=32)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    embedded = [Image.open(io.BytesIO(figure.extract_image(item[0])["image"])).convert("RGB")
                for item in figure[0].get_images(full=True)]
    for picture in record["pictures"]:
        with Image.open(ROOT / picture["source"]) as image:
            image = image.convert("RGB")
            if not any(candidate.size == image.size and candidate.tobytes() == image.tobytes() for candidate in embedded):
                raise ValueError("standalone PDF changed source-writing pixels")
    current[32].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[32].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
        new_slide=33, visual_review_required=True, artifact_tool_available=False)
    result["both_pdf_source_writing_images_pixel_identical"] = True
    (HERE / "evidence/native_visual_access_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide 33 and preserved all 32 earlier native slide appearances.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args()
    {"generate": generate, "verify": verify, "render": render}[args.stage]()
