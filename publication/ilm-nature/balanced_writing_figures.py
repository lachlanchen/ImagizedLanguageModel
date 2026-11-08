#!/usr/bin/env python3
"""Append the assessed balanced-writing figure and preserve 40 prior slides."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image

from conditional_learning_figures import multi_chart
from generate_assignment_editable_figures import NS, Slide, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "d8b2aa551a8e3c606e19eeb9f7b96247f9426815"
QA = ROOT / "artifacts/balanced_writing_publication_20261108"
FIGURES = ("Figure_S26_balanced_writing",)
SOURCES = {name: ROOT / "references/evidence" / folder / "result.json" for name, folder in
    (("control", "autonomous_writing_control_result"), ("pilot", "autonomous_writer_result"),
     ("balanced", "balanced_autonomous_writer_result"))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    data = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    for key, value in data.items():
        if value["complete_outputs"] != 64 or len(value["rows"]) != 64:
            raise ValueError("complete assessed cohort required")
        for name, digest in value["reports"].items():
            if sha(SOURCES[key].parent / name) != digest:
                raise ValueError("bound source report changed")
        for row in value["rows"]:
            field = "control" if key == "control" else "answer"
            if sha(SOURCES[key].parent / row["files"][field]) != row["png_sha256"]:
                raise ValueError("actual unmodified output image changed")
        for split in ("train", "validation"):
            if value["summary"][split] != dict(cases=32, ocr_exact=sum(r["ocr_exact"] for r in value["rows"] if r["split"] == split)):
                raise ValueError("complete OCR counts differ")
    if data["control"]["summary"] != {"train": dict(cases=32, ocr_exact=5), "validation": dict(cases=32, ocr_exact=10)}:
        raise ValueError("preserved control differs")
    if data["pilot"]["summary"] != {"train": dict(cases=32, ocr_exact=15), "validation": dict(cases=32, ocr_exact=13)}:
        raise ValueError("preserved pilot differs")
    reviewed = json.loads((SOURCES["balanced"].parent / "raw_review.json").read_text())
    if not reviewed["complete_cohort_reviewed"] or reviewed["result_sha256"] != sha(SOURCES["balanced"]):
        raise ValueError("complete current raw review required before publication")
    return data


def generate():
    data = evidence()
    QA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as z:
        original = {n: z.read(n) for n in z.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 40:
        raise ValueError("immutable 40-slide source required")
    charts, pictures = [], []
    source_part = "ppt/slides/slide40.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])

    def override(name, kind):
        if name in parts or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("new part collision: " + name)
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def chart(slide, categories, series, labels, maximum, unit, x, y, w, h, decimals=False):
        number = 48 + len(charts)
        name, book = f"ppt/charts/chart{number}.xml", f"ppt/embeddings/balanced_chart{number}.xlsx"
        node, workbook = multi_chart(parts["ppt/charts/chart47.xml"], categories, series, labels, maximum, unit)
        for prop in node.findall(".//a:defRPr", NS) + node.findall(".//a:rPr", NS):
            prop.set("sz", "1800")
        if decimals:
            for fmt in node.findall(".//c:numFmt", NS):
                fmt.set("formatCode", "0.0")
        node.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(node), xml(rels), workbook
        slide.chart(slide.relation("chart", "../charts/" + Path(name).name), x, y, w, h)
        charts.append(dict(part=name, workbook=book, categories=categories, series=series, labels=labels))

    def picture(slide, source, name, owner, x, y, width):
        stable = HERE / "figures" / (name + ".png")
        stable.write_bytes(source.read_bytes())
        part = "ppt/media/" + stable.name
        if part in parts:
            raise ValueError("new image collision")
        parts[part] = stable.read_bytes()
        with Image.open(stable) as im:
            slide.picture(slide.relation("image", "../media/" + stable.name), name,
                x, y, width, width * im.height / im.width)
        pictures.append(dict(part=part, source=str(stable.relative_to(ROOT)), sha256=sha(stable), slide=owner))

    def append(slide, number, description):
        slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
        note, part = f"ppt/notesSlides/balanced_notes{number}.xml", f"ppt/slides/slide{number}.xml"
        slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
        override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        note_source = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
        notes = ET.fromstring(parts[note_source])
        for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
            ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if ph is not None and ph.get("type") == "body":
                for i, text in enumerate(shape.findall(".//a:t", NS)):
                    text.text = description if i == 0 else ""
        rels = ET.fromstring(parts[relpath(note_source)])
        for node in rels:
            if node.get("Type").endswith("/slide"):
                node.set("Target", f"../slides/slide{number}.xml")
        override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        parts[note], parts[relpath(note)] = xml(notes), xml(rels)
        used = {r.get("Id") for r in relationships}
        rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
        element("pkg:Relationship", relationships, Id=rid, Type=NS["r"] + "/slide", Target=f"slides/slide{number}.xml")
        element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)

    s = Slide(parts[source_part])
    s.shape("Balanced exposure and autonomous image writing", 45, 24, 1510, 60, 31, True)
    s.shape("a  Complete OCR-exact copies (32 per split)", 45, 105, 745, 40, 22, True)
    names = (("control", "Old loss +32"), ("pilot", "Final image +32"), ("balanced", "Then +128"))
    values = [(label, [data[key]["summary"][split]["ocr_exact"] for split in ("train", "validation")]) for key, label in names]
    chart(s, ["Seen", "Held out"], values, [v for _, v in values], 32, 8, 45, 155, 725, 280)
    s.shape("b  Held-out copying by language (16 each)", 825, 105, 730, 40, 22, True)
    by_language = [(label, [sum(r["ocr_exact"] for r in data[key]["rows"] if r["split"] == "validation" and r["language"] == language)
        for language in ("zh", "en")]) for key, label in names]
    chart(s, ["Chinese", "English"], by_language, [v for _, v in by_language], 16, 4, 825, 155, 730, 280)
    trained = json.loads((SOURCES["balanced"].parent / "training_report.json").read_text())
    passed = data["balanced"]["exploratory_progress_rule_met"]
    s.shape("128 additional updates\nFour complete passes over 32 examples\n64 Chinese and 64 English draws", 45, 490, 450, 100, 18)
    s.shape(f"Training: {trained['elapsed_seconds']:,.1f} s\nPeak: {trained['peak_allocated_vram_gib_this_invocation']:.3f} GiB\nFull 50-step trajectory gradient", 45, 615, 450, 100, 18)
    s.shape("Continuation rule: " + ("met" if passed else "not met") +
        "\nCopying does not establish semantic QA.\nExtra updates and exposure change together.", 45, 740, 450, 110, 17)
    s.shape("c  First two held-out inputs: complete actual pages", 530, 470, 1025, 38, 22, True)
    for col, label in enumerate(("Target", "Starting pilot", "After +128")):
        s.shape(label, 530 + col * 340, 515, 330, 30, 17, True)
    for index, row in enumerate(data["balanced"]["rows"][32:34]):
        if row["id"] != f"copy/validation/{index:03}":
            raise ValueError("fixed first two cases required")
        for col, suffix in enumerate(("reference", "starting", "answer")):
            picture(s, SOURCES["balanced"].parent / row["files"][suffix], f"balanced_case{index}_{suffix}",
                41, 530 + col * 340, 550 + index * 160, 320)
    s.shape("Canonical writing style; reused development cases; inherited 3.900B transformer + 84.046M VAE.", 530, 873, 1025, 25, 14)
    append(s, 41, "Figure S26. Complete old-objective control, initial final-image pilot, and balanced continuation. "
        "Control/pilot retrospectively compared at32 additional updates, equal exposure but different compute/objective weights. "
        "Balanced128 updates continue pilot AdamW32to160 and four complete example permutations. "
        "Raw first two validation cases fixed before evaluation, no repair. Counts are independent OCR observations, "
        "not manual writing accuracy or semantic QA. Reused development; no independent-seed replication. "
        + json.dumps(dict(charts=charts, exploratory_progress_rule_met=passed, source_paths={k:str(v.relative_to(ROOT)) for k,v in SOURCES.items()})))
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relationships)):
        parts[name] = xml(node)
    target = QA / "draft.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in parts.items():
            z.writestr(name, raw)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=40,
        slides=41, original_chart_count=47, charts=charts, pictures=pictures,
        sources_sha256={str(p.relative_to(ROOT)): sha(p) for p in SOURCES.values()},
        builder_sha256=sha(__file__), artifact_tool_available=False,
        powerpoint_app_inspection=False, general_ilm_goal_complete=False)
    (QA / "deck_build.json").write_text(json.dumps(record, indent=2) + "\n")
    verify()


def verify():
    evidence()
    record = json.loads((QA / "deck_build.json").read_text())
    target = QA / "draft.pptx"
    if sha(target) != record["pptx_sha256"] or sha(__file__) != record["builder_sha256"]:
        raise ValueError("deck/builder binding changed")
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old, zipfile.ZipFile(target) as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError("earlier native part changed: " + name)
        for name in new.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 41:
            raise ValueError("41 slides required")
        for spec in record["charts"]:
            chart = ET.fromstring(new.read(spec["part"]))
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])), data_only=True)
            series = chart.findall(".//c:barChart/c:ser", NS)
            if len(series) != len(spec["series"]):
                raise ValueError("native series count changed")
            for j, node in enumerate(series):
                values = [float(n.text) for n in node.findall("c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
                labels = [n.text for n in node.findall("c:dLbls/c:dLbl/c:tx/c:rich/a:p/a:r/a:t", NS)]
                if values != spec["series"][j][1] or labels != [str(v) for v in spec["labels"][j]]:
                    raise ValueError("native chart values or displayed labels changed")
                if [book.active.cell(i + 2, j + 2).value for i in range(len(values))] != values:
                    raise ValueError("native chart workbook differs")
                if [str(book["Displayed counts"].cell(i + 2, j + 2).value) for i in range(len(values))] != labels:
                    raise ValueError("displayed labels workbook differs")
        for picture in record["pictures"]:
            if hashlib.sha256(new.read(picture["part"])).hexdigest() != picture["sha256"]:
                raise ValueError("original evidence PNG changed")
    return record


def render():
    from build_paired_writing_publication import build_slot
    record = verify(); output = QA / "render"; output.mkdir(exist_ok=True)
    source = QA / "baseline.pptx"; source.write_bytes(baseline("ILM_Figures_editable.pptx"))
    with build_slot():
        command = ["soffice", "-env:UserInstallation=" + (QA / "libreoffice-profile").as_uri(), "--headless", "--convert-to",
            'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            "--outdir", str(output), str(source), str(QA / "draft.pptx")]
        with (QA / "libreoffice.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                if process.wait(timeout=180):
                    raise RuntimeError("owned converter failed")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
    old, current = fitz.open(output / "baseline.pdf"), fitz.open(output / "draft.pdf")
    prefix = fitz.open(stream=baseline("ILM_Figures_review.pdf"), filetype="pdf")
    if (len(old), len(current), len(prefix)) != (40, 41, 40):
        raise ValueError("rendered slide count differs")
    preserved = []
    for i in range(40):
        a, b = old[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    review = fitz.open(); review.insert_pdf(prefix); review.insert_pdf(current, from_page=40, to_page=40)
    review.save(HERE / "ILM_Figures_review.pdf")
    hashes = {}
    for index, name in enumerate(FIGURES, 40):
        figure = fitz.open(); figure.insert_pdf(current, from_page=index, to_page=index)
        figure.save(HERE / "figures" / (name + ".pdf"))
        pixels = []
        for item in figure[0].get_images(full=True):
            with Image.open(io.BytesIO(figure.extract_image(item[0])["image"])) as image:
                pixels.append((image.size, image.convert("RGB").tobytes()))
        for pic in record["pictures"]:
            if pic["slide"] == index + 1:
                with Image.open(ROOT / pic["source"]) as image:
                    if (image.size, image.convert("RGB").tobytes()) not in pixels:
                        raise ValueError("new figure PDF altered measured PNG pixels")
        current[index].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (name + ".png"))
        (HERE / "figures" / (name + ".svg")).write_text(current[index].get_svg_image())
        hashes[name] = sha(HERE / "figures" / (name + ".png"))
    shutil.copy2(QA / "draft.pptx", HERE / "ILM_Figures_editable.pptx")
    (HERE / "evidence/balanced_writing_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")
    rendered = dict(pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"), review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=hashes, displayed_images_pixel_identical=True,
        artifact_tool_available=False, font_substitution="Arial rendered with Liberation Sans",
        powerpoint_app_inspection=False, visual_review_required=True)
    (HERE / "evidence/balanced_writing_figure_render.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print("Rendered slide 41; all 40 prior slide appearances and raw source images preserved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args(); {"generate": generate, "verify": verify, "render": render}[args.stage]()
