#!/usr/bin/env python3
"""Append measured, editable writing figures while preserving 38 prior slides."""
import argparse
import copy
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
BASE = "5c458a2bc2f490769ca94a084293f5961e75b44d"
QA = ROOT / "artifacts/autonomous_writing_publication_20261108"
FIGURES = ("Figure_S24_recognition_loss", "Figure_S25_autonomous_writing")
SOURCES = {name: ROOT / "references/evidence" / folder / "result.json" for name, folder in
    (("recognition", "recognition_guided_writing_result"), ("pilot", "autonomous_writer_result"),
     ("preflight", "autonomous_writer_preflight_result"))}
SOURCES["reader"] = ROOT / "references/evidence/autonomous_writing_loss_result/report.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    data = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
    for key in ("recognition", "pilot"):
        value = data[key]
        if len(value["rows"]) != value["complete_outputs"] or value["complete_outputs"] != 64:
            raise ValueError("complete writing cohort required")
        for name, digest in value["reports"].items():
            if sha(SOURCES[key].parent / name) != digest:
                raise ValueError("source report changed: " + name)
        for row in value["rows"]:
            if sha(SOURCES[key].parent / row["files"]["answer"]) != row["png_sha256"]:
                raise ValueError("actual output PNG changed")
    if data["recognition"]["summary"] != {"train": {"cases": 32, "ocr_exact": 6}, "validation": {"cases": 32, "ocr_exact": 9}}:
        raise ValueError("recognition result differs")
    if data["pilot"]["summary"] != {"train": {"cases": 32, "ocr_exact": 15}, "validation": {"cases": 32, "ocr_exact": 13}}:
        raise ValueError("complete pilot result differs")
    if not data["pilot"]["exploratory_progress_rule_met"] or data["pilot"]["writing_qualification"]:
        raise ValueError("exploratory progress and failed writing boundary changed")
    if not data["preflight"]["first_actual_differentiable_training_forward_native_pixel_exact"]:
        raise ValueError("actual differentiable forward must reproduce native pixels")
    if sum(len(v["steps"]) for v in data["preflight"]["trajectory_proof"].values()) != 100:
        raise ValueError("both full native trajectories required")
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
    if len(slides) != 38:
        raise ValueError("immutable 38-slide source required")
    charts, pictures = [], []
    source_part = "ppt/slides/slide38.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])

    def override(name, kind):
        if name in parts or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("new part collision: " + name)
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def chart(slide, categories, series, labels, maximum, unit, x, y, w, h, decimals=False):
        number = 45 + len(charts)
        name, book = f"ppt/charts/chart{number}.xml", f"ppt/embeddings/autonomous_chart{number}.xlsx"
        node, workbook = multi_chart(parts["ppt/charts/chart44.xml"], categories, series, labels, maximum, unit)
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
        note, part = f"ppt/notesSlides/autonomous_notes{number}.xml", f"ppt/slides/slide{number}.xml"
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
    s.shape("Recognition loss does not ensure faithful visible writing", 45, 24, 1510, 60, 31, True)
    s.shape("a  Independent OCR-exact copies (32 per split)", 45, 105, 760, 40, 22, True)
    chart(s, ["Seen", "Held out"], [("Flow loss", [6, 10]), ("+ Recognition", [6, 9])],
        [[6, 10], [6, 9]], 32, 8, 45, 155, 725, 285)
    s.shape("b  Training-reader target NLL (nats)", 825, 105, 730, 40, 22, True)
    reader = data["reader"]
    values = [round(v, 6) for v in (reader["assisted_training_mean_nll"], reader["summary"]["train"]["mean_autonomous_nll"], reader["summary"]["validation"]["mean_autonomous_nll"])]
    chart(s, ["Assisted\nupdates", "Autonomous\nseen", "Autonomous\nheld out"], [("Reader loss", values)],
        [[f"{x:.3f}" for x in values]], 1.2, .3, 825, 155, 730, 285, True)
    s.shape("255 assisted updates and final-state free outputs are different regimes; this is not a causal decomposition.", 45, 456, 1510, 36, 17)
    s.shape("c  Reader ignores a visible duplicated line", 45, 510, 1510, 38, 22, True)
    row = next(r for r in data["recognition"]["rows"] if r["id"] == "copy/validation/026")
    s.shape("Reference", 45, 557, 430, 30, 18, True)
    s.shape("Actual generated page", 515, 557, 430, 30, 18, True)
    picture(s, SOURCES["recognition"].parent / row["files"]["reference"], "autonomous_loss_reference026", 39, 45, 595, 430)
    picture(s, SOURCES["recognition"].parent / row["files"]["answer"], "autonomous_loss_generated026", 39, 515, 595, 430)
    s.shape("Reader returns the expected string.\nGold NLL: 0.0424; clean reference: 0.000192.\nRecognition does not count every visible repetition.", 1005, 580, 550, 150, 20)
    s.shape("6/32 and 9/32 with the independent evaluator; 14/32 and 19/32 with the training reader.\nThe reader is not an independent judge. No reliable writer or semantic QA is established.", 1005, 746, 550, 115, 17)
    s.shape("Same first 32 image pairs; 512 updates; source backbone and condition fixed. Complete 64-case outputs retained.", 45, 858, 1510, 30, 16)
    append(s, 39, "Figure S24. Matched recognition-loss comparison and complete autonomous reader diagnostic. Independent OCR train/held-out: flow 6/32,10/32; recognition 6/32,9/32. Reader NLL assisted0.0315187, autonomous seen1.0416348,held0.5749592; different noise regimes and checkpoint states,not a causal decomposition. Actual unmodified validation026 reference and duplicated output; reader returns expected string at NLL0.042427. Training-reader exact14/32,19/32 is not independent evaluation. Sources: references/evidence/recognition_guided_writing_result/result.json; references/evidence/autonomous_writing_loss_result/report.json. Models: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B ; https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf . "+json.dumps(charts))

    s = Slide(parts[source_part])
    s.shape("Autonomous image-to-image learning remains below reliable writing", 45, 24, 1510, 60, 30, True)
    s.shape("a  Full sampling trajectory; targets enter only the final loss", 45, 105, 1510, 38, 22, True)
    nodes = [("Input\nimage", 45, 180), ("Frozen image\nencoder", 305, 200),
        ("50 native sampling steps\nTrain rank-16 adapters", 580, 320),
        ("Frozen VAE\ndecoder", 980, 220), ("Answer\nimage", 1290, 240)]
    for words, x, width in nodes:
        s.shape(words, x, 160, width, 85, 19, False, True, "FFF1DA" if x == 580 else "EDF2F6")
    for a, b in zip(nodes, nodes[1:]):
        s.arrow(a[1] + a[2], 202, b[1], 202)
    s.shape("Independent noise + constant condition", 45, 265, 470, 34, 17)
    s.arrow(530, 280, 600, 280); s.arrow(600, 280, 600, 245)
    s.shape("Training only: reader CE + 10 balanced pixel L1", 960, 310, 570, 49, 18, False, True, "FFF1DA")
    s.arrow(1410, 245, 1410, 310)
    s.arrow(960, 335, 830, 335); s.arrow(830, 335, 830, 245)
    s.shape("Full 50-step gradient", 555, 347, 360, 28, 16)
    s.shape("32 updates: 1,135.452 s; 12.057 GiB peak allocated. Runtime: 3.900B transformer + 84.046M VAE.", 45, 391, 1510, 34, 17)
    s.shape("b  OCR-exact copies (n=32)", 45, 450, 460, 38, 22, True)
    chart(s, ["Seen", "Held out"], [("Starting", [6, 9]), ("After 32", [15, 13])],
        [[6, 9], [15, 13]], 32, 8, 45, 500, 450, 260)
    s.shape("Progress rule met; 30/32 writing floor not met.\nNo matched extra-training control.", 45, 786, 460, 77, 17)
    s.shape("c  First two held-out inputs: actual pages", 530, 450, 1025, 38, 22, True)
    for col, label in enumerate(("Target", "Starting", "After pilot")):
        s.shape(label, 530 + col * 340, 499, 330, 30, 17, True)
    for index, row in enumerate(data["pilot"]["rows"][32:34]):
        if row["id"] != f"copy/validation/{index:03}":
            raise ValueError("prospectively fixed first two held-out cases required")
        for col, suffix in enumerate(("reference", "starting", "answer")):
            picture(s, SOURCES["pilot"].parent / row["files"][suffix],
                f"autonomous_case{index}_{suffix}", 40, 530 + col * 340, 535 + index * 165, 330)
    s.shape("Unwanted punctuation remains in Chinese; the English phrase restores its missing letter.", 530, 872, 1025, 26, 15)
    append(s, 40, "Figure S25. Full native autonomous final-image optimization following DRaFT (Clark et al.,ICLR2024,https://arxiv.org/abs/2309.17400),not a new backpropagation algorithm. Pixel/reader targets enter only after generation; no noisy correct answer is supplied to sampler. 32 additional updates from recognition checkpoint; same 64 native inference seeds. OCR exact seen6/32to15/32,held9/32to13/32; canonical-font exploratory improvement,not writing/QA admission. First fixed validation000Chinese and001English all target/starting/finalPNG bytes unchanged. Inherited pretraining, full runtime3,899,903,488transformer+84,046,115VAE;24,358,912trainable;560,528,640readertrainingonly. 32draws22distinctseeninputs. Extra training,precision correction and loss terms not isolated. Source: references/evidence/autonomous_writer_result/result.json. "+json.dumps(charts[-1]))
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relationships)):
        parts[name] = xml(node)
    target = QA / "draft.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in parts.items():
            z.writestr(name, raw)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=38,
        slides=40, original_chart_count=44, charts=charts, pictures=pictures,
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
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 40:
            raise ValueError("40 slides required")
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
    if (len(old), len(current), len(prefix)) != (38, 40, 38):
        raise ValueError("rendered slide count differs")
    preserved = []
    for i in range(38):
        a, b = old[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    review = fitz.open(); review.insert_pdf(prefix); review.insert_pdf(current, from_page=38, to_page=39)
    review.save(HERE / "ILM_Figures_review.pdf")
    hashes = {}
    for index, name in enumerate(FIGURES, 38):
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
    (HERE / "evidence/autonomous_writing_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")
    rendered = dict(pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"), review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=hashes, displayed_images_pixel_identical=True,
        artifact_tool_available=False, font_substitution="Arial rendered with Liberation Sans",
        powerpoint_app_inspection=False, visual_review_required=True)
    (HERE / "evidence/autonomous_writing_figure_render.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print("Rendered slides 39/40; all 38 earlier slide appearances and evidence PNGs preserved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args(); {"generate": generate, "verify": verify, "render": render}[args.stage]()
