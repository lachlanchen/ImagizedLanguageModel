#!/usr/bin/env python3
"""Append measured visual-state and canvas controls, preserving 34 native slides."""
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
BASE = "585ad1c"
QA = ROOT / "artifacts/native_state_alignment_publication_20261107"
FIGURE = "Figure_S20_state_alignment"
RESULT = ROOT / "references/evidence/native_state_alignment_result/result.json"
SOURCE = ROOT / "references/evidence/native_canvas_control_result/result.json"
FIT = ROOT / "references/evidence/native_state_fit_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    result, canvas, fit = [json.loads(path.read_text()) for path in (RESULT, SOURCE, FIT)]
    for path, value in ((RESULT, result), (SOURCE, canvas), (FIT, fit)):
        if value["general_ilm_goal_complete"]:
            raise ValueError("no independent ILM claim admitted")
        for name, digest in value["files_sha256"].items():
            if sha(path.parent / name) != digest:
                raise ValueError("changed measured evidence: " + name)
    values = [[result["before"]["en"]["summary"]["normal_exact_stopped"], result["after"]["en"]["summary"]["normal_exact_stopped"],
        result["before"]["zh"]["normal"]["exact_stopped"], result["after"]["zh"]["normal"]["exact_stopped"]],
        [canvas["arms"][arm]["copy"]["languages"]["zh"]["exact_stopped"] for arm in ("legacy", "square", "tight")]]
    if values != [[11,16,0,0],[13,0,0]] or result["actual_updates"] != 2560 or canvas["actual_calls"] != 480:
        raise ValueError("completed primary values changed")
    if fit["groups"]["learner_image/copy/zh"]["training_target_exact_stopped"] != 0:
        raise ValueError("seen Chinese fit result changed")
    return result, canvas, values


def generate():
    result, source, values = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 34:
        raise ValueError("current 34-slide baseline required")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("native part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    specs = []
    settings = [(["English\nsource", "English\nlearned", "Chinese\nsource", "Chinese\nlearned"], 32, 8),
                (["Original\nstrip", "White-padded\nsquare", "Tight\ncrop"], 16, 4)]
    for number, numbers, (categories, maximum, unit) in zip((40, 41), values, settings):
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
        book = f"ppt/embeddings/state_alignment_chart{number}.xlsx"
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        specs.append(dict(part=name, workbook=book, values=numbers, categories=categories))
    source_part = "ppt/slides/slide34.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Visual-state learning and canvas dependence", 45, 24, 1510, 58, 31, True)
    slide.shape("a  Learned correction to a frozen native reader", 45, 92, 1500, 42, 23, True)
    for words, x, y, w, h, fill in (
        ("Complete prompt\n1024-square pixels",45,157,200,105,"EDF2F6"),
        ("432 view → frozen encoder\nCoarse features C",300,138,350,62,"EDF2F6"),
        ("1024 view → 9 fine windows\nLearned adapters → pooled F",300,226,350,68,"FFF1DA"),
        ("C + G(F − C)\nG starts exactly at zero",715,179,350,76,"FFF1DA"),
        ("Frozen language body H\nAuxiliary text head U",1130,179,390,76,"EDF2F6")):
        slide.shape(words,x,y,w,h,19,False,True,fill)
    for coordinates in ((245,183,300,170),(245,237,300,260),(650,170,715,207),(650,260,715,237),(1065,217,1130,217)):
        slide.arrow(*coordinates)
    slide.shape("Training only: original-text pre-answer states at seven layers; answer CE + normalized state MSE",45,306,1500,36,18)
    slide.shape("b  Held-out QA: exact stopped answers / 32",45,361,725,40,22,True)
    slide.chart(slide.relation("chart","../charts/chart40.xml"),45,407,700,224)
    slide.shape("Copying: English 7/16 → 16/16; Chinese 0/16 → 0/16",45,640,740,36,19)
    slide.shape("c  Separate fixed dense source: Chinese copies / 16",835,361,720,40,22,True)
    slide.chart(slide.relation("chart","../charts/chart41.xml"),835,407,700,224)
    slide.shape("English copies: 16/16 → 7/16 → 0/16; QA stays 0/32",835,640,720,36,19)
    source_image = ROOT / "artifacts/image_edit_learning/strip_pages/0000_input.png"
    full_image = ROOT / "artifacts/image_edit_learning/pages_v2/0000_input.png"
    with Image.open(source_image) as short, Image.open(full_image) as full:
        restored = Image.new("RGB",full.size,"white"); restored.paste(short,(0,0))
        if restored.tobytes() != full.convert("RGB").tobytes():
            raise ValueError("display strip does not preserve the complete seen request")
    stable = HERE / "figures/state_alignment_seen_request.png"
    stable.write_bytes(source_image.read_bytes())
    part = "ppt/media/state_alignment_seen_request.png"
    parts[part] = stable.read_bytes()
    slide.shape("d  Seen copying request; white display padding omitted",45,697,760,35,19,True)
    slide.picture(slide.relation("image","../media/"+Path(part).name),"Complete seen copying request",45,738,640,160)
    pictures = [dict(part=part,source=str(stable.relative_to(ROOT)),sha256=sha(stable))]
    slide.shape("Raw auxiliary text readout (not an answer image)",835,697,720,35,19,True)
    slide.shape("Requested: 花多地地山\nLearner: 花多梅树山\nAll 8 tested seen Chinese strings change characters",835,742,720,106,21)
    slide.shape("Four of six reader gates fail. No independent image-answer model is qualified.",835,860,720,35,17)
    for prop in slide.root.findall(".//a:rPr",NS):
        element("a:ea",prop,typeface="Noto Sans CJK SC")
    slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/state_alignment_notes35.xml"
    slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
    part = "ppt/slides/slide35.xml"
    override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S20. Completed state-alignment learner: 2560 updates, 640 examples, 10,849,280 trainable parameters, complete adapted model 2,841,626,336 plus codec 126,892,531. "
        "English QA 11 to16/32; Chinese0/32; copying English7 to16/16 and Chinese0/16. First fixed seen Chinese request shown from original strip, restoring its square white exterior is pixel exact. "
        "The learner received the full square; this display crop preserves all requested characters. Raw readout 花多梅树山 changes requested 花多地地山. "
        "Separate fixed dense checkpoint canvas control: copying Chinese13/16 on1024x256 strips,0/16 after white square padding,0/16 tight; English16,7,0/16. All480 calls retained,160 legacy IDs replayexact. "
        "VAE features/positions/context change with geometry despite preserved pixels. Four of six state-reader gates fail. No continuous image answers or general ILM qualification. "
        "Sources: references/evidence/native_state_alignment_result, native_state_fit_result, native_canvas_control_result. CMRC2018/SQuAD1.1 CC-BY-SA4.0; Show-o2 NUS Show Lab Apache2.0. " + json.dumps(dict(chart_values=values),ensure_ascii=False))
    for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            for i, node in enumerate(shape.findall(".//a:t", NS)):
                node.text = description if i == 0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target", "../slides/slide35.xml")
    override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note], parts[relpath(note)] = xml(notes), xml(rels)
    used = {r.get("Id") for r in relations}
    rid = next(f"rId{i}" for i in range(1, 1000) if f"rId{i}" not in used)
    element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target="slides/slide35.xml")
    element("p:sldId", slides, id=max(int(s.get("id")) for s in slides) + 1).set(q("r:id"), rid)
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relations)):
        parts[name] = xml(node)
    QA.mkdir(parents=True,exist_ok=True)
    target = QA / "draft.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=34, slides=35,
        charts=specs, pictures=pictures, builder_sha256=sha(__file__), source_result_sha256=sha(RESULT),
        source_control_sha256=sha(SOURCE), fit_result_sha256=sha(FIT), artifact_tool_available=False, general_ilm_goal_complete=False)
    (QA / "deck_build.json").write_text(json.dumps(record, indent=2) + "\n")


def verify():
    _, _, values = evidence()
    record_path = QA / "deck_build.json" if (QA / "deck_build.json").exists() else HERE / "evidence/native_state_alignment_deck_build.json"
    record = json.loads(record_path.read_text())
    target = QA / "draft.pptx" if (QA / "draft.pptx").exists() else HERE / "ILM_Figures_editable.pptx"
    if record["pptx_sha256"] != sha(target) or record["builder_sha256"] != sha(__file__) or record["source_result_sha256"] != sha(RESULT) or record["source_control_sha256"] != sha(SOURCE) or record["fit_result_sha256"] != sha(FIT):
        raise ValueError("source/deck binding changed")
    mutable = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old, zipfile.ZipFile(target) as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError("earlier native part changed: " + name)
        for name in new.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 35:
            raise ValueError("current deck must contain 35 slides")
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
    target = QA / "draft.pptx" if (QA / "draft.pptx").exists() else HERE / "ILM_Figures_editable.pptx"
    QA.mkdir(parents=True, exist_ok=True)
    with build_slot():
        source = QA / "baseline.pptx"
        source.write_bytes(baseline("ILM_Figures_editable.pptx"))
        output = QA / "render"
        output.mkdir(exist_ok=True)
        command = ["soffice", "-env:UserInstallation=" + (QA / "libreoffice-profile").as_uri(),
            "--headless", "--convert-to", 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            "--outdir", str(output), str(source), str(target)]
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
    current = fitz.open(output / (target.stem + ".pdf"))
    prefix = fitz.open(stream=baseline("ILM_Figures_review.pdf"), filetype="pdf")
    if (len(prior), len(current), len(prefix)) != (34, 35, 34):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(34):
        a, b = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=34, to_page=34)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=34, to_page=34)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    embedded = [Image.open(io.BytesIO(figure.extract_image(item[0])["image"])).convert("RGB")
                for item in figure[0].get_images(full=True)]
    for picture in record["pictures"]:
        with Image.open(ROOT / picture["source"]) as image:
            image = image.convert("RGB")
            if not any(candidate.size == image.size and candidate.tobytes() == image.tobytes() for candidate in embedded):
                raise ValueError("standalone PDF changed input-page pixels")
    current[34].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[34].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
        new_slide=35, visual_review_required=True, artifact_tool_available=False)
    result["displayed_input_image_pixel_identical"] = True
    import shutil
    if target != HERE / "ILM_Figures_editable.pptx":
        shutil.copy2(target,HERE / "ILM_Figures_editable.pptx")
    (HERE / "evidence/native_state_alignment_deck_build.json").write_text(json.dumps(record,indent=2)+"\n")
    (HERE / "evidence/native_state_alignment_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide 35 and preserved all 34 earlier native slide appearances.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args()
    {"generate": generate, "verify": verify, "render": render}[args.stage]()
