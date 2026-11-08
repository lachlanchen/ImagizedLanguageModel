#!/usr/bin/env python3
"""Append measured ink/QA figures and preserve all 41 prior native slides."""
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
BASE = "b5501de2e925ec86c25f726b1261bb7f65da5825"
QA = ROOT / "artifacts/ink_mass_publication_20261108"
FIGURES = ("Figure_S27_ink_mass", "Figure_S28_full_page_preflight")
SOURCES = {name: ROOT / "references/evidence" / folder for name, folder in (
    ("comparison", "ink_mass_comparison_result/result.json"),
    ("audit", "writing_physical_loss_audit.json"),
    ("preflight", "native_paired_language_preflight_result/result.json"))}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    data = {k: json.loads(path.read_text()) for k,path in SOURCES.items()}
    r, a, q = data["comparison"], data["audit"], data["preflight"]
    if (r["complete_cases"] != 64 or r["complete_final_neural_outputs"] != 128
            or not r["all_final_pngs_equal_neural_tensors"] or not r["optimizer_steps_all_192"]):
        raise ValueError("complete authenticated matched comparison required")
    for name,digest in r["reports"].items():
        if sha(SOURCES["comparison"].parent/name) != digest: raise ValueError("comparison report changed")
    for row in r["rows"]:
        for key,digest in row["files_sha256"].items():
            if sha(SOURCES["comparison"].parent/row["files"][key]) != digest: raise ValueError("raw comparison image changed")
    for arm,counts in (("control",(16,16)),("ink_mass",(18,19))):
        for split,count in zip(("train","validation"),counts):
            observed=sum(row["observations"][arm]["ocr_exact"] for row in r["rows"] if row["split"]==split)
            if r["summary"][arm][split] != dict(cases=32,ocr_exact=observed) or count != observed:
                raise ValueError("complete matched counts differ")
    reviewed=json.loads((SOURCES["comparison"].parent/"raw_review.json").read_text())
    if not reviewed["complete_cohort_reviewed"] or reviewed["result_sha256"] != sha(SOURCES["comparison"]):
        raise ValueError("complete bound raw review required")
    if a["complete_cases"] != 64 or a["optimizer_updates"] or a["neural_model_calls"]:
        raise ValueError("reference corruption diagnostic must remain CPU-only")
    if (q["normal_cases"] != 4 or q["normal_answers_ocr_exact"] != 0 or len(q["rows"]) != 8
            or not q["raw_review_completed"] or q["preflight_weights_used_for_inference"]):
        raise ValueError("complete fixed failed QA preflight required")
    for name,digest in q["reports"].items():
        if sha(SOURCES["preflight"].parent/name) != digest: raise ValueError("preflight report changed")
    for row in q["rows"]:
        for item in row["files"].values():
            if sha(SOURCES["preflight"].parent/item["file"]) != item["sha256"]: raise ValueError("actual QA image changed")
    for row in q["codec"]:
        for key in ("original","decoded"):
            if sha(SOURCES["preflight"].parent/row[key]["file"]) != row[key]["sha256"]: raise ValueError("codec image changed")
    if any(r[k] for k in ("writing_qualification","semantic_QA_evaluated","general_ilm_goal_complete")) or q["language_quality_qualified"]:
        raise ValueError("failed language/writing boundary changed")
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
    if len(slides) != 41:
        raise ValueError("immutable 41-slide source required")
    charts, pictures = [], []
    source_part = "ppt/slides/slide41.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])

    def override(name, kind):
        if name in parts or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("new part collision: " + name)
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def chart(slide, categories, series, labels, maximum, unit, x, y, w, h, decimals=False):
        number = 50 + len(charts)
        name, book = f"ppt/charts/chart{number}.xml", f"ppt/embeddings/ink_mass_chart{number}.xlsx"
        node, workbook = multi_chart(parts["ppt/charts/chart49.xml"], categories, series, labels, maximum, unit)
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

    def picture(slide, source, name, owner, x, y, width, crop=None):
        stable = HERE / "figures" / (name + ".png")
        if crop is None: stable.write_bytes(source.read_bytes())
        else:
            with Image.open(source) as original: original.crop(crop).save(stable)
        part = "ppt/media/" + stable.name
        if part in parts: raise ValueError("new image collision")
        parts[part] = stable.read_bytes()
        with Image.open(stable) as im:
            slide.picture(slide.relation("image", "../media/" + stable.name), name,
                x, y, width, width * im.height / im.width)
        pictures.append(dict(part=part, source=str(stable.relative_to(ROOT)), sha256=sha(stable), slide=owner,
            original_source=str(source.relative_to(ROOT)),original_sha256=sha(source),crop_box=crop))

    def append(slide, number, description):
        slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
        note, part = f"ppt/notesSlides/ink_mass_notes{number}.xml", f"ppt/slides/slide{number}.xml"
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

    r,a,qdata = data["comparison"],data["audit"],data["preflight"]
    s = Slide(parts[source_part])
    s.shape("Fixed target ink mass improves some copies",45,24,1510,60,31,True)
    s.shape("a  Duplicate / deletion physical cost (%)",45,105,745,40,22,True)
    ratios=[100*a["summary"][key]["duplicate_to_blank_ratio_median"] for key in ("balanced_l1","ink_mass_kl")]
    chart(s,["Balanced L1","Ink-mass KL"],[("Median on 64 reference pages",ratios)],
        [[f"{v:.2f}%" for v in ratios]],100,25,45,155,725,280)
    s.shape("b  Complete OCR-exact copies (%)",825,105,730,40,22,True)
    counts=[("Control",[16,16,4,12]),("Ink-mass",[18,19,6,13])]
    denoms=[32,32,16,16]
    chart(s,["Seen (32)","Held (32)","Held ZH (16)","Held EN (16)"],
        [(name,[100*v/n for v,n in zip(values,denoms)]) for name,values in counts],
        [[f"{v}/{n}" for v,n in zip(values,denoms)] for _,values in counts],100,25,825,155,730,280)
    s.shape("One matched permutation: 32 updates\nSame weights, Adam state, inputs and noise\nOnly physical loss changes; reader CE stays",45,490,450,110,18)
    s.shape("KL scale: 0.403568; temperature: 0.05\nFixed target ink mass; unclipped RGB\nTraining peaks: 12.032 / 12.033 GiB",45,620,450,105,17)
    s.shape("Exploratory rule passes; writing floor fails.\nCPU corruptions are not model outputs.\nNo semantic QA in these two arms.",45,750,450,105,17)
    s.shape("c  Fixed first two held-out cases: complete actual pages",530,470,1025,38,21,True)
    for col,label in enumerate(("Target","Control","Ink-mass")):
        s.shape(label,530+col*340,515,330,30,17,True)
    for i,row in enumerate(r["rows"][32:34]):
        if row["id"]!=f"copy/validation/{i:03}":raise ValueError("fixed copy cases differ")
        for col,key in enumerate(("reference","control","ink_mass")):
            picture(s,SOURCES["comparison"].parent/row["files"][key],f"ink_mass_case{i}_{key}",42,
                530+col*340,550+i*160,320)
    s.shape("Single seed; reused development; inherited 3.900B transformer + 84.046M VAE; all 128 final outputs retained.",530,873,1025,25,14)
    append(s,42,"Figure S27. CPU diagnostic corruptions and complete matched model assessment. "
        "Control uses reader CE +10balanced L1; treatment replaces only physical term with0.403568ink-mass Bernoulli KL. "
        "Both32updates from identical optimizer160to192; identical schedules/seeds. "
        "Full raw first2heldout cases fixed before evaluation; all128outputs retained. "
        "All exploratory gates pass but reliable writing remains unqualified; no QA trained/evaluated. "+json.dumps(r["summary"]))

    s=Slide(parts[source_part])
    s.shape("Full pages survive encoding; answering remains untrained",45,24,1510,60,30,True)
    s.shape("a  Complete first input and native reconstruction",45,105,980,40,22,True)
    s.shape("Original 1024 x 1024 input",45,158,450,25,17,True)
    s.shape("Frozen native VAE reconstruction",530,158,450,25,17,True)
    codec=qdata["codec"][0]
    for col,key in enumerate(("original","decoded")):
        picture(s,SOURCES["preflight"].parent/codec[key]["file"],"ink_mass_qa_"+key,43,45+485*col,190,450)
    s.shape("Four full pages; features: 128 x 64 x 64\nForeground F1: 97.834-98.180% (physical reconstruction)",45,654,980,58,17)
    s.shape("First input question region (display inset only)",45,716,980,28,17,True)
    picture(s,SOURCES["preflight"].parent/codec["original"]["file"],"ink_mass_qa_question_inset",43,45,748,940,crop=(0,552,1024,648))
    s.shape("Two real QA updates discarded: 12.203 GiB. Fresh inference reloads the fixed earlier checkpoint.\n0/4 normal answers; eight raw normal/control pages retained. Small preflight, not language learning.",45,846,980,51,15)
    s.shape("b  First two natural questions\nReference versus fixed baseline",1070,105,490,55,21,True)
    normal=[row for row in qdata["rows"] if row["variant"]=="normal"]
    if [row["id"] for row in normal[:2]] != ["TRAIN_3802_QUERY_0","TRAIN_3802_QUERY_1"]:
        raise ValueError("fixed first QA pair differs")
    for i,row in enumerate(normal[:2]):
        y=150+365*i
        s.shape(f"Q{i+1} reference answer (whitespace cropped)",1070,y,490,22,16,True)
        picture(s,SOURCES["preflight"].parent/row["files"]["reference"]["file"],f"ink_mass_qa{i}_reference_crop",43,
            1070,y+22,500,crop=(0,24,512,80))
        s.shape("Unrepaired native output: complete 512 x 256 page",1070,y+81,500,22,14)
        picture(s,SOURCES["preflight"].parent/row["files"]["answer"]["file"],f"ink_mass_qa{i}_answer",43,1070,y+106,500)
    append(s,43,"Figure S28. Full first source page and native VAE reconstruction; actual first two unoptimized QA outputs. "
        "Both model outputs shown complete. Canonical reference whitespace cropped at(0,24,512,80); original input question inset(0,552,1024,648). "
        "Model always receives complete1024square inputs. Four codec pages and eight standalone native outputs authenticated. "
        "Two discarded updates are execution checks; inference restoresfc031checkpoint with noQAoptimization, no reader/font/answer inputs. "
        "No semantic-learning or full-benchmark claim.")
    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relationships)):
        parts[name] = xml(node)
    target = QA / "draft.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in parts.items():
            z.writestr(name, raw)
    record = dict(baseline_commit=BASE, pptx_sha256=sha(target), preserved_native_slides=41,
        slides=43, original_chart_count=49, charts=charts, pictures=pictures,
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
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst", NS)) != 43:
            raise ValueError("43 slides required")
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
            original=ROOT/picture["original_source"]
            if sha(original)!=picture["original_sha256"]:raise ValueError("original source evidence changed")
            with Image.open(original) as image,Image.open(ROOT/picture["source"]) as shown:
                desired=image.crop(picture["crop_box"]) if picture["crop_box"] else image
                if desired.convert("RGB").size!=shown.convert("RGB").size or desired.convert("RGB").tobytes()!=shown.convert("RGB").tobytes():
                    raise ValueError("displayed crop differs from declared source rectangle")
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
    if (len(old), len(current), len(prefix)) != (41, 43, 41):
        raise ValueError("rendered slide count differs")
    preserved = []
    for i in range(41):
        a, b = old[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    review = fitz.open(); review.insert_pdf(prefix); review.insert_pdf(current, from_page=41, to_page=42)
    review.save(HERE / "ILM_Figures_review.pdf")
    hashes = {}
    for index, name in enumerate(FIGURES, 41):
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
    (HERE / "evidence/ink_mass_writing_deck_build.json").write_text(json.dumps(record, indent=2) + "\n")
    rendered = dict(pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"), review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=hashes, displayed_images_pixel_identical=True,
        artifact_tool_available=False, font_substitution="Arial rendered with Liberation Sans",
        powerpoint_app_inspection=False, visual_review_required=True)
    (HERE / "evidence/ink_mass_writing_figure_render.json").write_text(json.dumps(rendered, indent=2) + "\n")
    print("Rendered slides 42-43; all 41 prior slide appearances and raw source images preserved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("generate", "verify", "render"), required=True)
    args = p.parse_args(); {"generate": generate, "verify": verify, "render": render}[args.stage]()
