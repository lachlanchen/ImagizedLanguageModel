#!/usr/bin/env python3
"""Append an editable, source-bound figure for the completed matched readout failure."""
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

import fitz
import openpyxl
from PIL import Image

from generate_assignment_editable_figures import (
    NS, Slide, chart_part, element, q, relpath, resolve, xml,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "1f182c4"
QA = ROOT/"artifacts/readout_failure_figures_20261107"
FIGURE = "Figure_S15_readout_failure"
RESULT = ROOT/"references/evidence/adaptive_readout_v81_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE+":publication/ilm-nature/"+name], cwd=ROOT)


def evidence():
    value = json.loads(RESULT.read_text())
    run = ROOT/"artifacts/adaptive_readout_v81eval_20261107"
    verified = value["verification"]
    if (sha(run/"report.json") != value["report_sha256"]
            or sha(run/"verification.json") != value["verification_sha256"]
            or not verified["all_64_outputs_independently_scored"]
            or verified["training_fit_repair_pass"] or value["language_quality_claim"]):
        raise ValueError("completed rejected readout evidence required")
    for name,digest in value["raw_png_sha256"].items():
        if sha(RESULT.parent/name) != digest:raise ValueError("raw readout PNG changed")
    summary = verified["summaries"]
    return value, summary, [100*summary[a]["free_f1"] for a in ("control","adapted")], [
        100*summary[a]["assisted_f1"] for a in ("control","adapted")]


def generate():
    value,summaries,free,history = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name:archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst",NS)
    size = presentation.find("p:sldSz",NS)
    if len(slides) != 29 or (int(size.get('cx')),int(size.get('cy'))) != (15240000,8572500):
        raise ValueError("preserve the existing 29-slide template and geometry")
    def override(name,kind):
        if any(n.get("PartName") == "/"+name for n in types):
            raise ValueError("new native part collides with a preserved part")
        element("ct:Override",types,PartName="/"+name,ContentType=kind)
    specifications = []
    for number,categories,values in (
            (28,["Control","Adapted"],free),
            (29,["Control","Adapted"],history)):
        chart,workbook = chart_part(parts["ppt/charts/chart12.xml"],categories,values,
            "Foreground F1 (%)",0,100,"296486")
        for parent in chart.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        for node in chart.findall(".//a:defRPr",NS):
            node.set("sz","1900")
        for axis in chart.findall(".//c:valAx",NS):
            unit = axis.find("c:majorUnit",NS)
            if unit is None:unit = element("c:majorUnit",axis)
            unit.set("val","25")
            formatting = axis.find("c:numFmt",NS)
            if formatting is None:
                formatting = element("c:numFmt",formatCode="0",sourceLinked="0")
                following = {q("c:"+n) for n in ("majorTickMark","minorTickMark","tickLblPos","spPr","txPr","crossAx")}
                index = next(i for i,n in enumerate(axis) if n.tag in following)
                axis.insert(index,formatting)
            formatting.set("formatCode","0");formatting.set("sourceLinked","0")
        for node in chart.findall(".//c:numFmt",NS):
            node.set("formatCode","0");node.set("sourceLinked","0")
        bar = chart.find(".//c:barChart",NS)
        for node in bar.findall("c:dLbls",NS):bar.remove(node)
        labels = element("c:dLbls")
        element("c:numFmt",labels,formatCode="0.000",sourceLinked="0")
        element("c:dLblPos",labels,val="outEnd")
        for tag,show in (("showLegendKey",0),("showVal",1),("showCatName",0),
                         ("showSerName",0),("showPercent",0),("showBubbleSize",0)):
            element("c:"+tag,labels,val=show)
        bar.insert(list(bar).index(bar.find("c:axId",NS)),labels)
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/readout_failure_chart{number}.xlsx"
        if name in original or book in original:
            raise ValueError("new chart overwrites preserved evidence")
        chart.find("c:externalData",NS).set(q("r:id"),"rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship",rels,Id="rId1",Type=NS["r"]+"/package",Target="../embeddings/"+Path(book).name)
        parts[name],parts[relpath(name)],parts[book] = xml(chart),xml(rels),workbook
        override(name,"application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        specifications.append(dict(part=name,workbook=book,categories=categories,values=values))
    source_part = "ppt/slides/slide29.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Trainable readout leaves autonomous writing inaccurate",45,26,1510,58,31,True)
    slide.shape("a  Autonomous writing / F1 (%)",65,110,730,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart28.xml"),45,154,730,290)
    slide.shape("b  Correct preceding history / F1 (%)",845,110,710,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart29.xml"),825,154,730,290)
    slide.shape("Exact complete images: 0/32 in both arms",80,447,700,38,19)
    slide.shape("Exact complete images: 0/32 in both arms",865,447,700,38,19)
    slide.shape("c  First four fixed questions / complete raw images",65,500,1460,45,23,True)
    slide.shape("Reference",220,550,600,36,20,True)
    slide.shape("Adapted autonomous output",900,550,620,36,20,True)
    pictures = []
    for i in range(4):
        y = 598+66*i
        slide.shape(f"Case {i:02}",65,y+10,140,40,20)
        for x,kind in ((225,"reference"),(900,"answer")):
            relative = f"references/evidence/adaptive_readout_v81_result/raw/adapted_{i:02}_{kind}.png"
            source = ROOT/relative
            expected = value["raw_png_sha256"]["raw/"+source.name]
            if sha(source) != expected:raise ValueError("raw diagnostic picture changed")
            with Image.open(source) as image:w,h = image.size
            if h != 32 or w > 320:raise ValueError("first four complete images no longer fit")
            part = f"ppt/media/readout_failure_{i:02}_{kind}.png"
            if part in original:raise ValueError("new image part collides")
            parts[part] = source.read_bytes()
            slide.picture(slide.relation("image","../media/"+Path(part).name),
                f"Unmodified case {i:02} {kind}",x,y,w*2,h*2)
            pictures.append(dict(part=part,source=relative,sha256=expected))
    slide.shape("32 training questions; 2,000 updates/arm. Supplied-code F1: 99.9953% (28/32 exact). Language remains unqualified.",45,867,1510,32,17)
    layout = next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout"))
    slide.relation("slideLayout",layout)
    note = "ppt/notesSlides/readout_failure_notes30.xml"
    slide.relation("notesSlide","../notesSlides/"+Path(note).name)
    part = "ppt/slides/slide30.xml"
    parts[part],parts[relpath(part)] = xml(slide.root),xml(slide.rels)
    override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    source_note = next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S15. Source: references/evidence/adaptive_readout_v81_result/result.json. "
        "Fixed32 training questions,16 same-passage pairs,2000 matched updates/arm,B4,8000 draws. "
        "Free F1 control71.1767465%,adapted72.7475410%;correct incremental-history F1 "
        "77.3722725%,79.6293432%;all exact0/32. Supplied codes99.9953159%,28/32. "
        "53,096,961 adapted trainables;7,583,225,971 complete parameters include external pretraining. "
        "The original exact floating-stop comparison failed;separate evaluation performs no optimization "
        "and replays33 prior autonomous streams exactly. Three of four original training-fit gates fail. "
        "The first four fixed cases are shown without selection by outcome;rawPNGbytes unchanged at2x placement. "
        "All32cases/rawimages retained in source receipt. No semantic accuracy or general language success claimed.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp",NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
        if ph is not None and ph.get("type") == "body":
            for i,node in enumerate(shape.findall(".//a:t",NS)):node.text = description if i==0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):node.set("Target","../slides/slide30.xml")
    parts[note],parts[relpath(note)] = xml(notes),xml(rels)
    override(note,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    used = {r.get("Id") for r in relations};i = 1
    while f"rId{i}" in used:i += 1
    rid = f"rId{i}"
    element("pkg:Relationship",relations,Id=rid,Type=NS["r"]+"/slide",Target="slides/slide30.xml")
    element("p:sldId",slides,id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"),rid)
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),
                      ("ppt/_rels/presentation.xml.rels",relations)):
        parts[name] = xml(node)
    target = HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in parts.items():archive.writestr(name,data)
    receipt = dict(baseline_commit=BASE,pptx_sha256=sha(target),slides=30,preserved_native_slides=29,
        charts=specifications,pictures=pictures,builder_sha256=sha(Path(__file__)),
        sources_sha256={"matched_result":sha(RESULT)},
        artifact_tool_available=False,language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/readout_failure_deck_build.json").write_text(json.dumps(receipt,indent=2)+"\n")


def verify():
    _,_,free,history = evidence()
    value = json.loads((HERE/"evidence/readout_failure_deck_build.json").read_text())
    if value["pptx_sha256"] != sha(HERE/"ILM_Figures_editable.pptx") or value["builder_sha256"] != sha(Path(__file__)):
        raise ValueError("figure source/deck changed")
    mutable = {"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):raise ValueError("preserved native part changed: "+name)
        for name in new.namelist():
            if name.endswith((".xml",".rels")):ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS)) != 30:
            raise ValueError("wrong slide count")
        for spec,expected in zip(value["charts"],(free,history)):
            chart = ET.fromstring(new.read(spec["part"]))
            numbers = [float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
            book = openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])),data_only=True)
            actual = [book.active.cell(i+2,2).value for i in range(len(expected))]
            labels = [book.active.cell(i+2,1).value for i in range(len(expected))]
            if numbers != expected or any(abs(a-b)>1e-12 for a,b in zip(actual,expected)) or labels != spec["categories"]:
                raise ValueError("native chart/workbook source mismatch")
        for picture in value["pictures"]:
            if hashlib.sha256(new.read(picture["part"])).hexdigest() != picture["sha256"] or sha(ROOT/picture["source"]) != picture["sha256"]:
                raise ValueError("embedded physical writing image changed")
    return value


def render():
    build = verify();QA.mkdir(parents=True,exist_ok=True)
    memory = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
    if memory['MemAvailable']<24*1024**2 or (memory['SwapTotal'] and 1-memory['SwapFree']/memory['SwapTotal']>.75):
        raise RuntimeError("shared-host memory policy prevents conversion")
    reference = QA/"baseline.pptx";reference.write_bytes(baseline("ILM_Figures_editable.pptx"))
    output = QA/"render";output.mkdir(exist_ok=True)
    command = ["soffice","-env:UserInstallation="+(QA/"libreoffice-profile").as_uri(),
        "--headless","--convert-to","pdf","--outdir",str(output),str(reference),str(HERE/"ILM_Figures_editable.pptx")]
    with (QA/"libreoffice.log").open('w') as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if process.wait(timeout=180):raise RuntimeError("owned figure converter failed")
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    prefix = fitz.open(stream=baseline("ILM_Figures_review.pdf"),filetype="pdf")
    prior = fitz.open(output/"baseline.pdf");new = fitz.open(output/"ILM_Figures_editable.pdf")
    if len(prefix) != 29 or len(prior) != 29 or len(new) != 30:raise ValueError("incorrect rendered slide count")
    for i in range(29):
        if prior[i].get_pixmap(alpha=False).samples != new[i].get_pixmap(alpha=False).samples:
            raise ValueError("fresh appearance changed for old slide")
    canonical = fitz.open();canonical.insert_pdf(prefix);canonical.insert_pdf(new,from_page=29,to_page=29)
    canonical.save(HERE/"ILM_Figures_review.pdf");canonical.close()
    figure = fitz.open();figure.insert_pdf(new,from_page=29,to_page=29)
    figure.save(HERE/"figures"/(FIGURE+".pdf"));figure.close()
    new[29].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures"/(FIGURE+".png"))
    (HERE/"figures"/(FIGURE+".svg")).write_text(new[29].get_svg_image(text_as_path=False))
    final = fitz.open(HERE/"ILM_Figures_review.pdf");preserved = []
    for i in range(29):
        pix = prefix[i].get_pixmap(alpha=False)
        if pix.samples != final[i].get_pixmap(alpha=False).samples:raise ValueError("canonical prefix appearance changed")
        preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(pix.samples).hexdigest()))
    for document in (prefix,prior,new,final):document.close()
    receipt = dict(pptx_sha256=build['pptx_sha256'],review_pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved,fresh_same_renderer_prefix_equal=True,
        figure_sha256={suffix:sha(HERE/"figures"/(FIGURE+"."+suffix)) for suffix in ('pdf','png','svg')},
        renderer_sha256=sha(Path(__file__)),headless_process_exited=True,visual_review_required=True,
        language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/readout_failure_figure_render.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('generate','verify','render'),required=True)
    args = parser.parse_args();dict(generate=generate,verify=verify,render=render)[args.stage]()
