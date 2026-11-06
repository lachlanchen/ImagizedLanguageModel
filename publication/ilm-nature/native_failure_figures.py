#!/usr/bin/env python3
"""Append an editable, source-bound figure for the completed native failure."""
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
BASE = "b4855be"
QA = ROOT/"artifacts/native_failure_figures_20261106"
FIGURE = "Figure_S14_native_failure"
RECEIPTS = ("native_chinese_v80_result", "native_corpus_fit_v80_result")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE+":publication/ilm-nature/"+name], cwd=ROOT)


def evidence():
    main, diagnosis = [json.loads((ROOT/"references/evidence"/(name+".json")).read_text()) for name in RECEIPTS]
    run = ROOT/main["source_run"]
    for name,digest in main["source_files_sha256"].items():
        if sha(run/name) != digest:
            raise ValueError("completed native audit changed")
    raw = ROOT/diagnosis["source_run"]
    if (sha(raw/"report.json") != diagnosis["source_report_sha256"]
            or sha(raw/"verification.json") != diagnosis["source_verification_sha256"]
            or main["audit"]["bounded_chinese_development_pass"]
            or main["general_ilm_goal_complete"] or diagnosis["language_quality_claim"]):
        raise ValueError("failed source evidence differs or acquired a success claim")
    m = main["verification"]["measurements"]["cmrc"]
    s = diagnosis["summary"]
    return main, diagnosis, [100*m[k]["foreground_f1"] for k in ("normal", "blank")], [
        100*s[k] for k in ("free_f1", "assisted_f1", "supplied_f1")]


def generate():
    main,diagnosis,development,controls = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {name:archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst",NS)
    size = presentation.find("p:sldSz",NS)
    if len(slides) != 28 or (int(size.get('cx')),int(size.get('cy'))) != (15240000,8572500):
        raise ValueError("preserve the existing 28-slide template and geometry")
    def override(name,kind):
        if any(n.get("PartName") == "/"+name for n in types):
            raise ValueError("new native part collides with a preserved part")
        element("ct:Override",types,PartName="/"+name,ContentType=kind)
    specifications = []
    for number,categories,values in (
            (26,["Normal\nquestion","Question\nerased"],development),
            (27,["Free\ngeneration","Correct\nhistory","Supplied\ncodes"],controls)):
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
        book = f"ppt/embeddings/native_failure_chart{number}.xlsx"
        if name in original or book in original:
            raise ValueError("new chart overwrites preserved evidence")
        chart.find("c:externalData",NS).set(q("r:id"),"rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship",rels,Id="rId1",Type=NS["r"]+"/package",Target="../embeddings/"+Path(book).name)
        parts[name],parts[relpath(name)],parts[book] = xml(chart),xml(rels),workbook
        override(name,"application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        specifications.append(dict(part=name,workbook=book,categories=categories,values=values))
    source_part = "ppt/slides/slide28.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Conditional writing fails after full-context training",45,26,1510,58,31,True)
    slide.shape("a  Chinese development / F1 (%)",65,112,730,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart26.xml"),45,170,730,320)
    slide.shape("b  Training controls / F1 (%)",845,112,710,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart27.xml"),825,170,730,320)
    slide.shape("Exact images: 0/128 in both routes",80,505,680,38,19)
    slide.shape("Exact images: 0/32, 0/32, 28/32",865,505,665,38,19)
    slide.shape("c  First fixed question pair / complete raw writing images",65,560,1460,45,23,True)
    columns = [(245,"reference","Reference"),(570,"answer","Free generation"),
        (895,"assisted","Correct history"),(1220,"supplied","Supplied codes")]
    pictures = []
    for x,_,label in columns:
        slide.shape(label,x,618,300,36,20,True)
    for i,y in enumerate((675,775)):
        slide.shape("Question "+("A" if i==0 else "B"),65,y+8,170,40,20)
        for x,kind,_ in columns:
            relative = f"references/evidence/native_corpus_fit_v80_raw/{i:02}_{kind}.png"
            source = ROOT/relative
            expected = diagnosis["raw_pngs_sha256"][source.name]
            if sha(source) != expected:
                raise ValueError("raw diagnostic picture changed")
            with Image.open(source) as image:
                w,h = image.size
            if h != 32 or w > 128:
                raise ValueError("first fixed pair no longer fits the complete-image layout")
            part = f"ppt/media/native_failure_{i:02}_{kind}.png"
            if part in original:raise ValueError("new image part collides")
            parts[part] = source.read_bytes()
            slide.picture(slide.relation("image","../media/"+Path(part).name),
                f"Unmodified case {i:02} {kind}",x+5,y,w*2,h*2)
            pictures.append(dict(part=part,source=relative,sha256=expected))
    slide.shape("40,000 updates; 7.583B total / 46.900M trainable parameters. Image agreement does not establish semantic accuracy.",45,867,1510,32,17)
    layout = next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout"))
    slide.relation("slideLayout",layout)
    note = "ppt/notesSlides/native_failure_notes29.xml"
    slide.relation("notesSlide","../notesSlides/"+Path(note).name)
    part = "ppt/slides/slide29.xml"
    parts[part],parts[relpath(part)] = xml(slide.root),xml(slide.rels)
    override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    source_note = next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("Figure S14. Sources: references/evidence/native_chinese_v80_result.json and "
        "native_corpus_fit_v80_result.json. Completed 40000-update Chinese training on3193records,160000draws; "
        "7,583,221,875 complete parameters including externally pretrained weights,46,899,713 trained. "
        "Chinese128normal F1=36.504416%,question-erased25.403559%,exact0/128both; both-correct0/64. "
        "Fixed32 training cases free46.234673%,correct-history/knownlength50.240558%,supplied-code99.995316%. "
        "Exact counts0/32,0/32,28/32. First blocks equal32/32. Images show first original pair in cohort order, "
        "all complete rawPNGbytes unchanged at uniform2x placement; all32cases retained in receipt. "
        "No image repair, semantic transcription score, language qualification or efficiency superiority claimed.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp",NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
        if ph is not None and ph.get("type") == "body":
            for i,node in enumerate(shape.findall(".//a:t",NS)):node.text = description if i==0 else ""
    rels = ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):node.set("Target","../slides/slide29.xml")
    parts[note],parts[relpath(note)] = xml(notes),xml(rels)
    override(note,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    used = {r.get("Id") for r in relations};i = 1
    while f"rId{i}" in used:i += 1
    rid = f"rId{i}"
    element("pkg:Relationship",relations,Id=rid,Type=NS["r"]+"/slide",Target="slides/slide29.xml")
    element("p:sldId",slides,id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"),rid)
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),
                      ("ppt/_rels/presentation.xml.rels",relations)):
        parts[name] = xml(node)
    target = HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in parts.items():archive.writestr(name,data)
    receipt = dict(baseline_commit=BASE,pptx_sha256=sha(target),slides=29,preserved_native_slides=28,
        charts=specifications,pictures=pictures,builder_sha256=sha(Path(__file__)),
        sources_sha256={n:sha(ROOT/"references/evidence"/(n+".json")) for n in RECEIPTS},
        artifact_tool_available=False,language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/native_failure_deck_build.json").write_text(json.dumps(receipt,indent=2)+"\n")


def verify():
    _,diagnosis,development,controls = evidence()
    value = json.loads((HERE/"evidence/native_failure_deck_build.json").read_text())
    if value["pptx_sha256"] != sha(HERE/"ILM_Figures_editable.pptx") or value["builder_sha256"] != sha(Path(__file__)):
        raise ValueError("figure source/deck changed")
    mutable = {"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):raise ValueError("preserved native part changed: "+name)
        for name in new.namelist():
            if name.endswith((".xml",".rels")):ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS)) != 29:
            raise ValueError("wrong slide count")
        for spec,expected in zip(value["charts"],(development,controls)):
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
    if len(prefix) != 28 or len(prior) != 28 or len(new) != 29:raise ValueError("incorrect rendered slide count")
    for i in range(28):
        if prior[i].get_pixmap(alpha=False).samples != new[i].get_pixmap(alpha=False).samples:
            raise ValueError("fresh appearance changed for old slide")
    canonical = fitz.open();canonical.insert_pdf(prefix);canonical.insert_pdf(new,from_page=28,to_page=28)
    canonical.save(HERE/"ILM_Figures_review.pdf");canonical.close()
    figure = fitz.open();figure.insert_pdf(new,from_page=28,to_page=28)
    figure.save(HERE/"figures"/(FIGURE+".pdf"));figure.close()
    new[28].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures"/(FIGURE+".png"))
    (HERE/"figures"/(FIGURE+".svg")).write_text(new[28].get_svg_image(text_as_path=False))
    final = fitz.open(HERE/"ILM_Figures_review.pdf");preserved = []
    for i in range(28):
        pix = prefix[i].get_pixmap(alpha=False)
        if pix.samples != final[i].get_pixmap(alpha=False).samples:raise ValueError("canonical prefix appearance changed")
        preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(pix.samples).hexdigest()))
    for document in (prefix,prior,new,final):document.close()
    receipt = dict(pptx_sha256=build['pptx_sha256'],review_pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved,fresh_same_renderer_prefix_equal=True,
        figure_sha256={suffix:sha(HERE/"figures"/(FIGURE+"."+suffix)) for suffix in ('pdf','png','svg')},
        renderer_sha256=sha(Path(__file__)),headless_process_exited=True,visual_review_required=True,
        language_quality_claim=False,general_ilm_goal_complete=False)
    (HERE/"evidence/native_failure_figure_render.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('generate','verify','render'),required=True)
    args = parser.parse_args();dict(generate=generate,verify=verify,render=render)[args.stage]()
