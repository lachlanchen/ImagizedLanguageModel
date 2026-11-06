#!/usr/bin/env python3
"""Append, render and verify one native source-reader evidence figure."""

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

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE="71e0fc4"
PREFIX="publication/ilm-nature/"
QA=ROOT/"artifacts/native_context_figures_v80_20261106"
RECEIPT=ROOT/"references/evidence/native_context_v80.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git","show",BASE+":"+PREFIX+name],cwd=ROOT)


def evidence():
    value=json.loads(RECEIPT.read_text())
    suffix=ROOT/"artifacts/suffix_kv_v78_20261106"
    execution=ROOT/"artifacts/native_image_execution_v79r2_20261106"
    v=value["suffix_visibility"];e=value["native_execution"]
    if (sha(suffix/"verification.json")!=v["verification_sha256"]
            or sha(execution/"report.json")!=e["report_sha256"]
            or sha(execution/"supplemental_verification.json")!=e["supplemental_verification_sha256"]
            or e["trained"] or e["optimizer_updates"] or not e["replay"]["offline_checks_pass"]):
        raise ValueError("completed source visibility and untrained native execution required")
    c=v["summaries"]["copy"];t=v["summaries"]["training"]
    return value,[c["full_exact"],c["exact"]],[t["full_exact"],t["exact"]]


def generate():
    value,copy_counts,training_counts=evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original={name:archive.read(name) for name in archive.namelist()}
    parts=dict(original)
    types=ET.fromstring(parts["[Content_Types].xml"])
    presentation=ET.fromstring(parts["ppt/presentation.xml"])
    relationships=ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides=presentation.find("p:sldIdLst",NS)
    if len(slides)!=27:
        raise ValueError("preserve the existing 27-slide template")
    def override(name,kind):
        element("ct:Override",types,PartName="/"+name,ContentType=kind)
    specifications=[]
    for number,categories,counts,maximum,unit in (
        (24,["Full prompt\nKV","Suffix keys\nvisible"],copy_counts,12,3),
        (25,["Full prompt\nKV","Suffix keys\nvisible"],training_counts,32,8)):
        chart,workbook=chart_part(parts["ppt/charts/chart12.xml"],categories,counts,
            "Exact canonical answers",0,maximum,"296486")
        for parent in chart.iter():
            for child in list(parent):
                if child.tag==q("c:title"):
                    parent.remove(child)
        for axis in chart.findall(".//c:valAx",NS):
            node=axis.find("c:majorUnit",NS)
            if node is None:
                node=element("c:majorUnit",axis)
            node.set("val",str(unit))
        for node in chart.findall(".//a:defRPr",NS):
            node.set("sz","1900")
        for node in chart.findall(".//c:numFmt",NS):
            node.set("formatCode","0");node.set("sourceLinked","0")
        for node in chart.findall(".//c:formatCode",NS):
            node.text="0"
        bar=chart.find(".//c:barChart",NS)
        for node in bar.findall("c:dLbls",NS):
            bar.remove(node)
        labels=element("c:dLbls")
        element("c:numFmt",labels,formatCode="0",sourceLinked="0")
        element("c:dLblPos",labels,val="outEnd")
        for tag,show in (("showLegendKey",0),("showVal",1),("showCatName",0),("showSerName",0),("showPercent",0),("showBubbleSize",0)):
            element("c:"+tag,labels,val=show)
        bar.insert(list(bar).index(bar.find("c:axId",NS)),labels)
        part=f"ppt/charts/chart{number}.xml"
        book=f"ppt/embeddings/native_context_chart{number}.xlsx"
        if part in original or book in original:
            raise ValueError("new chart collides with a preserved native part")
        chart.find("c:externalData",NS).set(q("r:id"),"rId1")
        rels=element("pkg:Relationships")
        element("pkg:Relationship",rels,Id="rId1",Type=NS["r"]+"/package",Target="../embeddings/"+Path(book).name)
        parts[part],parts[relpath(part)],parts[book]=xml(chart),xml(rels),workbook
        override(part,"application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        specifications.append(dict(part=part,workbook=book,categories=categories,values=counts,denominator=maximum))
    slide=Slide(parts["ppt/slides/slide27.xml"])
    slide.shape("Context visibility and native image continuation",45,26,1510,58,31,True)
    slide.shape("a  Exact copy answers / 12 requests",65,112,730,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart24.xml"),45,170,730,340)
    slide.shape("b  Exact answers / 32 training questions",845,112,710,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart25.xml"),825,170,730,340)
    slide.shape("c  New image actor: execution only; no trained language result",65,535,1460,48,23,True)
    slide.shape("Prompt\npixels",45,620,180,72,22,False,True,"F2F4F5")
    slide.arrow(225,656,275,656)
    slide.shape("Frozen visual reader\n+ 36-layer native body\nFull prompt KV",275,602,345,108,20,False,True,"E8F1F7")
    slide.arrow(620,656,670,656)
    slide.shape("Trainable clean-code\npredictor + stop",670,612,300,88,21,False,True,"F8F1ED")
    slide.arrow(970,656,1020,656)
    slide.shape("Fixed codec\ndecoder",1020,620,230,72,21,False,True,"F2F4F5")
    slide.arrow(1250,656,1300,656)
    slide.shape("Emitted glyph\nimage block",1300,620,255,72,21,False,True,"E8F1F7")
    slide.arrow(1427,692,1427,750)
    slide.shape("Fixed image\nencoder",1300,750,255,64,21,False,True,"F2F4F5")
    slide.arrow(1300,782,970,782)
    slide.shape("Trainable image input\n2 observations / block",670,750,300,64,21,False,True,"F8F1ED")
    slide.arrow(670,782,550,782,False)
    slide.arrow(550,782,550,710)
    slide.shape("7.583B total parameters; 46.900M trainable interface parameters. External pretraining retained; 16.638 GiB execution peak.",45,850,1510,40,17)
    source_part="ppt/slides/slide27.xml"
    source_rels=ET.fromstring(parts[relpath(source_part)])
    layout=next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout"))
    slide.relation("slideLayout",layout)
    note_part="ppt/notesSlides/native_context_notes28.xml"
    slide.relation("notesSlide","../notesSlides/"+Path(note_part).name)
    part="ppt/slides/slide28.xml"
    parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
    override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    source_note=next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes=ET.fromstring(parts[source_note])
    description=("Figure S13. Source: references/evidence/native_context_v80.json. "
        "Exact stopped canonical source-helper answers with full prompt KV versus only final52 suffix keys visible: "
        "copy8/12 versus0/12; fixedtraining19/32 versus2/32. Older keys are masked, not physically deleted. "
        "Original positions and44 first decisions remain unchanged. Same44 BF16 reader memories. "
        "All1239 source projection choices replay onCPU; no independent whole-native forward replay. "
        "This rejects the tested frozen suffix interface, not all learned compression. The native image actor retains "
        "full promptKV, learns image input, full-dimensional clean code output and stop, and rereads emitted pixels. "
        "Its first block uses the initial reader state; later blocks query the full cache. Actual36-layer execution "
        "uses16.638GiB,7,582,980,210 completeparameters and46,899,713 trainableparameters. "
        "Zerooptimizerupdates: not a trained language result. Externallypretrained frozen weights retained. "
        "Original checker errors preserved; supplemental feedback GPU hashes exact,CPUmaxerror2.98e-6 vs1e-4.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp",NS):
        ph=shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
        if ph is not None and ph.get("type")=="body":
            for i,node in enumerate(shape.findall(".//a:t",NS)):
                node.text=description if i==0 else ""
    rels=ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target","../slides/slide28.xml")
    parts[note_part],parts[relpath(note_part)]=xml(notes),xml(rels)
    override(note_part,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    used={r.get("Id") for r in relationships};i=1
    while f"rId{i}" in used:
        i+=1
    rid=f"rId{i}"
    element("pkg:Relationship",relationships,Id=rid,Type=NS["r"]+"/slide",Target="slides/slide28.xml")
    element("p:sldId",slides,id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"),rid)
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),("ppt/_rels/presentation.xml.rels",relationships)):
        parts[name]=xml(node)
    target=HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in parts.items():
            archive.writestr(name,data)
    receipt=dict(baseline_commit=BASE,pptx_sha256=sha(target),source_evidence_sha256=sha(RECEIPT),
        charts=specifications,slides=28,prior_native_slides_preserved=27,artifact_tool_available=False,
        builder_sha256=sha(Path(__file__)),language_quality_claim=False)
    (HERE/"evidence/native_context_deck_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Appended native evidence slide28 with two editable, source-bound charts.")


def verify_native():
    _,copy_counts,training_counts=evidence()
    build=json.loads((HERE/"evidence/native_context_deck_build.json").read_text())
    if (sha(HERE/"ILM_Figures_editable.pptx")!=build["pptx_sha256"] or sha(RECEIPT)!=build["source_evidence_sha256"]
            or sha(Path(__file__))!=build["builder_sha256"]):
        raise ValueError("figure source or native deck changed")
    mutable={"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        for name in old.namelist():
            if name not in mutable and old.read(name)!=new.read(name):
                raise ValueError("old native figure part changed: "+name)
        for name in new.namelist():
            if name.endswith((".xml",".rels")):
                ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS))!=28:
            raise ValueError("wrong slide inventory")
        for spec,expected in zip(build["charts"],(copy_counts,training_counts)):
            chart=ET.fromstring(new.read(spec["part"]))
            counts=[int(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
            book=openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])),data_only=True)
            values=[book.active.cell(i+2,2).value for i in range(len(expected))]
            categories=[book.active.cell(i+2,1).value for i in range(len(expected))]
            if counts!=expected or values!=expected or categories!=spec["categories"]:
                raise ValueError("chart cache/workbook differs from actual source")
    return build


def render():
    build=verify_native()
    QA.mkdir(parents=True,exist_ok=True)
    available={line.split(":",1)[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.split()[1].isdigit()}
    if available["MemAvailable"]<24*1024**2 or (available["SwapTotal"] and 1-available["SwapFree"]/available["SwapTotal"]>.75):
        raise RuntimeError("shared-host memory policy prevents rendering")
    reference=QA/"baseline.pptx";reference.write_bytes(baseline("ILM_Figures_editable.pptx"))
    output=QA/"render";output.mkdir(exist_ok=True)
    profile=QA/"libreoffice-profile"
    command=["soffice","-env:UserInstallation="+profile.as_uri(),"--headless","--convert-to","pdf",
        "--outdir",str(output),str(reference),str(HERE/"ILM_Figures_editable.pptx")]
    with (QA/"libreoffice.log").open("w") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            if process.wait(timeout=180):
                raise RuntimeError("owned figure conversion failed")
        finally:
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
    prefix=fitz.open(stream=baseline("ILM_Figures_review.pdf"),filetype="pdf")
    prior=fitz.open(output/"baseline.pdf");new=fitz.open(output/"ILM_Figures_editable.pdf")
    if len(prefix)!=27 or len(prior)!=27 or len(new)!=28:
        raise ValueError("incorrect canonical or fresh slide count")
    for i in range(27):
        if prior[i].get_pixmap(alpha=False).samples!=new[i].get_pixmap(alpha=False).samples:
            raise ValueError("appended content changed a fresh old slide render")
    canonical=fitz.open();canonical.insert_pdf(prefix);canonical.insert_pdf(new,from_page=27,to_page=27)
    canonical.save(HERE/"ILM_Figures_review.pdf");canonical.close()
    figure=fitz.open();figure.insert_pdf(new,from_page=27,to_page=27)
    figure.save(HERE/"figures/Figure_S13_native_context.pdf");figure.close()
    new[27].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures/Figure_S13_native_context.png")
    (HERE/"figures/Figure_S13_native_context.svg").write_text(new[27].get_svg_image(text_as_path=False))
    final=fitz.open(HERE/"ILM_Figures_review.pdf")
    preserved=[]
    for i in range(27):
        pix=prefix[i].get_pixmap(alpha=False)
        if pix.samples!=final[i].get_pixmap(alpha=False).samples:
            raise ValueError("canonical figure prefix changed")
        preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(pix.samples).hexdigest()))
    for document in (prefix,prior,new,final):
        document.close()
    receipt=dict(pptx_sha256=build["pptx_sha256"],review_pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved,fresh_same_renderer_prefix_equal=True,
        figure_sha256={suffix:sha(HERE/"figures"/("Figure_S13_native_context."+suffix)) for suffix in ("pdf","png","svg")},
        renderer_sha256=sha(Path(__file__)),headless_process_exited=True,visual_review_required=True,
        language_quality_claim=False)
    (HERE/"evidence/native_context_figure_render.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Rendered28 slides; all27 previous native and canonical appearances are preserved.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("generate","render","verify"),required=True)
    args=parser.parse_args()
    {"generate":generate,"render":render,"verify":verify_native}[args.stage]()
