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
BASE="84cef10"
PREFIX="publication/ilm-nature/"
QA=ROOT/"artifacts/reader_content_figures_v77_20261106"
RECEIPT=ROOT/"references/evidence/reader_content_v77_result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git","show",BASE+":"+PREFIX+name],cwd=ROOT)


def evidence():
    value=json.loads(RECEIPT.read_text())
    run=ROOT/value["source_run"]
    if (sha(run/"verification.json")!=value["verification_sha256"]
            or not value["verification"]["offline_checks_pass"]
            or value["verification"]["normal_memory_matches"]!=44
            or value["verification"]["compact_memory_sufficiency_proven"]):
        raise ValueError("source-bound helper result required")
    copy=json.loads((ROOT/"references/evidence/copy_transfer_v75_result.json").read_text())
    fit=json.loads((ROOT/"references/evidence/corpus_fit_v75_result.json").read_text())
    if (copy["verification_sha256"]!=value["source_copy_verification_sha256"]
            or fit["source_verification_sha256"]!=value["source_fit_verification_sha256"]):
        raise ValueError("original failed image actor comparison changed")
    s=value["verification"]["summaries"]
    return value,[copy["summary"]["groups"]["all"]["exact"],s["copy/normal"]["exact"],
        s["copy/text"]["exact"],s["copy/blank"]["exact"]], [fit["verification"]["summary"]["free_exact"],
        s["training/normal"]["exact"],s["training/text"]["exact"]]


def generate():
    value,copy_counts,training_counts=evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original={name:archive.read(name) for name in archive.namelist()}
    parts=dict(original)
    types=ET.fromstring(parts["[Content_Types].xml"])
    presentation=ET.fromstring(parts["ppt/presentation.xml"])
    relationships=ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides=presentation.find("p:sldIdLst",NS)
    if len(slides)!=26:
        raise ValueError("preserve the existing 26-slide template")
    def override(name,kind):
        element("ct:Override",types,PartName="/"+name,ContentType=kind)
    specifications=[]
    for number,categories,counts,maximum,unit in (
        (22,["ILM\nimage","Source\nimage","Source\ntext","Blank\nimage"],copy_counts,12,3),
        (23,["ILM\nimage","Source\nimage","Source\ntext"],training_counts,32,8)):
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
        book=f"ppt/embeddings/reader_content_chart{number}.xlsx"
        if part in original or book in original:
            raise ValueError("new chart collides with a preserved native part")
        chart.find("c:externalData",NS).set(q("r:id"),"rId1")
        rels=element("pkg:Relationships")
        element("pkg:Relationship",rels,Id="rId1",Type=NS["r"]+"/package",Target="../embeddings/"+Path(book).name)
        parts[part],parts[relpath(part)],parts[book]=xml(chart),xml(rels),workbook
        override(part,"application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        specifications.append(dict(part=part,workbook=book,categories=categories,values=counts,denominator=maximum))
    slide=Slide(parts["ppt/slides/slide26.xml"])
    slide.shape("Reader content before image writing",45,26,1510,58,31,True)
    slide.shape("a  Exact copied answers / 12 new requests",65,112,730,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart22.xml"),45,170,730,355)
    slide.shape("b  Exact answers / 32 fixed training questions",845,112,710,48,23,True)
    slide.chart(slide.relation("chart","../charts/chart23.xml"),825,170,730,355)
    slide.shape("c  Shared observation; different continuation information",65,550,1460,48,23,True)
    slide.shape("Same prompt\npixels",45,655,200,82,22,False,True,"F2F4F5")
    slide.shape("Imported image +\ncontextual reader",300,655,300,82,22,False,True,"E8F1F7")
    slide.arrow(245,696,300,696)
    slide.arrow(600,696,650,696,False)
    slide.arrow(650,625,650,775,False)
    slide.arrow(650,625,700,625)
    slide.arrow(650,775,700,775)
    slide.shape("Full layerwise\nvisual/context KV cache",700,590,320,70,21,False,True,"F8F1ED")
    slide.shape("52 final states (BF16)\n44/44 match exactly",700,740,320,70,21,False,True,"F2F4F5")
    slide.arrow(1020,625,1110,625)
    slide.arrow(1020,775,1110,775)
    slide.shape("Offline source helper\nGreedy text output",1110,590,445,70,21,False,True,"F8F1ED")
    slide.shape("Independent image writer\nAutonomous glyph output",1110,740,445,70,21,False,True,"E8F1F7")
    slide.shape("100 fixed offline helper calls. Full-KV success does not prove compact-memory sufficiency or successful ILM image generation.",45,850,1510,40,17)
    source_part="ppt/slides/slide26.xml"
    source_rels=ET.fromstring(parts[relpath(source_part)])
    layout=next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout"))
    slide.relation("slideLayout",layout)
    note_part="ppt/notesSlides/reader_content_notes27.xml"
    slide.relation("notesSlide","../notesSlides/"+Path(note_part).name)
    part="ppt/slides/slide27.xml"
    parts[part],parts[relpath(part)]=xml(slide.root),xml(slide.rels)
    override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    source_note=next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes=ET.fromstring(parts[source_note])
    description=("Figure S12. Source: references/evidence/reader_content_v77_result.json; verified receipt "+value["verification_sha256"]+
        ". Same12 original copy requests and same32 fixed training questions as the failed V75 image actor. "
        "Charts use literal stopped canonical-answer counts, with outer whitespace trim only in the source helper. "
        "Image actor values are unchanged original physical-raster counts. Source helper text/blank controls are separate. "
        "All44 normal prefill memory suffixes match the original52-state BF16 bytes exactly. The first source projection "
        "uses the last compact-memory vector, while subsequent source decoding uses full layerwise visual/context KV. "
        "Thus source-helper success does not establish sufficiency of the compact memory or independent ILM success. "
        "100 source-helper calls; no optimizer updates or new independent image-actor calls. External pretraining retained. "
        "All725 retained source projection choices replay exactly; the complete vision/context forward pass is not independently replayed.")
    for shape in notes.findall("p:cSld/p:spTree/p:sp",NS):
        ph=shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
        if ph is not None and ph.get("type")=="body":
            for i,node in enumerate(shape.findall(".//a:t",NS)):
                node.text=description if i==0 else ""
    rels=ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"):
            node.set("Target","../slides/slide27.xml")
    parts[note_part],parts[relpath(note_part)]=xml(notes),xml(rels)
    override(note_part,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    used={r.get("Id") for r in relationships};i=1
    while f"rId{i}" in used:
        i+=1
    rid=f"rId{i}"
    element("pkg:Relationship",relationships,Id=rid,Type=NS["r"]+"/slide",Target="slides/slide27.xml")
    element("p:sldId",slides,id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"),rid)
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),("ppt/_rels/presentation.xml.rels",relationships)):
        parts[name]=xml(node)
    target=HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in parts.items():
            archive.writestr(name,data)
    receipt=dict(baseline_commit=BASE,pptx_sha256=sha(target),source_evidence_sha256=sha(RECEIPT),
        charts=specifications,slides=27,prior_native_slides_preserved=26,artifact_tool_available=False,
        builder_sha256=sha(Path(__file__)),language_quality_claim=False)
    (HERE/"evidence/reader_content_deck_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Appended native evidence slide27 with two editable, source-bound charts.")


def verify_native():
    _,copy_counts,training_counts=evidence()
    build=json.loads((HERE/"evidence/reader_content_deck_build.json").read_text())
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
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS))!=27:
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
    if len(prefix)!=26 or len(prior)!=26 or len(new)!=27:
        raise ValueError("incorrect canonical or fresh slide count")
    for i in range(26):
        if prior[i].get_pixmap(alpha=False).samples!=new[i].get_pixmap(alpha=False).samples:
            raise ValueError("appended content changed a fresh old slide render")
    canonical=fitz.open();canonical.insert_pdf(prefix);canonical.insert_pdf(new,from_page=26,to_page=26)
    canonical.save(HERE/"ILM_Figures_review.pdf");canonical.close()
    figure=fitz.open();figure.insert_pdf(new,from_page=26,to_page=26)
    figure.save(HERE/"figures/Figure_S12_reader_content.pdf");figure.close()
    new[26].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures/Figure_S12_reader_content.png")
    (HERE/"figures/Figure_S12_reader_content.svg").write_text(new[26].get_svg_image(text_as_path=False))
    final=fitz.open(HERE/"ILM_Figures_review.pdf")
    preserved=[]
    for i in range(26):
        pix=prefix[i].get_pixmap(alpha=False)
        if pix.samples!=final[i].get_pixmap(alpha=False).samples:
            raise ValueError("canonical figure prefix changed")
        preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(pix.samples).hexdigest()))
    for document in (prefix,prior,new,final):
        document.close()
    receipt=dict(pptx_sha256=build["pptx_sha256"],review_pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved,fresh_same_renderer_prefix_equal=True,
        figure_sha256={suffix:sha(HERE/"figures"/("Figure_S12_reader_content."+suffix)) for suffix in ("pdf","png","svg")},
        renderer_sha256=sha(Path(__file__)),headless_process_exited=True,visual_review_required=True,
        language_quality_claim=False)
    (HERE/"evidence/reader_content_figure_render.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Rendered27 slides; all26 previous native and canonical appearances are preserved.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("generate","render","verify"),required=True)
    args=parser.parse_args()
    {"generate":generate,"render":render,"verify":verify_native}[args.stage]()
