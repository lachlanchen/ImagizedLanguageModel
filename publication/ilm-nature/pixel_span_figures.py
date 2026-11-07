#!/usr/bin/env python3
"""Append measured source-pixel copying without rewriting 35 native slides."""
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
BASE = "67d9523"
QA = ROOT / "artifacts/contextual_pixel_spans_publication_20261107"
FIGURE = "Figure_S21_contextual_pixel_spans"
RESULT = ROOT / "references/evidence/contextual_pixel_spans_result/result.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    r = json.loads(RESULT.read_text())
    for name, digest in r["files_sha256"].items():
        if sha(RESULT.parent / name) != digest:
            raise ValueError("measured raster evidence changed")
    keys = ("development/copy/normal/zh", "development/copy/normal/en", "train/qa/normal/zh",
            "development/qa/normal/zh", "development/qa/question_erased/zh", "development/qa/passage_erased/zh")
    if [(r["groups"][k]["exact"], r["groups"][k]["cases"]) for k in keys] != [(16,16),(16,16),(32,32),(9,64),(2,64),(0,64)]:
        raise ValueError("completed primary counts required")
    if r["component_qualified"] or r["general_ilm_goal_complete"] or sum(r["gates"].values()) != 1 or r["both_correct_pairs"] != 1:
        raise ValueError("reader qualification still fails")
    if r["training"]["updates"] != 3200 or len(r["native_replay_checks"]) != 4:
        raise ValueError("complete optimizer and actual native replay required")
    return r, [100.,100.,14.0625,3.125,0.]


def generate():
    result, values = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as archive:
        original = {n: archive.read(n) for n in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 35: raise ValueError("preserved 35-slide baseline required")

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types):
            raise ValueError("native part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    categories = ["New copies\n32/32", "Seen QA\n32/32", "Held-out QA\n9/64", "Question erased\n2/64", "Passage erased\n0/64"]
    chart, workbook = chart_part(parts["ppt/charts/chart12.xml"], categories, values, "Exact image answers (%)", 0, 100, "296486")
    for parent in chart.iter():
        for child in list(parent):
            if child.tag in (q("c:title"), q("c:dLbls")): parent.remove(child)
    for node in chart.findall(".//a:defRPr", NS): node.set("sz", "1500")
    for axis in chart.findall(".//c:valAx", NS):
        major = axis.find("c:majorUnit", NS)
        if major is None: major = element("c:majorUnit", axis)
        major.set("val", "25")
        formatting = axis.find("c:numFmt", NS)
        if formatting is None:
            formatting = element("c:numFmt", formatCode="0", sourceLinked="0")
            following = {q("c:" + n) for n in ("majorTickMark","minorTickMark","tickLblPos","spPr","txPr","crossAx")}
            axis.insert(next(i for i,n in enumerate(axis) if n.tag in following), formatting)
    for node in chart.findall(".//c:numFmt", NS):
        node.set("formatCode", "0"); node.set("sourceLinked", "0")
    name, book = "ppt/charts/chart42.xml", "ppt/embeddings/pixel_span_chart42.xlsx"
    chart.find("c:externalData", NS).set(q("r:id"), "rId1")
    rels = element("pkg:Relationships")
    element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
    override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
    override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
    source_part = "ppt/slides/slide35.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])
    slide = Slide(parts[source_part])
    slide.shape("Exact copying with weak question answering",45,24,1510,58,31,True)
    slide.shape("a  Image-only contextual span selection",45,92,1500,40,23,True)
    for words,x,y,w,h,fill in (
        ("Writing image\nX",45,155,190,90,"EDF2F6"),
        ("Frozen visual +\ncontextual body",290,155,300,90,"EDF2F6"),
        ("Learned span head\n2.38M parameters",645,155,270,90,"FFF1DA"),
        ("Select source span\nOriginal pixels",975,155,250,90,"EDF2F6"),
        ("Answer image\nCopied from X",1285,155,270,90,"EDF2F6")):
        slide.shape(words,x,y,w,h,19,False,True,fill)
    for c in ((235,200,290,200),(590,200,645,200),(915,200,975,200),(1225,200,1285,200)):
        slide.arrow(*c)
    slide.arrow(140,245,140,292,False); slide.arrow(140,292,1100,292,False); slide.arrow(1100,292,1100,245)
    slide.shape("Original source pixels",330,255,400,32,18)
    slide.shape("Normalized scores over visible spans; duplicate images share probability mass",45,315,1500,36,19)
    slide.shape("b  Exact image answers (%); counts below",45,380,800,40,22,True)
    slide.chart(slide.relation("chart","../charts/chart42.xml"),45,430,790,260)
    slide.shape("Both questions correct: 1/32 paragraph pairs",45,711,780,35,19)
    slide.shape("3/4 requirements fail; source-copy component only",45,750,790,36,19)
    slide.shape("c  First fixed held-out answer pair",875,380,680,40,22,True)
    slide.shape("1. What are the fish's common names?",875,426,680,26,17)
    slide.shape("2. Which ocean regions does it inhabit?",875,610,680,26,17)
    normal = [r for r in result["cases"] if r["split"] == "development" and r["task"] == "qa" and r["condition"] == "normal"]
    if [r["id"] for r in normal[:2]] != ["DEV_1595_QUERY_0", "DEV_1595_QUERY_1"]:
        raise ValueError("fixed first two development questions required")
    pictures = []
    for i, (record, reference_y, output_y) in enumerate(zip(normal[:2], (456,640), (527,715)),1):
        for kind, relative, y, label in (("ref",record["references"][0],reference_y,f"Reference {i}"),
                                       ("output",record["output"],output_y,f"Output {i}: " + ("correct" if record["exact"] else "wrong"))):
            source = RESULT.parent / relative
            stable = HERE / "figures" / f"pixel_span_{kind}{i}.png"
            stable.write_bytes(source.read_bytes())
            part = "ppt/media/" + stable.name
            if part in parts: raise ValueError("image part collision")
            parts[part] = stable.read_bytes()
            slide.shape(label,875,y,670,30,18,True)
            with Image.open(stable) as image:
                scale = min(1.15,660 / image.width)
                slide.picture(slide.relation("image","../media/"+stable.name),"Unmodified " + label,
                    875,y+29,image.width*scale,image.height*scale)
            pictures.append(dict(part=part,source=str(stable.relative_to(ROOT)),sha256=sha(stable)))
    slide.shape("7.531B full runtime with external pretraining; 320 training images, 3,200 updates",45,817,1510,34,18)
    slide.shape("Source copies cannot write answers absent from the input; free generation remains unachieved",45,856,1510,35,18)
    slide.relation("slideLayout",next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
    note = "ppt/notesSlides/pixel_span_notes36.xml"
    slide.relation("notesSlide","../notesSlides/"+Path(note).name)
    part = "ppt/slides/slide36.xml"
    override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
    parts[part],parts[relpath(part)] = xml(slide.root),xml(slide.rels)
    source_note = next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
    notes = ET.fromstring(parts[source_note])
    description = ("FigureS21. Completed contextual source-pixel copying component. Fullruntime7,531,277,171 parameters, learnedhead2,378,370, externallypretrainedQwen3VL body. "
        "All320trainingIDsseen80times,3200updates. Newcopy32/32,seenQA32/32,heldoutQA9/64,questionerased2/64,passageerased0/64,bothpairs1/32. Threeoffourrequirementsfail. "
        "NativeimageinferencefourcallsreplayfeaturesandpixelsincludingwrongQA. Theseareoriginalsourcepixelcrops,notfreeglyphsynthesisorqualifiedgeneralreading. "
        "Currentunitssplitinkcolumns,notcharacters;somewrongspanscutcharacters. Originalreference/outputPNGspreservedcompletely. "
        "Sources: references/evidence/contextual_pixel_spans_result/result.json; CMRC2018 Cui2019 CC-BY-SA4.0; Qwen3VL Alibaba Apache2.0. " + json.dumps(dict(percentages=values,categories=categories)))
    for shape in notes.findall("p:cSld/p:spTree/p:sp",NS):
        ph=shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
        if ph is not None and ph.get("type")=="body":
            for i,node in enumerate(shape.findall(".//a:t",NS)): node.text=description if i==0 else ""
    rels=ET.fromstring(parts[relpath(source_note)])
    for node in rels:
        if node.get("Type").endswith("/slide"): node.set("Target","../slides/slide36.xml")
    override(note,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
    parts[note],parts[relpath(note)]=xml(notes),xml(rels)
    used={r.get("Id") for r in relations};rid=next(f"rId{i}" for i in range(1,1000) if f"rId{i}" not in used)
    element("pkg:Relationship",relations,Id=rid,Type=NS["r"]+"/slide",Target="slides/slide36.xml")
    element("p:sldId",slides,id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"),rid)
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),("ppt/_rels/presentation.xml.rels",relations)):
        parts[name]=xml(node)
    QA.mkdir(parents=True,exist_ok=True);target=QA/"draft.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,data in parts.items(): archive.writestr(name,data)
    record=dict(baseline_commit=BASE,pptx_sha256=sha(target),preserved_native_slides=35,slides=36,
        charts=[dict(part="ppt/charts/chart42.xml",workbook=book,values=values,categories=categories)],pictures=pictures,
        builder_sha256=sha(__file__),source_result_sha256=sha(RESULT),artifact_tool_available=False,general_ilm_goal_complete=False)
    (QA/"deck_build.json").write_text(json.dumps(record,indent=2)+"\n")


def verify():
    _,values=evidence()
    record_path=QA/"deck_build.json" if (QA/"deck_build.json").exists() else HERE/"evidence/contextual_pixel_spans_deck_build.json"
    record=json.loads(record_path.read_text())
    target=QA/"draft.pptx" if (QA/"draft.pptx").exists() else HERE/"ILM_Figures_editable.pptx"
    if record["pptx_sha256"]!=sha(target) or record["builder_sha256"]!=sha(__file__) or record["source_result_sha256"]!=sha(RESULT):
        raise ValueError("source/deck binding changed")
    mutable={"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old,zipfile.ZipFile(target) as new:
        for name in old.namelist():
            if name not in mutable and old.read(name)!=new.read(name): raise ValueError("earliernativepartchanged: "+name)
        for name in new.namelist():
            if name.endswith((".xml",".rels")): ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS))!=36: raise ValueError("36slidesrequired")
        spec=record["charts"][0];chart=ET.fromstring(new.read(spec["part"]))
        actual=[float(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
        book=openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])),data_only=True)
        if actual!=values or [book.active.cell(i+2,2).value for i in range(len(values))]!=values: raise ValueError("nativechart/workbookdiffers")
        for picture in record["pictures"]:
            if hashlib.sha256(new.read(picture["part"])).hexdigest()!=picture["sha256"]: raise ValueError("rawanswerpixelschanged")
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
    if (len(prior), len(current), len(prefix)) != (35, 36, 35):
        raise ValueError("incorrect rendered counts")
    preserved = []
    for i in range(35):
        a, b = prior[i].get_pixmap(alpha=False), current[i].get_pixmap(alpha=False)
        if a.samples != b.samples:
            raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    canonical = fitz.open()
    canonical.insert_pdf(prefix)
    canonical.insert_pdf(current, from_page=35, to_page=35)
    canonical.save(HERE / "ILM_Figures_review.pdf")
    figure = fitz.open()
    figure.insert_pdf(current, from_page=35, to_page=35)
    figure.save(HERE / "figures" / (FIGURE + ".pdf"))
    embedded = [Image.open(io.BytesIO(figure.extract_image(item[0])["image"])).convert("RGB")
                for item in figure[0].get_images(full=True)]
    for picture in record["pictures"]:
        with Image.open(ROOT / picture["source"]) as image:
            image = image.convert("RGB")
            if not any(candidate.size == image.size and candidate.tobytes() == image.tobytes() for candidate in embedded):
                raise ValueError("standalone PDF changed input-page pixels")
    current[35].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(HERE / "figures" / (FIGURE + ".png"))
    (HERE / "figures" / (FIGURE + ".svg")).write_text(current[35].get_svg_image())
    result = dict(pptx_sha256=record["pptx_sha256"], review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        preserved_prior_slide_pixels=preserved, figure_png_sha256=sha(HERE / "figures" / (FIGURE + ".png")),
        new_slide=36, visual_review_required=True, artifact_tool_available=False)
    result["displayed_answer_images_pixel_identical"] = True
    import shutil
    if target != HERE / "ILM_Figures_editable.pptx":
        shutil.copy2(target,HERE / "ILM_Figures_editable.pptx")
    (HERE / "evidence/contextual_pixel_spans_deck_build.json").write_text(json.dumps(record,indent=2)+"\n")
    (HERE / "evidence/contextual_pixel_spans_figure_render.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Rendered slide 36 and preserved all 35 earlier native slide appearances.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=("generate","verify","render"),required=True)
    args=p.parse_args()
    {"generate":generate,"verify":verify,"render":render}[args.stage]()
