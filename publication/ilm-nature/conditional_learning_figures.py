#!/usr/bin/env python3
"""Append two measured, native editable figures; preserve 36 earlier slides."""
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

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "11f2edce0d2f25e21f597217e02afe70f36c1cf8"
QA = ROOT / "artifacts/conditional_learning_publication_20261108"
RESULT = ROOT / "references/evidence/contextual_pixel_adaptation_result/result.json"
MEMORY = ROOT / "references/evidence/contextual_query_memory_result/result.json"
HISTORY = ROOT / "references/evidence/history_prefix_alignment_result.json"
FIGURES = ("Figure_S22_contextual_learning", "Figure_S23_history_prefix")
COLORS = ("296486", "AE603D", "4D7D66")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def baseline(name):
    return subprocess.check_output(["git", "show", BASE + ":publication/ilm-nature/" + name], cwd=ROOT)


def evidence():
    r, m, h = [json.loads(p.read_text()) for p in (RESULT, MEMORY, HISTORY)]
    for name, digest in r["files_sha256"].items():
        if sha(RESULT.parent / name) != digest:
            raise ValueError("comparison evidence changed: " + name)
    if r["native_assessment_calls"] != 512 or len(r["standalone_native_replays"]) != 8:
        raise ValueError("complete native assessment/replay required")
    if any(not a["answer_pixels_exact"] or not a["events_exact"] for a in r["standalone_native_replays"]):
        raise ValueError("standalone replay failed")
    for arm, expected in (("frozen", [32, 28, 23, 3, 1, 0]), ("adapted", [32, 27, 26, 3, 0, 0])):
        a = r["arms"][arm]
        actual = [a["new_copy_exact"], a["groups"]["train/qa/normal/zh"]["exact"], a["normal_qa_exact"],
                  a["both_correct_pairs"], a["groups"]["development/qa/question_erased/zh"]["exact"],
                  a["groups"]["development/qa/passage_erased/zh"]["exact"]]
        if actual != expected or a["component_qualified"] or sum(a["gates"].values()) != 2:
            raise ValueError("fixed comparison counts/gates changed")
    if r["useful_improvement_over_control"] or r["general_ilm_goal_complete"]:
        raise ValueError("failed improvement/full-goal boundary changed")
    if m["groups"]["development"]["changed_selected_images"] != 53 or m["groups"]["development"]["own_exact"] != 9:
        raise ValueError("earlier-head diagnostic changed")
    if h["totals"]["intermediate/matched_state_pairs"] != 2 or h["totals"]["intermediate/boundaries"] != 75:
        raise ValueError("history denominators changed")
    if h["source_model_forward_calls"] or h["optimizer_updates"]:
        raise ValueError("history audit must remain metadata-only")
    return r, m, h


def multi_chart(template, categories, series, labels, maximum, unit):
    """Native clustered chart, native XLSX, explicit displayed count labels."""
    first = series[0]
    root, _ = chart_part(template, categories, first[1], first[0], 0, maximum, COLORS[0])
    chart = root.find("c:chart", NS)
    bar = chart.find("c:plotArea/c:barChart", NS)
    prototype = copy.deepcopy(bar.find("c:ser", NS))
    for old_labels in prototype.findall("c:dLbls", NS):
        prototype.remove(old_labels)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in (q("c:title"), q("c:dLbls"), q("c:legend")):
                parent.remove(child)
    for child in list(bar):
        if child.tag == q("c:ser"):
            bar.remove(child)
    bar.find("c:grouping", NS).set("val", "clustered")
    for j, ((name, values), counts) in enumerate(zip(series, labels)):
        s = copy.deepcopy(prototype)
        s.find("c:idx", NS).set("val", str(j)); s.find("c:order", NS).set("val", str(j))
        for t in ("c:tx", "c:val"):
            s.remove(s.find(t, NS))
        tx = element("c:tx"); element("c:v", tx).text = name; s.insert(2, tx)
        for node in s.findall("c:spPr/a:solidFill/a:srgbClr", NS): node.set("val", COLORS[j])
        ref = element("c:numRef", element("c:val", s))
        element("c:f", ref).text = f"Sheet1!${chr(66+j)}$2:${chr(66+j)}${len(values)+1}"
        cache = element("c:numCache", ref)
        element("c:formatCode", cache).text = "0.000"
        element("c:ptCount", cache, val=len(values))
        for i, value in enumerate(values): element("c:v", element("c:pt", cache, idx=i)).text = str(value)
        dl = element("c:dLbls", s)
        for i, value in enumerate(counts):
            d = element("c:dLbl", dl); element("c:idx", d, val=i)
            rich = element("c:rich", element("c:tx", d))
            element("a:bodyPr", rich); element("a:lstStyle", rich)
            run = element("a:r", element("a:p", rich))
            props = element("a:rPr", run, sz="1400")
            element("a:srgbClr", element("a:solidFill", props), val="1D2B34")
            element("a:latin", props, typeface="Arial")
            element("a:t", run).text = str(value)
            element("c:dLblPos", d, val="outEnd")
            element("c:showLegendKey", d, val="0")
        for tag in ("showLegendKey", "showVal", "showCatName", "showSerName", "showPercent", "showBubbleSize"):
            element("c:" + tag, dl, val="0")
        s.remove(dl)
        s.insert(list(s).index(s.find("c:cat", NS)), dl)
        bar.insert(2+j, s)
    for axis in chart.findall("c:plotArea/c:valAx", NS):
        major = axis.find("c:majorUnit", NS)
        if major is None: major = element("c:majorUnit", axis)
        major.set("val", str(unit))
    for node in root.findall(".//a:defRPr", NS): node.set("sz", "1400")
    for node in root.findall(".//c:numFmt", NS):
        node.set("formatCode", "0"); node.set("sourceLinked", "0")
    legend = element("c:legend", chart)
    element("c:legendPos", legend, val="b"); element("c:overlay", legend, val="0")
    text = element("c:txPr", legend); element("a:bodyPr", text); element("a:lstStyle", text)
    p = element("a:p", text); element("a:defRPr", element("a:pPr", p), sz="1400")
    chart.remove(legend)
    chart.insert(list(chart).index(chart.find("c:plotArea", NS))+1, legend)
    book = openpyxl.Workbook(); sheet = book.active; sheet.title = "Sheet1"
    sheet.append(["Condition"] + [s[0] for s in series])
    for i, category in enumerate(categories): sheet.append([category] + [s[1][i] for s in series])
    count_sheet = book.create_sheet("Displayed counts")
    count_sheet.append(["Condition"] + [s[0] for s in series])
    for i, category in enumerate(categories): count_sheet.append([category] + [s[i] for s in labels])
    buffer = io.BytesIO(); book.save(buffer)
    return root, buffer.getvalue()


def generate():
    result, _, history = evidence()
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as z:
        original = {n: z.read(n) for n in z.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relations = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 36: raise ValueError("36-slide immutable baseline required")
    charts, pictures = [], []

    def override(name, kind):
        if name in original or any(n.get("PartName") == "/" + name for n in types): raise ValueError("part collision")
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def add_chart(slide, categories, series, counts, maximum, unit, x, y, w, h):
        number = 43 + len(charts)
        name, book = f"ppt/charts/chart{number}.xml", f"ppt/embeddings/conditional_chart{number}.xlsx"
        chart, workbook = multi_chart(parts["ppt/charts/chart42.xml"], categories, series, counts, maximum, unit)
        chart.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        parts[name], parts[relpath(name)], parts[book] = xml(chart), xml(rels), workbook
        slide.chart(slide.relation("chart", "../charts/" + Path(name).name), x, y, w, h)
        charts.append(dict(part=name, workbook=book, categories=categories, series=series, counts=counts))

    def picture(slide, source, name, owner, x, y, maxwidth, scale):
        stable = HERE / "figures" / (name + ".png"); stable.write_bytes(source.read_bytes())
        part = "ppt/media/" + stable.name
        if part in parts: raise ValueError("picture collision")
        parts[part] = stable.read_bytes()
        with Image.open(stable) as im:
            ratio = min(scale, maxwidth / im.width)
            slide.picture(slide.relation("image", "../media/" + stable.name), "Unmodified " + name,
                          x, y, im.width * ratio, im.height * ratio)
        pictures.append(dict(part=part, source=str(stable.relative_to(ROOT)), sha256=sha(stable), slide=owner))

    source_part = "ppt/slides/slide36.xml"
    source_rels = ET.fromstring(parts[relpath(source_part)])

    def append(slide, number, description):
        slide.relation("slideLayout", next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout")))
        note, part = f"ppt/notesSlides/conditional_notes{number}.xml", f"ppt/slides/slide{number}.xml"
        slide.relation("notesSlide", "../notesSlides/" + Path(note).name)
        override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        source_note = next(resolve(source_part, n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
        notes = ET.fromstring(parts[source_note])
        for shape in notes.findall("p:cSld/p:spTree/p:sp", NS):
            ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if ph is not None and ph.get("type") == "body":
                for i, node in enumerate(shape.findall(".//a:t", NS)): node.text = description if i == 0 else ""
        rels = ET.fromstring(parts[relpath(source_note)])
        for node in rels:
            if node.get("Type").endswith("/slide"): node.set("Target", f"../slides/slide{number}.xml")
        override(note, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        parts[note], parts[relpath(note)] = xml(notes), xml(rels)
        used = {r.get("Id") for r in relations}; rid = next(f"rId{i}" for i in range(1,1000) if f"rId{i}" not in used)
        element("pkg:Relationship", relations, Id=rid, Type=NS["r"] + "/slide", Target=f"slides/slide{number}.xml")
        element("p:sldId", slides, id=max(int(s.get("id")) for s in slides)+1).set(q("r:id"), rid)

    slide = Slide(parts[source_part])
    slide.shape("Contextual learning improves few answers and misses reader requirements",45,24,1510,60,31,True)
    slide.shape("a  Same source-pixel motor; matched learning schedule",45,98,1510,40,22,True)
    nodes = [("Writing\nimage",45,150),("Frozen\nvision",240,210),
             ("Contextual attention\nControl: frozen\nAdapted: rank 8",495,280),
             ("Learned span head\nBoth arms",820,210),("Select source span\nOriginal pixels",1075,240),("Answer\nimage",1360,195)]
    for text,x,w in nodes: slide.shape(text,x,150,w,102,18,False,True,"FFF1DA" if x in (495,820) else "EDF2F6")
    for a,b in zip(nodes,nodes[1:]): slide.arrow(a[1]+a[2],201,b[1],201)
    slide.arrow(120,252,120,291,False); slide.arrow(120,291,1195,291,False); slide.arrow(1195,291,1195,252)
    slide.shape("Original source pixels",290,257,450,30,17)
    slide.shape("3,257 updates per arm; 2.378M / 8.572M learned parameters; 7.531B pretrained full runtime",45,306,1510,34,17)
    slide.shape("b  Exact image answers (%); labels give correct counts",45,356,790,40,22,True)
    counts = [[32,28,23,3,1,0],[32,27,26,3,0,0]]; denominators=[32,32,64,32,64,64]
    categories=["New copies\n(n=32)","Seen QA\n(n=32)","Unseen QA\n(n=64)","Both correct\n(n=32 pairs)","Question erased\n(n=64)","Passage erased\n(n=64)"]
    series=[(name,[100*c/n for c,n in zip(cs,denominators)]) for name,cs in zip(("Control","Adapted"),counts)]
    add_chart(slide,categories,series,counts,110,25,45,406,765,298)
    slide.shape("Gain: +3 answers, +0 complete pairs",45,713,790,32,19,True)
    slide.shape("Both fail reader floors; adaptation misses improvement rule",45,748,790,40,17)
    slide.shape("c  First fixed unseen question pair",850,356,705,40,22,True)
    normal = {arm:[c for c in result["arms"][arm]["cases"] if c["split"]=="development" and c["task"]=="qa" and c["condition"]=="normal"] for arm in ("frozen","adapted")}
    if [r["id"] for r in normal["frozen"][:2]] != ["DEV_1595_QUERY_0","DEV_1595_QUERY_1"]: raise ValueError("fixed first pair changed")
    for i, (heading,top) in enumerate((("1. What are the fish's common names?",404),("2. Which ocean regions does it inhabit?",574))):
        slide.shape(heading,850,top,705,30,17)
        rows=[("Reference",normal["frozen"][i]["references"][0],"reference"),("Control",normal["frozen"][i]["output"],"control"),("Adapted",normal["adapted"][i]["output"],"adapted")]
        for j,(label,path,kind) in enumerate(rows):
            y=top+36+j*42; slide.shape(label,850,y,145,33,17,True)
            picture(slide,RESULT.parent/path,f"learning_{kind}{i+1}",37,1005,y,545,1.15)
    slide.shape("Both add a comma in Q1; both reproduce Q2 exactly",850,738,705,42,15)
    slide.shape("d  Separate task-memory substitution: earlier head",45,793,1510,32,20,True)
    slide.shape("53/64 development selections change; original score 9/64. Hybrid features, not native new prompts.",45,831,1510,32,17)
    slide.shape("Source crops cannot compose absent answers. Earlier-head substitution is not a third comparison arm.",45,868,1510,28,15)
    append(slide,37,"Figure S22. Matched contextual learning, same original-pixel motor. Both full runtimes 7,531,277,171 externally pretrained parameters. Counts control/adapted: copies32/32 both,seen28/32vs27/32,unseen23/64vs26/64,pairs3/32both,question erased1/64vs0/64,passage erased0/64both. Both fail natural/pair floors; adaptation misses prospective gain rule. Passage erasure removes all correct source support. All examples unmodified raw PNGs, fixed first pair. Earlier head memory intervention53/64changes at9/64original is separate, hybrid not native. CMRC2018 Cui et al CC-BY-SA4.0; Qwen3-VL Alibaba Apache2.0. Sources: references/evidence/contextual_pixel_adaptation_result/result.json; contextual_query_memory_result/result.json. "+json.dumps(charts[-1]))

    slide = Slide(parts[source_part])
    slide.shape("Image steps and language pieces usually describe different histories",45,24,1510,60,31,True)
    slide.shape("a  Audit identical preceding content before comparing states",45,98,1510,40,22,True)
    for label,y,units,state in (("Image writer",150,"64-column image blocks\nTwo tile embeddings per block","Writer state"),("Language helper",250,"Variable native language pieces\nComplete decoded text prefix","Helper state")):
        slide.shape(label,45,y+15,200,50,20,True)
        slide.shape(units,255,y,365,80,18,False,True,"EDF2F6")
        slide.shape(state,680,y,285,80,19,False,True,"EDF2F6")
        slide.arrow(620,y+40,680,y+40); slide.arrow(965,y+40,1075,240)
    slide.shape("Match histories by visible pixels\nFull reference's rendering origin\nA saved helper state is also required",1075,165,480,150,19,False,True,"FFF1DA")
    slide.shape("Equal ordinal step does not establish equal history",45,337,1510,28,16)
    slide.shape("b  All retained boundary counts",45,377,790,38,22,True)
    counts=[[32,75,32],[32,7,32],[32,2,19]]
    add_chart(slide,["Initial","Intermediate","Terminal"],list(zip(("Boundaries","Prefix raster","Saved state"),counts)),counts,85,20,45,425,765,290)
    slide.shape("All 32 initial states agree after common BF16 rounding",45,725,790,45,17)
    slide.shape("c  Both eligible intermediate matches",850,377,705,38,22,True)
    for i,(case_index,boundary_index,top) in enumerate(((17,3,425),(23,1,594)),1):
        c=history["cases"][case_index]; b=c["boundaries"][boundary_index]; match=b["matches"][0]
        slide.shape(Path(c["source"]).stem,850,top,705,32,18,True)
        prefix=c["canonical_prefixes"][b["canonical_prefix_indices"][0]]["image"]
        prefix_source=ROOT/history["source_run"]/prefix
        if not prefix_source.exists(): prefix_source=HERE/"figures"/f"history_matched_prefix{i}.png"
        if sha(prefix_source)!=history["local_files_sha256"][prefix]: raise ValueError("history prefix pixels changed")
        picture(slide,prefix_source,f"history_matched_prefix{i}",38,855,top+41,690,1.25)
        slide.shape(f"Image positions: {2*b['position']}     Helper positions: {match['source_position']}",850,top+89,705,32,17)
        slide.shape(f"Normalized state distance: {match['normalized_state_rmse']:.3f} (uncalibrated)",850,top+122,705,32,16)
    slide.shape("68/75 intermediate boundaries have no complete-prefix raster; 5 more lack a saved matching state",45,776,1510,34,17)
    slide.shape("All 216 saved source-prefix positions decode to complete UTF-8; 13 terminal states are missing",45,815,1510,34,17)
    slide.shape("Existing traces only: zero new neural calls, zero updates. No alignment treatment or writer repair tested.",45,858,1510,34,17)
    append(slide,38,"Figure S23. Existing-trace history audit, not alignment training. Counts initial32/32/32,intermediate75/7/2,terminal32/32/19 for boundaries/prefix rasters/saved states. All32initial BF16statesagree;68/75intermediate boundaries raster incompatible;5eligible missing saved states. All216savedsourceprefixes completeUTF8. Both intermediate correspondences retained with original fulloriginprefixPNG: TRAIN1779Q1image6/helper6distance1.4969523; TRAIN2146Q1image2/helper6distance1.4067816. Distances uncalibrated, not language quality or unique causal evidence. Zero neural calls,zerooptimizerupdates,noalignedtraining. Source: references/evidence/history_prefix_alignment_result.json. CMRC2018 Cui et al CC-BY-SA4.0; Qwen3-VL Alibaba Apache2.0. "+json.dumps(charts[-1]))
    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),("ppt/_rels/presentation.xml.rels",relations)): parts[name]=xml(node)
    QA.mkdir(parents=True,exist_ok=True); target=QA/"draft.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
        for name,data in parts.items(): z.writestr(name,data)
    record=dict(baseline_commit=BASE,pptx_sha256=sha(target),preserved_native_slides=36,slides=38,charts=charts,pictures=pictures,
                builder_sha256=sha(__file__),sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in (RESULT,MEMORY,HISTORY)},
                artifact_tool_available=False,powerpoint_app_inspection=False,general_ilm_goal_complete=False)
    (QA/"deck_build.json").write_text(json.dumps(record,indent=2)+"\n")
    verify()


def verify():
    evidence()
    record_path=QA/"deck_build.json" if (QA/"deck_build.json").exists() else HERE/"evidence/conditional_learning_deck_build.json"
    record=json.loads(record_path.read_text())
    target=QA/"draft.pptx" if (QA/"draft.pptx").exists() else HERE/"ILM_Figures_editable.pptx"
    if sha(target)!=record["pptx_sha256"] or sha(__file__)!=record["builder_sha256"]: raise ValueError("deck/builder binding changed")
    for name,digest in record["sources_sha256"].items():
        if sha(ROOT/name)!=digest: raise ValueError("source binding changed")
    mutable={"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    with zipfile.ZipFile(io.BytesIO(baseline("ILM_Figures_editable.pptx"))) as old,zipfile.ZipFile(target) as new:
        for name in old.namelist():
            if name not in mutable and old.read(name)!=new.read(name): raise ValueError("earlier part changed: "+name)
        for name in new.namelist():
            if name.endswith((".xml",".rels")): ET.fromstring(new.read(name))
        if len(ET.fromstring(new.read("ppt/presentation.xml")).find("p:sldIdLst",NS))!=38: raise ValueError("38 slides required")
        for spec in record["charts"]:
            chart=ET.fromstring(new.read(spec["part"])); book=openpyxl.load_workbook(io.BytesIO(new.read(spec["workbook"])),data_only=True)
            all_series=chart.findall(".//c:barChart/c:ser",NS)
            if len(all_series)!=len(spec["series"]): raise ValueError("native chart series count changed")
            for j,s in enumerate(all_series):
                values=[float(n.text) for n in s.findall("c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
                labels=[int(n.text) for n in s.findall("c:dLbls/c:dLbl/c:tx/c:rich/a:p/a:r/a:t",NS)]
                if values!=spec["series"][j][1] or labels!=spec["counts"][j]: raise ValueError("native chart data changed")
                if [book.active.cell(i+2,j+2).value for i in range(len(values))]!=values: raise ValueError("workbook differs")
                if [book["Displayed counts"].cell(i+2,j+2).value for i in range(len(values))]!=labels: raise ValueError("count workbook differs")
                categories=[n.text for n in s.findall("c:cat/c:strRef/c:strCache/c:pt/c:v",NS)]
                if categories!=spec["categories"]: raise ValueError("chart categories changed")
        for pic in record["pictures"]:
            if hashlib.sha256(new.read(pic["part"])).hexdigest()!=pic["sha256"]: raise ValueError("raw image changed")
    return record


def render():
    from build_paired_writing_publication import build_slot
    record=verify(); target=QA/"draft.pptx"; output=QA/"render"; output.mkdir(exist_ok=True)
    with build_slot():
        source=QA/"baseline.pptx"; source.write_bytes(baseline("ILM_Figures_editable.pptx"))
        command=["soffice","-env:UserInstallation="+(QA/"libreoffice-profile").as_uri(),"--headless","--convert-to",
                 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
                 "--outdir",str(output),str(source),str(target)]
        with (QA/"libreoffice.log").open("w") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (QA/"render_status.json").write_text(json.dumps(dict(pid=process.pid,owned_converter=True,state="running")))
            try:
                if process.wait(timeout=180): raise RuntimeError("owned converter failed")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
    prior,current=fitz.open(output/"baseline.pdf"),fitz.open(output/"draft.pdf")
    prefix=fitz.open(stream=baseline("ILM_Figures_review.pdf"),filetype="pdf")
    if (len(prior),len(current),len(prefix))!=(36,38,36): raise ValueError("rendered slide count differs")
    preserved=[]
    for i in range(36):
        a,b=prior[i].get_pixmap(alpha=False),current[i].get_pixmap(alpha=False)
        if a.samples!=b.samples: raise ValueError("earlier slide appearance changed")
        preserved.append(hashlib.sha256(b.samples).hexdigest())
    canonical=fitz.open(); canonical.insert_pdf(prefix); canonical.insert_pdf(current,from_page=36,to_page=37)
    canonical.save(HERE/"ILM_Figures_review.pdf")
    hashes={}
    for i,name in enumerate(FIGURES,36):
        figure=fitz.open(); figure.insert_pdf(current,from_page=i,to_page=i); figure.save(HERE/"figures"/(name+".pdf"))
        embedded=[]
        for item in figure[0].get_images(full=True):
            with Image.open(io.BytesIO(figure.extract_image(item[0])["image"])) as im: embedded.append((im.size,im.convert("RGB").tobytes()))
        for pic in record["pictures"]:
            if pic["slide"]!=i+1: continue
            with Image.open(ROOT/pic["source"]) as im:
                if (im.size,im.convert("RGB").tobytes()) not in embedded: raise ValueError("PDF raw image changed")
        current[i].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures"/(name+".png"))
        (HERE/"figures"/(name+".svg")).write_text(current[i].get_svg_image())
        hashes[name]=sha(HERE/"figures"/(name+".png"))
    shutil.copy2(target,HERE/"ILM_Figures_editable.pptx")
    (HERE/"evidence/conditional_learning_deck_build.json").write_text(json.dumps(record,indent=2)+"\n")
    rendered=dict(pptx_sha256=sha(target),review_pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),preserved_prior_slide_pixels=preserved,
                  figure_png_sha256=hashes,displayed_images_pixel_identical=True,artifact_tool_available=False,
                  font_substitution="Arial rendered with Liberation Sans",powerpoint_app_inspection=False,visual_review_required=True)
    (HERE/"evidence/conditional_learning_figure_render.json").write_text(json.dumps(rendered,indent=2)+"\n")
    (QA/"render_status.json").write_text(json.dumps(dict(state="completed",owned_converter_live=False)))
    print("Rendered slides 37/38; all 36 earlier slide appearances and raw evidence images preserved.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("generate","verify","render"),required=True)
    args=parser.parse_args(); {"generate":generate,"verify":verify,"render":render}[args.stage]()
