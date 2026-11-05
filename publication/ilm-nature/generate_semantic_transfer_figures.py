#!/usr/bin/env python3
"""Append editable reader evidence and continuous-writing method diagrams."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE/"baselines/v72-visual-semantics/baseline_manifest.json"
READER = ROOT/"references/evidence/native_visual_transfer_v71_result.json"
EXECUTION = ROOT/"references/evidence/semantic_ilm_v72_execution.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    baseline = json.loads(BASELINE.read_text())
    revision = baseline["source_commit"]
    name = "publication/ilm-nature/ILM_Figures_editable.pptx"
    raw = subprocess.check_output(["git","show",revision+":"+name],cwd=ROOT)
    if hashlib.sha256(raw).hexdigest()!=baseline["files_sha256"]["ILM_Figures_editable.pptx"]:
        raise ValueError("preserved source deck changed")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        original = {name:archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst",NS)
    size = presentation.find("p:sldSz",NS)
    if len(slides)!=24 or (size.get("cx"),size.get("cy"))!=("15240000","8572500"):
        raise ValueError("preserve the actual 24-slide template")
    evidence = json.loads(READER.read_text())
    execution = json.loads(EXECUTION.read_text())
    if (not evidence["offline_checks_pass"] or evidence["helper_projection_argmax_agreements"]!=742
            or evidence["image_actor_generated_images"]!=0 or not execution["passed"]
            or execution["trainable_parameters"]!=38651649 or execution["language_quality_claim"]):
        raise ValueError("verified source evidence or scope changed")

    def override(name,kind):
        element("ct:Override",types,PartName="/"+name,ContentType=kind)

    def add_slide(number,slide,notes):
        source_part = "ppt/slides/slide24.xml"
        source_rels = ET.fromstring(parts[relpath(source_part)])
        layout = next(n.get("Target") for n in source_rels if n.get("Type").endswith("/slideLayout"))
        slide.relation("slideLayout",layout)
        note_part = f"ppt/notesSlides/semantic_transfer_notes{number}.xml"
        slide.relation("notesSlide","../notesSlides/"+Path(note_part).name)
        part = f"ppt/slides/slide{number}.xml"
        parts[part],parts[relpath(part)] = xml(slide.root),xml(slide.rels)
        override(part,"application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        source_note = next(resolve(source_part,n.get("Target")) for n in source_rels if n.get("Type").endswith("/notesSlide"))
        note = ET.fromstring(parts[source_note])
        for shape in note.findall("p:cSld/p:spTree/p:sp",NS):
            ph = shape.find("p:nvSpPr/p:nvPr/p:ph",NS)
            if ph is not None and ph.get("type")=="body":
                for i,text in enumerate(shape.findall(".//a:t",NS)):
                    text.text = notes if i==0 else ""
        rels = ET.fromstring(parts[relpath(source_note)])
        for item in rels:
            if item.get("Type").endswith("/slide"):
                item.set("Target",f"../slides/slide{number}.xml")
        parts[note_part],parts[relpath(note_part)] = xml(note),xml(rels)
        override(note_part,"application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        used = {n.get("Id") for n in relationships}
        index = 1
        while f"rId{index}" in used:
            index += 1
        rid = f"rId{index}"
        element("pkg:Relationship",relationships,Id=rid,Type=NS["r"]+"/slide",Target=f"slides/slide{number}.xml")
        node = element("p:sldId",slides,id=max(int(n.get("id")) for n in slides)+1)
        node.set(q("r:id"),rid)

    # One editable count chart, with its source-derived embedded workbook.
    routes = ["raw/normal","raw/blank","native_budget/normal","native_budget/blank","text"]
    counts = [evidence["summaries"][route]["exact"] for route in routes]
    if counts!=[1,0,4,0,3] or any(evidence["summaries"][r]["count"]!=8 for r in routes):
        raise ValueError("fixed eight-question comparison changed")
    categories = ["Original\nimage","Original\nQ erased","Native budget\nimage","Native budget\nQ erased","Text\nhelper"]
    chart,workbook = chart_part(parts["ppt/charts/chart12.xml"],categories,counts,"Exact answers / 8",0,8,"296486")
    for parent in chart.iter():
        for child in list(parent):
            if child.tag==q("c:title"):
                parent.remove(child)
    for unit in chart.findall(".//c:majorUnit",NS):
        unit.set("val","2")
    for run in chart.findall(".//a:defRPr",NS):
        run.set("sz","1900")
    for fmt in chart.findall(".//c:numFmt",NS)+chart.findall(".//c:formatCode",NS):
        if fmt.tag==q("c:formatCode"):
            fmt.text="0"
        else:
            fmt.set("formatCode","0")
            fmt.set("sourceLinked","0")
    for show in chart.findall(".//c:showVal",NS):
        show.set("val","1")
    bar=chart.find(".//c:barChart",NS)
    labels=element("c:dLbls")
    element("c:numFmt",labels,formatCode="0",sourceLinked="0")
    element("c:dLblPos",labels,val="outEnd")
    for tag,value in (("showLegendKey",0),("showVal",1),("showCatName",0),("showSerName",0),("showPercent",0),("showBubbleSize",0)):
        element("c:"+tag,labels,val=value)
    bar.insert(list(bar).index(bar.find("c:axId",NS)),labels)
    book_part="ppt/embeddings/semantic_transfer_chart21.xlsx"
    chart_part_name="ppt/charts/chart21.xml"
    chart.find("c:externalData",NS).set(q("r:id"),"rId1")
    rels=element("pkg:Relationships")
    element("pkg:Relationship",rels,Id="rId1",Type=NS["r"]+"/package",Target="../embeddings/"+Path(book_part).name)
    parts[chart_part_name],parts[relpath(chart_part_name)],parts[book_part]=xml(chart),xml(rels),workbook
    override(chart_part_name,"application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
    override(book_part,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    reading=Slide(parts["ppt/slides/slide24.xml"])
    reading.shape("Numeric image controls retain bounded reading",45,26,1510,58,31,True)
    reading.shape("a  Exact Chinese answers | the same eight reused questions",65,112,900,52,23,True)
    reading.chart(reading.relation("chart","../charts/chart21.xml"),45,180,930,505)
    reading.shape("Q erased: only question ink is removed, with wrap coordinates retained.",65,715,910,63,20)
    reading.shape("b  Diagnostic boundary",1040,112,510,52,23,True)
    reading.shape("Prompt pixels\nFixed numeric task frame",1055,190,460,95,22,False,True,"F2F4F5")
    reading.arrow(1285,285,1285,325)
    reading.shape("Imported vision + context\nVocabulary module absent",1055,325,460,100,22,False,True,"E8F1F7")
    reading.arrow(1285,425,1285,465)
    reading.shape("Continuous hidden states",1055,465,460,65,22,False,True,"F2F4F5")
    reading.arrow(1285,530,1285,570)
    reading.shape("Helper only\nSource symbolic text decoder",1055,570,460,100,22,False,True,"F8F1ED")
    reading.shape("Text decoding is excluded from the image actor.\nThis check generates no image answers.",1040,715,510,75,19)
    reading.shape("All 32 input images and 742 retained decoding choices verify. The complete neural forward pass is not independently replayed.",45,832,1510,45,17)
    add_slide(25,reading,"Figure S10. Source: references/evidence/native_visual_transfer_v71_result.json, verified receipt "+evidence["verification_sha256"]+". Eight fixed reused Chinese development questions, 40 helper generations. Exact counts in source order are 1,0,4,0,3 of eight. Observer scaling uses the declared Pillow bilinear factor-32 budget and adds no textual information. The separate helper uses a source vocabulary projection, tokenizer and token-row lookup for greedy symbolic decoding, none attached to the image actor. All 742 retained projection decisions replay as exact argmax; whole vision/context forward replay is not claimed. This small reused diagnostic is not population language accuracy or image-output success.")

    method=Slide(parts["ppt/slides/slide24.xml"])
    method.shape("Causal image writing with frozen visual semantics",45,26,1510,58,31,True)
    method.shape("a  Training | complete answer images supervise continuous writing",65,112,1460,48,23,True)
    boxes=[("Prompt image X",45,205),("Frozen image reader\n+ contextual body",300,260),
        ("Memory M\n52 × 4096 values",630,205),("Causal writer Tθ\n8 layers, width 512",905,260),
        ("Joint code flow\nvθ(zᵤ, u, hₜ)",1235,310)]
    for text,x,w in boxes:
        method.shape(text,x,190,w,100,22,False,True,"E8F1F7" if "writer" in text or "flow" in text else "F2F4F5")
    for a,b in [(250,300),(560,630),(835,905),(1165,1235)]:
        method.arrow(a,240,b,240)
    method.shape("Frozen writing encoder",45,355,370,75,21,False,True,"F2F4F5")
    method.shape("Complete answer-image stream",45,308,470,40,19)
    method.shape("Past writing codes only\nCausal mask excludes zₜ and later",610,355,505,75,20,False,True,"F2F4F5")
    method.arrow(415,392,610,392)
    method.arrow(1035,355,1035,290)
    method.shape("zᵤ = (1−u)e + u zₜ",1210,355,340,75,23,False,True,"F2F4F5")
    method.arrow(1380,355,1380,290)
    method.arrow(415,415,505,415,False)
    method.arrow(505,415,505,467,False)
    method.arrow(505,467,1380,467,False)
    method.arrow(1380,467,1380,430)
    method.shape("Detached target zₜ + Gaussian e",640,435,520,40,19)
    method.shape("L = mean ||vθ − (zₜ − e)||² + balanced stop loss",200,495,1200,48,25,False,True)
    method.shape("b  Inference | every generated block becomes physical visual history",65,575,1460,48,23,True)
    inference=[("Prompt-only\nmemory M",45,255),("Causal writer\nPrompt + reread images",365,325),
        ("Joint flow\n32 midpoint steps",755,270),("LayerNorm\nDecode + threshold",1090,245),("Output\n32 × 64 px",1380,170)]
    for text,x,w in inference:
        method.shape(text,x,650,w,95,19 if x==1090 else 21,False,True,"E8F1F7" if "writer" in text or "flow" in text else "F2F4F5")
    for a,b in [(300,365),(690,755),(1025,1090),(1335,1380)]:
        method.arrow(a,697,b,697)
    method.arrow(1465,745,1465,820,False)
    method.arrow(1465,820,525,820,False)
    method.arrow(525,820,525,745)
    method.shape("Re-encode actual emitted pixels",720,773,620,42,21)
    method.shape("Learned stop ≥ 0.5\nUniform cap: 128 blocks",65,775,400,65,19)
    method.shape("38.65M trainable writer parameters. External reader pretraining is retained. Full language training and audit are pending.",45,858,1510,36,16)
    add_slide(26,method,"Figure S11. Implemented V72 candidate, not a completed language result. Sources: references/semantic_ilm_v72_protocol.md; ilm/visual_lm/semantic_image_memory.py; ilm/visual_lm/semantic_writing_model.py; references/evidence/semantic_ilm_v72_execution.json. All 12,976 complete training examples are admitted; only two explicit white termination blocks are removed. Prompt memory is computed without answer labels. The frozen inherited reader supplies all 52 final task-frame states in BF16. The 38,651,649-parameter causal transformer and three-residual-block joint flow train with complete image-code histories and up to 16 sampled target losses per example. Production budget: 40,000 updates, batch 16. Inference has no tokenizer, OCR, language output head, persistent glyph inventory or teacher service. Generated code coordinates receive non-affine LayerNorm, are decoded and thresholded, then actual output pixels are re-encoded for feedback. The structural stop head is checked after every prefix, including after the uniform 128-block cap. Execution checks pass but full autonomous language quality remains unproven. Diagram contains no fabricated generated sample.")

    for name,node in (("[Content_Types].xml",types),("ppt/presentation.xml",presentation),("ppt/_rels/presentation.xml.rels",relationships)):
        parts[name]=xml(node)
    mutable={"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
    if any(parts[name]!=value for name,value in original.items() if name not in mutable):
        raise ValueError("a previous slide/chart/media part changed")
    target=HERE/"ILM_Figures_editable.pptx"
    temporary=target.with_suffix(".partial")
    with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,value in parts.items():
            archive.writestr(name,value)
    temporary.replace(target)
    sources={str(p.relative_to(ROOT)):sha(p) for p in [READER,EXECUTION,
        ROOT/"references/semantic_ilm_v72_protocol.md",ROOT/"ilm/visual_lm/semantic_writing_model.py",Path(__file__)]}
    receipt=dict(baseline_commit=revision,baseline_pptx_sha256=hashlib.sha256(raw).hexdigest(),
        pptx_sha256=sha(target),slides=26,new_slides=[25,26],original_parts_preserved=len(original)-3,
        chart=dict(part=chart_part_name,workbook=book_part,categories=categories,values=counts,denominator=8),
        sources_sha256=sources,artifact_tool_available=False,
        implementation="native OOXML append; editable shapes, connectors and chart workbook",
        existing_slides_preserved=True,language_quality_claim=False,method_status="candidate under training")
    (HERE/"evidence/semantic_transfer_deck_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Appended native slides 25-26; retained all preceding slide/chart/media parts.")


if __name__=="__main__":
    main()
