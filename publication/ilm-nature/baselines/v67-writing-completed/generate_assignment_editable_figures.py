#!/usr/bin/env python3
"""Append native OOXML figures without rewriting the preserved deck parts.

The local Artifact Tool runtime is unavailable. This builder uses ZIP/XML,
native chart data/workbooks and unchanged source bitmaps, not slide screenshots.
"""

import copy
import io
import json
import posixpath
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import openpyxl

from generate_assignment_evidence import HERE, ROOT, completed_report


NS = dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    r="http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    c="http://schemas.openxmlformats.org/drawingml/2006/chart",
    pkg="http://schemas.openxmlformats.org/package/2006/relationships",
    ct="http://schemas.openxmlformats.org/package/2006/content-types")
for prefix in ("p", "a", "r", "c"):
    ET.register_namespace(prefix, NS[prefix])


def q(name):
    prefix, local = name.split(":")
    return "{"+NS[prefix]+"}"+local


def element(tag, parent=None, **attrs):
    node = ET.Element(q(tag), {k: str(v) for k, v in attrs.items()})
    if parent is not None:
        parent.append(node)
    return node


def xml(node):
    # Some office importers require OPC infrastructure elements to use the
    # default namespace even though a prefixed form is XML-equivalent.
    infrastructure = node.tag.startswith(("{"+NS["ct"]+"}", "{"+NS["pkg"]+"}"))
    if infrastructure:
        namespace = node.tag.split("}")[0][1:]
        ET.register_namespace("", namespace)
    return ET.tostring(node, encoding="utf-8", xml_declaration=True)


def relpath(part):
    return posixpath.join(posixpath.dirname(part), "_rels", posixpath.basename(part)+".rels")


def resolve(part, target):
    return target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(posixpath.dirname(part), target))


def px(value):
    return str(round(value*9525))


class Slide:
    def __init__(self, template):
        self.root = ET.fromstring(template)
        self.tree = self.root.find("p:cSld/p:spTree", NS)
        for child in list(self.tree)[2:]:
            self.tree.remove(child)
        self.id = 1
        self.rels = element("pkg:Relationships")

    def relation(self, kind, target):
        rid = f"rId{len(self.rels)+1}"
        element("pkg:Relationship", self.rels, Id=rid, Type=NS["r"]+"/"+kind, Target=target)
        return rid

    def shape(self, words, x, y, w, h, size=22, bold=False, center=False, fill=None):
        self.id += 1
        shape = element("p:sp", self.tree)
        nv = element("p:nvSpPr", shape)
        element("p:cNvPr", nv, id=self.id, name=f"Editable text {self.id}")
        element("p:cNvSpPr", nv, txBox="1")
        element("p:nvPr", nv)
        properties = element("p:spPr", shape)
        transform = element("a:xfrm", properties)
        element("a:off", transform, x=px(x), y=px(y))
        element("a:ext", transform, cx=px(w), cy=px(h))
        element("a:avLst", element("a:prstGeom", properties, prst="rect"))
        if fill:
            element("a:srgbClr", element("a:solidFill", properties), val=fill)
            element("a:srgbClr", element("a:solidFill", element("a:ln", properties, w="9525")), val="677B87")
        else:
            element("a:noFill", properties)
            element("a:noFill", element("a:ln", properties))
        body = element("p:txBody", shape)
        element("a:bodyPr", body, wrap="square", lIns=px(5), rIns=px(5), tIns=px(3), bIns=px(3), anchor="ctr" if fill else "t")
        element("a:lstStyle", body)
        for line in words.split("\n"):
            paragraph = element("a:p", body)
            element("a:pPr", paragraph, algn="ctr" if center else "l")
            run = element("a:r", paragraph)
            properties = element("a:rPr", run, lang="en-US", sz=round(size*100), b="1" if bold else "0")
            element("a:srgbClr", element("a:solidFill", properties), val="1D2B34")
            element("a:latin", properties, typeface="Arial")
            element("a:t", run).text = line
        return shape

    def arrow(self, x1, y1, x2, y2, head=True):
        self.id += 1
        shape = element("p:cxnSp", self.tree)
        nv = element("p:nvCxnSpPr", shape)
        element("p:cNvPr", nv, id=self.id, name=f"Editable connector {self.id}")
        element("p:cNvCxnSpPr", nv)
        element("p:nvPr", nv)
        properties = element("p:spPr", shape)
        transform = element("a:xfrm", properties, flipH="1" if x2 < x1 else "0", flipV="1" if y2 < y1 else "0")
        element("a:off", transform, x=px(min(x1, x2)), y=px(min(y1, y2)))
        element("a:ext", transform, cx=px(abs(x2-x1)), cy=px(abs(y2-y1)))
        element("a:avLst", element("a:prstGeom", properties, prst="line"))
        line = element("a:ln", properties, w="19050")
        element("a:srgbClr", element("a:solidFill", line), val="536674")
        if head:
            element("a:tailEnd", line, type="triangle", w="med", len="med")

    def picture(self, relationship, name, x, y, w, h):
        self.id += 1
        pic = element("p:pic", self.tree)
        nv = element("p:nvPicPr", pic)
        element("p:cNvPr", nv, id=self.id, name=name)
        element("a:picLocks", element("p:cNvPicPr", nv), noChangeAspect="1")
        element("p:nvPr", nv)
        fill = element("p:blipFill", pic)
        element("a:blip", fill).set(q("r:embed"), relationship)
        element("a:fillRect", element("a:stretch", fill))
        prop = element("p:spPr", pic)
        trans = element("a:xfrm", prop)
        element("a:off", trans, x=px(x), y=px(y))
        element("a:ext", trans, cx=px(w), cy=px(h))
        element("a:avLst", element("a:prstGeom", prop, prst="rect"))

    def chart(self, relationship, x, y, w, h):
        self.id += 1
        frame = element("p:graphicFrame", self.tree)
        nv = element("p:nvGraphicFramePr", frame)
        element("p:cNvPr", nv, id=self.id, name=f"Native evidence chart {self.id}")
        element("p:cNvGraphicFramePr", nv)
        element("p:nvPr", nv)
        trans = element("p:xfrm", frame)
        element("a:off", trans, x=px(x), y=px(y))
        element("a:ext", trans, cx=px(w), cy=px(h))
        data = element("a:graphicData", element("a:graphic", frame), uri=NS["c"])
        element("c:chart", data).set(q("r:id"), relationship)


def chart_part(template, categories, values, series_name, minimum, maximum, color, intervals=None):
    root = ET.fromstring(template)
    chart = root.find("c:chart", NS)
    legend = chart.find("c:legend", NS)
    if legend is not None:
        chart.remove(legend)
    bar = chart.find("c:plotArea/c:barChart", NS)
    old = bar.findall("c:ser", NS)
    index = list(bar).index(old[0])
    for item in old:
        bar.remove(item)
    series = element("c:ser")
    element("c:idx", series, val="0")
    element("c:order", series, val="0")
    text_ref = element("c:strRef", element("c:tx", series))
    element("c:f", text_ref).text = "Sheet1!$B$1"
    cache = element("c:strCache", text_ref)
    element("c:ptCount", cache, val="1")
    element("c:v", element("c:pt", cache, idx="0")).text = series_name
    element("a:srgbClr", element("a:solidFill", element("c:spPr", series)), val=color)
    element("c:invertIfNegative", series, val="0")
    if intervals:
        errors = element("c:errBars", series)
        element("c:errDir", errors, val="y")
        element("c:errBarType", errors, val="both")
        element("c:errValType", errors, val="cust")
        element("c:noEndCap", errors, val="0")
        for tag, column, numbers in (("plus", "E", [hi-v for v, (lo, hi) in zip(values, intervals)]),
                                     ("minus", "F", [v-lo for v, (lo, hi) in zip(values, intervals)])):
            ref = element("c:numRef", element("c:"+tag, errors))
            element("c:f", ref).text = f"Sheet1!${column}$2:${column}${len(values)+1}"
            cache = element("c:numCache", ref)
            element("c:formatCode", cache).text = "0.000"
            element("c:ptCount", cache, val=len(numbers))
            for i, value in enumerate(numbers):
                element("c:v", element("c:pt", cache, idx=i)).text = repr(value)
        error_style = element("c:spPr", errors)
        element("a:srgbClr", element("a:solidFill", element("a:ln", error_style, w="25400")), val="172A38")
    ref = element("c:strRef", element("c:cat", series))
    element("c:f", ref).text = f"Sheet1!$A$2:$A${len(categories)+1}"
    cache = element("c:strCache", ref)
    element("c:ptCount", cache, val=len(categories))
    for i, name in enumerate(categories):
        element("c:v", element("c:pt", cache, idx=i)).text = name
    ref = element("c:numRef", element("c:val", series))
    element("c:f", ref).text = f"Sheet1!$B$2:$B${len(values)+1}"
    cache = element("c:numCache", ref)
    element("c:formatCode", cache).text = "0.000"
    element("c:ptCount", cache, val=len(values))
    for i, value in enumerate(values):
        element("c:v", element("c:pt", cache, idx=i)).text = repr(value)
    bar.insert(index, series)
    for label in bar.findall("c:dLbls/c:numFmt", NS):
        label.set("formatCode", "0.000")
    for axis in chart.findall("c:plotArea/c:catAx", NS):
        tick = axis.find("c:tickLblPos", NS)
        if tick is not None:
            tick.set("val", "low")
    for axis in chart.findall("c:plotArea/c:valAx", NS):
        scaling = axis.find("c:scaling", NS)
        for tag, value in (("min", minimum), ("max", maximum)):
            node = scaling.find("c:"+tag, NS)
            if node is None:
                node = element("c:"+tag, scaling)
            node.set("val", str(value))
        unit = axis.find("c:majorUnit", NS)
        if unit is not None:
            unit.set("val", "1")
    for run in root.findall(".//a:defRPr", NS):
        run.set("sz", "1150")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(["Condition", series_name, "Lower", "Upper", "Plus", "Minus"])
    for i, (name, value) in enumerate(zip(categories, values)):
        lo, hi = intervals[i] if intervals else (value, value)
        sheet.append([name, value, lo, hi, hi-value, value-lo])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return root, buffer.getvalue()


def main():
    report = completed_report()
    baseline = HERE/"baselines/v62-transcript-bounds/ILM_Figures_editable.pptx"
    with zipfile.ZipFile(baseline) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slides = presentation.find("p:sldIdLst", NS)
    if len(slides) != 17:
        raise ValueError("preserved seventeen-slide deck required")
    size = presentation.find("p:sldSz", NS)
    if (int(size.get("cx")), int(size.get("cy"))) != (15240000, 8572500):
        raise ValueError("preserve actual reference dimensions")

    def override(part, kind):
        if not any(item.get("PartName") == "/"+part for item in types):
            element("ct:Override", types, PartName="/"+part, ContentType=kind)

    def presentation_relation(kind, target):
        used = {item.get("Id") for item in relationships}
        index = 1
        while f"rId{index}" in used:
            index += 1
        rid = f"rId{index}"
        element("pkg:Relationship", relationships, Id=rid, Type=NS["r"]+"/"+kind, Target=target)
        return rid

    method = HERE/"protocol-v63/question_assignment_method_editable.pptx"
    with zipfile.ZipFile(method) as archive:
        source = {name: archive.read(name) for name in archive.namelist()}
    source_types = ET.fromstring(source["[Content_Types].xml"])
    kinds = {item.get("PartName").lstrip("/"): item.get("ContentType")
        for item in source_types if item.tag == q("ct:Override")}
    mapping = {"ppt/slides/slide1.xml": "ppt/slides/slide18.xml"}
    imported = set()

    def import_part(old):
        if old in imported:
            return mapping[old]
        if old not in mapping:
            mapping[old] = posixpath.join(posixpath.dirname(old), "v63_"+posixpath.basename(old))
        new = mapping[old]
        imported.add(old)
        parts[new] = source[old]
        if old in kinds:
            override(new, kinds[old])
        old_rels = relpath(old)
        if old_rels in source:
            rels = ET.fromstring(source[old_rels])
            for item in rels:
                if item.get("TargetMode") == "External":
                    continue
                target = import_part(resolve(old, item.get("Target")))
                item.set("Target", posixpath.relpath(target, posixpath.dirname(new)))
            parts[relpath(new)] = xml(rels)
        return new

    import_part("ppt/slides/slide1.xml")
    for old in sorted(imported):
        kind = kinds.get(old, "")
        if kind.endswith("slideMaster+xml"):
            collection = presentation.find("p:sldMasterIdLst", NS)
            rid = presentation_relation("slideMaster", posixpath.relpath(mapping[old], "ppt"))
            node = element("p:sldMasterId", collection, id=max(int(x.get("id")) for x in collection)+1)
            node.set(q("r:id"), rid)
        if kind.endswith("notesMaster+xml"):
            collection = presentation.find("p:notesMasterIdLst", NS)
            if collection is None:
                collection = element("p:notesMasterIdLst")
                presentation.insert(1, collection)
            node = element("p:notesMasterId", collection)
            node.set(q("r:id"), presentation_relation("notesMaster", posixpath.relpath(mapping[old], "ppt")))
    node = element("p:sldId", slides, id=max(int(x.get("id")) for x in slides)+1)
    node.set(q("r:id"), presentation_relation("slide", "slides/slide18.xml"))

    def add_slide(number, slide, notes):
        part = f"ppt/slides/slide{number}.xml"
        source_rels = ET.fromstring(parts["ppt/slides/_rels/slide17.xml.rels"])
        layout = next(item.get("Target") for item in source_rels if item.get("Type").endswith("/slideLayout"))
        slide.relation("slideLayout", layout)
        slide.relation("notesSlide", f"../notesSlides/v63_notes{number}.xml")
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        notes_part = f"ppt/notesSlides/v63_notes{number}.xml"
        template_notes = next(resolve("ppt/slides/slide17.xml", x.get("Target")) for x in source_rels if x.get("Type").endswith("/notesSlide"))
        note = ET.fromstring(parts[template_notes])
        for shape in note.findall("p:cSld/p:spTree/p:sp", NS):
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if placeholder is not None and placeholder.get("type") == "body":
                texts = shape.findall(".//a:t", NS)
                for i, text in enumerate(texts):
                    text.text = notes if i == 0 else ""
        note_rels = ET.fromstring(parts[relpath(template_notes)])
        for item in note_rels:
            if item.get("Type").endswith("/slide"):
                item.set("Target", f"../slides/slide{number}.xml")
        parts[notes_part], parts[relpath(notes_part)] = xml(note), xml(note_rels)
        override(notes_part, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        node = element("p:sldId", slides, id=max(int(x.get("id")) for x in slides)+1)
        node.set(q("r:id"), presentation_relation("slide", f"slides/slide{number}.xml"))

    result = Slide(parts["ppt/slides/slide17.xml"])
    result.shape("Question assignment: completed free-writing test", 45, 25, 1510, 70, 31, True)
    result.shape("Strict-raster F1 (%)", 65, 110, 650, 40, 21, True)
    result.shape("Paired differences with 95% intervals (pp)", 830, 110, 710, 40, 21, True)
    routes = ("control", "control_blank", "aligned", "aligned_blank")
    categories = ("Control", "Control blank Q", "Aligned", "Aligned blank Q")
    values = [100*report["measurements"][r]["raster_f1"] for r in routes]
    contrasts = ("aligned_minus_control_raster_f1", "control_minus_control_blank_raster_f1", "aligned_minus_aligned_blank_raster_f1")
    means = [100*report["comparisons"][k]["mean"] for k in contrasts]
    intervals = [[100*v for v in report["comparisons"][k]["ci95"]] for k in contrasts]
    for number, cats, vals, title, low, high, color, ci, x, w in (
        (14, categories, values, "Raster F1 (%)", 0, 4.2, "28638B", None, 65, 650),
        (15, ("Aligned − control", "Control − blank Q", "Aligned − blank Q"), means, "Paired difference (pp)", -2, 4, "28638B", intervals, 825, 710)):
        root, workbook = chart_part(parts["ppt/charts/chart12.xml"], cats, vals, title, low, high, color, ci)
        chart_name = f"ppt/charts/chart{number}.xml"
        book_name = f"ppt/embeddings/v63_chart{number}.xlsx"
        data = root.find("c:externalData", NS)
        data.set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"]+"/package", Target=f"../embeddings/v63_chart{number}.xlsx")
        parts[chart_name], parts[relpath(chart_name)], parts[book_name] = xml(root), xml(rels), workbook
        override(chart_name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book_name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        result.chart(result.relation("chart", f"../charts/chart{number}.xml"), x, 150, w, 270)
    result.shape("Exact answers: control 2/256, aligned 1/256. Both-correct pairs: 0/393 in each arm.", 65, 425, 1470, 42, 21, True)
    result.shape("First two fixed primary answers", 65, 472, 1470, 35, 18, True)
    for i in range(2):
        for j, route in enumerate(("reference", "control", "aligned")):
            y = 510+(i*3+j)*58
            result.shape(f"{i+1}: {route.capitalize()}", 65, y+10, 160, 38, 17)
            media = f"ppt/media/v63_raw_{i}_{route}.png"
            parts[media] = (HERE/f"evidence/v63_raw/{i}_{route}.png").read_bytes()
            result.picture(result.relation("image", "../media/"+posixpath.basename(media)),
                f"Unrepaired audit row {i} {route}", 240, y, 1120, 56)
    result.shape("Unrepaired pixels and white padding. One seed, reused development and external pretraining. Both reading gates fail.", 45, 862, 1510, 34, 13)
    add_slide(19, result, "Sources: evidence/v63_development_report.json and assignment_source_hashes.json; Figure 8; Supplementary Note 22. Paired intervals resample 198 articles, 2000 draws, seed 20266329. All six writing images preserve original bits with white padding.")

    actor = Slide(parts["ppt/slides/slide17.xml"])
    actor.shape("Complete continuous image actor", 45, 25, 1510, 75, 32, True)
    actor.shape("7.548B total parameters     25.58M trainable     17.181 GiB peak allocated", 65, 115, 1470, 50, 23)
    boxes = ((65,210,300,110,"Prompt image\nCompleted blocks",False),
        (440,210,300,110,"27-layer vision encoder\nFrozen FP32 weights",False),
        (815,210,340,110,"36-layer contextual body\nRank-eight adapters",True),
        (1240,210,300,110,"Two continuous\nwriting queries",True),
        (815,485,340,120,"Six-layer flow writer\n19.37M parameters",True),
        (1240,485,300,120,"32 × 64\noutput pixels",False))
    for x,y,w,h,label,trainable in boxes:
        actor.shape(label,x,y,w,h,21,False,True,"EAF2F7" if trainable else "F2F4F5")
    for start,end,head in (((365,265),(440,265),True),((740,265),(815,265),True),((1155,265),(1240,265),True),
        ((1390,320),(1390,400),False),((1390,400),(985,400),False),((985,400),(985,485),True),((1155,545),(1240,545),True)):
        actor.arrow(*start,*end,head)
    actor.shape("Each completed block is a separate encoder image.\nCausal history excludes current and future answer pixels.\nGeneration rereads only completed output blocks.",65,485,660,140,21)
    actor.shape("Yt = (1 − t) U + t Y       U ~ N(0, I)       Lflow = E ||v(Yt,t,h) − (Y − U)||²",65,690,1470,70,25)
    actor.shape("Frozen contextual weights use BF16 storage with FP32 computation. A separate classifier predicts stopping.",65,790,1470,60,18)
    actor.shape("Execution checks pass. Writing and language quality remain unqualified.",65,850,1470,40,18,True)
    add_slide(20,actor,"Sources: evidence/grounded_actor_preflight.json, evidence/inherited_context_check.json; Figure S7 and Supplementary Note 23. This is an execution-only result. The two-update output is noise. Frozen inherited pretraining is external work; no source vocabulary or model service enters actor inference.")

    parts["ppt/presentation.xml"], parts["ppt/_rels/presentation.xml.rels"], parts["[Content_Types].xml"] = xml(presentation), xml(relationships), xml(types)
    if "docProps/app.xml" in parts:
        app = ET.fromstring(parts["docProps/app.xml"])
        for node in app.iter():
            if node.tag.endswith("}Slides"):
                node.text = "20"
        parts["docProps/app.xml"] = xml(app)
    target = HERE/"ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as output:
        for name,value in parts.items():
            output.writestr(name,value)
    print("Saved 20 native editable slides; preserved the first seventeen slide parts")


if __name__ == "__main__":
    main()
