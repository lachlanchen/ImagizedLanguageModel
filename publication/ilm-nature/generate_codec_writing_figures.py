#!/usr/bin/env python3
"""Append source-bound native writing figures without changing old deck parts.

The local Artifact Tool package is unavailable. This uses the established
native OOXML builder, editable charts/workbooks and unrepaired source PNGs.
"""

import io
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
import zipfile

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / "baselines/v63-completed-actor"


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipts = {}
    sources = {}
    for name in ("writer_factorial_v65_result", "direct_writer_v66_result", "codec_output_v67_result",
            "codec_visual_actor_preflight_result", "continuous_writing_geometry_result"):
        path = ROOT / "references/evidence" / (name + ".json")
        receipts[name] = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = sha(path)
        (HERE / "evidence" / (name + ".json")).write_bytes(path.read_bytes())
    fitting, direct, codec = (receipts[x] for x in
        ("writer_factorial_v65_result", "direct_writer_v66_result", "codec_output_v67_result"))
    if direct["writing_component_pass"] or not codec["output_interface_pass"]:
        raise ValueError("preserve the actual failed-direct / qualified-codec decisions")
    with zipfile.ZipFile(BASELINE / "ILM_Figures_editable.pptx") as archive:
        original = {name:archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slide_ids = presentation.find("p:sldIdLst", NS)
    if len(slide_ids) != 20:
        raise ValueError("the complete twenty-slide baseline is required")
    size = presentation.find("p:sldSz", NS)
    if (size.get("cx"), size.get("cy")) != ("15240000", "8572500"):
        raise ValueError("preserve reference canvas dimensions")

    def override(name, kind):
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def add_slide(number, slide, note_text):
        old_part = "ppt/slides/slide20.xml"
        old_rels = ET.fromstring(parts[relpath(old_part)])
        layout = next(x.get("Target") for x in old_rels if x.get("Type").endswith("/slideLayout"))
        slide.relation("slideLayout", layout)
        note_part = f"ppt/notesSlides/writing_notes{number}.xml"
        slide.relation("notesSlide", "../notesSlides/" + Path(note_part).name)
        part = f"ppt/slides/slide{number}.xml"
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        override(part, "application/vnd.openxmlformats-officedocument.presentationml.slide+xml")
        source_note = next(resolve(old_part, x.get("Target")) for x in old_rels if x.get("Type").endswith("/notesSlide"))
        note = ET.fromstring(parts[source_note])
        for shape in note.findall("p:cSld/p:spTree/p:sp", NS):
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            if placeholder is not None and placeholder.get("type") == "body":
                for index, text in enumerate(shape.findall(".//a:t", NS)):
                    text.text = note_text if index == 0 else ""
        note_rels = ET.fromstring(parts[relpath(source_note)])
        for item in note_rels:
            if item.get("Type").endswith("/slide"):
                item.set("Target", f"../slides/slide{number}.xml")
        parts[note_part], parts[relpath(note_part)] = xml(note), xml(note_rels)
        override(note_part, "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml")
        used = {x.get("Id") for x in relationships}
        index = 1
        while f"rId{index}" in used:
            index += 1
        rid = f"rId{index}"
        element("pkg:Relationship", relationships, Id=rid, Type=NS["r"] + "/slide", Target=f"slides/slide{number}.xml")
        node = element("p:sldId", slide_ids, id=max(int(x.get("id")) for x in slide_ids) + 1)
        node.set(q("r:id"), rid)

    chart_data = []
    next_chart = 1 + max(int(re.fullmatch(r"ppt/charts/chart(\d+)\.xml", name)[1])
        for name in parts if re.fullmatch(r"ppt/charts/chart(\d+)\.xml", name))

    def chart(slide, categories, values, x, colors):
        number = next_chart + len(chart_data)
        root, workbook = chart_part(parts["ppt/charts/chart12.xml"], categories, values,
            "Foreground F1 (%)", 0, 110, colors[0])
        for parent in root.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        for unit in root.findall(".//c:majorUnit", NS):
            unit.set("val", "20")
        for axis_format in root.findall(".//c:valAx/c:numFmt", NS):
            axis_format.set("formatCode", "0")
        labels = root.find(".//c:barChart/c:dLbls", NS)
        if labels is not None:
            show = labels.find("c:showVal", NS)
            if show is not None:
                show.set("val", "1")
        for run in root.findall(".//a:defRPr", NS):
            run.set("sz", "1800")
        series = root.find(".//c:barChart/c:ser", NS)
        insertion = list(series).index(series.find("c:cat", NS))
        for index, color in enumerate(colors):
            point = element("c:dPt")
            element("c:idx", point, val=index)
            element("a:srgbClr", element("a:solidFill", element("c:spPr", point)), val=color)
            series.insert(insertion + index, point)
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/writing_chart{number}.xlsx"
        root.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package",
            Target="../embeddings/" + Path(book).name)
        parts[name], parts[relpath(name)], parts[book] = xml(root), xml(rels), workbook
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        slide.chart(slide.relation("chart", "../charts/" + Path(name).name), x, 155, 700, 290)
        chart_data.append(dict(part=name, workbook=book, categories=categories, values=values))

    raw = []

    def picture(slide, source, receipt, relative, destination, x, y):
        if sha(source) != receipt["files_sha256"][relative]:
            raise ValueError("verified raw writing image changed")
        sources[str(source.relative_to(ROOT))] = sha(source)
        target = HERE / "evidence/writing_raw" / destination
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(source.read_bytes())
        media = "ppt/media/writing_" + destination
        parts[media] = source.read_bytes()
        slide.picture(slide.relation("image", "../media/" + Path(media).name),
            "Unrepaired " + destination, x, y, 128, 64)
        raw.append(dict(slide=21, media=media, evidence=str(target.relative_to(HERE)),
            source=str(source.relative_to(ROOT)), sha256=sha(source), x=x, y=y))

    evidence = Slide(parts["ppt/slides/slide20.xml"])
    evidence.shape("Writing fidelity depends on the output representation", 45, 25, 1510, 65, 31, True)
    evidence.shape("a   Training F1 (%) | n=64", 65, 110, 710, 42, 23, True)
    evidence.shape("b   Development F1 (%) | n=128", 835, 110, 710, 42, 23, True)
    arms = ("inherited_flow", "inherited_direct", "pixels_flow", "pixels_direct")
    categories = ["Reader\nflow", "Reader\ndirect", "Pixels\nflow", "Pixels\ndirect"]
    chart(evidence, categories, [100 * fitting["arms"][a]["measurements"]["aligned"]["foreground_f1"] for a in arms],
        65, ["939B9F", "28638B", "939B9F", "28638B"])
    chart(evidence, ["Direct\naligned", "Direct\nshuffled", "Codec\naligned", "Codec\nshuffled"],
        [100 * item["measurements"][route]["foreground_f1"] for item, route in
            ((direct, "aligned"), (direct, "shuffled"), (codec, "aligned"), (codec, "shuffled"))],
        835, ["AE593F", "D7B7AA", "28638B", "9FBFCF"])
    evidence.shape("Exact blocks: 0/64, 63/64, 0/64, 64/64", 85, 440, 700, 36, 18)
    evidence.shape("Exact aligned: direct 3/128; fixed codec 126/128", 850, 440, 690, 36, 18)
    evidence.shape("c   First two fitting rows", 65, 495, 770, 40, 22, True)
    evidence.shape("d   First two development rows", 950, 495, 620, 40, 22, True)
    labels = ("Reference", "Reader flow", "Reader direct", "Pixels flow", "Pixels direct")
    for col, label in enumerate(labels):
        evidence.shape(label, 150 + 146 * col, 545, 145, 32, 15, center=True)
    for col, label in enumerate(("Reference", "Direct", "Codec")):
        evidence.shape(label, 1010 + 174 * col, 545, 165, 32, 16, center=True)
    for row in range(2):
        y = 590 + row * 105
        evidence.shape(str(row + 1), 90, y + 15, 45, 35, 18)
        evidence.shape(str(row + 1), 955, y + 15, 45, 35, 18)
        for col, arm in enumerate(("reference", *arms)):
            relative = f"reference/{row:03}.png" if arm == "reference" else f"{arm}/aligned/{row:03}.png"
            picture(evidence, ROOT / "artifacts/writer_factorial_v65_20261105" / relative,
                fitting, relative, f"fit_{row}_{arm}.png", 160 + col * 146, y)
        for col, route in enumerate(("reference", "direct", "codec")):
            receipt = direct if route == "direct" else codec
            folder = "direct_writer_learning_v66_20261105" if route == "direct" else "codec_output_v67_20261105"
            relative = f"development/aligned/{row:03}.png" if route == "direct" else f"{('reference' if route == 'reference' else 'aligned')}/{row:03}.png"
            picture(evidence, ROOT / "artifacts" / folder / relative, receipt, relative,
                f"dev_{row}_{route}.png", 1030 + col * 174, y)
    evidence.shape("Fitting uses training images. Development compares supplied representations with unequal prior training; neither is question answering.",
        45, 835, 1510, 45, 15)
    add_slide(21, evidence, "Sources: evidence/writer_factorial_v65_result.json, direct_writer_v66_result.json and codec_output_v67_result.json. Figure 9; Supplementary Notes 24–25. All 16 raw PNGs preserve source bytes. The same 128 development cases use different training histories; no causal efficiency comparison is claimed.")

    actor = Slide(parts["ppt/slides/slide20.xml"])
    actor.shape("Reading and writing use separate continuous spaces", 45, 25, 1510, 65, 31, True)
    actor.shape("7.539B total parameters     9.352M trainable     16.953 GiB peak allocated", 65, 110, 1470, 45, 23)
    boxes = ((50, 210, "Prompt pixels\nGenerated history", False),
        (295, 210, "27-layer reader\nFrozen weights", False),
        (540, 250, "36-layer context\nAdapters + 2 queries", True),
        (825, 210, "4096 → 768\nTrainable projection", True),
        (1070, 210, "Fixed writing\ndecoder", False),
        (1315, 230, "32 × 64\noutput pixels", False))
    for x, width, label, trainable in boxes:
        actor.shape(label, x, 225, width, 115, 21, center=True,
            fill="EAF2F7" if trainable else "F2F4F5")
    for left, right in zip(boxes, boxes[1:]):
        actor.arrow(left[0] + left[1], 282, right[0], 282)
    for start, end, head in (((1430, 225), (1430, 195), False), ((1430, 195), (155, 195), False),
            ((155, 195), (155, 225), True)):
        actor.arrow(*start, *end, head)
    actor.shape("Generated images become the next history", 400, 158, 800, 28, 16, center=True)
    for x, width, label in ((65, 250, "Target image Y\nTraining only"), (390, 270, "Fixed code encoder\n2 × 768 values"),
            (740, 280, "Detached target z*\nNo label path at inference")):
        actor.shape(label, x, 500, width, 110, 20, center=True, fill="F2F4F5")
    actor.arrow(315, 555, 390, 555)
    actor.arrow(660, 555, 740, 555)
    actor.shape("Code + image losses\nLearned structural stop", 1130, 500, 360, 110, 21, center=True, fill="EAF2F7")
    actor.arrow(1020, 555, 1130, 555)
    for start, end, head in (((930, 340), (930, 450), False), ((930, 450), (1310, 450), False),
            ((1310, 450), (1310, 500), True)):
        actor.arrow(*start, *end, head)
        line = actor.tree[-1].find(".//a:ln", NS)
        element("a:prstDash", line, val="dash")
    actor.shape("Predicted code and image", 960, 405, 570, 35, 18, center=True)
    actor.shape("L = MSE(N(ẑ), N(z*)) + MSE(D(ẑ), Y) + Lstop", 65, 675, 1460, 65, 27, center=True)
    actor.shape("Frozen decoder weights still pass gradients into the predicted codes and contextual adapters.", 65, 755, 1470, 45, 20)
    actor.shape("All 24 complete-model execution checks pass. Two-update output is fragmented; autonomous language remains unqualified.",
        45, 835, 1510, 45, 15)
    add_slide(22, actor, "Sources: evidence/codec_visual_actor_preflight_result.json; Figure S8; Supplementary Note 26. GPU-0 peak allocated 16.95316457748413 GiB. All 27 vision and 36 contextual layers remain present. The fixed codec encoder supplies detached training labels only; generation receives prompt pixels and generated history. This figure does not display a trained-language result.")
    parts["[Content_Types].xml"], parts["ppt/presentation.xml"], parts["ppt/_rels/presentation.xml.rels"] = xml(types), xml(presentation), xml(relationships)
    changed_allowed = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    for name, value in original.items():
        if name not in changed_allowed and parts[name] != value:
            raise ValueError("an existing deck part changed")
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    (HERE / "evidence/codec_writing_deck_build.json").write_text(json.dumps(dict(
        baseline_sha256=sha(BASELINE / target.name), pptx_sha256=sha(target),
        original_parts_preserved=len(original) - 3, slides=22, new_slides=[21, 22],
        chart_data=chart_data, raw_images=raw, source_sha256=sources,
        artifact_tool_available=False, implementation="native OOXML append; no slide screenshots"), indent=2) + "\n")
    print("Appended two native editable figures; preserved all original slide/chart parts")


if __name__ == "__main__":
    main()
