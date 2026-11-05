#!/usr/bin/env python3
"""Append native V68 result figures while preserving all existing deck parts."""

import copy
import hashlib
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import openpyxl
from PIL import Image
import torch

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / "baselines/v67-writing-completed"
RUN = ROOT / "artifacts/codec_language_v68_20261105"
DIAG = ROOT / "artifacts/codec_language_v68_diagnosis_20261105"
PINS = {
    "codec_language_v68_autonomous_result": "ca09f2fb88374406cc2ddfc8e44637776ddbea2fc5c18240ebce83b8df9c026c",
    "codec_language_v68_diagnosis_result": "69446b336ee39016758ad9229bbb5d48788dfcf931a44303fa69e83275c2b461",
    "codec_language_v68_training_integrity": "c62957f679820f882ba9bb8ea2fd4ae01c2aa0a3feda9d98e92617a97630d05b",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if torch.cuda.is_initialized():
        raise ValueError("publication evidence preparation uses CPU only")
    receipts, sources = {}, {}
    for name, digest in PINS.items():
        path = ROOT / "references/evidence" / (name + ".json")
        if sha(path) != digest:
            raise ValueError("source-bound evidence changed")
        receipts[name] = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = digest
        (HERE / "evidence" / path.name).write_bytes(path.read_bytes())
    audit = receipts["codec_language_v68_autonomous_result"]
    diagnostic = receipts["codec_language_v68_diagnosis_result"]
    if (audit["bounded_chinese_development_pass"] or not audit["all_verification_checks_pass"]
            or not diagnostic["offline_physical_checks_pass"] or diagnostic["raw_cases_verified"] != 32):
        raise ValueError("retain the verified failed-language decision")
    raw_report = json.loads((RUN / "development/report.json").read_text())
    diag_report = json.loads((DIAG / "generation_report.json").read_text())
    if sha(RUN / "development/report.json") != audit["report_sha256"] or sha(DIAG / "generation_report.json") != diagnostic["generation_report_sha256"]:
        raise ValueError("actual source reports differ from their receipts")
    sources[str((RUN / "development/report.json").relative_to(ROOT))] = sha(RUN / "development/report.json")
    sources[str((DIAG / "generation_report.json").relative_to(ROOT))] = sha(DIAG / "generation_report.json")
    prepared = torch.load(DIAG / "prepared.pt", map_location="cpu", weights_only=True)
    with zipfile.ZipFile(BASELINE / "ILM_Figures_editable.pptx") as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    types = ET.fromstring(parts["[Content_Types].xml"])
    presentation = ET.fromstring(parts["ppt/presentation.xml"])
    relationships = ET.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    slide_ids = presentation.find("p:sldIdLst", NS)
    if len(slide_ids) != 22:
        raise ValueError("the verified twenty-two-slide baseline is required")
    size = presentation.find("p:sldSz", NS)
    if (size.get("cx"), size.get("cy")) != ("15240000", "8572500"):
        raise ValueError("preserve template dimensions")

    def override(name, kind):
        element("ct:Override", types, PartName="/" + name, ContentType=kind)

    def add_slide(number, slide, notes):
        old_part = "ppt/slides/slide22.xml"
        old_rels = ET.fromstring(parts[relpath(old_part)])
        slide.relation("slideLayout", next(x.get("Target") for x in old_rels if x.get("Type").endswith("/slideLayout")))
        note_part = f"ppt/notesSlides/codec_language_notes{number}.xml"
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
                    text.text = notes if index == 0 else ""
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

    chart_data, raw = [], []
    next_chart = 1 + max(int(re.fullmatch(r"ppt/charts/chart(\d+)\.xml", name)[1])
        for name in parts if re.fullmatch(r"ppt/charts/chart(\d+)\.xml", name))

    def chart(slide, categories, series_values, series_names, colors, x, y, w, h, point_colors=None):
        number = next_chart + len(chart_data)
        root, _ = chart_part(parts["ppt/charts/chart12.xml"], categories, series_values[0], series_names[0], 0, 110, colors[0])
        for parent in root.iter():
            for child in list(parent):
                if child.tag == q("c:title"):
                    parent.remove(child)
        bar = root.find(".//c:barChart", NS)
        original_series = bar.find("c:ser", NS)
        position = list(bar).index(original_series)
        bar.remove(original_series)
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(["Condition", *series_names])
        for row, category in enumerate(categories):
            sheet.append([category, *[values[row] for values in series_values]])
        for index, (values, title, color) in enumerate(zip(series_values, series_names, colors)):
            series = copy.deepcopy(original_series)
            column = openpyxl.utils.get_column_letter(index + 2)
            series.find("c:idx", NS).set("val", str(index))
            series.find("c:order", NS).set("val", str(index))
            series.find("c:tx/c:strRef/c:f", NS).text = f"Sheet1!${column}$1"
            series.find("c:tx/c:strRef/c:strCache/c:pt/c:v", NS).text = title
            series.find("c:spPr/a:solidFill/a:srgbClr", NS).set("val", color)
            series.find("c:val/c:numRef/c:f", NS).text = f"Sheet1!${column}$2:${column}${len(values)+1}"
            cache = series.find("c:val/c:numRef/c:numCache", NS)
            for item in list(cache):
                if item.tag == q("c:pt"):
                    cache.remove(item)
            for i, value in enumerate(values):
                element("c:v", element("c:pt", cache, idx=i)).text = repr(value)
            if point_colors:
                insertion = list(series).index(series.find("c:cat", NS))
                for i, color in enumerate(point_colors):
                    point = element("c:dPt")
                    element("c:idx", point, val=i)
                    element("a:srgbClr", element("a:solidFill", element("c:spPr", point)), val=color)
                    series.insert(insertion + i, point)
            bar.insert(position + index, series)
        for tag, value in (("grouping", "clustered"), ("overlap", "0"), ("gapWidth", "120")):
            node = bar.find("c:" + tag, NS)
            if node is not None:
                node.set("val", value)
        if len(series_values) > 1:
            chart_root = root.find("c:chart", NS)
            legend = element("c:legend")
            element("c:legendPos", legend, val="b")
            element("c:layout", legend)
            element("c:overlay", legend, val="0")
            text = element("c:txPr", legend)
            element("a:bodyPr", text)
            element("a:lstStyle", text)
            properties = element("a:defRPr", element("a:pPr", element("a:p", text)), sz="1600")
            element("a:latin", properties, typeface="Arial")
            chart_root.insert(list(chart_root).index(chart_root.find("c:plotArea", NS)) + 1, legend)
        for unit in root.findall(".//c:majorUnit", NS):
            unit.set("val", "20")
        for axis_format in root.findall(".//c:valAx/c:numFmt", NS):
            axis_format.set("formatCode", "0")
            axis_format.set("sourceLinked", "0")
        for cache_format in root.findall(".//c:ser/c:val/c:numRef/c:numCache/c:formatCode", NS):
            cache_format.text = "0.0"
        labels = root.find(".//c:barChart/c:dLbls", NS)
        if labels is not None:
            show = labels.find("c:showVal", NS)
            if show is not None:
                show.set("val", "1")
            for fmt in labels.findall("c:numFmt", NS):
                fmt.set("formatCode", "0.0")
                fmt.set("sourceLinked", "0")
        for run in root.findall(".//a:defRPr", NS):
            run.set("sz", "1600" if len(series_values) > 1 else "1800")
        name = f"ppt/charts/chart{number}.xml"
        book = f"ppt/embeddings/codec_language_chart{number}.xlsx"
        root.find("c:externalData", NS).set(q("r:id"), "rId1")
        rels = element("pkg:Relationships")
        element("pkg:Relationship", rels, Id="rId1", Type=NS["r"] + "/package", Target="../embeddings/" + Path(book).name)
        stream = io.BytesIO()
        workbook.save(stream)
        # Excel's serializer rounds binary floats. Match its actual decimal
        # cells exactly in the native cache, retaining full receipt precision
        # in chart_data and testing the round-trip numerical error separately.
        workbook_bytes = stream.getvalue()
        with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as saved:
            sheet_xml = ET.fromstring(saved.read("xl/worksheets/sheet1.xml"))
        worksheet_ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        stored = {cell.get("r"): cell.find("s:v", worksheet_ns).text
            for cell in sheet_xml.findall(".//s:c", worksheet_ns)
            if cell.find("s:v", worksheet_ns) is not None}
        for i, series in enumerate(bar.findall("c:ser", NS)):
            column = openpyxl.utils.get_column_letter(i + 2)
            for j, value in enumerate(series.findall("c:val/c:numRef/c:numCache/c:pt/c:v", NS)):
                value.text = stored[f"{column}{j+2}"]
        parts[name], parts[relpath(name)], parts[book] = xml(root), xml(rels), workbook_bytes
        override(name, "application/vnd.openxmlformats-officedocument.drawingml.chart+xml")
        override(book, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        slide.chart(slide.relation("chart", "../charts/" + Path(name).name), x, y, w, h)
        chart_data.append(dict(part=name, workbook=book, categories=categories, series_names=series_names, values=series_values))

    def picture(slide, source, destination, x, y, scale, slide_number, crop=None):
        array = np.asarray(Image.open(source).convert("L"))
        if crop:
            left, top, right, bottom = crop
            array = array[top:bottom, left:right]
        target = HERE / "evidence/language_raw" / destination
        target.parent.mkdir(exist_ok=True)
        if crop:
            Image.fromarray(array).save(target)
        else:
            target.write_bytes(source.read_bytes())
        media = "ppt/media/codec_language_" + destination
        parts[media] = target.read_bytes()
        height, width = array.shape
        slide.picture(slide.relation("image", "../media/" + Path(media).name),
            "Measured writing " + destination, x, y, width * scale, height * scale)
        sources[str(source.relative_to(ROOT))] = sha(source)
        raw.append(dict(slide=slide_number, media=media, evidence=str(target.relative_to(HERE)),
            source=str(source.relative_to(ROOT)), source_sha256=sha(source), crop=crop,
            visible_pixel_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            pixel_shape=list(array.shape), x=x, y=y, scale=scale))

    measurement = audit["independently_recomputed_measurements"]["cmrc"]
    question = audit["independently_recomputed_question_effect"]
    evidence = Slide(parts["ppt/slides/slide22.xml"])
    evidence.shape("Question effects persist when accurate writing fails", 45, 25, 1510, 65, 31, True)
    evidence.shape("a   Chinese foreground F1 (%) | n=128", 65, 110, 710, 42, 23, True)
    chart(evidence, ["Full prompt", "Blank question"],
        [[100 * measurement[route]["foreground_f1"] for route in ("normal", "blank")]],
        ["Foreground F1 (%)"], ["28638B"], 65, 155, 700, 290, ["28638B", "AABFCB"])
    evidence.shape(f"Paired gain: {100*question['mean']:.2f} points [{100*question['ci95'][0]:.2f}, {100*question['ci95'][1]:.2f}]", 80, 445, 710, 40, 18)
    evidence.shape("b   Exact answers and learned stopping (%)", 835, 110, 710, 42, 23, True)
    chart(evidence, ["Exact\nanswers", "Both correct\npairs", "Learned\nstop"],
        [[100 * measurement["normal"]["exact"], 100 * audit["paired_both_correct"], 100 * measurement["normal"]["stopped"]]],
        ["Rate (%)"], ["28638B"], 835, 155, 700, 290, ["AE593F", "AE593F", "939B9F"])
    evidence.shape("0/128 exact answers; 0/64 complete pairs; 128/128 stops", 845, 445, 710, 40, 18)
    evidence.shape("c   First two fixed Chinese cases | complete, unmodified streams", 65, 505, 1470, 40, 22, True)
    for text, x in (("Reference", 185), ("Full prompt", 610), ("Blank question", 1015)):
        evidence.shape(text, x, 555, 400, 35, 18)
    for line, index in enumerate((16, 17)):
        row = prepared[index]
        if (row["id"], row["source"]) != (("DEV_477_QUERY_0", "DEV_477_QUERY_1")[line], "cmrc"):
            raise ValueError("show only the first two originally fixed cases")
        y = 615 + line * 95
        evidence.shape(str(line + 1), 90, y + 5, 45, 35, 18)
        reference = DIAG / f"case_{index:02}_reference.png"
        actual = np.asarray(Image.open(reference).convert("L"))
        expected = (row["reference"][0, 0].numpy() >= 0).astype(np.uint8) * 255
        if not np.array_equal(actual, expected):
            raise ValueError("published reference does not match measured source pixels")
        picture(evidence, reference, f"audit_{line}_reference.png", 190, y, 1.25, 23)
        for route, x in (("normal", 615), ("blank", 1020)):
            relative = f"predictions/{route}/cmrc_{row['id']}.png"
            source = RUN / "development" / relative
            if sha(source) != audit["raw_files_sha256"][relative]:
                raise ValueError("unmodified autonomous PNG changed")
            picture(evidence, source, f"audit_{line}_{route}.png", x, y, 1.25, 23)
    evidence.shape("All 384 generations and original gates are verified. Question dependence and stopping pass; answer quality fails.", 45, 835, 1510, 45, 15)
    add_slide(23, evidence, "Figure 10. Source: evidence/codec_language_v68_autonomous_result.json. All 384 raw generations and 192 original first references are verified. Chinese primary n=128, paired articles n=64, 2,000 article bootstrap draws. Three answer-quality gates fail; matched question gain and learned stopping pass. All six images are the complete first two fixed cases, with aspect ratio preserved and no output repair. External foundation pretraining is inherited.")

    diagnosis = Slide(parts["ppt/slides/slide22.xml"])
    diagnosis.shape("Prediction fails before generated-history drift", 45, 25, 1510, 65, 31, True)
    diagnosis.shape("a   Foreground F1 (%) | eight fixed cases per group", 65, 110, 1470, 42, 23, True)
    order = ("train/cmrc", "train/dolly", "development/cmrc", "development/dolly")
    chart(diagnosis, ["Training\nChinese", "Training\nEnglish", "Development\nChinese", "Development\nEnglish"],
        [[100 * diagnostic["groups"][group][key] for group in order]
            for key in ("supplied_code_f1", "exact_history_f1", "autonomous_f1")],
        ["Supplied codes", "Exact history", "Autonomous"], ["28638B", "AE593F", "939B9F"], 65, 155, 1470, 330)
    diagnosis.shape("Exact history supplies previous images and answer length; it is an assisted diagnostic.", 80, 490, 1460, 40, 19)
    diagnosis.shape("b   First 32-by-64 block, before generated history", 65, 550, 1470, 40, 22, True)
    for text, x in (("Reference", 555), ("Supplied-code reconstruction", 885), ("Initial prediction", 1270)):
        diagnosis.shape(text, x - 25, 600, min(360, 1580 - (x - 25)), 35, 17)
    for line, index in enumerate((0, 16)):
        y = 650 + line * 90
        diagnosis.shape("Training first case" if line == 0 else "Development first case", 90, y + 15, 390, 40, 19)
        for name, x in (("reference", 575), ("supplied", 945), ("exact_history", 1320)):
            source = DIAG / f"case_{index:02}_{name}.png"
            picture(diagnosis, source, f"diagnosis_{line}_{name}_first_block.png", x, y, 2., 24, [0, 0, 64, 32])
    diagnosis.shape("No first predicted block is exact. All 32 initial predictions match the autonomous path; no diagnostic weights update.", 45, 835, 1510, 45, 15)
    add_slide(24, diagnosis, "Figure S9. Source: evidence/codec_language_v68_diagnosis_result.json; registered 32 cases, eight per split/source. Exact-history writing is assisted with correct preceding images and known length, including forced continuation after a predicted stop. The 16 development autonomous streams are reused from the original audit. The lower panel explicitly shows only the first 64-pixel block from the first fixed training and development Chinese cases; complete measurements remain unchanged. Offline physical checks pass; native neural forward replay is not claimed. No first block is exact and conditional prediction fails even on training examples.")

    for name, node in (("[Content_Types].xml", types), ("ppt/presentation.xml", presentation), ("ppt/_rels/presentation.xml.rels", relationships)):
        parts[name] = xml(node)
    allowed = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    if any(parts[name] != value for name, value in original.items() if name not in allowed):
        raise ValueError("a previous deck part changed")
    target = HERE / "ILM_Figures_editable.pptx"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    values = {
        "CodecLanguageFOne": 100 * measurement["normal"]["foreground_f1"],
        "CodecLanguageBlankFOne": 100 * measurement["blank"]["foreground_f1"],
        "CodecLanguageQuestionGain": 100 * question["mean"],
        "CodecLanguageCILow": 100 * question["ci95"][0],
        "CodecLanguageCIHigh": 100 * question["ci95"][1],
    }
    (HERE / "codec_language_values.tex").write_text("% Generated from the fixed autonomous verification receipt.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + f"{value:.3f}" + "}" for name, value in values.items()) + "\n")
    (HERE / "codec_language_diagnosis_table.tex").write_text(
        "% Generated from the fixed diagnostic measurement receipt.\n" +
        "\\begin{tabular}{lrrrr}\n\\toprule\n" +
        "Group & Supplied codes & Exact history & Autonomous & Code MSE" + r" \\" + "\n\\midrule\n" +
        "\n".join(label + " & " + " & ".join(f"{100*diagnostic['groups'][group][key]:.3f}"
            for key in ("supplied_code_f1", "exact_history_f1", "autonomous_f1")) +
            f" & {diagnostic['groups'][group]['normalized_code_mse']:.4f}" + r" \\"
            for group, label in zip(order, ("Training, Chinese", "Training, English", "Development, Chinese", "Development, English"))) +
        "\n\\bottomrule\n\\end{tabular}\n")
    sources[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    (HERE / "evidence/codec_language_deck_build.json").write_text(json.dumps(dict(
        baseline_sha256=sha(BASELINE / target.name), pptx_sha256=sha(target),
        original_parts_preserved=len(original) - 3, slides=24, new_slides=[23, 24],
        chart_data=chart_data, raw_images=raw, source_sha256=sources,
        artifact_tool_available=False, implementation="native OOXML append with editable charts/workbooks; no slide screenshots",
        output_repair=False, language_quality_claim=False), indent=2) + "\n")
    print("Appended two native editable figures; preserved all 22 previous slides")


if __name__ == "__main__":
    main()
