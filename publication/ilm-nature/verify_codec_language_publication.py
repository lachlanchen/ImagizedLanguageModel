#!/usr/bin/env python3
"""Verify the completed V68 publication against pinned scores and physical pixels.

This checks publication provenance, not a second native neural forward pass.
It supersedes the earlier active-deck validators by preserving their entire
verified 22-slide package and its appearance in an immutable Git baseline.
"""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from decimal import Decimal

import fitz
import numpy as np
import openpyxl
from PIL import Image
import torch

from generate_codec_language_figures import PINS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / "baselines/v67-writing-completed"
DIAG = ROOT / "artifacts/codec_language_v68_diagnosis_20261105"
RUN = ROOT / "artifacts/codec_language_v68_20261105"
NS = dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
          a="http://schemas.openxmlformats.org/drawingml/2006/main",
          r="http://schemas.openxmlformats.org/officeDocument/2006/relationships",
          c="http://schemas.openxmlformats.org/drawingml/2006/chart")
FINAL_PDFS = ("manuscript.pdf", "manuscript_review.pdf",
              "supplementary_information.pdf", "redlined_manuscript.pdf",
              "redlined_supplementary_information.pdf", "ILM_Figures_review.pdf",
              "figures/Figure_10_conditional_writing.pdf",
              "figures/Figure_S9_prediction_diagnosis.pdf")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    return np.asarray(Image.open(path).convert("L"))


def source_pixels(row):
    value = pixels(ROOT / row["source"])
    if row["crop"]:
        left, top, right, bottom = row["crop"]
        value = value[top:bottom, left:right]
    return value


def verify_pdf_rasters(path, page_index, rows):
    """Check every visible placement, including deduplicated and mirrored XObjects."""
    doc = fitz.open(path)
    page = doc[page_index]
    found, seen = [], set()
    for item in page.get_images(full=True):
        xref = item[0]
        if xref in seen:
            continue
        seen.add(xref)
        value = pixels(io.BytesIO(doc.extract_image(xref)["image"]))
        if value.shape[0] != 32:
            continue
        for rect, transform in page.get_image_rects(xref, transform=True):
            if transform.b or transform.c or not transform.a or not transform.d:
                raise ValueError("rotated/degenerate evidence image")
            actual = value[::-1] if transform.d < 0 else value
            actual = actual[:, ::-1] if transform.a < 0 else actual
            # LibreOffice rounds frame edges by about 0.04 PDF point. The
            # source PPT frame is checked exactly below; allow 0.2% relative
            # aspect rounding here, without modifying or resampling pixels.
            if abs((rect.width / rect.height) / (value.shape[1] / 32) - 1) > .002:
                raise ValueError("writing evidence aspect ratio changed")
            found.append((round(rect.y0, 2), rect.x0, actual))
    expected = sorted(rows, key=lambda row: (row["y"], row["x"]))
    if len(found) != len(expected):
        raise ValueError(f"visible evidence count differs: {path}, page {page_index+1}")
    for (_, _, actual), row in zip(sorted(found, key=lambda v: v[:2]), expected):
        if not np.array_equal(actual, source_pixels(row)):
            raise ValueError(f"PDF evidence differs from source: {path}")


def figure_page(path, caption):
    doc = fitz.open(path)
    pages = [i for i, page in enumerate(doc) if caption in page.get_text()]
    if len(pages) != 1:
        raise ValueError(f"expected one actual figure caption: {path}, {pages}")
    return pages[0]


def verify():
    if torch.cuda.is_initialized():
        raise ValueError("publication verification is CPU only")
    build = json.loads((HERE / "evidence/codec_language_deck_build.json").read_text())
    if sha(HERE / "ILM_Figures_editable.pptx") != build["pptx_sha256"]:
        raise ValueError("deck changed after generation")
    receipts = {}
    for name, digest in PINS.items():
        public = ROOT / "references/evidence" / (name + ".json")
        copied = HERE / "evidence" / public.name
        if sha(public) != digest or copied.read_bytes() != public.read_bytes():
            raise ValueError("pinned public receipt changed")
        receipts[name] = json.loads(public.read_text())
    audit = receipts["codec_language_v68_autonomous_result"]
    diag = receipts["codec_language_v68_diagnosis_result"]
    train = receipts["codec_language_v68_training_integrity"]
    for name, digest in build["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("source evidence changed: " + name)
    if audit["gates"] != dict(chinese_exact=False, chinese_foreground_f1=False,
            both_correct_pairs=False, question_gain=True,
            question_gain_ci_positive=True, learned_stopping=True):
        raise ValueError("original six decisions changed")
    if (audit["bounded_chinese_development_pass"] or not audit["all_verification_checks_pass"]
            or diag["native_neural_forward_replayed"] or diag["optimizer_updates"] != 0
            or not diag["reference_history_and_length_supplied"]
            or train["recorded_updates_verified"] != 10000
            or not all(train["checks"].values())):
        raise ValueError("experimental scope changed")
    baseline = json.loads((BASELINE / "baseline_manifest.json").read_text())
    retained = 0
    for name, digest in baseline["files_sha256"].items():
        raw = (BASELINE / name).read_bytes()
        original = subprocess.check_output(["git", "show", baseline["source_commit"] +
            ":publication/ilm-nature/" + name], cwd=ROOT)
        if sha(BASELINE / name) != digest or raw != original:
            raise ValueError("immutable Git baseline changed: " + name)
        if name.startswith("figures/") or name == "nature_reading.sty":
            if (HERE / name).read_bytes() != raw:
                raise ValueError("previous figure or template changed")
            retained += name.startswith("figures/")
    cmrc = audit["independently_recomputed_measurements"]["cmrc"]
    question = audit["independently_recomputed_question_effect"]
    order = ("train/cmrc", "train/dolly", "development/cmrc", "development/dolly")
    values = [[[100 * cmrc[route]["foreground_f1"] for route in ("normal", "blank")]],
              [[0., 100 * audit["paired_both_correct"], 100 * cmrc["normal"]["stopped"]]],
              [[100 * diag["groups"][group][key] for group in order]
               for key in ("supplied_code_f1", "exact_history_f1", "autonomous_f1")]]
    macro_values = dict(CodecLanguageFOne=values[0][0][0],
        CodecLanguageBlankFOne=values[0][0][1], CodecLanguageQuestionGain=100*question["mean"],
        CodecLanguageCILow=100*question["ci95"][0], CodecLanguageCIHigh=100*question["ci95"][1])
    macros = (HERE / "codec_language_values.tex").read_text()
    for key, value in macro_values.items():
        if "\\newcommand{\\" + key + "}{" + f"{value:.3f}" + "}" not in macros:
            raise ValueError("generated numerical macro differs")
    table = (HERE / "codec_language_diagnosis_table.tex").read_text()
    for group, label in zip(order, ("Training, Chinese", "Training, English",
                                   "Development, Chinese", "Development, English")):
        item = diag["groups"][group]
        expected = label + " & " + " & ".join(f"{100*item[k]:.3f}" for k in
            ("supplied_code_f1", "exact_history_f1", "autonomous_f1"))
        expected += f" & {item['normalized_code_mse']:.4f}"
        if expected not in table:
            raise ValueError("diagnostic table differs")
    with zipfile.ZipFile(BASELINE / "ILM_Figures_editable.pptx") as old, \
            zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as deck:
        allowed = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
        preserved = [name for name in old.namelist() if name not in allowed]
        if any(old.read(name) != deck.read(name) for name in preserved):
            raise ValueError("previous slide/chart/workbook/media/theme changed")
        for name in deck.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(deck.read(name))
        root = ET.fromstring(deck.read("ppt/presentation.xml"))
        if len(root.findall("p:sldIdLst/p:sldId", NS)) != 24:
            raise ValueError("expected 24 slides")
        if len(build["chart_data"]) != 3 or len(build["raw_images"]) != 12:
            raise ValueError("new evidence counts differ")
        for chart, expected in zip(build["chart_data"], values):
            if not np.allclose(chart["values"], expected, atol=1e-12, rtol=0):
                raise ValueError("native chart build values differ from receipt")
            root = ET.fromstring(deck.read(chart["part"]))
            series = root.findall(".//c:barChart/c:ser", NS)
            if len(series) != len(expected):
                raise ValueError("native chart series count differs")
            book = openpyxl.load_workbook(io.BytesIO(deck.read(chart["workbook"])), data_only=True).active
            with zipfile.ZipFile(io.BytesIO(deck.read(chart["workbook"]))) as archive:
                worksheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            sn = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            serialized = {cell.get("r"): cell.find("s:v", sn).text
                for cell in worksheet.findall(".//s:c", sn) if cell.find("s:v", sn) is not None}
            for i, (ser, numbers) in enumerate(zip(series, expected)):
                points = ser.findall("c:val/c:numRef/c:numCache/c:pt/c:v", NS)
                cache = [float(v.text) for v in points]
                cells = [book.cell(j+2, i+2).value for j in range(len(numbers))]
                column = openpyxl.utils.get_column_letter(i+2)
                if any(Decimal(v.text) != Decimal(serialized[f"{column}{j+2}"]) for j, v in enumerate(points)):
                    raise ValueError("native cache differs from exact serialized worksheet value")
                if not np.allclose([cache, cells], [numbers, numbers], atol=1e-12, rtol=0):
                    raise ValueError("chart cache or embedded workbook differs")
                categories = [v.text for v in ser.findall("c:cat/c:strRef/c:strCache/c:pt/c:v", NS)]
                if categories != chart["categories"] or [book.cell(j+2, 1).value for j in range(len(numbers))] != categories:
                    raise ValueError("editable chart labels differ")
                if book.cell(1, i+2).value != chart["series_names"][i]:
                    raise ValueError("editable series label differs")
        for row in build["raw_images"]:
            expected = source_pixels(row)
            if (sha(ROOT / row["source"]) != row["source_sha256"]
                    or list(expected.shape) != row["pixel_shape"]
                    or hashlib.sha256(expected.tobytes()).hexdigest() != row["visible_pixel_sha256"]
                    or not np.array_equal(pixels(HERE / row["evidence"]), expected)
                    or deck.read(row["media"]) != (HERE / row["evidence"]).read_bytes()):
                raise ValueError("PPT/copy/cropped image differs from source")
            if row["slide"] == 23 and row["crop"] is not None:
                raise ValueError("main figure must show complete streams")
            part = f"ppt/slides/slide{row['slide']}.xml"
            rels = ET.fromstring(deck.read(f"ppt/slides/_rels/slide{row['slide']}.xml.rels"))
            target = "../media/" + Path(row["media"]).name
            rid = [rel.get("Id") for rel in rels if rel.get("Target") == target]
            if len(rid) != 1:
                raise ValueError("evidence relationship missing or ambiguous")
            slide = ET.fromstring(deck.read(part))
            pictures = [pic for pic in slide.findall("p:cSld/p:spTree/p:pic", NS)
                if pic.find("p:blipFill/a:blip", NS).get("{"+NS["r"]+"}embed") == rid[0]]
            if len(pictures) != 1 or pictures[0].find("p:blipFill/a:srcRect", NS) is not None:
                raise ValueError("unexpected PPT crop or image reuse")
            transform = pictures[0].find("p:spPr/a:xfrm", NS)
            offset, extent = transform.find("a:off", NS), transform.find("a:ext", NS)
            geometry = [int(offset.get("x")), int(offset.get("y")), int(extent.get("cx")), int(extent.get("cy"))]
            expected_geometry = [row["x"], row["y"], expected.shape[1]*row["scale"], expected.shape[0]*row["scale"]]
            if geometry != [round(v*9525) for v in expected_geometry]:
                raise ValueError("PPT source frame or uniform display scale changed")
    # Reconstruct the selected diagnostic PNGs from source-bound physical tensors.
    report = json.loads((DIAG / "generation_report.json").read_text())
    prepared = torch.load(DIAG / "prepared.pt", map_location="cpu", weights_only=True)
    for index in (0, 16, 17):
        casepath = DIAG / f"case_{index:02}.pt"
        if sha(casepath) != report["raw_files_sha256"][casepath.name]:
            raise ValueError("source-bound diagnostic tensor changed")
        data = torch.load(casepath, map_location="cpu", weights_only=True)
        reference = (prepared[index]["reference"][0, 0].numpy() >= 0).astype(np.uint8)*255
        if not np.array_equal(pixels(DIAG / f"case_{index:02}_reference.png"), reference):
            raise ValueError("reference PNG differs from prepared source pixels")
        if index in (0, 16):
            for route in ("supplied", "exact_history"):
                tensor = data[route + "_pixels"][:, 0]
                array = np.concatenate([(v.numpy() >= 0).astype(np.uint8)*255 for v in tensor], axis=1)
                if not np.array_equal(pixels(DIAG / f"case_{index:02}_{route}.png"), array):
                    raise ValueError("diagnostic image differs from retained scores")
        if index in (16, 17):
            if prepared[index]["id"] != ("DEV_477_QUERY_0" if index == 16 else "DEV_477_QUERY_1"):
                raise ValueError("first fixed displayed case changed")
            for route in ("normal", "blank"):
                relative = f"predictions/{route}/cmrc_{prepared[index]['id']}.png"
                if sha(RUN / "development" / relative) != audit["raw_files_sha256"][relative]:
                    raise ValueError("published autonomous stream changed")
    oldpdf = fitz.open(BASELINE / "ILM_Figures_review.pdf")
    newpdf = fitz.open(HERE / "ILM_Figures_review.pdf")
    if len(oldpdf) != 22 or len(newpdf) != 24:
        raise ValueError("review slide count differs")
    for i in range(22):
        a, b = (doc[i].get_pixmap(matrix=fitz.Matrix(.75, .75)) for doc in (oldpdf, newpdf))
        if (a.width, a.height, a.samples) != (b.width, b.height, b.samples):
            raise ValueError("old rendered slide changed: " + str(i+1))
    mainrows = [r for r in build["raw_images"] if r["slide"] == 23]
    supprows = [r for r in build["raw_images"] if r["slide"] == 24]
    for index, rows in ((22, mainrows), (23, supprows)):
        verify_pdf_rasters(HERE / "ILM_Figures_review.pdf", index, rows)
    for name, rows in ((FINAL_PDFS[-2], mainrows), (FINAL_PDFS[-1], supprows)):
        verify_pdf_rasters(HERE / name, 0, rows)
    figure_pages = {}
    for name, caption, rows in (
            ("manuscript", "Question dependence persists when accurate answer writing fails.", mainrows),
            ("manuscript_review", "Question dependence persists when accurate answer writing fails.", mainrows),
            ("supplementary_information", "Conditional writing fails with exact reference history.", supprows)):
        page = figure_page(HERE / (name + ".pdf"), caption)
        verify_pdf_rasters(HERE / (name + ".pdf"), page, rows)
        figure_pages[name] = page + 1
        doc = fitz.open(HERE / (name + ".pdf"))
        text = "\n".join(p.get_text() for p in doc)
        if any(value not in text for value in ("32.671", "19.508", "13.163", "10.933", "15.730", "16.965")):
            raise ValueError("compiled manuscript lacks current rounded values")
        log = (HERE / ("." + name) / (name + ".log")).read_text()
        if any(term in log for term in ("Overfull", "undefined references", "undefined citations", "LaTeX Error")):
            raise ValueError("unresolved compiled layout/reference error")
    redline = json.loads((HERE / "evidence/codec_language_redline_build.json").read_text())
    if redline["baseline_commit"] != baseline["source_commit"]:
        raise ValueError("wrong redline baseline")
    if redline["builder_sha256"] != sha(HERE / "build_codec_language_redlines.py"):
        raise ValueError("redline builder changed after compilation")
    for name, row in redline["documents"].items():
        if sha(HERE / (name + ".pdf")) != row["pdf_sha256"] or sha(HERE / (name + ".tex")) != row["tex_sha256"]:
            raise ValueError("redline differs from verified build")
        for relative, digest in row["new_inputs_sha256"].items():
            if sha(HERE / relative) != digest:
                raise ValueError("redline is stale relative to manuscript inputs")
    visual = json.loads((HERE / "evidence/codec_language_visual_qa.json").read_text())
    for name, digest in visual["files_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("output changed after visual review")
    for name in ("package_integrity", "layout_geometry"):
        check = json.loads((HERE / "evidence" / ("codec_language_" + name + ".json")).read_text())
        if check["finding_count"] or check["slide_count"] != 24 or check.get("warning_count", 0):
            raise ValueError("PPT structural/layout validation failed")
    result = dict(baseline_commit=baseline["source_commit"], immutable_git_baseline_files=len(baseline["files_sha256"]),
        previous_deck_parts_unchanged=len(preserved), previous_slides_render_identically=22,
        previous_figure_files_unchanged=retained, slides=24,
        new_native_chart_workbooks_and_caches_verified=3,
        new_chart_caches_equal_serialized_worksheet_decimals=True, ppt_evidence_images_exact=12,
        main_complete_streams=6, supplementary_explicit_first_block_excerpts=6,
        deck_pdf_rasters_exact=12, standalone_pdf_rasters_exact=12,
        article_review_supplement_pdf_rasters_exact=18, figure_pages=figure_pages,
        original_six_decisions_retained=True, matching_redlines_verified=True,
        pdfs={name: dict(sha256=sha(HERE/name), pages=len(fitz.open(HERE/name))) for name in FINAL_PDFS},
        pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"),
        manual_visual_qa_receipt_sha256=sha(HERE / "evidence/codec_language_visual_qa.json"),
        artifact_tool_available=False, native_ooxml_fallback=True, native_powerpoint_playback_checked=False,
        native_neural_forward_replayed=False, pdf_aspect_relative_rounding_tolerance=.002,
        ppt_uniform_scale_exact=True, output_repair=False, language_quality_claim=False,
        general_ilm_goal_complete=False, verifier_sha256=sha(Path(__file__)))
    (HERE / "evidence/codec_language_publication_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
