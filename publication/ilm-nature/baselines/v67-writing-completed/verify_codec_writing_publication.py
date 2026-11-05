#!/usr/bin/env python3
"""Bind new writing figures to native charts, raw pixels and immutable sources."""

import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import fitz
import numpy as np
import openpyxl
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = HERE / "baselines/v63-completed-actor"
NS = dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    c="http://schemas.openxmlformats.org/drawingml/2006/chart")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pdf_rasters(path, number, raw):
    document = fitz.open(path)
    page = document[number]
    found, seen = [], set()
    for item in page.get_images(full=True):
        xref = item[0]
        if xref in seen:
            continue
        seen.add(xref)
        value = np.asarray(Image.open(io.BytesIO(document.extract_image(xref)["image"])).convert("L"))
        if value.shape != (32, 64):
            continue
        # Office/PDF can deduplicate identical reference/direct images. Verify
        # every visible placement, rather than rejecting legitimate reuse.
        for rect, transform in page.get_image_rects(xref, transform=True):
            if transform.b or transform.c or not transform.a or not transform.d:
                raise ValueError("rotated/degenerate writing evidence")
            image = value[::-1] if transform.d < 0 else value
            image = image[:, ::-1] if transform.a < 0 else image
            found.append((rect.y0, rect.x0, image))
    expected = sorted(raw, key=lambda x:(x["y"], x["x"]))
    if len(found) != len(expected):
        raise ValueError("displayed raw-image count differs: " + str(path))
    for (_, _, actual), row in zip(sorted(found, key=lambda x:x[:2]), expected):
        original = np.asarray(Image.open(ROOT / row["source"]).convert("L"))
        if not np.array_equal(actual, original):
            raise ValueError("displayed writing pixels differ from source: " + str(path))


def verify():
    build = json.loads((HERE / "evidence/codec_writing_deck_build.json").read_text())
    if sha(HERE / "ILM_Figures_editable.pptx") != build["pptx_sha256"]:
        raise ValueError("editable figures changed after generation")
    for name, digest in build["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("writing source evidence changed")
    baseline = json.loads((BASELINE / "baseline_manifest.json").read_text())
    retained_figures = 0
    for name, digest in baseline["files_sha256"].items():
        if sha(BASELINE / name) != digest:
            raise ValueError("immutable completed-actor baseline changed")
        if name.startswith("figures/") or name == "nature_reading.sty":
            if sha(HERE / name) != digest:
                raise ValueError("an earlier scientific figure or style changed")
            retained_figures += name.startswith("figures/")
    fitting = json.loads((ROOT / "references/evidence/writer_factorial_v65_result.json").read_text())
    direct = json.loads((ROOT / "references/evidence/direct_writer_v66_result.json").read_text())
    codec = json.loads((ROOT / "references/evidence/codec_output_v67_result.json").read_text())
    expected_values = [
        [100 * fitting["arms"][arm]["measurements"]["aligned"]["foreground_f1"]
            for arm in ("inherited_flow", "inherited_direct", "pixels_flow", "pixels_direct")],
        [100 * value["measurements"][route]["foreground_f1"]
            for value, route in ((direct, "aligned"), (direct, "shuffled"), (codec, "aligned"), (codec, "shuffled"))],
    ]
    with zipfile.ZipFile(BASELINE / "ILM_Figures_editable.pptx") as old, zipfile.ZipFile(HERE / "ILM_Figures_editable.pptx") as deck:
        allowed = {"[Content_Types].xml", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
        preserved = [name for name in old.namelist() if name not in allowed]
        if any(old.read(name) != deck.read(name) for name in preserved):
            raise ValueError("an original slide/chart/media/theme part changed")
        for name in deck.namelist():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(deck.read(name))
        presentation = ET.fromstring(deck.read("ppt/presentation.xml"))
        if len(presentation.findall("p:sldIdLst/p:sldId", NS)) != 22:
            raise ValueError("twenty-two editable slides required")
        for chart, values in zip(build["chart_data"], expected_values):
            parsed = ET.fromstring(deck.read(chart["part"]))
            actual = [float(x.text) for x in parsed.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
            if not np.allclose(actual, values, atol=1e-12, rtol=0):
                raise ValueError("native chart cache differs from audited scores")
            workbook = openpyxl.load_workbook(io.BytesIO(deck.read(chart["workbook"])), data_only=True).active
            if not np.allclose([workbook.cell(i + 2, 2).value for i in range(4)], values, atol=1e-12, rtol=0):
                raise ValueError("editable workbook differs from audited scores")
        for row in build["raw_images"]:
            expected = (ROOT / row["source"]).read_bytes()
            if deck.read(row["media"]) != expected or (HERE / row["evidence"]).read_bytes() != expected:
                raise ValueError("native PPT/evidence PNG changed source bytes")
    old_pdf = fitz.open(BASELINE / "ILM_Figures_review.pdf")
    new_pdf = fitz.open(HERE / "ILM_Figures_review.pdf")
    if len(new_pdf) != 22 or len(old_pdf) != 20:
        raise ValueError("reference and current review slide counts differ")
    identical = []
    for index in range(20):
        before = old_pdf[index].get_pixmap(matrix=fitz.Matrix(.75, .75))
        after = new_pdf[index].get_pixmap(matrix=fitz.Matrix(.75, .75))
        identical.append(before.width == after.width and before.height == after.height and before.samples == after.samples)
    if not all(identical):
        raise ValueError("a preserved slide's rendered appearance changed")
    verify_pdf_rasters(HERE / "ILM_Figures_review.pdf", 20, build["raw_images"])
    verify_pdf_rasters(HERE / "figures/Figure_9_writing_space.pdf", 0, build["raw_images"])
    article = fitz.open(HERE / "manuscript.pdf")
    pages = [i for i, page in enumerate(article) if "Faithful supplied-code writing" in page.get_text()]
    if len(pages) != 1:
        raise ValueError("locate the actual new figure caption in the article")
    verify_pdf_rasters(HERE / "manuscript.pdf", pages[0], build["raw_images"])
    requirements = {
        "manuscript":("56.563", "99.998", "126", "16.953"),
        "manuscript_review":("56.563", "99.998", "16.953"),
        "supplementary_information":("34.567", "99.995", "56.563", "99.998", "16.953", "99.7517", "88.3693"),
    }
    for name, required in requirements.items():
        document = fitz.open(HERE / (name + ".pdf"))
        text = "\n".join(page.get_text() for page in document)
        if any(value not in text for value in required):
            raise ValueError("compiled document lacks current evidence: " + name)
        log = (HERE / ("." + name) / (name + ".log")).read_text()
        if any(term in log for term in ("Overfull", "undefined references", "undefined citations", "LaTeX Error")):
            raise ValueError("compiled layout/reference error: " + name)
    redline = json.loads((HERE / "evidence/codec_writing_redline_build.json").read_text())
    for name, value in redline["documents"].items():
        if sha(HERE / (name + ".pdf")) != value["pdf_sha256"] or sha(HERE / (name + ".tex")) != value["tex_sha256"]:
            raise ValueError("matching redline differs from its verified build")
    result = dict(previous_deck_parts_unchanged=len(preserved), previous_slides_render_identically=20,
        previous_figure_files_unchanged=retained_figures, source_pngs_exact=16,
        ppt_rasters_exact=16, standalone_pdf_rasters_exact=16, deck_pdf_rasters_exact=16,
        article_pdf_rasters_exact=16, article_new_figure_page=pages[0] + 1,
        native_chart_workbooks_and_caches_verified=2, matching_redlines_verified=True,
        pptx_sha256=sha(HERE / "ILM_Figures_editable.pptx"),
        review_pdf_sha256=sha(HERE / "ILM_Figures_review.pdf"),
        artifact_tool_available=False, native_ooxml_fallback=True,
        output_repair=False, language_quality_claim=False, verifier_sha256=sha(Path(__file__)))
    (HERE / "evidence/codec_writing_publication_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
