#!/usr/bin/env python3
"""Verify V63 figures against frozen scores, source pixels and old deck parts."""

import hashlib
import io
import json
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

import fitz
import numpy as np
import openpyxl
from PIL import Image
import torch

from generate_assignment_evidence import HERE,ROOT,RUN,completed_report
from generate_v57_evidence import pad_answer
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions

NS = dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    c="http://schemas.openxmlformats.org/drawingml/2006/chart",
    r="http://schemas.openxmlformats.org/officeDocument/2006/relationships")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pdf_images(path,page_number,expected):
    document = fitz.open(path)
    page = document[page_number]
    images = [item for item in page.get_images(full=True) if page.get_image_rects(item[0])]
    if len(images) != len(expected):
        raise ValueError("unexpected displayed image count: "+str(path))
    actual_rows = []
    for item in images:
        placements = page.get_image_rects(item[0],transform=True)
        if len(placements) != 1:
            raise ValueError("image reuse/cropping must be reviewed")
        rect,matrix = placements[0]
        if matrix.b or matrix.c or not matrix.a or not matrix.d:
            raise ValueError("rotated or degenerate writing")
        image = np.asarray(Image.open(io.BytesIO(document.extract_image(item[0])["image"])).convert("L"))
        if matrix.d < 0:
            image = image[::-1]
        if matrix.a < 0:
            image = image[:,::-1]
        actual_rows.append((rect.y0,rect.x0,image))
    for (_,_,actual),reference in zip(sorted(actual_rows,key=lambda row:row[:2]),expected):
        if not np.array_equal(actual,reference):
            raise ValueError("displayed PDF pixels differ from source: "+str(path))


def verify():
    report = completed_report()
    for name,expected_hash in json.loads((HERE/"evidence/assignment_source_hashes.json").read_text()).items():
        if sha(ROOT/name) != expected_hash:
            raise ValueError("recorded source evidence changed: "+name)
    records = {r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")}
    renderer = VisualQARenderer()
    expected = []
    for i,identifier in enumerate(report["primary_ids"][:2]):
        for route in ("reference","control","aligned"):
            cells = renderer(records[identifier]["references"][0]["text"]) if route=="reference" else torch.load(
                RUN/"development/predictions"/route/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]
            if len(cells)>20:
                raise ValueError("display would truncate ink")
            image = (255*(1-pad_answer(cells,20))).astype(np.uint8)
            asset = np.asarray(Image.open(HERE/f"evidence/v63_raw/{i}_{route}.png").convert("L"))
            if not np.array_equal(image,asset):
                raise ValueError("source asset was repaired")
            expected.append(image)
    check_pdf_images(HERE/"figures/Figure_8_question_assignment.pdf",0,expected)
    check_pdf_images(HERE/"ILM_Figures_review.pdf",18,expected)
    old = HERE/"baselines/v62-transcript-bounds"
    exempt = {"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels","docProps/app.xml"}
    with zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as deck,zipfile.ZipFile(old/"ILM_Figures_editable.pptx") as base:
        unchanged = [name for name in base.namelist() if name not in exempt]
        for name in unchanged:
            if base.read(name) != deck.read(name):
                raise ValueError("prior deck part changed: "+name)
        with zipfile.ZipFile(HERE/"protocol-v63/question_assignment_method_editable.pptx") as method:
            if deck.read("ppt/slides/slide18.xml") != method.read("ppt/slides/slide1.xml"):
                raise ValueError("verified native method slide changed")
        slide = ET.fromstring(deck.read("ppt/slides/slide19.xml"))
        rels = {r.get("Id"):r.get("Target") for r in ET.fromstring(deck.read("ppt/slides/_rels/slide19.xml.rels"))}
        pictures = slide.findall("p:cSld/p:spTree/p:pic",NS)
        if len(pictures) != 6:
            raise ValueError("six native source raster images required")
        for pic,reference in zip(pictures,expected):
            if pic.find("p:blipFill/a:srcRect",NS) is not None:
                raise ValueError("cropped source picture")
            rid = pic.find("p:blipFill/a:blip",NS).get("{"+NS["r"]+"}embed")
            name = posixpath.normpath(posixpath.join("ppt/slides",rels[rid]))
            if not np.array_equal(np.asarray(Image.open(io.BytesIO(deck.read(name))).convert("L")),reference):
                raise ValueError("PPT raw image changed")
        routes = ("control","control_blank","aligned","aligned_blank")
        contrasts = ("aligned_minus_control_raster_f1","control_minus_control_blank_raster_f1","aligned_minus_aligned_blank_raster_f1")
        for number,values in ((14,[100*report["measurements"][r]["raster_f1"] for r in routes]),
                (15,[100*report["comparisons"][c]["mean"] for c in contrasts])):
            chart = ET.fromstring(deck.read(f"ppt/charts/chart{number}.xml"))
            book = openpyxl.load_workbook(io.BytesIO(deck.read(f"ppt/embeddings/v63_chart{number}.xlsx")),data_only=True).active
            cached = [float(e.text) for e in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
            if not np.allclose(cached,values,atol=1e-12,rtol=0):
                raise ValueError("chart cache differs from frozen audit")
            for i,value in enumerate(values):
                if abs(book.cell(i+2,2).value-value)>1e-12:
                    raise ValueError("native workbook differs from frozen audit")
                if number==15:
                    lo,hi = [100*v for v in report["comparisons"][contrasts[i]]["ci95"]]
                    for column,target in ((3,lo),(4,hi),(5,hi-value),(6,value-lo)):
                        if abs(book.cell(i+2,column).value-target)>1e-12:
                            raise ValueError("native interval workbook differs")
            if number==15:
                for tag,values in (("plus",[100*(report['comparisons'][c]['ci95'][1]-report['comparisons'][c]['mean']) for c in contrasts]),
                        ("minus",[100*(report['comparisons'][c]['mean']-report['comparisons'][c]['ci95'][0]) for c in contrasts])):
                    cached = [float(e.text) for e in chart.findall(f".//c:errBars/c:{tag}/c:numRef/c:numCache/c:pt/c:v",NS)]
                    if not np.allclose(cached,values,atol=1e-12,rtol=0):
                        raise ValueError("native errorbar cache differs")
        for name in deck.namelist():
            if name.endswith((".xml",".rels")):
                ET.fromstring(deck.read(name))
    old_pdf,new_pdf = fitz.open(old/"ILM_Figures_review.pdf"),fitz.open(HERE/"ILM_Figures_review.pdf")
    exact_render = []
    lossless_image_changes = []
    for i in range(17):
        a,b = old_pdf[i].get_pixmap(matrix=fitz.Matrix(.75,.75)),new_pdf[i].get_pixmap(matrix=fitz.Matrix(.75,.75))
        exact_render.append(a.samples==b.samples and a.width==b.width and a.height==b.height)
        if not exact_render[-1]:
            if old_pdf[i].get_text("words") != new_pdf[i].get_text("words"):
                raise ValueError("earlier text or geometry changed")
            # Lossless export repairs the earlier renderer's JPEG compression,
            # without editing any source bitmap or object. Require every
            # changed pixel to be within an original displayed image rectangle.
            changed = np.any(np.frombuffer(a.samples,dtype=np.uint8).reshape(a.height,a.width,3)
                != np.frombuffer(b.samples,dtype=np.uint8).reshape(b.height,b.width,3),axis=-1)
            mask = np.zeros_like(changed)
            for item in old_pdf[i].get_images(full=True):
                for rect in old_pdf[i].get_image_rects(item[0]):
                    x0,y0 = max(0,int(rect.x0*.75)-2),max(0,int(rect.y0*.75)-2)
                    x1,y1 = min(a.width,int(np.ceil(rect.x1*.75))+2),min(a.height,int(np.ceil(rect.y1*.75))+2)
                    mask[y0:y1,x0:x1] = True
            if np.any(changed & ~mask):
                raise ValueError("earlier rendering changed outside source image areas")
            lossless_image_changes.append(i+1)
    manifest = json.loads((old/"baseline_manifest.json").read_text())
    figure_count = 0
    for name,expected_hash in manifest["files_sha256"].items():
        if name.startswith("figures/") or name=="nature_reading.sty":
            if sha(HERE/name)!=expected_hash:
                raise ValueError("prior figure/style changed")
            figure_count += name.startswith("figures/")
    result = dict(previous_deck_parts_unchanged=len(unchanged),previous_slides_render_identically=sum(exact_render),
        previous_slides_with_only_lossless_image_export_changes=lossless_image_changes,
        previous_figure_files_unchanged=figure_count,method_slide_xml_unchanged=True,
        source_pngs_exact=6,pdf_rasters_exact=6,ppt_rasters_exact=6,deck_pdf_rasters_exact=6,
        native_charts_verified=[14,15],custom_interval_workbook_and_caches_verified=True,
        output_repair=False,pptx_sha256=sha(HERE/"ILM_Figures_editable.pptx"),
        pdf_sha256=sha(HERE/"ILM_Figures_review.pdf"),verifier_sha256=sha(Path(__file__)))
    (HERE/"evidence/assignment_publication_integrity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    verify()
