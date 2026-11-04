#!/usr/bin/env python3
"""Verify that displayed PDF/PPT rasters preserve the actual V61 answer bits."""

import io
import json

import fitz
import numpy as np
from PIL import Image
from pptx import Presentation
import torch
import zipfile

from generate_evidence import HERE,ROOT
from generate_native_pixel_evidence import RUN
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions


def main():
    report=json.loads((RUN/"development/report.json").read_text())
    rows={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")}
    renderer=VisualQARenderer();expected=[];ppt_expected=[]
    for identifier in report["primary_ids"][:2]:
        for route in ("reference","trained","trained_blank"):
            cells=(renderer(rows[identifier]["references"][0]["text"]) if route=="reference" else
                torch.load(RUN/"development/predictions"/route/(identifier+".pt"),map_location="cpu",weights_only=False)["images"])
            bitmap=np.full((32,512),255,dtype=np.uint8)
            for i,cell in enumerate(cells[:16]):
                bitmap[:,32*i:32*(i+1)]=(255*(1-cell[0].numpy())).astype(np.uint8)
            expected.append(bitmap)
            if route!="trained_blank":ppt_expected.append(bitmap)
    pdf=fitz.open(HERE/"figures/Figure_6_native_pixel_language.pdf");page=pdf[0]
    images=page.get_images(full=True)
    if len(images)!=6:raise ValueError("six separate raw rasters required")
    for item,reference in zip(images,expected):
        xref=item[0];placements=page.get_image_rects(xref,transform=True)
        if len(placements)!=1:raise ValueError("unexpected raw image reuse")
        matrix=placements[0][1]
        if matrix.b or matrix.c or matrix.a==0 or matrix.d==0:
            raise ValueError("raw image rotated or degenerate")
        actual=np.asarray(Image.open(io.BytesIO(pdf.extract_image(xref)["image"])).convert("L"))
        # Matplotlib stores image rows inverted and places them with a negative
        # PDF scale. Compare displayed coordinates, not the storage convention.
        if matrix.d<0:actual=actual[::-1]
        if matrix.a<0:actual=actual[:,::-1]
        if not np.array_equal(actual,reference):raise ValueError("PDF altered a raw image pixel")
    prs=Presentation(HERE/"ILM_Figures_editable.pptx")
    pictures=[shape for shape in prs.slides[14].shapes if shape.shape_type==13]
    if len(pictures)!=4:raise ValueError("four raw PPT rasters required")
    for shape,reference in zip(pictures,ppt_expected):
        actual=np.asarray(Image.open(io.BytesIO(shape.image.blob)).convert("L"))
        if not np.array_equal(actual,reference):raise ValueError("PPT altered a raw image pixel")
    with zipfile.ZipFile(HERE/"baselines/v60/ILM_Figures_editable.pptx") as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        if not all(old.read(f"ppt/slides/slide{i}.xml")==new.read(f"ppt/slides/slide{i}.xml") for i in range(1,15)):
            raise ValueError("an earlier slide changed")
    receipt=dict(pdf_rasters_exact_in_display_coordinates=6,ppt_rasters_exact=4,
        prior_fourteen_slide_xml_unchanged=True,output_repair=False,
        pdf_placement_orientation_accounted_for=True)
    (HERE/"evidence/native_pixel_raster_integrity.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":main()
