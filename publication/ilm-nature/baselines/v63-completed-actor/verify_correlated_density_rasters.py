#!/usr/bin/env python3
"""Compare displayed V62 PDF/PPT writing directly with immutable source bits."""

import io
import json
import zipfile

import fitz
import numpy as np
from PIL import Image
from pptx import Presentation
import torch

from generate_correlated_density_evidence import RUN,completed_report
from generate_evidence import HERE,ROOT
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions


def main():
    report=completed_report()
    records={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")};renderer=VisualQARenderer()
    expected=[]
    for identifier in report["primary_ids"][:2]:
        for route in ("reference","k1","k8"):
            cells=renderer(records[identifier]["references"][0]["text"]) if route=="reference" else torch.load(RUN/"development/predictions"/route/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]
            if len(cells)>20:raise ValueError("raw display would truncate writing")
            raster=np.full((32,640),255,dtype=np.uint8)
            for i,cell in enumerate(cells):raster[:,i*32:(i+1)*32]=(255*(1-cell[0].numpy())).astype(np.uint8)
            expected.append(raster)
    pdf=fitz.open(HERE/"figures/Figure_7_correlated_density.pdf");page=pdf[0]
    images=page.get_images(full=True)
    if len(images)!=6:raise ValueError("six distinct raw PDF images required")
    for item,reference in zip(images,expected):
        xref=item[0];placements=page.get_image_rects(xref,transform=True)
        if len(placements)!=1:raise ValueError("unexpected raw image reuse")
        matrix=placements[0][1]
        if matrix.b or matrix.c or not matrix.a or not matrix.d:raise ValueError("rotated or degenerate image")
        actual=np.asarray(Image.open(io.BytesIO(pdf.extract_image(xref)["image"])).convert("L"))
        if matrix.d<0:actual=actual[::-1]
        if matrix.a<0:actual=actual[:,::-1]
        if not np.array_equal(actual,reference):raise ValueError("PDF changed source pixels")
    deck=Presentation(HERE/"ILM_Figures_editable.pptx")
    pictures=[shape for shape in deck.slides[16].shapes if shape.shape_type==13]
    if len(pictures)!=6:raise ValueError("six raw PPT images required")
    for shape,reference in zip(pictures,expected):
        actual=np.asarray(Image.open(io.BytesIO(shape.image.blob)).convert("L"))
        if not np.array_equal(actual,reference):raise ValueError("PPT changed source pixels")
    with zipfile.ZipFile(HERE/"baselines/v61/ILM_Figures_editable.pptx") as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        if not all(old.read(f"ppt/slides/slide{i}.xml")==new.read(f"ppt/slides/slide{i}.xml") for i in range(1,16)):
            raise ValueError("a previous slide changed")
    receipt=dict(pdf_rasters_exact_in_display_coordinates=6,ppt_rasters_exact=6,
        prior_fifteen_slide_xml_unchanged=True,output_repair=False,pdf_placement_orientation_accounted_for=True)
    (HERE/"evidence/correlated_density_raster_integrity.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":main()
