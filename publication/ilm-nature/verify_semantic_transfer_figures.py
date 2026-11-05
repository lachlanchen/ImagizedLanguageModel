#!/usr/bin/env python3
"""Bind editable figures to source scores, model configuration and old slides."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
QA=ROOT/"artifacts/semantic_figures_v72_20261106"
NS=dict(p="http://schemas.openxmlformats.org/presentationml/2006/main",
    a="http://schemas.openxmlformats.org/drawingml/2006/main",
    c="http://schemas.openxmlformats.org/drawingml/2006/chart")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    build=json.loads((HERE/"evidence/semantic_transfer_deck_build.json").read_text())
    render=json.loads((HERE/"evidence/semantic_transfer_figure_render.json").read_text())
    baseline=json.loads((HERE/"baselines/v72-visual-semantics/baseline_manifest.json").read_text())
    for name,digest in build["sources_sha256"].items():
        if sha(ROOT/name)!=digest:
            raise ValueError("figure implementation/evidence changed")
    if (sha(HERE/"ILM_Figures_editable.pptx")!=build["pptx_sha256"]
            or render["pptx_sha256"]!=build["pptx_sha256"]
            or sha(HERE/"render_semantic_transfer_figures.py")!=render["renderer_sha256"]
            or sha(HERE/"ILM_Figures_review.pdf")!=render["rendered_pdf_sha256"]
            or build["language_quality_claim"] or render["language_quality_claim"]):
        raise ValueError("current figure files or scope changed")
    source_prefix="publication/ilm-nature/"
    old_deck=subprocess.check_output(["git","show",baseline["source_commit"]+":"+source_prefix+"ILM_Figures_editable.pptx"],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(old_deck)) as old,zipfile.ZipFile(HERE/"ILM_Figures_editable.pptx") as new:
        mutable={"[Content_Types].xml","ppt/presentation.xml","ppt/_rels/presentation.xml.rels"}
        preserved=0
        for name in old.namelist():
            if name not in mutable:
                if old.read(name)!=new.read(name):
                    raise ValueError("old native deck part changed: "+name)
                preserved+=1
        if preserved!=build["original_parts_preserved"]:
            raise ValueError("old part inventory differs")
        for name in new.namelist():
            if name.endswith((".xml",".rels")):
                ET.fromstring(new.read(name))
        presentation=ET.fromstring(new.read("ppt/presentation.xml"))
        if len(presentation.find("p:sldIdLst",NS))!=26:
            raise ValueError("new deck must contain 26 slides")
        reader=json.loads((ROOT/"references/evidence/native_visual_transfer_v71_result.json").read_text())
        expected=[reader["summaries"][route]["exact"] for route in
            ("raw/normal","raw/blank","native_budget/normal","native_budget/blank","text")]
        chart=ET.fromstring(new.read(build["chart"]["part"]))
        cache=[int(n.text) for n in chart.findall(".//c:ser/c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
        workbook=openpyxl.load_workbook(io.BytesIO(new.read(build["chart"]["workbook"])),data_only=True)
        values=[workbook.active.cell(i+2,2).value for i in range(5)]
        categories=[workbook.active.cell(i+2,1).value for i in range(5)]
        if cache!=expected or values!=expected or categories!=build["chart"]["categories"] or expected!=[1,0,4,0,3]:
            raise ValueError("native chart cache/workbook differs from actual source")
        new_text=[]
        for number in (25,26):
            slide=ET.fromstring(new.read(f"ppt/slides/slide{number}.xml"))
            new_text.append("\n".join(n.text or "" for n in slide.findall(".//a:t",NS)))
            for transform in slide.findall(".//a:xfrm",NS):
                off,ext=transform.find("a:off",NS),transform.find("a:ext",NS)
                if off is not None and ext is not None:
                    x,y,cx,cy=[int(n) for n in (off.get("x"),off.get("y"),ext.get("cx"),ext.get("cy"))]
                    if min(x,y,cx,cy)<0 or x+cx>15240000 or y+cy>8572500:
                        raise ValueError("new native shape leaves the canvas")
        if any(text not in new_text[0] for text in ("Helper only","Vocabulary module absent","no image answers")):
            raise ValueError("reader helper boundary is missing")
        if any(text not in new_text[1] for text in ("52 × 4096","8 layers, width 512","32 midpoint steps",
            "Re-encode actual emitted pixels","128 blocks","0/128 exact; 29.386%")):
            raise ValueError("deployed computation or candidate boundary is missing")
    original_pdf=subprocess.check_output(["git","show",baseline["source_commit"]+":"+source_prefix+"ILM_Figures_review.pdf"],cwd=ROOT)
    with fitz.open(stream=original_pdf,filetype="pdf") as old,fitz.open(HERE/"ILM_Figures_review.pdf") as new:
        if len(new)!=26:
            raise ValueError("complete canonical PDF inventory differs")
        for i in range(24):
            if old[i].get_pixmap(alpha=False).samples!=new[i].get_pixmap(alpha=False).samples:
                raise ValueError("canonical previous slide pixels changed")
        for i,name in [(24,"Figure_S10_visual_semantics"),(25,"Figure_S11_continuous_writer")]:
            with fitz.open(HERE/"figures"/(name+".pdf")) as figure:
                if new[i].get_pixmap(alpha=False).samples!=figure[0].get_pixmap(alpha=False).samples:
                    raise ValueError("paper figure is not the same rendered native slide")
            for suffix,digest in render["new_figures_sha256"][name].items():
                if sha(HERE/"figures"/(name+"."+suffix))!=digest:
                    raise ValueError("paper figure export changed")
    retained_assets=0
    for name,digest in baseline["files_sha256"].items():
        if name.startswith("figures/") or name.endswith(".sty"):
            if sha(HERE/name)!=digest:
                raise ValueError("prior individual figure/style changed")
            retained_assets+=1
    receipt=dict(pptx_sha256=build["pptx_sha256"],figure_pdf_sha256=render["rendered_pdf_sha256"],
        native_deck_parts_preserved=preserved,canonical_prior_pdf_slides_preserved=24,
        prior_figure_style_assets_preserved=retained_assets,native_chart_workbook_matches_source=True,
        new_figures_match_native_render=True,source_bound_numeric_counts=expected,
        offline_checks_pass=True,visual_review_required=True,language_quality_claim=False)
    (HERE/"evidence/semantic_transfer_figure_verification.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=="__main__":
    verify()
