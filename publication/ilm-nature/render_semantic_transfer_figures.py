#!/usr/bin/env python3
"""Render the append-only deck and verify all 24 prior slide appearances."""

import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess

import fitz
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
QA=ROOT/"artifacts/semantic_figures_v72_20261106"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    values={line.split(":",1)[0]:int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split()[1].isdigit()}
    if (values["MemAvailable"]<24*1024**2 or
            (values["SwapTotal"] and 1-values["SwapFree"]/values["SwapTotal"]>.75)):
        raise RuntimeError("shared-host memory policy prevents rendering now")
    QA.mkdir(parents=True,exist_ok=True)
    render=QA/"render"
    render.mkdir(exist_ok=True)
    baseline=json.loads((HERE/"baselines/v72-visual-semantics/baseline_manifest.json").read_text())
    baseline_deck=subprocess.check_output(["git","show",baseline["source_commit"]+":publication/ilm-nature/ILM_Figures_editable.pptx"],cwd=ROOT)
    if hashlib.sha256(baseline_deck).hexdigest()!=baseline["files_sha256"]["ILM_Figures_editable.pptx"]:
        raise ValueError("source baseline deck changed")
    baseline_path=QA/"baseline.pptx"
    baseline_path.write_bytes(baseline_deck)
    profile=QA/"libreoffice-profile"
    command=["soffice","-env:UserInstallation="+profile.as_uri(),"--headless","--convert-to","pdf",
        "--outdir",str(render),str(HERE/"ILM_Figures_editable.pptx"),str(baseline_path)]
    with (QA/"libreoffice.log").open("w") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (QA/"render_runtime.json").write_text(json.dumps(dict(pid=process.pid,process_group=process.pid,
            profile=str(profile),status="running"),indent=2)+"\n")
        try:
            code=process.wait(timeout=180)
            if code:
                raise RuntimeError("owned LibreOffice render failed")
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            try:
                os.killpg(process.pid,0)
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            (QA/"render_runtime.json").write_text(json.dumps(dict(pid=None,profile=str(profile),status="exited"),indent=2)+"\n")
    build=json.loads((HERE/"evidence/semantic_transfer_deck_build.json").read_text())
    baseline=json.loads((HERE/"baselines/v72-visual-semantics/baseline_manifest.json").read_text())
    original=subprocess.check_output(["git","show",baseline["source_commit"]+":publication/ilm-nature/ILM_Figures_review.pdf"],cwd=ROOT)
    candidate=render/"ILM_Figures_editable.pdf"
    preserved=[]
    fresh_preserved=[]
    renderer_differences=[]
    fresh_baseline=render/"baseline.pdf"
    if not fresh_baseline.exists():
        raise ValueError("fresh baseline renderer comparison required")
    reference=fitz.open(fresh_baseline)
    with fitz.open(stream=original,filetype="pdf") as old,fitz.open(candidate) as new:
        if len(old)!=24 or len(new)!=26:
            raise ValueError("incorrect old/new slide inventory")
        for i in range(24):
            before=old[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            after=new[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            fresh=reference[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
            if (fresh.width!=after.width or fresh.height!=after.height or fresh.samples!=after.samples):
                raise ValueError("new content changes a fresh prior-slide render: "+str(i+1))
            fresh_preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(after.samples).hexdigest()))
            if before.width!=after.width or before.height!=after.height or before.samples!=after.samples:
                renderer_differences.append(i+1)
        canonical=fitz.open()
        canonical.insert_pdf(old)
        canonical.insert_pdf(new,from_page=24,to_page=25)
        canonical.save(QA/"canonical_figures.pdf")
        canonical.close()
        figures={}
        for i,name in [(24,"Figure_S10_visual_semantics"),(25,"Figure_S11_continuous_writer")]:
            page=new[i]
            document=fitz.open()
            document.insert_pdf(new,from_page=i,to_page=i)
            path=HERE/"figures"/(name+".pdf")
            document.save(path)
            document.close()
            page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(HERE/"figures"/(name+".png"))
            (HERE/"figures"/(name+".svg")).write_text(page.get_svg_image(text_as_path=False))
            figures[name]={suffix:sha(HERE/"figures"/(name+"."+suffix)) for suffix in ("pdf","png","svg")}
    reference.close()
    target=HERE/"ILM_Figures_review.pdf"
    target.write_bytes((QA/"canonical_figures.pdf").read_bytes())
    with fitz.open(stream=original,filetype="pdf") as old,fitz.open(target) as new:
        for i in range(24):
            before=old[i].get_pixmap(alpha=False)
            after=new[i].get_pixmap(alpha=False)
            if before.samples!=after.samples:
                raise ValueError("canonical prior PDF pixels changed")
            preserved.append(dict(slide=i+1,pixels_sha256=hashlib.sha256(after.samples).hexdigest()))
    receipt=dict(pptx_sha256=sha(HERE/"ILM_Figures_editable.pptx"),rendered_pdf_sha256=sha(target),slides=26,
        renderer_sha256=sha(Path(__file__)),
        previous_slide_renders_preserved=preserved,new_figures_sha256=figures,
        fresh_baseline_vs_append_preserved=fresh_preserved,older_renderer_difference_slides=renderer_differences,
        canonical_prefix_reused_without_image_changes=True,
        visual_review_required=True,headless_profile_exited=True,artifact_tool_available=False,
        source_deck_build_sha256=sha(HERE/"evidence/semantic_transfer_deck_build.json"),language_quality_claim=False)
    (HERE/"evidence/semantic_transfer_figure_render.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Rendered 26 slides and verified pixel-identical preservation of all 24 earlier slides.")


if __name__=="__main__":
    main()
