#!/usr/bin/env python3
"""Render the isolated46-slide native-flow revision and preserve prior slides."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

import fitz
from PIL import Image

from generate_assignment_editable_figures import NS
from prepare_native_qa_publication import FIGURES, HERE, ROOT, active_manifest, baseline, sha
from render_native_qa_publication import pdf_pixels, render_slot, verify_stage as verify_qa


def verify(stage,source):
    verify_qa(source)
    record=json.loads((stage/'preparation.json').read_text())
    if (record['builder_sha256']!=sha(HERE/'prepare_native_flow_publication.py')
            or record['source_stage_preparation_sha256']!=sha(source/'preparation.json')):
        raise ValueError('staged producer or predecessor changed')
    for name,digest in record['output_files_sha256'].items():
        if sha(stage/name)!=digest:
            raise ValueError('staged output changed: '+name)
    mutable={'[Content_Types].xml','ppt/presentation.xml','ppt/_rels/presentation.xml.rels'}
    with zipfile.ZipFile(source/'draft.pptx') as old,zipfile.ZipFile(stage/'draft.pptx') as new:
        for name in old.namelist():
            if name not in mutable and old.read(name)!=new.read(name):
                raise ValueError('prior package part changed: '+name)
        if len(ET.fromstring(new.read('ppt/presentation.xml')).find('p:sldIdLst',NS))!=46:
            raise ValueError('complete46-slide stage required')
        for picture in record['pictures']:
            if hashlib.sha256(new.read(picture['part'])).hexdigest()!=picture['sha256']:
                raise ValueError('original model evidence image changed')
    return record


def render(stage,source):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('hide CUDA for publication conversion')
    with render_slot():
        record=verify(stage,source)
        before=active_manifest()
        output=stage/'render';output.mkdir(exist_ok=False)
        predecessor=stage/'predecessor.pptx'
        predecessor.write_bytes((source/'draft.pptx').read_bytes())
        command=['soffice','-env:UserInstallation='+(stage/'libreoffice-profile').as_uri(),
            '--headless','--convert-to',
            'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir',str(output),str(predecessor),str(stage/'draft.pptx')]
        with (stage/'libreoffice.log').open('x') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic()-started>180:
                        raise TimeoutError('owned Office conversion exceeded180 seconds')
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                if process.returncode:raise RuntimeError('Office conversion failed')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        old,new=fitz.open(output/'predecessor.pdf'),fitz.open(output/'draft.pdf')
        prefix=fitz.open(stream=baseline('ILM_Figures_review.pdf'),filetype='pdf')
        if (len(old),len(new),len(prefix))!=(45,46,43):
            raise ValueError('converted page counts differ')
        preserved=[]
        for i in range(45):
            a,b=old[i].get_pixmap(alpha=False),new[i].get_pixmap(alpha=False)
            if (a.width,a.height,a.samples)!=(b.width,b.height,b.samples):
                raise ValueError('prior rendered slide appearance changed: '+str(i+1))
            preserved.append(hashlib.sha256(b.samples).hexdigest())
        review=fitz.open();review.insert_pdf(prefix);review.insert_pdf(new,from_page=43,to_page=45)
        review.save(output/'ILM_Figures_review.pdf')
        figures={}
        required=set()
        with zipfile.ZipFile(stage/'draft.pptx') as archive:
            for picture in record['pictures']:
                with Image.open(io.BytesIO(archive.read(picture['part']))) as image:
                    required.add((image.size,hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()))
        for i,name in enumerate((*FIGURES,'Figure_S31_native_visual_flow'),43):
            figure=fitz.open();figure.insert_pdf(new,from_page=i,to_page=i);figure.save(output/(name+'.pdf'))
            if i==45 and not required<=pdf_pixels(figure):
                raise ValueError('rendered diagram changes original evidence pixels')
            new[i].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(output/(name+'.png'))
            (output/(name+'.svg')).write_text(new[i].get_svg_image())
            figures[name]={suffix:sha(output/(name+'.'+suffix)) for suffix in ('pdf','png','svg')}
        for path in output.glob('*.pdf'):
            check=subprocess.run(['qpdf','--check',str(path)],capture_output=True,text=True)
            (output/(path.stem+'.qpdf.log')).write_text(check.stdout+check.stderr)
            if check.returncode:raise ValueError('PDF structure check failed: '+path.name)
        if active_manifest()!=before:
            raise ValueError('active publication files changed during isolated rendering')
        rendered=dict(renderer_sha256=sha(__file__),source_preparation_sha256=sha(stage/'preparation.json'),
            pptx_sha256=sha(stage/'draft.pptx'),preserved_prior45_slide_pixels=preserved,
            review_pdf_sha256=sha(output/'ILM_Figures_review.pdf'),figure_files_sha256=figures,
            displayed_evidence_images_pixel_identical=True,active_publication_files_unchanged=True,
            rendered=True,visual_review_completed=False,PowerPoint_app_inspection=False,
            font_substitution='Arial rendered with Liberation Sans',active_manuscript_updated=False,
            Nutstore_updated=False,general_ilm_goal_complete=False)
        (output/'render.json').write_text(json.dumps(rendered,indent=2)+'\n')
        print('Rendered46 native slides; prior45 match exactly. Inspect changed figure before promotion.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/native_qa_publication_stage_v1')
    args=parser.parse_args();render(args.input.resolve(),args.source.resolve())
