#!/usr/bin/env python3
"""Build matching article PDFs and enforce native-figure pixel preservation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STAGE=ROOT/'artifacts/native_reader_writing_publication_v2'


def main():
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for directory,name in [('reading','manuscript'),('review','manuscript_review')]:
        folder=STAGE/directory;folder.mkdir(exist_ok=True)
        result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
            '-outdir='+str(folder),name+'.tex'],cwd=HERE,env=env,capture_output=True,text=True)
        (folder/'native_reader_build.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError('Article build failed: '+name)
        log=(folder/(name+'.log')).read_text()
        if any(t in log for t in ['Overfull','undefined references','undefined citations','LaTeX Error']):
            raise ValueError('Article layout/reference error: '+name)
    result=subprocess.run([sys.executable,str(HERE/'build_native_reader_writing_redline.py')],
        cwd=HERE,env=env,capture_output=True,text=True)
    (STAGE/'native_reader_redline_build.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Matching redline build failed')
    result=subprocess.run([sys.executable,str(HERE/'verify_native_reader_writing_figure.py')],
        cwd=HERE,env=env,capture_output=True,text=True)
    (STAGE/'native_reader_figure_audit.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Native chart/pixel preservation audit failed')
    for directory,name in [('reading','manuscript'),('review','manuscript_review')]:
        shutil.copy2(STAGE/directory/(name+'.pdf'),HERE/(name+'.pdf'))
    data=json.loads((STAGE/'figure_audit.json').read_text())
    print(json.dumps(dict(builds_complete=True,original_pixels_preserved=True,
        PDF_pages={k:v['pages'] for k,v in data['PDFs'].items()},manual_visual_QA_still_required=True)))


if __name__=='__main__':main()
