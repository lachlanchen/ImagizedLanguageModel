#!/usr/bin/env python3
"""Render isolated QA publication drafts after all owned model jobs exit.

No active paper, figure or Nutstore alias is changed. Successful conversion
still requires inspection before any later publication promotion.
"""
import argparse
from contextlib import contextmanager
import fcntl
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

from prepare_native_qa_publication import BASE, FIGURES, HERE, RESULT, ROOT, active_manifest, baseline, evidence, sha
from generate_assignment_editable_figures import NS


def admission():
    """Observe actual project processes; a queued assessment also owns its slot."""
    owned = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            cwd = (directory / 'cwd').resolve(strict=True)
            if cwd != ROOT and ROOT not in cwd.parents:
                continue
            words = (directory / 'cmdline').read_bytes().split(b'\0')
            candidates = [Path(os.fsdecode(w)) for w in words if w.endswith(b'.py')]
            candidates = [(cwd/p).resolve() if not p.is_absolute() else p.resolve() for p in candidates]
            script = next((str(p.relative_to(ROOT)) for p in candidates if ROOT in p.parents), '')
            if not script or not Path(script).name.startswith(('train_', 'infer_', 'run_')):
                continue
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            owned.append(dict(pid=int(directory.name), start_ticks=stat[19], script=script))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
        if line.split()[1].isdigit()}
    ram_ok = memory['MemAvailable'] >= 24*1024**2
    swap_fraction = (memory['SwapTotal']-memory['SwapFree'])/memory['SwapTotal'] if memory['SwapTotal'] else 0
    return dict(ready=not owned and ram_ok and swap_fraction <= .75, live_project_jobs=owned,
        available_RAM_GiB=memory['MemAvailable']/1024**2, swap_used_fraction=swap_fraction,
        shared_lock_not_yet_acquired=True)


@contextmanager
def render_slot():
    observed = admission()
    if not observed['ready']:
        raise RuntimeError('publication render is not admitted: '+json.dumps(observed))
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not admission()['ready']:
            raise RuntimeError('project model admission changed before render')
        yield


def verify_stage(stage):
    record = json.loads((stage/'preparation.json').read_text())
    if record['baseline_commit'] != BASE or record['source_result_sha256'] != sha(RESULT):
        raise ValueError('staged baseline/result changed')
    if record['builder_sha256'] != sha(HERE/'prepare_native_qa_publication.py'):
        raise ValueError('staged producer changed')
    for name, digest in record['output_files_sha256'].items():
        if sha(stage/name) != digest:
            raise ValueError('staged deck or TeX fragment changed: '+name)
    if (HERE/'ILM_Figures_editable.pptx').read_bytes() != baseline('ILM_Figures_editable.pptx'):
        raise ValueError('active deck is no longer the declared baseline')
    evidence()
    mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
    with zipfile.ZipFile(io.BytesIO(baseline('ILM_Figures_editable.pptx'))) as old, zipfile.ZipFile(stage/'draft.pptx') as new:
        for name in old.namelist():
            if name not in mutable and old.read(name) != new.read(name):
                raise ValueError('preserved baseline part changed: '+name)
        if len(ET.fromstring(new.read('ppt/presentation.xml')).find('p:sldIdLst', NS)) != 45:
            raise ValueError('complete staged45-slide deck required')
        for picture in record['pictures']:
            raw = new.read(picture['part'])
            if hashlib.sha256(raw).hexdigest() != picture['sha256']:
                raise ValueError('staged evidence image changed')
            original = RESULT.parent/picture['original']
            if sha(original) != picture['original_sha256']:
                raise ValueError('original evidence changed')
            with Image.open(io.BytesIO(raw)) as shown, Image.open(original) as source:
                desired = source.crop(picture['crop_box']) if picture['crop_box'] else source
                a, b = desired.convert('RGB'), shown.convert('RGB')
                if a.size != b.size or a.tobytes() != b.tobytes():
                    raise ValueError('staged pixels differ from declared original/crop')
    return record


def pdf_pixels(document):
    result = set()
    for page in document:
        for item in page.get_images(full=True):
            with Image.open(io.BytesIO(document.extract_image(item[0])['image'])) as image:
                result.add((image.size, hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()))
    return result


def render(stage):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for the publication process')
    # Refuse before starting conversion or creating output while model work is live.
    with render_slot():
        record = verify_stage(stage)
        before = active_manifest()
        output = stage/'render'
        output.mkdir(exist_ok=False)
        original = stage/'baseline.pptx'
        with original.open('xb') as handle:
            handle.write(baseline('ILM_Figures_editable.pptx'))
        command = ['soffice', '-env:UserInstallation='+(stage/'libreoffice-profile').as_uri(), '--headless', '--convert-to',
            'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir', str(output), str(original), str(stage/'draft.pptx')]
        with (stage/'libreoffice.log').open('x') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if time.monotonic()-started > 180:
                        raise TimeoutError('owned Office conversion exceeded180 seconds')
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                if process.returncode:
                    raise RuntimeError('owned Office conversion failed')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
        old, new = fitz.open(output/'baseline.pdf'), fitz.open(output/'draft.pdf')
        prefix = fitz.open(stream=baseline('ILM_Figures_review.pdf'), filetype='pdf')
        if (len(old), len(new), len(prefix)) != (43, 45, 43):
            raise ValueError('converted slide counts differ')
        preserved = []
        for i in range(43):
            a, b = old[i].get_pixmap(alpha=False), new[i].get_pixmap(alpha=False)
            if (a.width, a.height, a.samples) != (b.width, b.height, b.samples):
                raise ValueError('prior slide appearance changed: '+str(i+1))
            preserved.append(hashlib.sha256(b.samples).hexdigest())
        review = fitz.open(); review.insert_pdf(prefix); review.insert_pdf(new, from_page=43, to_page=44)
        review.save(output/'ILM_Figures_review.pdf')
        figures = {}
        with zipfile.ZipFile(stage/'draft.pptx') as archive:
            required = set()
            for picture in record['pictures']:
                with Image.open(io.BytesIO(archive.read(picture['part']))) as image:
                    required.add((image.size, hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()))
        for i, name in enumerate(FIGURES, 43):
            figure = fitz.open(); figure.insert_pdf(new, from_page=i, to_page=i)
            figure.save(output/(name+'.pdf'))
            if i == 44 and not required <= pdf_pixels(figure):
                raise ValueError('rendered figure changed original input/reference/answer pixels')
            new[i].get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(output/(name+'.png'))
            (output/(name+'.svg')).write_text(new[i].get_svg_image())
            figures[name] = {suffix:sha(output/(name+'.'+suffix)) for suffix in ('pdf','png','svg')}
        for path in sorted(output.glob('*.pdf')):
            check = subprocess.run(['qpdf','--check',str(path)],capture_output=True,text=True)
            (output/(path.stem+'.qpdf.log')).write_text(check.stdout+check.stderr)
            if check.returncode: raise ValueError('PDF structure check failed: '+path.name)
        if active_manifest() != before:
            raise ValueError('active publication files changed during isolated rendering')
        rendered = dict(source_preparation_sha256=sha(stage/'preparation.json'),
            renderer_sha256=sha(__file__), pptx_sha256=sha(stage/'draft.pptx'),
            review_pdf_sha256=sha(output/'ILM_Figures_review.pdf'), preserved_prior_slide_pixels=preserved,
            figure_files_sha256=figures, displayed_images_pixel_identical=True, active_publication_files_unchanged=True,
            font_substitution='Arial rendered with Liberation Sans',
            rendered=True, visual_review_completed=False, PowerPoint_app_inspection=False,
            active_manuscript_updated=False, Nutstore_updated=False, general_ilm_goal_complete=False)
        (output/'render.json').write_text(json.dumps(rendered,indent=2)+'\n')
        print('Rendered isolated45-slide draft; inspect slides44/45 before promotion.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('check-admission','verify','render'), required=True)
    parser.add_argument('--input', type=Path)
    args = parser.parse_args()
    if args.stage == 'check-admission':
        print(json.dumps(admission(),indent=2))
    elif args.input is None:
        parser.error('--input is required for verify/render')
    elif args.stage == 'verify':
        verify_stage(args.input.resolve()); print('Isolated deck and original evidence verified; no render started.')
    else:
        render(args.input.resolve())
