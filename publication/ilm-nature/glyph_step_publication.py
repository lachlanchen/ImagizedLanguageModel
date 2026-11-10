#!/usr/bin/env python3
"""Stage S39/S40 and their completed results together; promote only reviewed PDFs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl
from PIL import Image

from native_flow_manuscript import build_slot, owned_command, protected, expanded, TOOL, TOOL_SHA
from render_native_qa_publication import pdf_pixels
import motor_gate_publication as previous
import motor_gate_figures as motor_figure
import motor_gate_text as motor_text
from glyph_step_data import HERE, ROOT, collect, load, require, sha
from glyph_step_figures import append, FIGURE, NS
from glyph_step_text import RESULTS, NOTE

BASE = previous.BASE
DOCS = previous.DOCS
CODE = (*previous.CODE, 'glyph_step_publication.py', 'glyph_step_data.py',
        'glyph_step_figures.py', 'glyph_step_text.py')
NAMES = (*previous.NAMES, 'glyph_step_results.tex', 'glyph_step_note.tex',
         *('figures/'+FIGURE+'.'+suffix for suffix in ('pdf', 'png', 'svg')))
RESULT_INSERT = '\\input{glyph_step_results.tex}\n\n'
NOTE_INSERT = '\\input{glyph_step_note.tex}\n\n'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def identity():
    result = {}
    for name in CODE:
        path = HERE / name
        require(subprocess.check_output(['git', 'show', 'HEAD:publication/ilm-nature/'+name], cwd=ROOT)
                == path.read_bytes(), 'commit publication sources first: '+name)
        result[name] = sha(path)
    return result


def source_audit():
    code = identity(); motor = motor_figure.source_data(); previous.measure_check(motor)
    glyph = collect()
    for token in ['0.951778', '0.972856', '194/256', '229/256', '0.341013', '0.354426',
                  '12/16', '3/16', '688,128', '17,848,833', '3.208']:
        require(token in NOTE, 'measured result absent from Note 61: '+token)
    require(glyph['fixed_documents'] == [0, 5, 10, 15] and len(glyph['cells']) == 32,
            'fixed C3 example grid differs')
    require('No main-fit complete-answer result' in NOTE, 'unfinished candidate must remain explicit')
    bound = dict(motor['source_files_sha256'])
    bound.update(glyph['source_files_sha256'])
    for path in motor_figure.SOURCES.iterdir():
        if path.is_file(): bound[str(path.relative_to(ROOT))] = sha(path)
    for name in previous.EXTRA:
        path = ROOT/'references/evidence'/name; bound[str(path.relative_to(ROOT))] = sha(path)
    for name in ['references/manuscript_motor_gate_revision_plan.md',
                 'references/manuscript_glyph_step_revision_plan.md']:
        bound[name] = sha(ROOT/name)
    for name in ['scripts/audit_cached_glyph_teacher_labels.py', 'scripts/audit_glyph_target_coverage.py',
                 'references/cached_glyph_teacher_agreement_result.md', 'references/glyph_target_coverage_result.md',
                 *('references/'+p for p in ['semantic_readout_distillation_protocol.md',
                    'contrastive_glyph_readout_protocol.md', 'robust_glyph_decoder_protocol.md',
                    'semantic_motor_flow_protocol.md', 'spatial_semantic_motor_protocol.md'])]:
        bound[name] = sha(ROOT/name)
    return dict(code_sha256=code, evidence_sha256=bound, motor=motor, glyph=glyph,
                neural_calls=0, Office_calls=0, LaTeX_calls=0, active_publication_changed=False)


def prepare(stage):
    with build_slot():
        data = source_audit(); before = protected()
        for name in before:
            require((HERE/name).read_bytes() == previous.baseline(name), 'active baseline differs: '+name)
        stage.mkdir(parents=True, exist_ok=False); paper = stage/'paper'; paper.mkdir()
        for path in HERE.iterdir():
            if path.is_file() and path.suffix in ('.tex', '.sty') and not path.name.startswith('redlined_'):
                shutil.copy2(path, paper/path.name)
        shutil.copytree(HERE/'figures', paper/'figures')
        old_figure = motor_figure.append(paper)
        new_figure = append(paper)
        for name, text in [('motor_gate_results.tex', motor_text.RESULTS), ('motor_gate_note.tex', motor_text.NOTE),
                           ('glyph_step_results.tex', RESULTS), ('glyph_step_note.tex', NOTE)]:
            (paper/name).write_text(text)
        article = (paper/'article_content.tex').read_text(); anchor = r'\section*{Discussion}'
        require(article.count(anchor) == 1, 'unique main Results boundary required')
        (paper/'article_content.tex').write_text(article.replace(anchor,
            previous.RESULT_INSERT+RESULT_INSERT+anchor))
        supplement = (paper/'supplementary_information.tex').read_text()
        anchor = '\\clearpage\n\\section*{Reproduction}'
        require(supplement.count(anchor) == 1, 'unique SI Reproduction boundary required')
        (paper/'supplementary_information.tex').write_text(supplement.replace(anchor,
            previous.NOTE_INSERT+NOTE_INSERT+anchor))
        require(protected() == before, 'active publication changed during staging')
        save(stage/'preparation.json', dict(baseline_commit=BASE, code_sha256=data['code_sha256'],
            evidence_sha256=data['evidence_sha256'], active_before_sha256=before,
            figures=[old_figure, new_figure], abstract_unchanged=True,
            completed_copy_C3_and_helper_results_included=True, unfinished_tail_language_results_included=False,
            prepared_sha256={str(p.relative_to(paper)): sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
            delivery_date=time.strftime('%Y%m%d'), active_publication_updated=False, general_ILM_goal_complete=False))


def verify(stage):
    meta = load(stage/'preparation.json')
    require(protected() == meta['active_before_sha256'], 'active reviewed publication changed')
    require(identity() == meta['code_sha256'], 'publication code changed after staging')
    for name, digest in meta['evidence_sha256'].items():
        require(sha(ROOT/name) == digest, 'bound evidence changed: '+name)
    for name, digest in meta['prepared_sha256'].items():
        require(sha(stage/'paper'/name) == digest, 'prepared source or figure changed: '+name)
    return meta


def render(stage):
    with build_slot():
        meta = verify(stage); folder = stage/'render'; folder.mkdir(exist_ok=False)
        predecessor = stage/'predecessor.pptx'; shutil.copy2(HERE/'ILM_Figures_editable.pptx', predecessor)
        owned_command(['soffice', '-env:UserInstallation='+(stage/'office-profile').as_uri(), '--headless',
            '--convert-to', 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir', str(folder), str(predecessor), str(stage/'paper/ILM_Figures_editable.pptx')],
            stage, stage/'office.log', 180)
        old = fitz.open(folder/'predecessor.pdf'); new = fitz.open(folder/'ILM_Figures_editable.pdf')
        require((len(old), len(new)) == (53, 55), 'actual slide counts differ')
        for index in range(53):
            a, b = old[index].get_pixmap(alpha=False), new[index].get_pixmap(alpha=False)
            require((a.width, a.height, a.samples) == (b.width, b.height, b.samples),
                    'predecessor slide appearance changed')
        with zipfile.ZipFile(HERE/'ILM_Figures_editable.pptx') as a, zipfile.ZipFile(stage/'paper/ILM_Figures_editable.pptx') as b:
            mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
            require(all(a.read(n) == b.read(n) for n in a.namelist() if n not in mutable),
                    'old slide/chart/media part changed')
            for figure in meta['figures']:
                for row in figure['charts']:
                    node = ET.fromstring(b.read(row['part']))
                    values = [float(n.text) for n in node.findall(
                        'c:chart/c:plotArea/c:barChart/c:ser/c:val/c:numRef/c:numCache/c:pt/c:v', NS)]
                    book = openpyxl.load_workbook(io.BytesIO(b.read(row['workbook'])), data_only=True)
                    require(values == row['values'] and all(abs(book.active.cell(i+2, 2).value-v) < 1e-10
                            for i, v in enumerate(values)), 'native chart/workbook data changed')
                for row in figure['media']:
                    require(hashlib.sha256(b.read(row['part'])).hexdigest() == row['sha256'],
                            'embedded raw source PNG changed')
        raw_paths = [ROOT/row['source'] for row in meta['figures'][0]['media']]
        raw_paths += [ROOT/row['file'] for row in meta['figures'][1]['media']]
        expected = {(Image.open(p).size, hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest())
                    for p in raw_paths}
        require(expected <= pdf_pixels(new), 'rendered PDF did not retain the complete raw image objects')
        review = fitz.open(HERE/'ILM_Figures_review.pdf'); require(len(review) == 53, 'review baseline differs')
        review.insert_pdf(new, from_page=53, to_page=54); review.save(stage/'paper/ILM_Figures_review.pdf')
        figures = {}
        for index, name in [(53, motor_figure.FIGURE), (54, FIGURE)]:
            doc = fitz.open(); doc.insert_pdf(new, from_page=index, to_page=index)
            doc.save(stage/'paper/figures'/(name+'.pdf'))
            new[index].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(stage/'paper/figures'/(name+'.png'))
            (stage/'paper/figures'/(name+'.svg')).write_text(new[index].get_svg_image())
            figures[name] = {suffix: sha(stage/'paper/figures'/(name+'.'+suffix)) for suffix in ['pdf', 'png', 'svg']}
        save(stage/'render.json', dict(preparation_sha256=sha(stage/'preparation.json'), native_slides=55,
            native_charts=61, prior53_slide_appearances_exact=True, prior57_chart_parts_exact=True,
            four_native_chart_workbooks_exact=True, complete_embedded_output_pixels_preserved=True,
            PowerPoint_application_inspected=False, actual_visual_review_pending=True,
            font_substitution='Arial rendered with Liberation Sans', figure_files_sha256=figures))


def redline(stage, source, stem):
    require(sha(TOOL) == TOOL_SHA, 'pinned latexdiff changed')
    folder = stage/'redline_inputs'/stem; folder.mkdir(parents=True, exist_ok=False); old, new = {}, {}
    (folder/'old.tex').write_text(expanded(source, previous.baseline, old))
    (folder/'new.tex').write_text(expanded(source, lambda n: (stage/'paper'/n).read_bytes(), new))
    result = subprocess.run(['perl', str(TOOL), '--encoding=utf8', '--type=CFONT', '--subtype=SAFE',
        '--floattype=IDENTICAL', '--math-markup=0', 'old.tex', 'new.tex'], cwd=folder,
        capture_output=True, text=True, check=True, timeout=90)
    notice = r'\noindent{\color{blue}\footnotesize Change-marked copy relative to the preserved baseline (2ca07ea0).}\par\medskip'+'\n'
    (stage/'paper'/(stem+'.tex')).write_text(result.stdout.replace(r'\begin{document}',
        r'\begin{document}'+'\n'+notice, 1))
    return dict(old_inputs_sha256=old, new_inputs_sha256=new, pinned_latexdiff_sha256=TOOL_SHA)


def build(stage):
    with build_slot():
        verify(stage); reviewed = load(stage/'figure_review.json')
        require(reviewed['render_sha256'] == sha(stage/'render.json') and reviewed['S39_actually_viewed']
                and reviewed['S40_actually_viewed'] and reviewed['approved_for_build'],
                'actually inspect both figure exports before building')
        logs = stage/'build_logs'; logs.mkdir(exist_ok=False)
        redlines = {stem: redline(stage, source, stem) for source, stem in
                    [('manuscript_review.tex', 'redlined_manuscript'),
                     ('supplementary_information.tex', 'redlined_supplementary_information')]}
        pdfs, pages, locations = {}, {}, {}
        for stem in DOCS:
            aux = stage/'aux'/stem; aux.mkdir(parents=True, exist_ok=False)
            owned_command(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                '-outdir='+str(aux), stem+'.tex'], stage/'paper', logs/(stem+'.log'), 180)
            log = (aux/(stem+'.log')).read_text()
            require(not any(token in log for token in ['Overfull', 'undefined references', 'undefined citations', 'LaTeX Error']),
                    'new layout/reference error: '+stem)
            pdf = stage/'paper'/(stem+'.pdf'); shutil.copy2(aux/(stem+'.pdf'), pdf)
            owned_command(['qpdf', '--check', str(pdf)], stage/'paper', logs/(stem+'.qpdf.log'), 30)
            doc = fitz.open(pdf); text = ' '.join(' '.join(p.get_text().split()) for p in doc)
            require('0.972856' in text and '0.951778' in text, 'completed copy comparison absent')
            if stem in ['supplementary_information', 'redlined_supplementary_information']:
                require('Supplementary Note 61' in text and '0.810537' in text and '0.341013' in text,
                        'new SI controls absent')
            pdfs[pdf.name] = sha(pdf); pages[stem] = len(doc)
            locations[stem] = {token: [i+1 for i, page in enumerate(doc) if token in ' '.join(page.get_text().split())]
                for token in ['Improved glyph copying', 'Sharp known-form', 'Supplementary Note 60',
                              'Supplementary Note 61', '0.972856', '0.810537', 'Fig. S40']}
            print('Built '+stem+': '+str(len(doc))+' pages', flush=True)
        verify(stage)
        save(stage/'build.json', dict(PDFs_sha256=pdfs, pages=pages, locations=locations, redlines=redlines,
            preparation_sha256=sha(stage/'preparation.json'), render_sha256=sha(stage/'render.json'),
            no_overfull_or_undefined=True, actual_PDF_review_pending=True, active_publication_updated=False))


def promote(stage):
    with build_slot():
        meta = verify(stage); built = load(stage/'build.json'); reviewed = load(stage/'review.json')
        require(reviewed['build_sha256'] == sha(stage/'build.json') and reviewed['all_changed_PDF_pages_actually_viewed'],
                'actual matching PDF inspection required')
        require(reviewed['PDFs_sha256'] == built['PDFs_sha256'] and set(reviewed['staged_files_sha256']) == set(NAMES),
                'reviewed promotion set incomplete')
        for name, digest in reviewed['staged_files_sha256'].items():
            require(sha(stage/'paper'/name) == digest, 'reviewed artifact changed')
        article = (stage/'paper/article_content.tex').read_text()
        supplement = (stage/'paper/supplementary_information.tex').read_text()
        require(article.replace(previous.RESULT_INSERT, '').replace(RESULT_INSERT, '')
                == previous.baseline('article_content.tex').decode(), 'unplanned main-text change')
        require(supplement.replace(previous.NOTE_INSERT, '').replace(NOTE_INSERT, '')
                == previous.baseline('supplementary_information.tex').decode(), 'unplanned SI change')
        require((stage/'paper/frontmatter.tex').read_bytes() == previous.baseline('frontmatter.tex'), 'abstract changed')
        folder = HERE/'baselines/pre-glyph-step-2ca07ea0'; folder.mkdir(exist_ok=False)
        save(folder/'baseline_manifest.json', dict(source_commit=BASE,
            files_sha256=meta['active_before_sha256'], recoverable_from_Git=True))
        for name in NAMES: shutil.copy2(stage/'paper'/name, HERE/name)
        for name in ['preparation', 'render', 'figure_review', 'build', 'review']:
            shutil.copy2(stage/(name+'.json'), HERE/'evidence'/('glyph_step_'+name+'.json'))
        save(stage/'promotion.json', dict(files_sha256={name: sha(HERE/name) for name in NAMES},
            baseline_commit=BASE, active_publication_updated=True, general_ILM_goal_complete=False))


def sync(stage):
    with build_slot():
        from promote_native_flow_manuscript import DEST, ALIASES, atomic_copy
        promoted = load(stage/'promotion.json'); prepared = load(stage/'preparation.json')
        for name, digest in promoted['files_sha256'].items():
            require(sha(HERE/name) == digest, 'promoted artifact changed')
        require(DEST.is_dir(), 'authorized Nutstore ILM destination absent')
        package = DEST/('ILM_NatureStyle_'+prepared['delivery_date']+'_GlyphFeedbackControls')
        package.mkdir(exist_ok=False); files, aliases = {}, {}
        for path in sorted((stage/'paper').rglob('*')):
            if path.is_file():
                relative = path.relative_to(stage/'paper'); atomic_copy(path, package/relative)
                files[str(relative)] = sha(package/relative)
        for name in ['preparation', 'render', 'figure_review', 'build', 'review', 'promotion']:
            path = stage/(name+'.json'); atomic_copy(path, package/'evidence'/path.name)
            files['evidence/'+path.name] = sha(path)
        for name, alias in ALIASES.items():
            target = DEST/alias
            if target.exists():
                digest = sha(target)
                require(digest in [prepared['active_before_sha256'][name], sha(HERE/name)],
                        'preserve Nutstore alias with another revision')
                if digest != sha(HERE/name):
                    backup = target.with_name(target.stem+'_before_glyph_feedback_'+digest[:12]+target.suffix)
                    if backup.exists(): require(sha(backup) == digest, 'different backup exists')
                    else: atomic_copy(target, backup)
            atomic_copy(HERE/name, target); require(sha(target) == sha(HERE/name), 'local alias copy differs')
            aliases[alias] = sha(target)
        receipt = dict(package=package.name, files_sha256=files, aliases_sha256=aliases,
                       local_hashes_verified=True, cloud_propagation_verified=False)
        save(stage/'sync.json', receipt); save(HERE/'evidence/glyph_step_sync.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['source-audit', 'prepare', 'render', 'build', 'promote', 'sync'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if args.stage == 'source-audit':
        require(not output.exists(), 'preserve previous source audits'); save(output, source_audit())
    else: globals()[args.stage](output)
