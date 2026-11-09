#!/usr/bin/env python3
"""Stage and verify the complete-learning paper without replacing active files."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import fitz
import openpyxl

from complete_pixel_learning_figures import FIGURES, EVIDENCE, append
from complete_pixel_learning_text import SUMMARY, RESULTS, DISCUSSION, NOTE
from generate_assignment_editable_figures import NS
from native_flow_manuscript import HERE, ROOT, DOCS, TOOL, TOOL_SHA, build_slot, expanded, owned_command, protected, require, save, sha
from render_native_qa_publication import pdf_pixels

BASE = 'd387a49ed923cf5a912cda9c0db143cbd95cf2e0'
CODE = ('complete_pixel_publication.py', 'complete_pixel_learning_figures.py', 'complete_pixel_learning_text.py')
RESULT_INSERT = '\\subsection*{Complete image-answer fitting and failed transfer}\n\\input{complete_pixel_learning_results.tex}\n\n'
DISCUSSION_ANCHOR = 'task, beyond the conditional-likelihood gain established here.\n'


def load(path):
    import json
    return json.loads(path.read_text())


def baseline(name):
    return subprocess.check_output(['git', 'show', BASE + ':publication/ilm-nature/' + name], cwd=ROOT)


def prepare(stage):
    with build_slot():
        audit = load(EVIDENCE / 'independent_audit.json')
        review = load(EVIDENCE / 'review_and_manifest.json')
        require(review['all32_final_training_outputs_and_references_actually_viewed']
                and review['all48_development_outputs_and_owning_prompts_and_all_references_actually_viewed'],
                'complete raw-output review required')
        require(audit['groups']['final_train_cmrc_normal']['exact'] == 32
                and audit['groups']['development_cmrc_normal']['exact'] == 0
                and audit['groups']['development_dolly_normal']['exact'] == 0,
                'completed independently audited fitting and transfer result required')
        cold = ROOT / 'references/evidence/complete_pixel_runtime_qualification'
        require(load(cold / 'report.json')['all_actual_PNG_replays_bit_exact'], 'actual cold inference qualification required')
        bound = {}
        for folder in (EVIDENCE, cold):
            for name, digest in load(folder / 'review_and_manifest.json')['files_sha256'].items():
                require(sha(folder / name) == digest, 'source evidence changed: ' + name)
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    bound[str(path.relative_to(ROOT))] = sha(path)
        monitor_path = ROOT / 'references/evidence/complete_pixel_language_update4000/review_and_manifest.json'
        bound[str(monitor_path.relative_to(ROOT))] = sha(monitor_path)
        monitors = load(monitor_path)['matched_whole_cohort_monitors']
        before = protected()
        for name in before:
            require((HERE / name).read_bytes() == baseline(name), 'active source differs from the reviewed baseline: ' + name)
        code = {name: sha(HERE / name) for name in CODE}
        for name, digest in code.items():
            raw = subprocess.check_output(['git', 'show', 'HEAD:publication/ilm-nature/' + name], cwd=ROOT)
            require(hashlib.sha256(raw).hexdigest() == digest, 'commit the publication source first: ' + name)
        stage.mkdir(parents=True, exist_ok=False)
        paper = stage / 'paper'
        paper.mkdir()
        for source in HERE.iterdir():
            if source.is_file() and source.suffix in ('.tex', '.sty') and not source.name.startswith('redlined_'):
                shutil.copy2(source, paper / source.name)
        shutil.copytree(HERE / 'figures', paper / 'figures')
        figures = append(paper, audit, monitors)
        (paper / 'complete_pixel_learning_results.tex').write_text(RESULTS)
        (paper / 'complete_pixel_learning_note.tex').write_text(NOTE)
        front = (paper / 'frontmatter.tex').read_text()
        anchor = r'\newcommand{\PaperSummary}'
        require(front.count(anchor) == 1, 'unique abstract required')
        (paper / 'frontmatter.tex').write_text(front[:front.index(anchor)] + anchor + '{' + SUMMARY + '}\n')
        article = (paper / 'article_content.tex').read_text()
        require(article.count(r'\section*{Discussion}') == 1 and article.count(DISCUSSION_ANCHOR) == 1, 'unique insertion anchors required')
        article = article.replace(r'\section*{Discussion}', RESULT_INSERT + r'\section*{Discussion}')
        article = article.replace(DISCUSSION_ANCHOR, DISCUSSION_ANCHOR + '\n' + DISCUSSION + '\n')
        (paper / 'article_content.tex').write_text(article)
        supplement = (paper / 'supplementary_information.tex').read_text()
        anchor = '\\clearpage\n\\section*{Reproduction}'
        require(supplement.count(anchor) == 1, 'unique supplement anchor required')
        (paper / 'supplementary_information.tex').write_text(supplement.replace(anchor, '\\input{complete_pixel_learning_note.tex}\n\n' + anchor))
        require(protected() == before, 'active publication changed during staging')
        save(stage / 'preparation.json', dict(baseline_commit=BASE, code_sha256=code,
            bound_evidence_sha256=bound, active_before_sha256=before, figures=figures,
            prepared_files_sha256={str(p.relative_to(paper)): sha(p) for p in sorted(paper.rglob('*')) if p.is_file()},
            active_publication_updated=False, general_ilm_goal_complete=False))
        print('Prepared 50 native slides, 54 charts and complete-learning manuscript sources.')


def verify(stage):
    meta = load(stage / 'preparation.json')
    require(protected() == meta['active_before_sha256'], 'active publication changed')
    for name, digest in meta['code_sha256'].items():
        require(sha(HERE / name) == digest, 'staged publication builder changed')
    for name, digest in meta['bound_evidence_sha256'].items():
        require(sha(ROOT / name) == digest, 'source evidence changed')
    for name, digest in meta['prepared_files_sha256'].items():
        require(sha(stage / 'paper' / name) == digest, 'prepared source changed: ' + name)
    return meta


def render(stage):
    with build_slot():
        meta = verify(stage)
        folder = stage / 'render'
        folder.mkdir(exist_ok=False)
        predecessor = stage / 'predecessor.pptx'
        shutil.copy2(HERE / 'ILM_Figures_editable.pptx', predecessor)
        owned_command(['soffice', '-env:UserInstallation=' + (stage / 'office-profile').as_uri(), '--headless',
            '--convert-to', 'pdf:impress_pdf_Export:{"UseLosslessCompression":{"type":"boolean","value":"true"},"ReduceImageResolution":{"type":"boolean","value":"false"}}',
            '--outdir', str(folder), str(predecessor), str(stage / 'paper/ILM_Figures_editable.pptx')],
            stage, stage / 'office.log', 180)
        old, new = fitz.open(folder / 'predecessor.pdf'), fitz.open(folder / 'ILM_Figures_editable.pdf')
        require((len(old), len(new)) == (48, 50), 'native slide count differs')
        for i in range(48):
            a, b = old[i].get_pixmap(alpha=False), new[i].get_pixmap(alpha=False)
            require((a.width, a.height, a.samples) == (b.width, b.height, b.samples), 'old slide appearance changed: ' + str(i + 1))
        with zipfile.ZipFile(HERE / 'ILM_Figures_editable.pptx') as a, zipfile.ZipFile(stage / 'paper/ILM_Figures_editable.pptx') as b:
            mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
            require(all(a.read(n) == b.read(n) for n in a.namelist() if n not in mutable), 'prior editable part changed')
            chart = ET.fromstring(b.read('ppt/charts/chart54.xml'))
            values = [float(n.text) for n in chart.findall('c:chart/c:plotArea/c:barChart/c:ser/c:val/c:numRef/c:numCache/c:pt/c:v', NS)]
            book = openpyxl.load_workbook(io.BytesIO(b.read('ppt/embeddings/complete_pixel_NLL.xlsx')), data_only=True)
            require(all(abs(book.active.cell(i + 2, 2).value - value) <= abs(value) * 1e-14 for i, value in enumerate(values))
                    and len(values) == 4, 'native chart and linked workbook differ')
        review = fitz.open(HERE / 'ILM_Figures_review.pdf')
        require(len(review) == 48, 'review predecessor count differs')
        review.insert_pdf(new, from_page=48, to_page=49)
        review.save(stage / 'paper/ILM_Figures_review.pdf')
        figures = {}
        for i, name in enumerate(FIGURES, 48):
            figure = fitz.open()
            figure.insert_pdf(new, from_page=i, to_page=i)
            figure.save(stage / 'paper/figures' / (name + '.pdf'))
            new[i].get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False).save(stage / 'paper/figures' / (name + '.png'))
            (stage / 'paper/figures' / (name + '.svg')).write_text(new[i].get_svg_image())
            if i == 49:
                expected = {(tuple(m['RGB_size']), m['RGB_pixels_sha256']) for m in meta['figures']['media']}
                require(expected <= pdf_pixels(figure), 'figure conversion changed original/cropped image pixels')
            figures[name] = {suffix: sha(stage / 'paper/figures' / (name + '.' + suffix)) for suffix in ('pdf', 'png', 'svg')}
        save(stage / 'render.json', dict(preparation_sha256=sha(stage / 'preparation.json'),
            prior48_slide_appearances_exact=True, prior53_chart_and_workbook_parts_preserved=True,
            new_chart_matches_embedded_workbook=True, figure_files_sha256=figures,
            review_pdf_sha256=sha(stage / 'paper/ILM_Figures_review.pdf'), original_evidence_pixels_preserved=True,
            PowerPoint_application_inspected=False, font_substitution='Arial rendered with Liberation Sans',
            actual_visual_review_pending=True))
        print('Rendered 50 slides. Prior48 match exactly; inspect S34/S35 before building.')


def redline(stage, source, stem):
    require(sha(TOOL) == TOOL_SHA, 'pinned latexdiff changed')
    folder = stage / 'redline_inputs' / stem
    folder.mkdir(parents=True, exist_ok=False)
    old, new = {}, {}
    (folder / 'old.tex').write_text(expanded(source, baseline, old))
    (folder / 'new.tex').write_text(expanded(source, lambda name: (stage / 'paper' / name).read_bytes(), new))
    result = subprocess.run(['perl', str(TOOL), '--encoding=utf8', '--type=CFONT', '--subtype=SAFE',
        '--floattype=IDENTICAL', '--math-markup=0', 'old.tex', 'new.tex'], cwd=folder, capture_output=True, text=True, check=True)
    notice = r'\noindent{\color{blue}\footnotesize Change-marked copy relative to the preserved writing-controls baseline (d387a49).}\par\medskip' + '\n'
    text = result.stdout.replace(r'\begin{document}', r'\begin{document}' + '\n' + notice, 1)
    marked = False
    if source == 'manuscript_review.tex':
        text, count = re.subn(r'(\\begin\{thebibliography\}\{[^}]+\})',
            lambda match: match[0] + '\n' + r'\setlength{\itemsep}{0pt}\setlength{\parsep}{0pt}', text)
        require(count == 1, 'unique redline bibliography required for compact spacing')
        token = r'\newcommand{\PaperSummary}'
        match = re.search(r'(?m)^' + re.escape(token), text)
        require(match is not None, 'active abstract missing from redline')
        left = text.index('{', match.start() + len(token))
        depth, right = 1, left + 1
        while depth:
            require(right < len(text), 'unclosed abstract')
            if text[right] in '{}':
                depth += 1 if text[right] == '{' else -1
            right += 1
        text = text[:left] + '{' + r'\DIFadd{' + SUMMARY + '}}' + text[right:]
        marked = True
    (stage / 'paper' / (stem + '.tex')).write_text(text)
    return dict(old_inputs_sha256=old, new_inputs_sha256=new, complete_replacement_abstract_marked=marked)


def build(stage):
    with build_slot():
        verify(stage)
        review = load(stage / 'figure_review.json')
        require(review['render_receipt_sha256'] == sha(stage / 'render.json') and review['both_new_figures_actually_viewed'], 'review actual figures before build')
        logs = stage / 'build_logs'
        logs.mkdir(exist_ok=False)
        redlines = {stem: redline(stage, source, stem) for source, stem in
                    [('manuscript_review.tex', 'redlined_manuscript'), ('supplementary_information.tex', 'redlined_supplementary_information')]}
        outputs, pages, locations = {}, {}, {}
        terms = ('Complete image-answer fitting', 'Supplementary Note 57', 'Fig. S34', 'Fig. S35', '32/32', '18.191', '668')
        for stem in DOCS:
            auxiliary = stage / 'aux' / stem
            auxiliary.mkdir(parents=True, exist_ok=False)
            owned_command(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', '-outdir=' + str(auxiliary), stem + '.tex'],
                          stage / 'paper', logs / (stem + '.build.log'), 180)
            log = (auxiliary / (stem + '.log')).read_text()
            require(not any(t in log for t in ('Overfull', 'undefined references', 'undefined citations', 'LaTeX Error')), 'layout/LaTeX error: ' + stem)
            path = stage / 'paper' / (stem + '.pdf')
            shutil.copy2(auxiliary / (stem + '.pdf'), path)
            document = fitz.open(path)
            text = ' '.join(' '.join(p.get_text().split()) for p in document)
            require('32/32' in text and '18.191' in text, 'complete-learning result absent: ' + stem)
            if 'supplementary' in stem:
                require(all(term in text for term in ('Supplementary Note 57', 'Fig. S34', 'Fig. S35', '668')), 'new supplement content missing')
            if stem == 'redlined_manuscript':
                spans = [s for b in document[0].get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
                require(any(s['color'] == 255 and 'Written language' in s['text'] for s in spans), 'replacement abstract lacks compiled blue markup')
            owned_command(['qpdf', '--check', str(path)], stage / 'paper', logs / (stem + '.qpdf.log'), 30)
            outputs[path.name], pages[stem] = sha(path), len(document)
            locations[stem] = {term: [i + 1 for i, p in enumerate(document) if term in ' '.join(p.get_text().split())] for term in terms}
            print('Built ' + stem + ': ' + str(len(document)) + ' pages', flush=True)
        verify(stage)
        save(stage / 'build.json', dict(preparation_sha256=sha(stage / 'preparation.json'), PDFs_sha256=outputs,
            render_receipt_sha256=sha(stage / 'render.json'), pages=pages, locations=locations, redlines=redlines,
            no_overfull_or_undefined=True, active_publication_updated=False, actual_pdf_review_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'render', 'build'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.stage](args.output.resolve())
