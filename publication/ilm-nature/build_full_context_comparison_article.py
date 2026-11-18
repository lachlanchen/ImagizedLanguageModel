"""Prepare Figure 32, compile the three manuscript editions and preserve measured inputs."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

import build_ranking_completed_article as base
from scripts.diagnose_visual_context_gradients import require, write_new

HERE, ROOT = base.HERE, base.ROOT
WORK = ROOT / 'artifacts/full_context_comparison_publication_v1'
FIG = 'Figure_32_full_context_comparison'
BASELINE = '9cd13feb'
PAPER = HERE / 'figures'
STAGE = WORK / 'article'
SKILL = Path.home() / '.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations'
NODE = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
base.FIGURES.extend([
    ('Figure_28_glyph_score_gaps', 'glyph_score_gap_caption.tex', 'fig:glyph-score-gaps'),
    ('Figure_29_reward_calibration', 'reward_calibration_caption.tex', 'fig:reward-calibration'),
    ('Figure_30_white_target_completed', 'white_target_completed_caption.tex', 'fig:white-target-completed'),
    ('Figure_31_visual_context_evidence', 'visual_context_completed_caption.tex', 'fig:visual-context-evidence'),
    (FIG, 'full_context_comparison_caption.tex', 'fig:full-context-comparison')])
base.STAGE = STAGE
_base_literal = base.literal_figures
base.literal_figures = lambda: _base_literal().replace('build_ranking_completed_article.py', Path(__file__).name)


def baseline():
    files = [p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.tex', '.pdf', '.sty')]
    changed = subprocess.check_output(['git','diff','--name-only',BASELINE,'--',*files,PAPER], cwd=ROOT)
    require(not changed, 'Current article differs from the designated immediate baseline')
    before = STAGE / 'before'
    before.mkdir(parents=True, exist_ok=False)
    for path in files:
        shutil.copy2(path, before / path.name)
    write_new(STAGE / 'before_manifest.json', dict(source_commit=BASELINE,
        files_sha256={p.name:base.sha(p) for p in files},
        figures_sha256={str(p.relative_to(HERE)):base.sha(p) for p in PAPER.rglob('*') if p.is_file()}))


def environment(stage):
    return dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', UV_THREADPOOL_SIZE='2', RUNTIME_NODE_MODULES=str(NODE/'node_modules'),
                RUNTIME_NODE=str(NODE/'bin/node'), RUNTIME_PYTHON=sys.executable, RUNTIME_BIN_DIR=str(NODE/'bin'),
                ILM_MOTOR_PAPER_STAGE=str(stage), ILM_MOTOR_PAPER_CHECK_ONLY='1')


def costs():
    return dict(child_seconds=0., minimum_available_host_gib=base.host()['available_gib'],
                peak_child_tree_rss_gib=0., children=[])


def figure(attempt):
    stage = WORK / f'figure_{attempt:02d}'
    stage.mkdir(exist_ok=False)
    base.STAGE = stage
    (stage / 'node_modules').symlink_to(NODE / 'node_modules', target_is_directory=True)
    module = stage / 'full_context_comparison_figure.mjs'
    shutil.copy2(HERE / module.name, module)
    base.run([str(NODE/'bin/node'),str(module),str(ROOT),str(stage),str(SKILL),sys.executable],
             stage / 'export.log', environment(stage), costs())


def build():
    base.preserved()
    env, usage = environment(STAGE), costs()
    for folder, name in base.DOCUMENTS[:2]:
        out = STAGE / folder
        out.mkdir(exist_ok=False)
        base.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
                  '-outdir='+str(out),name+'.tex'],out/'build.log',env,usage)
    base.run([sys.executable,str(HERE/'build_motor_diagnostics_redline.py')],STAGE/'redline.log',env,usage)
    docs = {}
    for folder, name in base.DOCUMENTS:
        logs = (STAGE / folder / (name+'.log')).read_text()
        errors = [s for s in logs.splitlines() if any(t in s for t in ('Overfull','Float too large',
                  'undefined references','undefined citations','multiply defined','LaTeX Error'))]
        require(not errors, str((name, errors)))
        docs[name] = base.sha(STAGE / folder / (name+'.pdf'))
    write_new(STAGE / 'build_receipt.json',dict(documents=docs, sources={p.name:base.sha(p) for p in HERE.glob('*.tex')},
              baseline=BASELINE, redline_baseline=base.load(STAGE/'redline_build.json')['baseline_commit']))


def promote():
    import hashlib
    import io
    import xml.etree.ElementTree as ET
    import zipfile
    from openpyxl import load_workbook

    build = base.load(STAGE / 'build_receipt.json')
    inspection = base.load(STAGE / 'inspection.json')
    reviewed = base.load(STAGE / 'visual_review.json')
    baseline = base.preserved()
    require(all(base.sha(HERE/name) == value for name,value in build['sources'].items()), 'TeX source changed')
    evidence = ROOT/'references/evidence/trainable_visual_context_result_20261117.json'
    data = base.load(evidence)
    deck = PAPER/'Full_context_comparison.pptx'
    samples = base.load(WORK/'figure_01/samples.json')
    namespaces = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
                  'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    with zipfile.ZipFile(deck) as archive:
        hashes = {hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist() if n.startswith('ppt/media/')}
        require(all(s['png_sha256'] in hashes for s in samples), 'Embedded original answer changed')
        sheet, = [n for n in archive.namelist() if n.endswith('.xlsx')]
        rows = list(load_workbook(io.BytesIO(archive.read(sheet)),read_only=True,data_only=True).active.values)[1:]
        for index, arm in enumerate(('trainable','frozen')):
            observed = [(int(r[index*2].replace(',','')),r[index*2+1]) for r in rows]
            expected = [(v['step'],v['exact']) for v in data['all_snapshot_summaries'] if v['arm'] == arm]
            require(observed == expected,'Native workbook values changed')
        chart, = [n for n in archive.namelist() if '/charts/' in n and n.endswith('.xml')]
        require(len(ET.fromstring(archive.read(chart)).findall('.//c:ser',namespaces)) == 2,'Expected two chart series')
        table = ET.fromstring(archive.read('ppt/slides/slide1.xml')).find('.//a:tbl',namespaces)
        values = [[c.find('.//a:t',namespaces).text for c in r.findall('a:tc',namespaces)] for r in table.findall('a:tr',namespaces)]
        require(values[1:] == [['Learned context','12','0','0'],['Frozen context','2','10','0']], 'Book counts differ')
    targets = {}
    for folder, name in base.DOCUMENTS:
        source = STAGE/folder/(name+'.pdf')
        result = inspection['documents'][name]
        require(base.sha(source) == result['pdf_sha256'] == reviewed[name]['pdf_sha256'] == build['documents'][name]
                and reviewed[name]['overview_pages'] == result['pages'] and reviewed[name]['detailed_pages'],
                'Incomplete or changed visual review')
        targets[HERE/(name+'.pdf')] = source
    targets[HERE/'redlined_manuscript.tex'] = STAGE/'redlines/redlined_manuscript/redlined_manuscript.tex'
    receipt = HERE/'evidence/full_context_comparison_article_build.json'
    require(not receipt.exists(), 'Preserve previous promotion')
    for target, source in targets.items():
        require(base.sha(target) in (baseline['files_sha256'][target.name],base.sha(source)), 'Active file changed')
    for target, source in targets.items():
        temporary = target.with_name(target.name+'.promoting')
        shutil.copy2(source,temporary)
        temporary.replace(target)
    write_new(receipt,dict(promoted=True,baseline_git=BASELINE,cumulative_redline_git=build['redline_baseline'],
              source_result_sha256=base.sha(evidence),inspection=inspection,visual_review=reviewed,
              prior_figure_files_verified=len(baseline['figures_sha256']),figure=dict(editable_sha256=base.sha(deck),
              chart_series=2,charts=1,workbooks=1,tables=1,samples=samples),office_application_export_verified=False,
              all_root_tex_sha256={p.name:base.sha(p) for p in HERE.glob('*.tex')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('baseline','figure','captions','build','inspect','promote'))
    p.add_argument('--attempt',type=int,default=1)
    args=p.parse_args()
    with (ROOT/'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        h = base.host()
        require(h['available_gib'] >= 48 and h['some'] <= .5 and h['full'] <= .1, 'Host admission')
        if args.action == 'baseline':
            baseline()
        elif args.action == 'figure':
            figure(args.attempt)
        elif args.action == 'captions':
            (HERE/'complete_continuation_figures.tex').write_text(base.literal_figures())
        elif args.action == 'build':
            build()
        elif args.action == 'inspect':
            base.inspect()
        else:
            promote()
