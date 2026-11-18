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


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('baseline','figure','captions','build','inspect'))
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
        else:
            base.inspect()
