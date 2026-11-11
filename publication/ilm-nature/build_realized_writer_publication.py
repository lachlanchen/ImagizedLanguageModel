#!/usr/bin/env python3
"""Build the bounded article update without replacing earlier evidence receipts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = Path(os.environ.get('ILM_REALIZED_PAPER_STAGE',
                            str(ROOT / 'artifacts/realized_writer_article_v1')))


def run(arguments, log, env):
    result = subprocess.run(arguments, cwd=HERE, env=env, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Build failed; retained log: ' + str(log))


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    check_only = os.environ.get('ILM_REALIZED_PAPER_CHECK_ONLY') == '1'
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        ILM_REALIZED_PAPER_STAGE=str(STAGE), ILM_MOTOR_PAPER_STAGE=str(STAGE),
        ILM_MOTOR_PAPER_CHECK_ONLY='1')
    if not (STAGE / 'preserved_assets.json').exists():
        shutil.copy2(HERE / 'evidence/realized_writer_preserved_assets.json',
                     STAGE / 'preserved_assets.json')
    for folder, name in [('reading', 'manuscript'), ('review', 'manuscript_review')]:
        directory = STAGE / folder; directory.mkdir(exist_ok=True)
        run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
             '-outdir=' + str(directory), name + '.tex'], directory / 'build.log', env)
        log = (directory / (name + '.log')).read_text()
        if any(x in log for x in ['Overfull', 'undefined references', 'undefined citations', 'LaTeX Error']):
            raise ValueError('Layout/reference error: ' + name)
    # Its check-only route writes the newly expanded redline and receipt solely
    # in this stage, retaining the immutable baseline and previous public audit.
    run([sys.executable, str(HERE / 'build_motor_diagnostics_redline.py')],
        STAGE / 'redline_console.log', env)
    run([sys.executable, str(HERE / 'verify_motor_diagnostics_figure.py')],
        STAGE / 'original_glyph_console.log', env)
    (STAGE / 'figure_audit.json').replace(STAGE / 'original_glyph_audit.json')
    run([sys.executable, str(HERE / 'verify_realized_writer_figure.py')],
        STAGE / 'figure_console.log', env)
    audit = json.loads((STAGE / 'figure_audit.json').read_text())
    if not check_only:
        for folder, name in [('reading', 'manuscript'), ('review', 'manuscript_review')]:
            shutil.copy2(STAGE / folder / (name + '.pdf'), HERE / (name + '.pdf'))
        redline = STAGE / 'redlines/redlined_manuscript'
        for suffix in ['tex', 'pdf']:
            shutil.copy2(redline / ('redlined_manuscript.' + suffix),
                         HERE / ('redlined_manuscript.' + suffix))
        shutil.copy2(STAGE / 'redline_build.json', HERE / 'evidence/realized_writer_redline_build.json')
        shutil.copy2(STAGE / 'figure_audit.json', HERE / 'evidence/realized_writer_figure_audit.json')
        shutil.copy2(STAGE / 'original_glyph_audit.json', HERE / 'evidence/realized_writer_original_glyph_audit.json')
    receipt = dict(all_builds_complete=True, matching_article_PDFs_verified=True,
        active_files_updated=not check_only,
        pages={name: value['pages'] for name, value in audit['PDFS'].items()},
        prior_assets_preserved=audit['prior_assets_preserved'],
        manual_visual_QA_still_required=True, new_optimizer_updates=0)
    (STAGE / 'build_state.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
