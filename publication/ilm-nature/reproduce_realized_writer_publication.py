#!/usr/bin/env python3
"""Rebuild outside figure-authoring caches and compare every article page raster."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / 'artifacts/realized_writer_article_reproduction_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(existing_build=False):
    if not existing_build:
        STAGE.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', ILM_REALIZED_PAPER_STAGE=str(STAGE),
                   ILM_REALIZED_PAPER_CHECK_ONLY='1')
        result = subprocess.run([sys.executable, str(HERE / 'build_realized_writer_publication.py')],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        (STAGE / 'build_console.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('Independent build failed; retained log')
    state = json.loads((STAGE / 'build_state.json').read_text())
    if not state['all_builds_complete'] or state['active_files_updated']:
        raise ValueError('Require the separate check-only build')
    forbidden = ['realized_writer_publication_v1', 'motor_diagnostics_publication_v1']
    inputs = []
    for path in STAGE.rglob('*.fls'):
        for line in path.read_text().splitlines():
            if line.startswith('INPUT '):
                if any(name in line for name in forbidden):
                    raise ValueError('Private figure-authoring cache used by TeX: ' + line)
                inputs.append(line[6:])
    if not inputs:
        raise ValueError('Require actual TeX input traces')
    records = []
    for name, relative in [('manuscript', 'reading/manuscript.pdf'),
        ('manuscript_review', 'review/manuscript_review.pdf'),
        ('redlined_manuscript', 'redlines/redlined_manuscript/redlined_manuscript.pdf')]:
        original, repeated = HERE / (name + '.pdf'), STAGE / relative
        a, b = fitz.open(original), fitz.open(repeated)
        if len(a) != len(b):
            raise ValueError('Reproduced page count differs: ' + name)
        pages = []
        for i in range(len(a)):
            first = a[i].get_pixmap(alpha=False); second = b[i].get_pixmap(alpha=False)
            if ((first.width, first.height, first.n) != (second.width, second.height, second.n)
                    or first.samples != second.samples):
                raise ValueError(f'Reproduced page differs: {name} {i + 1}')
            pages.append(dict(page=i + 1, raster_sha256=hashlib.sha256(first.samples).hexdigest()))
        records.append(dict(document=name, current_pdf_sha256=sha(original),
            reproduced_pdf_sha256=sha(repeated), pages=pages, all_page_rasters_bit_exact=True))
    result = dict(source_sha256=sha(Path(__file__)), builder_sha256=sha(HERE / 'build_realized_writer_publication.py'),
        documents=records, all_article_pages=sum(len(x['pages']) for x in records),
        every_page_raster_bit_exact=True, TeX_input_trace_entries_checked=len(inputs),
        private_figure_authoring_cache_in_TeX_inputs=False,
        system_TeX_and_pinned_latexdiff_installations_reused=True,
        separate_build_without_active_file_replacement=True,
        claim_boundary='Article rendering reproducibility only; no new model generation or language measurement.')
    (HERE / 'evidence/realized_writer_reproduction.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(all_page_rasters_bit_exact=True, article_pages=result['all_article_pages'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-build', action='store_true')
    main(parser.parse_args().existing_build)
