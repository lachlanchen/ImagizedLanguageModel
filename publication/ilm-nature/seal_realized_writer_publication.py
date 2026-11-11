#!/usr/bin/env python3
"""Bind inspected current PDFs, source files, native charts and reproduction."""
import hashlib
import json
from pathlib import Path

import fitz

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = ROOT / 'artifacts/realized_writer_article_v1'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manual = load(STAGE / 'visual_qa_manifest.json')
    if not manual['records'] or not all(row['actually_viewed'] for row in manual['records']):
        raise ValueError('Inspect every selected changed/continued page first')
    names = dict(reading='manuscript', review='manuscript_review', redline='redlined_manuscript')
    docs = {key: fitz.open(HERE / (name + '.pdf')) for key, name in names.items()}
    rasters = {}
    for row in manual['records']:
        path = ROOT / row['file']
        if sha(path) != row['sha256']:
            raise ValueError('Inspected raster changed')
        current = docs[row['variant']][row['page'] - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        viewed = fitz.Pixmap(path)
        if (current.width, current.height, current.samples) != (viewed.width, viewed.height, viewed.samples):
            raise ValueError('Inspected raster does not match current PDF')
        rasters[row['file']] = row['sha256']
    figure = load(HERE / 'evidence/realized_writer_figure_audit.json')
    pixels = load(HERE / 'evidence/realized_writer_original_glyph_audit.json')
    reproduction = load(HERE / 'evidence/realized_writer_reproduction.json')
    if not (figure['matched_article_PDFs_verified'] and figure['previous_assets_unchanged']
            and reproduction['every_page_raster_bit_exact']):
        raise ValueError('Complete figure/preservation/reproduction checks required')
    for row in reproduction['documents']:
        if sha(HERE / (row['document'] + '.pdf')) != row['current_pdf_sha256']:
            raise ValueError('PDF changed after independent reproduction')
    if not all(row['original_tile_occurrences_verified'] == 20 for row in pixels['PDFs'].values()):
        raise ValueError('Preserve all previous original glyph occurrences')
    redline = load(HERE / 'evidence/realized_writer_redline_build.json')
    tex = redline['documents']['redlined_manuscript']['new_inputs_sha256']
    for name, digest in tex.items():
        if sha(HERE / name) != digest:
            raise ValueError('TeX source changed after compilation: ' + name)
    files = set(tex) | {'manuscript.pdf', 'manuscript_review.pdf', 'redlined_manuscript.pdf',
        'redlined_manuscript.tex', 'nature_reading.sty', 'README.md',
        'figures/Realized_writer_figures.pptx', 'figures/Figure_22_realized_writer.pdf',
        'figures/Figure_22_realized_writer.png', 'build_realized_writer_publication.py',
        'reproduce_realized_writer_publication.py', 'seal_realized_writer_publication.py',
        'prepare_realized_writer_evidence.py', 'realized_writer_figures.mjs',
        'verify_realized_writer_figure.py', 'format_motor_chart_labels.py'}
    files |= {'evidence/' + path.name for path in (HERE / 'evidence').glob('realized_writer_*.json')
              if path.name not in ['realized_writer_publication_build.json', 'realized_writer_local_sync.json']}
    receipt = dict(previous_publication_commit='fdc619d5',
        redline_baseline_commit=redline['baseline_commit'], files_sha256={name: sha(HERE / name) for name in sorted(files)},
        pdf_pages={key: len(doc) for key, doc in docs.items()}, native_charts=4,
        all_native_caches_and_literal_workbooks_match=True, prior_assets_unchanged=224,
        all20previous_original_tile_occurrences_in_all_article_PDFs_verified=True,
        figure_audit_sha256=sha(HERE / 'evidence/realized_writer_figure_audit.json'),
        original_glyph_audit_sha256=sha(HERE / 'evidence/realized_writer_original_glyph_audit.json'),
        figure_bindings_sha256=sha(HERE / 'evidence/realized_writer_figure_bindings.json'),
        measured_evidence_sha256=sha(HERE / 'evidence/realized_writer_measured_data.json'),
        visual_QA=dict(visual_QA_complete=True, reviewer='Model-based visual inspection in the authoring session',
            pages=manual['selected_pages'], inspected_rasters_sha256=rasters,
            inspected_rasters_reproduced_from_current_PDFs=True, independent_human_review=False),
        reproduction_sha256=sha(HERE / 'evidence/realized_writer_reproduction.json'),
        all160_article_page_rasters_reproduced=reproduction['all_article_pages'] == 160,
        no_private_figure_authoring_cache_required=True,
        used_primary_references_verified=['https://arxiv.org/abs/2309.17400', 'https://arxiv.org/abs/2210.02747'],
        preexisting_bibliography_not_comprehensively_reverified=True,
        full_rollout_first_only_gate_pass_included=True, both_rollout_language_scores_unmeasured=True,
        completed216_balanced_language_failure_included=True,
        F1_chart_values_display_six_decimals=True, model_comparison_not_learning_curve=True,
        new_optimizer_updates_in_this_revision=0, usable_ILM_qualified=False)
    (HERE / 'evidence/realized_writer_publication_build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(pages=receipt['pdf_pages'], inspected_pages=len(rasters),
                         independent_reproduction_pages=reproduction['all_article_pages'], sealed=True)))


if __name__ == '__main__':
    main()
