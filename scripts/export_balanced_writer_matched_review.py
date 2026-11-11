#!/usr/bin/env python3
"""Export every completed balanced-writer output without private runtime paths."""
import json
import os
from pathlib import Path
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.compare_balanced_writer_matched import RUN
from scripts.review_balanced_writer_matched import authenticate_review
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/export_balanced_writer_matched_review.py'
DESTINATION = ROOT / 'references/evidence/balanced_writer_matched_20261111'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only export')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + OWN], cwd=ROOT)
            == (ROOT / OWN).read_bytes(), 'Commit the export implementation before use')
    review = RUN / 'review'
    mapping, manual = authenticate_review(RUN, review / 'readings.json')
    summary_path = ROOT / 'references/evidence/balanced_writer_matched_learning_20261111.json'
    summary = json.loads(summary_path.read_text())
    require(summary['independent_per_answer_and_aggregate_recount_passed']
            and summary['all216_initial_context_states_bit_exact_to_parent'],
            'Complete verified whole-language analysis required')
    require(not DESTINATION.exists(), 'Preserve earlier public export')
    manifest = json.loads((review / 'manifest.json').read_text())
    DESTINATION.mkdir()
    files = {}
    for board in manifest['opaque_boards']:
        require(sha(review / board['file']) == board['sha256'], 'Reviewed board changed')
        shutil.copy2(review / board['file'], DESTINATION / board['file'])
        files[board['file']] = sha(DESTINATION / board['file'])
    for prefix in manifest['prefix_bindings']:
        require(sha(review / 'pages' / prefix['file']) == prefix['sha256'], 'Viewed prefix changed')
        shutil.copy2(review / 'pages' / prefix['file'], DESTINATION / prefix['file'])
        files[prefix['file']] = sha(DESTINATION / prefix['file'])
    records = []
    for layout in ['fixed_cells', 'proportional']:
        scored = json.loads((review / layout / 'scores.json').read_text())
        for row in scored['records']:
            opaque = next(key for key, value in mapping.items()
                          if value['layout'] == layout and value['case'] == row['case'])
            records.append(dict(opaque=opaque, layout=layout, case=row['case'],
                kind=row['kind'], condition=row['condition'],
                stopped=row['stopped'], emitted_glyphs=row['emitted_glyphs'],
                capped=row['emitted_glyphs'] == 128 and not row['stopped'],
                image_sha256=row['image_sha256'], reading=manual['readings'][opaque],
                readable=row['readable'], complete_exact=row['complete_exact'],
                coherent_complete=row['coherent_complete']))
    require(len(records) == 216 and len({r['opaque'] for r in records}) == 216,
            'All216 actual outputs required')
    public = dict(schema='balanced-writer-matched-output-review-v1',
        exporter_sha256=sha(ROOT / OWN), learning_summary_sha256=sha(summary_path),
        original_readings_sha256=sha(review / 'readings.json'),
        original_reading_audit_sha256=sha(review / 'reading_audit.json'),
        original_manifest_sha256=sha(review / 'manifest.json'), records=records,
        boards=manifest['opaque_boards'],
        book_prefixes=[dict(file=p['file'], sha256=p['sha256']) for p in manifest['prefix_bindings']],
        all216_actual_outputs_viewed=True, all54boards_viewed=True, all12prefixes_viewed=True,
        readings_frozen_before_reference_scoring=True,
        model_reviewer_prior_cohort_familiarity=True, independent_human_adjudication=False,
        uncertain_partials_unrepaired=True, original_output_pixels_unrepaired=True,
        enlargement='Nearest-neighbor display repetition only, before reading freeze; original board pixels unchanged.',
        generalized_language_learning_proven=False, goal_achieved=False)
    save_json(DESTINATION / 'review.json', public)
    files['review.json'] = sha(DESTINATION / 'review.json')
    receipt = dict(source_sha256=sha(ROOT / OWN), files_sha256=files,
        original_output_images=216, complete_opaque_boards=54, original_book_prefixes=12,
        private_runtime_paths_exported=False, source_learning_summary_sha256=sha(summary_path))
    save_json(ROOT / 'references/evidence/balanced_writer_matched_review_export_20261111.json', receipt)
    save_json(RUN / 'public_review_export.json', receipt)
    print(json.dumps(dict(original_outputs=216, boards=54, prefixes=12, files=len(files))))


if __name__ == '__main__':
    main()
