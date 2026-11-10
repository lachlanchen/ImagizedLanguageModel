#!/usr/bin/env python3
"""Publish reviewed model outputs, without redistributing C3 input pages/text."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import require


def load(path):
    return json.loads(Path(path).read_text())


def main(run, output):
    own = 'scripts/archive_c3_glyph_assessment.py'
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU archive requires hidden CUDA')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + own], cwd=ROOT)
            == (ROOT / own).read_bytes(), 'commit archiver first')
    require(not output.exists(), 'preserve prior archives')
    require(load(run / 'generation/status.json')['phase'] == 'completed', 'finish every generation first')
    raw = load(run / 'raw_review/report.json')
    manual = load(run / 'raw_review/manual_review.json')
    scores = load(run / 'raw_review/scores.json')
    require(raw['generation_report_sha256'] == sha(run / 'generation/report.json')
            and manual['raw_review_report_sha256'] == sha(run / 'raw_review/report.json')
            and scores['manual_review_sha256'] == sha(run / 'raw_review/manual_review.json'),
            'review/score ancestry differs')
    require(manual['all16_boards_actually_viewed'] and manual['all128_actual_streams_viewed']
            and manual['all5_zoom_boards_actually_viewed'] and len(raw['records']) == 128
            and raw['all_pixel_and_feedback_image_hashes_verified'], 'complete actual review required')
    output.mkdir(parents=True)
    (output / 'raw_outputs').mkdir()
    files = {}

    def copy(source, destination, expected):
        require(sha(source) == expected, 'reviewed file changed')
        shutil.copy2(source, output / destination)
        require(sha(output / destination) == expected, 'archive copy differs')
        files[str(destination)] = expected

    for row in raw['records']:
        require(sha(run / 'generation' / row['tensor']) == row['tensor_sha256'], 'raw tensor changed')
        copy(run / 'generation' / row['png'], Path('raw_outputs') / row['png'], row['png_sha256'])
    for row in raw['boards']:
        copy(run / 'raw_review' / row['file'], Path(row['file']), row['sha256'])
    for name, digest in manual['additional_nearest_neighbor_zoom_sha256'].items():
        copy(run / 'raw_review' / name, Path(name), digest)
    for name in ['report.json', 'manual_review.json']:
        copy(run / 'raw_review' / name, Path('review_' + name), sha(run / 'raw_review' / name))
    # Original reference strings and source pages remain in ignored local data.
    public = dict(scores)
    public['records'] = []
    for row in scores['records']:
        row = dict(row)
        expected = row.pop('expected')
        row['expected_utf8_sha256'] = hashlib.sha256(expected.encode()).hexdigest()
        public['records'].append(row)
    public['reference_strings_redistributed'] = False
    save_json(output / 'scores.json', public)
    generation = load(run / 'generation/report.json')
    resources = {k: generation[k] for k in ['elapsed_seconds', 'resource_wait_seconds',
        'peak_allocated_vram_gib', 'minimum_available_gib']}
    save_json(output / 'resources.json', resources)
    save_json(output / 'archive.json', dict(source_sha256=sha(ROOT / own),
        preparation_sha256=sha(run / 'preparation.json'),
        generation_report_sha256=sha(run / 'generation/report.json'),
        original_local_scores_sha256=sha(run / 'raw_review/scores.json'),
        checkpoint_versions=load(run / 'preparation.json')['versions'], files_sha256=files,
        streams=128, actual_review_boards=16, additional_zoom_boards=5,
        every_output_PNG_preserved=True, runtime_output_pixels_repaired=False,
        input_pages_redistributed=False, source_reference_strings_redistributed=False,
        model_calls=0, CUDA_calls=0))
    notice = ROOT / 'data/evaluation/c3_m_development_v1/license.txt'
    shutil.copy2(notice, output / 'C3_SOURCE_LICENSE.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('artifacts/c3_glyph_language_assessment_v1'))
    parser.add_argument('--output', type=Path, default=Path('references/evidence/c3_glyph_language_assessment_v1'))
    args = parser.parse_args()
    main(args.run.resolve(), args.output.resolve())
