#!/usr/bin/env python3
"""Archive every reviewed complete image stream and its unchanged score."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require


def load(path):
    return json.loads(Path(path).read_text())


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU archive requires hidden CUDA')
    own = 'scripts/archive_semantic_streams.py'
    require(subprocess.check_output(['git', 'show', 'HEAD:' + own], cwd=ROOT)
            == (ROOT / own).read_bytes(), 'commit archiver first')
    require(not output.exists(), 'preserve prior archives')
    require(load(run / 'assessment/status.json')['phase'] == 'completed', 'finish every stream first')
    review = load(run / 'raw_review/manual_review.json')
    require(review['all40_actual_raw_outputs_viewed'] and review['all10_boards_viewed']
            and review['raw_review_report_sha256'] == sha(run / 'raw_review/report.json'),
            'actual complete-stream review required')
    report = load(run / 'raw_review/report.json')
    require(len(report['records']) == 40 and len(report['boards']) == 10, 'incomplete review')
    # Re-run the exact recording checks in a fresh CPU review directory before
    # publishing; the original human review remains tied to its original boards.
    import numpy as np
    from PIL import Image
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    raw = output / 'raw_outputs'
    raw.mkdir()
    files = {}
    for row in report['records']:
        tensor = run / 'assessment' / row['file']
        png = run / 'assessment' / row['png']
        require(sha(tensor) == row['sha256'] and sha(png) == row['png_sha256'], 'original changed')
        v = torch.load(tensor, weights_only=True, map_location='cpu')
        images = v['images']
        require(tuple(images.shape) == (row['glyphs'], 1, 32, 32)
                and bool(((images == -1) | (images == 1)).all()), 'invalid actual pixels')
        require(len(v['feedback']) == len(images) and len(v['stop_probabilities']) == len(images) + 1,
                'decision history differs')
        require(all(p < .5 for p in v['stop_probabilities'][:-1])
                and (v['stop_probabilities'][-1] >= .5) == row['stopped']
                and row['stopped'] != row['capped'], 'autonomous stop differs')
        canvas = np.full((32, max(1, len(images)) * 32), 255, np.uint8)
        for i, tile in enumerate(images):
            require(tensor_hash(tile[None]) == v['feedback'][i]['image_sha256'], 'feedback pixels differ')
            canvas[:, i*32:(i+1)*32] = ((tile[0].numpy() + 1) * 127.5).astype(np.uint8)
        require(np.array_equal(canvas, np.asarray(Image.open(png).convert('L'))), 'PNG differs from tensor')
        shutil.copy2(png, raw / row['png'])
        files['raw_outputs/' + row['png']] = sha(png)
    for board in report['boards']:
        source = run / 'raw_review' / board
        require(sha(source) == review['board_sha256'][board], 'actually viewed board differs')
        shutil.copy2(source, output / board)
        files[board] = sha(source)
    for name in ['report.json', 'manual_review.json', 'canonical_scores.json']:
        source = run / 'raw_review' / name
        shutil.copy2(source, output / ('review_' + name))
        files['review_' + name] = sha(source)
    assessment = load(run / 'assessment/report.json')
    assessment.pop('admission', None)
    save_json(output / 'assessment.json', assessment)
    save_json(output / 'archive.json', dict(source_sha256=sha(ROOT / own),
        checkpoint_sha256=load(run / 'fit/report.json')['checkpoint_sha256'],
        original_assessment_sha256=sha(run / 'assessment/report.json'),
        files_sha256=files, streams=40, actually_viewed_boards=10,
        every_PNG_pixel_verified=True, all_feedback_image_hashes_verified=True,
        feedback_embedding_hashes_cold_recomputed=False, image_outputs_repaired=False,
        neural_forward_calls=0, CUDA_calls=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.output.resolve())
