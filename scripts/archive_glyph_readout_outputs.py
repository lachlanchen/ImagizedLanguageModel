#!/usr/bin/env python3
"""Publish all completed fixed-state pixels without neural calls or repair."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/archive_glyph_readout_outputs.py'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only archive')
    require(not output.exists(), 'preserve prior archives')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + OWN], cwd=ROOT)
            == (ROOT / OWN).read_bytes(), 'commit archive code first')
    torch.set_num_threads(2)
    q = run / 'qualify'
    require(load(q / 'status.json')['phase'] == 'completed', 'qualification must be terminal')
    report = load(q / 'report.json')
    review = load(q / 'visual_review.json')
    require(review['report_sha256'] == sha(q / 'report.json')
            and review['all_three_boards_actually_viewed'], 'actually inspect the three fixed boards first')
    fit = load(run / 'fit/report.json')
    require(sha(run / 'fit/checkpoint_latest.pt') == fit['checkpoint_sha256'], 'checkpoint changed')
    output.mkdir(parents=True)
    for source, name in [(q / 'report.json', 'qualification.json'),
                         (q / 'records.json', 'records.json'),
                         (q / 'visual_review.json', 'visual_review.json'),
                         (run / 'fit/report.json', 'fit.json')]:
        shutil.copy2(source, output / name)
    for name in report['boards']:
        require(sha(q / name) == review['boards_sha256'][name], 'viewed board changed')
        shutil.copy2(q / name, output / name)
    files = []
    for name, digest in report['raw_outputs_sha256'].items():
        source = q / name
        require(sha(source) == digest, 'saved raw pixels changed')
        value = torch.load(source, weights_only=True, map_location='cpu')
        bits = value['ink']
        require(bits.ndim == 4 and tuple(bits.shape[1:]) == (1, 32, 32)
                and bool(((bits == 0) | (bits == 1)).all()), 'invalid whole-glyph images')
        n = len(bits)
        canvas = np.full((((n + 63) // 64) * 32, 2048), 255, np.uint8)
        for i, glyph in enumerate(bits):
            y, x = (i // 64) * 32, (i % 64) * 32
            canvas[y:y + 32, x:x + 32] = 255 * (1 - glyph[0].numpy().astype(np.uint8))
        filename = source.stem + '_all.png'
        Image.fromarray(canvas).convert('1', dither=Image.Dither.NONE).save(output / filename)
        restored = np.asarray(Image.open(output / filename).convert('L'))
        require(np.array_equal(restored, canvas), 'PNG packing changed actual pixels')
        files.append(dict(file=filename, sha256=sha(output / filename), source_tensor_sha256=digest,
                          glyphs=n, columns=64, cell_size=[32, 32], labels=value['labels'].tolist()))
    receipt = dict(source_sha256=sha(ROOT / OWN), preparation_sha256=sha(run / 'preparation.json'),
        qualification_sha256=sha(q / 'report.json'), fit_sha256=sha(run / 'fit/report.json'),
        visual_review_sha256=sha(q / 'visual_review.json'), packed_files=files,
        packed_glyph_images=sum(r['glyphs'] for r in files), padding_is_white=True,
        every_packed_pixel_verified=True, image_outputs_repaired=False,
        all_outputs_individually_human_graded=False, actual_sample_boards_viewed=3,
        neural_forward_calls=0, CUDA_calls=0)
    (output / 'archive.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(packed_glyph_images=receipt['packed_glyph_images'],
                          PNG_files=len(files), all_packed_pixels_verified=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.output.resolve())
