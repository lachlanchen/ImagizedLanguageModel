#!/usr/bin/env python3
"""Archive both complete conditions, all raw answers and every qualification pixel."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/archive_glyph_tail_pair.py'


def load(path):
    return json.loads(Path(path).read_text())


def reviewed(run):
    require(load(run/'assessment/status.json')['phase']=='completed', 'finish all answers before archiving')
    raw = load(run/'raw_review/report.json'); manual = load(run/'raw_review/manual_review.json')
    scores = load(run/'raw_review/scores.json'); q = load(run/'qualify/report.json')
    qr = load(run/'qualify/visual_review.json')
    require(len(raw['records']) == len(scores['records']) == 72 and len(raw['boards']) == 18,
            'retain the complete fixed cohort')
    require(manual['all72_actual_streams_viewed'] and manual['all18_boards_actually_viewed']
            and manual['raw_review_report_sha256'] == sha(run/'raw_review/report.json')
            and scores['manual_review_sha256'] == sha(run/'raw_review/manual_review.json'), 'complete actual review required')
    require(qr['all8_qualification_boards_actually_viewed'] and len(q['boards']) == 8
            and qr['report_sha256'] == sha(run/'qualify/report.json'), 'complete actual qualification review required')
    require(raw['assessment_report_sha256'] == sha(run/'assessment/report.json'), 'assessment ancestry differs')
    require(load(run/'fit/report.json')['checkpoint_sha256'] == q['checkpoint_sha256']
            == load(run/'assessment/report.json')['checkpoint_sha256'], 'mixed checkpoint conditions')
    return raw, manual, scores, q, qr


def archive(run, output, records):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    raw, manual, scores, q, qr = records
    m = load(run/'preparation.json')
    output.mkdir(); (output/'raw_outputs').mkdir(); (output/'qualification').mkdir()
    files = {}

    def copy(path, destination, expected):
        require(sha(path) == expected, 'reviewed source changed')
        shutil.copy2(path, output/destination)
        require(sha(output/destination) == expected, 'archive copy differs')
        files[str(destination)] = expected

    for row in raw['records']:
        path = run/'assessment'/row['file']; require(sha(path) == row['sha256'], 'raw answer tensor changed')
        value = torch.load(path, weights_only=True, map_location='cpu'); pixels = value['images']
        require(tuple(pixels.shape) == (row['glyphs'], 1, 32, 32)
                and bool(((pixels == -1) | (pixels == 1)).all()), 'invalid complete-glyph stream')
        require(len(value['feedback']) == len(pixels) and len(value['states']) == len(value['stop_probabilities']) == len(pixels)+1,
                'decision count differs')
        require(all(p < .5 for p in value['stop_probabilities'][:-1])
                and (value['stop_probabilities'][-1] >= .5) == row['stopped']
                and row['stopped'] != row['capped'], 'stop/cap outcome differs')
        require(not row['capped'] or len(pixels) == 64, 'incorrect registered cap')
        actual = np.full((32, max(1, len(pixels))*32), 255, np.uint8)
        for i, tile in enumerate(pixels):
            require(tensor_hash(tile[None]) == value['feedback'][i]['image_sha256'], 'actual feedback image differs')
            actual[:, i*32:(i+1)*32] = ((tile[0].numpy()+1)*127.5).astype(np.uint8)
        png = run/'assessment'/row['png']
        require(np.array_equal(actual, np.asarray(Image.open(png).convert('L'))), 'PNG differs from actual emitted pixels')
        copy(png, Path('raw_outputs')/row['png'], row['png_sha256'])
    for row in raw['boards']: copy(run/'raw_review'/row['file'], Path(row['file']), row['sha256'])
    for name in q['boards']:
        copy(run/'qualify'/name, Path('qualification')/name, qr['boards_sha256'][name])
    primitive = ROOT/m['primitive']/'training.pt'
    require(sha(primitive) == m['bindings_sha256'][str(primitive.relative_to(ROOT))], 'canonical physical evaluator changed')
    targets = torch.load(primitive, weights_only=True, map_location='cpu')['ink']
    physical = load(run/'qualify/records.json'); packed = []
    for key in ['training', 'copy_held']:
        name = key+'_outputs.pt'; path = run/'qualify'/name
        require(sha(path) == q['raw_outputs_sha256'][name], 'qualification raw tensor changed')
        v = torch.load(path, weights_only=True, map_location='cpu'); bits, labels = v['ink'], v['labels']
        require(bits.ndim == 4 and tuple(bits.shape[1:]) == (1,32,32)
                and bool(((bits == 0) | (bits == 1)).all()), 'invalid qualification glyphs')
        rows = physical[key]
        require(len(bits) == len(labels) == len(rows) and labels.tolist() == [r['label'] for r in rows],
                'qualification order or labels differ')
        actual = bits[:,0].numpy().astype(np.uint8); gold = targets[labels,0].numpy().astype(np.uint8)
        denominator = actual.sum((1,2))+gold.sum((1,2)); intersection = (actual & gold).sum((1,2))
        f1 = np.where(denominator == 0, 1., 2*intersection/np.maximum(denominator,1))
        require(np.allclose(f1, [r['f1'] for r in rows], rtol=0, atol=1e-12), 'physical scores differ from saved pixels')
        require(np.array_equal((actual == gold).all((1,2)), [r['exact'] for r in rows]), 'physical exact count differs')
        canvas = np.full((((len(bits)+63)//64)*32,2048),255,np.uint8)
        for i, glyph in enumerate(actual):
            canvas[(i//64)*32:(i//64+1)*32,(i%64)*32:(i%64+1)*32] = 255*(1-glyph)
        destination = Path('qualification')/(key+'_all.png')
        Image.fromarray(canvas).convert('1', dither=Image.Dither.NONE).save(output/destination)
        require(np.array_equal(np.asarray(Image.open(output/destination).convert('L')),canvas), 'packing changed source pixels')
        files[str(destination)] = sha(output/destination)
        packed.append(dict(file=str(destination),glyphs=len(bits),columns=64,cell_size=[32,32],
            source_tensor_sha256=q['raw_outputs_sha256'][name],all_saved_F1_and_exact_flags_recomputed=True))
    for source, destination in [
        (run/'raw_review/report.json','review_report.json'), (run/'raw_review/manual_review.json','manual_review.json'),
        (run/'raw_review/scores.json','scores.json'), (run/'fit/report.json','fit.json'),
        (run/'qualify/visual_review.json','qualification/visual_review.json'),
        (run/'qualify/records.json','qualification/records.json')]:
        copy(source, Path(destination), sha(source))
    for source, destination in [(q,'qualification/report.json'),(load(run/'assessment/report.json'),'assessment.json')]:
        public = dict(source); public.pop('admission',None); save_json(output/destination,public)
        files[destination] = sha(output/destination)
    return dict(run=str(run.relative_to(ROOT)), checkpoint_sha256=q['checkpoint_sha256'], files_sha256=files,
        streams=72,raw_review_boards=18,qualification_boards=8,packed_qualification=packed,
        full_pixel_and_feedback_image_hash_verification=True,feedback_embeddings_cold_recomputed=False,
        individually_human_graded_all_qualification_images=False,output_pixels_repaired=False,
        neural_forward_calls=0,CUDA_initialized=torch.cuda.is_initialized(),
        full_fit_report_sha256=sha(run/'fit/report.json'),full_assessment_report_sha256=sha(run/'assessment/report.json'))


def main(primary, control, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only archive')
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT) == (ROOT/OWN).read_bytes(), 'commit archiver first')
    require(not output.exists(), 'preserve previous pair archives')
    a, b = reviewed(primary), reviewed(control)
    require({r['key']:r['input_sha256'] for r in a[0]['records']}
            == {r['key']:r['input_sha256'] for r in b[0]['records']}, 'paired conditions used different inputs')
    for run, other in [(primary,control),(control,primary)]:
        require((ROOT/load(run/'comparison_registration.json')['companion_run']).resolve() == other,
                'archive both registered conditions together')
    output.mkdir(parents=True)
    conditions = {name:archive(run,output/name,data) for name,run,data in
                  [('contextual',primary,a),('readout_only',control,b)]}
    require(not any(r['CUDA_initialized'] for r in conditions.values()), 'CPU archive initialized CUDA')
    license_path = ROOT/'data/visual_qa/cmrc2018_short/LICENCE'
    shutil.copy2(license_path,output/'CMRC_SOURCE_LICENCE')
    save_json(output/'archive.json',dict(source_sha256=sha(ROOT/OWN),conditions=conditions,
        total_streams=144,both_conditions_retained=True,all_52_actual_review_boards_retained=True,
        source_dataset='CMRC 2018; see existing pinned preparation provenance',
        source_dataset_preparation_sha256=sha(ROOT/'data/visual_qa/cmrc2018_short/preparation_report.json'),
        source_dataset_license_sha256=sha(license_path),general_ILM_goal_achieved=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--control',type=Path,default=Path('artifacts/glyph_tail_readout_control_v2'))
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args(); main(a.primary.resolve(),a.control.resolve(),a.output.resolve())
