#!/usr/bin/env python3
"""Prepare all opaque motor outputs and reuse unchanged whole-answer scoring."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.infer_glyph_context_motor import study, verified_case
from scripts.train_glyph_context_motor import load, EVALUATION
from scripts.review_compact_visual_span import score
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/review_glyph_context_motor.py'
LAYOUTS = ['fixed_cells', 'proportional']
EVALUATIONS = dict(fixed_cells=EVALUATION,
    proportional=ROOT/'artifacts/glyph_tail_input_presentation_v1/evaluation')
CONTROLS = dict(fixed_cells=ROOT/'artifacts/glyph_tail_compact_cohort_v1',
    proportional=ROOT/'artifacts/glyph_tail_input_presentation_v1')


def identity():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only raw-output review')
    require(subprocess.check_output(['git', 'show', 'HEAD:'+OWN], cwd=ROOT) == (ROOT/OWN).read_bytes(),
            'Commit output-first review implementation before use')
    torch.set_num_threads(2)
    return sha(ROOT/OWN)


def prepare(run):
    code = identity(); study(run)
    manifest = load(run/'evaluation_manifest.json'); rows = manifest['inputs']
    reports = {layout:load(run/('generation_'+layout)/'report.json') for layout in LAYOUTS}
    require(len(rows) == 216 and all(r['all_inputs_completed'] for r in reports.values()),
            'Complete every216stream before review')
    review = run/'review'; require(not review.exists(), 'Preserve earlier review')
    mapping = {}
    for row in rows:
        generation = run/('generation_'+row['layout'])
        report = reports[row['layout']]
        require(report['request_sha256'] == sha(generation/'request.json'), 'Generation request changed')
        actual = verified_case(generation/row['id'], row)
        require(actual in report['records'], 'Generated case absent from complete report')
        opaque = hashlib.sha256(('context-motor-review-v1\0'+row['id']+'\0'
            +report['motor_checkpoint_sha256']).encode()).hexdigest()[:20]
        require(opaque not in mapping, 'Opaque ID collision')
        mapping[opaque] = dict(case=row['original_case'], layout=row['layout'], condition=row['condition'],
            step=12000, checkpoint_sha256=report['motor_checkpoint_sha256'],
            run_report_sha256=sha(generation/'report.json'), image_sha256=actual['image_sha256'],
            stopped=actual['stopped'], emitted_glyphs=actual['emitted_glyphs'], elapsed_seconds=actual['elapsed_seconds'],
            source=str((generation/row['id']/'answer.png').relative_to(ROOT)), input_image=row['image'])
    review.mkdir(); pages = review/'pages'; pages.mkdir()
    for opaque, row in mapping.items():
        shutil.copy2(ROOT/row['source'], pages/(opaque+'.png'))
        if row['case'].startswith('book_') and row['condition'] == 'normal':
            shutil.copy2(ROOT/row['input_image'], pages/(opaque+'_prefix.png'))
    save_json(review/'mapping.json', dict(training_registration_sha256=sha(run/'registration.json'), rows=mapping))
    readings = {k:dict(text=None, uncertain=None, coherent_continuation=None) for k in sorted(mapping)}
    save_json(review/'readings_template.json', dict(evaluator=None, readings=readings))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    boards = []; ordered = sorted(mapping)
    for first in range(0, len(ordered), 4):
        keys = ordered[first:first+4]
        pictures = [Image.open(pages/(k+'.png')).convert('L') for k in keys]
        width = max(700, max(p.width for p in pictures))+16
        height = sum(p.height+34 for p in pictures)
        canvas = Image.new('L', (width, height), 245); draw = ImageDraw.Draw(canvas); y = 0; tiles = []
        for key, picture in zip(keys, pictures):
            row = mapping[key]
            draw.text((8, y+4), f"{key}  n={row['emitted_glyphs']} stop={row['stopped']}", font=font, fill=0)
            canvas.paste(picture, (8, y+28))
            tiles.append(dict(opaque=key, x=8, y=y+28, width=picture.width, height=picture.height))
            y += picture.height+34
        name = f'board_{first//4+1:02}.png'; canvas.save(review/name)
        boards.append(dict(file=name, sha256=sha(review/name), tiles=tiles))
    require(len(boards) == 54 and len(list(pages.glob('*_prefix.png'))) == 12, 'Complete display coverage required')
    save_json(review/'manifest.json', dict(source_sha256=code, mapping_sha256=sha(review/'mapping.json'),
        generation_reports_sha256={layout:sha(run/('generation_'+layout)/'report.json') for layout in LAYOUTS},
        original_images=216, opaque_boards=boards, book_prefixes=12,
        references_opened=False, original_pixels_repaired=False,
        review_disclosure='One model review with prior cohort familiarity; opaque layout/case identities; no independent human adjudication.'))
    print(json.dumps(dict(actual_output_images=216, opaque_boards=54, book_prefixes=12, references_opened=False)))


def authenticate_review(run, readings):
    code = identity(); study(run)
    review = run/'review'; manifest = load(review/'manifest.json'); audit = load(review/'reading_audit.json')
    require(code == manifest['source_sha256'] and sha(review/'mapping.json') == manifest['mapping_sha256'],
            'Review source/mapping changed')
    require(audit['readings_sha256'] == sha(readings) and audit['mapping_sha256'] == sha(review/'mapping.json')
            and audit['review_manifest_sha256'] == sha(review/'manifest.json')
            and audit['all216_actual_output_images_viewed'] and audit['all54_boards_actually_viewed']
            and audit['all12_book_prefixes_viewed'] and audit['frozen_before_reference_scoring'],
            'Finish and freeze every actual output reading before scoring')
    mapping = load(review/'mapping.json')['rows']; manual = load(readings)
    require(set(mapping) == set(manual['readings']) and isinstance(manual['evaluator'], str), 'Incomplete whole-response readings')
    for layout, expected in manifest['generation_reports_sha256'].items():
        require(sha(run/('generation_'+layout)/'report.json') == expected, 'Generation changed after review')
    for key, row in mapping.items():
        require(sha(review/'pages'/(key+'.png')) == row['image_sha256'], 'Viewed output changed')
    for board in manifest['opaque_boards']:
        require(sha(review/board['file']) == board['sha256'], 'Viewed board changed')
        with Image.open(review/board['file']) as surface:
            for tile in board['tiles']:
                with Image.open(review/'pages'/(tile['opaque']+'.png')) as original:
                    crop = surface.crop((tile['x'], tile['y'], tile['x']+tile['width'], tile['y']+tile['height']))
                    require(crop.tobytes() == original.convert('L').tobytes(), 'Board tile differs from raw output')
    return mapping, manual


def scores(run, readings):
    mapping, manual = authenticate_review(run, readings)
    review = run/'review'; all_scores = {}
    for layout in LAYOUTS:
        destination = review/layout; require(not destination.exists(), 'Preserve earlier scores')
        destination.mkdir(); (destination/'pages').symlink_to((review/'pages').resolve(), target_is_directory=True)
        selected = {key:{k:v for k, v in row.items() if k not in ['layout', 'source', 'input_image']}
                    for key, row in mapping.items() if row['layout'] == layout}
        save_json(destination/'mapping.json', dict(preparation_sha256=sha(EVALUATIONS[layout]/'preparation.json'), rows=selected))
        subset = dict(evaluator=manual['evaluator'], readings={k:manual['readings'][k] for k in selected})
        save_json(destination/'readings.json', subset)
        with redirect_stdout(io.StringIO()):
            score(EVALUATIONS[layout], destination, destination/'readings.json')
        result = load(destination/'scores.json')
        # Reuse the unchanged scorer in a new directory, independent of saved aggregates.
        with tempfile.TemporaryDirectory(prefix='ilm-context-motor-score-') as temporary:
            folder = Path(temporary)
            shutil.copy2(destination/'mapping.json', folder/'mapping.json')
            (folder/'pages').symlink_to((review/'pages').resolve(), target_is_directory=True)
            with redirect_stdout(io.StringIO()):
                score(EVALUATIONS[layout], folder, destination/'readings.json')
            require(load(folder/'scores.json') == result, 'Whole-response score recount differs')
        all_scores[layout] = result
    normal = lambda result:[r for r in result['records'] if r['kind'] == 'qa' and r['condition'] == 'normal']
    retained = {}
    for layout in LAYOUTS:
        old = {r['case'] for r in normal(load(CONTROLS[layout]/'review/scores.json')) if r['complete_exact']}
        new = {r['case'] for r in normal(all_scores[layout]) if r['complete_exact']}
        require(len(old) == (2 if layout == 'fixed_cells' else 6), 'Original retention denominator changed')
        retained[layout] = dict(required=sorted(old), retained=sorted(old & new), lost=sorted(old-new), all_retained=old <= new)
    proportional = next(r for r in all_scores['proportional']['results'] if r['kind'] == 'qa' and r['condition'] == 'normal')
    pilot = (proportional['complete_exact'] >= 8 and proportional['paired_both_correct'] >= 2
             and proportional['readable'] >= 11 and all(x['all_retained'] for x in retained.values()))
    save_json(review/'comparison.json', dict(results={layout:all_scores[layout]['results'] for layout in LAYOUTS},
        original_exact_retention=retained, prospective_development_pilot_passed=pilot,
        readings_sha256=sha(readings), review_manifest_sha256=sha(review/'manifest.json'),
        whole_response_scores_recounted=True, reused_development_cohort=True,
        untouched_confirmation_complete=False, cold_runtime_reproduction_complete=False, goal_achieved=False))
    print(json.dumps(dict(results={layout:all_scores[layout]['results'] for layout in LAYOUTS},
                         pilot_passed=pilot, goal_achieved=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'score'])
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_context_motor_v1')
    parser.add_argument('--readings', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.run.resolve())
    else:
        require(args.readings is not None, 'Supply complete frozen raw-output readings')
        scores(args.run.resolve(), args.readings.resolve())
