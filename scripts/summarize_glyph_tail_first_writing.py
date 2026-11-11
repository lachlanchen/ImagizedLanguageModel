#!/usr/bin/env python3
"""Independently recount saved first-image pixels and export the complete CPU audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.audit_glyph_tail_first_writing import validate, tensor_sha

OWN = 'scripts/summarize_glyph_tail_first_writing.py'


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, where='value'):
    """Compare every field, retaining zeros, missing cases and original rounding."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Different fields: '+where)
        for key in expected:
            same(actual[key], expected[key], where+'.'+key)
    elif isinstance(expected, list):
        require(len(actual) == len(expected), 'Different length: '+where)
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, where+f'[{i}]')
    elif isinstance(expected, float):
        require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), 'Different number: '+where)
    else:
        require(actual == expected, 'Different value: '+where)


def recount(images, targets, present=None):
    """Vectorized integer contingency counts, independent of the generating scorer."""
    a = images.astype(bool).reshape(len(images), -1)
    b = targets.astype(bool).reshape(len(targets), -1)
    tp = np.count_nonzero(a & b, axis=1)
    fp = np.count_nonzero(a & ~b, axis=1)
    fn = np.count_nonzero(~a & b, axis=1)
    present = np.ones(len(images), bool) if present is None else np.asarray(present, bool)
    rows = []
    for i in range(len(images)):
        t, p, n = int(tp[i]), int(fp[i]), int(fn[i])
        denominator = 2*t+p+n
        rows.append(dict(tp=t, fp=p, fn=n, target_ink=t+n, predicted_ink=t+p,
            pixel_exact=bool(present[i] and p+n == 0),
            foreground_f1=(2*t/denominator if denominator else 1.) if present[i] else 0.))
    return rows


def group(rows):
    if not rows:
        return dict(cases=0, exact=0, mean_foreground_f1=None, aggregate_foreground_f1=None)
    t, p, n = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
    return dict(cases=len(rows), exact=sum(r['pixel_exact'] for r in rows),
        mean_foreground_f1=sum(r['foreground_f1'] for r in rows)/len(rows),
        aggregate_foreground_f1=2*t/(2*t+p+n) if 2*t+p+n else 1., tp=t, fp=p, fn=n)


def main(run, output, public):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA for independent CPU scoring')
    require(subprocess.check_output(['git', 'show', 'HEAD:'+OWN], cwd=ROOT) == (ROOT/OWN).read_bytes(),
            'Commit the independent scorer before running it')
    require(not output.exists() and not public.exists(), 'Preserve previous analyses and exports')
    torch.set_num_threads(2)
    r = validate(run)
    report_path = run/'execution/report.json'
    report = load(report_path)
    require(report['registration_sha256'] == sha(run/'registration.json'), 'Registration changed')
    raw_path = run/'execution/pixels.npz'
    require(sha(raw_path) == report['raw_pixels_sha256'], 'Original pixel archive changed')
    require(not report['new_optimizer_updates'] and not report['new_contextual_model_calls']
            and not report['new_autonomous_ILM_images'] and not report['CUDA_initialized'],
            'This is a frozen CPU audit, not new language generation')
    prepared = torch.load(run/'prepared_pixels.pt', map_location='cpu', weights_only=True)
    with np.load(raw_path, allow_pickle=False) as archive:
        require(set(archive.files) == {'gold_ink', 'original_first_ink', 'supplied_reconstruction_ink'},
                'Unexpected pixel archive contents')
        arrays = {k: archive[k] for k in archive.files}
    for name, shape in [('gold_ink', (745, 1, 32, 32)), ('original_first_ink', (192, 1, 32, 32)),
                        ('supplied_reconstruction_ink', (745, 1, 32, 32))]:
        require(arrays[name].shape == shape and arrays[name].dtype == np.uint8
                and np.isin(arrays[name], [0, 1]).all(), 'Noncanonical pixels: '+name)
    for name in ['gold_ink', 'original_first_ink']:
        require(np.array_equal(arrays[name], prepared[name].numpy()), 'Prepared pixels changed: '+name)
    jobs, targets = r['jobs'], r['targets']
    require(len(jobs) == 192 and len(targets) == 32, 'Keep all registered cases')
    for i, job in enumerate(jobs):
        require(tensor_sha(prepared['original_first_states'][i]) == job['original_first_state_sha256'],
                'Original first state changed')
        require(not job['first_present'] or tensor_sha(prepared['original_first_ink'][i])
                == job['original_first_ink_sha256'], 'Original first image changed')
    # Recount coverage from the authenticated primitive data, not stored Boolean totals.
    primitive = ROOT/'artifacts/semantic_glyph_bridge_v1'
    inventory = load(primitive/'inventory.json')['records']
    training = torch.load(primitive/'training.pt', map_location='cpu', weights_only=True)
    characters = {c for row in inventory for c in row['chars']}
    bitmaps = {hashlib.sha256(p.numpy().tobytes()).hexdigest() for p in training['ink']}
    full = recount(arrays['supplied_reconstruction_ink'], arrays['gold_ink'])
    full_rows, first_offsets, covered = [], [], []
    missing, complete = Counter(), 0
    offset = 0
    for target in targets:
        require(target['offset'] == offset and target['glyphs'] == len(target['answer']), 'Target offsets differ')
        gold = prepared['gold_ink'][offset:offset+target['glyphs']]
        require(tensor_sha(gold) == target['source_glyphs_sha256'], 'Source target image changed')
        char_flags, bitmap_flags = [], []
        first_offsets.append(offset)
        for position, character in enumerate(target['answer']):
            char_ok = character in characters
            bitmap_ok = hashlib.sha256(arrays['gold_ink'][offset+position].tobytes()).hexdigest() in bitmaps
            char_flags.append(char_ok); bitmap_flags.append(bitmap_ok); covered.append(char_ok)
            if not char_ok:
                missing[character] += 1
            full_rows.append(dict(case=target['case'], position=position, primitive_character_covered=char_ok,
                primitive_bitmap_covered=bitmap_ok, **full[offset+position]))
        same(char_flags, target['primitive_character_coverage'])
        same(bitmap_flags, target['primitive_bitmap_coverage'])
        complete += all(char_flags)
        offset += target['glyphs']
    require(offset == 745 and len(characters) == 4096 and len(inventory) == 4087, 'Coverage denominator changed')
    coverage = dict(target_questions=32, first_character_covered=sum(t['primitive_character_coverage'][0] for t in targets),
        first_bitmap_covered=sum(t['primitive_bitmap_coverage'][0] for t in targets), all_target_positions=745,
        all_target_character_covered=sum(covered),
        all_target_bitmap_covered=sum(x['primitive_bitmap_covered'] for x in full_rows),
        complete_answers_character_covered=complete, missing_characters=dict(missing),
        continuous_decoder_not_inventory_limited_by_construction=True)
    offsets = [j['target_first_offset'] for j in jobs]
    first = recount(arrays['original_first_ink'], arrays['gold_ink'][offsets], [j['first_present'] for j in jobs])
    records = [dict(job=j, original_first=x, supplied_first=full[j['target_first_offset']])
               for j, x in zip(jobs, first)]
    aggregates, subgroups = [], []
    for layout in ['fixed_cells', 'proportional']:
        for condition in ['normal', 'question_blank', 'passage_blank']:
            ix = [i for i, j in enumerate(jobs) if j['layout'] == layout and j['condition'] == condition]
            require(len(ix) == 32, 'Incomplete layout/erasure arm')
            aggregates.append(dict(layout=layout, condition=condition, **group([first[i] for i in ix])))
            if condition != 'normal':
                continue
            for native in [False, True]:
                for glyph in [False, True]:
                    chosen = [first[i] for i in ix if jobs[i]['native_complete_exact'] == native
                              and jobs[i]['glyph_complete_exact'] == glyph]
                    subgroups.append(dict(layout=layout, native_complete_exact=native,
                                          glyph_complete_exact=glyph, **group(chosen)))
    metrics = dict(records=records, supplied_all_target_positions=full_rows, primitive_coverage=coverage,
        original_first_aggregates=aggregates, outcome_subgroups=subgroups,
        supplied_first_32=group([full[i] for i in first_offsets]), supplied_all_targets=group(full),
        supplied_covered_targets=group([x for x, c in zip(full, covered) if c]),
        supplied_missing_targets=group([x for x, c in zip(full, covered) if not c]),
        initial_stop_count=sum(j['initial_stop_probability'] >= .5 for j in jobs))
    for key, value in metrics.items():
        same(value, report[key], key)
    # Verify every displayed tile; no resampling, repair, or selected-case omission.
    job_index = {j['id']:i for i, j in enumerate(jobs)}
    checked = 0
    for board_number, board in enumerate(report['boards']):
        path = run/'execution'/board['file']
        require(sha(path) == board['sha256'], 'Board changed')
        with Image.open(path) as image:
            require(image.size == (840, 1424) and image.mode == 'L', 'Board geometry changed')
            pixels = np.asarray(image)
        for row, target in enumerate(targets[16*board_number:16*(board_number+1)]):
            n = target['offset']; case = target['case']
            shown = [arrays['gold_ink'][n], arrays['supplied_reconstruction_ink'][n],
                arrays['original_first_ink'][job_index['fixed_cells__'+case+'_normal']],
                arrays['original_first_ink'][job_index['proportional__'+case+'_normal']]]
            for x, original in zip([155, 320, 500, 710], shown):
                expected = (255*(1-original[0])).repeat(2, 0).repeat(2, 1)
                y = 55+84*row
                require(np.array_equal(pixels[y:y+64, x:x+64], expected), 'Displayed tile differs from original')
                checked += 1
    require(checked == 128 and not torch.cuda.is_initialized(), 'Incomplete CPU display verification')
    output.mkdir(parents=True); public.mkdir(parents=True)
    for source, name in [(raw_path, 'pixels.npz')]+[(run/'execution'/b['file'], Path(b['file']).name) for b in report['boards']]:
        shutil.copy2(source, public/name)
        require(sha(source) == sha(public/name), 'Public export differs')
    value = dict(schema='glyph-tail-first-writing-learning-v1', **metrics,
        registration_sha256=sha(run/'registration.json'), original_report_sha256=sha(report_path),
        checkpoint_sha256=r['checkpoint_sha256'], audit_source_sha256=r['sources'], independent_scorer_sha256=sha(ROOT/OWN),
        public_files_sha256={p.name:sha(p) for p in public.iterdir()},
        compatible_primitive_characters=4096, compatible_primitive_images=4087, verified_display_tiles=checked,
        font=dict(name='NotoSansCJKSC', collection_index=2, size=26, baseline=26, tile=[32, 32],
                  sha256='b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a'),
        all192_original_streams_preserved=True, all745_original_targets_authenticated=True,
        all192_first_state_bindings_verified=True, independent_integer_pixel_recount_passed=True,
        CPU_reconstruction_is_answer_supplied=True, first_pixel_mismatch_is_not_semantic_error=True,
        visual_review='All32 rows viewed in two labelled boards; prior cohort familiarity; no independent human review.',
        helper_or_development_answers_used_for_training=False, new_optimizer_updates=0,
        new_contextual_model_calls=0, new_autonomous_ILM_images=0, goal_achieved=False,
        CPU_reconstruction_resources={k:report[k] for k in ['elapsed_seconds', 'peak_RSS_gib', 'resource_wait_seconds', 'CUDA_initialized']})
    require(not any(s in json.dumps(value, ensure_ascii=False) for s in ['/home/', '"pid"', 'boot_id']),
            'Private runtime metadata in public evidence')
    save_json(public.parent/'glyph_tail_first_writing_learning_20261111.json', value)
    save_json(output/'verification.json', dict(all_checks_passed=True, raw_pixels_sha256=sha(raw_path),
        source_report_sha256=sha(report_path), public_evidence_sha256=sha(public.parent/'glyph_tail_first_writing_learning_20261111.json'),
        checked_display_tiles=checked, CPU_only=True, new_model_calls=0))
    print(json.dumps(dict(all_checks_passed=True, original_streams=192, target_positions=745,
                         verified_display_tiles=checked, goal_achieved=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_tail_first_writing_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/glyph_tail_first_writing_learning_v1')
    parser.add_argument('--public', type=Path, default=ROOT/'references/evidence/glyph_tail_first_writing_20261111')
    args = parser.parse_args()
    main(args.run.resolve(), args.output.resolve(), args.public.resolve())
