#!/usr/bin/env python3
"""Bind Figure 22 to complete language records and original diagnostic arrays."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = ROOT / 'references/evidence'
STAGE = Path(os.environ.get('ILM_REALIZED_PAPER_STAGE',
                            str(ROOT / 'artifacts/realized_writer_publication_v1')))
BASELINE = '61addf3f'
LAYOUTS = ['fixed_cells', 'proportional']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def preserve():
    files = {}
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
        BASELINE, '--', 'publication/ilm-nature'], cwd=ROOT, text=True).splitlines()
    for name in names:
        relative = str(Path(name).relative_to('publication/ilm-nature'))
        if not (relative.startswith('figures/') or relative == 'nature_reading.sty'
                or relative.startswith(('supplementary_information',
                                        'redlined_supplementary'))):
            continue
        raw = subprocess.check_output(['git', 'show', BASELINE + ':' + name], cwd=ROOT)
        if raw.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            expected = next(line.split('sha256:', 1)[1] for line in raw.decode().splitlines()
                            if line.startswith('oid sha256:'))
        else:
            expected = hashlib.sha256(raw).hexdigest()
        if sha(HERE / relative) != expected:
            raise ValueError('Earlier asset changed: ' + relative)
        files[relative] = expected
    return dict(baseline_commit=BASELINE, LFS_content_identity_used=True,
                files_sha256=files)


def recount_physical(path, archive, routes):
    record = load(path)
    if not (record['visual_review_complete'] and record['all512_displayed_tiles_bit_exact']):
        raise ValueError('Complete diagnostic review required')
    with np.load(archive, allow_pickle=False) as arrays:
        gold = arrays['gold_ink']
        if gold.shape != (128, 1, 32, 32) or len(record['records']) != 128:
            raise ValueError('Keep all original positions')
        scores = {}
        for group in ['train_first', 'train_later', 'fixed_cells', 'proportional']:
            selected = [i for i, row in enumerate(record['records']) if row['group'] == group]
            if len(selected) != 32:
                raise ValueError('Keep all32 positions in each pool')
            scores[group] = {}
            for route, key in routes.items():
                predicted = arrays[key]
                if predicted.shape != gold.shape or not np.isin(predicted, [0, 1]).all():
                    raise ValueError('Original binary glyphs required')
                values, exact = [], 0
                for index in selected:
                    p, g = predicted[index].astype(bool), gold[index].astype(bool)
                    tp, fp, fn = int((p & g).sum()), int((p & ~g).sum()), int((~p & g).sum())
                    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.
                    original = record['records'][index]['scores'][route]
                    if ((tp, fp, fn) != (original['tp'], original['fp'], original['fn'])
                            or f1 != original['foreground_f1']):
                        raise ValueError('Per-position integer recount differs')
                    values.append(f1)
                    exact += int(np.array_equal(p, g))
                summary = dict(cases=32, exact=exact, mean_foreground_f1=float(np.mean(values)),
                    median_foreground_f1=float(np.median(values)),
                    at_least_095=sum(value >= .95 for value in values))
                if summary != record['groups'][group][route]:
                    raise ValueError('Physical group recount differs')
                scores[group][route] = summary
    return scores


def main():
    # Fail before exporting a figure if the live language study is unfinished.
    balanced_path = EVIDENCE / 'balanced_writer_matched_learning_20261111.json'
    if not balanced_path.exists():
        raise ValueError('Finish all216 outputs and frozen whole-answer scoring first')
    balanced = load(balanced_path)
    if not (balanced['actual_outputs'] == 216
            and balanced['independent_per_answer_and_aggregate_recount_passed']
            and balanced['all216_initial_context_states_bit_exact_to_parent']
            and balanced['opaque_boards_viewed'] == 54 and balanced['book_prefixes_viewed'] == 12):
        raise ValueError('Complete authenticated language evidence required')
    paths = dict(balanced_language=balanced_path,
        flow_language=EVIDENCE / 'glyph_context_motor_learning_20261111.json',
        parent_language=EVIDENCE / 'glyph_tail_first_writing_learning_20261111.json',
        rollout=EVIDENCE / 'rollout_endpoint_motor_learning_20261111.json',
        balanced_physical=EVIDENCE / 'balanced_writer_matched_diagnosis_learning_20261111.json')
    archives = dict(rollout=EVIDENCE / 'rollout_endpoint_motor_20261111/pixels.npz',
        balanced=EVIDENCE / 'balanced_writer_matched_diagnosis_20261111/pixels.npz')
    rollout = recount_physical(paths['rollout'], archives['rollout'],
        dict(candidate='candidate_ink', control='control_ink', original_flow='original_flow_ink',
             parent='parent_ink', codec='supplied_ink'))
    physical = recount_physical(paths['balanced_physical'], archives['balanced'],
        dict(candidate='candidate_ink', parent='parent_ink', flow='flow_ink', codec='supplied_ink'))
    if not load(paths['rollout'])['whole_language_scores_unmeasured_for_both_new_arms']:
        raise ValueError('Do not substitute an unregistered rollout language score')
    flow, parent = load(paths['flow_language']), load(paths['parent_language'])
    qa = {model: {} for model in ['parent', 'flow', 'balanced']}
    for layout in LAYOUTS:
        jobs = [row['job'] for row in parent['records']
                if row['job']['layout'] == layout and row['job']['condition'] == 'normal']
        pairs = {}
        for job in jobs:
            pairs.setdefault(job['target_case'].split('_')[1], []).append(job)
        if len(jobs) != 32 or len(pairs) != 16 or not all(len(x) == 2 for x in pairs.values()):
            raise ValueError('Original QA denominators changed')
        qa['parent'][layout] = dict(exact=sum(j['glyph_complete_exact'] for j in jobs),
            pairs=sum(all(j['glyph_complete_exact'] for j in pair) for pair in pairs.values()),
            readable=sum(j['glyph_readable'] for j in jobs))
        for name, result in [('flow', flow), ('balanced', balanced)]:
            row = next(r for r in result['results'][layout]
                       if r['kind'] == 'qa' and r['condition'] == 'normal')
            if row['cases'] != 32:
                raise ValueError('Keep all32 normal questions')
            qa[name][layout] = dict(exact=row['complete_exact'], pairs=row['paired_both_correct'],
                                    readable=row['readable'])
    STAGE.mkdir(parents=True, exist_ok=True)
    value = dict(source_sha256={name: sha(p) for name, p in paths.items()},
        raw_archives_sha256={name: sha(p) for name, p in archives.items()},
        rollout_physical=rollout, balanced_physical=physical, qa=qa,
        balanced_pilot_passed=balanced['prospective_development_pilot_passed'],
        balanced_union_retention=balanced['union_exact_retention'],
        all256_physical_positions_recounted=True,
        language_independently_recounted_in_bound_source=True,
        both_rollout_language_scores=None, unmeasured_language_not_zero=True,
        inherited_contextual_parameters=7522463984, new_optimizer_updates=0,
        comparisons_not_learning_curve=True, F1_display_decimal_places=6,
        full_precision_F1_retained=True)
    (STAGE / 'measured_data.json').write_text(json.dumps(value, indent=2) + '\n')
    preserved = preserve()
    (STAGE / 'preserved_assets.json').write_text(json.dumps(preserved, indent=2) + '\n')
    print(json.dumps(dict(physical_positions_recounted=256, complete_balanced_outputs=216,
                         preserved_assets=len(preserved['files_sha256']))))


if __name__ == '__main__':
    main()
