#!/usr/bin/env python3
"""One-sided CPU falsification; an incomplete teacher run can never pass."""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from scripts import calibrate_isolated_glyph_judge_gpu0 as recovery

base = recovery.base
ROOT = base.ROOT
REG = ROOT / 'references/evidence/isolated_glyph_partial_falsification_registration_20261116.json'
OUT = ROOT / 'references/evidence/isolated_glyph_partial_falsification_20261116.json'
PROTOCOL = ROOT / 'references/isolated_glyph_partial_falsification_20261116.md'


def reference_failure(reading, expected):
    if reading is None:
        return 'unknown'
    if not reading['accepted']:
        return 'observed_abstention'
    return 'observed_wrong_character' if reading['text'] != expected else None


def register():
    recovery.configure(); trial = base.load(base.REG); inventory = base.authenticate(trial)
    failure = base.load(base.RUN / 'failure.json')
    paths = sorted((base.RUN / 'responses').glob('*.json'))
    if len(paths) != failure['completed_responses'] or len(paths) != 402:
        raise ValueError('Preserve the full first aborted recovery cache')
    keys = [p.stem for p in paths]
    if keys != inventory['query_order'][:len(keys)] or (base.RUN / 'transcription_freeze.json').exists():
        raise ValueError('Require the exact interrupted prefix, never selected readings')
    responses = {p.stem: base.sha(p) for p in paths}
    # Hash bytes only. Do not open response contents, reference strings or review labels here.
    freeze_path = base.RUN / 'partial_response_freeze.json'
    base.write_new(freeze_path, dict(registration_sha256=base.sha(base.REG),
        response_sha256=responses, completed=402, prescribed=526, missing=124,
        incomplete=True, calibration_pass_impossible=True))
    sources = {base.relative(p): base.sha(p) for p in [Path(__file__), PROTOCOL, base.REG,
        base.RUN / 'failure.json', recovery.RUNTIME / 'transcribe_shutdown.json', freeze_path,
        ROOT / 'tests/test_partial_isolated_falsification.py']}
    base.write_new(REG, dict(schema='partial-isolated-judge-falsification-registration-v1',
        input_sha256=sources, response_sha256=responses, complete_cohort=64,
        observed_queries=402, missing_queries=124, whole_run_calibration_complete=False,
        rule='Any observed wrong/abstained clean reference, known-correct forced-zero answer, or fully read wrong/uncertain full-content acceptance proves a necessary fixed condition false. Missing data remain unknown. No pass is possible.',
        resources=dict(cpu_threads=2, elapsed_seconds=300, maximum_rss_gib=2, minimum_host_available_gib=24),
        training_authorized=False, neural_calls=0, action_on_no_failure='Inconclusive; do not pass or resume the aborted run.',
        action_on_failure='Reject fixed judge using necessary-condition violation; no fit or further teacher run.'))
    print(json.dumps(dict(registered=base.relative(REG), frozen=402, missing=124, neural_calls=0)))


def run():
    began = time.monotonic(); recovery.configure(); reg = base.load(REG)
    base.git_admission()  # Recovery registration remains committed; this analysis is separately bound below.
    for name, digest in reg['input_sha256'].items():
        if base.sha(ROOT / name) != digest:
            raise ValueError('Fixed analysis source changed: ' + name)
    trial = base.load(base.REG); inventory = base.authenticate(trial)
    committed = __import__('subprocess').check_output(['git', 'show', 'HEAD:' + base.relative(REG)], cwd=ROOT)
    if __import__('hashlib').sha256(committed).hexdigest() != base.sha(REG):
        raise ValueError('Commit and push the one-sided analysis before unmasking')
    readings = {}
    for key, digest in reg['response_sha256'].items():
        path = base.RUN / 'responses' / (key + '.json')
        if base.sha(path) != digest:
            raise ValueError('Preserve every frozen first response')
        raw = base.load(path)
        if raw['cell'] != key or raw['reading'] != base.judge.parse_response(raw['response']):
            raise ValueError('Strict reading differs from preserved response')
        readings[key] = raw['reading']
    for key, row in inventory['cells'].items():
        if row['blank']:
            readings[key] = dict(text=' ', accepted=True, reason='physically_white')
    refs = {r['id']: r for r in base.load(base.REFS)['reference_rows'] if r['kind'] == 'qa'}
    truth = {layout: {r['case']: r for r in base.load(base.OLD / 'review' / (layout + '_core_scores.json'))['records']
                     if r['step'] == 41298 and r['kind'] == 'qa' and r['condition'] == 'normal'}
             for layout in ('fixed_cells', 'proportional')}
    records = []; reference_failures = []; reference_unknown = []; aliases = {}
    forced_zero = []; false_content = []
    for case in inventory['cases']:
        base.guard(began, 300)
        old = truth[case['layout']][case['case']]
        text = refs[case['case'].removesuffix('_normal')]['answer']
        flags = [inventory['cells'][k]['blank'] for k in case['reference_cells']]
        for position, (key, char, blank) in enumerate(zip(case['reference_cells'], text, flags)):
            expected = ' ' if blank else char
            aliases.setdefault(key, set()).add(expected)
            reason = reference_failure(readings.get(key), expected)
            row = dict(case=case['case'], layout=case['layout'], position=position,
                       cell=key, expected=expected, reading=readings.get(key), reason=reason)
            if reason == 'unknown': reference_unknown.append(row)
            elif reason: reference_failures.append(row)
        known = case['stopped'] and all(k in readings for k in case['emitted_cells'])
        row = dict(case=case['case'], layout=case['layout'], stopped=case['stopped'],
            complete_exact=old['complete_exact'], readable=old['readable'], uncertain=old['reading']['uncertain'],
            emitted_glyphs=case['emitted_glyphs'], all_emitted_cells_observed=known,
            missing_cell_occurrences=(sum(k not in readings for k in case['emitted_cells'])
                                      if case['stopped'] else case['emitted_glyphs']))
        # Known abstention forces zero regardless of the missing cells.
        zero = not case['stopped'] or any(k in readings and not readings[k]['accepted'] for k in case['emitted_cells'])
        row['reward_provably_zero'] = zero
        if zero and old['complete_exact']: forced_zero.append(dict(case=case['case'], layout=case['layout']))
        if known or not case['stopped']:
            value = base.judge.score_sequence(case['emitted_cells'], readings, text, flags, stopped=case['stopped'])
            row.update(value)
            if value['complete_content'] and (not old['readable'] or not old['complete_exact']):
                false_content.append(dict(case=case['case'], layout=case['layout'], frozen_uncertain=old['reading']['uncertain']))
        else:
            row.update(reward=0.0 if zero else None, complete_content=None, transcript=None)
        records.append(row)
    if (len(records), sum(r['complete_exact'] for r in records), sum(r['readable'] for r in records)) != (64, 9, 18):
        raise ValueError('Retain the complete original cohort')
    violations = dict(observed_clean_reference_failures=len(reference_failures),
        known_correct_forced_zero=len(forced_zero), fully_read_uncredited_content_accepted=len(false_content))
    rejected = any(violations.values())
    base.guard(began, 300)
    output = dict(schema='partial-isolated-judge-falsification-v1', registration_sha256=base.sha(REG),
        original_trial_registration_sha256=base.sha(base.REG), calibration_complete=False,
        qualification_rejected=rejected, partial_evidence_inconclusive=not rejected,
        observed_queries=402, missing_queries=124, cases=records, violations=violations,
        observed_reference_failures=reference_failures, unknown_reference_occurrences=len(reference_unknown),
        physical_reference_aliases={k: sorted(v) for k, v in aliases.items() if len(v) > 1},
        known_correct_forced_zero=forced_zero, fully_read_false_content=false_content,
        all_original_readings_unchanged=True, reused_development=True, independent_human_adjudication=False,
        elapsed_seconds=time.monotonic() - began, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        new_model_calls=0, new_optimizer_updates=0, training_authorized=False, goal_achieved=False)
    base.write_new(OUT, output)
    print(json.dumps({k: output[k] for k in ['calibration_complete','qualification_rejected', 'violations',
        'unknown_reference_occurrences','physical_reference_aliases','elapsed_seconds','peak_rss_gib']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=('register','run'))
    args = parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '2' or os.environ.get('OMP_NUM_THREADS') != '2':
        raise ValueError('Require the fixed two-thread CPU analysis')
    register() if args.action == 'register' else run()
