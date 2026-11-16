#!/usr/bin/env python3
"""Recount the completed white-target comparison from its sealed saved pixels."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/white_target_ranking_v1'
OUTPUT = ROOT / 'references/evidence/white_target_physical_result_20261116.json'
REGISTRATION_SHA = '04b908f488f5175bff1df8da1a7d38676696c03a3ab5f70faa6a199a1757d828'
GATE_SHA = '597b46676f48ab15f4c9d8aab35ca1678a4c722d0901f799259974e65aac6123'
FROZEN_SHA = 'b56d9833291715c0894345b78349e366875e1f5035f64c98f009f158a6826711'
FIELDS = ['positions', 'source_whitespace', 'false_ink', 'false_blanks', 'source_pixel_exact']


def load(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def build():
    began = time.perf_counter()

    def guard():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        assert int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2
        assert time.perf_counter() - began <= 900
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2

    guard()
    from scripts import review_white_target_physical as workflow
    workflow.configure()
    (qualify, registration, report, manifest, ids, bindings), session = workflow.review.session_context()
    assert bindings['registration_sha256'] == REGISTRATION_SHA
    assert sha(qualify / 'gate.json') == GATE_SHA
    assert sha(qualify / 'frozen_readings.json') == FROZEN_SHA
    frozen = workflow.review.read_frozen_bundle(qualify, manifest, ids, bindings)
    workflow.checked_readings(frozen['readings'], ids)
    gate = load(qualify / 'gate.json')
    assert gate['bindings'] == bindings and gate['frozen_readings_sha256'] == FROZEN_SHA
    assert not gate['passed'] and gate['readable_groups_passed'] and gate['credited_candidate_ink_passed']
    assert not gate['whitespace_passed'] and not gate['false_blanks_passed']
    assert gate['blank_flags_verified_against_opaque_pixels'] and not gate['credited_empty_candidate_ids']
    assert (gate['maximum_false_ink'], gate['maximum_false_blanks']) == (11, 37)
    assert len(ids) == 2080 and len(manifest['boards']) == 33 and len(report['records']) == 124
    fit, audit = [load(RUN / stage / 'report.json') for stage in ['fit', 'audit']]
    assert fit['complete'] and audit['complete'] and report['complete']
    assert audit['fit_report_sha256'] == sha(RUN / 'fit/report.json')
    assert audit['retained_updates'] == 41298 and audit['executed_events_all_attempts'] == 41300
    assert audit['retained_target_exposure'] == 6108741 and audit['retained_white_target_exposure'] == 68808
    assert [s['step'] for s in audit['snapshots']] == [0, 10324, 20649, 30973, 41298]
    assert all(s['trainable_tensors'] == 131 for s in audit['snapshots'])
    control_root = ROOT / registration['control_physical_pixels']
    control_report_path = control_root.parent / 'report.json'
    control_report = load(control_report_path)
    controls = {row['job']['id']: row for row in control_report['records']}
    gate_cases = {row['case']: row for row in gate['complete_case_counts']}
    assert len(gate_cases) == 124
    counts = defaultdict(lambda: dict(positions=0, source_whitespace=0, false_ink=0,
        false_blanks=0, source_pixel_exact=0, source_mask_nll_sum=0.0))
    cases = []
    for record in report['records']:
        guard()
        job = record['job']; control = controls[job['id']]
        assert control['job'] == job and control['target_glyphs'] == record['target_glyphs']
        path = qualify / 'cases' / job['id'] / 'pixels.npz'
        old_path = control_root / job['id'] / 'pixels.npz'
        assert sha(path) == record['pixels_sha256'] and sha(old_path) == control['pixels_sha256']
        with np.load(path, allow_pickle=False) as arrays, np.load(old_path, allow_pickle=False) as prior:
            gold = arrays['gold_ink']
            assert np.array_equal(gold, prior['gold_ink'])
            assert np.array_equal(arrays['parent_ink'], prior['candidate_ink'])
            assert gold.shape == (record['target_glyphs'], 1, 32, 32)
            assert gold.dtype == np.uint8 and np.isin(gold, [0, 1]).all()
            white = np.count_nonzero(gold.reshape(len(gold), -1), axis=1) == 0
            for role, name, metrics in [('candidate', 'candidate_ink', record), ('parent', 'parent_ink', control)]:
                output = arrays[name]
                assert output.shape == gold.shape and output.dtype == np.uint8 and np.isin(output, [0, 1]).all()
                empty = np.count_nonzero(output.reshape(len(output), -1), axis=1) == 0
                exact = np.count_nonzero((output != gold).reshape(len(gold), -1), axis=1) == 0
                assert int(exact.sum()) == metrics['canonical_exact']
                row = dict(role=role, case=job['id'], split=job['split'], kind=job['kind'], layout=job['layout'],
                    positions=len(gold), source_whitespace=int(white.sum()),
                    false_ink=int(np.count_nonzero(white & ~empty)),
                    false_blanks=int(np.count_nonzero(~white & empty)), source_pixel_exact=int(exact.sum()))
                if role == 'candidate':
                    assert all(row[k] == gate_cases[job['id']][k] for k in FIELDS[:-1])
                cases.append(row)
                key = (role, job['split'], job['kind'], job['layout'])
                for field in FIELDS:
                    counts[key][field] += row[field]
                counts[key]['source_mask_nll_sum'] += metrics['pattern_nll_mean'] * len(gold)
    groups = []
    for key, value in sorted(counts.items()):
        value = dict(value)
        value['source_mask_nll_mean'] = value.pop('source_mask_nll_sum') / value['positions']
        groups.append(dict(zip(['role', 'split', 'kind', 'layout'], key), **value))
    totals = {}
    for role in ['candidate', 'parent']:
        rows = [r for r in groups if r['role'] == role]
        totals[role] = {k: sum(r[k] for r in rows) for k in FIELDS}
        selected = [r for r in gate['groups'] if r['role'] == role]
        totals[role]['readable_nonblank_selected'] = sum(r['readable_nonblank'] for r in selected)
        totals[role]['selected_positions'] = sum(r['count'] for r in selected)
    assert totals['candidate']['positions'] == totals['parent']['positions'] == 12597
    assert totals['candidate']['source_whitespace'] == totals['parent']['source_whitespace'] == 252
    assert (totals['candidate']['false_ink'], totals['candidate']['false_blanks']) == (12, 54)
    assert (totals['parent']['false_ink'], totals['parent']['false_blanks']) == (25, 37)
    assert (totals['candidate']['readable_nonblank_selected'], totals['parent']['readable_nonblank_selected']) == (1016, 1010)
    controller = load(RUN / 'controller_status.json')
    assert controller['pid'] is None and controller['phase'] == 'awaiting_opaque_physical_review'
    assert not (RUN / 'language').exists()
    assert not (ROOT / 'references/evidence/white_target_language_registration_v1.json').exists()
    ended = [json.loads(line) for line in (RUN / 'controller_journal.jsonl').read_text().splitlines()
             if json.loads(line)['event'] == 'ended']
    assert [row['action'] for row in ended] == ['train', 'audit', 'qualify']
    assert all(row['returncode'] == 0 for row in ended)
    entry_history = sorted((qualify / 'review/entry_attempts').glob('*.json'))
    guard()
    result = dict(schema='white-target-physical-result-v1', complete=True, evidence_date='2026-11-16',
        physical_gate_passed=False, readable_groups_passed=True, credited_candidate_ink_passed=True,
        whitespace_passed=False, false_blanks_passed=False, maximum_false_ink=11, maximum_false_blanks=37,
        language_quality_unmeasured=True, goal_achieved=False,
        new_neural_calls_from_exporter=0, new_optimizer_updates_from_exporter=0,
        registration_sha256=REGISTRATION_SHA, gate_sha256=GATE_SHA, frozen_readings_sha256=FROZEN_SHA,
        fit_report_sha256=sha(RUN / 'fit/report.json'), audit_report_sha256=sha(RUN / 'audit/report.json'),
        qualification_report_sha256=sha(qualify / 'report.json'), control_qualification_sha256=sha(control_report_path),
        review_session_sha256=sha(qualify / 'review/session.json'),
        freeze_receipt_sha256=sha(qualify / 'review/freeze_receipt.json'), exporter_sha256=sha(Path(__file__)),
        model_account=fit['model_account'], retained_updates=41298, extra_executed_replay_updates=2,
        source_count=13766, retained_target_exposure=6108741, retained_white_target_exposure=68808,
        snapshots=audit['snapshots'], physical_review=dict(boards=33, images=2080,
            reviewer_kind=session['reviewer_kind'], prior_development_familiarity=True,
            independent_human_adjudication=False,
            uncertain_readings=sum(r['uncertain'] for r in frozen['readings'].values()),
            pixel_empty_readings=sum(r['blank'] for r in frozen['readings'].values()),
            credited_empty_candidate_ids=[], one_freeze_before_owner_access=True, frozen_readings_repaired=False,
            original_unfrozen_entries_and_rejections_preserved=True,
            entry_history_sha256={p.name: sha(p) for p in entry_history}),
        readable_groups=gate['groups'], complete_groups=groups, totals=totals, complete_case_counts=cases,
        resources=dict(coordinator_stage_seconds={r['action']: r['elapsed_seconds'] for r in ended},
            fit={k: fit[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
            audit={k: audit[k] for k in ['elapsed_seconds', 'peak_rss_gib', 'minimum_available_gib']},
            qualification={k: report[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
            physical_gate_cpu={k: gate[k] for k in ['elapsed_seconds', 'peak_rss_gib']},
            inherited_pretraining_and_prior_work_excluded=True))
    return result, dict(elapsed_seconds=time.perf_counter() - began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Use CPU-only export'
    result, cost = build()
    if args.verify:
        assert load(OUTPUT) == result, 'Deterministic public result differs'
    else:
        with OUTPUT.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
    print(json.dumps(dict(verified=args.verify, output=str(OUTPUT.relative_to(ROOT)), result_sha256=sha(OUTPUT), **cost)))


if __name__ == '__main__':
    main()
