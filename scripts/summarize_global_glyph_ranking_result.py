#!/usr/bin/env python3
"""Export the completed ranking comparison without neural calls or new grades."""
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
RUN = ROOT / 'artifacts/global_glyph_ranking_v1'
OUTPUT = ROOT / 'references/evidence/global_glyph_ranking_physical_result_20261115.json'
REGISTRATION_SHA = '8e928fdc78a397bfaca9dd3812b836723cffcdc1234772d802e8f815675c6c7f'
GATE_SHA = '55cc3f8797338665cd341cab0a85ed0c79280bf7342cd28ff0b37b624f302309'
FROZEN_SHA = '2e7e99c81b0fcda771b27ee50664c16b0a458fcb2cb5249a4d3eae0aaaabfc93'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def build():
    began = time.perf_counter()

    def guard():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        assert int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2
        assert time.perf_counter() - began <= 900
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2

    guard()
    from scripts import review_global_glyph_ranking as workflow
    workflow.configure()
    values, session = workflow.review.session_context()
    qualify, registration, report, manifest, ids, bindings = values
    assert bindings['registration_sha256'] == REGISTRATION_SHA
    assert sha(qualify / 'gate.json') == GATE_SHA
    assert sha(qualify / 'frozen_readings.json') == FROZEN_SHA
    frozen = workflow.review.read_frozen_bundle(qualify, manifest, ids, bindings)
    workflow.checked_readings(frozen['readings'], ids)
    gate = load(qualify / 'gate.json')
    assert gate['bindings'] == bindings and gate['frozen_readings_sha256'] == FROZEN_SHA
    assert not gate['passed'] and gate['readable_groups_passed']
    assert gate['credited_candidate_ink_passed'] and not gate['whitespace_passed']
    assert gate['blank_flags_verified_against_opaque_pixels']
    assert gate['source_whitespace'] == 252 and gate['false_ink'] == 25
    assert gate['maximum_false_ink'] == 11 and not gate['credited_empty_candidate_ids']
    assert len(ids) == 2080 and len(manifest['boards']) == 33
    assert len(report['records']) == 124
    fit, audit = [load(RUN / stage / 'report.json') for stage in ['fit', 'audit']]
    assert fit['complete'] and audit['complete'] and report['complete']
    assert audit['fit_report_sha256'] == sha(RUN / 'fit/report.json')
    assert audit['retained_updates'] == 41298 and audit['retained_target_exposure'] == 6108741
    assert audit['executed_events_all_attempts'] == 41300
    assert [s['step'] for s in audit['snapshots']] == [0, 10324, 20649, 30973, 41298]
    assert all(s['trainable_tensors'] == 131 for s in audit['snapshots'])
    control_root = ROOT / registration['control_physical_pixels']
    control_report_path = control_root.parent / 'report.json'
    control_report = load(control_report_path)
    controls = {r['job']['id']: r for r in control_report['records']}
    counts = defaultdict(lambda: dict(positions=0, source_whitespace=0, false_ink=0,
        false_blanks=0, source_pixel_exact=0, source_mask_nll_sum=0.0))
    case_counts = []
    for record in report['records']:
        guard()
        job = record['job']
        control = controls[job['id']]
        assert control['job'] == job and control['target_glyphs'] == record['target_glyphs']
        path = qualify / 'cases' / job['id'] / 'pixels.npz'
        assert sha(path) == record['pixels_sha256']
        old = control_root / job['id'] / 'pixels.npz'
        assert sha(old) == control['pixels_sha256']
        with np.load(path, allow_pickle=False) as arrays, np.load(old, allow_pickle=False) as prior:
            gold = arrays['gold_ink']
            assert np.array_equal(gold, prior['gold_ink'])
            assert np.array_equal(arrays['parent_ink'], prior['candidate_ink'])
            assert gold.shape == (record['target_glyphs'], 1, 32, 32)
            white = ~gold.reshape(len(gold), -1).any(1)
            for role, name, metrics in [('candidate', 'candidate_ink', record), ('parent', 'parent_ink', control)]:
                output = arrays[name]
                assert output.dtype == np.uint8 and np.isin(output, [0, 1]).all()
                blank = ~output.reshape(len(output), -1).any(1)
                exact = (output == gold).reshape(len(output), -1).all(1)
                assert int(exact.sum()) == metrics['canonical_exact']
                key = (role, job['split'], job['kind'], job['layout'])
                row = dict(role=role, case=job['id'], split=job['split'], kind=job['kind'],
                    layout=job['layout'], positions=len(gold), source_whitespace=int(white.sum()),
                    false_ink=int((white & ~blank).sum()), false_blanks=int((~white & blank).sum()),
                    source_pixel_exact=int(exact.sum()))
                case_counts.append(row)
                for field in ['positions', 'source_whitespace', 'false_ink', 'false_blanks', 'source_pixel_exact']:
                    counts[key][field] += row[field]
                counts[key]['source_mask_nll_sum'] += metrics['pattern_nll_mean'] * len(gold)
    complete_groups = []
    for key, value in sorted(counts.items()):
        value = dict(value)
        value['source_mask_nll_mean'] = value.pop('source_mask_nll_sum') / value['positions']
        complete_groups.append(dict(zip(['role', 'split', 'kind', 'layout'], key), **value))
    totals = {}
    for role in ['candidate', 'parent']:
        rows = [r for r in complete_groups if r['role'] == role]
        totals[role] = {k: sum(r[k] for r in rows) for k in
            ['positions', 'source_whitespace', 'false_ink', 'false_blanks', 'source_pixel_exact']}
        totals[role]['readable_nonblank_selected'] = sum(r['readable_nonblank'] for r in gate['groups'] if r['role'] == role)
        totals[role]['selected_positions'] = sum(r['count'] for r in gate['groups'] if r['role'] == role)
    assert totals['candidate']['positions'] == totals['parent']['positions'] == 12597
    assert totals['candidate']['source_whitespace'] == totals['parent']['source_whitespace'] == 252
    assert totals['candidate']['false_blanks'] == 37 and totals['parent']['false_blanks'] == 711
    assert totals['candidate']['false_ink'] == 25 and totals['parent']['false_ink'] == 11
    assert totals['candidate']['readable_nonblank_selected'] == 1002
    assert totals['parent']['readable_nonblank_selected'] == 981
    controller = load(RUN / 'controller_status.json')
    assert controller['pid'] is None and controller['phase'] == 'awaiting_opaque_physical_review'
    assert not (RUN / 'language').exists()
    assert not (ROOT / 'references/evidence/global_glyph_ranking_language_registration_v1.json').exists()
    journal = [json.loads(s) for s in (RUN / 'controller_journal.jsonl').read_text().splitlines()]
    ended = [r for r in journal if r['event'] == 'ended']
    assert [r['action'] for r in ended] == ['train', 'audit', 'qualify']
    assert all(r['returncode'] == 0 for r in ended)
    guard()
    result = dict(schema='global-glyph-ranking-physical-result-v1', complete=True,
        evidence_date='2026-11-15', physical_gate_passed=False,
        readable_groups_passed=True, credited_candidate_ink_passed=True, whitespace_passed=False,
        maximum_false_ink=11, language_quality_unmeasured=True, goal_achieved=False,
        new_neural_calls_from_exporter=0, new_optimizer_updates_from_exporter=0,
        registration_sha256=REGISTRATION_SHA, gate_sha256=GATE_SHA,
        fit_report_sha256=sha(RUN / 'fit/report.json'), audit_report_sha256=sha(RUN / 'audit/report.json'),
        qualification_report_sha256=sha(qualify / 'report.json'),
        control_qualification_sha256=sha(control_report_path), frozen_readings_sha256=FROZEN_SHA,
        review_session_sha256=sha(qualify / 'review/session.json'),
        freeze_receipt_sha256=sha(qualify / 'review/freeze_receipt.json'),
        exporter_sha256=sha(Path(__file__)), model_account=fit['model_account'],
        retained_updates=41298, extra_executed_replay_updates=2, source_count=13766,
        retained_target_exposure=6108741, snapshots=audit['snapshots'],
        physical_review=dict(boards=33, images=2080, reviewer_kind=session['reviewer_kind'],
            prior_development_familiarity=True, independent_human_adjudication=False,
            uncertain_readings=sum(r['uncertain'] for r in frozen['readings'].values()),
            pixel_empty_readings=sum(r['blank'] for r in frozen['readings'].values()),
            blank_flags_verified_against_original_pixels=True, credited_empty_candidate_ids=[],
            original_unfrozen_entry_history_preserved=True,
            pre_freeze_transcription_correction_sha256=sha(qualify / 'review_history/pre_freeze_alignment_correction_20261115.json'),
            one_freeze_before_owner_access=True, frozen_readings_repaired=False),
        readable_groups=gate['groups'], complete_groups=complete_groups, totals=totals,
        complete_case_counts=case_counts,
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
    print(json.dumps(dict(verified=args.verify, output=str(OUTPUT.relative_to(ROOT)),
        result_sha256=sha(OUTPUT), **cost)))


if __name__ == '__main__':
    main()
