#!/usr/bin/env python3
"""Export the completed context-gradient comparison without neural execution.

Frozen review scores remain immutable. Pixel inventories and paired saved losses
are post-hoc diagnostics, not replacement readability or language scores.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/joint_context_learning_v1'
CONTROL = ROOT / 'artifacts/continuation_joint_writer_v1'
REGISTRATION = ROOT / 'references/evidence/joint_context_learning_registration_20261114.json'
EXPECTED = 'd168b5f24ad2b8dbcf607ab1f0ab06728fe914b1e186dffea93a31535e99c833'
OUT = ROOT / 'references/evidence/joint_context_physical_result_20261114.json'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(guard):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only export required')
    require(sha(REGISTRATION) == sha(RUN / 'registration.json') == EXPECTED, 'Registration changed')
    registration = load(REGISTRATION)
    for name, digest in registration['source_sha256'].items():
        guard(); require(sha(ROOT / name) == digest, 'Registered source changed: ' + name)
    controller = load(RUN / 'controller_status.json')
    require(controller['pid'] is None and controller['phase'] == 'awaiting_opaque_physical_review',
            'Original serial pipeline must finish first')
    fit, audit, qualify = [load(RUN / folder / 'report.json') for folder in ['fit', 'audit', 'qualify']]
    require(all(x['complete'] and x['registration_sha256'] == EXPECTED for x in [fit, audit, qualify]),
            'Complete bound reports required')
    require(audit['fit_report_sha256'] == sha(RUN / 'fit/report.json'), 'Audited fit changed')
    require(fit['retained_updates'] == audit['retained_updates'] == 41298
            and fit['target_exposure'] == audit['retained_target_exposure'] == 6108741,
            'Complete original exposure required')
    snapshots = load(RUN / 'fit/snapshots.json')
    require([int(k) for k in snapshots] == [0, 10324, 20649, 30973, 41298], 'Keep every fixed snapshot')
    for row in snapshots.values():
        guard(); require(sha(ROOT / row['file']) == row['sha256'], 'Saved snapshot changed')

    from scripts.review_joint_writer_physical import aggregate, read_frozen_bundle, verify_pixels
    from scripts.evaluate_joint_context_language import review_scope, ink_audit
    with review_scope() as review:
        (q, reg, report, boards, ids, bindings), session = review.session_context()
        frozen = read_frozen_bundle(q, boards, ids, bindings)
        mapping = load(q / 'mapping.json')['rows']
        groups, passed = aggregate(reg['qualification_jobs'], mapping, frozen['readings'], EXPECTED, .9)
        gate = load(q / 'gate.json')
        require(groups == gate['groups'] and passed == gate['passed'] is False, 'Preserve original failed gate')
        verify_pixels(q, report, boards, mapping, frozen, guard)
        ink = ink_audit(q, mapping, frozen, bindings, guard)
    saved_ink = load(q / 'failed_gate_ink_audit.json')
    require(all(saved_ink[k] == v for k, v in ink.items()), 'Independent ink audit differs')
    require(len(ids) == len(frozen['readings']) == 2080 and len(boards['boards']) == 33,
            'Complete opaque cohort required')

    import numpy as np
    old_records = {x['job']['id']: x for x in load(CONTROL / 'qualify/report.json')['records']}
    pixel_groups = defaultdict(lambda: defaultdict(float))
    require(qualify['all124_complete'] and len(qualify['records']) == 124, 'All qualification cases required')
    for record in qualify['records']:
        guard(); job = record['job']; old = old_records[job['id']]
        require(job == old['job'], 'Control cohort changed')
        p = q / 'cases' / job['id'] / 'pixels.npz'
        old_p = CONTROL / 'qualify/cases' / job['id'] / 'pixels.npz'
        require(sha(p) == record['pixels_sha256'] and sha(old_p) == old['pixels_sha256'], 'Case pixels changed')
        with np.load(p, allow_pickle=False) as arrays, np.load(old_p, allow_pickle=False) as original:
            gold = arrays['gold_ink']; candidate = arrays['candidate_ink']; parent = arrays['parent_ink']
            require(np.array_equal(gold, original['gold_ink'])
                    and np.array_equal(parent, original['candidate_ink']), 'Unchanged control pixels required')
            require(gold.shape == candidate.shape == parent.shape == (record['target_glyphs'], 1, 32, 32),
                    'Complete physical image stream required')
            for array in [gold, candidate, parent]:
                require(np.all((array == 0) | (array == 1)), 'Binary original pixels required')
            sums = [x.reshape(len(x), -1).sum(1) for x in [gold, candidate, parent]]
            group = pixel_groups[(job['split'], job['kind'], job['layout'])]
            group['cases'] += 1; group['positions'] += len(gold); group['source_blank'] += int((sums[0] == 0).sum())
            group['candidate_source_pattern_nll_sum'] += record['pattern_nll_mean'] * len(gold)
            group['control_source_pattern_nll_sum'] += old['pattern_nll_mean'] * len(gold)
            for role, array, count in [('candidate', candidate, sums[1]), ('control', parent, sums[2])]:
                group[role + '_blank'] += int((count == 0).sum())
                group[role + '_false_blank'] += int(((count == 0) & (sums[0] > 0)).sum())
                group[role + '_canonical_pixel_exact'] += int((array == gold).reshape(len(gold), -1).all(1).sum())
                selected = np.asarray(job['positions'], dtype=np.int64)
                group[role + '_selected_blank'] += int((count[selected] == 0).sum())
    pixel_rows = []
    for key, group in sorted(pixel_groups.items()):
        row = dict(zip(['split', 'kind', 'layout'], key)); row.update(group)
        for role in ['candidate', 'control']:
            row[role + '_source_pattern_nll_mean'] = row.pop(role + '_source_pattern_nll_sum') / group['positions']
        pixel_rows.append(row)
    require(sum(x['positions'] for x in pixel_rows) == 12597, 'All complete physical positions required')

    def ledger(path):
        raw = path.read_bytes(); require(raw.endswith(b'\n'), 'Incomplete ledger line')
        return [v for line in raw.splitlines() if (v := json.loads(line)).get('event_kind') == 'scheduled']
    current = ledger(RUN / 'fit/ledger_001.jsonl'); control = ledger(CONTROL / 'fit/ledger_001.jsonl')
    require(len(current) == len(control) == 41298, 'Complete matched ledgers required')
    loss_groups = defaultdict(list)
    for new, old in zip(current, control):
        guard()
        require(all(new[k] == old[k] for k in ['step', 'source_index', 'source_id', 'layout'])
                and new['target_glyphs'] == old['glyphs'], 'Paired complete-source schedule differs')
        require(math.isfinite(new['image_loss']) and math.isfinite(old['nll']), 'Finite paired NLL required')
        epoch = (new['step'] - 1) // 13766 + 1
        loss_groups[(epoch, new['kind'], new['layout'])].append((new['image_loss'], old['nll'], new['target_glyphs']))
    require(sum(x['target_glyphs'] for x in current) == 6108741, 'Keep all retained exposures')
    loss_rows = []
    for key, values in sorted(loss_groups.items()):
        n = len(values); positions = sum(v[2] for v in values)
        loss_rows.append(dict(epoch=key[0], kind=key[1], layout=key[2], sources=n, target_exposure=positions,
            candidate_source_mean_nll=statistics.fmean(v[0] for v in values),
            control_source_mean_nll=statistics.fmean(v[1] for v in values),
            candidate_glyph_weighted_nll=sum(v[0] * v[2] for v in values) / positions,
            control_glyph_weighted_nll=sum(v[1] * v[2] for v in values) / positions,
            candidate_lower_loss_sources=sum(v[0] < v[1] for v in values),
            source_mean_nll_delta=statistics.fmean(v[0] - v[1] for v in values)))
    costs = defaultdict(float)
    for event in (json.loads(line) for line in (RUN / 'controller_journal.jsonl').read_text().splitlines()):
        if event['event'] == 'ended':
            require(event['returncode'] == 0, 'Preserve any failed attempt separately')
            costs[event['action']] += event['elapsed_seconds']
    files = ['registration.json', 'controller_status.json', 'controller_journal.jsonl', 'fit/report.json',
        'fit/snapshots.json', 'fit/ledger_001.jsonl', 'audit/report.json', 'qualify/report.json', 'qualify/boards.json',
        'qualify/gate.json', 'qualify/frozen_readings.json', 'qualify/review/freeze_receipt.json',
        'qualify/mapping.json', 'qualify/failed_gate_ink_audit.json']
    return dict(schema='joint-context-completed-physical-result-v1', export_source_sha256=sha(Path(__file__)),
        registration_sha256=EXPECTED, input_sha256={name: sha(RUN / name) for name in files},
        control_input_sha256={name: sha(CONTROL / name) for name in ['fit/ledger_001.jsonl', 'qualify/report.json']},
        training={k: fit[k] for k in ['retained_updates', 'extra_executed_resume_updates', 'training_cases', 'epochs',
            'target_exposure', 'model_account', 'snapshots', 'elapsed_seconds', 'resource_wait_seconds',
            'peak_allocated_vram_gib', 'minimum_available_gib']},
        audit={k: audit[k] for k in ['complete', 'retained_updates', 'executed_events_all_attempts',
            'retained_target_exposure', 'actual_saved_replay_weights_optimizer_rng_exact', 'elapsed_seconds', 'peak_rss_gib']},
        qualification_cases=124, reviewed_boards=33, reviewed_images=2080, groups=groups,
        physical_gate_passed=False, independent_ink_audit=ink,
        qualification_resources={k: qualify[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib']},
        cumulative_stage_child_seconds=dict(costs), complete_supplied_history_pixel_diagnostics=pixel_rows,
        paired_online_training_likelihood=loss_rows,
        diagnostics_are_post_hoc_not_regrading=True, original_readings_and_scores_changed=False,
        review=dict(reviewer=session['reviewer'], reviewer_kind=session['reviewer_kind'],
            prior_development_familiarity=True, independent_human_adjudication=False,
            frozen_readings_sha256=sha(q / 'frozen_readings.json'), readings_frozen_before_owner_unmask=True),
        autonomous_language_quality='unmeasured; physical prerequisite failed', new_autonomous_streams=0,
        new_model_calls=0, new_optimizer_updates=0, goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); began = time.perf_counter(); minimum = float('inf')
    def guard():
        nonlocal minimum
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0]) / 1024**2; minimum = min(minimum, available)
        require(available >= 24 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2
                and time.perf_counter() - began <= 900, 'CPU analysis resource ceiling reached')
    guard(); result = summarize(guard)
    resources = dict(elapsed_seconds=time.perf_counter() - began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, minimum_available_gib=minimum)
    if args.verify:
        saved = load(OUT); saved.pop('analysis_resources'); require(saved == result, 'Completed export differs')
    else:
        from scripts.review_joint_writer_physical import write_new
        write_new(OUT, dict(result, analysis_resources=resources))
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), verified=args.verify,
        physical_gate_passed=False, independent_ink_candidate_credited_empty=result['independent_ink_audit']['candidate_credited_empty'],
        pixel_diagnostics=result['complete_supplied_history_pixel_diagnostics'],
        paired_online_training_likelihood=result['paired_online_training_likelihood'], analysis_resources=resources)))


if __name__ == '__main__':
    main()
