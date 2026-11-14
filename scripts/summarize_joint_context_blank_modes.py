#!/usr/bin/env python3
"""Independently recount saved probability arrays; never reload a neural model."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
from collections import Counter

from scripts import analyze_joint_context_blank_modes as probe

ROOT = probe.ROOT
DEST = probe.DEST
OUT = ROOT / 'references/evidence/joint_context_blank_modes_result_20261114.json'


def summarize(guard):
    require = probe.require
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only recount required')
    registration = probe.validate(); report = probe.load(DEST / 'report.json'); end = probe.load(DEST / 'attempts/end.json')
    require(report['complete'] and end['complete'] and end['error'] is None
            and report['registration_sha256'] == probe.sha(probe.PUBLIC), 'Completed registered probe required')
    import numpy as np
    original = probe.load(probe.ORIGINAL / 'qualify/report.json')
    originals = {r['job']['id']: r for r in original['records']}
    jobs = {r['id']: r for r in registration['qualification_jobs']}
    require(len(report['cases']) == len(jobs) == 124 and len({r['case'] for r in report['cases']}) == 124,
            'Every bound case required once')
    groups = {}; files = {}; maximum_error = 0.; largest_deficit = 0.
    for row in report['cases']:
        guard(); p = ROOT / row['file']; require(probe.sha(p) == row['sha256'], 'Probability array changed')
        files[row['file']] = row['sha256']; job = jobs[row['case']]
        with np.load(p, allow_pickle=False) as data:
            values = {k: data[k] for k in data.files}
            n = row['positions']; require(all(v.shape == (n,) and np.isfinite(v).all() for v in values.values()), 'Complete finite probability vectors required')
            source_blank, emitted_blank = values['source_blank'], values['emitted_blank']
            require(source_blank.dtype == emitted_blank.dtype == np.bool_, 'Original typed pixel flags required')
            q = probe.ORIGINAL / 'qualify/cases' / job['id'] / 'pixels.npz'
            original_record = originals[job['id']]
            require(probe.sha(q) == original_record['pixels_sha256'], 'Original pixels changed')
            with np.load(q, allow_pickle=False) as pixels:
                require(np.array_equal(source_blank, pixels['gold_ink'].reshape(n, -1).sum(1) == 0)
                        and np.array_equal(emitted_blank, pixels['candidate_ink'].reshape(n, -1).sum(1) == 0), 'Saved blank flags differ from pixels')
            false = emitted_blank & ~source_blank
            better = values['source_logp'] >= values['blank_logp'] + registration['numerical_margin_nats']
            certified = values['nonempty_upper_bound'] < values['blank_logp'] - registration['numerical_margin_nats']
            require(not (false & better & certified).any(), 'Source contradicts certified mode')
            require((values['blank_argmax_disagreements'][emitted_blank] == 0).all(), 'Blank greedy path differs')
            require(np.allclose(values['blank_logp'][emitted_blank], values['emitted_logp'][emitted_blank], atol=1e-5, rtol=0), 'Same image probability differs')
            error = abs(float((-values['source_logp'] / 128).mean()) - original_record['pattern_nll_mean'])
            require(error == row['source_nll_mean_error'] and error <= registration['source_nll_mean_max_error'], 'Original source likelihood differs')
            maximum_error = max(maximum_error, error)
            largest_deficit = max(largest_deficit, float(values['saved_greedy_logp_deficit'].max()))
            key = '/'.join(job[k] for k in ['split', 'kind', 'layout']); group = groups.setdefault(key, Counter())
            group.update(positions=n, source_blank=int(source_blank.sum()), emitted_blank=int(emitted_blank.sum()),
                false_blank=int(false.sum()), source_preferred_false_blank=int((false & better).sum()),
                certified_blank_global_mode_false_blank=int((false & certified).sum()),
                saved_blank_argmax_disagreements=int(values['blank_argmax_disagreements'][emitted_blank].sum()),
                all_saved_path_argmax_disagreements=int(values['saved_argmax_disagreements'].sum()))
    require({k: dict(v) for k, v in groups.items()} == report['groups'], 'Independent group recount differs')
    decision = probe.decision(groups); require(decision == report['decision'], 'Fixed decision recount differs')
    require(maximum_error == report['maximum_source_nll_mean_error']
            and largest_deficit <= registration['maximum_saved_greedy_logp_deficit_nats'], 'Reproduction validity differs')
    require(sum(v['positions'] for v in groups.values()) == 12597
            and sum(v['false_blank'] for v in groups.values()) == 711, 'Complete failure cohort required')
    files.update({str((DEST / name).relative_to(ROOT)): probe.sha(DEST / name)
                  for name in ['registration.json', 'report.json', 'attempts/start.json', 'attempts/end.json']})
    return dict(schema='joint-context-blank-modes-completed-result-v1', export_source_sha256=probe.sha(Path(__file__)),
        registration_sha256=probe.sha(probe.PUBLIC), input_sha256=files, cases=124, positions=12597,
        groups={k: dict(v) for k, v in groups.items()}, decision=decision,
        false_blanks=711, source_preferred_false_blanks=sum(v['source_preferred_false_blank'] for v in groups.values()),
        certified_blank_global_modes=sum(v['certified_blank_global_mode_false_blank'] for v in groups.values()),
        maximum_source_nll_mean_error=maximum_error, maximum_saved_greedy_logp_deficit=largest_deficit,
        restored_inventory_sha256=__import__('hashlib').sha256(json.dumps(report['actual_restored_inventory'], sort_keys=True).encode()).hexdigest(),
        all131_actual_parameters_restored=report['all131_actual_parameters_restored'], model_account=report['model_account'],
        actual_context_recomputed=True, stale_predictor_cache_used=False,
        model_freezing_is_execution_assertion_not_second_neural_replay=True,
        original_physical_gate_passed=False, original_readings_and_scores_changed=False,
        source_frozen_review_blank_flag_inconsistencies=28, source_frozen_review_credited_empty_images=6,
        probe_resources={k: end[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
        generated_glyphs=0, new_optimizer_updates=0, new_autonomous_streams=0, recount_neural_calls=0,
        full_language_quality='unmeasured; physical prerequisite failed', goal_achieved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    began = time.perf_counter(); minimum = float('inf')
    def guard():
        nonlocal minimum
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(mem['MemAvailable'].split()[0]) / 1024**2; minimum = min(minimum, available)
        probe.require(available >= 24 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2
                      and time.perf_counter() - began <= 900, 'CPU recount resource ceiling reached')
    guard(); value = summarize(guard)
    cost = dict(elapsed_seconds=time.perf_counter() - began, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                minimum_available_host_gib=minimum)
    if args.verify:
        old = probe.load(OUT); old.pop('recount_resources'); probe.require(old == value, 'Export differs')
    else:
        probe.write_new(OUT, dict(value, recount_resources=cost))
    print(json.dumps(dict(verified=args.verify, output=str(OUT.relative_to(ROOT)), decision=value['decision'],
        source_preferred=value['source_preferred_false_blanks'], certified_blank_global_modes=value['certified_blank_global_modes'],
        recount_resources=cost)))


if __name__ == '__main__':
    main()
