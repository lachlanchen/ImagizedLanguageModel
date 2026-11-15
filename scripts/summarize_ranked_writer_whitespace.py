#!/usr/bin/env python3
"""CPU recount of the ranked writer's new whitespace-density diagnostic."""
import argparse
from collections import Counter
import json
import os
import resource
import time

from scripts import analyze_ranked_writer_whitespace as probe


def build(guard):
    import numpy as np
    p = probe; require = p.require
    reg = p.validate(); report = p.load(p.DEST / 'report.json'); end = p.load(p.DEST / 'attempts/end.json')
    require(report['complete'] and end['complete'] and end['error'] is None
            and report['registration_sha256'] == p.sha(p.PUBLIC), 'Completed single diagnostic required')
    mapping = p.load(p.ORIGINAL / 'qualify/mapping.json')['rows']
    readings = p.load(p.ORIGINAL / 'qualify/frozen_readings.json')['readings']
    originals = {r['job']['id']: r for r in p.load(p.ORIGINAL / 'qualify/report.json')['records']}
    jobs = {r['id']: r for r in reg['qualification_jobs']}
    require(len(report['cases']) == len(jobs) == 124 and len({r['case'] for r in report['cases']}) == 124,
            'All original cases required exactly once')
    groups, files, maximum_error, disagreements = {}, {}, 0., 0
    for row in report['cases']:
        guard(); path = p.ROOT / row['file']; require(p.sha(path) == row['sha256'], 'Probability vector changed')
        files[row['file']] = row['sha256']; job = jobs[row['case']]; n = row['positions']
        with np.load(path, allow_pickle=False) as arrays:
            v = {k: arrays[k] for k in arrays.files}
        require(all(a.shape == (n,) and np.isfinite(a).all() for a in v.values()), 'Complete finite vectors required')
        require(not v['saved_argmax_disagreements'].any(), 'Saved greedy choices disagree')
        disagreements += int(v['saved_argmax_disagreements'].sum())
        require(v['source_blank'].dtype == v['emitted_blank'].dtype == np.bool_, 'Typed pixel flags required')
        original = p.ORIGINAL / 'qualify/cases' / job['id'] / 'pixels.npz'
        require(p.sha(original) == originals[job['id']]['pixels_sha256'], 'Original qualification changed')
        with np.load(original, allow_pickle=False) as arrays:
            for key, field in [('source_blank', 'gold_ink'), ('emitted_blank', 'candidate_ink')]:
                require(np.array_equal(v[key], ~arrays[field].reshape(n, -1).any(1)), 'Pixel flags differ')
        error = abs(float((-v['source_logp'] / 128).mean()) - originals[job['id']]['pattern_nll_mean'])
        require(error == row['source_nll_mean_error'] and error <= reg['source_nll_mean_max_error'], 'Original source likelihood differs')
        maximum_error = max(maximum_error, error)
        empty = v['emitted_blank']
        require(np.allclose(v['blank_logp'][empty], v['emitted_logp'][empty], atol=1e-5, rtol=0), 'Identical blank paths differ')
        selected, credited = p.selected_flags(job, n, mapping, readings)
        count = p.case_counts(v['source_blank'], empty, v['blank_logp'], v['emitted_logp'], selected, credited)
        certified = v['blank_logp'] > v['nonempty_upper_bound'] + .001
        count.update(certified_blank_global_mode=int(certified.sum()),
            certified_blank_global_mode_source_white=int((certified & v['source_blank']).sum()),
            certified_blank_global_mode_false_ink=int((certified & v['source_blank'] & ~empty).sum()))
        key = '/'.join(job[k] for k in ['split', 'kind', 'layout']); groups.setdefault(key, Counter()).update(count)
    groups = {k: dict(v) for k, v in groups.items()}; decision = p.decision(groups)
    require(report['groups'] == groups and report['decision'] == decision
            and maximum_error == report['maximum_source_nll_mean_error'], 'Independent diagnostic recount differs')
    require(report['all131_actual_parameters_restored'] and report['actual_context_recomputed']
            and report['model_tensors_unchanged'] and not report['stale_predictor_cache_used'], 'Actual frozen actor required')
    require(report['generated_glyphs'] == report['new_optimizer_updates'] == report['new_autonomous_streams'] == 0
            and not report['original_gate_changed'], 'No generation, optimization or repaired gate allowed')
    for name in ['registration.json', 'report.json', 'attempts/start.json', 'attempts/end.json']:
        path = p.DEST / name; files[str(path.relative_to(p.ROOT))] = p.sha(path)
    return dict(schema='ranked-writer-whitespace-density-result-v1', complete=True,
        exporter_sha256=p.sha(__file__), registration_sha256=p.sha(p.PUBLIC), input_sha256=files,
        cases=124, positions=12597, groups=groups, decision=decision,
        all131_actual_parameters_restored=True, actual_context_recomputed=True,
        restored_inventory=report['actual_restored_inventory'], model_account=report['model_account'],
        maximum_source_nll_mean_error=maximum_error, saved_argmax_disagreements=disagreements,
        model_freezing_is_execution_assertion_not_second_neural_replay=True,
        physical_gate_passed=False, original_gate_and_readings_changed=False,
        generated_glyphs=0, optimizer_updates=0, autonomous_answers=0, recount_neural_calls=0,
        counterfactual_only=True, language_quality_unmeasured=True, goal_achieved=False,
        resources={k: end[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']})


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    probe.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only recount required')
    began = time.perf_counter()
    def guard():
        memory = dict(line.split(':', 1) for line in probe.Path('/proc/meminfo').read_text().splitlines())
        probe.require(int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2
            and time.perf_counter() - began <= 900, 'CPU recount resource ceiling reached')
    guard(); value = build(guard)
    if args.verify:
        probe.require(probe.load(probe.EXPORT) == value, 'Original deterministic export differs')
    else:
        probe.write_new(probe.EXPORT, value)
    print(json.dumps(dict(verified=args.verify, result_sha256=probe.sha(probe.EXPORT), decision=value['decision'],
        elapsed_seconds=time.perf_counter() - began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2)))


if __name__ == '__main__':
    main()
