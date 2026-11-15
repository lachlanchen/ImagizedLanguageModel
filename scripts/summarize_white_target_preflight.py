#!/usr/bin/env python3
"""Authenticate the six-case white-target preflight without a neural replay."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import time

from scripts import preflight_white_target_ranking as preflight

OUT = preflight.ROOT / 'references/evidence/white_target_preflight_result_20261115.json'


def summarize(guard):
    require, load, sha = preflight.require, preflight.load, preflight.sha
    r = preflight.validate(); folder = preflight.RUN / 'execution'
    report, end, outer = [load(p) for p in [folder / 'report.json', folder / 'end.json', preflight.RUN / 'outer_end.json']]
    require(report['complete'] and end['complete'] and end['error'] is None and outer['returncode'] == 0
            and report['registration_sha256'] == sha(preflight.PUBLIC), 'Completed single bound preflight required')
    require(report['actual_trainable_tensors'] == 131 and report['actual_optimizer_updates'] == 4
            and report['all_updates_reset'], 'Every actual learned tensor/update/reset required')
    require(len(report['cases']) == len(r['cases']) == 6, 'All six complete sources required')
    cases = []; bindings = {name: sha(preflight.RUN / name) for name in
        ['registration.json', 'execution/report.json', 'execution/end.json', 'outer_start.json', 'outer_end.json']}
    for expected, row in zip(r['cases'], report['cases']):
        guard()
        require(row['case'] == expected['id'] and row['target_glyphs'] == expected['source']['target_glyphs']
            and row['white_targets'] == expected['expected_white_targets']
            and row['nonempty_targets'] + row['white_targets'] == row['target_glyphs']
            and row['frozen_parameters_and_affines_unchanged'] and not row['updates_carried'], 'Original full source/reset differs')
        name = 'execution/' + row['case'] + '.json'; bindings[name] = sha(preflight.RUN / name)
        require(load(preflight.RUN / name) == row, 'Individual case report differs')
        item = {k: row[k] for k in ['case', 'target_glyphs', 'nonempty_targets', 'white_targets',
            'image_loss', 'ranking_loss', 'white_ranking_loss', 'stop_loss', 'loss',
            'state_gradient_norm', 'image_microbatches', 'optimizer_updates']}
        require(all(math.isfinite(item[k]) for k in ['image_loss', 'ranking_loss', 'white_ranking_loss',
            'stop_loss', 'loss', 'state_gradient_norm']), 'Finite case metrics required')
        require((item['white_ranking_loss'] > 0) == bool(item['white_targets']), 'New term must act on actual white targets only')
        if row['case'].startswith('qa_'):
            errors = row['gradient_comparison']; require(set(errors) == set(r['initial_trainable_inventory']), 'All actual QA gradients required')
            for name, x in errors.items():
                require(x['reference_norm'] > 0 and math.isfinite(x['reference_norm'])
                    and (x['relative_l2'] <= r['gradient_relative_l2_tolerance'] or x['maximum_absolute'] <= r['gradient_absolute_tolerance']), 'Actual QA gradient L2 differs')
                require(math.isfinite(x['maximum_absolute']) and math.isfinite(x['relative_l2'])
                    and math.isfinite(x['reference_maximum']) and x['reference_maximum'] > 0
                    and x['maximum_absolute'] <= r['gradient_absolute_tolerance']
                        + r['gradient_maximum_scaled_tolerance'] * x['reference_maximum'], 'Actual QA maximum gradient differs')
            require(abs(row['reference_loss'] - row['loss']) <= r['loss_absolute_tolerance'] and row['optimizer_updates'] == 0, 'Complete QA loss/update differs')
            item.update(actual_gradient_tensors=131, maximum_relative_l2=max(x['relative_l2'] for x in errors.values()),
                maximum_absolute=max(x['maximum_absolute'] for x in errors.values()), reference_loss=row['reference_loss'])
        else:
            norms, changed = row['gradient_norms'], row['changed_elements']
            require(set(norms) == set(changed) == set(r['initial_trainable_inventory'])
                and all(math.isfinite(v) and v > 0 for v in norms.values())
                and all(type(v) is int and v > 0 for v in changed.values()) and row['optimizer_updates'] == 1,
                'All four actual complete-book updates required')
            item.update(actual_gradient_tensors=131, actual_updated_tensors=131)
        cases.append(item)
    require(sum(c['optimizer_updates'] for c in cases) == 4
            and sum(c['white_targets'] > 0 for c in cases) == 2, 'Four reset steps and two real mixed-white layouts required')
    require(end['elapsed_seconds'] <= r['maximum_seconds'] and end['resource_wait_seconds'] <= r['pressure_seconds']
            and end['peak_allocated_vram_gib'] <= r['maximum_allocated_vram_gib']
            and end['minimum_available_gib'] >= r['minimum_runtime_host_gib'], 'Registered resource ceilings required')
    return dict(schema='white-target-ranking-real-source-preflight-result-v1', complete=True,
        registration_sha256=sha(preflight.PUBLIC), export_source_sha256=sha(Path(__file__)), input_sha256=bindings,
        cases=cases, mixed_white_selection=r['mixed_white_selection'], model_account=report['model_account'],
        all_actual131gradients_checked=True, actual_optimizer_updates=4, all_updates_reset=True,
        updates_carried_into_fit=False,
        frozen_state_counters_and_numerics_are_bound_execution_checks_not_independent_neural_replay=True,
        execution_resources={k: end[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
        outer_elapsed_seconds=outer['elapsed_seconds'], preserved_prior_sources=len(load(preflight.PROBE)['source_sha256']),
        new_full_fit_executed=False, new_autonomous_images=0, language_quality='unmeasured',
        artifact_validation_neural_calls=0, goal_achieved=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--verify', action='store_true'); args = p.parse_args()
    preflight.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only artifact validation required')
    began = time.perf_counter()
    def guard():
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        preflight.require(int(memory['MemAvailable'].split()[0]) >= 24 * 1024**2
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2
            and time.perf_counter() - began <= 900, 'CPU artifact validation resource ceiling reached')
    guard(); value = summarize(guard)
    if args.verify: preflight.require(preflight.load(OUT) == value, 'Completed preflight export differs')
    else: preflight.write_new(OUT, value)
    print(json.dumps(dict(verified=args.verify, output=str(OUT.relative_to(preflight.ROOT)), result_sha256=preflight.sha(OUT),
        complete=True, actual_optimizer_updates=4, all_updates_reset=True,
        artifact_validation_seconds=time.perf_counter() - began)))


if __name__ == '__main__': main()
