#!/usr/bin/env python3
"""Authenticate the completed real-source ranking preflight without replaying it."""
import argparse
import json
import math
import os
from pathlib import Path
import time

from scripts import preflight_global_glyph_ranking as preflight

OUT = preflight.ROOT / 'references/evidence/global_glyph_ranking_preflight_result_20261114.json'


def summarize():
    require, load, sha = preflight.require, preflight.load, preflight.sha
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only artifact validation required')
    r = preflight.validate(); folder = preflight.RUN / 'execution'
    report = load(folder / 'report.json'); end = load(folder / 'end.json')
    require(report['complete'] and end['complete'] and end['error'] is None
            and report['registration_sha256'] == sha(preflight.PUBLIC), 'Completed bound preflight required')
    require(report['actual_trainable_tensors'] == 131 and report['actual_optimizer_updates'] == 2
            and report['all_updates_reset'], 'Every actual learned tensor/update/reset required')
    require(len(report['cases']) == len(r['cases']) == 4, 'All four complete sources required')
    cases = []; bindings = {name: sha(preflight.RUN / name) for name in ['registration.json', 'execution/report.json', 'execution/end.json', 'launch.json']}
    for expected, row in zip(r['cases'], report['cases']):
        require(row['case'] == expected['id'] and row['target_glyphs'] == expected['source']['target_glyphs']
                and row['frozen_parameters_and_affines_unchanged'] and not row['updates_carried'], 'Original complete source/reset differs')
        name = 'execution/' + row['case'] + '.json'; bindings[name] = sha(preflight.RUN / name)
        require(load(preflight.RUN / name) == row, 'Individual case report differs')
        item = {k: row[k] for k in ['case', 'target_glyphs', 'nonempty_targets', 'image_loss', 'ranking_loss',
            'stop_loss', 'loss', 'state_gradient_norm', 'image_microbatches', 'optimizer_updates']}
        require(all(math.isfinite(item[k]) for k in ['image_loss', 'ranking_loss', 'stop_loss', 'loss', 'state_gradient_norm']), 'Finite case metrics required')
        if row['case'].startswith('qa_'):
            errors = row['gradient_comparison']; require(set(errors) == set(r['initial_trainable_inventory']), 'All actual QA gradients required')
            for x in errors.values():
                require(x['reference_norm'] > 0 and math.isfinite(x['reference_norm'])
                        and (x['relative_l2'] <= r['gradient_relative_l2_tolerance'] or x['maximum_absolute'] <= r['gradient_absolute_tolerance']), 'Actual QA gradient tolerance differs')
            require(abs(row['reference_loss'] - row['loss']) <= r['loss_absolute_tolerance'] and row['optimizer_updates'] == 0, 'Complete QA loss/update differs')
            item.update(actual_gradient_tensors=131, maximum_relative_l2=max(x['relative_l2'] for x in errors.values()),
                maximum_absolute=max(x['maximum_absolute'] for x in errors.values()), reference_loss=row['reference_loss'])
        else:
            norms, changed = row['gradient_norms'], row['changed_elements']
            require(set(norms) == set(changed) == set(r['initial_trainable_inventory'])
                    and all(math.isfinite(v) and v > 0 for v in norms.values())
                    and all(type(v) is int and v > 0 for v in changed.values()) and row['optimizer_updates'] == 1,
                    'Both actual complete-book updates required')
            item.update(actual_gradient_tensors=131, actual_updated_tensors=131)
        cases.append(item)
    return dict(schema='global-glyph-ranking-real-source-preflight-result-v1',
        registration_sha256=sha(preflight.PUBLIC), export_source_sha256=sha(Path(__file__)), input_sha256=bindings,
        cases=cases, model_account=report['model_account'], all_actual131gradients_checked=True,
        actual_optimizer_updates=2, all_updates_reset=True, updates_carried_into_fit=False,
        frozen_state_and_numerics_are_bound_execution_checks_not_independent_neural_replay=True,
        execution_resources={k: end[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
        original64sources_unchanged=True, new_full_fit_executed=False, new_autonomous_images=0,
        language_quality='unmeasured', artifact_validation_neural_calls=0, goal_achieved=False)


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--verify', action='store_true');args = p.parse_args();began = time.perf_counter()
    value = summarize()
    if args.verify:preflight.require(preflight.load(OUT) == value, 'Completed preflight export differs')
    else:preflight.write_new(OUT, value)
    print(json.dumps(dict(verified=args.verify, output=str(OUT.relative_to(preflight.ROOT)), complete=True,
        actual_optimizer_updates=2, all_updates_reset=True, artifact_validation_seconds=time.perf_counter() - began)))


if __name__ == '__main__':main()
