#!/usr/bin/env python3
"""Bind the complete matched white-target comparison before any GPU execution."""
import argparse
import os
import shutil
import subprocess

from ilm.visual_lm.white_target_training import (
    ROOT, RUN, PUBLIC, CONTROL, WRITER, PREFLIGHT, PARENT,
    expected_initial, tensor_inventory, external_runtime, load, sha, require)
from scripts.preflight_visual_continuation import runtime
from scripts.review_joint_writer_physical import write_new

NEW = (
    'ilm/visual_lm/white_target_training.py', 'scripts/prepare_white_target_training.py',
    'scripts/train_white_target_comparison.py', 'scripts/audit_white_target_training.py',
    'scripts/run_white_target_training.py', 'scripts/review_white_target_physical.py',
    'tests/test_white_target_training_contract.py', 'references/white_target_training_protocol.md',
)


def sources():
    result = dict(load(PREFLIGHT / 'registration.json')['source_sha256'])
    for name in NEW:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit comparison source before binding: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Preserve bound source: ' + name)
    return result


def register():
    import numpy as np
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only registration')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve existing trial'); torch.set_num_threads(2)
    pre = load(PREFLIGHT / 'execution/report.json')
    require(pre['complete'] and len(pre['cases']) == 6 and pre['all_updates_reset']
            and pre['actual_optimizer_updates'] == 4 and pre['actual_trainable_tensors'] == 131,
            'Completed reset real-whitespace preflight required')
    old = load(CONTROL / 'registration.json'); value = old.copy()
    bindings = dict(old['bindings_sha256']); bindings.update(load(PREFLIGHT / 'registration.json')['input_sha256'])
    def bind(path):
        name, digest = str(path.relative_to(ROOT)), sha(path)
        require(name not in bindings or bindings[name] == digest, 'Conflicting preserved input: ' + name)
        bindings[name] = digest
    for path in [CONTROL / 'registration.json', CONTROL / 'fit/report.json', CONTROL / 'audit/report.json',
                 CONTROL / 'qualify/report.json', CONTROL / 'qualify/gate.json', CONTROL / 'qualify/frozen_readings.json',
                 CONTROL / 'qualify/review/session.json', CONTROL / 'qualify/review/freeze_receipt.json',
                 CONTROL / 'qualify/boards.json', CONTROL / 'qualify/mapping.json',
                 WRITER / 'initial_parameters.pt', WRITER / 'cache/normalization.pt', WRITER / 'schedule.npy',
                 PREFLIGHT / 'registration.json', PREFLIGHT / 'execution/report.json', PREFLIGHT / 'execution/end.json',
                 PREFLIGHT / 'outer_start.json', PREFLIGHT / 'outer_end.json',
                 ROOT / 'references/evidence/white_target_ranking_preflight_registration_20261115.json',
                 ROOT / 'references/evidence/white_target_preflight_result_20261115.json',
                 ROOT / 'references/evidence/global_glyph_ranking_physical_result_20261115.json',
                 ROOT / 'references/evidence/ranked_writer_whitespace_result_20261115.json',
                 CONTROL / 'whitespace_density/report.json', CONTROL / 'whitespace_density/attempts/end.json']:
        bind(path)
    for job in old['qualification_jobs']:
        bind(ROOT / job['image']); bind(CONTROL / 'qualify/cases' / job['id'] / 'pixels.npz')
    rows = load(PARENT / 'training_manifest.json')['rows']; schedule = np.load(WRITER / 'schedule.npy', allow_pickle=False)
    require(schedule.shape == (41298, 2) and len(rows) == 13766, 'Complete original schedule required')
    for epoch in schedule.reshape(3, 13766, 2):
        require(np.array_equal(np.sort(epoch[:, 0]), np.arange(13766)), 'Each source exactly once per epoch')
    require(np.array_equal(np.bincount(schedule[schedule[:, 1] == 1, 0], minlength=13766), np.full(13766, 2)),
            'One fixed and two proportional presentations required')
    require(sum(rows[int(i)]['target_glyphs'] for i in schedule[:, 0]) == 6108741, 'Complete target exposure required')
    for name, digest in bindings.items(): require(sha(ROOT / name) == digest, 'Bound input changed: ' + name)
    require(shutil.disk_usage(ROOT).free >= 48 * 1024**3, 'Retain48GiB free disk')
    value.update(schema='complete-white-target-ranking-v1', source_sha256=sources(), bindings_sha256=bindings,
        runtime=runtime(), schedule_file=str((WRITER / 'schedule.npy').relative_to(ROOT)),
        external_program_sha256=external_runtime(), initial_trainable_inventory=tensor_inventory(expected_initial()),
        trainable_parameters=5005760, trainable_tensors=131, model_account=pre['model_account'],
        objective='complete physical image likelihood, nonempty source-over-blank ranking, white-target first-nonzero-bound ranking and frozen-stop loss',
        ranking_strength=1., white_strength=1., margin_nats=1., stop_loss_weight=.1,
        context_recomputed_each_update=True, predictor_state_cache_used_for_training=False,
        body_prompt_feedback_stop_and_affines_frozen=True, continuation_query_and_adapters_trainable=True,
        initial_head_updates=0, preflight_updates_carried=False,
        fit_seconds=108000, fit_pressure_seconds=7200, qualify_seconds=3600, qualify_pressure_seconds=300,
        minimum_entry_host_gib=48, minimum_runtime_host_gib=24, maximum_allocated_vram_gib=22,
        audit_seconds=900, audit_peak_rss_gib=8, cpu_threads=2, automatic_retry=False,
        unchanged_control='Completed nonempty-ranking actor; identical fresh head, point-final context, normalization, full schedule and decoder',
        control_physical_pixels=str((CONTROL / 'qualify/cases').relative_to(ROOT)),
        original_writer_initialization=str(WRITER.relative_to(ROOT)),
        physical_review_workflow='scripts/review_white_target_physical.py',
        physical_blank_flags_from_opaque_pixels_before_freeze=True, physical_credited_ink_required=True,
        source_whitespace_positions=252, maximum_false_ink_on_source_whitespace=11, maximum_false_blanks=37,
        physical_conditions_are_prospective_and_do_not_repair_old_reviews=True,
        autonomous_evaluation_protocol='Unchanged708stream plan and original pilot/retention/dependence/long-book rules. Restore all131 actual learned tensors. Separate committed post-gate execution binding after all7 readable groups, credited ink, <=11falseink/252white and <=37falseblanks.',
        language_quality_unmeasured=True, goal_achieved=False)
    RUN.mkdir(); write_new(RUN / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), sources=len(value['source_sha256']), bound_inputs=len(bindings),
        training_sources=13766, updates=41298, target_exposure=6108741)


def validate():
    r = load(PUBLIC)
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact registration before execution')
    require(r['source_sha256'] == sources() and r['runtime'] == runtime(), 'Source/runtime differs')
    require(r['schema'] == 'complete-white-target-ranking-v1'
        and r['ranking_strength'] == r['white_strength'] == r['margin_nats'] == 1., 'Fixed white-target objective required')
    require(r['source_whitespace_positions'] == 252 and r['maximum_false_ink_on_source_whitespace'] == 11
            and r['maximum_false_blanks'] == 37, 'Preserve physical safeguards')
    for name, digest in r['bindings_sha256'].items(): require(sha(ROOT / name) == digest, 'Bound input/control changed: ' + name)
    require(sha(ROOT / r['schedule_file']) == r['schedule_sha256'], 'Original complete schedule changed')
    return r


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['register', 'validate']); args = p.parse_args()
    print(json.dumps(register() if args.action == 'register' else dict(valid=bool(validate()))))
