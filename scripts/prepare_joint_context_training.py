#!/usr/bin/env python3
"""Register the complete context-gradient intervention, without neural execution."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

from ilm.visual_lm.joint_context_training import (ROOT, RUN, PUBLIC, CONTROL, PREFLIGHT, PARENT,
    expected_initial, tensor_inventory, external_runtime, load, sha, require)
from scripts.preflight_visual_continuation import runtime
from scripts.review_joint_writer_physical import write_new

NEW = ('ilm/visual_lm/joint_context_training.py', 'scripts/prepare_joint_context_training.py',
       'scripts/train_joint_context_learning.py', 'scripts/audit_joint_context_learning.py',
       'scripts/run_joint_context_learning.py', 'scripts/review_joint_context_physical.py',
       'scripts/review_joint_writer_physical.py', 'scripts/evaluate_joint_writer_language.py',
       'references/joint_context_learning_protocol.md')


def sources():
    result = load(PREFLIGHT / 'registration.json')['source_sha256'].copy()
    for name in NEW:
        require((ROOT / name).read_bytes() == subprocess.check_output(
            ['git', 'show', 'HEAD:' + name], cwd=ROOT), 'Commit source before binding: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Bound source changed: ' + name)
    return result


def register():
    import numpy as np
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only registration')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve existing trial')
    torch.set_num_threads(2)
    pre = load(PREFLIGHT / 'execution/report.json')
    require(pre['complete'] and len(pre['records']) == 4 and pre['all_trainables_restored_to_registered_initialization']
            and not pre['updates_carried_into_fit'], 'Completed real-source preflight and reset required')
    old = load(CONTROL / 'registration.json'); value = old.copy()
    for name in ['execution_recovery_amendment', 'prospective_binding_amendment']:
        value.pop(name, None)
    bindings = old['bindings_sha256'].copy()
    def bind(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for path in [CONTROL / 'registration.json', CONTROL / 'schedule.npy', CONTROL / 'initial_parameters.pt',
                 CONTROL / 'cache/normalization.pt', CONTROL / 'fit/report.json', CONTROL / 'audit/report.json',
                 CONTROL / 'qualify/report.json', CONTROL / 'qualify/gate.json', CONTROL / 'qualify/frozen_readings.json',
                 PREFLIGHT / 'registration.json', PREFLIGHT / 'execution/report.json',
                 PREFLIGHT / 'recovery/external_sigterm_20261113/completion.json',
                 ROOT / 'references/evidence/joint_writer_blank_density_result_20261113.json']:
        bind(path)
    for job in old['qualification_jobs']:
        bind(ROOT / job['image'])
        bind(CONTROL / 'qualify/cases' / job['id'] / 'pixels.npz')
    rows = load(PARENT / 'training_manifest.json')['rows']
    schedule = np.load(CONTROL / 'schedule.npy', allow_pickle=False)
    require(schedule.shape == (41298, 2) and len(rows) == 13766, 'Complete original schedule required')
    for epoch in schedule.reshape(3, 13766, 2):
        require(np.array_equal(np.sort(epoch[:, 0]), np.arange(13766)), 'Each source once per epoch')
    require(np.array_equal(np.bincount(schedule[schedule[:, 1] == 1, 0], minlength=13766), np.full(13766, 2)),
            'Original fixed/proportional exposure required')
    require(sum(rows[int(i)]['target_glyphs'] for i in schedule[:, 0]) == 6108741, 'Complete target exposure required')
    require(shutil.disk_usage(ROOT).free >= 48 * 1024**3, 'Retain48GiB free disk')
    initial = tensor_inventory(expected_initial())
    value.update(schema='complete-joint-context-learning-v1', source_sha256=sources(), bindings_sha256=bindings,
        runtime=runtime(), schedule_file=str((CONTROL / 'schedule.npy').relative_to(ROOT)),
        external_program_sha256=external_runtime(),
        initial_trainable_inventory=initial, trainable_parameters=5005760, trainable_tensors=131,
        model_account=pre['model_account'], stop_loss_weight=.1,
        objective='complete proper image likelihood plus balanced frozen-stop loss; exact state VJP',
        context_recomputed_each_update=True, predictor_state_cache_used_for_training=False,
        body_query_feedback_stop_frozen=False, body_prompt_feedback_stop_and_affines_frozen=True,
        continuation_query_and_adapters_trainable=True,
        fit_seconds=108000, fit_pressure_seconds=7200, qualify_seconds=3600, qualify_pressure_seconds=300,
        minimum_entry_host_gib=48, minimum_runtime_host_gib=24, maximum_allocated_vram_gib=22,
        audit_seconds=900, audit_peak_rss_gib=8, automatic_retry=False,
        unchanged_control='Completed head-only trial: identical fresh head, point-final continuation initialization, normalization and schedule; old frozen verdict retained',
        control_physical_pixels=str((CONTROL / 'qualify/cases').relative_to(ROOT)),
        initial_head_updates=0, preflight_updates_carried=False,
        autonomous_evaluation_protocol='All708 streams specified by unchanged evaluate_joint_writer_language.plan; actual131-tensor snapshots restored before each arm. A separate committed post-gate execution binding is required before neural generation.',
        physical_review_workflow='scripts/review_joint_context_physical.py',
        language_quality_unmeasured=True, goal_achieved=False)
    RUN.mkdir()
    write_new(RUN / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), trainable_parameters=5005760, sources=13766, updates=41298)


def validate():
    r = load(PUBLIC)
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit/push exact registration first')
    require(r['source_sha256'] == sources() and r['runtime'] == runtime(), 'Source/runtime differs')
    for name, digest in r['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound input/control changed: ' + name)
    require(sha(ROOT / r['schedule_file']) == r['schedule_sha256'], 'Original schedule changed')
    return r


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['register', 'validate'])
    args = p.parse_args()
    print(json.dumps(register() if args.action == 'register' else dict(valid=bool(validate()))))
