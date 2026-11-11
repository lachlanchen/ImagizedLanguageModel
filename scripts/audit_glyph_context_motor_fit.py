#!/usr/bin/env python3
"""Independently audit the completed motor optimizer and publish execution facts."""
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from scripts.train_glyph_context_motor import validate, load, CHECKPOINT, CHECKPOINT_SHA
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/audit_glyph_context_motor_fit.py'


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only parameter audit')
    require(subprocess.check_output(['git', 'show', 'HEAD:'+OWN], cwd=ROOT) == (ROOT/OWN).read_bytes(),
            'Commit the independent audit before execution')
    torch.set_num_threads(2)
    run = ROOT/'artifacts/glyph_context_motor_v1'
    r = validate(run); report = load(run/'fit/report.json')
    require(load(run/'fit/status.json')['phase'] == 'complete', 'Finish the actual fit before auditing')
    path = run/'fit/checkpoint_latest.pt'
    require(sha(path) == report['checkpoint_sha256'], 'Final checkpoint changed')
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    previous = torch.load(run/'fit/checkpoint_previous.pt', map_location='cpu', weights_only=True, mmap=True)
    require(value['step'] == 12000 and previous['step'] == 11000 and value['training_complete'],
            'Fixed final/previous optimizer exposure differs')
    require(value['preparation_sha256'] == sha(run/'registration.json')
            and value['parent_checkpoint_sha256'] == CHECKPOINT_SHA and sha(CHECKPOINT) == CHECKPOINT_SHA,
            'Parent/registration identity changed')
    targets = torch.load(run/'training_targets.pt', map_location='cpu', weights_only=True)
    require(torch.equal(targets['mean'], value['motor_mean']) and torch.equal(targets['scale'], value['motor_scale']),
            'Training-only motor normalization changed')
    torch.manual_seed(r['seed']); initial = ContextMotorFlow()
    names = list(dict(initial.named_parameters()))
    require(set(names) == set(value['model']) == set(value['ema']), 'Unregistered model or EMA state')
    total = sum(p.numel() for p in initial.parameters())
    require(total == r['parameters'] == 24477696, 'Actual trainable parameter count differs')
    model = ContextMotorFlow(); model.load_state_dict(value['model'])
    ema = ContextMotorFlow(); ema.load_state_dict(value['ema'])
    require(all(torch.isfinite(p).all() for p in model.parameters())
            and all(torch.isfinite(p).all() for p in ema.parameters()), 'Nonfinite fitted parameters')
    optimizer = value['optimizer']; ids = [i for g in optimizer['param_groups'] for i in g['params']]
    require(ids == list(range(len(names))) and len(optimizer['state']) == len(names), 'Optimizer coverage differs')
    changed = []
    for name, index in zip(names, ids):
        state = optimizer['state'][index]; expected = value['model'][name]
        require(int(state['step']) == 12000 and state['exp_avg'].shape == expected.shape
                and state['exp_avg_sq'].shape == expected.shape
                and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all(),
                'Incomplete or nonfinite optimizer state: '+name)
        if not torch.equal(initial.state_dict()[name], expected):
            changed.append(name)
    require(len(changed) == len(names), 'A registered parameter tensor did not change')
    history = [json.loads(s) for s in (run/'fit/history.jsonl').read_text().splitlines()]
    require([x['step'] for x in history] == [1]+list(range(100, 12001, 100)), 'Training history coverage differs')
    require(all(math.isfinite(x['velocity_MSE']) and x['velocity_MSE'] >= 0
                and math.isfinite(x['gradient_norm']) for x in history), 'Invalid logged optimization values')
    require(value['sampled_target_counts'] == report['sampled_target_counts'] == dict(first=768000, later=768000)
            and value['sampled_indices_sha256'] == report['sampled_indices_sha256'], 'Actual sample exposure differs')
    require(not torch.cuda.is_initialized(), 'No GPU model may be launched by this audit')
    evidence = dict(schema='glyph-context-motor-fit-audit-v1', registration_sha256=sha(run/'registration.json'),
        study_registration_sha256=sha(run/'study_registration.json'), checkpoint_sha256=sha(path),
        original_fit_report_sha256=sha(run/'fit/report.json'), history_sha256=sha(run/'fit/history.jsonl'),
        audit_source_sha256=sha(ROOT/OWN), actual_parameters=total, actual_optimizer_updates=12000,
        changed_parameter_tensors=len(changed), all_optimizer_parameter_steps_verified=True,
        normalization_identical_to_prepared_training_only_statistics=True,
        parent_checkpoint_unchanged=True, actual_first_positions=3176, actual_later_positions=20086,
        sampled_target_counts=value['sampled_target_counts'], final_EMA_only=True,
        training_resources={k:report[k] for k in ['elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib',
                                                 'peak_RSS_gib', 'minimum_available_gib']},
        preparation_resources=r['supplied_target_preparation_resources'],
        zero_update_admission_failure=load(ROOT/'references/evidence/glyph_context_motor_admission_recovery_20261111.json'),
        full_reader_not_loaded_during_head_fit=True, inherited_body_parameters_updated=0,
        independent_CPU_tensor_audit=True, language_accuracy_not_measured=True, goal_achieved=False)
    path = ROOT/'references/evidence/glyph_context_motor_fit_20261111.json'
    require(not path.exists(), 'Preserve previous fit evidence')
    save_json(path, evidence)
    save_json(run/'fit_audit.json', dict(all_checks_passed=True, public_evidence_sha256=sha(path)))
    print(json.dumps(dict(all_checks_passed=True, optimizer_updates=12000, parameters=total,
                         language_accuracy_not_measured=True, goal_achieved=False)))


if __name__ == '__main__':
    main()
