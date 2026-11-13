#!/usr/bin/env python3
"""Publish the completed real-source preflight without rerunning the model."""
import json
from pathlib import Path

from ilm.visual_lm.joint_writer_training import ROOT, load, sha, require
from scripts.review_joint_writer_physical import write_new


def main():
    run = ROOT / 'artifacts/joint_context_preflight_v1'
    r = load(run / 'registration.json'); report = load(run / 'execution/report.json')
    recovery = load(run / 'recovery/external_sigterm_20261113/completion.json')
    require(report['complete'] and report['registration_sha256'] == sha(run / 'registration.json'), 'Completed registered preflight required')
    require(report['all_trainables_restored_to_registered_initialization'] and not report['updates_carried_into_fit'], 'All preflight changes must be reset')
    for name, digest in r['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'Original preflight source changed')
    records = []
    for item in report['records']:
        row = {k: item[k] for k in ['case', 'source_id', 'target_glyphs', 'layout', 'loss', 'seconds', 'optimizer_updates']}
        if 'gradient_comparison' in item:
            row.update(gradient_tensors=len(item['gradient_comparison']),
                maximum_relative_gradient_error=max(v['relative_l2'] for v in item['gradient_comparison'].values()),
                maximum_absolute_gradient_error=max(v['maximum_absolute'] for v in item['gradient_comparison'].values()))
        else:
            row.update(gradient_tensors=len(item['gradient_norms']), actual_adam_states=item['adam_states'])
        records.append(row)
    value = dict(schema='joint-context-real-source-preflight-result-v1', complete=True,
        registration_sha256=sha(run / 'registration.json'), private_report_sha256=sha(run / 'execution/report.json'),
        records=records, model_account=report['model_account'], frozen_parameters_unchanged=report['frozen_parameters_unchanged'],
        feedback_and_condition_affine_buffers_bit_exact=report['feedback_and_condition_affine_buffers_bit_exact'],
        all_trainables_restored_to_registered_initialization=True, optimizer_updates=2, updates_carried_into_fit=False,
        peak_allocated_vram_gib=report['peak_allocated_vram_gib'], minimum_available_gib=report['minimum_available_gib'],
        resource_wait_seconds=report['resource_wait_seconds'], monitor_elapsed_seconds=report['elapsed_seconds'],
        original_sigterm_preserved=True, signal_sender_unknown=True,
        recovered_attempt_seconds=recovery['recovered_attempt_seconds'],
        cumulative_wall_seconds_upper_bound=recovery['cumulative_wall_seconds_upper_bound'],
        recovery_completion_sha256=sha(run / 'recovery/external_sigterm_20261113/completion.json'),
        original_peak_unmeasured=True, original_termination_cause_unproved=True,
        computational_precondition_only=True, new_autonomous_images=0, language_evaluated=False, goal_achieved=False)
    destination = ROOT / 'references/evidence/joint_context_preflight_result_20261114.json'
    write_new(destination, value)
    print(json.dumps(dict(result=str(destination.relative_to(ROOT)), sha256=sha(destination))))


if __name__ == '__main__':
    main()
