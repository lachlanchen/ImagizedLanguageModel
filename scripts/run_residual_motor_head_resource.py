#!/usr/bin/env python3
"""One physical-motor admission amendment; all model/quality fields unchanged."""
import argparse
import copy
import json
import os

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
REG = ROOT / 'references/evidence/residual_motor_head_admission_registration_20261118.json'
RUN = ROOT / 'artifacts/residual_plan_motor_head_resource_v1'
PARENT = ROOT / 'references/evidence/residual_motor_comparison_registration_20261118.json'
EXTRA = ('scripts/run_residual_motor_head_resource.py', 'ilm/visual_lm/residual_motor_resources.py',
         'references/residual_motor_head_resource_protocol_20261118.md',
         'references/evidence/residual_motor_admission_failure_20261118.json',
         'references/evidence/residual_motor_scheduling_failure_20261118.json',
         'references/evidence/residual_motor_comparison_registration_20261118.json',
         'artifacts/residual_plan_motor_comparison_v1/failure.json',
         'artifacts/residual_plan_motor_comparison_v1/status.json',
         'artifacts/residual_plan_motor_comparison_v1.exit.json',
         'artifacts/residual_plan_motor_admission_recovery_scheduling_v1.json',
         'artifacts/residual_plan_motor_admission_recovery_v1.exit.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    failure = original.load(ROOT / 'references/evidence/residual_motor_scheduling_failure_20261118.json')
    original.require(failure['worker_launches'] == 0 and failure['owned_processes_terminated'], 'Prior scheduler must be terminal without a worker')
    reg = copy.deepcopy(original.load(PARENT))
    reg['schema'] = 'ilm-residual-motor-head-resource-v1'
    reg['input_sha256'].update({n: original.sha(ROOT / n) for n in EXTRA})
    reg['resources']['minimum_entry_host_gib'] = 40
    reg['head_resource_amendment'] = dict(parent_registration_sha256=original.sha(PARENT),
        output_run=str(RUN.relative_to(ROOT)), runtime_host_reserve_gib=24, maximum_process_rss_gib=8,
        additional_entry_margin_gib=8, physical_motor_parameters=2385763, no_contextual_foundation_loaded=True,
        quality_criteria_unchanged=True, model_and_sources_unchanged=True, maximum_worker_launches=1,
        original_full_context_admission_source_unchanged=True)
    original.write_new(REG, reg)
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(not RUN.exists(), 'Preserve every execution')
    from scripts import train_residual_plan_motor as worker
    from ilm.visual_lm import visual_context_deployment_resources as resources
    from ilm.visual_lm.residual_motor_resources import physical_motor_gpu
    worker.REG = REG
    worker.RUN = RUN
    worker.FILES = tuple(dict.fromkeys((*worker.FILES, *EXTRA)))
    previous = resources.full_reader_gpu
    resources.full_reader_gpu = physical_motor_gpu
    try:
        worker.run()
    finally:
        resources.full_reader_gpu = previous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
