#!/usr/bin/env python3
"""Fresh complete motor fit with the qualified global deterministic flag."""
import argparse
import copy
import json
import os

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
REG = ROOT / 'references/evidence/residual_motor_deterministic_comparison_registration_20261118.json'
RUN = ROOT / 'artifacts/residual_plan_motor_deterministic_v1'
PARENT = ROOT / 'references/evidence/residual_motor_loaded_comparison_registration_20261118.json'
EXTRA = ('scripts/run_deterministic_residual_motor.py',
         'references/residual_motor_deterministic_comparison_protocol_20261118.md',
         'references/residual_motor_replay_diagnostic_result_20261118.md',
         'references/evidence/residual_motor_replay_diagnostic_result_20261118.json',
         'references/evidence/residual_motor_replay_diagnostic_registration_20261118.json',
         'references/evidence/residual_motor_loaded_comparison_registration_20261118.json',
         'artifacts/residual_motor_replay_diagnostic_v1/result.json',
         'artifacts/residual_motor_replay_diagnostic_v1/original.json',
         'artifacts/residual_motor_replay_diagnostic_v1/deterministic.json',
         'artifacts/residual_motor_replay_diagnostic_v1.exit.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    result = original.load(ROOT / 'references/evidence/residual_motor_replay_diagnostic_result_20261118.json')
    original.require(result['deterministic_setting_qualified'] and result['all_trial_updates_reset'], 'Actual numerical qualification required')
    original.require(not result['modes'][0]['all_replays_exact'] and result['modes'][1]['all_replays_exact'], 'Retain observed control and qualified intervention')
    reg = copy.deepcopy(original.load(PARENT))
    reg['schema'] = 'ilm-residual-motor-deterministic-comparison-v1'
    reg['input_sha256'].update({n: original.sha(ROOT / n) for n in EXTRA})
    reg['head_resource_amendment']['output_run'] = str(RUN.relative_to(ROOT))
    reg['execution_amendment'] = dict(parent_registration_sha256=original.sha(PARENT),
        global_deterministic_algorithms=True, warn_only=False, fresh_original_initialization=True,
        actual_numerical_qualification=True, mathematical_model_loss_data_quality_and_resources_unchanged=True)
    original.write_new(REG, reg)
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(not RUN.exists(), 'Preserve every execution')
    import torch
    from scripts import train_residual_plan_motor_loaded as worker
    previous = torch.are_deterministic_algorithms_enabled()
    worker.REG = REG
    worker.RUN = RUN
    worker.FILES = tuple(dict.fromkeys((*worker.FILES, *EXTRA)))
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        worker.run()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
