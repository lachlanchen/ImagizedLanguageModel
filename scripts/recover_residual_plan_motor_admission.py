#!/usr/bin/env python3
"""One separately bound admission recovery; unchanged model and quality rules."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
REG = ROOT / 'references/evidence/residual_motor_admission_recovery_registration_20261118.json'
RUN = ROOT / 'artifacts/residual_plan_motor_admission_recovery_v1'
PARENT = ROOT / 'references/evidence/residual_motor_comparison_registration_20261118.json'
EXTRA = ('scripts/recover_residual_plan_motor_admission.py',
         'references/residual_motor_admission_recovery_protocol_20261118.md',
         'references/evidence/residual_motor_admission_failure_20261118.json',
         'references/evidence/residual_motor_comparison_registration_20261118.json',
         'artifacts/residual_plan_motor_comparison_v1/failure.json',
         'artifacts/residual_plan_motor_comparison_v1/status.json',
         'artifacts/residual_plan_motor_comparison_v1.exit.json',
         'artifacts/residual_plan_motor_comparison_v1.log')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    parent = original.load(PARENT)
    failure = original.load(ROOT / 'references/evidence/residual_motor_admission_failure_20261118.json')
    original.require(failure['no_model_constructed'] and failure['retained_updates'] == 0, 'Only the unexecuted admission failure may recover')
    result = copy.deepcopy(parent)
    result['schema'] = 'ilm-residual-motor-admission-recovery-v1'
    result['input_sha256'].update({n: original.sha(ROOT / n) for n in EXTRA})
    result['admission_recovery'] = dict(parent_registration_sha256=original.sha(PARENT),
        original_failure_preserved=True, changed_model_or_training=False, changed_quality_or_resource_ceiling=False,
        original_run='artifacts/residual_plan_motor_comparison_v1', output_run=str(RUN.relative_to(ROOT)),
        scheduling_entry_host_gib=50, stable_observations=2, observation_interval_seconds=10,
        maximum_scheduling_wait_seconds=300, total_deadline_includes_scheduling=True, maximum_worker_launches=1)
    original.write_new(REG, result)
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(not RUN.exists(), 'Preserve every execution')
    began = time.perf_counter()
    reg = original.load(REG)
    for name, digest in reg['input_sha256'].items():
        original.require(original.sha(ROOT / name) == digest, 'Recovery binding changed: ' + name)
    record_path = ROOT / 'artifacts/residual_plan_motor_admission_recovery_scheduling_v1.json'
    original.require(not record_path.exists(), 'Do not repeat scheduling')
    observations = []
    stable = 0
    while True:
        memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        pressure = {line.split()[0]: float(line.split()[1].split('=')[1]) for line in Path('/proc/pressure/memory').read_text().splitlines()}
        gpu = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
        sample = dict(elapsed_seconds=time.perf_counter() - began, available_host_gib=memory['MemAvailable'] / 1024**2,
                      some_pressure_avg10=pressure['some'], full_pressure_avg10=pressure['full'], free_gpu_mib=gpu,
                      free_disk_gib=shutil.disk_usage(ROOT).free / 1024**3)
        admitted = sample['available_host_gib'] >= 50 and pressure['some'] <= .5 and pressure['full'] <= .1 and gpu >= 20 * 1024 and sample['free_disk_gib'] >= 48
        stable = stable + 1 if admitted else 0
        sample['consecutive_admitted_observations'] = stable
        observations.append(sample)
        record = dict(registration_sha256=original.sha(REG), phase='waiting', observations=observations, automatic_retry=False, worker_launches=0)
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        if stable >= 2:
            break
        if time.perf_counter() - began + 10 > 300:
            record.update(phase='terminal_scheduling_reserve_unavailable', elapsed_seconds=time.perf_counter() - began)
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            raise RuntimeError('Recovery scheduling reserve unavailable; no worker launched')
        time.sleep(10)
    remaining = 2400 - (time.perf_counter() - began)
    original.require(remaining > 0, 'Total deadline exhausted before worker')
    def expired(_s, _f):
        raise TimeoutError('Recovery total deadline including scheduling')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    record.update(phase='worker_starting', worker_launches=1, scheduling_elapsed_seconds=time.perf_counter() - began)
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    from scripts import train_residual_plan_motor as worker
    worker.REG = REG
    worker.RUN = RUN
    worker.FILES = tuple(dict.fromkeys((*worker.FILES, *EXTRA)))
    try:
        worker.run()
        record['phase'] = 'terminal_worker_complete'
    except BaseException as error:
        record.update(phase='terminal_worker_exception', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        record['total_elapsed_seconds'] = time.perf_counter() - began
        record_path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
