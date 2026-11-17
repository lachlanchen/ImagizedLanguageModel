#!/usr/bin/env python3
"""Bind and prepare complete training/evaluation physical correction targets."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
RUN = ROOT / 'artifacts/residual_plan_motor_preparation_v1'
REG = ROOT / 'references/evidence/residual_motor_preparation_registration_20261118.json'
PUBLIC = ROOT / 'references/evidence/residual_motor_preparation_result_20261118.json'
FILES = ('scripts/prepare_residual_plan_motor.py',
         'scripts/diagnose_visual_context_gradients.py', 'scripts/score_pixel_language_first_blocks.py', 'ilm/visual_lm/residual_plan_motor.py',
         'tests/test_residual_plan_motor.py', 'references/evidence/residual_plan_motor_cpu_20261118.json',
         'references/residual_plan_motor_direction_20261118.md',
         'references/evidence/spatial_capacity_comparison_result_20261118.json',
         'references/evidence/scalar_codec_interrupted_result_20261118.json',
         'ilm/visual_lm/binary_plan_glyph_density.py', 'ilm/visual_lm/spatial_plan_codec.py',
         'ilm/visual_lm/occupancy_glyph_density.py', 'ilm/visual_lm/conditional_pixel_density.py',
         'artifacts/spatial_capacity_comparison_v1/binary_final.pt',
         'artifacts/spatial_capacity_comparison_v1/binary_final.npz',
         'artifacts/nonlinear_feedback_alignment_v1/prepared.pt',
         'artifacts/binary_codec_preparation_v1/partition.pt')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    original.write_new(REG, dict(schema='ilm-residual-motor-preparation-v1',
        input_sha256={n: original.sha(ROOT / n) for n in FILES},
        packages={n: importlib.metadata.version(n) for n in ['torch', 'numpy']},
        source_images=68948, train_images=64041, held_images=4907, code_bits=256,
        retained_frozen_codec='artifacts/spatial_capacity_comparison_v1/binary_final.pt',
        no_new_neural_calls=True, no_training=True, no_source_alias_resplit=True,
        no_reserved_language_access=True, automatic_retry=False, full_language_fit_authorized=False,
        resources=dict(maximum_seconds=180, maximum_rss_gib=2, cpu_threads=2,
                       minimum_entry_host_gib=48, minimum_runtime_host_gib=24)))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation only')
    original.require(not RUN.exists() and not PUBLIC.exists(), 'Preserve every execution')
    reg = original.load(REG)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in [str(REG.relative_to(ROOT)), *[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git', 'show', 'HEAD:' + name]) == (ROOT / name).read_bytes(), 'Commit bound sources before preparation')
    original.require(subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], text=True, timeout=30).split()[0] == commit, 'Push before preparation')
    import numpy as np
    import torch
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2)
    began = time.perf_counter()
    samples = []
    def guard():
        host = host_resources()
        samples.append(host)
        original.require(host['available_gib'] >= 24 and time.perf_counter() - began < 180, 'Preparation reserve/deadline')
        original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, 'Preparation RSS ceiling')
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = host_resources()
        original.require(host['available_gib'] >= 48 and host['some_pressure_avg10'] <= .5 and host['full_pressure_avg10'] <= .1, 'Preparation admission')
        RUN.mkdir()
        try:
            for name, digest in reg['input_sha256'].items():
                guard()
                original.require(original.sha(ROOT / name) == digest, 'Bound source changed: ' + name)
            original.require(reg['packages'] == {n: importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
            control = original.load(ROOT / 'references/evidence/spatial_capacity_comparison_result_20261118.json')
            arm = next(a for a in control['arms'] if a['mode'] == 'binary')
            original.require(original.sha(ROOT / arm['checkpoint_file']) == arm['checkpoint_sha256'], 'Frozen final codec changed')
            original.require(original.sha(ROOT / arm['final']['raw_file']) == arm['final']['raw_sha256'], 'Saved base images/plans changed')
            original.require(control['all_275792_reconstructions_recounted'] and not control['support'], 'Use audited unqualified base without relabelling it')
            source = torch.load(ROOT / 'artifacts/nonlinear_feedback_alignment_v1/prepared.pt', weights_only=True, map_location='cpu')
            partition = torch.load(ROOT / 'artifacts/binary_codec_preparation_v1/partition.pt', weights_only=True, map_location='cpu')
            packed = source['packed_ink']
            original.require(packed.dtype == torch.uint8 and packed.shape == (68948, 128), 'Complete source bytes required')
            with np.load(ROOT / arm['final']['raw_file']) as data:
                base, plans = data['ink'], data['plan']
            original.require(base.shape == (68948, 1, 32, 32) and base.dtype == np.uint8 and np.isin(base, [0, 1]).all(), 'Saved physical bases required')
            original.require(plans.shape == (68948, 32) and plans.dtype == np.uint8, 'Complete actual hard plans required')
            base_packed = torch.from_numpy(np.packbits(base.reshape(68948, 1024), axis=1, bitorder='big'))
            residual_packed = base_packed ^ packed
            original.require(torch.equal(base_packed ^ residual_packed, packed), 'Every original target byte must recover exactly')
            original.require(len(partition['train_index']) == 64041 and len(partition['held_index']) == 4907, 'Retain original physical partition')
            path = RUN / 'physical_targets.pt'
            torch.save(dict(plan_patterns=torch.from_numpy(plans.copy()), base_packed=base_packed,
                            residual_packed=residual_packed, source_images=68948, code_bits=256), path)
            guard()
            for name, digest in reg['input_sha256'].items():
                original.require(original.sha(ROOT / name) == digest, 'Bound source mutated: ' + name)
            result = dict(schema='ilm-residual-motor-preparation-result-v1', source_commit=commit,
                registration_sha256=original.sha(REG), cache_file=str(path.relative_to(ROOT)), cache_sha256=original.sha(path),
                source_images=68948, train_images=64041, held_images=4907, all_original_target_bytes_recover=True,
                exact_original_physical_partition_retained=True, old_base_quality_unchanged=True,
                old_codec_still_unqualified=True, no_neural_calls=True, no_training=True,
                no_reserved_language_access=True, full_language_fit_authorized=False, goal_achieved=False,
                resources=dict(elapsed_seconds=time.perf_counter() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                               minimum_available_host_gib=min(v['available_gib'] for v in samples), cpu_threads=2))
            original.write_new(RUN / 'result.json', result)
            original.write_new(PUBLIC, result)
            print(json.dumps(result), flush=True)
        except BaseException as error:
            original.write_new(RUN / 'failure.json', dict(error_type=type(error).__name__, error=str(error), automatic_retry=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
