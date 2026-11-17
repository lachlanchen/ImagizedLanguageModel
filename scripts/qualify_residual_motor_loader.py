#!/usr/bin/env python3
"""CPU-only complete-cache qualification of the corrected plan-byte boundary."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
REG = ROOT / 'references/evidence/residual_motor_loader_registration_20261118.json'
RESULT = ROOT / 'references/evidence/residual_motor_loader_result_20261118.json'
FILES = ('scripts/qualify_residual_motor_loader.py', 'ilm/visual_lm/residual_motor_data.py',
         'tests/test_residual_motor_data.py', 'ilm/visual_lm/binary_plan_glyph_density.py',
         'references/evidence/residual_motor_loading_failure_20261118.json',
         'artifacts/residual_plan_motor_preparation_v1/physical_targets.pt',
         'references/evidence/residual_motor_preparation_result_20261118.json')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration')
    original.write_new(REG, dict(schema='ilm-residual-motor-loader-v1', input_sha256={n: original.sha(ROOT / n) for n in FILES},
        packages={n: importlib.metadata.version(n) for n in ['torch', 'numpy']}, source_plans=68948, code_bits=256,
        resources=dict(entry_host_gib=40, runtime_host_gib=24, maximum_rss_gib=2, cpu_threads=2, maximum_seconds=120),
        no_neural_calls=True, no_training=True, automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not RESULT.exists(), 'CPU-only fresh qualification')
    import torch
    from ilm.visual_lm.binary_plan_glyph_density import pack_plan
    from ilm.visual_lm.residual_motor_data import decode_motor_patterns
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2)
    began = time.perf_counter()
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = host_resources()
        original.require(host['available_gib'] >= 40 and host['some_pressure_avg10'] <= .5 and host['full_pressure_avg10'] <= .1, 'CPU loader admission')
        reg = original.load(REG)
        for n, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT / n) == digest, 'Loader binding changed: ' + n)
        original.require(reg['packages'] == {n: importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
        data = torch.load(ROOT / 'artifacts/residual_plan_motor_preparation_v1/physical_targets.pt', weights_only=True, map_location='cpu')
        patterns = data['plan_patterns']
        original.require(patterns.shape == (68948, 32) and patterns.dtype == torch.uint8, 'Complete actual serialized plans required')
        bits = decode_motor_patterns(patterns)
        original.require(bits.shape == (68948, 256) and torch.equal(pack_plan(bits).to(torch.uint8), patterns), 'Every actual plan byte must recover')
        previous_scalar_comparison = bool((patterns < 256).all())
        original.require(not previous_scalar_comparison and bool((patterns.long() < 256).all()), 'Observed dtype failure must reproduce without changing values')
        current = host_resources()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        elapsed = time.perf_counter() - began
        original.require(current['available_gib'] >= 24 and peak < 2 * 1024**3 and elapsed < 120, 'Loader qualification resource ceiling')
        result = dict(schema='ilm-residual-motor-loader-result-v1', registration_sha256=original.sha(REG),
                      source_plans=68948, serialized_bytes=68948 * 32, decoded_bits=68948 * 256,
                      all_actual_plan_bytes_roundtrip=True, original_byte_values_unchanged=True,
                      original_uint8_comparison_fails=True, widened_integer_comparison_passes=True,
                      no_neural_calls=True, no_training=True, no_reserved_language_access=True,
                      resources=dict(elapsed_seconds=elapsed, peak_rss_bytes=peak, entry_host_gib=host['available_gib'], final_host_gib=current['available_gib'], cpu_threads=2))
        original.write_new(RESULT, result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
