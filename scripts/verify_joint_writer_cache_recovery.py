#!/usr/bin/env python3
"""Verify saved joint-writer arrays before a manual, budgeted cache recovery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

BEGAN = time.perf_counter()
BEGAN_UNIX = time.time()

import numpy as np
import torch

from ilm.visual_lm.glyph_joint_density import unpack_glyphs
from ilm.visual_lm.joint_writer_training import (
    ROOT, RUN, CachedSequences, cache_directory, load, previous_seconds, require, sha,
)


def atomic_json(path, value):
    partial = path.with_suffix('.tmp')
    partial.write_text(json.dumps(value, indent=2) + '\n')
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only verification')
    require(not args.output.exists(), 'Preserve the first verification receipt')
    for rel in ['controller_status.json', 'cache/status.json']:
        require(load(RUN/rel)['pid'] is None, 'Original runtime must have exited')
    torch.set_num_threads(2)
    baseline = load(args.baseline)
    registration = load(args.baseline.parent/'original/registration.json')
    remaining = registration['cache_seconds'] - previous_seconds('cache')
    require(remaining > 0, 'Original cumulative cache budget exhausted')
    deadline = min(900., remaining)
    journal_path = RUN/'controller_journal.jsonl'
    with journal_path.open('a', buffering=1) as journal:
        def event(value):
            journal.write(json.dumps(dict(value, action='cache', child_pid=os.getpid(),
                                         kind='cpu_recovery_verification')) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        event(dict(event='started', unix=BEGAN_UNIX))
        result = dict(complete=False, model_calls=0, device='cpu', pid=os.getpid(),
                      baseline_sha256=sha(args.baseline), source_sha256=sha(Path(__file__)))
        error = None
        checked = 0
        minimum_ram = float('inf')
        def guard():
            nonlocal minimum_ram
            memory = {line.split(':', 1)[0]: int(line.split()[1])
                      for line in Path('/proc/meminfo').read_text().splitlines()}
            available = memory['MemAvailable']/1024**2
            minimum_ram = min(minimum_ram, available)
            require(available >= 24, 'Retain24GiB host RAM')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 <= 8, 'Verification RSS cap exceeded')
            require(time.perf_counter()-BEGAN <= deadline, 'Verification time cap exceeded')
        try:
            guard()
            cached = CachedSequences()
            require(not cached.record_hashes, 'This recovery expects the unfinished cache report')
            positions = 0
            for entry in baseline['records']:
                guard()
                directory = ROOT/entry['directory']
                require(sha(directory/'record.json') == entry['record_sha256'], 'Original cache record changed')
                states, patterns, record = cached.get(directory)
                require(record['states_sha256'] == entry['states_sha256'] and
                        record['patterns_sha256'] == entry['patterns_sha256'], 'Original array binding changed')
                positions += len(patterns)
                checked += 1
                del states, patterns
                if checked % 1000 == 0:
                    print(json.dumps(dict(verified_training_records=checked, total=27532)), flush=True)
            require(checked == 27532 and positions == 4072494, 'Complete original training cache required')
            golds = torch.load(ROOT/'artifacts/continuation_clean_history_v1/targets.pt',
                               map_location='cpu', weights_only=True)
            jobs = [job for job in registration['qualification_jobs'] if job['split'] == 'train']
            for index, job in enumerate(jobs):
                guard()
                directory = cache_directory(int(job['id'].split('_')[1]), job['layout'])
                if index == 0:
                    try:
                        _states, _record = cached.get(directory)
                    except ValueError as failure:
                        require(str(failure) == 'too many values to unpack (expected 2)', 'Unexpected original failure')
                        result['original_two_value_failure_reproduced'] = True
                    else:
                        raise ValueError('Original failing interface was not reproduced')
                states, patterns, record = cached.get(directory)
                require(record['gold_sha256'] == job['target_sha256'], 'Qualification target binding changed')
                require(torch.equal(unpack_glyphs(patterns.long()), golds[job['id']]), 'Qualification pixels differ')
                del states, patterns
            norm_path = RUN/'cache/normalization.pt'
            require(sha(norm_path) == baseline['normalization_sha256'], 'Original normalization changed')
            norm = torch.load(norm_path, map_location='cpu', weights_only=True)
            require(norm['source_sequences'] == 13766 and norm['target_states'] == positions and
                    norm['layouts'] == 'fixed1:proportional2', 'Training-only normalization coverage changed')
            require(all(value.shape == (4096,) and torch.isfinite(value).all()
                        for value in [norm['mean'], norm['scale']]), 'Invalid normalization')
            guard()
            result.update(complete=True, training_records=checked, target_positions=positions,
                          training_qualification_cases=len(jobs), all_array_hashes_shapes_dtypes_finite=True,
                          qualification_target_hashes_and_pixels_exact=True,
                          normalization_sha256=sha(norm_path), original_training_arrays_rewritten=False,
                          original_normalization_rewritten=False, cuda_initialized=torch.cuda.is_initialized())
        except BaseException as failure:
            error = failure
            result['error'] = repr(failure)
        finally:
            result.update(elapsed_seconds=time.perf_counter()-BEGAN,
                          peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                          minimum_available_gib=minimum_ram, verified_training_records=checked,
                          original_cache_budget_seconds=registration['cache_seconds'])
            atomic_json(args.output, result)
            event(dict(event='ended', unix=time.time(), elapsed_seconds=result['elapsed_seconds'],
                       returncode=1 if error else 0, report_sha256=sha(args.output)))
        if error:
            raise error
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
