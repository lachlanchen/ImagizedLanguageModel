#!/usr/bin/env python3
"""One explicitly registered recovery; preserves the original preflight source."""
import time
BEGAN = time.perf_counter()

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/joint_context_preflight_v1'
RECOVERY = RUN / 'recovery/external_sigterm_20261113'
PUBLIC = ROOT / 'references/evidence/joint_context_preflight_registration_20261113.json'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from scripts.review_joint_writer_physical import require, write_new
    registration = load(PUBLIC); amendment = registration['execution_recovery_amendment']
    record_path = ROOT / amendment['record']
    require(sha(record_path) == amendment['sha256'], 'Recovery record changed')
    record = load(record_path)
    require(sha(Path(__file__)) == record['recovery_source_sha256'], 'Recovery wrapper changed')
    require(Path(__file__).read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:scripts/recover_joint_context_preflight.py'], cwd=ROOT), 'Commit recovery source first')
    require(not Path('/proc/1951199').exists(), 'Original worker remains present')
    require(not (RUN / 'execution').exists(), 'Preserve the original execution before manual recovery')
    manifest = load(RECOVERY / 'preserved_manifest.json')
    require(sha(RECOVERY / 'preserved_manifest.json') == record['preserved_manifest_sha256'], 'Archive index changed')
    for name, digest in manifest.items():
        require(sha(RECOVERY / 'original' / name) == digest, 'Original evidence changed: ' + name)
    gpu_root = ROOT.parent / 'LazyPromotion/.local/gpu-userspace-595.84-20261101/root/usr'
    external = dict(compatibility_wrapper=Path.home() / '.local/bin/with-current-nvidia',
        strace=Path('/usr/bin/strace'), cuda_driver=gpu_root / 'lib/x86_64-linux-gnu/libcuda.so.595.84',
        nvml=gpu_root / 'lib/x86_64-linux-gnu/libnvidia-ml.so.595.84')
    require({name: sha(path) for name, path in external.items()} == record['external_program_sha256'],
            'Registered per-process compatibility/tracing installation changed')
    require(os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == str(gpu_root / 'lib/x86_64-linux-gnu'),
            'Use the existing per-process NVIDIA compatibility wrapper')
    tracer = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines()
                      if x.startswith('TracerPid:')))
    require(tracer > 0 and sha(Path('/proc', str(tracer), 'exe')) == record['external_program_sha256']['strace'],
            'Run this one recovery under the registered signal tracer')
    with (RECOVERY / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time(),
            signal_tracer_pid=tracer,
            original_wall_seconds_debit_upper_bound=record['prior_wall_seconds_debit_upper_bound'])
        write_new(RECOVERY / 'launch.json', identity)
        error = None
        try:
            from scripts import preflight_joint_context_learning as preflight
            # Only shorten the original time allowance. No model/data/gradient
            # setting changes, and no prior optimizer state exists to carry.
            preflight.BEGAN = BEGAN - record['prior_wall_seconds_debit_upper_bound']
            value = preflight.execute()
            print(json.dumps(value), flush=True)
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            elapsed = time.perf_counter() - BEGAN
            write_new(RECOVERY / 'completion.json', dict(identity, error=error,
                recovered_attempt_seconds=elapsed,
                cumulative_wall_seconds_upper_bound=elapsed + record['prior_wall_seconds_debit_upper_bound'],
                original_limits_not_reset=True, automatic_retry=False))


if __name__ == '__main__':
    main()
