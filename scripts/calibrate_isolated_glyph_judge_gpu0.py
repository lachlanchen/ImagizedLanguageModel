#!/usr/bin/env python3
"""GPU0 placement recovery; reuse the fixed judge, pixels and scoring contract."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from scripts import calibrate_isolated_glyph_judge as base

ROOT = base.ROOT
ORIGINAL = ROOT / 'artifacts/isolated_glyph_judge_v1'
RUNTIME = ROOT / 'artifacts/isolated_glyph_teacher_gpu0_v2'
BINARY = ROOT.parent / 'LocalLLM/.local/ollama/bin/ollama'
WEIGHTS = ROOT.parent / 'LocalLLM/.local/models/ollama'
PORT = 11438


def canonical_parameters(value):
    """Ollama prints this map in arbitrary order; preserve each actual value."""
    return sorted(tuple(line.split(None, 1)) for line in value.splitlines() if line.strip())


def configure():
    base.RUN = ROOT / 'artifacts/isolated_glyph_judge_gpu0_v2'
    base.REG = ROOT / 'references/evidence/isolated_glyph_judge_gpu0_registration_20261116.json'
    base.OUT = ROOT / 'references/evidence/isolated_glyph_judge_gpu0_calibration_20261116.json'
    old_protocol = base.PROTOCOL
    base.PROTOCOL = ROOT / 'references/isolated_glyph_judge_gpu0_recovery_20261116.md'
    base.judge.ORIGIN = f'http://127.0.0.1:{PORT}'
    original_fingerprint = base.judge.fingerprint
    def fingerprint():
        value = original_fingerprint()
        parameters = base.judge.local_json('/api/show', dict(model=base.judge.MODEL))['parameters']
        value['parameters_sha256'] = hashlib.sha256(json.dumps(
            canonical_parameters(parameters), ensure_ascii=False).encode()).hexdigest()
        value['parameter_digest_policy'] = 'Sorted default parameter fields, preserving values; request overrides separately bound.'
        return value
    base.judge.fingerprint = fingerprint
    original_sources = base.sources
    def sources():
        rows, bound = original_sources()
        paths = [Path(__file__), old_protocol, ROOT / 'tests/test_isolated_teacher_placement.py',
                 ROOT / 'references/evidence/isolated_glyph_judge_registration_20261116.json',
                 ORIGINAL / 'failure.json', ORIGINAL / 'owner.json',
                 RUNTIME / 'register_owner.json'] + sorted((ORIGINAL / 'responses').glob('*.json'))
        for path in paths:
            bound[base.relative(path)] = base.sha(path)
        return rows, bound
    base.sources = sources


def identity(pid):
    path = Path('/proc') / str(pid)
    return dict(pid=pid, start_ticks=int((path / 'stat').read_text().rsplit(')', 1)[1].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


@contextmanager
def server(action):
    began = time.monotonic(); base.guard(began, 60, reserve=48)
    if base.gpu_state()[0]['free_mib'] < 20 * 1024:
        raise RuntimeError('GPU0 entry reserve not met')
    # Both shared servers are device-1 pinned. Leave them and their owners alone.
    with socket.socket() as check:
        check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        check.bind(('127.0.0.1', PORT))
    uuid = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True, timeout=10).strip()
    if not uuid.startswith('GPU-') or '\n' in uuid:
        raise ValueError('Require one physical GPU0 UUID')
    if not BINARY.is_file() or not WEIGHTS.is_dir():
        raise FileNotFoundError('Reuse the verified shared binary and weights')
    env = {key: value for key, value in os.environ.items() if not key.startswith('OLLAMA_')}
    env.update(CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
        OLLAMA_HOST=f'127.0.0.1:{PORT}', OLLAMA_MODELS=str(WEIGHTS),
        OLLAMA_CONTEXT_LENGTH='2048', OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1',
        OLLAMA_KEEP_ALIVE='60s', OLLAMA_NO_CLOUD='1',
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # Exclusive log creation prevents a second launcher or silent recovery.
    with (RUNTIME / (action + '_server.log')).open('x') as log:
        child = subprocess.Popen([str(BINARY), 'serve'], env=env, cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        own = identity(child.pid)
        try:
            exposed = dict(v.split('=', 1) for v in (Path('/proc') / str(child.pid) / 'environ')
                           .read_text().split('\0') if '=' in v)
            if exposed.get('CUDA_VISIBLE_DEVICES') != uuid or exposed.get('OLLAMA_HOST') != env['OLLAMA_HOST']:
                raise ValueError('Owned server device binding differs')
            base.write_new(RUNTIME / (action + '_owner.json'), dict(**own,
                supervisor=identity(os.getpid()), physical_gpu_uuid=uuid,
                port=PORT, executable_sha256=base.sha(BINARY), existing_weight_store=str(WEIGHTS),
                started_unix=time.time(), shared_service_modified=False))
            while True:
                if child.poll() is not None:
                    raise RuntimeError('Owned teacher server exited before readiness')
                if time.monotonic() - began > 30:
                    raise TimeoutError('Owned teacher metadata readiness timeout')
                try:
                    base.judge.local_json('/api/version', timeout=1)
                    break
                except OSError:
                    time.sleep(.1)
            yield
        finally:
            # This process group was created here. Never target a shared daemon.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
            with socket.socket() as check:
                check.settimeout(1)
                port_closed = check.connect_ex(('127.0.0.1', PORT)) != 0
            base.write_new(RUNTIME / (action + '_shutdown.json'), dict(**own,
                server_exit_code=child.returncode, port_closed=port_closed,
                elapsed_seconds=time.monotonic() - began, final_gpu=base.gpu_state(),
                ended_unix=time.time()))
            if not port_closed:
                raise RuntimeError('Owned teacher port remains open after cleanup')


def main(action):
    configure()
    if os.environ.get('OPENBLAS_NUM_THREADS') != '2' or os.environ.get('OMP_NUM_THREADS') != '2':
        raise ValueError('Require the registered two-thread client')
    if action == 'score':
        base.score(); return
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with server(action):
            if action == 'register':
                base.register()
                value = base.load(base.REG)
                value['execution_recovery'] = dict(previous_attempt_incomplete=True,
                    previous_completed_responses=1, previous_failed_placement=True,
                    fixed_origin=base.judge.ORIGIN, pinned_physical_gpu=0,
                    reuse_existing_binary_and_weights=True, temporary_server=True,
                    original_prompt_model_pixels_score_and_gates_unchanged=True,
                    all_queries_repeated_prospectively_without_inspecting_first_reading=True)
                base.REG.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
            else:
                base.transcribe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'transcribe', 'score'))
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Stop the owned teacher and preserve partial evidence')
    signal.signal(signal.SIGTERM, interrupted)
    main(args.action)
