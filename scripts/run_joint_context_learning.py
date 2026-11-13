#!/usr/bin/env python3
"""Single serial fit/audit/qualification owner; no automatic recovery or generation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ilm.visual_lm.joint_context_training import ROOT, RUN, load, sha, require, stage_identity, require_process_runtime
from scripts.prepare_joint_context_training import validate
from scripts.score_pixel_language_first_blocks import owner_live
from scripts.train_visual_continuation import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); registration = validate(); require_process_runtime(registration)
    status = RUN / 'controller_status.json'
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists():
            require(not owner_live(load(status)) and args.resume, 'Preserve prior controller; manual recovery required')
            prior = load(status)
            if prior.get('child_identity'):
                require(not owner_live(prior['child_identity']), 'Original child remains live')
        identity = stage_identity(); child = None
        atomic_json(status, dict(identity, phase='starting', child_pid=None))
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'Controller received signal{signum}')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONPATH=str(ROOT))
        try:
            for action, folder in [('train', 'fit'), ('audit', 'audit'), ('qualify', 'qualify')]:
                report = RUN / folder / 'report.json'
                if report.exists():
                    require(load(report)['complete'], 'Preserve incomplete report')
                    continue
                child_status = RUN / folder / 'status.json'
                if child_status.exists():
                    require(not owner_live(load(child_status)) and args.resume, 'Preserve old stage; explicit resume required')
                command = ([sys.executable, '-u', 'scripts/audit_joint_context_learning.py'] if action == 'audit'
                           else [sys.executable, '-u', 'scripts/train_joint_context_learning.py', action])
                if child_status.exists() and action != 'audit':
                    command.append('--resume')
                env = dict(environment, CUDA_VISIBLE_DEVICES='') if action == 'audit' else environment
                with (RUN / (action + '_console.log')).open('a') as log, (RUN / 'controller_journal.jsonl').open('a', buffering=1) as journal:
                    began = time.perf_counter(); child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    child_identity = dict(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19], boot_id=identity['boot_id'])
                    atomic_json(status, dict(identity, phase=action, child_pid=child.pid, child_identity=child_identity))
                    journal.write(json.dumps(dict(event='started', action=action, child_pid=child.pid, child_identity=child_identity, unix=time.time())) + '\n')
                    journal.flush(); os.fsync(journal.fileno())
                    try:
                        result = child.wait()
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=20)
                            except subprocess.TimeoutExpired:
                                atomic_json(status, dict(identity, phase='interrupted_child_still_live', child_pid=child.pid, child_identity=child_identity))
                                raise
                        raise
                    finally:
                        if child.poll() is not None:
                            journal.write(json.dumps(dict(event='ended', action=action, child_pid=child.pid, unix=time.time(),
                                elapsed_seconds=time.perf_counter() - began, returncode=child.returncode,
                                report_sha256=sha(report) if report.exists() else None)) + '\n')
                            journal.flush(); os.fsync(journal.fileno())
                    require(result == 0 and report.exists() and load(report)['complete'], 'Stage failed; preserve before manual recovery')
                    child = None
            atomic_json(status, dict(identity, phase='awaiting_opaque_physical_review', child_pid=None,
                full_corpus_fit_complete=True, physical_gate_unmeasured=True, language_quality_unmeasured=True))
        except BaseException as exc:
            value = load(status)
            atomic_json(status, dict(value, phase='failed' if child is None or child.poll() is not None else 'interrupted_child_still_live', error=repr(exc)))
            raise
        finally:
            atomic_json(status, dict(load(status), pid=None))


if __name__ == '__main__':
    main()
