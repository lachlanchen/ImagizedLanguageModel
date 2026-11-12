#!/usr/bin/env python3
"""Wait for resources, then run unchanged registered evaluators serially.

No model imports, training, changed generation rules or automatic child retries.
The original admission failure and all stage costs must remain available.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = {'evaluate': 'evaluation', 'parent-long-books': 'parent_long_books'}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def live(record):
    pid = record.get('pid')
    if not isinstance(pid, int):
        return False
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == record.get('start_ticks') and (
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record.get('boot_id'))
    except FileNotFoundError:
        return False


def prior_costs(journals):
    costs = {action: 0.0 for action in ACTIONS}
    for path in journals:
        active = {}
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            action = row['action']
            if action not in ACTIONS:
                continue
            key = (action, row['child_pid'])
            if row['event'] == 'started':
                require(key not in active, 'Duplicate child start receipt')
                active[key] = row
            elif row['event'] == 'ended':
                require(key in active, 'Child end without start receipt')
                active.pop(key)
                seconds = row['elapsed_seconds']
                require(math.isfinite(seconds) and seconds >= 0, 'Invalid prior child cost')
                costs[action] += seconds
            else:
                raise RuntimeError('Unknown child event')
        require(not active, 'Unaccounted interrupted child: preserve and reconcile its cost first')
    return costs


def budget_seconds(maximum, spent):
    require(math.isfinite(maximum) and math.isfinite(spent) and 0 <= spent < maximum, 'Stage budget exhausted')
    # Terminate the owned process group before the cumulative ceiling, with a
    # ten-second cleanup margin. This only tightens the registered wall limit.
    remaining = maximum - spent - 10.0
    require(remaining > 0, 'Insufficient remaining stage budget')
    return remaining


def admitted(resources, minimum_ram, minimum_gpu_mib):
    return (resources['available_gib'] >= minimum_ram + 2.0 and
            resources['gpu_free_mib'] >= minimum_gpu_mib and resources['mutex_free'] and
            resources['some_avg10'] <= 0.5 and resources['full_avg10'] <= 0.1)


def evaluator_environment(base):
    # Match numerics() in the registered evaluator before it imports CUDA.
    # A tmux launch must not depend on a particular interactive shell's exports.
    return dict(base, CUDA_VISIBLE_DEVICES='0', PYTHONPATH=str(ROOT),
                CUBLAS_WORKSPACE_CONFIG=':4096:8')


def resources():
    mem = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
    pressure = {x.split()[0]: float(x.split()[1].split('=')[1]) for x in Path('/proc/pressure/memory').read_text().splitlines()}
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = True
        except BlockingIOError:
            free = False
    gpu = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip())
    return dict(available_gib=mem['MemAvailable'] / 1048576, gpu_free_mib=gpu, mutex_free=free,
                some_avg10=pressure['some'], full_avg10=pressure['full'])


def stop_owned(child):
    if child.poll() is not None:
        return
    require(os.getpgid(child.pid) == child.pid, 'Refuse to terminate an unowned process group')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def main():
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Recovery interrupted; preserve child cost and owned-process cleanup')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--name', default='evaluation_recovery_001')
    parser.add_argument('--wait-seconds', type=float, default=86400)
    args = parser.parse_args()
    run = args.run.resolve()
    require(args.name.startswith('evaluation_recovery_') and Path(args.name).name == args.name, 'Invalid recovery name')
    require(math.isfinite(args.wait_seconds) and args.wait_seconds > 0, 'Invalid admission wait limit')
    p = load(run / 'preparation.json')
    require(load(run / 'fit/report.json')['complete'], 'Complete training first; this launcher never trains')
    audit = load(run / 'completion_audit_20261113.json')
    require(audit['registration_sha256'] == sha(run / 'preparation.json') and
            audit['audit_source_sha256'] == sha(ROOT / 'scripts/audit_visual_continuation_completion.py') and
            audit['training_report_sha256'] == sha(run / 'fit/report.json') and
            audit['all_final_adam_counters'] == p['total_steps'], 'Completion audit differs')
    failure = run / 'admission_failure_20261113_001'
    receipt = load(failure / 'receipt.json')
    require(receipt['training_complete'] and receipt['generated_outputs_before_retry'] == 0,
            'Reconcile the preserved failed attempt before recovery')
    for name, binding in receipt['preserved'].items():
        require(sha(failure / name) == binding['sha256'], 'Preserved failure evidence changed')
    lock = (ROOT / 'artifacts/ilm_evaluation_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path in [run / 'controller_status.json', run / 'fit/status.json', *[run / x / 'status.json' for x in ACTIONS.values()]]:
        if path.exists():
            require(not live(load(path)), 'Original stage owner remains live')
    destination = run / args.name
    journals = [run / 'controller_children.jsonl', *sorted(run.glob('evaluation_recovery_*/children.jsonl'))]
    costs = prior_costs(journals)
    destination.mkdir(exist_ok=False)
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())
    began = time.perf_counter()
    waiting = 0.0
    child = None
    write(destination / 'start.json', dict(identity, source_sha256=sha(Path(__file__)),
        registration_sha256=sha(run / 'preparation.json'), completion_audit_sha256=sha(run / 'completion_audit_20261113.json'),
        prior_child_cost_seconds=costs, registered_stage_limit_seconds=p['evaluation_maximum_seconds'],
        original_entry_ram_gib=p['minimum_entry_host_ram_gib'], conservative_prelaunch_headroom_gib=2,
        admission_wait_limit_seconds=args.wait_seconds, original_evaluators_unchanged=True))
    def status(phase, **extra):
        write(destination / 'status.json', dict(identity, phase=phase, elapsed_seconds=time.perf_counter() - began,
              admission_wait_seconds=waiting, child_pid=child.pid if child and child.poll() is None else None, **extra))
    try:
        for action, folder in ACTIONS.items():
            report_path = run / folder / 'report.json'
            if report_path.exists() and load(report_path)['complete']:
                continue
            status('waiting_for_admission', pending_action=action)
            while True:
                observed = resources()
                status('waiting_for_admission', pending_action=action, resources=observed)
                if admitted(observed, p['minimum_entry_host_ram_gib'], p['minimum_free_gpu_gib'] * 1024):
                    break
                require(waiting < args.wait_seconds, 'Admission wait limit exhausted; no model launched')
                before = time.perf_counter(); time.sleep(min(10, args.wait_seconds - waiting)); waiting += time.perf_counter() - before
            for name, expected in p['source_sha256'].items():
                require(sha(ROOT / name) == expected, f'Registered source changed: {name}')
            limit = budget_seconds(p['evaluation_maximum_seconds'], costs[action])
            env = evaluator_environment(os.environ)
            command = [sys.executable, '-u', 'scripts/train_visual_continuation.py', action, '--run', str(run)]
            child_began = time.perf_counter()
            with (destination / f'{action}_console.log').open('x') as log, (destination / 'children.jsonl').open('a') as journal:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                def event(kind, **extra):
                    journal.write(json.dumps(dict(event=kind, action=action, child_pid=child.pid, unix=time.time(), **extra)) + '\n')
                    journal.flush(); os.fsync(journal.fileno())
                event('started', command=command)
                try:
                    while child.poll() is None:
                        require(time.perf_counter() - child_began < limit, 'Cumulative stage wall budget exhausted')
                        status(action, remaining_child_wall_seconds=limit - (time.perf_counter() - child_began))
                        try:
                            child.wait(timeout=min(2.0, max(0.01, limit - (time.perf_counter() - child_began))))
                        except subprocess.TimeoutExpired:
                            pass
                finally:
                    stop_owned(child)
                    elapsed = time.perf_counter() - child_began
                    event('ended', elapsed_seconds=elapsed, returncode=child.returncode)
                    costs[action] += elapsed
                require(child.returncode == 0 and report_path.exists() and load(report_path)['complete'], 'Child incomplete; preserve attempt before another retry')
                require(costs[action] <= p['evaluation_maximum_seconds'], 'Cumulative stage cost exceeded')
        require(load(run / 'evaluation/report.json')['total_streams'] == 708 and
                load(run / 'parent_long_books/report.json')['total_streams'] == 24, 'Incomplete registered output coverage')
        status('preparing_opaque_review')
        if not (run / 'review/mapping.json').exists():
            subprocess.run([sys.executable, 'scripts/review_visual_continuation.py', 'prepare', '--run', str(run)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT)), check=True, timeout=180)
        write(destination / 'report.json', dict(complete=True, stage_child_cost_seconds=costs,
            prelaunch_admission_wait_seconds=waiting, total_elapsed_seconds=time.perf_counter() - began,
            generated_streams=732, original_training_restarted=False, model_or_generation_rules_changed=False,
            language_scoring_complete=False, review_mapping_sha256=sha(run / 'review/mapping.json')))
        status('ready_for_complete_output_review')
    except BaseException as error:
        status('failed', error=repr(error), accounted_stage_child_cost_seconds=costs)
        raise
    finally:
        if child:
            stop_owned(child)
        record = load(destination / 'status.json'); record['pid'] = None; write(destination / 'status.json', record)


if __name__ == '__main__':
    main()
