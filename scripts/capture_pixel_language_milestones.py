#!/usr/bin/env python3
"""Preserve fixed assessed own states while the sole GPU learner continues.

The watcher is a small standard-library process. A short CPU subprocess checks
and archives each learned state after its owning assessment has completed.
No foundation weights, optimizer states or reference answers are archived.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEPS = (4000, 8000)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def live(owner):
    path = Path('/proc', str(owner['pid']), 'stat')
    try:
        return (owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and path.read_text().rsplit(')', 1)[1].split()[19] == owner['start_ticks'])
    except FileNotFoundError:
        return False


def capture(run, step):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for archival')
    require(step in STEPS, 'only the fixed 4000/8000 assessments are archived')
    import torch
    from ilm.visual_lm.complete_pixel_runtime import read_pixel_checkpoint
    from ilm.visual_lm.pixel_language_diversity import state_digest
    torch.set_num_threads(2)
    assessment = run/'execution'/f'assessment_{step:05}'/'report.json'
    expected = load(assessment)
    require(expected['update'] == step, 'owning assessment update differs')
    archive = run/'milestone_models'
    archive.mkdir(exist_ok=True)
    path = archive/f'actor_update{step:05}.pt'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        record = load(receipt)
        require(sha(path) == record['archive_sha256'] and record['model_sha256'] == expected['model_sha256']
            and record['assessment_sha256'] == sha(assessment), 'immutable milestone changed')
        return record
    require(not path.exists(), 'preserve an unexpected archive without a receipt')
    for name in ('checkpoint_latest.pt', 'checkpoint_previous.pt'):
        candidate = run/'execution'/name
        try:
            state, _, identity = read_pixel_checkpoint(candidate, run, allow_incomplete=True)
        except FileNotFoundError:
            continue  # An atomic rolling rename may leave latest absent briefly.
        if identity['checkpoint_update'] != step:
            del state
            continue
        digest = state_digest(state)
        require(digest == expected['model_sha256'], 'checkpoint does not own the assessed images')
        temporary = path.with_suffix('.pt.partial')
        require(not temporary.exists(), 'preserve an interrupted partial archive')
        torch.save(dict(model=state, update=step, training_complete=False,
            preparation_sha256=identity['preparation_sha256'], frozen_weights_saved=False,
            parent_checkpoint_sha256=identity['checkpoint_sha256']), temporary)
        temporary.replace(path)
        restored, _, archived_identity = read_pixel_checkpoint(path, run, allow_incomplete=True)
        require(state_digest(restored) == digest, 'saved own state changed in archival')
        record = dict(update=step, archive_sha256=archived_identity['checkpoint_sha256'],
            model_sha256=digest, owning_model_tensors=len(state),
            parent_checkpoint_sha256=identity['checkpoint_sha256'],
            preparation_sha256=identity['preparation_sha256'], assessment_sha256=sha(assessment),
            archival_code_sha256=sha(__file__), frozen_weights_saved=False,
            optimizer_saved=False, model_inference_calls=0, GPU_calls=0,
            selected_by_quality=False, language_quality_claim=False)
        save(receipt, record)
        return record
    raise ValueError(f'fixed update {step} no longer exists in either rolling checkpoint')


def watch(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for the lightweight watcher')
    source = Path(__file__).resolve()
    relative = str(source.relative_to(ROOT))
    require(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+relative], cwd=ROOT)).hexdigest() == sha(source),
            'commit the archival implementation first')
    archive = run/'milestone_models'
    archive.mkdir(exist_ok=True)
    with (archive/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner_status = load(run/'execution/status.json')
        owner = {key: owner_status[key] for key in ('pid', 'boot_id', 'start_ticks')}
        require(live(owner), 'the original model process must be verified live')
        watcher = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
        prior = archive/'watch_status.json'
        if prior.exists():
            saved = load(prior)
            require(not saved.get('pid') or not live(saved), 'an existing watcher is still live')
        completed = []
        def status(phase, **extra):
            save(prior, dict(phase=phase, **watcher, owner=owner, completed=completed,
                archival_code_sha256=sha(source), **extra))
        try:
            for step in STEPS:
                while not (run/'execution'/f'assessment_{step:05}'/'report.json').exists():
                    require(live(owner), 'original model owner exited before remaining assessments')
                    state = load(run/'execution/status.json')
                    require(state['phase'] not in ('failed', 'completed'), 'model execution ended before the requested assessment')
                    status('waiting_for_fixed_assessment', waiting_for=step,
                           model_phase=state['phase'], model_update=state.get('update'))
                    time.sleep(30)
                info = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines()}
                require(info['MemAvailable'] >= 24*1024**2 and (info['SwapTotal'] == 0 or
                    (info['SwapTotal']-info['SwapFree'])/info['SwapTotal'] <= .75), 'RAM/swap threshold reached before CPU archival')
                status('checking_and_archiving_own_state', waiting_for=step)
                subprocess.run([sys.executable, str(source), '--run', str(run), '--capture-update', str(step)], check=True, cwd=ROOT)
                completed.append(step)
            status('completed')
        except BaseException as error:
            status('failed', reason=repr(error))
            raise
        finally:
            if prior.exists():
                save(prior, dict(load(prior), pid=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--capture-update', type=int, choices=STEPS)
    args = parser.parse_args()
    if args.capture_update is not None:
        print(json.dumps(capture(args.run.resolve(), args.capture_update), indent=2))
    else:
        watch(args.run.resolve())
