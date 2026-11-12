#!/usr/bin/env python3
"""CPU audit of the completed registered fit; never generate or score language."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def event_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audit(run):
    # Import only in the explicit audit, with no model or image-generation imports.
    import numpy as np
    import torch

    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Run this audit with CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    p = load(run / 'preparation.json')
    report = load(run / 'fit/report.json')
    registration = digest(run / 'preparation.json')
    require(report['complete'] and report['actual_optimizer_steps'] == p['total_steps'], 'Incomplete fit')
    require(report['registration_sha256'] == registration, 'Completion registration differs')
    require(p['epochs'] == report['epochs'] == 3, 'Wrong epoch count')
    require(not report['language_quality_evaluated'] and not report['preflight_updates_carried'], 'Wrong completion claim')
    for name, expected in p['source_sha256'].items():
        require(digest(ROOT / name) == expected, f'Registered source changed: {name}')
    for name, key in [('training_manifest.json', 'training_manifest_sha256'), ('schedule.npy', 'schedule_sha256')]:
        require(digest(run / name) == p[key], f'Registered data changed: {name}')
    rows = load(run / 'training_manifest.json')['rows']
    schedule = np.load(run / 'schedule.npy', allow_pickle=False)
    require(len(rows) == p['training_cases'] and schedule.shape == (p['total_steps'], 2), 'Schedule size differs')
    snapshots = load(run / 'fit/snapshots.json')
    require(snapshots == report['snapshots'] and sorted(map(int, snapshots)) == p['snapshot_steps'], 'Snapshot inventory differs')
    snapshot_checks = {}
    for step, record in snapshots.items():
        path = ROOT / record['file']
        before = path.stat()
        actual = digest(path)
        after = path.stat()
        require(record['step'] == int(step) and actual == record['sha256'], 'Snapshot bytes differ')
        require((before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns), 'Snapshot changed during audit')
        snapshot_checks[step] = dict(sha256=actual, bytes=after.st_size)
    ledger = run / 'fit/ledger_attempt_001.jsonl'
    require(len(list((run / 'fit').glob('ledger_attempt_*.jsonl'))) == 1, 'Audit additional training attempts explicitly')
    previous = '0' * 64
    steps = 0
    exposure = 0
    replays = []
    kinds = Counter()
    layouts = Counter()
    presentations = Counter()
    epochs = [Counter() for _ in range(3)]
    loss_range = [float('inf'), -float('inf')]
    with ledger.open() as f:
        for line in f:
            event = json.loads(line)
            if event['event_kind'] == 'discarded_resume_verification':
                replays.append(event['step'])
                continue
            require(event['event_kind'] == 'scheduled', 'Unknown training event')
            actual = event.pop('event_sha256')
            require(event['previous_event_sha256'] == previous and event_digest(event) == actual, 'Broken ledger chain')
            previous = actual
            steps += 1
            require(event['step'] == steps and event['attempt'] == 1, 'Duplicate, missing or reordered update')
            index, natural = map(int, schedule[steps - 1])
            row = rows[index]
            layout = 'proportional' if natural else 'fixed_cells'
            require(event['row_index'] == index and event['case'] == row['id'] and event['kind'] == row['kind'], 'Wrong scheduled source')
            require(event['layout'] == layout and event['target_glyphs'] == row['target_glyphs'], 'Partial or relabelled target')
            for key in ('loss', 'image_loss', 'stop_loss', 'gradient_norm_before_clip', 'learning_rate', 'elapsed_seconds'):
                require(math.isfinite(event[key]), f'Nonfinite training measurement: {key}')
            peak, warmup, total = p['peak_learning_rate'], p['learning_rate_warmup'], p['total_steps']
            lr = peak * steps / warmup if steps <= warmup else peak * (
                p['learning_rate_final_fraction'] + (1 - p['learning_rate_final_fraction']) *
                (1 + math.cos(math.pi * (steps - warmup) / (total - warmup))) / 2)
            require(math.isclose(event['learning_rate'], lr, rel_tol=1e-12), 'Learning-rate schedule differs')
            exposure += event['target_glyphs']
            kinds[event['kind']] += 1
            layouts[layout] += 1
            presentations[index, natural] += 1
            epochs[(steps - 1) // len(rows)][index] += 1
            loss_range = [min(loss_range[0], event['loss']), max(loss_range[1], event['loss'])]
    require(steps == p['total_steps'] and exposure == p['scheduled_target_exposure'], 'Incomplete source exposure')
    require(exposure == report['scheduled_target_exposure'] and sorted(replays) == [5, 6], 'Wrong nominal/replay accounting')
    require(all(len(e) == len(rows) and set(e.values()) == {1} for e in epochs), 'Epoch lacks complete unique coverage')
    require(all(presentations[i, 0] == 1 and presentations[i, 1] == 2 for i in range(len(rows))), 'Wrong per-source layouts')
    initial = torch.load(ROOT / snapshots['0']['file'], map_location='cpu', weights_only=True)
    final = torch.load(ROOT / snapshots[str(p['total_steps'])]['file'], map_location='cpu', weights_only=True)
    require(final['completed_training'] and final['step'] == final['total_steps'] == steps, 'Wrong final checkpoint')
    require(final['registration_sha256'] == registration and final['source_sha256'] == p['source_sha256'], 'Wrong final identity')
    weights = final['trainable_weights']
    require(weights.keys() == initial['trainable_weights'].keys(), 'Parameter inventory changed')
    parameters = sum(t.numel() for t in weights.values())
    require(parameters == p['trainable_parameters'], 'Wrong trainable parameter count')
    changed = 0
    for name, tensor in weights.items():
        require(tensor.device.type == 'cpu' and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), 'Invalid final tensor')
        changed += not torch.equal(tensor, initial['trainable_weights'][name])
    optimizer = final['optimizer']
    indices = [i for group in optimizer['param_groups'] for i in group['params']]
    require(len(indices) == len(set(indices)) == len(weights) and set(indices) == set(optimizer['state']), 'Incomplete Adam inventory')
    for state in optimizer['state'].values():
        require(int(state['step']) == steps, 'Adam counter differs from full fit')
        require(all(bool(torch.isfinite(state[k]).all()) for k in ('exp_avg', 'exp_avg_sq')), 'Nonfinite Adam state')
    require(not torch.cuda.is_initialized(), 'CPU audit initialized CUDA')
    return dict(schema='visual-continuation-full-completion-audit-v1', registration_sha256=registration,
        training_report_sha256=digest(run / 'fit/report.json'), ledger_sha256=digest(ledger),
        source_files_verified=len(p['source_sha256']), scheduled_updates=steps, scheduled_target_glyphs=exposure,
        complete_unique_source_coverage_each_epoch=True, each_source_fixed_once_proportional_twice=True,
        kinds=dict(kinds), layouts=dict(layouts), discarded_replay_updates=replays, ledger_final_event_sha256=previous,
        training_loss_range=loss_range, snapshots=snapshot_checks, final_trainable_tensors=len(weights),
        final_trainable_parameters=parameters, changed_trainable_tensors=changed,
        actual_final_adam_states=len(optimizer['state']), all_final_adam_counters=steps,
        final_weights_and_moments_finite=True, frozen_body_versions_unchanged_reported=report['frozen_parameter_versions_unchanged'],
        frozen_body_independently_loaded=False, cumulative_training_seconds_reported=report['cumulative_elapsed_seconds'],
        peak_allocated_vram_gib_reported=report['peak_allocated_vram_gib'],
        no_new_training_or_inference=True, language_quality_measured=False, goal_achieved=False,
        audit_source_sha256=digest(Path(__file__)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'Preserve the existing audit output')
    result = audit(args.run.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))
