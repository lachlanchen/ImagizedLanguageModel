"""Streaming complete-sequence training primitives for the joint image writer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT
from .continuation_training import TrainingMonitor, learning_rate

PARENT = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
RUN = ROOT / 'artifacts/continuation_joint_writer_v1'
PUBLIC = ROOT / 'references/evidence/continuation_joint_writer_registration_20261113.json'
SEED = 2026111339


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_positions(length):
    require(length > 0, 'Nonempty complete target required')
    return [0] + (np.linspace(1, length-1, min(8, length-1), dtype=np.int64).tolist() if length > 1 else [])


def cache_directory(index, layout):
    return RUN / 'cache/train' / layout / f'{index:05}'


class Monitor(TrainingMonitor):
    def __init__(self, folder, seconds, pressure, allocated_gib):
        super().__init__(folder, seconds, pressure)
        self.allocated_gib = allocated_gib
        self.next_disk = 0.

    def guard(self):
        super().guard()
        require(torch.cuda.max_memory_allocated()/1024**3 <= self.allocated_gib, 'Stage VRAM limit reached')
        now = time.perf_counter()
        if now >= self.next_disk:
            require(shutil.disk_usage(RUN).free >= 48*1024**3, 'Retain48GiB free disk')
            self.next_disk = now + 2

    def receipt(self):
        value = super().receipt()
        if not np.isfinite(value['minimum_available_gib']):
            value['minimum_available_gib'] = None
        return value


def previous_seconds(action):
    journal = RUN / 'controller_journal.jsonl'
    if not journal.exists():
        return 0.
    pending = set(); elapsed = 0.
    for line in journal.read_text().splitlines():
        r = json.loads(line)
        if r['action'] != action:
            continue
        if r['event'] == 'started':
            pending.add(r['child_pid'])
        else:
            require(r['child_pid'] in pending, 'Unmatched stage end receipt')
            pending.remove(r['child_pid']); elapsed += r['elapsed_seconds']
    require(not pending - {os.getpid()}, 'Account for prior unterminated controller journal before recovery')
    return elapsed


class CachedSequences:
    """Authenticate each immutable array on first access; keep no state corpus in RAM."""
    def __init__(self):
        self.verified = {}
        report = RUN/'cache/report.json'
        if report.exists():
            value = load(report)
            self.record_hashes = {r['directory']:r['record_sha256'] for r in value['records']+value['qualification']}
        else:
            self.record_hashes = {}

    def get(self, directory):
        if self.record_hashes:
            key = str(directory.relative_to(ROOT))
            require(key in self.record_hashes and sha(directory/'record.json') == self.record_hashes[key],
                    'Cache record changed')
        record = load(directory / 'record.json')
        values = []
        for name, key in [('states.npy', 'states_sha256'), ('patterns.npy', 'patterns_sha256')]:
            path = directory / name
            stat = path.stat(); identity = (stat.st_size, stat.st_mtime_ns, record[key])
            if self.verified.get(str(path)) != identity:
                require(sha(path) == record[key], 'Cached array changed: ' + str(path))
                self.verified[str(path)] = identity
            values.append(np.load(path, allow_pickle=False))
        states, patterns = values
        require(states.shape == (record['glyphs']+1, 4096) and states.dtype == np.float32,
                'Complete FP32 causal states required')
        require(patterns.shape == (record['glyphs'], 128) and patterns.dtype == np.uint8,
                'Complete physical target patterns required')
        require(np.isfinite(states).all(), 'Nonfinite cached state')
        return torch.from_numpy(states), torch.from_numpy(patterns), record


def update(model, optimizer, states, patterns, position, total, monitor):
    """Exact equal-sequence reduction, including a non-divisible last microbatch."""
    require(len(states) == len(patterns)+1 and len(patterns) > 0, 'Predictor/target boundaries differ')
    optimizer.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    value = 0.; count = len(patterns)
    for start in range(0, count, 64):
        monitor.guard()
        end = min(start+64, count)
        # states[t] precedes target t. The terminal state never enters image loss.
        h = states[start:end].to(device)
        target = patterns[start:end].to(device=device, dtype=torch.long)
        logits = model(h, target).float()
        loss = -logits.log_softmax(-1).gather(-1, target[..., None]).sum() / (count*128)
        require(bool(torch.isfinite(loss)), 'Nonfinite complete-image likelihood')
        loss.backward(); value += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    lr = learning_rate(position, total, peak=2e-4, warmup=500)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step(); monitor.guard()
    return dict(step=position, glyphs=count, nll=value, learning_rate=lr,
                gradient_norm_before_clip=float(norm))


def condition_moments(states):
    """Mean and second moment over target-predicting states, excluding the terminal."""
    x = states[:-1].double()
    return x.mean(0), x.square().mean(0)
