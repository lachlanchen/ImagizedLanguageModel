"""Provenance and accounting for the complete whole-glyph ranking comparison."""
import json
import math
import os
from pathlib import Path
import time

import torch

from .joint_writer_training import ROOT, PARENT, RUN as WRITER, load, sha, require
from .semantic_glyph_loop import tensor_hash

CONTROL = ROOT / 'artifacts/joint_context_learning_v1'
RUN = ROOT / 'artifacts/global_glyph_ranking_v1'
PUBLIC = ROOT / 'references/evidence/global_glyph_ranking_registration_20261114.json'
PREFLIGHT = ROOT / 'artifacts/global_glyph_ranking_preflight_v1'
PARAMETERS = 5005760


def check_training_event(event, r, expected_nonempty=None):
    """Independently check objective accounting, including actual source whitespace."""
    for key in ['image_loss', 'ranking_loss', 'stop_loss', 'loss', 'state_gradient_norm',
                'gradient_norm_before_clip', 'learning_rate', 'elapsed_seconds']:
        require(math.isfinite(event[key]) and event[key] >= 0, 'Invalid training metric: ' + key)
    expected = event['image_loss'] + r['ranking_strength'] * event['ranking_loss'] + .1 * event['stop_loss']
    require(math.isclose(event['loss'], expected, rel_tol=1e-10, abs_tol=1e-10), 'Objective components differ')
    require(type(event['nonempty_targets']) is int and 0 <= event['nonempty_targets'] <= event['target_glyphs'],
            'Invalid visible-ink target count')
    if expected_nonempty is not None:
        require(event['nonempty_targets'] == expected_nonempty, 'Ranking mask differs from complete source pixels')
    if event['nonempty_targets'] == 0:
        require(event['ranking_loss'] == 0., 'True whitespace must not receive anti-blank ranking')


def expected_initial():
    """CPU construction from authenticated original tensors, without a base-model copy."""
    r = load(WRITER / 'registration.json')
    parent = torch.load(ROOT / r['parent_checkpoint']['file'], weights_only=True, map_location='cpu')
    values = {n: p for n, p in parent['trainable_weights'].items()
              if n == 'dynamics.query' or n.startswith('dynamics.adapters.')}
    head = torch.load(WRITER / 'initial_parameters.pt', weights_only=True, map_location='cpu')
    values.update({'bridge.density.' + n: p for n, p in head.items()})
    require(len(values) == 131 and sum(p.numel() for p in values.values()) == PARAMETERS,
            'Every actual context/head trainable required')
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all()) for p in values.values()),
            'Finite FP32 initial parameters required')
    return values


def tensor_inventory(values):
    return {n: dict(shape=list(p.shape), elements=p.numel(), sha256=tensor_hash(p))
            for n, p in sorted(values.items())}


def previous_seconds(action):
    path = RUN / 'controller_journal.jsonl'
    elapsed = 0.; pending = set()
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        if row['action'] != action:
            continue
        if row['event'] == 'started':
            require(row['child_pid'] not in pending, 'Duplicate child start')
            pending.add(row['child_pid'])
        else:
            require(row['event'] == 'ended' and row['child_pid'] in pending, 'Unmatched child end')
            pending.remove(row['child_pid'])
            require(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] >= 0, 'Invalid cost')
            elapsed += row['elapsed_seconds']
    require(not pending - {os.getpid()}, 'Reconcile prior unterminated attempt before recovery')
    return elapsed


def check_checkpoint(value, r, step=None):
    require(value['schema'] == 'global-glyph-ranking-checkpoint-v1'
            and value['registration_sha256'] == sha(PUBLIC)
            and value['source_sha256'] == r['source_sha256']
            and value['runtime'] == r['runtime'], 'Checkpoint provenance differs')
    require(value['objective'] == r['objective'] and value['ranking_strength'] == r['ranking_strength'] == 1.
            and value['margin_nats'] == r['margin_nats'] == 1.
            and value['normalization_sha256'] == r['bindings_sha256'][str((WRITER / 'cache/normalization.pt').relative_to(ROOT))],
            'Checkpoint objective/normalization differs')
    require(step is None or value['step'] == step, 'Wrong snapshot step')
    require(value['completed_training'] is (value['step'] == r['total_steps']), 'Wrong completion metadata')
    inventory = tensor_inventory(value['trainable_weights'])
    expected = r['initial_trainable_inventory']
    require(inventory.keys() == expected.keys(), 'Wrong trainable names')
    require(all(inventory[n]['shape'] == expected[n]['shape'] for n in expected), 'Wrong trainable shapes')
    require(all(p.dtype == torch.float32 and bool(torch.isfinite(p).all())
                for p in value['trainable_weights'].values()), 'Nonfinite/non-FP32 trainable')
    require(sum(p.numel() for p in value['trainable_weights'].values()) == PARAMETERS, 'Wrong parameter count')
    if value['step'] == 0:
        require(inventory == expected, 'Initial context/head draw changed')
    else:
        state = value['optimizer']['state']
        require(len(state) == len(expected) and {int(s['step']) for s in state.values()} == {value['step']},
                'Every actual Adam counter must equal the saved position')
        require(all(s[k].dtype == torch.float32 and bool(torch.isfinite(s[k]).all())
                    for s in state.values() for k in ['exp_avg', 'exp_avg_sq']), 'Invalid Adam moments')


def stage_identity():
    return dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())


def external_runtime():
    root = ROOT.parent / 'LazyPromotion/.local/gpu-userspace-595.84-20261101/root/usr/lib/x86_64-linux-gnu'
    paths = dict(compatibility_wrapper=Path.home() / '.local/bin/with-current-nvidia',
                 strace=Path('/usr/bin/strace'), cuda_driver=root / 'libcuda.so.595.84',
                 nvml=root / 'libnvidia-ml.so.595.84')
    return {name: sha(path) for name, path in paths.items()}


def require_process_runtime(r):
    root = ROOT.parent / 'LazyPromotion/.local/gpu-userspace-595.84-20261101/root/usr/lib/x86_64-linux-gnu'
    require(external_runtime() == r['external_program_sha256'], 'Registered compatibility/tracing runtime changed')
    require(os.environ.get('LD_LIBRARY_PATH', '').split(':')[0] == str(root),
            'Launch with the existing per-process NVIDIA compatibility wrapper')
