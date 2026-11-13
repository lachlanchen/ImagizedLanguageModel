#!/usr/bin/env python3
"""CPU-only independent recount of complete sources, ledger and actual optimizer."""
import time
BEGAN = time.perf_counter()

import json
import os
from pathlib import Path
import resource

import numpy as np
import torch

from ilm.visual_lm.joint_context_training import (ROOT, RUN, PUBLIC, PARENT,
    load, sha, require, check_checkpoint, previous_seconds)
from ilm.visual_lm.continuation_training import TrainingImages, canonical_digest, same_nested
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.prepare_joint_context_training import validate
from scripts.train_visual_continuation import atomic_json


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only audit required')
    torch.set_num_threads(2); r = validate(); minimum = float('inf')
    remaining = r['audit_seconds'] - previous_seconds('audit')
    require(remaining > 0, 'Cumulative audit allowance exhausted')
    folder = RUN / 'audit'; folder.mkdir(exist_ok=False)
    def guard():
        nonlocal minimum
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        minimum = min(minimum, mem['MemAvailable'] / 1024**2)
        require(minimum >= 24 and time.perf_counter() - BEGAN <= remaining, 'CPU audit resource ceiling reached')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 <= r['audit_peak_rss_gib'], 'CPU audit RSS ceiling reached')
    fit = load(RUN / 'fit/report.json')
    require(fit['complete'] and fit['registration_sha256'] == sha(PUBLIC), 'Complete source-bound fit required')
    rows = load(PARENT / 'training_manifest.json')['rows']; schedule = np.load(ROOT / r['schedule_file'], allow_pickle=False)
    events = {}; executed = 0
    for ledger in sorted((RUN / 'fit').glob('ledger_*.jsonl')):
        with ledger.open() as stream:
            for line in stream:
                row = json.loads(line); digest = row.pop('event_sha256')
                require(canonical_digest(row) == digest, 'Training event changed')
                require(digest not in events, 'Duplicate event identity')
                events[digest] = row; executed += 1
    ordered = []; pointer = fit['ledger_final_sha256']; seen = set()
    while pointer != '0' * 64:
        require(pointer in events and pointer not in seen, 'Broken/cyclic retained chain')
        seen.add(pointer); row = events[pointer]; ordered.append(row); pointer = row['previous_event_sha256']
    ordered.reverse(); retained = [x for x in ordered if x['event_kind'] == 'scheduled']
    require(len(retained) == 41298 and [x['step'] for x in retained] == list(range(1, 41299)), 'Every retained update required')
    exposure = 0
    for event, (index, style) in zip(retained, schedule):
        row = rows[int(index)]; layout = 'fixed_cells' if style == 0 else 'proportional'
        require(event['source_index'] == int(index) and event['source_id'] == row['id'] and event['layout'] == layout
                and event['target_glyphs'] == row['target_glyphs'] and event['target_sha256'] == row['target_sha256']
                and event['input_sha256'] == row['images'][layout]['sha256'], 'Complete source/layout exposure differs')
        require(event['image_microbatches'] == (row['target_glyphs'] + 63) // 64, 'Incomplete image microbatch reduction')
        exposure += event['target_glyphs']
    require(exposure == 6108741, 'Retained target exposure differs')
    replay = [x for x in ordered if x['event_kind'] == 'discarded_resume_verification']
    require([x['step'] for x in replay] == [4, 5], 'Two actual nonretained resume updates required')
    data = TrainingImages(PARENT); positions = 0
    for index, row in enumerate(rows):
        guard()
        for layout in ['fixed_cells', 'proportional']:
            require(sha(ROOT / row['images'][layout]['file']) == row['images'][layout]['sha256'], 'Original prompt changed')
        gold = data.corpus[row['corpus_index']]['ink'][row['cut']:]
        require(len(gold) == row['target_glyphs'] and tensor_hash(gold) == row['target_sha256'], 'Complete physical target changed')
        positions += len(gold)
    require(positions == 2036247, 'All eligible complete targets required')
    snapshots = load(RUN / 'fit/snapshots.json'); require(snapshots == fit['snapshots'], 'Fitted snapshot list differs')
    require(sorted(map(int, snapshots)) == r['snapshot_steps'], 'Every fixed checkpoint required')
    checked = []
    for key, item in snapshots.items():
        guard(); path = ROOT / item['file']; require(sha(path) == item['sha256'], 'Snapshot changed')
        value = torch.load(path, weights_only=True, map_location='cpu'); check_checkpoint(value, r, int(key))
        checked.append(dict(step=int(key), trainable_tensors=len(value['trainable_weights']), sha256=item['sha256']))
        del value
    control = load(RUN / 'fit/resume_control.json')
    require(control['passed'] and control['exact_model_optimizer_rng'], 'Successful actual replay required')
    for file, key in [('resume_control3.pt', 'source_file_sha256'), ('resume_control5.pt', 'uninterrupted_file_sha256'),
                      ('resume_replayed5.pt', 'replayed_file_sha256')]:
        require(sha(RUN / 'fit' / file) == control[key], 'Actual replay checkpoint changed')
    a, b = [torch.load(RUN / 'fit' / file, weights_only=True, map_location='cpu')
            for file in ['resume_control5.pt', 'resume_replayed5.pt']]
    for name in ['trainable_weights', 'optimizer', 'rng']:
        same_nested(a[name], b[name])
    require(fit['frozen_parameters_and_affines_unchanged'] and fit['complete_context_recomputed_each_update']
            and not fit['predictor_state_cache_used_for_training'], 'Bound execution invariant missing')
    guard()
    atomic_json(folder / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC),
        fit_report_sha256=sha(RUN / 'fit/report.json'), retained_updates=len(retained), executed_events_all_attempts=executed,
        retained_target_exposure=exposure, complete_source_targets=positions, complete_source_count=len(rows),
        snapshots=checked, actual_saved_replay_weights_optimizer_rng_exact=True,
        freezing_is_bound_execution_assertion_not_second_neural_replay=True, new_model_calls=0, new_optimizer_updates=0,
        elapsed_seconds=time.perf_counter() - BEGAN, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        minimum_available_gib=minimum, language_quality_unmeasured=True, goal_achieved=False))
    print(json.dumps(dict(audit_complete=True, retained_updates=len(retained), target_exposure=exposure)), flush=True)


if __name__ == '__main__':
    main()
