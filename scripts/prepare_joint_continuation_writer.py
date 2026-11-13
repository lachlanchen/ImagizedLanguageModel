#!/usr/bin/env python3
"""Freeze the complete data, unchanged model and full-language comparison."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from ilm.visual_lm.joint_writer_training import (ROOT, PARENT, RUN, PUBLIC, SEED,
    load, sha, require, selected_positions)
from ilm.visual_lm.continuation_training import layout_schedule
from ilm.visual_lm.glyph_joint_density import GlyphJointDensity
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.preflight_visual_continuation import runtime
from scripts.train_visual_continuation import atomic_json

NEW = ('ilm/visual_lm/glyph_joint_density.py', 'ilm/visual_lm/conditional_pixel_density.py',
       'ilm/visual_lm/joint_writer_training.py', 'scripts/prepare_joint_continuation_writer.py',
       'scripts/train_joint_continuation_writer.py', 'scripts/run_joint_continuation_writer.py',
       'scripts/audit_joint_continuation_writer.py',
       'tests/test_glyph_joint_density.py', 'tests/test_joint_writer_sequence_training.py',
       'references/continuation_joint_writer_protocol.md')


def sources():
    names = set(load(PARENT / 'preparation.json')['source_sha256']) | set(NEW)
    names |= {'scripts/diagnose_continuation_clean_history.py'}
    for name in names:
        require((ROOT/name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT),
                'Commit source before neural preparation: '+name)
    return {name:sha(ROOT/name) for name in sorted(names)}


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU-only registration')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve existing study')
    code = sources(); torch.set_num_threads(2)
    parent = load(PARENT / 'preparation.json'); manifest = load(PARENT/'training_manifest.json')
    rows = manifest['rows']; require(len(rows) == 13766, 'Complete original exclusions and corpus required')
    require(sum(r['target_glyphs'] for r in rows) == 2036247, 'Complete target exposure required')
    require(shutil.disk_usage(ROOT).free >= 120*1024**3, 'Need120GiB free disk before caching')
    final = load(PARENT/'fit/snapshots.json')['41298']
    require(sha(ROOT/final['file']) == final['sha256'], 'Final parent checkpoint changed')
    bindings = {}
    def bind(p):
        bindings[str(p.relative_to(ROOT))] = sha(p)
    for path in [PARENT/'preparation.json', PARENT/'training_manifest.json', PARENT/'fit/report.json',
                 PARENT/'fit/snapshots.json', ROOT/final['file'],
                 PARENT/'evaluation/report.json', PARENT/'review/frozen_readings.json',
                 PARENT/'review/summary.json', PARENT/'review/protocol_pilot_audit.json',
                 ROOT/'data/visual_sequence_corpus_v1/report.json',
                 ROOT/'artifacts/continuation_clean_history_v1/registration.json',
                 ROOT/'artifacts/continuation_clean_history_v1/targets.pt',
                 ROOT/'artifacts/continuation_clean_history_v1/execution/report.json',
                 ROOT/'references/evidence/continuation_clean_history_result_20261113.json']:
        bind(path)
    qualify = load(ROOT/'artifacts/continuation_clean_history_v1/registration.json')['jobs']
    for job in qualify:
        job['positions'] = selected_positions(job['target_glyphs'])
    require(len(qualify) == 124, 'All fixed conditional-writing cases required')
    schedule = layout_schedule(rows, 3, SEED)
    require(schedule.shape == (41298, 2), 'Three complete epochs required')
    for epoch in schedule.reshape(3,13766,2):
        require(np.array_equal(np.sort(epoch[:,0]),np.arange(13766)), 'Every source once per epoch required')
    require(np.array_equal(np.bincount(schedule[schedule[:,1]==1,0],minlength=13766),np.full(13766,2)),
            'Each source must use proportional layout twice')
    require(sum(rows[int(index)]['target_glyphs'] for index in schedule[:,0]) == 6108741,
            'Scheduled target exposure differs')
    RUN.mkdir(parents=True); np.save(RUN/'schedule.npy', schedule, allow_pickle=False)
    torch.manual_seed(SEED)
    initial=GlyphJointDensity(torch.zeros(4096),torch.ones(4096))
    atomic_tensor(RUN/'initial_parameters.pt',{k:v.detach().clone() for k,v in initial.named_parameters()})
    value = dict(schema='complete-joint-continuation-writer-v1', source_sha256=code,
        bindings_sha256=bindings, runtime=runtime(), parent_checkpoint=final,
        source_manifest=str((PARENT/'training_manifest.json').relative_to(ROOT)),
        training_cases=13766, training_counts=parent['counts'], complete_target_positions_per_layout=2036247,
        cache_training_records=27532, cache_target_predicting_states=4072494, cache_state_dtype='float32',
        schedule_sha256=sha(RUN/'schedule.npy'), initial_parameters_sha256=sha(RUN/'initial_parameters.pt'),
        seed=SEED, epochs=3, total_steps=41298,
        retained_target_exposure=6108741, snapshot_steps=[0, 10324, 20649, 30973, 41298],
        normalization='training-only equal-sequence fixed:proportional=1:2; std floor1e-4',
        objective='mean proper physical-mask NLL within each complete sequence; one source per update',
        microbatch_glyphs=64, trainable_head_parameters=2642368,
        learning_rate=2e-4, warmup=500, final_lr=2e-5, weight_decay=.01, gradient_clip=1.,
        checkpoint_interval=250, cache_seconds=86400, cache_pressure_seconds=7200,
        fit_seconds=43200, fit_pressure_seconds=3600, qualify_seconds=3600, qualify_pressure_seconds=300,
        audit_seconds=900, audit_peak_rss_gib=8,
        cache_peak_allocated_gib=22, head_peak_allocated_gib=8, runtime_host_reserve_gib=24,
        cache_entry_host_gib=48, head_entry_host_gib=24, free_device_entry_gib=20,
        cache_disk_limit_gib=100, minimum_free_disk_gib=48,
        qualification_jobs=qualify, physical_gate_readable_nonblank_fraction=.9,
        physical_gate_requires_all_groups=True, physical_outputs_reviewed_before_owner_unmask=True,
        evaluation_inputs=parent['evaluation_inputs'], maximum_core_glyphs=128, maximum_long_book_glyphs=640,
        new_candidate_streams_after_physical_gate=708, generation_seconds=28800,
        pilot=parent['pilot'], exact_retention=parent['exact_retention'],
        unchanged_control='Original complete final41298 candidate outputs; reuse original frozen scores explicitly',
        original_control_reexecuted=False, body_query_feedback_stop_frozen=True,
        prior_development_familiarity=True, independent_human_adjudication=False,
        untouched_confirmation_required=True, language_quality_unmeasured=True, goal_achieved=False)
    atomic_json(RUN/'registration.json', value); atomic_json(PUBLIC, value)
    print(json.dumps(dict(registration_sha256=sha(PUBLIC), train_sources=13766,
        cache_records=27532, target_exposure=6108741)), flush=True)


def validate():
    r = load(RUN/'registration.json')
    require(r['source_sha256'] == sources() and r['runtime'] == runtime(), 'Registered source/runtime changed')
    require(PUBLIC.read_bytes() == (RUN/'registration.json').read_bytes() ==
            subprocess.check_output(['git','show','HEAD:'+str(PUBLIC.relative_to(ROOT))],cwd=ROOT),
            'Commit registration before neural execution')
    for name, digest in r['bindings_sha256'].items():
        require(sha(ROOT/name) == digest, 'Bound corpus/model/control changed: '+name)
    require(sha(RUN/'schedule.npy') == r['schedule_sha256'], 'Complete schedule changed')
    require(sha(RUN/'initial_parameters.pt') == r['initial_parameters_sha256'], 'Registered fresh initialization changed')
    return r


if __name__ == '__main__':
    register()
