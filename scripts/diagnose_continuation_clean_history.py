#!/usr/bin/env python3
"""Initial/final full-model clean-history control, with long cache agreement."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, sha
from ilm.visual_lm.continuation_training import (TrainingImages, TrainingMonitor,
    build_actor, load, restore_trainable)
from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss, tensor_hash
from ilm.visual_lm.visual_continuation import prefill_visual_prompt
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.preflight_visual_continuation import runtime
from scripts.infer_glyph_tail_language import prepare_prompt
from scripts.score_pixel_language_first_blocks import require
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.train_visual_continuation import atomic_json

FIT = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
RUN = ROOT / 'artifacts/continuation_clean_history_v1'
PUBLIC = ROOT / 'references/evidence/continuation_clean_history_registration_20261113.json'
SELF = str(Path(__file__).resolve().relative_to(ROOT))
PLAN = 'references/continuation_clean_history_protocol.md'


def code_bindings():
    names = set(load(FIT / 'preparation.json')['source_sha256'])
    names.update([SELF, PLAN, 'ilm/visual_lm/continuation_training.py',
                  'ilm/visual_lm/visual_continuation.py', 'scripts/train_visual_continuation.py'])
    for name in names:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name]),
                'Commit diagnostic source before registration: ' + name)
    return {name: sha(ROOT / name) for name in sorted(names)}


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration with hidden CUDA')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve prior study')
    code = code_bindings(); torch.set_num_threads(2)
    prep = load(FIT / 'preparation.json'); rows = load(FIT / 'training_manifest.json')['rows']
    bindings = {}
    def bind(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for path in [FIT / 'preparation.json', FIT / 'training_manifest.json', FIT / 'fit/report.json',
                 FIT / 'evaluation/report.json', FIT / 'review/frozen_readings.json',
                 FIT / 'review/summary.json', ROOT / 'data/visual_sequence_corpus_v1/report.json',
                 ROOT / 'artifacts/compact_visual_span_evaluation_v1/references.json',
                 ROOT / 'artifacts/visual_continuation_trajectories_v1/report.json']:
        bind(path)
    snapshots = load(FIT / 'fit/snapshots.json')
    checkpoints = {k: snapshots[k] for k in ['0', '41298']}
    for value in checkpoints.values():
        require(sha(ROOT / value['file']) == value['sha256'], 'Checkpoint changed')
        bind(ROOT / value['file'])
    data = TrainingImages(FIT); jobs = []; gold = {}
    longest = max((i for i, r in enumerate(rows) if r['kind'] == 'book'),
                  key=lambda i: (rows[i]['target_glyphs'], rows[i]['id']))
    for kind in ['qa', 'book']:
        chosen = sorted((i for i, r in enumerate(rows) if r['kind'] == kind),
                        key=lambda i: __import__('hashlib').sha256(rows[i]['id'].encode()).hexdigest())[:12]
        if kind == 'book' and longest not in chosen:
            chosen[-1] = longest
        for index in chosen:
            for layout in [0, 1]:
                prompt, target, row, key = data.get(index, layout)
                name = f'train_{index:05}_{key}'
                job = dict(id=name, split='train', kind=kind, layout=key,
                           image=row['images'][key]['file'], source_id=row['id'],
                           long_agreement=index == longest and layout == 0)
                jobs.append(job); gold[name] = target; bind(ROOT / job['image'])
    corpus_path = ROOT / 'data/visual_sequence_corpus_v1'
    corpus = VisualSequenceCorpus(corpus_path, split='development',
                                  expected_report_sha256=prep['corpus_report_sha256'])
    source_indices = {}; offset = 0
    for shard in corpus.shards:
        path = corpus_path / next(n for n in shard['files'] if n.endswith('_sources.json'))
        require(sha(path) == shard['files'][path.name], 'Development source metadata changed')
        for i, row in enumerate(load(path)):
            require(row['id'] not in source_indices, 'Duplicate development source')
            source_indices[row['id']] = offset + i
        offset += shard['records']; bind(path)
    refs = {r['id']: r for r in load(ROOT /
        'artifacts/compact_visual_span_evaluation_v1/references.json')['reference_rows']}
    prior = {}
    for arm in load(FIT / 'evaluation/report.json')['arms']:
        if arm['arm'] == 'candidate_core128' and arm['step'] in [0, 41298]:
            for record in arm['records']:
                if record['input']['condition'] == 'normal':
                    path = FIT / 'evaluation' / f"step_{arm['step']:06}" / 'core128' / record['input']['id'] / 'output.pt'
                    require(sha(path) == record['tensor_sha256'], 'Original output changed')
                    bind(path)
                    prior[arm['step'], record['input']['id']] = str(path.relative_to(ROOT))
    for row in prep['evaluation_inputs']:
        if row['condition'] != 'normal':
            continue
        reference = refs[row['original_case'].removesuffix('_normal')]
        item = corpus[source_indices[reference['source_id']]]
        cut = item['prompt_glyphs'] if reference['kind'] == 'qa' else reference['prefix_glyphs']
        target = item['ink'][cut:].clone()
        require(len(target) == len(reference['answer']) > 0, 'Complete development target differs')
        name = row['id']; gold[name] = target; bind(ROOT / row['image'])
        jobs.append(dict(id=name, split='development', kind=reference['kind'], layout=row['layout'],
                         image=row['image'], source_id=reference['source_id'], long_agreement=False,
                         original_outputs={str(step): prior[step, name] for step in [0, 41298]}))
    require(len(jobs) == len(gold) == 124, 'Keep all prescribed124 jobs')
    for job in jobs:
        job.update(target_glyphs=len(gold[job['id']]), target_sha256=tensor_hash(gold[job['id']]))
    RUN.mkdir(parents=True); atomic_tensor(RUN / 'targets.pt', gold)
    r = dict(schema='continuation-clean-history-control-v1', source_sha256=code, bindings_sha256=bindings,
        checkpoints=checkpoints, jobs=jobs, targets_sha256=sha(RUN / 'targets.pt'), runtime=runtime(),
        new_optimizer_updates=0, new_autonomous_generations=0, supplied_reference_history=True,
        state_tolerance=.002, maximum_seconds=1800, pressure_wait_allowance_seconds=300,
        minimum_entry_host_ram_gib=48, minimum_runtime_host_ram_gib=24, minimum_free_vram_gib=20,
        maximum_allocated_vram_gib=22, dense_fraction=.9, later_fraction_f1_095=.9,
        autonomous_language_evidence=False, goal_achieved=False)
    atomic_json(RUN / 'registration.json', r); atomic_json(PUBLIC, r)
    print(json.dumps(dict(jobs=len(jobs), target_positions=sum(len(v) for v in gold.values()),
                          registration_sha256=sha(PUBLIC))), flush=True)


@torch.no_grad()
def predict(actor, prompt, ink, monitor, agreement=False):
    monitor.guard(); memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
    signed = (1 - 2 * ink.float()).to(actor.dynamics.query.device)
    feedback = actor.bridge.feedback_embedding(signed)[None]
    x = actor.dynamics.embeddings(feedback)
    parallel = actor.dynamics(x, memory, checkpoint_layers=False)
    logits, _, stop = actor.bridge.predict(parallel[0].float())
    result = dict(predicted_ink=(logits[:-1].sigmoid() >= .5).to(torch.uint8).cpu(),
        stop_probabilities=stop.sigmoid().float().cpu(),
        balanced_image_loss=float(reconstruction_loss(logits[:-1], signed)),
        first_state=parallel[0, 0].float().cpu(),
        geometry=geometry, supplied_reference_history=True, stopping_bypassed=True)
    if agreement:
        history = None; errors = []; changed = []; serial_stops = []
        for i in range(len(x[0])):
            monitor.position['boundary'] = i; monitor.guard()
            h, history = actor.dynamics(x[:, i:i+1], memory, checkpoint_layers=False,
                                        previous=history, return_history=True)
            errors.append(float((h - parallel[:, i:i+1]).abs().max()))
            other, _, end = actor.bridge.predict(h[:, -1].float())
            changed.append(float(((other.sigmoid() >= .5) != (logits[i:i+1].sigmoid() >= .5)).float().mean()))
            serial_stops.append(float(end.sigmoid()[0]))
        result['agreement'] = dict(maximum_absolute_state_error=max(errors), per_boundary_state_error=errors,
            per_boundary_changed_pixel_fraction=changed, serial_stop_probabilities=serial_stops)
    return result


def execute():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Physical GPU0 only')
    r = load(RUN / 'registration.json')
    require(r['source_sha256'] == code_bindings() and r['runtime'] == runtime(), 'Registered source/runtime differs')
    for name, digest in r['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound source or evidence changed: ' + name)
    require(sha(RUN / 'targets.pt') == r['targets_sha256'], 'Prepared target pixels changed')
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() ==
            subprocess.check_output(['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))]),
            'Commit registration before neural execution')
    out = RUN / 'execution'; out.mkdir(exist_ok=False)
    monitor = TrainingMonitor(out, r['maximum_seconds'], r['pressure_wait_allowance_seconds'])
    monitor.status('admitting'); hooks = []
    try:
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        from scripts.train_semantic_glyph_language import numerics
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance = build_actor(device, load(FIT / 'preparation.json')['seed']); actor.eval()
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard())
                     for layer in actor.interface.actor.context.layers]
            targets = torch.load(RUN / 'targets.pt', weights_only=True, map_location='cpu')
            records = []
            ordered = sorted(r['jobs'], key=lambda j: (not j['long_agreement'], j['id']))
            for step, snapshot in r['checkpoints'].items():
                checkpoint = torch.load(ROOT / snapshot['file'], weights_only=True, map_location='cpu')
                require(checkpoint['runtime'] == r['runtime'], 'Checkpoint runtime differs')
                restore_trainable(actor, checkpoint['trainable_weights']); del checkpoint
                versions = [(p, p._version) for p in actor.parameters()]
                for index, job in enumerate(ordered):
                    monitor.position = dict(step=int(step), completed=index, total=124, case=job['id'])
                    monitor.status('clean_history'); monitor.guard()
                    if job['split'] == 'development':
                        image, _ = prepare_prompt(ROOT / job['image'])
                        prompt = torch.from_numpy(np.asarray(image).copy()).float()[None, None]/127.5-1
                    else:
                        with Image.open(ROOT / job['image']) as image:
                            prompt = torch.from_numpy(np.asarray(image.convert('L')).copy()).float()[None, None]/127.5-1
                    target = targets[job['id']]
                    require(tensor_hash(target) == job['target_sha256'], 'Reference target changed')
                    value = predict(actor, prompt, target, monitor, job['long_agreement'])
                    directory = out / f'step_{int(step):06}' / job['id']; directory.mkdir(parents=True)
                    atomic_tensor(directory / 'predictions.pt', value)
                    record = dict(job=job, step=int(step), predictions_file=str((directory / 'predictions.pt').relative_to(ROOT)),
                                  predictions_sha256=sha(directory / 'predictions.pt'),
                                  balanced_image_loss=value['balanced_image_loss'])
                    if job['split'] == 'development':
                        original = torch.load(ROOT / job['original_outputs'][step], weights_only=True, map_location='cpu')
                        record['original_first_state_maximum_error'] = float((value['first_state'] - original['states'][0]).abs().max())
                        record['original_observer_geometry_equal'] = value['geometry'] == original['observer_geometry']
                        atomic_json(directory / 'report.json', record)
                        require(record['original_first_state_maximum_error'] <= r['state_tolerance'] and
                                record['original_observer_geometry_equal'], 'Original first state/geometry not reproduced')
                        del original
                    if job['long_agreement']:
                        record['agreement'] = value['agreement']
                        atomic_json(directory / 'report.json', record)
                        require(value['agreement']['maximum_absolute_state_error'] <= r['state_tolerance'],
                                'Long parallel/incremental state agreement fails; preserve predictions before repair')
                    records.append(record); atomic_json(directory / 'report.json', record)
                    require(all(p._version == version for p, version in versions), 'Frozen parameter changed')
                    print(json.dumps(dict(step=step, case=job['id'], targets=len(target),
                        image_loss=value['balanced_image_loss'], agreement=record.get('agreement', {}).get('maximum_absolute_state_error'))), flush=True)
                    del value, prompt; gc.collect()
            atomic_json(out / 'report.json', dict(complete=len(records) == 248, records=records,
                registration_sha256=sha(PUBLIC), provenance=provenance, new_optimizer_updates=0,
                new_autonomous_generations=0, autonomous_language_evidence=False,
                all_parameter_versions_unchanged=True, **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        for hook in hooks:
            hook.remove()
        status = load(out / 'status.json')
        atomic_json(out / 'terminal.json', status)
        atomic_json(out / 'status.json', dict(status, pid=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    args = parser.parse_args()
    (register if args.action == 'register' else execute)()
