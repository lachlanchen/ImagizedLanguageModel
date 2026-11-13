#!/usr/bin/env python3
"""Cache/train/qualify the full-corpus joint writer, with no live lexical teacher."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from ilm.visual_lm.joint_writer_training import (ROOT, PARENT, RUN, PUBLIC, SEED, load, sha, require,
    selected_positions, cache_directory, Monitor, previous_seconds, CachedSequences, update, condition_moments)
from ilm.visual_lm.continuation_training import (TrainingImages, build_actor, restore_trainable,
    rng_state, set_rng_state, same_nested, canonical_digest)
from ilm.visual_lm.visual_continuation import prefill_visual_prompt
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs, unpack_glyphs
from scripts.prepare_joint_continuation_writer import validate
from scripts.infer_glyph_tail_language import prepare_prompt
from scripts.score_pixel_language_first_blocks import owner_live
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu
from scripts.train_semantic_glyph_language import numerics
from scripts.train_visual_continuation import atomic_json


def begin(action, r, resume=False):
    folder = RUN/('fit' if action == 'train' else action)
    if (folder/'status.json').exists():
        require(not owner_live(load(folder/'status.json')), 'Original worker is still live')
        require(resume or (folder/'report.json').exists(), 'Preserve failed attempt; explicit manual resume required')
    folder.mkdir(exist_ok=True)
    key = 'fit' if action == 'train' else action
    remaining = r[key+'_seconds'] - previous_seconds(action)
    require(remaining > 0, 'Cumulative stage budget exhausted')
    old_wait = 0.
    for path in (folder/'attempts').glob('*.json') if (folder/'attempts').exists() else []:
        old_wait += load(path).get('resource_wait_seconds', 0.)
    pressure = r[key+'_pressure_seconds'] - old_wait
    require(pressure >= 0, 'Cumulative pressure allowance exhausted')
    monitor = Monitor(folder, remaining, pressure, 22 if action == 'cache' else 8)
    monitor.status('admitting')
    return folder, monitor


def finish(folder, monitor, error=None):
    monitor.status('failed' if error else 'complete', **({'error':repr(error)} if error else {}))
    receipt = dict(load(folder/'status.json'), **monitor.receipt())
    (folder/'attempts').mkdir(exist_ok=True)
    atomic_json(folder/'attempts'/f'{time.time_ns()}.json', receipt)
    atomic_json(folder/'status.json', dict(receipt, pid=None))


def frozen_parent(device, r):
    meta = load(PARENT/'preparation.json')
    actor, provenance = build_actor(device, meta['seed'])
    snapshot = torch.load(ROOT/r['parent_checkpoint']['file'], weights_only=True, map_location='cpu')
    require(snapshot['step'] == 41298 and snapshot['runtime'] == r['runtime'], 'Exact final reader required')
    restore_trainable(actor, snapshot['trainable_weights'])
    actor.requires_grad_(False).eval()
    return actor, provenance


@torch.no_grad()
def cache_one(actor, prompt, gold, directory, identity, monitor, cached):
    if directory.exists():
        states, patterns, record = cached.get(directory)
        require(record['identity'] == identity and record['gold_sha256'] == tensor_hash(gold), 'Cached case differs')
        require(torch.equal(unpack_glyphs(patterns.long()), gold), 'Cached targets differ')
        return states, record
    monitor.guard()
    memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
    signed = (1-2*gold.float()).to(actor.dynamics.query.device)
    feedback = actor.bridge.feedback_embedding(signed)[None]
    states = actor.dynamics(actor.dynamics.embeddings(feedback), memory,
                           checkpoint_layers=False)[0].float().cpu().contiguous()
    patterns = pack_glyphs(gold)
    require(len(states) == len(gold)+1 and torch.isfinite(states).all(), 'Complete finite causal states required')
    require(torch.equal(unpack_glyphs(patterns), gold), 'Source pixels did not pack losslessly')
    directory.parent.mkdir(parents=True, exist_ok=True)
    partial = directory.with_name('.'+directory.name+f'.{time.time_ns()}.partial'); partial.mkdir()
    np.save(partial/'states.npy', states.numpy(), allow_pickle=False)
    np.save(partial/'patterns.npy', patterns.to(torch.uint8).numpy(), allow_pickle=False)
    record = dict(identity=identity, glyphs=len(gold), gold_sha256=tensor_hash(gold), geometry=geometry,
        states_sha256=sha(partial/'states.npy'), patterns_sha256=sha(partial/'patterns.npy'),
        array_bytes=sum((partial/n).stat().st_size for n in ['states.npy', 'patterns.npy']),
        causal_predictor_before_current_target=True, terminal_has_no_image_target=True,
        state_dtype='float32', target_packing_lossless=True)
    atomic_json(partial/'record.json', record); partial.rename(directory)
    return states, record


def cache(r, resume):
    folder, monitor = begin('cache', r, resume); hooks = []; error = None
    try:
        require(not (folder/'report.json').exists(), 'Cache already complete; use existing authenticated data')
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance = frozen_parent(device, r)
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard())
                     for layer in actor.interface.actor.context.layers]
            versions = [(p, p._version) for p in actor.parameters()]
            images = TrainingImages(PARENT); cached = CachedSequences()
            first = torch.zeros(4096, dtype=torch.float64); second = first.clone()
            records = []; normalization_weight = 0; positions = 0; generated_bytes = 0
            for index, row in enumerate(images.rows):
                for layout in [0, 1]:
                    prompt, gold, _, key = images.get(index, layout)
                    monitor.position = dict(completed=len(records), total=27532, source=index, layout=key)
                    if len(records) % 25 == 0:
                        monitor.status('caching_training_predictors')
                    monitor.guard()
                    identity = dict(split='train', source_index=index, source_id=row['id'], kind=row['kind'],
                                    layout=key, input_sha256=row['images'][key]['sha256'])
                    directory = cache_directory(index, key)
                    states, record = cache_one(actor, prompt, gold, directory, identity, monitor, cached)
                    mean, square = condition_moments(states); weight = 1 if layout == 0 else 2
                    first += weight*mean; second += weight*square; normalization_weight += weight
                    positions += len(gold); generated_bytes += record['array_bytes']
                    require(generated_bytes <= 100*1024**3, 'Cache storage ceiling reached')
                    records.append(dict(directory=str(directory.relative_to(ROOT)), record_sha256=sha(directory/'record.json'),
                                        glyphs=len(gold), array_bytes=record['array_bytes']))
                    if len(records) % 100 == 0:
                        require(all(p._version == v for p,v in versions), 'Frozen reader changed')
                        print(json.dumps(dict(stage='cache', completed=len(records), total=27532,
                            target_states=positions, array_gib=generated_bytes/1024**3)), flush=True)
                    del states, gold, prompt
            require(len(records) == 27532 and positions == 4072494 and normalization_weight == 41298,
                    'Complete both-layout training cache required')
            mean = first/normalization_weight
            scale = (second/normalization_weight-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
            require(torch.isfinite(mean).all() and torch.isfinite(scale).all(), 'Finite training-only normalization required')
            atomic_tensor(folder/'normalization.pt', dict(mean=mean.float(), scale=scale.float(),
                source_sequences=13766, layouts='fixed1:proportional2', target_states=positions))
            golds = torch.load(ROOT/'artifacts/continuation_clean_history_v1/targets.pt', weights_only=True, map_location='cpu')
            qualification = []
            for i, job in enumerate(r['qualification_jobs']):
                monitor.position = dict(completed=i, total=124, case=job['id']); monitor.status('caching_qualification_only')
                if job['split'] == 'train':
                    directory = cache_directory(int(job['id'].split('_')[1]), job['layout'])
                    states, _, record = cached.get(directory)
                else:
                    directory = folder/'qualification'/job['id']
                    canvas, _ = prepare_prompt(ROOT/job['image'])
                    prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                    identity = dict(split='development', case=job['id'], source_id=job['source_id'],
                        kind=job['kind'], layout=job['layout'], input_sha256=sha(ROOT/job['image']))
                    states, record = cache_one(actor, prompt, golds[job['id']], directory, identity, monitor, cached)
                    old = torch.load(ROOT/job['original_outputs']['41298'], weights_only=True, map_location='cpu')
                    require(record['geometry'] == old['observer_geometry'] and
                            (states[0]-old['states'][0]).abs().max() <= .002, 'Original reader prefill not reproduced')
                    generated_bytes += record['array_bytes']
                require(record['gold_sha256'] == job['target_sha256'], 'Qualification target binding differs')
                qualification.append(dict(job=job, directory=str(directory.relative_to(ROOT)),
                                          record_sha256=sha(directory/'record.json')))
                del states
            require(all(p._version == v for p,v in versions), 'Frozen reader parameter changed')
            require(generated_bytes <= 100*1024**3, 'Cache storage ceiling reached')
            atomic_json(folder/'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), records=records,
                qualification=qualification, training_records=27532, target_states=positions, array_bytes=generated_bytes,
                normalization_sha256=sha(folder/'normalization.pt'), provenance=provenance,
                training_sources=13766, development_in_fit_or_normalization=False, optimizer_updates=0,
                frozen_parameter_versions_unchanged=True, **monitor.receipt()))
    except BaseException as e:
        error = e; raise
    finally:
        for hook in hooks:
            hook.remove()
        finish(folder, monitor, error)


def checkpoint(model, optimizer, step, r, chain):
    return dict(schema='joint-continuation-head-checkpoint-v1', step=step,
        registration_sha256=sha(PUBLIC), cache_report_sha256=sha(RUN/'cache/report.json'),
        parent_checkpoint=r['parent_checkpoint'], source_sha256=r['source_sha256'], runtime=r['runtime'],
        model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()), rng=rng_state(), ledger_sha256=chain,
        completed_training=step == r['total_steps'], only_head_trainable=True)


def restore(model, optimizer, value):
    model.load_state_dict(value['model']); optimizer.load_state_dict(value['optimizer']); set_rng_state(value['rng'])


@torch.no_grad()
def neural_checks(model, states, patterns):
    indices = sorted({0, len(patterns)//2, len(patterns)-1})
    device = next(model.parameters()).device
    h = states[indices].to(device); target = patterns[indices].to(device=device, dtype=torch.long)
    full = model(h, target); condition = model.condition_features(h); history = None; values = []
    for j in range(128):
        one, history = model.step(condition, target[:,j-1] if j else None, j, history)
        values.append(one)
    serial = torch.stack(values, 1)
    tv = float((.5*(serial.softmax(-1)-full.softmax(-1)).abs().sum(-1)).max())
    generated = model.generate(h); replay = model(h, generated).argmax(-1)
    result = dict(probability_total_variation_maximum=tv,
        raw_logit_absolute_maximum=float((serial-full).abs().max()),
        generated_prefix_greedy_choices_exact=bool(torch.equal(generated,replay)), states=len(indices))
    require(tv <= 1e-4 and result['generated_prefix_greedy_choices_exact'], 'Physical head cache agreement failed')
    return result


def train(r, resume):
    folder, monitor = begin('train', r, resume); error = None; ledger = None
    try:
        meta = load(RUN/'cache/report.json'); require(meta['complete'], 'Complete full corpus cache first')
        require(sha(RUN/'cache/normalization.pt') == meta['normalization_sha256'], 'Normalization changed')
        require(not (folder/'report.json').exists(), 'Training already complete')
        with bridge_gpu(dict(minimum_entry_ram_gib=24.)) as (device, admission):
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            norm = torch.load(RUN/'cache/normalization.pt', weights_only=True, map_location='cpu')
            model = GlyphJointDensity(norm['mean'], norm['scale']).to(device).train()
            require(sum(p.numel() for p in model.parameters()) == 2642368, 'Exact registered head required')
            initial=torch.load(RUN/'initial_parameters.pt',weights_only=True,map_location='cpu')
            require(all(torch.equal(p.detach().cpu(),initial[name]) for name,p in model.named_parameters()),
                    'Head does not start from the registered fresh draw')
            del initial
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.01,
                betas=(.9,.999), eps=1e-8, foreach=False)
            schedule = np.load(RUN/'schedule.npy', allow_pickle=False)
            cases = CachedSequences(); start = 0; chain = '0'*64
            snapshots = {}; snapshot_index = folder/'snapshots.json'
            if resume:
                pointer = load(folder/'resume.json'); require(sha(ROOT/pointer['file']) == pointer['sha256'], 'Resume file changed')
                value = torch.load(ROOT/pointer['file'], weights_only=True, map_location='cpu')
                require(value['registration_sha256'] == sha(PUBLIC) and value['cache_report_sha256'] == sha(RUN/'cache/report.json'), 'Resume binding differs')
                restore(model, optimizer, value); start = value['step']; chain = value['ledger_sha256']
                snapshots = load(snapshot_index)
                require(not any(int(k) > start for k in snapshots), 'Preserve abandoned later snapshots before manual resume')
            attempt = len(list(folder.glob('ledger_*.jsonl')))+1
            ledger = (folder/f'ledger_{attempt:03}.jsonl').open('x', buffering=1)
            def emit(event):
                nonlocal chain
                event = dict(event, previous_event_sha256=chain)
                chain = canonical_digest(event)
                ledger.write(json.dumps(dict(event, event_sha256=chain))+'\n')
            def snapshot(step):
                path = folder/'snapshots'/f'step_{step:06}.pt'; path.parent.mkdir(exist_ok=True)
                require(not path.exists(), 'Preserve existing fixed snapshot')
                atomic_tensor(path, checkpoint(model, optimizer, step, r, chain))
                snapshots[str(step)] = dict(file=str(path.relative_to(ROOT)), sha256=sha(path), step=step)
                atomic_json(snapshot_index, snapshots)
            if start == 0:
                require(not resume, 'Nonzero checkpoint required for resumed execution')
                snapshot(0)
            resumed_test = load(folder/'resume_control.json') if (folder/'resume_control.json').exists() else None
            control_events = {}; checks = []
            if start == 0:
                j, style = map(int, schedule[0]); style = 'fixed_cells' if style == 0 else 'proportional'
                a, b, _ = cases.get(cache_directory(j, style))
                checks.append(dict(step=0, **neural_checks(model, a, b)))
                atomic_json(folder/'numerical_checks.json', checks)
            for step in range(start+1, r['total_steps']+1):
                index, layout = map(int, schedule[step-1]); key = 'fixed_cells' if layout == 0 else 'proportional'
                monitor.position = dict(step=step, total=r['total_steps'], source=index, layout=key)
                if step % 25 == 1:
                    monitor.status('training_complete_sequences')
                states, patterns, record = cases.get(cache_directory(index, key))
                event = update(model, optimizer, states, patterns, step, r['total_steps'], monitor)
                emit(dict(event, event_kind='scheduled', source_index=index, layout=key,
                          source_id=record['identity']['source_id']))
                if step in [4,5]:
                    control_events[step] = event.copy()
                if step == 3:
                    atomic_tensor(folder/'resume_control3.pt', checkpoint(model, optimizer, step, r, chain))
                if step == 5 and resumed_test is None:
                    uninterrupted = checkpoint(model, optimizer, 5, r, chain)
                    atomic_tensor(folder/'resume_control5.pt', uninterrupted)
                    value = torch.load(folder/'resume_control3.pt', weights_only=True, map_location='cpu')
                    restore(model, optimizer, value); replay_events = []
                    for replay_step in [4,5]:
                        j, style = map(int, schedule[replay_step-1]); style = 'fixed_cells' if style == 0 else 'proportional'
                        a, b, _ = cases.get(cache_directory(j, style))
                        replay_events.append(update(model, optimizer, a, b, replay_step, r['total_steps'], monitor))
                    require(replay_events == [control_events[4],control_events[5]], 'Resume losses or gradient norms differ')
                    actual = checkpoint(model, optimizer, 5, r, chain)
                    for name in ['model','optimizer','rng']:
                        same_nested(actual[name], uninterrupted[name])
                    resumed_test = dict(passed=True, restored_step=3, replayed_steps=[4,5], extra_executed_updates=2,
                        exact_model_optimizer_rng=True, source_file_sha256=sha(folder/'resume_control3.pt'),
                        uninterrupted_file_sha256=sha(folder/'resume_control5.pt'))
                    atomic_json(folder/'resume_control.json', resumed_test)
                    for replay in replay_events:
                        emit(dict(replay, event_kind='discarded_resume_verification'))
                    del uninterrupted, value, actual
                if step in [5, r['total_steps']]:
                    checks.append(dict(step=step, **neural_checks(model, states, patterns)))
                    atomic_json(folder/'numerical_checks.json', checks)
                if step in r['snapshot_steps']:
                    snapshot(step)
                if step % r['checkpoint_interval'] == 0 or step == r['total_steps']:
                    ledger.flush(); os.fsync(ledger.fileno())
                    path = folder/'current.pt'; previous = folder/'previous.pt'
                    if previous.exists():
                        previous.unlink()
                    if path.exists():
                        os.link(path, previous)
                    atomic_tensor(path, checkpoint(model, optimizer, step, r, chain))
                    atomic_json(folder/'resume.json', dict(file=str(path.relative_to(ROOT)), sha256=sha(path), step=step))
                if step % 100 == 0:
                    print(json.dumps(dict(stage='fit', step=step, nll=event['nll'], **monitor.receipt())), flush=True)
                del states, patterns
            require(resumed_test and resumed_test['passed'], 'Actual optimizer/RNG resume control required')
            counts = [int(v['step'].item()) for v in optimizer.state.values()]
            require(len(counts)==len(list(model.parameters())) and set(counts) == {41298},
                    'Every actual Adam counter must reach the registered final step')
            atomic_json(folder/'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), snapshots=snapshots,
                retained_updates=41298, extra_executed_resume_updates=2, training_cases=13766, epochs=3,
                target_exposure=6108741, parent_unchanged=True, head_parameters=2642368,
                optimizer_counters=counts, ledger_final_sha256=chain, cache_report_sha256=sha(RUN/'cache/report.json'),
                language_quality_unmeasured=True, **monitor.receipt()))
    except BaseException as e:
        error = e; raise
    finally:
        if ledger is not None:
            ledger.close()
        finish(folder, monitor, error)


def qualify(r, resume):
    folder, monitor = begin('qualify', r, resume); error = None
    try:
        require(load(RUN/'fit/report.json')['complete'] and load(RUN/'audit/report.json')['complete'],
                'Complete full-corpus fit and independent audit first')
        require(not (folder/'report.json').exists(), 'Qualification already complete')
        require(not list(folder.glob('cases/*')), 'Preserve partial qualification before a manual recovery')
        meta = load(RUN/'cache/report.json'); old_report = load(ROOT/'artifacts/continuation_clean_history_v1/execution/report.json')
        parents = {x['job']['id']:x for x in old_report['records'] if x['step'] == 41298}
        mapping = {}; records = []
        with bridge_gpu(dict(minimum_entry_ram_gib=24.)) as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            norm = torch.load(RUN/'cache/normalization.pt', weights_only=True, map_location='cpu')
            model = GlyphJointDensity(norm['mean'], norm['scale']).to(device).eval()
            snapshot = load(RUN/'fit/snapshots.json')['41298']; require(sha(ROOT/snapshot['file']) == snapshot['sha256'], 'Final head changed')
            value = torch.load(ROOT/snapshot['file'], weights_only=True, map_location='cpu')
            model.load_state_dict(value['model']); del value
            cases = CachedSequences()
            for i, item in enumerate(meta['qualification']):
                job = item['job']; monitor.position = dict(completed=i, total=124, case=job['id'])
                monitor.status('conditional_physical_writing'); monitor.guard()
                states, patterns, _ = cases.get(ROOT/item['directory'])
                output = []; nll = []
                for start in range(0,len(patterns),64):
                    monitor.guard(); h = states[start:start+min(64,len(patterns)-start)].to(device)
                    output.append(model.write(h).cpu())
                    target = patterns[start:start+64].to(device=device, dtype=torch.long)
                    nll.extend((-model(h,target).log_softmax(-1).gather(-1,target[...,None]).squeeze(-1).mean(1)).cpu().tolist())
                ink = torch.cat(output); gold = unpack_glyphs(patterns.long())
                parent_record = parents[job['id']]; parent_path = ROOT/parent_record['predictions_file']
                require(sha(parent_path) == parent_record['predictions_sha256'], 'Parent conditional pixels changed')
                parent = torch.load(parent_path, weights_only=True, map_location='cpu')['predicted_ink']
                require(parent.shape == ink.shape == gold.shape, 'Keep every complete target position')
                directory = folder/'cases'/job['id']; directory.mkdir(parents=True)
                np.savez_compressed(directory/'pixels.npz', candidate_ink=ink.numpy(), parent_ink=parent.numpy(), gold_ink=gold.numpy())
                a = ink.numpy().reshape(len(ink),-1); b = gold.numpy().reshape(len(gold),-1)
                numerator = 2*(a*b).sum(1); denominator = a.sum(1)+b.sum(1)
                f1 = np.divide(numerator,denominator,out=np.ones(len(a)),where=denominator>0)
                record = dict(job=job, pixels_sha256=sha(directory/'pixels.npz'), target_glyphs=len(ink),
                    pattern_nll_mean=float(np.mean(nll)), canonical_f1_mean=float(f1.mean()),
                    canonical_f1_per_position=f1.tolist(), canonical_exact=int(np.all(a==b,axis=1).sum()),
                    supplied_reference_history=True, autonomous_language_evidence=False)
                atomic_json(directory/'report.json',record); records.append(record)
                for role, tiles in [('candidate',ink),('parent',parent)]:
                    for position in job['positions']:
                        identity = hashlib.sha256(f'{sha(PUBLIC)}/{role}/{job["id"]}/{position}'.encode()).hexdigest()[:20]
                        require(identity not in mapping, 'Opaque physical ID collision')
                        mapping[identity] = dict(role=role, case=job['id'], split=job['split'], kind=job['kind'],
                            layout=job['layout'], position=position, pixels_sha256=record['pixels_sha256'])
                        pages = folder/'pages'; pages.mkdir(exist_ok=True)
                        Image.fromarray(255-255*tiles[position,0].numpy()).save(pages/(identity+'.png'))
                        mapping[identity]['image_sha256']=sha(pages/(identity+'.png'))
                print(json.dumps(dict(case=job['id'], generated_glyphs=len(ink))), flush=True)
            atomic_json(folder/'mapping.json', dict(registration_sha256=sha(PUBLIC), rows=mapping))
            identifiers = sorted(mapping); boards = []
            for start in range(0,len(identifiers),64):
                ids = identifiers[start:start+64]; canvas = Image.new('L',(1120,848),255); draw = ImageDraw.Draw(canvas)
                for j,key in enumerate(ids):
                    x=(j%8)*140; y=(j//8)*106
                    image=Image.open(folder/'pages'/(key+'.png')).resize((64,64),Image.Resampling.NEAREST)
                    canvas.paste(image,(x+36,y+4)); draw.text((x+5,y+76),key,fill=0)
                path=folder/f'board_{start//64+1:03}.png';canvas.save(path)
                boards.append(dict(file=path.name,sha256=sha(path),ids=ids))
            atomic_json(folder/'readings_template.json', dict(readings={key:dict(text=None,uncertain=None,blank=None) for key in identifiers},
                all_boards_actually_viewed=False, owner_mapping_opened=False))
            atomic_json(folder/'boards.json',dict(boards=boards,images=len(mapping),all_outputs_reviewed=False))
            atomic_json(folder/'report.json',dict(complete=True,registration_sha256=sha(PUBLIC),records=records,
                mapping_sha256=sha(folder/'mapping.json'),boards_sha256=sha(folder/'boards.json'),
                all124_complete=True, new_optimizer_updates=0, new_autonomous_answers=0,
                physical_gate_unmeasured=True, language_quality_unmeasured=True, **monitor.receipt()))
    except BaseException as e:
        error=e;raise
    finally:
        finish(folder,monitor,error)


def score_physical(readings_path):
    folder=RUN/'qualify';report=load(folder/'report.json');receipt=load(readings_path)
    require(report['complete'] and receipt.get('all_boards_actually_viewed') is True, 'Inspect every physical output first')
    require(receipt.get('mapping_sha256')==sha(folder/'mapping.json'), 'Freeze readings against actual mapping')
    require(receipt.get('boards_sha256')==sha(folder/'boards.json'), 'Freeze readings against actual boards')
    require(not (folder/'frozen_readings.json').exists(), 'Preserve frozen readings')
    data=load(folder/'mapping.json')['rows'];readings=receipt['readings'];require(set(data)==set(readings),'Retain all selected tiles')
    boards=load(folder/'boards.json')['boards']
    require({k for b in boards for k in b['ids']}==set(data),'Every physical page must be displayed')
    for b in boards:
        require(sha(folder/b['file'])==b['sha256'],'Reviewed board changed')
    for key, v in readings.items():
        require(isinstance(v['text'],str) and type(v['uncertain']) is bool and type(v['blank']) is bool,'Complete readings required')
        require(sha(folder/'pages'/(key+'.png'))==data[key]['image_sha256'],'Reviewed pixels changed')
    atomic_json(folder/'frozen_readings.json', receipt)
    from collections import defaultdict
    groups=defaultdict(list)
    for key,row in data.items():
        v=readings[key];groups[row['role'],row['split'],row['kind'],row['layout']].append(
            bool(v['text'].strip()) and not v['uncertain'] and not v['blank'])
    results=[dict(zip(['role','split','kind','layout'],k),count=len(v),readable_nonblank=sum(v),fraction=sum(v)/len(v))
             for k,v in sorted(groups.items())]
    gate=all(x['fraction']>=.9 for x in results if x['role']=='candidate')
    atomic_json(folder/'gate.json',dict(passed=gate,groups=results,frozen_readings_sha256=sha(folder/'frozen_readings.json'),
        physical_only=True,canonical_match_not_required=True,complete_language_unmeasured=True,goal_achieved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['cache','train','qualify','score-physical'])
    p.add_argument('--resume',action='store_true');p.add_argument('--readings',type=Path);a=p.parse_args()
    if a.action=='score-physical':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU review scoring only');validate();score_physical(a.readings)
    else:
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8',
                'Set physical GPU0 and deterministic CUBLAS before launch')
        r=validate();{'cache':cache,'train':train,'qualify':qualify}[a.action](r,a.resume)
