#!/usr/bin/env python3
"""Train complete joint image/context likelihood, then qualify physical writing."""
import time
BEGAN = time.perf_counter()

import argparse
import copy
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw
import torch

from ilm.visual_lm.joint_context_training import (ROOT, RUN, PUBLIC, CONTROL, PARENT,
    load, sha, require, tensor_inventory, check_checkpoint, previous_seconds, require_process_runtime)
from ilm.visual_lm.joint_context_model import build_joint_context_actor, differentiable_states
from ilm.visual_lm.joint_continuation_learning import backward_complete_image_likelihood
from ilm.visual_lm.continuation_training import (TrainingImages, trainable_state, restore_trainable,
    rng_state, set_rng_state, same_nested, canonical_digest, learning_rate)
from ilm.visual_lm.joint_writer_training import Monitor
from ilm.visual_lm.glyph_joint_density import pack_glyphs
from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from ilm.visual_lm.visual_continuation import prefill_visual_prompt
from scripts.prepare_joint_context_training import validate
from scripts.score_pixel_language_first_blocks import owner_live
from scripts.train_semantic_glyph_language import numerics
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.train_visual_continuation import atomic_json
from scripts.infer_glyph_tail_language import prepare_prompt


def begin(action, r, resume):
    folder = RUN / ('fit' if action == 'train' else action)
    require(not (folder / 'report.json').exists(), 'Completed stage must not repeat')
    if (folder / 'status.json').exists():
        prior = load(folder / 'status.json')
        require(not owner_live(prior) and resume, 'Preserve prior attempt; manual resume required')
        require(prior.get('pid') is None, 'Reconcile unterminated worker and pressure cost before recovery')
    else:
        require(not resume, 'No prior stage to resume')
    folder.mkdir(exist_ok=True)
    key = 'fit' if action == 'train' else action
    seconds = r[key + '_seconds'] - previous_seconds(action)
    waited = sum(load(p)['resource_wait_seconds'] for p in (folder / 'attempts').glob('*.json'))
    pressure = r[key + '_pressure_seconds'] - waited
    require(seconds > 0 and pressure >= 0, 'Cumulative stage allowance exhausted')
    monitor = Monitor(folder, seconds, pressure, r['maximum_allocated_vram_gib'])
    monitor.began = BEGAN  # Imports, validation and setup use the same stage ceiling.
    monitor.status('admitting')
    return folder, monitor


def finish(folder, monitor, error):
    receipt = dict(complete=error is None and (folder / 'report.json').exists(),
                   error=repr(error) if error else None, **monitor.receipt())
    monitor.status('failed' if error else 'complete', error=repr(error) if error else None)
    receipt.update(load(folder / 'status.json'))
    (folder / 'attempts').mkdir(exist_ok=True)
    atomic_json(folder / 'attempts' / f'{time.time_ns()}.json', receipt)
    atomic_json(folder / 'status.json', dict(receipt, pid=None))


def frozen_checker(actor):
    versions = [(p, p._version) for p in actor.parameters() if not p.requires_grad]
    names = {'semantic_mean', 'semantic_scale', 'density.condition_mean', 'density.condition_scale'}
    buffers = {n: b.detach().clone() for n, b in actor.bridge.named_buffers() if n in names}
    require(len(buffers) == 4, 'Four fixed affine buffers required')
    def verify():
        require(all(p._version == v and p.grad is None for p, v in versions), 'Frozen parameter changed/received gradient')
        current = dict(actor.bridge.named_buffers())
        require(all(torch.equal(current[n], b) for n, b in buffers.items()), 'Frozen affine buffer changed')
    return verify


def checkpoint(actor, optimizer, step, r, chain):
    return dict(schema='joint-context-checkpoint-v1', step=step, registration_sha256=sha(PUBLIC),
        source_sha256=r['source_sha256'], runtime=r['runtime'], parent_checkpoint=r['parent_checkpoint'],
        trainable_weights=trainable_state(actor), optimizer=copy.deepcopy(optimizer.state_dict()),
        rng=rng_state(), ledger_sha256=chain, completed_training=step == r['total_steps'],
        normalization_sha256=sha(CONTROL / 'cache/normalization.pt'), only_context_and_density_trainable=True)


def restore(actor, optimizer, value, r):
    check_checkpoint(value, r)
    restore_trainable(actor, value['trainable_weights'])
    optimizer.load_state_dict(value['optimizer']); set_rng_state(value['rng'])


def update(actor, optimizer, data, schedule, position, r, monitor):
    began = time.perf_counter()
    index, style = map(int, schedule[position - 1])
    prompt, ink, row, layout = data.get(index, style)
    monitor.position = dict(step=position, total=r['total_steps'], source=index, layout=layout,
                            source_id=row['id'], target_glyphs=len(ink))
    monitor.guard(); optimizer.zero_grad(set_to_none=True)
    states, memory, _ = differentiable_states(actor, prompt, ink)
    event = backward_complete_image_likelihood(states, actor.bridge.density, actor.bridge.stop, ink,
        microbatch_glyphs=r['microbatch_glyphs'], guard=monitor.guard)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    require(len(parameters) == 131 and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in parameters), 'All actual context/head gradients must be finite and present')
    norm = torch.nn.utils.clip_grad_norm_(parameters, r['gradient_clip'], error_if_nonfinite=True)
    lr = learning_rate(position, r['total_steps'], peak=r['learning_rate'], warmup=r['warmup'])
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step(); monitor.guard(); torch.cuda.synchronize()
    event.update(step=position, source_index=index, source_id=row['id'], kind=row['kind'], layout=layout,
        target_sha256=row['target_sha256'], input_sha256=row['images'][layout]['sha256'],
        learning_rate=lr, gradient_norm_before_clip=float(norm), elapsed_seconds=time.perf_counter() - began)
    del states, memory, prompt, ink
    return event


def comparable(event):
    return {k: v for k, v in event.items() if k != 'elapsed_seconds'}


def train(r, resume):
    folder, monitor = begin('train', r, resume); hooks = []; error = None; ledger = None
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance, account = build_joint_context_actor(device)
            require(account == r['model_account'], 'Actual complete model differs')
            require(tensor_inventory(trainable_state(actor)) == r['initial_trainable_inventory'], 'Fresh initialization differs')
            frozen_ok = frozen_checker(actor)
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
            parameters = [p for p in actor.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=r['learning_rate'], weight_decay=r['weight_decay'],
                                         betas=(.9, .999), eps=1e-8, foreach=False)
            data = TrainingImages(PARENT); schedule = np.load(ROOT / r['schedule_file'], allow_pickle=False)
            chain = '0' * 64; start = 0; snapshots = {}
            if resume:
                pointer = load(folder / 'resume.json'); require(sha(ROOT / pointer['file']) == pointer['sha256'], 'Resume file changed')
                value = torch.load(ROOT / pointer['file'], weights_only=True, map_location='cpu')
                restore(actor, optimizer, value, r); start = value['step']; chain = value['ledger_sha256']
                snapshots = load(folder / 'snapshots.json')
                require(not any(int(k) > start for k in snapshots), 'Preserve abandoned later fixed snapshots before recovery')
                del value
            ledger = (folder / f'ledger_{len(list(folder.glob("ledger_*.jsonl"))) + 1:03}.jsonl').open('x', buffering=1)
            def emit(event):
                nonlocal chain
                event = dict(event, previous_event_sha256=chain); chain = canonical_digest(event)
                ledger.write(json.dumps(dict(event, event_sha256=chain)) + '\n')
            def save_snapshot(step):
                frozen_ok(); path = folder / 'snapshots' / f'step_{step:06}.pt'; path.parent.mkdir(exist_ok=True)
                require(not path.exists(), 'Preserve fixed snapshot')
                value = checkpoint(actor, optimizer, step, r, chain); check_checkpoint(value, r, step)
                atomic_tensor(path, value)
                snapshots[str(step)] = dict(file=str(path.relative_to(ROOT)), sha256=sha(path), step=step)
                atomic_json(folder / 'snapshots.json', snapshots)
            if not resume:
                save_snapshot(0)
            control = load(folder / 'resume_control.json') if (folder / 'resume_control.json').exists() else None
            control_events = {}
            for step in range(start + 1, r['total_steps'] + 1):
                event = update(actor, optimizer, data, schedule, step, r, monitor)
                emit(dict(event, event_kind='scheduled'))
                if step in [4, 5]:
                    control_events[step] = comparable(event)
                if step == 3:
                    atomic_tensor(folder / 'resume_control3.pt', checkpoint(actor, optimizer, step, r, chain))
                if step == 5 and control is None:
                    uninterrupted = checkpoint(actor, optimizer, step, r, chain)
                    atomic_tensor(folder / 'resume_control5.pt', uninterrupted)
                    saved = torch.load(folder / 'resume_control3.pt', weights_only=True, map_location='cpu')
                    restore(actor, optimizer, saved, r); replays = []
                    for number in [4, 5]:
                        replays.append(update(actor, optimizer, data, schedule, number, r, monitor))
                    require([comparable(e) for e in replays] == [control_events[4], control_events[5]], 'Resume losses/gradients differ')
                    actual = checkpoint(actor, optimizer, step, r, chain)
                    for key in ['trainable_weights', 'optimizer', 'rng']:
                        same_nested(actual[key], uninterrupted[key])
                    atomic_tensor(folder / 'resume_replayed5.pt', actual)
                    control = dict(passed=True, exact_model_optimizer_rng=True, restored_step=3, replayed_steps=[4, 5],
                        extra_executed_updates=2, source_file_sha256=sha(folder / 'resume_control3.pt'),
                        uninterrupted_file_sha256=sha(folder / 'resume_control5.pt'),
                        replayed_file_sha256=sha(folder / 'resume_replayed5.pt'))
                    atomic_json(folder / 'resume_control.json', control)
                    for replay in replays:
                        emit(dict(replay, event_kind='discarded_resume_verification'))
                    del uninterrupted, saved, actual
                if step in r['snapshot_steps']:
                    save_snapshot(step)
                if step % r['checkpoint_interval'] == 0 or step == r['total_steps']:
                    frozen_ok(); ledger.flush(); os.fsync(ledger.fileno())
                    path = folder / 'current.pt'; previous = folder / 'previous.pt'
                    if previous.exists():
                        require(previous.is_file() and not previous.is_symlink(), 'Ordinary generated previous checkpoint required')
                        previous.unlink()
                    if path.exists():
                        os.link(path, previous)
                    atomic_tensor(path, checkpoint(actor, optimizer, step, r, chain))
                    atomic_json(folder / 'resume.json', dict(file=str(path.relative_to(ROOT)), sha256=sha(path), step=step))
                if step <= 6 or step % 25 == 0:
                    monitor.status('training_complete_sequences', loss=event['loss'])
                    print(json.dumps(dict(stage='fit', step=step, loss=event['loss'], **monitor.receipt())), flush=True)
                del event
            require(control and control['passed'], 'Actual restoration/replay proof required')
            frozen_ok(); check_checkpoint(checkpoint(actor, optimizer, r['total_steps'], r, chain), r, r['total_steps'])
            atomic_json(folder / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), snapshots=snapshots,
                retained_updates=41298, extra_executed_resume_updates=2, training_cases=13766, epochs=3,
                target_exposure=6108741, ledger_final_sha256=chain, model_account=account, provenance=provenance,
                frozen_parameters_and_affines_unchanged=True, complete_context_recomputed_each_update=True,
                predictor_state_cache_used_for_training=False, language_quality_unmeasured=True, **monitor.receipt()))
    except BaseException as exc:
        error = exc; raise
    finally:
        for hook in hooks:
            hook.remove()
        if ledger is not None:
            ledger.close()
        finish(folder, monitor, error)


@torch.no_grad()
def qualify(r, resume):
    folder, monitor = begin('qualify', r, resume); error = None; hooks = []
    try:
        require(load(RUN / 'fit/report.json')['complete'] and load(RUN / 'audit/report.json')['complete'], 'Complete fit/audit first')
        require(not list(folder.glob('cases/*')), 'Preserve any partial qualification before explicit recovery')
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, _, _ = build_joint_context_actor(device)
            snapshot = load(RUN / 'fit/snapshots.json')['41298']; require(sha(ROOT / snapshot['file']) == snapshot['sha256'], 'Final snapshot changed')
            value = torch.load(ROOT / snapshot['file'], weights_only=True, map_location='cpu'); check_checkpoint(value, r, 41298)
            restore_trainable(actor, value['trainable_weights']); del value
            actor.requires_grad_(False).eval()
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
            versions = [(p, p._version) for p in [*actor.parameters(), *actor.buffers()]]
            targets = torch.load(ROOT / 'artifacts/continuation_clean_history_v1/targets.pt', weights_only=True, map_location='cpu')
            training = TrainingImages(PARENT)
            records = []; mapping = {}
            for i, job in enumerate(r['qualification_jobs']):
                monitor.position = dict(completed=i, total=124, case=job['id']); monitor.status('conditional_physical_writing')
                gold = targets[job['id']]; require(tensor_hash(gold) == job['target_sha256'], 'Complete reference pixels changed')
                if job['split'] == 'train':
                    prompt, original_gold, _, layout = training.get(int(job['id'].split('_')[1]), int(job['layout'] == 'proportional'))
                    require(layout == job['layout'] and torch.equal(original_gold, gold), 'Original training qualification differs')
                else:
                    canvas, _ = prepare_prompt(ROOT / job['image'])
                    prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None, None] / 127.5 - 1
                memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
                feedback = actor.bridge.feedback_embedding((1 - 2 * gold.float()).to(device))[None]
                states = actor.dynamics(actor.dynamics.embeddings(feedback), memory, checkpoint_layers=False)[0].float()
                require(states.shape == (len(gold) + 1, 4096), 'Complete target-preceding states required')
                patterns = pack_glyphs(gold).to(device); output = []; nll = []
                for start in range(0, len(gold), 64):
                    monitor.guard(); h = states[start:min(start + 64, len(gold))]; target = patterns[start:start + 64]
                    output.append(actor.bridge.density.write(h).cpu())
                    nll.extend((-actor.bridge.density(h, target).log_softmax(-1).gather(-1, target[..., None]).squeeze(-1).mean(1)).cpu().tolist())
                ink = torch.cat(output)
                old_path = CONTROL / 'qualify/cases' / job['id'] / 'pixels.npz'
                require(sha(old_path) == r['bindings_sha256'][str(old_path.relative_to(ROOT))], 'Unchanged head-only pixels differ')
                with np.load(old_path, allow_pickle=False) as old:
                    parent = torch.from_numpy(old['candidate_ink'].copy())
                    require(np.array_equal(old['gold_ink'], gold.numpy()), 'Control reference differs')
                require(ink.shape == parent.shape == gold.shape, 'Keep all complete positions')
                directory = folder / 'cases' / job['id']; directory.mkdir(parents=True)
                np.savez_compressed(directory / 'pixels.npz', candidate_ink=ink.numpy(), parent_ink=parent.numpy(), gold_ink=gold.numpy())
                a = ink.numpy().reshape(len(ink), -1); b = gold.numpy().reshape(len(gold), -1)
                den = a.sum(1) + b.sum(1); f1 = np.divide(2 * (a * b).sum(1), den, out=np.ones(len(a)), where=den > 0)
                record = dict(job=job, pixels_sha256=sha(directory / 'pixels.npz'), target_glyphs=len(gold), geometry=geometry,
                    pattern_nll_mean=float(np.mean(nll)), canonical_f1_mean=float(f1.mean()), canonical_f1_per_position=f1.tolist(),
                    canonical_exact=int(np.all(a == b, axis=1).sum()), supplied_reference_history=True, autonomous_language_evidence=False)
                atomic_json(directory / 'report.json', record); records.append(record)
                for role, tiles in [('candidate', ink), ('parent', parent)]:
                    for position in job['positions']:
                        key = hashlib.sha256(f'{sha(PUBLIC)}/{role}/{job["id"]}/{position}'.encode()).hexdigest()[:20]
                        require(key not in mapping, 'Opaque ID collision')
                        pages = folder / 'pages'; pages.mkdir(exist_ok=True)
                        Image.fromarray(255 - 255 * tiles[position, 0].numpy()).save(pages / (key + '.png'))
                        mapping[key] = dict(role=role, case=job['id'], split=job['split'], kind=job['kind'], layout=job['layout'],
                            position=position, pixels_sha256=record['pixels_sha256'], image_sha256=sha(pages / (key + '.png')))
                print(json.dumps(dict(case=job['id'], generated_glyphs=len(ink))), flush=True)
                del states, memory, feedback, patterns, ink, parent, gold, prompt
            require(all(p._version == version for p, version in versions), 'Qualification changed model tensors')
            atomic_json(folder / 'mapping.json', dict(registration_sha256=sha(PUBLIC), rows=mapping))
            identifiers = sorted(mapping); boards = []
            for start in range(0, len(identifiers), 64):
                ids = identifiers[start:start + 64]; canvas = Image.new('L', (1120, 848), 255); draw = ImageDraw.Draw(canvas)
                for j, key in enumerate(ids):
                    x, y = (j % 8) * 140, (j // 8) * 106
                    image = Image.open(folder / 'pages' / (key + '.png')).resize((64, 64), Image.Resampling.NEAREST)
                    canvas.paste(image, (x + 36, y + 4)); draw.text((x + 5, y + 76), key, fill=0)
                path = folder / f'board_{start // 64 + 1:03}.png'; canvas.save(path)
                boards.append(dict(file=path.name, sha256=sha(path), ids=ids))
            atomic_json(folder / 'boards.json', dict(boards=boards, images=len(mapping), all_outputs_reviewed=False))
            atomic_json(folder / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), records=records,
                mapping_sha256=sha(folder / 'mapping.json'), boards_sha256=sha(folder / 'boards.json'), all124_complete=True,
                new_optimizer_updates=0, new_autonomous_answers=0, physical_gate_unmeasured=True,
                language_quality_unmeasured=True, **monitor.receipt()))
    except BaseException as exc:
        error = exc; raise
    finally:
        for hook in hooks:
            hook.remove()
        finish(folder, monitor, error)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['train', 'qualify'])
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Use physicalGPU0 and deterministic CUBLAS')
    registration = validate()
    require_process_runtime(registration)
    {'train': train, 'qualify': qualify}[args.action](registration, args.resume)
