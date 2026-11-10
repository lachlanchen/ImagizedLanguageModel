#!/usr/bin/env python3
"""Continue a sealed glyph-tail fit, checking actual checkpoint continuation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require, owner_live

OWN = 'scripts/resume_glyph_tail_fit.py'
PROTOCOL = 'references/glyph_tail_fit_recovery.md'


def load(path):
    return json.loads(Path(path).read_text())


def source_identity():
    for name in [OWN, PROTOCOL]:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit recovery sources first')
    return {name: sha(ROOT / name) for name in [OWN, PROTOCOL]}


def audit(run, folder, checkpoint):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'audit requires hidden CUDA')
    from scripts.train_glyph_tail_adapter import verify
    import torch
    m = verify(run)
    for status in run.glob('fit*/status.json'):
        require(not owner_live(load(status)), 'another fit owner remains active')
    pipeline = load(run / 'pipeline.json')
    for key in ['child', 'supervisor']:
        require(not owner_live(dict(pid=pipeline.get(key + '_pid'),
                    start_ticks=pipeline.get(key + '_start_ticks'), boot_id=pipeline.get('boot_id'))),
                'coordinator or its child remains active')
    require(load(run / 'fit/status.json')['phase'] != 'completed', 'do not replace a completed fit')
    require(checkpoint.resolve().is_relative_to(run.resolve()), 'checkpoint must belong to this run')
    digest = sha(checkpoint)
    value = torch.load(checkpoint, weights_only=True, map_location='cpu')
    step = value['step']
    require(value['preparation_sha256'] == sha(run / 'preparation.json')
            and value['tail_start_layer'] == 32 and 0 < step < m['fit_steps'] - 8,
            'wrong or completed checkpoint')
    require(step % 1000 == 0 and value['optimizer']['state'], 'require an actual periodic optimizer checkpoint')
    require(len(value['optimizer']['param_groups']) == 2, 'optimizer groups differ')
    for group, initial in zip(value['optimizer']['param_groups'],
                              [m['adapter_learning_rate'], m['readout_learning_rate']]):
        expected = initial * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / m['fit_steps'])))
        require(group['lr'] == expected, 'checkpoint learning-rate position differs')
    require(sha(checkpoint) == digest, 'checkpoint changed during audit')
    folder.mkdir(exist_ok=False)
    save_json(folder / 'audit.json', dict(code_sha256=source_identity(), checkpoint=str(checkpoint.relative_to(ROOT)),
        checkpoint_sha256=digest, checkpoint_step=step, preparation_sha256=sha(run / 'preparation.json'),
        original_fit_status=load(run / 'fit/status.json'), original_pipeline=pipeline,
        fit_steps=m['fit_steps'], optimizer_schedule_changed=False, data_changed=False,
        pressure_wait_seconds=1800., original_pressure_wait_seconds=300.,
        roundtrip_proof_effective_updates=8, roundtrip_proof_discarded_updates=4,
        older_uncheckpointed_updates_not_recoverable=True))


def run_fit(run, folder):
    from scripts.train_glyph_tail_adapter import verify
    m = verify(run); a = load(folder / 'audit.json')
    require(source_identity() == a['code_sha256'], 'recovery sources changed')
    require(sha(run / 'preparation.json') == a['preparation_sha256'], 'preparation changed')
    checkpoint = ROOT / a['checkpoint']
    require(sha(checkpoint) == a['checkpoint_sha256'], 'audited checkpoint changed')
    for status in run.glob('fit*/status.json'):
        require(not owner_live(load(status)), 'a fit is still active')
    require(not (folder / 'status.json').exists(), 'preserve every recovery attempt')
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.glyph_tail_adapter import load_tail, adapter_state
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss, tensor_hash
    from scripts.train_semantic_glyph_language import bridge_model, numerics
    from scripts.train_semantic_glyph_bridge import bridge_gpu, atomic_tensor
    from scripts.train_genuine_glyph_readout import pools
    from scripts import distill_semantic_glyph_readout as base
    monitor = base.Monitor(folder, dict(maximum_stage_seconds=m['fit_seconds'],
        maximum_vram_gib=12., minimum_runtime_ram_gib=32., maximum_pressure_wait_seconds=1800.))
    monitor.status('admitting')
    try:
        rows = load(run / 'cache/report.json')['files']
        for row in rows:
            require(sha(ROOT / row['file']) == row['sha256'], 'frozen prefix changed')
        # The complete 55-GiB integrity read can itself coincide with host I/O
        # pressure. Wait before strict admission instead of failing without
        # any optimizer work; retain all live resource floors and the mutex.
        monitor.guard()
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device, admission):
            numerics(); torch.manual_seed(m['seed']); torch.cuda.manual_seed_all(m['seed'])
            tail, start = load_tail(device); tail.train()
            model = bridge_model({}, device, ROOT / m['bridge'])
            model.readout.requires_grad_(True); model.stop.requires_grad_(True)
            data = torch.load(ROOT / m['primitive'] / 'training.pt', weights_only=True, map_location='cpu')
            images = (1 - 2 * data['ink'].float()).to(device)
            copied = pools(ROOT / m['genuine'], load(ROOT / m['genuine'] / 'preparation.json'))['copy_train']
            copied = (copied[0].to(device), copied[1].to(device))
            body = [p for p in tail.parameters() if p.requires_grad]
            head = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW([dict(params=body, lr=m['adapter_learning_rate']),
                                    dict(params=head, lr=m['readout_learning_rate'])], weight_decay=.01)

            def restore(path):
                v = torch.load(path, weights_only=True, map_location='cpu')
                require(v['preparation_sha256'] == a['preparation_sha256'] and v['tail_start_layer'] == start,
                        'restored identity differs')
                model.load_state_dict(v['model'])
                params = dict(tail.named_parameters())
                require(set(v['body_tail_adapters']) == set(adapter_state(tail)), 'adapter keys differ')
                with torch.no_grad():
                    for name, tensor in v['body_tail_adapters'].items(): params[name].copy_(tensor.to(device))
                opt.load_state_dict(v['optimizer'])
                torch.set_rng_state(v['torch_rng']); torch.cuda.set_rng_state(v['cuda_rng'])
                return v['step']

            step = restore(checkpoint)

            def save(path, completed):
                atomic_tensor(path, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    body_tail_adapters=adapter_state(tail), tail_start_layer=start, optimizer=opt.state_dict(),
                    step=completed, language_training_complete=completed == m['fit_steps'],
                    preparation_sha256=a['preparation_sha256'], usable_language_demonstrated=False,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

            def update(index):
                monitor.position = dict(step=index, total=m['fit_steps']); monitor.guard()
                opt.zero_grad(set_to_none=True)
                if index % 4 == 3:
                    ix = torch.randint(len(copied[0]), (128,), device=device)
                    logits, _, _ = model.predict(copied[0][ix])
                    pixel = reconstruction_loss(logits, images[copied[1][ix]]); stop = pixel * 0
                    sample = dict(route='copy', indices_sha256=tensor_hash(ix))
                else:
                    ix = int(torch.randint(len(rows), ()))
                    value = torch.load(ROOT / rows[ix]['file'], weights_only=True, map_location='cpu', mmap=True)
                    prediction = tail(inputs_embeds=value['lower'].to(device),
                        position_ids=value['position_ids'].to(device), use_cache=False).last_hidden_state[0, value['boundary']-1:]
                    labels = value['labels'].to(device); logits, _, stops = model.predict(prediction)
                    pixel = reconstruction_loss(logits[:-1], images[labels[:-1]])
                    stop = .5 * (F.softplus(-stops[-1]) + F.softplus(stops[:-1]).mean())
                    sample = dict(route='question', index=ix)
                loss = pixel + stop
                require(bool(torch.isfinite(loss)), 'nonfinite continued fit loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(body + head, 1.); opt.step()
                decay = .1 + .9 * .5 * (1 + math.cos(math.pi * (index + 1) / m['fit_steps']))
                for group, rate in zip(opt.param_groups, [m['adapter_learning_rate'], m['readout_learning_rate']]):
                    group['lr'] = rate * decay
                return dict(step=index+1, pixel=float(pixel.detach()), stop=float(stop.detach()),
                            loss=float(loss.detach()), **sample)

            def fingerprint():
                result = dict(parameters=[tensor_hash(p.detach()) for p in body + head],
                    cpu_rng=tensor_hash(torch.get_rng_state()), cuda_rng=tensor_hash(torch.cuda.get_rng_state()),
                    optimizer=[])
                for group in opt.param_groups:
                    result['optimizer'].append(dict(lr=group['lr'], states=[
                        {k: tensor_hash(v.reshape(-1)) if torch.is_tensor(v) else v for k, v in opt.state[p].items()}
                        for p in group['params']]))
                return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()

            # Real question AND copy updates, including serialized optimizer/RNG,
            # not a reduced stand-in model or an evaluation-data-based choice.
            first = [update(i) for i in range(step, step+4)]
            boundary = folder / 'roundtrip_boundary.pt'; save(boundary, step+4)
            expected = [update(i) for i in range(step+4, step+8)]; expected_sha = fingerprint()
            require(restore(boundary) == step+4, 'roundtrip boundary changed')
            actual = [update(i) for i in range(step+4, step+8)]; actual_sha = fingerprint()
            require(actual == expected and actual_sha == expected_sha, 'actual optimizer continuation differs')
            save_json(folder / 'roundtrip.json', dict(first=first, uninterrupted=expected, resumed=actual,
                expected_state_sha256=expected_sha, resumed_state_sha256=actual_sha,
                all_trainable_weights_optimizer_buffers_RNG_bit_exact=True, effective_updates=8,
                additional_discarded_updates=4, boundary_sha256=sha(boundary), evaluation_inputs_used=False))
            frozen = [(p, p._version) for p in [*tail.parameters(), *model.parameters()] if not p.requires_grad]
            for index in range(step+8, m['fit_steps']):
                item = update(index)
                require(all(p._version == version for p, version in frozen), 'frozen parameter changed')
                if (index+1) % 100 == 0 or index % 100 == 0:
                    with (folder / 'history.jsonl').open('a') as stream: stream.write(json.dumps(item)+'\n')
                    monitor.status('training_glyph_context_and_readout', **item); print(json.dumps(item), flush=True)
                if (index+1) % 1000 == 0:
                    path = folder / 'checkpoint_latest.pt'
                    if path.exists(): path.replace(folder / 'checkpoint_previous.pt')
                    save(path, index+1)
            path = folder / 'checkpoint_latest.pt'
            summary = dict(preparation_sha256=a['preparation_sha256'], checkpoint_sha256=sha(path),
                actual_optimizer_updates=m['fit_steps'], trained_body_parameters=sum(p.numel() for p in body),
                trained_bridge_parameters=sum(p.numel() for p in head), all_checks_passed=True,
                all_frozen_parameter_versions_unchanged=True, full_body_loaded=False,
                language_accuracy_not_measured=True, recovery_audit_sha256=sha(folder / 'audit.json'),
                recovery_code_sha256=a['code_sha256'], resumed_from_step=step,
                actual_additional_updates_discarded_for_roundtrip=4,
                roundtrip_sha256=sha(folder / 'roundtrip.json'), cross_attempt_wall_time_complete=False,
                optimizer_schedule_changed=False, data_changed=False, **monitor.receipt())
            save_json(folder / 'report.json', summary)
            original = run / 'fit/checkpoint_latest.pt'
            preserved = folder / 'initial_interrupted_checkpoint.pt'
            os.link(original, preserved)
            temporary = run / 'fit/checkpoint_recovered.partial'
            os.link(path, temporary); os.replace(temporary, original)
            save_json(run / 'fit/report.json', summary)
        monitor.status('completed')
        save_json(run / 'fit/status.json', dict(load(folder / 'status.json'), pid=None,
            recovery=str(folder.relative_to(ROOT))))
    except BaseException as error:
        monitor.status('failed', reason=repr(error)); raise
    finally:
        save_json(folder / 'status.json', dict(load(folder / 'status.json'), pid=None))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['audit', 'run'], required=True)
    p.add_argument('--run', type=Path, default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--recovery-name', default='fit_recovery_01')
    p.add_argument('--checkpoint', type=Path)
    a = p.parse_args(); root = a.run.resolve()
    require(a.recovery_name.startswith('fit_recovery_') and '/' not in a.recovery_name,
            'invalid recovery folder')
    folder = root / a.recovery_name
    if a.stage == 'audit': audit(root, folder, (a.checkpoint or root/'fit/checkpoint_latest.pt').resolve())
    else: run_fit(root, folder)
