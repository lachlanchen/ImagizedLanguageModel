#!/usr/bin/env python3
"""Conditional pixel-loss ablation after the genuine-readout physical test.

Two fixed arms share the exact cached contexts, initialization and schedule.
The original body, actual-pixel feedback encoder and stopping weights stay fixed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
from scripts import train_genuine_glyph_readout as genuine
from scripts import distill_semantic_glyph_readout as base
from scripts import train_semantic_glyph_language as language
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu
from scripts.score_pixel_language_first_blocks import require

PARENT = ROOT / 'artifacts/genuine_glyph_readout_v1'
FILES = ('scripts/train_direct_pixel_readout.py', 'references/direct_pixel_readout_protocol.md')
ARMS = ('g_only', 'g_and_decoder_input')


def identity():
    for name in FILES:
        require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
                == (ROOT / name).read_bytes(), 'commit direct-pixel sources first')
    return {n: sha(ROOT / n) for n in FILES}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation')
    require(not output.exists(), 'preserve earlier ablations')
    require((PARENT / 'qualify/report.json').is_file(), 'finish genuine readout qualification first')
    require(language.load(PARENT / 'qualify/status.json')['phase'] == 'completed'
            and language.load(PARENT / 'qualify/status.json')['pid'] is None, 'parent qualification still active')
    m = genuine.check(PARENT)
    result = language.load(PARENT / 'qualify/report.json')
    require(not result['readout_gate_passed'], 'parent passed; assess its autonomous language first')
    s = result['summary']
    gate = m['gate']
    physical_failed = (s['genuine/copy_train']['mean_f1'] < gate['training_copy_mean_f1']
        or s['genuine/copy_held']['mean_f1'] < gate['held_copy_mean_f1']
        or s['genuine/copy_held']['fraction_f1_at_least095'] < gate['held_fraction095']
        or s['genuine/original_histories']['mean_f1'] <
           s['baseline/original_histories']['mean_f1'] - gate['old_cached_target_mean_f1_maximum_loss'])
    require(physical_failed, 'only stop criterion failed; pixel-loss ablation does not address it')
    code = identity()
    output.mkdir(parents=True)
    shared = dict(code_sha256=code, genuine_run=language.relative(PARENT),
        genuine_data_preparation_sha256=sha(PARENT / 'preparation.json'),
        genuine_cache_report_sha256=sha(PARENT / 'cache/report.json'),
        parent_qualification_sha256=sha(PARENT / 'qualify/report.json'),
        primitive=m['primitive'], baseline=m['baseline'], baseline_sha256=m['baseline_sha256'],
        seed=m['seed'], fit_steps=m['fit_steps'], batch_size=m['batch_size'],
        learning_rate=m['learning_rate'], gate=gate,
        maximum_generation_glyphs=m['maximum_generation_glyphs'],
        assessment_maximum_seconds=m['assessment_maximum_seconds'],
        minimum_entry_ram_gib=48., minimum_runtime_ram_gib=32., maximum_vram_gib=8.,
        maximum_stage_seconds=1800., maximum_pressure_wait_seconds=300.,
        architecture_and_parameter_count_unchanged=True,
        loss='balanced foreground/background pixel BCE only',
        semantic_anchor_regression=False, semantic_cosine_loss=False,
        actual_pixel_feedback_encoder_frozen=True, stop_and_affine_buffers_frozen=True,
        C3_inputs_or_references_used=False, development_training=False,
        model_quality_not_established=True)
    save_json(output / 'preparation.json', shared)
    for arm in ARMS:
        folder = output / arm
        folder.mkdir()
        save_json(folder / 'preparation.json', dict(shared, arm=arm,
            trained_modules=['readout'] + (['decoder_input'] if arm == ARMS[1] else [])))
    print(json.dumps(dict(prepared_arms=list(ARMS), parent_physical_gate_failed=True,
                          neural_calls=0, new_parameters=0)))


def check(folder):
    m = language.load(folder / 'preparation.json')
    require(identity() == m['code_sha256'], 'direct-pixel sources changed')
    parent = ROOT / m['genuine_run']
    old = genuine.check(parent)
    for file, key in [('preparation.json', 'genuine_data_preparation_sha256'),
                      ('cache/report.json', 'genuine_cache_report_sha256'),
                      ('qualify/report.json', 'parent_qualification_sha256')]:
        require(sha(parent / file) == m[key], 'genuine source changed: ' + file)
    require(sha(ROOT / m['baseline']) == m['baseline_sha256'], 'preserved baseline changed')
    require(m['arm'] in ARMS, 'unknown fixed arm')
    return m, old, parent


def fit(folder, m, old, parent, monitor, device):
    torch.manual_seed(m['seed'])
    torch.cuda.manual_seed_all(m['seed'])
    model = base.bridge(m, device)
    model.readout.requires_grad_(True)
    if m['arm'] == ARMS[1]:
        model.decoder_input.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    pools = genuine.pools(parent, old)
    training = [(h.to(device), y.to(device)) for route, (h, y) in pools.items() if route != 'copy_held']
    data = torch.load(ROOT / m['primitive'] / 'training.pt', weights_only=True, map_location='cpu')
    images = (1 - 2 * data['ink'].float()).to(device)
    del data  # No semantic vector target is used in either arm.
    opt = torch.optim.AdamW(params, lr=m['learning_rate'], weight_decay=.01)
    for step in range(1, m['fit_steps'] + 1):
        if step % 20 == 1:
            monitor.position = dict(arm=m['arm'], step=step, total=m['fit_steps'])
            monitor.guard()
        hs, ys = [], []
        for h, y in training:
            ids = torch.randint(len(h), (m['batch_size'] // 2,), device=device)
            hs.append(h[ids])
            ys.append(y[ids])
        h, labels = torch.cat(hs), torch.cat(ys)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            motor = model.readout(h).float()
            loss = reconstruction_loss(model.decode_logits(motor), images[labels])
        require(bool(torch.isfinite(loss)), 'nonfinite direct pixel loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        lr = m['learning_rate'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / m['fit_steps'])))
        for group in opt.param_groups:
            group['lr'] = lr
        if step % 100 == 0:
            row = dict(step=step, pixel_loss=float(loss), arm=m['arm'])
            with (monitor.path / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            monitor.status('fitting_direct_pixel_readout')
            print(json.dumps(row), flush=True)
    checkpoint = monitor.path / 'checkpoint_latest.pt'
    atomic_tensor(checkpoint, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        step=m['fit_steps'], language_training_complete=True, primitive_only=False,
        usable_language_demonstrated=False, motor_code_has_semantic_anchor_loss=False,
        preparation_sha256=sha(folder / 'preparation.json')))
    original = base.bridge(m, 'cpu')
    allowed = tuple(n + '.' for n in m['trained_modules'])
    require(all(torch.equal(original.state_dict()[k], v.cpu()) for k, v in model.state_dict().items()
                if not k.startswith(allowed)), 'a frozen feedback/stop/decoder-convolution weight changed')
    save_json(monitor.path / 'report.json', dict(checkpoint_sha256=sha(checkpoint), steps=m['fit_steps'],
        trained_modules=m['trained_modules'], trained_parameters=sum(p.numel() for p in params),
        unchanged_total_parameters=sum(p.numel() for p in model.parameters()),
        held_template_states_excluded=len(pools['copy_held'][0]),
        copy_training_states=len(pools['copy_train'][0]), old_histories=len(pools['original_histories'][0]),
        only_declared_modules_changed=True, body_loaded=False, teacher_loaded=False,
        semantic_targets_used=False, **monitor.receipt()))


@torch.no_grad()
def qualify(folder, m, old, parent, monitor, device):
    require(language.load(folder / 'fit/status.json')['phase'] == 'completed', 'finish fixed fit first')
    checkpoint = folder / 'fit/checkpoint_latest.pt'
    require(sha(checkpoint) == language.load(folder / 'fit/report.json')['checkpoint_sha256'], 'candidate changed')
    pools = genuine.pools(parent, old)
    data = torch.load(ROOT / m['primitive'] / 'training.pt', weights_only=True, map_location='cpu')
    gold = data['ink'].to(device).bool()
    summary, records, emitted, files = {}, [], {}, {}
    for version, path in [('baseline', ROOT / m['baseline']), ('candidate', checkpoint)]:
        model = base.bridge(m, device, path)
        for route, (states, labels) in pools.items():
            parts, f1s, exact, nonstop = [], [], 0, 0
            for start in range(0, len(states), 64):
                monitor.position = dict(model=version, route=route, start=start, total=len(states))
                monitor.guard()
                logits, _, stop = model.predict(states[start:start + 64].to(device))
                a = logits >= 0
                g = gold[labels[start:start + 64].to(device)]
                d = a.flatten(1).sum(-1) + g.flatten(1).sum(-1)
                f1 = torch.where(d == 0, torch.ones_like(d).float(), 2 * (a & g).flatten(1).sum(-1) / d)
                same = (a == g).flatten(1).all(-1)
                f1s.extend(f1.cpu().tolist())
                exact += int(same.sum())
                nonstop += int((stop.sigmoid() < .5).sum())
                parts.append(a.cpu())
                for i in range(len(a)):
                    records.append(dict(model=version, route=route, index=start + i,
                        f1=float(f1[i]), pixel_exact=bool(same[i]), initial_stop_probability=float(stop.sigmoid()[i])))
            key = version + '/' + route
            emitted[key] = torch.cat(parts)
            if version == 'baseline':
                # Cold replay against the same parent baseline, not its fitted candidate.
                prior = parent / 'qualify' / ('baseline_' + route + '.pt')
                require(sha(prior) == language.load(parent / 'qualify/report.json')['raw_outputs_sha256'][prior.name],
                        'parent baseline pixels changed')
                require(torch.equal(emitted[key], torch.load(prior, weights_only=True, map_location='cpu')['ink']),
                        'baseline replay differs; do not compare the arms')
            name = version + '_' + route + '.pt'
            atomic_tensor(monitor.path / name, dict(ink=emitted[key], labels=labels))
            files[name] = sha(monitor.path / name)
            summary[key] = dict(cases=len(states), mean_f1=sum(f1s) / len(f1s), pixel_exact=exact,
                fraction_f1_at_least095=sum(v >= .95 for v in f1s) / len(f1s), nonstop_fraction=nonstop / len(f1s))
        del model
    boards = []
    for route, (_, labels) in pools.items():
        ids = np.linspace(0, len(labels) - 1, 64, dtype=int)
        page = Image.new('L', (896, 392), 245)
        draw = ImageDraw.Draw(page)
        for i, index in enumerate(ids):
            x, y = (i % 8) * 112, (i // 8) * 49
            draw.text((x + 2, y), str(index), fill=0)
            for j, a in enumerate([gold[labels[index]].cpu(), emitted['baseline/' + route][index],
                                   emitted['candidate/' + route][index]]):
                page.paste(Image.fromarray(255 * (1 - a[0].numpy().astype(np.uint8))), (x + 2 + j * 34, y + 13))
        name = route + '_gold_baseline_candidate.png'
        page.resize((1792, 784), Image.Resampling.NEAREST).save(monitor.path / name)
        boards.append(name)
    gate = m['gate']
    train, held = summary['candidate/copy_train'], summary['candidate/copy_held']
    passed = (train['mean_f1'] >= gate['training_copy_mean_f1'] and held['mean_f1'] >= gate['held_copy_mean_f1']
        and held['fraction_f1_at_least095'] >= gate['held_fraction095']
        and held['nonstop_fraction'] >= gate['held_first_nonstop_fraction']
        and summary['candidate/original_histories']['mean_f1'] >=
            summary['baseline/original_histories']['mean_f1'] - gate['old_cached_target_mean_f1_maximum_loss'])
    save_json(monitor.path / 'records.json', records)
    save_json(monitor.path / 'report.json', dict(arm=m['arm'], summary=summary,
        readout_gate_passed=passed, raw_outputs_sha256=files, boards=boards,
        parent_baseline_pixels_cold_replay_exact=True, full_language_assessed=False,
        actual_PNG_review_pending=True, **monitor.receipt()))
    print(json.dumps(dict(arm=m['arm'], summary=summary, readout_gate_passed=passed)), flush=True)


def run(folder, stage):
    m, old, parent = check(folder)
    if stage == 'assessment':
        require(language.load(folder / 'qualify/report.json')['readout_gate_passed'], 'physical gate failed')
        review = language.load(folder / 'qualify/visual_review.json')
        require(review['report_sha256'] == sha(folder / 'qualify/report.json')
                and review['all_three_boards_actually_viewed'], 'view all raw qualification boards first')
    path = folder / stage
    path.mkdir(exist_ok=False)
    monitor = (language.Monitor(path, m['assessment_maximum_seconds']) if stage == 'assessment'
               else base.Monitor(path, m))
    monitor.status('admitting')
    try:
        if stage == 'assessment':
            language.assess(folder, old, monitor)
        else:
            with bridge_gpu(m) as (device, admission):
                language.numerics()
                monitor.guard()
                save_json(path / 'admission.json', admission)
                {'fit': fit, 'qualify': qualify}[stage](folder, m, old, parent, monitor, device)
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed', reason=repr(error))
        raise
    finally:
        save_json(path / 'status.json', dict(language.load(path / 'status.json'), pid=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'qualify', 'assessment'], required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/direct_pixel_readout_v1')
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output.resolve())
    else:
        require(args.arm is not None, 'select one fixed arm; never run two model jobs concurrently')
        run(args.output.resolve() / args.arm, args.stage)
