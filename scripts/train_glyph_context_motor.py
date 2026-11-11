#!/usr/bin/env python3
"""Train a complete continuous glyph head from authenticated real-reader states."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge, tensor_hash
from scripts.compare_glyph_tail_compact_cohort import CHECKPOINT, CHECKPOINT_SHA, EVALUATION
from scripts.score_pixel_language_first_blocks import host_resources, require
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu

OWN = 'scripts/train_glyph_context_motor.py'
PROTOCOL = 'references/glyph_context_motor_protocol.md'
SOURCES = (OWN, PROTOCOL, 'references/glyph_tail_context_writer_decision.md',
    'ilm/visual_lm/context_motor_flow.py', 'ilm/visual_lm/semantic_glyph_loop.py',
    'scripts/train_semantic_glyph_bridge.py', 'scripts/train_semantic_glyph_language.py',
    'scripts/score_pixel_language_first_blocks.py', 'scripts/train_balanced_glyph_writer.py')
CACHE = ROOT/'artifacts/balanced_glyph_writer_v1/cache'
CACHE_SHA = '201db406d906a2a1eaff916c14d24e256e77c24e9140c51c6e73cb42ab6aab5c'
PRIMITIVE = ROOT/'artifacts/semantic_glyph_bridge_v1'


def load(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity():
    hashes = {n:sha(ROOT/n) for n in SOURCES}
    for name, digest in hashes.items():
        committed = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == digest, 'Commit registered source first: '+name)
    return hashes


class Monitor:
    def __init__(self, folder, *, cpu=False):
        self.folder, self.cpu = folder, cpu
        self.began = time.perf_counter()
        self.waited, self.samples, self.step = 0., [], 0
        self.seconds = 600 if cpu else 1800
        self.wait_allowance = 120 if cpu else 600

    def status(self, phase, **extra):
        value = dict(phase=phase, pid=os.getpid(), step=self.step,
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            elapsed_seconds=time.perf_counter()-self.began)
        value.update(extra)
        save_json(self.folder/'status.json', value)

    def guard(self):
        if self.cpu:
            require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU only')
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 <= 4, 'CPU RSS cap reached')
        else:
            require(torch.cuda.max_memory_allocated()/1024**3 <= 8, 'Head-fit allocation cap reached')
        while True:
            elapsed = time.perf_counter()-self.began
            require(elapsed < self.seconds, 'Stage wall budget exhausted')
            v = host_resources(); self.samples.append(v)
            require(v['available_gib'] >= 24, 'Host runtime reserve reached')
            if v['some_pressure_avg10'] <= .5 and v['full_pressure_avg10'] <= .1:
                return
            remaining = min(self.wait_allowance-self.waited, self.seconds-elapsed)
            require(remaining > 0, 'Pressure wait budget exhausted')
            self.status('waiting_for_memory_pressure', resources=v)
            began = time.perf_counter(); time.sleep(min(5., remaining)); self.waited += time.perf_counter()-began

    def receipt(self):
        return dict(elapsed_seconds=time.perf_counter()-self.began, resource_wait_seconds=self.waited,
            peak_RSS_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            peak_allocated_vram_gib=0 if self.cpu else torch.cuda.max_memory_allocated()/1024**3,
            minimum_available_gib=min(v['available_gib'] for v in self.samples),
            CUDA_initialized=torch.cuda.is_initialized())


def evaluation_inputs():
    rows, bindings = [], {}
    for layout, run, base in [
        ('fixed_cells', ROOT/'artifacts/glyph_tail_compact_cohort_v1', EVALUATION/'inputs'),
        ('proportional', ROOT/'artifacts/glyph_tail_input_presentation_v1',
         ROOT/'artifacts/glyph_tail_input_presentation_v1/evaluation/inputs')]:
        registration = load(run/'registration.json')
        require(registration['checkpoint_sha256'] == CHECKPOINT_SHA and registration['maximum_glyphs'] == 128,
                'Preserve original complete-response control')
        bindings[relative(run/'registration.json')] = sha(run/'registration.json')
        bindings[relative(run/'review/scores.json')] = sha(run/'review/scores.json')
        for original in registration['inputs']:
            name = layout+'__'+original['id']
            path = base/original['image']
            require(sha(path) == original['sha256'], 'Physical evaluation input changed')
            seed = int.from_bytes(hashlib.sha256(('glyph-context-motor-v1\0'+name).encode()).digest()[:8], 'big') % (2**63)
            rows.append(dict(original, id=name, original_case=original['id'], layout=layout,
                             image=relative(path), seed=seed))
            bindings[relative(path)] = original['sha256']
    require(len(rows) == 216 and len({r['id'] for r in rows}) == 216, 'Keep all216 original streams')
    return rows, bindings


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not run.exists(), 'New CPU preparation only')
    require(host_resources()['available_gib'] >= 32, 'CPU entry reserve unavailable')
    torch.set_num_threads(2); code = identity()
    run.mkdir(parents=True); monitor = Monitor(run, cpu=True); monitor.status('preparing')
    try:
        monitor.guard()
        cache_report = load(CACHE/'report.json')
        require(cache_report['pool_sha256'] == CACHE_SHA and sha(CACHE/'pools.pt') == CACHE_SHA,
                'Actual parent training cache changed')
        require(cache_report['actual_parent_training_pixels_and_stop_counts_reproduced']
                and cache_report['all23262_training_pixel_scores_verified'], 'Actual-context qualification required')
        parent = load(ROOT/'artifacts/glyph_tail_adaptation_v2/preparation.json')
        source_cache = ROOT/'artifacts/glyph_tail_adaptation_v2/cache/report.json'
        require(sha(source_cache) == cache_report['source_cache_sha256'], 'Parent cache provenance changed')
        require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Preserve the exact contextual parent')
        pools = torch.load(CACHE/'pools.pt', map_location='cpu', weights_only=True)
        jobs = parent['training_jobs']
        require(len(jobs) == 3176, 'Preserve all authentic training questions')
        first_labels = torch.tensor([j['labels'][0] for j in jobs])
        later_labels = torch.tensor([label for j in jobs for label in j['labels'][1:]])
        require(torch.equal(first_labels, pools['first_labels']) and torch.equal(later_labels, pools['later_labels']),
                'First/later training labels do not reproduce source order')
        require(pools['first'].shape == (3176, 4096) and pools['later'].shape == (20086, 4096)
                and pools['terminal'].shape == (3176, 4096), 'Actual training-state shape differs')
        require(all(torch.isfinite(v).all() for v in pools.values()), 'Nonfinite cached training state')
        primitive = load(PRIMITIVE/'preparation.json')
        require(sha(PRIMITIVE/'training.pt') == primitive['training_sha256'], 'Primitive image identity changed')
        primitive_data = torch.load(PRIMITIVE/'training.pt', map_location='cpu', weights_only=True)
        images = 1-2*primitive_data['ink'].float()
        labels = torch.cat([first_labels, later_labels])
        require(int(labels.min()) >= 0 and int(labels.max()) < len(images), 'Invalid training-only image label')
        model = SemanticGlyphBridge().eval().requires_grad_(False)
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
        model.load_state_dict(payload['model']); del payload
        motor = []
        with torch.inference_mode():
            for start in range(0, len(images), 32):
                monitor.guard()
                motor.append(model.decoder_input(model.encode(images[start:start+32])).clone())
        motor = torch.cat(motor).float()
        require(motor.shape == (4087, 1024) and torch.isfinite(motor).all(), 'Invalid supplied motor features')
        a, b = motor[first_labels], motor[later_labels]
        mean = .5*(a.mean(0)+b.mean(0))
        variance = (.5*(a.square().mean(0)+b.square().mean(0))-mean.square()).clamp_min(0)
        scale = variance.sqrt().clamp_min(.05)
        atomic_tensor(run/'training_targets.pt', dict(motor=motor, mean=mean, scale=scale))
        evaluation, bindings = evaluation_inputs()
        for path in [CHECKPOINT, CACHE/'pools.pt', CACHE/'report.json', source_cache,
                     ROOT/'artifacts/glyph_tail_adaptation_v2/preparation.json',
                     ROOT/'artifacts/balanced_glyph_writer_v1/preparation.json',
                     PRIMITIVE/'preparation.json', PRIMITIVE/'training.pt']:
            bindings[relative(path)] = sha(path)
        save_json(run/'evaluation_manifest.json', dict(inputs=evaluation, maximum_glyphs=128,
            sampling_steps=32, sampling_candidates=1, raw_inputs_without_answers=True))
        torch.manual_seed(202611171)
        flow = ContextMotorFlow()
        monitor.guard()
        save_json(run/'registration.json', dict(schema='glyph-context-motor-v1', code_sha256=code,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            bindings_sha256=bindings, checkpoint=relative(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
            training_pool=relative(CACHE/'pools.pt'), training_targets_sha256=sha(run/'training_targets.pt'),
            evaluation_manifest_sha256=sha(run/'evaluation_manifest.json'),
            parameters=sum(p.numel() for p in flow.parameters()), first_states=3176, later_states=20086,
            seed=202611171, fit_steps=12000, batch_size=128, first_per_update=64, later_per_update=64,
            optimizer=dict(name='AdamW', learning_rate=.0002, betas=[.9, .99], weight_decay=.01,
                           clip_norm=1., warmup_updates=200, final_learning_rate_fraction=.1),
            motor_width=1024, velocity_hidden_width=1536, motor_coordinate_scale_floor=.05,
            normalization_uses_equal_first_later_training_mass=True,
            EMA='min(0.999,(1+step)/(10+step)); final only', sampler='one sample;32Heun intervals;last Euler',
            stopped_by_frozen_parent=True, actual_pixel_feedback_required=True,
            development_answers_or_helper_states_used_for_training=False,
            body_adapters_encoder_decoder_stop_frozen=True,
            small_fit_wall_seconds=1800, pressure_wait_seconds=600, small_fit_peak_vram_gib=8,
            minimum_host_entry_gib=48, minimum_host_runtime_gib=24,
            full_reader_stage_wall_seconds=3600, full_reader_peak_vram_gib=22,
            language_accuracy_not_measured=True, usable_ILM_demonstrated=False,
            supplied_target_preparation_resources=monitor.receipt()))
        monitor.status('prepared')
        print(json.dumps(dict(parameters=sum(p.numel() for p in flow.parameters()), train_positions=len(labels),
            evaluation_streams=len(evaluation), target_features=list(motor.shape), CUDA_initialized=False)))
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        save_json(run/'status.json', dict(load(run/'status.json'), pid=None))


def validate(run):
    r = load(run/'registration.json')
    require(r['code_sha256'] == identity(), 'Registered sources changed')
    for name, digest in r['bindings_sha256'].items():
        require(sha(ROOT/name) == digest, 'Bound input changed: '+name)
    require(sha(run/'training_targets.pt') == r['training_targets_sha256']
            and sha(run/'evaluation_manifest.json') == r['evaluation_manifest_sha256'], 'Prepared data changed')
    return r


def fit(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Use physical GPU0 only')
    r = validate(run); folder = run/'fit'; folder.mkdir(exist_ok=False)
    monitor = Monitor(folder); monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device, admission):
            save_json(folder/'admission.json', admission)
            from scripts.train_semantic_glyph_language import numerics
            numerics(); torch.set_num_threads(2); torch.manual_seed(r['seed']); torch.cuda.manual_seed_all(r['seed'])
            monitor.guard()
            data = torch.load(run/'training_targets.pt', map_location='cpu', weights_only=True)
            targets = ((data['motor']-data['mean'])/data['scale']).to(device)
            pools = torch.load(ROOT/r['training_pool'], map_location='cpu', weights_only=True)
            p = {k:v.to(device) for k, v in pools.items() if k != 'terminal'}
            del pools
            flow = ContextMotorFlow().to(device)
            require(sum(v.numel() for v in flow.parameters()) == r['parameters'], 'Architecture differs')
            ema = copy.deepcopy(flow).eval().requires_grad_(False)
            optimizer = torch.optim.AdamW(flow.parameters(), lr=.0002, betas=(.9, .99), weight_decay=.01)
            sampler = hashlib.sha256(); counts = dict(first=0, later=0)
            began_weights = {k:tensor_hash(v) for k, v in flow.state_dict().items()}
            for step in range(1, r['fit_steps']+1):
                monitor.step = step
                if step % 10 == 1:
                    monitor.guard()
                rate = min(1., step/200) * (.1+.9*.5*(1+math.cos(math.pi*max(0, step-200)/(r['fit_steps']-200))))
                for group in optimizer.param_groups:
                    group['lr'] = .0002*rate
                optimizer.zero_grad(set_to_none=True)
                states, labels = [], []
                for name in ['first', 'later']:
                    ix = torch.randint(len(p[name]), (64,), device=device)
                    sampler.update(name.encode()); sampler.update(ix.cpu().numpy().tobytes()); counts[name] += 64
                    states.append(p[name][ix]); labels.append(p[name+'_labels'][ix])
                state = torch.cat(states); y = targets[torch.cat(labels)]
                noise = torch.randn_like(y); t = torch.rand(len(y), device=device)
                mixed = (1-t[:, None])*noise+t[:, None]*y
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    velocity = flow(mixed, t, state)
                loss = (velocity-(y-noise)).square().mean()
                require(bool(torch.isfinite(loss)), 'Nonfinite flow loss')
                loss.backward(); norm = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.)
                require(bool(torch.isfinite(norm)), 'Nonfinite flow gradient')
                optimizer.step()
                decay = min(.999, (1+step)/(10+step))
                with torch.no_grad():
                    for average, parameter in zip(ema.parameters(), flow.parameters()):
                        average.lerp_(parameter, 1-decay)
                if step == 1 or step % 100 == 0:
                    row = dict(step=step, velocity_MSE=float(loss.detach()), gradient_norm=float(norm),
                               learning_rate=optimizer.param_groups[0]['lr'])
                    with (folder/'history.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    monitor.status('training_actual_context_motor', **row); print(json.dumps(row), flush=True)
                if step % 1000 == 0:
                    previous = folder/'checkpoint_latest.pt'
                    if previous.exists():
                        previous.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(previous, dict(schema='glyph-context-motor-checkpoint-v1',
                        model={k:v.detach().cpu() for k, v in flow.state_dict().items()},
                        ema={k:v.detach().cpu() for k, v in ema.state_dict().items()}, optimizer=optimizer.state_dict(),
                        motor_mean=data['mean'], motor_scale=data['scale'], step=step,
                        preparation_sha256=sha(run/'registration.json'), parent_checkpoint_sha256=CHECKPOINT_SHA,
                        sampled_target_counts=counts.copy(), sampled_indices_sha256=sampler.hexdigest(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                        training_complete=step == r['fit_steps'], language_accuracy_not_measured=True))
            monitor.guard(); torch.cuda.synchronize()
            require(counts == dict(first=768000, later=768000), 'Fixed training exposure differs')
            require(all(torch.isfinite(v).all() for v in ema.state_dict().values()), 'Nonfinite final EMA')
            changed = [k for k,v in flow.state_dict().items() if tensor_hash(v) != began_weights[k]]
            require(changed, 'No parameter updates occurred')
            save_json(folder/'report.json', dict(registration_sha256=sha(run/'registration.json'),
                checkpoint_sha256=sha(folder/'checkpoint_latest.pt'), actual_optimizer_updates=r['fit_steps'],
                actual_trainable_parameters=r['parameters'], sampled_target_counts=counts,
                sampled_indices_sha256=sampler.hexdigest(), changed_parameter_tensors=len(changed),
                full_reader_loaded=False, inherited_body_parameters_updated=0, base_bridge_loaded_during_fit=False,
                original_parent_checkpoint_sha256=CHECKPOINT_SHA, final_EMA_only=True,
                training_complete=True, language_accuracy_not_measured=True, goal_achieved=False, **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        save_json(folder/'status.json', dict(load(folder/'status.json'), pid=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit'])
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_context_motor_v1')
    args = parser.parse_args()
    {'prepare':prepare, 'fit':fit}[args.stage](args.run.resolve())
