#!/usr/bin/env python3
"""Frozen conditional writing diagnosis:64 training states and64 retained firsts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge, tensor_hash
from scripts.audit_glyph_tail_first_writing import validate as first_audit, pixels_metrics
from scripts.infer_glyph_context_motor import study
from scripts.score_pixel_language_first_blocks import require
from scripts.train_glyph_context_motor import load, Monitor, CHECKPOINT, CACHE, PRIMITIVE
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu
from scripts.train_semantic_glyph_language import numerics

OWN = 'scripts/diagnose_glyph_context_motor.py'
DECISION = 'references/glyph_context_motor_failure_decision.md'
MOTOR = ROOT/'artifacts/glyph_context_motor_v1'
FIRST = ROOT/'artifacts/glyph_tail_first_writing_v1'
NATIVE = ROOT/'references/evidence/glyph_tail_native_reader_learning_20261111.json'


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sources():
    names = [OWN, DECISION, 'ilm/visual_lm/context_motor_flow.py',
        'ilm/visual_lm/semantic_glyph_loop.py', 'scripts/train_glyph_context_motor.py',
        'scripts/audit_glyph_tail_first_writing.py', 'scripts/train_semantic_glyph_bridge.py',
        'scripts/train_semantic_glyph_language.py']
    for name in names:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) == (ROOT/name).read_bytes(),
                'Commit diagnosis source first: '+name)
    return {n:sha(ROOT/n) for n in names}


def selected(pool, count):
    ordered = sorted(range(count), key=lambda i:digest(f'motor-capacity-v1\0{pool}\0{i}'))[:32]
    return [(i, int(digest(f'motor-capacity-v1\0sample\0{pool}\0{i}')[:16], 16) % 2**63) for i in ordered]


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not output.exists(), 'Fresh CPU registration only')
    torch.set_num_threads(2); code = sources(); study(MOTOR); original = first_audit(FIRST)
    require(sha(MOTOR/'fit/checkpoint_latest.pt') == load(MOTOR/'fit/report.json')['checkpoint_sha256'],
            'Completed final EMA changed')
    output.mkdir(parents=True)
    primitive = torch.load(PRIMITIVE/'training.pt', map_location='cpu', weights_only=True)
    pools = torch.load(CACHE/'pools.pt', map_location='cpu', weights_only=True)
    motors = torch.load(MOTOR/'training_targets.pt', map_location='cpu', weights_only=True)
    old = torch.load(FIRST/'prepared_pixels.pt', map_location='cpu', weights_only=True)
    prior_pixels = np.load(FIRST/'execution/pixels.npz', allow_pickle=False)
    natives = {(r['layout'],r['case']):r for r in load(NATIVE)['records']
               if r['tail'] == 'adapted' and r['condition'] == 'normal' and r['layout'] in ['fixed_cells','proportional']}
    inventory = {tensor_hash(tile):i for i,tile in enumerate(primitive['ink'])}
    references = {r['case']:r for r in original['targets']}
    positions, states, gold, parent_ink, candidate_ink, supplied_ink, candidate_motor, noise = [], [], [], [], [], [], [], []
    for pool, count in [('first',3176),('later',20086)]:
        require(pools[pool].shape == (count,4096), 'Training pool shape changed')
        for index, seed in selected(pool, count):
            label = int(pools[pool+'_labels'][index]); state = pools[pool][index]; target = primitive['ink'][label]
            positions.append(dict(id=f'{pool}_{index:05}',group='train_'+pool,pool=pool,index=index,seed=seed,
                primitive_index=label,state_sha256=tensor_hash(state),target_sha256=tensor_hash(target)))
            states.append(state); gold.append(target)
            parent_ink.append(torch.zeros_like(target)); candidate_ink.append(torch.zeros_like(target))
            supplied_ink.append(torch.zeros_like(target)); candidate_motor.append(torch.zeros(1024)); noise.append(torch.zeros(1,1024))
    manifest = load(MOTOR/'evaluation_manifest.json'); by_id = {r['id']:r for r in manifest['inputs']}
    bindings = {}
    for index, job in enumerate(original['jobs']):
        if job['condition'] != 'normal':
            continue
        row = by_id[job['id']]; folder = MOTOR/('generation_'+job['layout'])/job['id']
        report = load(folder/'report.json'); path = folder/'legacy_output.pt'
        require(sha(path) == report['legacy_output_sha256'], 'Retained candidate stream changed')
        value = torch.load(path, map_location='cpu', weights_only=True)
        require(torch.equal(value['states'][0], old['original_first_states'][index]), 'Matched actual first state changed')
        reference = references[job['target_case']]; target = old['gold_ink'][job['target_first_offset']]
        primitive_index = inventory[tensor_hash(target)]
        native = natives[job['layout'],job['case']]; text = ''.join(native['text'].split())
        require(native['prefill_state_sha256'] == tensor_hash(value['states'][0]), 'Helper prefill binding changed')
        positions.append(dict(id=job['id'],group=job['layout'],original_case=job['case'],seed=row['seed'],
            primitive_index=primitive_index,state_sha256=tensor_hash(value['states'][0]),target_sha256=tensor_hash(target),
            canonical_first_character=reference['first_character'],native_first_character=text[:1],
            native_canonical_first_agree=text[:1] == reference['first_character'],
            actual_first_present=bool(len(value['images'])),candidate_initial_stop_probability=value['stop_probabilities'][0],
            parent_initial_stop_probability=job['initial_stop_probability'],
            candidate_whole_stopped=value['stopped'],candidate_whole_capped=value['capped']))
        states.append(value['states'][0]); gold.append(target); parent_ink.append(old['original_first_ink'][index])
        candidate_ink.append(((1-value['images'][0])/2).to(torch.uint8) if len(value['images']) else torch.zeros_like(target))
        supplied_ink.append(torch.from_numpy(prior_pixels['supplied_reconstruction_ink'][job['target_first_offset']].copy()))
        candidate_motor.append(value['motor_trace'][0]['normalized_motor'].flatten())
        noise.append(value['motor_trace'][0]['noise'])
        for p in [path,folder/'report.json']:
            bindings[str(p.relative_to(ROOT))] = sha(p)
    require(len(positions) == 128 and len({p['id'] for p in positions}) == 128, 'Every128 positions required')
    labels = torch.tensor([p['primitive_index'] for p in positions])
    atomic_tensor(output/'prepared.pt',dict(states=torch.stack(states),gold_ink=torch.stack(gold),
        parent_ink=torch.stack(parent_ink),candidate_ink=torch.stack(candidate_ink),supplied_ink=torch.stack(supplied_ink),
        candidate_normalized_motor=torch.stack(candidate_motor),noise=torch.stack(noise),
        target_motor=motors['motor'][labels],motor_mean=motors['mean'],motor_scale=motors['scale']))
    for p in [CHECKPOINT,MOTOR/'fit/checkpoint_latest.pt',MOTOR/'training_targets.pt',CACHE/'pools.pt',
              PRIMITIVE/'training.pt',FIRST/'registration.json',FIRST/'prepared_pixels.pt',
              FIRST/'execution/report.json',FIRST/'execution/pixels.npz',NATIVE,MOTOR/'review/comparison.json']:
        bindings[str(p.relative_to(ROOT))] = sha(p)
    registration = dict(schema='motor-capacity-v1',sources=code,bindings_sha256=bindings,
        prepared_sha256=sha(output/'prepared.pt'),positions=positions,sampler_intervals=32,
        samples_per_training_position=1,development_images_reused_without_resampling=True,
        diagnostic_one_glyph_regardless_of_stop=True,stopping_reported_separately=True,
        training_gate=dict(median_foreground_f1_minimum=.99,at_least_095_minimum=30,count=32),
        new_optimizer_updates=0,new_contextual_body_calls=0,whole_language_scores_unchanged=True,
        maximum_seconds=600,maximum_allocated_vram_gib=8,pressure_wait_allowance_seconds=120,
        minimum_entry_available_gib=48,minimum_runtime_available_gib=24,CUDA_initialized=torch.cuda.is_initialized())
    require(not registration['CUDA_initialized'], 'CPU preparation initialized CUDA')
    save_json(output/'registration.json',registration)
    public = ROOT/'references/evidence/glyph_context_motor_capacity_registration_20261111.json'
    require(not public.exists(), 'Preserve previous registration')
    save_json(public,registration)
    print(json.dumps(dict(positions=128,training=64,retained_development=64,registration_sha256=sha(output/'registration.json'))))


def validate(output):
    registration = load(output/'registration.json')
    require(registration['sources'] == sources() and registration['prepared_sha256'] == sha(output/'prepared.pt'),
            'Registered diagnostic source or prepared states changed')
    for name, expected in registration['bindings_sha256'].items():
        require(sha(ROOT/name) == expected, 'Bound source changed: '+name)
    public = ROOT/'references/evidence/glyph_context_motor_capacity_registration_20261111.json'
    require(load(public) == registration and subprocess.check_output(['git','show','HEAD:'+str(public.relative_to(ROOT))],
            cwd=ROOT) == public.read_bytes(), 'Commit all position identities before sampling')
    return registration


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Physical GPU0 diagnosis only')
    registration = validate(output); execution = output/'execution'; execution.mkdir(exist_ok=False)
    monitor = Monitor(execution); monitor.seconds = 600; monitor.wait_allowance = 120; monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission), torch.no_grad():
            numerics(); torch.set_num_threads(2); monitor.guard()
            data = torch.load(output/'prepared.pt',map_location='cpu',weights_only=True)
            checkpoint = torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
            parent = SemanticGlyphBridge().to(device).eval().requires_grad_(False)
            parent.load_state_dict(checkpoint['model']); del checkpoint
            checkpoint = torch.load(MOTOR/'fit/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
            flow = ContextMotorFlow().to(device).eval().requires_grad_(False); flow.load_state_dict(checkpoint['ema'])
            require(torch.equal(data['motor_mean'],checkpoint['motor_mean'])
                    and torch.equal(data['motor_scale'],checkpoint['motor_scale']), 'Motor normalization changed')
            del checkpoint
            versions = [(p,p._version) for m in [parent,flow] for p in m.parameters()]
            arrays = {k:data[k].clone() for k in ['gold_ink','parent_ink','candidate_ink','supplied_ink','noise']}
            predicted = data['candidate_normalized_motor'].clone(); parent_motors = []; records = []
            mean,scale = (data[k].to(device) for k in ['motor_mean','motor_scale'])
            for index, position in enumerate(registration['positions']):
                monitor.step=index; monitor.guard(); monitor.status('conditional_writing',position=position['id'])
                h = data['states'][index:index+1].to(device)
                old_motor = parent.decoder_input(parent.readout(h)).float()
                parent_motors.append(((old_motor-mean)/scale).cpu().flatten())
                stop = float(parent.stop(h).sigmoid().flatten()[0])
                if position['group'].startswith('train_'):
                    generator = torch.Generator(device=device).manual_seed(position['seed'])
                    normalized, noise = flow.sample(h,generator=generator)
                    predicted[index] = normalized.cpu().flatten(); arrays['noise'][index] = noise.cpu()
                    logits = parent.decoder((normalized*scale+mean).reshape(1,64,4,4))
                    arrays['candidate_ink'][index] = (logits.sigmoid()>=.5).to(torch.uint8).cpu()[0]
                    arrays['parent_ink'][index] = (parent.decoder(old_motor.reshape(1,64,4,4)).sigmoid()>=.5).to(torch.uint8).cpu()[0]
                    arrays['supplied_ink'][index] = (parent.decoder(data['target_motor'][index:index+1].to(device)
                        .reshape(1,64,4,4)).sigmoid()>=.5).to(torch.uint8).cpu()[0]
                target_normalized=(data['target_motor'][index]-data['motor_mean'])/data['motor_scale']
                scores={route:pixels_metrics(arrays[key][index].numpy(),arrays['gold_ink'][index].numpy(),
                    position.get('actual_first_present',True) if route=='candidate' else True)
                    for route,key in [('candidate','candidate_ink'),('parent','parent_ink'),('codec','supplied_ink')]}
                records.append(dict(**position,scores=scores,stop_probability=stop,
                    candidate_normalized_motor_MSE=float((predicted[index]-target_normalized).square().mean()),
                    parent_normalized_motor_MSE=float((parent_motors[-1]-target_normalized).square().mean())))
            monitor.guard(); require(all(p._version==v for p,v in versions), 'Frozen diagnostic parameter changed')
            atomic_tensor(execution/'raw.pt',dict(**arrays,candidate_normalized_motor=predicted,
                parent_normalized_motor=torch.stack(parent_motors),target_motor=data['target_motor']))
            np.savez_compressed(execution/'pixels.npz',**{k:v.numpy() for k,v in arrays.items() if k!='noise'})
            save_json(execution/'report.json',dict(registration_sha256=sha(output/'registration.json'),records=records,
                raw_sha256=sha(execution/'raw.pt'),pixels_sha256=sha(execution/'pixels.npz'),admission=admission,
                all_parameters_frozen=True,new_optimizer_updates=0,new_contextual_body_calls=0,
                development_images_reused_without_resampling=True,whole_language_scores_unchanged=True,
                supplied_or_cached_state_diagnosis_not_language=True,goal_achieved=False,**monitor.receipt()))
        monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:
        save_json(execution/'status.json',dict(load(execution/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_context_motor_capacity_v1')
    args=parser.parse_args();{'prepare':prepare,'run':run}[args.stage](args.output.resolve())
