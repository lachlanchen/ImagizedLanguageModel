#!/usr/bin/env python3
"""Paired full motor fit with one additional decoded endpoint objective."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow
from ilm.visual_lm.decoder_endpoint_loss import decoder_endpoint_loss
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,tensor_hash
from scripts.infer_glyph_context_motor import study
from scripts.score_pixel_language_first_blocks import require
from scripts.train_glyph_context_motor import load,Monitor,CACHE,PRIMITIVE,CHECKPOINT,CHECKPOINT_SHA
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.train_semantic_glyph_language import numerics

BASE=ROOT/'artifacts/glyph_context_motor_v1'
CAPACITY=ROOT/'artifacts/glyph_context_motor_capacity_v1'
OWN='scripts/train_decoder_aware_motor.py'
PROTOCOL='references/decoder_aware_motor_decision.md'


def sources():
    names=[OWN,PROTOCOL,'ilm/visual_lm/decoder_endpoint_loss.py','ilm/visual_lm/context_motor_flow.py',
        'ilm/visual_lm/semantic_glyph_loop.py','scripts/train_glyph_context_motor.py',
        'scripts/train_semantic_glyph_bridge.py','scripts/train_semantic_glyph_language.py']
    for name in names:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit objective sources before registration: '+name)
    return {name:sha(ROOT/name) for name in names}


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not run.exists(),'New CPU registration only')
    torch.set_num_threads(2);code=sources();study(BASE)
    prior=load(BASE/'fit/report.json');capacity=load(ROOT/'references/evidence/glyph_context_motor_capacity_learning_20261111.json')
    require(not any(capacity['training_writing_gates'].values()) and capacity['all512_displayed_tiles_bit_exact'],
            'Complete the original writing failure diagnosis first')
    names=[BASE/'registration.json',BASE/'fit/report.json',BASE/'fit/checkpoint_latest.pt',
        BASE/'evaluation_manifest.json',BASE/'training_targets.pt',CACHE/'pools.pt',PRIMITIVE/'training.pt',
        CHECKPOINT,CAPACITY/'registration.json',CAPACITY/'prepared.pt',
        ROOT/'references/evidence/glyph_context_motor_capacity_learning_20261111.json']
    bindings={str(p.relative_to(ROOT)):sha(p) for p in names}
    torch.manual_seed(202611171);initial=ContextMotorFlow()
    initial_hashes={k:tensor_hash(v) for k,v in initial.state_dict().items()}
    run.mkdir(parents=True)
    registration=dict(schema='decoder-aware-motor-v1',sources=code,bindings_sha256=bindings,
        original_control_registration_sha256=sha(BASE/'registration.json'),
        control_sampled_indices_sha256=prior['sampled_indices_sha256'],
        initial_parameter_sha256=initial_hashes,seed=202611171,fit_steps=12000,parameters=24477696,
        first_per_update=64,later_per_update=64,optimizer='same AdamW2e-4,beta(.9,.99),wd.01,clip1,200warmup,cosine to10%',
        decoded_endpoint_coefficient=1.,eligible_times_at_least=.75,
        pixel_loss='per-image foreground/background balanced BCE; mean of eligible samples',
        initialization_from_scratch_matching_control=True,not_a_warm_start_from_fitted_control=True,
        exact_original_training_exposure_required=True,no_extra_training_random_draws=True,
        final_EMA_only=True,sampler='unchanged one sample32Heun,last Euler',
        frozen_decoder_and_encoder=True,full_reader_loaded=False,
        training_diagnosis_positions_sha256=sha(CAPACITY/'registration.json'),
        training_gate=dict(median_foreground_f1_minimum=.99,at_least_095_minimum=30,per_pool_count=32),
        complete_language_pilot=dict(exact_minimum=10,pairs_minimum=2,readable_minimum=16,
            retain_union_proportional_exact=8,retain_original_fixed_exact=2),
        physical_gate_required_before_full216_language_generation=True,untouched_confirmation_required=True,
        maximum_seconds=1800,pressure_wait_allowance_seconds=600,maximum_allocated_vram_gib=8,
        minimum_entry_available_gib=48,minimum_runtime_available_gib=24,
        usable_language_qualified=False,goal_achieved=False)
    save_json(run/'registration.json',registration)
    path=ROOT/'references/evidence/decoder_aware_motor_registration_20261111.json'
    require(not path.exists(),'Preserve previous experiment registration');save_json(path,registration)
    print(json.dumps(dict(registration_sha256=sha(run/'registration.json'),parameters=24477696,fit_not_started=True)))


def validate(run):
    value=load(run/'registration.json');require(value['sources']==sources(),'Registered objective source changed')
    for name,digest in value['bindings_sha256'].items():require(sha(ROOT/name)==digest,'Bound input changed: '+name)
    public=ROOT/'references/evidence/decoder_aware_motor_registration_20261111.json'
    require(load(public)==value and subprocess.check_output(['git','show','HEAD:'+str(public.relative_to(ROOT))],
        cwd=ROOT)==public.read_bytes(),'Commit experiment registration before fitting')
    return value


def fit(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Physical GPU0 fit only')
    r=validate(run);folder=run/'fit';folder.mkdir(exist_ok=False)
    monitor=Monitor(folder);monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission):
            numerics();torch.set_num_threads(2);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            monitor.guard();save_json(folder/'admission.json',admission)
            data=torch.load(BASE/'training_targets.pt',map_location='cpu',weights_only=True)
            targets=((data['motor']-data['mean'])/data['scale']).to(device)
            pools=torch.load(CACHE/'pools.pt',map_location='cpu',weights_only=True)
            p={k:v.to(device) for k,v in pools.items() if k!='terminal'};del pools
            images=torch.load(PRIMITIVE/'training.pt',map_location='cpu',weights_only=True)['ink'].to(device)
            flow=ContextMotorFlow().to(device);ema=copy.deepcopy(flow).eval().requires_grad_(False)
            require({k:tensor_hash(v) for k,v in flow.state_dict().items()}==r['initial_parameter_sha256'],
                    'Paired control initialization differs')
            with torch.random.fork_rng(devices=[0]):
                parent=SemanticGlyphBridge().to(device).eval().requires_grad_(False)
                parent.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['model'])
            frozen=[(v,v._version) for v in parent.parameters()]
            optimizer=torch.optim.AdamW(flow.parameters(),lr=.0002,betas=(.9,.99),weight_decay=.01)
            sampler=hashlib.sha256();counts=dict(first=0,later=0);eligible_total=0
            mean,scale=data['mean'].to(device),data['scale'].to(device)
            for step in range(1,12001):
                monitor.step=step
                if step%10==1:monitor.guard()
                rate=min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-200)/11800)))
                for group in optimizer.param_groups:group['lr']=.0002*rate
                optimizer.zero_grad(set_to_none=True);states=[];labels=[]
                for name in ['first','later']:
                    ix=torch.randint(len(p[name]),(64,),device=device)
                    sampler.update(name.encode());sampler.update(ix.cpu().numpy().tobytes());counts[name]+=64
                    states.append(p[name][ix]);labels.append(p[name+'_labels'][ix])
                labels=torch.cat(labels);state=torch.cat(states);y=targets[labels]
                noise=torch.randn_like(y);t=torch.rand(len(y),device=device)
                mixed=(1-t[:,None])*noise+t[:,None]*y
                with torch.autocast('cuda',dtype=torch.bfloat16):velocity=flow(mixed,t,state)
                velocity_loss=(velocity-(y-noise)).square().mean()
                pixel_loss,eligible=decoder_endpoint_loss(velocity,mixed,t,mean,scale,parent.decoder,images[labels])
                eligible_total+=eligible;loss=velocity_loss+pixel_loss
                require(bool(torch.isfinite(loss)),'Nonfinite paired objective')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),1.)
                require(bool(torch.isfinite(norm)),'Nonfinite paired gradient');optimizer.step()
                decay=min(.999,(1+step)/(10+step))
                with torch.no_grad():
                    for average,param in zip(ema.parameters(),flow.parameters()):average.lerp_(param,1-decay)
                if step==1 or step%100==0:
                    row=dict(step=step,velocity_MSE=float(velocity_loss.detach()),decoded_BCE=float(pixel_loss.detach()),
                        total_loss=float(loss.detach()),eligible_samples=eligible,gradient_norm=float(norm),learning_rate=.0002*rate)
                    with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    monitor.status('training_decoder_aware_motor',**row);print(json.dumps(row),flush=True)
                if step%1000==0:
                    path=folder/'checkpoint_latest.pt'
                    if path.exists():path.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(path,dict(schema='decoder-aware-motor-checkpoint-v1',
                        model={k:v.detach().cpu() for k,v in flow.state_dict().items()},
                        ema={k:v.detach().cpu() for k,v in ema.state_dict().items()},optimizer=optimizer.state_dict(),
                        motor_mean=data['mean'],motor_scale=data['scale'],step=step,
                        preparation_sha256=sha(run/'registration.json'),parent_checkpoint_sha256=CHECKPOINT_SHA,
                        sampled_target_counts=counts.copy(),sampled_indices_sha256=sampler.hexdigest(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                        training_complete=step==12000,language_accuracy_not_measured=True))
            monitor.guard();torch.cuda.synchronize()
            require(counts==dict(first=768000,later=768000)
                and sampler.hexdigest()==r['control_sampled_indices_sha256'],'Paired training examples differ')
            control=torch.load(BASE/'fit/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
            require(torch.equal(torch.get_rng_state(),control['torch_rng'])
                and torch.equal(torch.cuda.get_rng_state().cpu(),control['cuda_rng'].cpu()),'Paired random stream differs')
            require(all(v._version==version for v,version in frozen),'Frozen bridge weights changed')
            require(all(torch.isfinite(v).all() for v in ema.parameters()),'Nonfinite final EMA')
            save_json(folder/'report.json',dict(registration_sha256=sha(run/'registration.json'),
                checkpoint_sha256=sha(folder/'checkpoint_latest.pt'),actual_optimizer_updates=12000,
                actual_trainable_parameters=24477696,sampled_target_counts=counts,sampled_indices_sha256=sampler.hexdigest(),
                eligible_pixel_supervision_samples=eligible_total,paired_initialization_and_final_RNG_verified=True,
                original_parent_unchanged=True,frozen_bridge_parameter_versions_unchanged=True,
                training_complete=True,final_EMA_only=True,full_reader_loaded=False,
                inherited_body_parameters_updated=0,language_accuracy_not_measured=True,goal_achieved=False,
                **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','fit'])
    parser.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_decoder_aware_motor_v1')
    args=parser.parse_args();{'prepare':prepare,'fit':fit}[args.stage](args.run.resolve())
