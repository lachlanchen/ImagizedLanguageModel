#!/usr/bin/env python3
"""Paired full-sampler endpoint fine-tuning, with all contextual weights frozen."""
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
from ilm.visual_lm.rollout_endpoint import heun_endpoint,balanced_ink_bce
from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge,tensor_hash
from scripts.infer_glyph_context_motor import study
from scripts.score_pixel_language_first_blocks import require
from scripts.train_glyph_context_motor import load,Monitor,CACHE,PRIMITIVE,CHECKPOINT,CHECKPOINT_SHA
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.train_semantic_glyph_language import numerics

BASE=ROOT/'artifacts/glyph_context_motor_v1'
CAPACITY=ROOT/'artifacts/glyph_context_motor_capacity_v1'
RUN=ROOT/'artifacts/glyph_rollout_endpoint_v1'
PROTOCOL='references/rollout_endpoint_motor_decision.md'
OWN='scripts/train_rollout_endpoint_motor.py'
PUBLIC=ROOT/'references/evidence/rollout_endpoint_motor_registration_20261111.json'
START_SHA='79160d69ecd8559a939cba144566610e61f3582b92e9ff3e8e141eb39bcdd976'
NAMES=[OWN,PROTOCOL,'ilm/visual_lm/rollout_endpoint.py','ilm/visual_lm/context_motor_flow.py',
    'ilm/visual_lm/semantic_glyph_loop.py','scripts/train_glyph_context_motor.py',
    'scripts/train_semantic_glyph_bridge.py','scripts/train_semantic_glyph_language.py',
    'scripts/assess_rollout_endpoint_motor.py','tests/test_rollout_endpoint.py']


def sources():
    for name in NAMES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit the prospective recipe and implementation first: '+name)
    return {n:sha(ROOT/n) for n in NAMES}


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not run.exists(),'New CPU registration only')
    torch.set_num_threads(2);code=sources();study(BASE)
    require(sha(BASE/'fit/checkpoint_latest.pt')==START_SHA,'Original final flow checkpoint changed')
    names=[BASE/'registration.json',BASE/'fit/report.json',BASE/'fit/checkpoint_latest.pt',
        BASE/'training_targets.pt',BASE/'evaluation_manifest.json',CACHE/'pools.pt',PRIMITIVE/'training.pt',
        CHECKPOINT,CAPACITY/'registration.json',CAPACITY/'prepared.pt',CAPACITY/'execution/raw.pt',
        CAPACITY/'execution/report.json',ROOT/'references/evidence/decoder_aware_motor_learning_20261111.json']
    previous=load(names[-1]);require(not previous['full_language_generation_run']
        and not any(previous['training_writing_gates'].values()),'Finish the preceding rejection first')
    checkpoint=torch.load(BASE/'fit/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
    initial={k:tensor_hash(v) for k,v in checkpoint['ema'].items()}
    registration=dict(schema='rollout-endpoint-motor-pair-v1',sources=code,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        bindings_sha256={str(p.relative_to(ROOT)):sha(p) for p in names},
        start_checkpoint_sha256=START_SHA,initial_parameter_sha256=initial,
        start_from_original_flow_final_EMA=True,failed_local_loss_checkpoint_used=False,
        arms=['control','rollout'],seed=202611181,fit_steps=2000,parameters=24477696,
        first_per_update=16,later_per_update=16,expected_sampled_target_counts=dict(first=32000,later=32000),
        optimizer=dict(name='AdamW',learning_rate=.00002,betas=[.9,.99],weight_decay=.01,
                       clip_norm=1.,warmup_updates=100,final_learning_rate_fraction=.1),
        EMA_decay=.99,final_EMA_only=True,endpoint_BCE_coefficient=1.,
        gradient_through_all32_intervals=True,recomputed_deterministic_intervals=True,
        no_detached_prefix_or_custom_adjoint=True,same_noise_for_FM_and_rollout=True,
        sampler='unchanged32Heun intervals,last Euler,one sample per glyph',
        frozen_reader_adapters_encoder_decoder_normalization_stop=True,full_reader_loaded=False,
        training_gate=dict(median_foreground_f1_minimum=.99,at_least_095_minimum=30,per_pool_count=32),
        complete_language_pilot=dict(exact_minimum=10,pairs_minimum=2,readable_minimum=16,
            retain_union_proportional_exact=8,retain_original_fixed_exact=2,
            matched_control_retention_and_no_count_regression_required=True,
            strictly_exceed_matched_control_exact_or_pairs_required=True),
        candidate_physical_gate_required_before_both_complete_language_trials=True,
        untouched_confirmation_required=True,maximum_seconds=dict(control=1800,rollout=3600),
        pressure_wait_allowance_seconds=600,maximum_allocated_vram_gib=8,
        minimum_entry_available_gib=48,minimum_runtime_available_gib=24,
        language_accuracy_unmeasured=True,goal_achieved=False)
    run.mkdir(parents=True);save_json(run/'registration.json',registration)
    require(not PUBLIC.exists(),'Preserve previous registration');save_json(PUBLIC,registration)
    print(json.dumps(dict(registration_sha256=sha(run/'registration.json'),arms=2,fit_not_started=True)))


def validate(run):
    r=load(run/'registration.json');require(r['sources']==sources(),'Registered sources changed')
    for name,digest in r['bindings_sha256'].items():require(sha(ROOT/name)==digest,'Bound input changed: '+name)
    require(load(PUBLIC)==r and subprocess.check_output(['git','show','HEAD:'+str(PUBLIC.relative_to(ROOT))],
        cwd=ROOT)==PUBLIC.read_bytes(),'Commit the registration before fitting')
    return r


def fit(run,arm):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','One worker on physicalGPU0')
    r=validate(run)
    if arm=='rollout':
        require(load(run/'fit_control/status.json')['phase']=='complete','Finish the matched control first')
    folder=run/('fit_'+arm);folder.mkdir(exist_ok=False)
    monitor=Monitor(folder);monitor.seconds=r['maximum_seconds'][arm];monitor.status('admitting')
    try:
        with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device,admission):
            numerics();torch.set_num_threads(2);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
            monitor.guard();save_json(folder/'admission.json',admission)
            data=torch.load(BASE/'training_targets.pt',map_location='cpu',weights_only=True)
            targets=((data['motor']-data['mean'])/data['scale']).to(device)
            pools=torch.load(CACHE/'pools.pt',map_location='cpu',weights_only=True)
            p={k:v.to(device) for k,v in pools.items() if k!='terminal'};del pools
            images=torch.load(PRIMITIVE/'training.pt',map_location='cpu',weights_only=True)['ink'].to(device)
            flow=ContextMotorFlow().to(device)
            starting=torch.load(BASE/'fit/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
            flow.load_state_dict(starting['ema']);del starting
            require({k:tensor_hash(v) for k,v in flow.state_dict().items()}==r['initial_parameter_sha256'],
                    'Warm-start EMA differs from the matched control')
            ema=copy.deepcopy(flow).eval().requires_grad_(False)
            with torch.random.fork_rng(devices=[0]):
                parent=SemanticGlyphBridge().to(device).eval().requires_grad_(False)
                parent.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)['model'])
            frozen=[(v,v._version) for v in list(parent.parameters())+list(parent.buffers())]
            # Same actual states and explicit local noise in both arms; no global random draw.
            with torch.no_grad():
                h=p['first'][:2]
                deployed,noise=flow.sample(h,generator=torch.Generator(device=device).manual_seed(202611182))
                direct=heun_endpoint(flow,h,noise,recompute=False)
                recomputed=heun_endpoint(flow,h,noise,recompute=True)
                require(torch.equal(deployed,direct) and torch.equal(direct,recomputed),
                        'Differentiable forward differs from the deployed sampler')
            optimizer=torch.optim.AdamW(flow.parameters(),lr=.00002,betas=(.9,.99),weight_decay=.01)
            sampler=hashlib.sha256();counts=dict(first=0,later=0)
            mean,scale=data['mean'].to(device),data['scale'].to(device)
            for step in range(1,r['fit_steps']+1):
                monitor.step=step
                if step%10==1:monitor.guard()
                rate=min(1.,step/100)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-100)/1900)))
                for group in optimizer.param_groups:group['lr']=.00002*rate
                optimizer.zero_grad(set_to_none=True);states=[];labels=[]
                for name in ['first','later']:
                    ix=torch.randint(len(p[name]),(16,),device=device)
                    sampler.update(name.encode());sampler.update(ix.cpu().numpy().tobytes());counts[name]+=16
                    states.append(p[name][ix]);labels.append(p[name+'_labels'][ix])
                labels=torch.cat(labels);state=torch.cat(states);y=targets[labels]
                noise=torch.randn_like(y);t=torch.rand(len(y),device=device)
                mixed=(1-t[:,None])*noise+t[:,None]*y
                with torch.autocast('cuda',dtype=torch.bfloat16):velocity=flow(mixed,t,state)
                velocity_loss=(velocity-(y-noise)).square().mean()
                pixel_loss=velocity_loss.detach()*0
                if arm=='rollout':
                    endpoint=heun_endpoint(flow,state,noise,recompute=True)
                    logits=parent.decoder((endpoint*scale+mean).reshape(-1,64,4,4)).float()
                    pixel_loss=balanced_ink_bce(logits,images[labels])
                loss=velocity_loss+pixel_loss
                require(bool(torch.isfinite(loss)),'Nonfinite paired loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(flow.parameters(),1.)
                require(bool(torch.isfinite(norm)),'Nonfinite full-path gradient');optimizer.step()
                with torch.no_grad():
                    for avg,param in zip(ema.parameters(),flow.parameters()):avg.lerp_(param,.01)
                if step==1 or step%50==0:
                    row=dict(step=step,arm=arm,velocity_MSE=float(velocity_loss.detach()),
                        realized_endpoint_BCE=float(pixel_loss.detach()),total_loss=float(loss.detach()),
                        gradient_norm=float(norm),learning_rate=.00002*rate,
                        elapsed_seconds=monitor.receipt()['elapsed_seconds'])
                    with (folder/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    monitor.status('training_'+arm,**row);print(json.dumps(row),flush=True)
                if step%250==0:
                    path=folder/'checkpoint_latest.pt'
                    if path.exists():path.replace(folder/'checkpoint_previous.pt')
                    atomic_tensor(path,dict(schema='rollout-endpoint-checkpoint-v1',arm=arm,
                        model={k:v.detach().cpu() for k,v in flow.state_dict().items()},
                        ema={k:v.detach().cpu() for k,v in ema.state_dict().items()},optimizer=optimizer.state_dict(),
                        motor_mean=data['mean'],motor_scale=data['scale'],step=step,
                        preparation_sha256=sha(run/'registration.json'),parent_checkpoint_sha256=CHECKPOINT_SHA,
                        sampled_target_counts=counts.copy(),sampled_indices_sha256=sampler.hexdigest(),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                        training_complete=step==r['fit_steps'],language_accuracy_not_measured=True))
            monitor.guard();torch.cuda.synchronize()
            require(counts==r['expected_sampled_target_counts'],'Paired exposure differs')
            require(all(v._version==version for v,version in frozen),'Frozen bridge changed')
            require(all(torch.isfinite(v).all() for v in ema.parameters()),'Nonfinite final EMA')
            require(any(tensor_hash(v)!=r['initial_parameter_sha256'][k] for k,v in flow.state_dict().items()),
                    'No real parameter update')
            if arm=='rollout':
                control=torch.load(run/'fit_control/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
                require(sampler.hexdigest()==control['sampled_indices_sha256'],'Paired examples differ')
                require(torch.equal(torch.get_rng_state(),control['torch_rng']) and
                    torch.equal(torch.cuda.get_rng_state().cpu(),control['cuda_rng'].cpu()),'Paired random stream differs')
            # Audit actual optimizer step tensors, not only the Python loop index.
            require(len(optimizer.state)==len(list(flow.parameters())) and
                all(int(v['step'])==2000 for v in optimizer.state.values()),'Optimizer update audit failed')
            save_json(folder/'report.json',dict(schema='rollout-endpoint-fit-v1',arm=arm,
                registration_sha256=sha(run/'registration.json'),checkpoint_sha256=sha(folder/'checkpoint_latest.pt'),
                actual_optimizer_updates=2000,actual_trainable_parameters=24477696,
                sampled_target_counts=counts,sampled_indices_sha256=sampler.hexdigest(),
                initial_EMA_identity_verified=True,actual_GPU_sampler_forward_bit_exact=True,
                paired_final_RNG_and_exposure_verified=arm=='rollout',
                frozen_bridge_parameter_versions_unchanged=True,all_optimizer_parameter_steps_verified=True,
                training_complete=True,final_EMA_only=True,full_reader_loaded=False,
                inherited_body_parameters_updated=0,language_accuracy_not_measured=True,goal_achieved=False,
                **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit']);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--arm',choices=['control','rollout'])
    args=parser.parse_args()
    if args.stage=='fit':
        require(args.arm is not None,'Select a registered arm');fit(args.run.resolve(),args.arm)
    else:prepare(args.run.resolve())
