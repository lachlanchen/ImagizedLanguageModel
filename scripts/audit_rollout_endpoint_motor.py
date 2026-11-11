#!/usr/bin/env python3
"""Independent CPU inspection of both actual paired optimizer/checkpoint states."""
import json
import os
from pathlib import Path

import torch

from ilm.image_edit.baseline import ROOT,sha,save_json
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.score_pixel_language_first_blocks import require
from scripts.train_rollout_endpoint_motor import RUN,validate,START_SHA,BASE


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU-only optimizer audit')
    torch.set_num_threads(2);r=validate(RUN);records={};finals={}
    start=torch.load(BASE/'fit/checkpoint_latest.pt',map_location='cpu',weights_only=True,mmap=True)
    require({k:tensor_hash(v) for k,v in start['ema'].items()}==r['initial_parameter_sha256'],
            'Registered initial EMA is not the original saved model')
    for arm in ['control','rollout']:
        folder=RUN/('fit_'+arm);report=json.loads((folder/'report.json').read_text())
        require(report['training_complete'] and report['actual_optimizer_updates']==2000,'Complete both fits first')
        checkpoint=folder/'checkpoint_latest.pt';require(sha(checkpoint)==report['checkpoint_sha256'],'Checkpoint identity differs')
        d=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        require(d['step']==2000 and d['training_complete'] and d['arm']==arm,'Final paired checkpoint required')
        require(torch.equal(d['motor_mean'],start['motor_mean']) and
            torch.equal(d['motor_scale'],start['motor_scale']),'Motor normalization changed')
        changed=[]
        for name,value in d['model'].items():
            require(bool(torch.isfinite(value).all()),'Nonfinite trained parameter')
            require(bool(torch.isfinite(d['ema'][name]).all()),'Nonfinite EMA')
            if tensor_hash(value)!=r['initial_parameter_sha256'][name]:changed.append(name)
        require(changed,'No actual flow update')
        require(set(d['model'])==set(r['initial_parameter_sha256']),'Parameter map differs')
        optimizer=d['optimizer'];ids=[i for group in optimizer['param_groups'] for i in group['params']]
        require(len(ids)==len(set(ids))==len(d['model'])==len(optimizer['state']),'Optimizer parameter coverage differs')
        for i in ids:
            state=optimizer['state'][i]
            require(int(state['step'])==2000,'Actual optimizer parameter step differs')
            for key in ['exp_avg','exp_avg_sq']:require(bool(torch.isfinite(state[key]).all()),'Nonfinite optimizer moment')
        require(d['sampled_target_counts']==dict(first=32000,later=32000),'Training exposure differs')
        require(d['preparation_sha256']==sha(RUN/'registration.json'),'Unregistered checkpoint')
        previous=torch.load(folder/'checkpoint_previous.pt',map_location='cpu',weights_only=True,mmap=True)
        require(previous['step']==1750 and not previous['training_complete'],'Preserve previous scheduled checkpoint')
        records[arm]=dict(checkpoint_sha256=sha(checkpoint),report_sha256=sha(folder/'report.json'),
            actual_parameter_optimizer_steps=2000,changed_parameters=len(changed),
            finite_model_EMA_and_optimizer_moments=True,sampled_target_counts=d['sampled_target_counts'],
            previous_checkpoint_step=1750,**{k:report[k] for k in ['elapsed_seconds','resource_wait_seconds','peak_allocated_vram_gib']})
        finals[arm]={k:d[k].clone() if isinstance(d[k],torch.Tensor) else d[k]
            for k in ['torch_rng','cuda_rng','sampled_indices_sha256']}
    require(finals['control']['sampled_indices_sha256']==finals['rollout']['sampled_indices_sha256'],
            'Paired sampled indices differ')
    require(torch.equal(finals['control']['torch_rng'],finals['rollout']['torch_rng'])
        and torch.equal(finals['control']['cuda_rng'],finals['rollout']['cuda_rng']),'Paired final random stream differs')
    result=dict(schema='rollout-endpoint-independent-fit-audit-v1',source_sha256=sha(Path(__file__)),
        registration_sha256=sha(RUN/'registration.json'),starting_flow_checkpoint_sha256=START_SHA,
        arms=records,actual_initialization_bound=True,paired_indices_and_final_CPU_CUDA_RNG_bit_exact=True,
        actual_checkpoint_optimizer_states_audited=True,language_quality_unmeasured=True,goal_achieved=False,
        CPU_CUDA_initialized=torch.cuda.is_initialized())
    save_json(RUN/'fit_audit.json',result)
    save_json(ROOT/'references/evidence/rollout_endpoint_motor_fit_20261111.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
