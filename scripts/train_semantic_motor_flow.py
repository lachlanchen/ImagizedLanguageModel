#!/usr/bin/env python3
"""Train/qualify a joint continuous motor prior on the existing glyph primitives."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_motor_flow import SemanticMotorFlow,generated_motor_images
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts import train_semantic_glyph_language as language
from scripts import distill_semantic_glyph_readout as base
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.score_pixel_language_first_blocks import require

LANGUAGE=ROOT/'artifacts/semantic_glyph_language_v1'
FILES=('ilm/visual_lm/semantic_motor_flow.py','ilm/visual_lm/semantic_writing_model.py',
       'scripts/train_semantic_motor_flow.py','references/semantic_motor_flow_protocol.md',
       'scripts/distill_semantic_glyph_readout.py')


def identity():
    values={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit joint motor sources first: '+n)
    return values


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation')
    require(not output.exists(),'preserve every existing and partial motor run')
    original=language.check(LANGUAGE);path=LANGUAGE/'fit/checkpoint_latest.pt'
    require(sha(path)==language.load(LANGUAGE/'fit/report.json')['checkpoint_sha256'],'language baseline changed')
    output.mkdir(parents=True)
    save_json(output/'preparation.json',dict(original,motor_code_sha256=identity(),
        parent_run=language.relative(LANGUAGE),parent_preparation_sha256=sha(LANGUAGE/'preparation.json'),
        baseline=language.relative(path),baseline_sha256=sha(path),
        seed=202611107,assessment_seed=202611108,fit_steps=8000,batch_size=128,learning_rate=.0001,
        sigma_min=.001,width=4096,layers=3,sampling_steps=32,sampling_candidates=4,
        gaussian_maximum=.8,mixture_minimum=.25,mixture_maximum=.45,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,maximum_vram_gib=8.,
        maximum_stage_seconds=1800.,maximum_pressure_wait_seconds=300.,
        no_language_targets=True,no_development_training=True,encoder_decoder_readout_stop_frozen=True,
        full_rank_velocity_output=True,selection_uses_only_learned_actual_image_encoder=True,
        primitive_gate=dict(clean_mean_f1=.99,clean_fraction095=.95,mixture_mean_f1=.95,mixture_fraction095=.8)))


def check(output):
    meta=language.load(output/'preparation.json');language.check(ROOT/meta['parent_run'])
    require(meta['motor_code_sha256']==identity(),'motor implementation changed')
    require(sha(ROOT/meta['parent_run']/'preparation.json')==meta['parent_preparation_sha256'] and
            sha(ROOT/meta['baseline'])==meta['baseline_sha256'],'baseline changed')
    return meta


def training_codes(meta,device):
    data=torch.load(ROOT/meta['primitive']/'training.pt',map_location='cpu',weights_only=True)
    codes=((data['semantic']-data['mean'])/data['scale']).to(device)
    return F.normalize(codes,dim=-1)*math.sqrt(4096),data


def fit(output,meta,monitor,device):
    torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    codes,_=training_codes(meta,device)
    model=SemanticMotorFlow(meta['width'],meta['layers']).to(device)
    ema=copy.deepcopy(model).requires_grad_(False).eval()
    parameters=list(model.parameters());averaged=list(ema.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=meta['learning_rate'],betas=(.9,.99),weight_decay=.01,fused=True)
    for step in range(1,meta['fit_steps']+1):
        if step%10==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        ids=torch.randint(len(codes),(meta['batch_size'],),device=device)
        other=(ids+torch.randint(1,len(codes),(len(ids),),device=device))%len(codes)
        target=codes[ids];route=torch.rand(len(ids),1,device=device)
        strength=meta['mixture_minimum']+(meta['mixture_maximum']-meta['mixture_minimum'])*torch.rand(len(ids),1,device=device)
        noise=torch.randn_like(target)
        mixed=(1-strength)*target+strength*codes[other]+.05*noise
        blurred=target+meta['gaussian_maximum']*torch.rand(len(ids),1,device=device)*noise
        c=torch.where(route<.6,mixed,torch.where(route<.8,blurred,target))
        c=F.normalize(c,dim=-1)*math.sqrt(4096)
        eps=torch.randn_like(target);t=torch.rand(len(ids),device=device)
        x=(1-(1-meta['sigma_min'])*t[:,None])*eps+t[:,None]*target
        velocity=target-(1-meta['sigma_min'])*eps
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x,t,c)
        loss=F.mse_loss(prediction,velocity)
        require(bool(torch.isfinite(loss)),'nonfinite motor velocity loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        decay=min(.999,(1+step)/(10+step))
        with torch.no_grad():torch._foreach_lerp_(averaged,parameters,1-decay)
        fraction=step/meta['fit_steps'];warm=min(1.,step/200)
        for group in optimizer.param_groups:group['lr']=meta['learning_rate']*warm*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
        if step%100==0:
            row=dict(step=step,velocity_mse=float(loss),ema_decay=decay,seconds=time.perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('fitting_joint_motor');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt'
    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in ema.state_dict().items()},
        step=meta['fit_steps'],width=meta['width'],layers=meta['layers'],ema=True,
        primitive_only=True,language_generation_qualified=False,
        preparation_sha256=sha(output/'preparation.json'),base_checkpoint_sha256=meta['baseline_sha256']))
    save_json(monitor.path/'report.json',dict(checkpoint_sha256=sha(path),steps=meta['fit_steps'],
        flow_parameters=sum(p.numel() for p in parameters),semantic_coordinates=4096,
        narrow_output_span_removed=True,base_model_loaded=False,base_checkpoint_changed=False,
        teacher_head_loaded=False,language_targets_used=False,**monitor.receipt()))


def load_flow(output,meta,device):
    require(language.load(output/'fit/status.json')['phase']=='completed','complete motor training first')
    path=output/'fit/checkpoint_latest.pt'
    require(sha(path)==language.load(output/'fit/report.json')['checkpoint_sha256'],'motor checkpoint changed')
    model=SemanticMotorFlow(meta['width'],meta['layers']).to(device)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    return model.requires_grad_(False).eval()


@torch.no_grad()
def qualify(output,meta,monitor,device):
    flow=load_flow(output,meta,device);bridge=base.bridge(meta,device)
    codes,data=training_codes(meta,device);gold=data['ink'].to(device).bool()
    summary={};records=[];outputs={};files={};boards=[];inputs={}
    for route in ['clean','mixture045']:
        torch.manual_seed(meta['assessment_seed']);torch.cuda.manual_seed_all(meta['assessment_seed'])
        selected=[];candidates=[];scores=[];choices=[];baseline=[];f1s=[];oracle=[];exact=0;route_inputs=[]
        for start in range(0,len(codes),16):
            monitor.position=dict(route=route,start=start,total=len(codes));monitor.guard()
            ids=torch.arange(start,min(start+16,len(codes)),device=device)
            c=codes[ids] if route=='clean' else .55*codes[ids]+.45*codes[(ids+1049)%len(codes)]
            c=F.normalize(c,dim=-1)*math.sqrt(4096);route_inputs.append(tensor_hash(c))
            value=generated_motor_images(flow,bridge,c,steps=meta['sampling_steps'],candidates=meta['sampling_candidates'])
            actual=value['images']<0;all_actual=value['candidate_images']<0;expected=gold[ids]
            denominators=all_actual.flatten(2).sum(-1)+expected[:,None].flatten(2).sum(-1)
            all_f1=torch.where(denominators==0,torch.ones_like(denominators).float(),
                2*(all_actual&expected[:,None]).flatten(2).sum(-1)/denominators)
            chosen=value['selected_indices'];rows=torch.arange(len(c),device=device)
            f1=all_f1[rows,chosen];same=(actual==expected).flatten(1).all(-1)
            f1s.extend(f1.cpu().tolist());oracle.extend(all_f1.max(-1).values.cpu().tolist());exact+=int(same.sum())
            selected.append(actual.cpu());candidates.append(all_actual.cpu());scores.append(value['sample_scores'].cpu());choices.append(chosen.cpu())
            baseline.append((bridge.decode_logits(c)>=0).cpu())
            for i in range(len(c)):
                records.append(dict(route=route,index=start+i,selected_f1=float(f1[i]),pixel_exact=bool(same[i]),
                    selected_candidate=int(chosen[i]),candidate_f1=all_f1[i].cpu().tolist(),
                    candidate_semantic_scores=value['sample_scores'][i].cpu().tolist()))
        outputs[route]=dict(ink=torch.cat(selected),candidate_ink=torch.cat(candidates),
            sample_scores=torch.cat(scores),selected_indices=torch.cat(choices),baseline_ink=torch.cat(baseline))
        name=route+'_all_outputs.pt';atomic_tensor(monitor.path/name,outputs[route]);files[name]=sha(monitor.path/name);inputs[route]=route_inputs
        summary[route]=dict(forms=len(codes),mean_selected_f1=sum(f1s)/len(f1s),selected_pixel_exact=exact,
            fraction_selected_f1_at_least095=sum(v>=.95 for v in f1s)/len(f1s),
            mean_oracle_candidate_f1=sum(oracle)/len(oracle),oracle_not_used_for_selection=True)
        indices=np.linspace(0,len(codes)-1,64,dtype=int).tolist()
        canvas=Image.new('L',(8*112,8*49),245);draw=ImageDraw.Draw(canvas)
        for i,index in enumerate(indices):
            x=(i%8)*112;y=(i//8)*49;draw.text((x+2,y),str(index),fill=0)
            for j,a in enumerate([gold[index].cpu(),outputs[route]['baseline_ink'][index],outputs[route]['ink'][index]]):
                p=(255*(1-a[0].numpy().astype(np.uint8))).astype(np.uint8);canvas.paste(Image.fromarray(p),(x+2+j*34,y+13))
        name=route+'_gold_baseline_flow.png';canvas.resize((canvas.width*2,canvas.height*2),Image.Resampling.NEAREST).save(monitor.path/name);boards.append(name)
        monitor.status('qualifying_generated_images',summary=summary);print(json.dumps({route:summary[route]}),flush=True)
    g=meta['primitive_gate']
    passed=(summary['clean']['mean_selected_f1']>=g['clean_mean_f1'] and
        summary['clean']['fraction_selected_f1_at_least095']>=g['clean_fraction095'] and
        summary['mixture045']['mean_selected_f1']>=g['mixture_mean_f1'] and
        summary['mixture045']['fraction_selected_f1_at_least095']>=g['mixture_fraction095'])
    save_json(monitor.path/'records.json',records)
    save_json(monitor.path/'report.json',dict(summary=summary,primitive_gate_passed=passed,
        raw_outputs_sha256=files,conditioning_inputs_sha256=inputs,boards=boards,
        actual_generated_images_scored=True,source_body_loaded=False,teacher_loaded=False,
        anchors_used_to_select_runtime_output=False,language_assessed=False,actually_viewed=False,**monitor.receipt()))


def run(output,stage):
    meta=check(output);folder=output/stage;folder.mkdir(exist_ok=False)
    monitor=base.Monitor(folder,meta);monitor.status('admitting')
    try:
        with bridge_gpu(meta) as (device,admission):
            language.numerics();monitor.guard();save_json(folder/'admission.json',admission)
            {'fit':fit,'qualify':qualify}[stage](output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(language.load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','fit','qualify'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
