#!/usr/bin/env python3
"""Existing spatial flow writer, conditioned on semantic code and neural ink plan."""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import time
import json

import numpy as np
from PIL import Image,ImageDraw
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.ink_writer import FovealInkFlow,FovealWriterConfig,flow_training_state,sample_foveal_ink
from scripts import train_semantic_glyph_language as language
from scripts import distill_semantic_glyph_readout as base
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.score_pixel_language_first_blocks import require

LANGUAGE=ROOT/'artifacts/semantic_glyph_language_v1'
FILES=('scripts/train_spatial_semantic_motor.py','references/spatial_semantic_motor_protocol.md',
       'ilm/visual_lm/ink_writer.py','ilm/visual_lm/ink_jepa.py','scripts/distill_semantic_glyph_readout.py')


def identity():
    result={n:sha(ROOT/n) for n in FILES}
    for n in FILES:require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit spatial motor sources first')
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation');require(not output.exists(),'preserve previous attempts')
    original=language.check(LANGUAGE);path=LANGUAGE/'fit/checkpoint_latest.pt'
    require(sha(path)==language.load(LANGUAGE/'fit/report.json')['checkpoint_sha256'],'baseline changed')
    output.mkdir(parents=True)
    save_json(output/'preparation.json',dict(original,spatial_code_sha256=identity(),parent_run=language.relative(LANGUAGE),
        parent_preparation_sha256=sha(LANGUAGE/'preparation.json'),baseline=language.relative(path),baseline_sha256=sha(path),
        seed=202611109,assessment_seed=202611110,fit_steps=8000,batch_size=128,learning_rate=.0002,
        base_channels=32,context_dim=256,sampling_steps=16,sampling_candidates=4,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,maximum_vram_gib=8.,maximum_stage_seconds=1800.,
        maximum_pressure_wait_seconds=300.,base_weights_unchanged=True,no_language_targets=True,
        primitive_gate=dict(clean_mean_f1=.99,clean_fraction095=.95,mixture_mean_f1=.95,mixture_fraction095=.8)))


def check(output):
    m=language.load(output/'preparation.json');language.check(ROOT/m['parent_run'])
    require(m['spatial_code_sha256']==identity(),'spatial sources changed')
    require(sha(ROOT/m['parent_run']/'preparation.json')==m['parent_preparation_sha256'] and
            sha(ROOT/m['baseline'])==m['baseline_sha256'],'original model changed')
    return m


def writer(meta,device):
    return FovealInkFlow(FovealWriterConfig(condition_dim=4096,base_channels=meta['base_channels'],
        context_dim=meta['context_dim'],condition_dropout=0.)).to(device)


def codes_and_images(meta,device):
    d=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    codes=F.normalize(((d['semantic']-d['mean'])/d['scale']).to(device),dim=-1)*math.sqrt(4096)
    return codes,2*d['ink'].float().to(device)-1


@torch.no_grad()
def samples(model,bridge,c,meta):
    c=F.normalize(c.float(),dim=-1)*math.sqrt(4096);n=meta['sampling_candidates']
    plan=2*bridge.decode_logits(c).sigmoid()-1
    repeated=c[:,None].expand(-1,n,-1).flatten(0,1)
    plans=plan[:,None].expand(-1,n,-1,-1,-1).flatten(0,1)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        image=sample_foveal_ink(model,repeated,plans,steps=meta['sampling_steps'],guidance_scale=1.)
    ink=(image>=0).reshape(len(c),n,1,32,32)
    actual=1-2*ink.float();encoded=bridge.encode(actual.flatten(0,1)).reshape(len(c),n,4096)
    scores=F.cosine_similarity(encoded,c[:,None],dim=-1);selected=scores.argmax(-1)
    return dict(ink=ink[torch.arange(len(c),device=c.device),selected],candidate_ink=ink,
                sample_scores=scores,selected_indices=selected)


def fit(output,meta,monitor,device):
    torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    codes,images=codes_and_images(meta,device);bridge=base.bridge(meta,device)
    model=writer(meta,device);ema=copy.deepcopy(model).requires_grad_(False).eval()
    params=list(model.parameters());averaged=list(ema.parameters())
    opt=torch.optim.AdamW(params,lr=meta['learning_rate'],betas=(.9,.99),weight_decay=.01)
    for step in range(1,meta['fit_steps']+1):
        if step%10==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        ids=torch.randint(len(codes),(meta['batch_size'],),device=device)
        other=(ids+torch.randint(1,len(codes),(len(ids),),device=device))%len(codes)
        target=images[ids]+.001*torch.randn_like(images[ids]);route=torch.rand(len(ids),1,device=device)
        noise=torch.randn_like(codes[ids]);a=.25+.2*torch.rand(len(ids),1,device=device)
        mixed=(1-a)*codes[ids]+a*codes[other]+.05*noise
        blurred=codes[ids]+.8*torch.rand(len(ids),1,device=device)*noise
        c=F.normalize(torch.where(route<.6,mixed,torch.where(route<.8,blurred,codes[ids])),dim=-1)*math.sqrt(4096)
        with torch.no_grad():plan=2*bridge.decode_logits(c).sigmoid()-1
        state,velocity,t,_=flow_training_state(target)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(state,t,c,plan)
        loss=F.mse_loss(prediction.float(),velocity)
        require(bool(torch.isfinite(loss)),'nonfinite spatial velocity loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        decay=min(.999,(1+step)/(10+step))
        with torch.no_grad():torch._foreach_lerp_(averaged,params,1-decay)
        fraction=step/meta['fit_steps']
        for group in opt.param_groups:group['lr']=meta['learning_rate']*min(1.,step/200)*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
        if step%100==0:
            row=dict(step=step,velocity_mse=float(loss),seconds=time.perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('fitting_spatial_motor');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt';atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in ema.state_dict().items()},
        step=meta['fit_steps'],base_checkpoint_sha256=meta['baseline_sha256'],ema=True,language_generation_qualified=False,
        preparation_sha256=sha(output/'preparation.json')))
    save_json(monitor.path/'report.json',dict(checkpoint_sha256=sha(path),steps=meta['fit_steps'],
        trained_parameters=sum(p.numel() for p in params),base_checkpoint_unchanged=sha(ROOT/meta['baseline'])==meta['baseline_sha256'],
        language_targets_used=False,teacher_loaded=False,source_body_loaded=False,**monitor.receipt()))


@torch.no_grad()
def qualify(output,meta,monitor,device):
    require(language.load(output/'fit/status.json')['phase']=='completed','complete spatial fit first')
    path=output/'fit/checkpoint_latest.pt';require(sha(path)==language.load(output/'fit/report.json')['checkpoint_sha256'],'motor changed')
    model=writer(meta,device);model.load_state_dict(torch.load(path,weights_only=True,map_location='cpu')['model']);model.eval()
    bridge=base.bridge(meta,device);codes,images=codes_and_images(meta,device);gold=images>0
    summary={};records=[];files={};boards=[]
    for route in ['clean','mixture045']:
        torch.manual_seed(meta['assessment_seed']);torch.cuda.manual_seed_all(meta['assessment_seed'])
        parts={k:[] for k in ['ink','candidate_ink','sample_scores','selected_indices','baseline_ink']};f1s=[];oracle=[];exact=0
        for start in range(0,len(codes),16):
            monitor.position=dict(route=route,start=start,total=len(codes));monitor.guard()
            ids=torch.arange(start,min(start+16,len(codes)),device=device)
            c=codes[ids] if route=='clean' else .55*codes[ids]+.45*codes[(ids+1049)%len(codes)]
            c=F.normalize(c,dim=-1)*math.sqrt(4096);value=samples(model,bridge,c,meta);all_actual=value['candidate_ink']
            expected=gold[ids];denom=all_actual.flatten(2).sum(-1)+expected[:,None].flatten(2).sum(-1)
            all_f1=torch.where(denom==0,torch.ones_like(denom).float(),2*(all_actual&expected[:,None]).flatten(2).sum(-1)/denom)
            choice=value['selected_indices'];f1=all_f1[torch.arange(len(c),device=device),choice]
            same=(value['ink']==expected).flatten(1).all(-1);exact+=int(same.sum());f1s.extend(f1.cpu().tolist());oracle.extend(all_f1.max(-1).values.cpu().tolist())
            value['baseline_ink']=bridge.decode_logits(c)>=0
            for k in parts:parts[k].append(value[k].cpu())
            for i in range(len(c)):records.append(dict(route=route,index=start+i,selected_f1=float(f1[i]),pixel_exact=bool(same[i]),
                selected_candidate=int(choice[i]),candidate_f1=all_f1[i].cpu().tolist(),sample_scores=value['sample_scores'][i].cpu().tolist()))
        raw={k:torch.cat(v) for k,v in parts.items()};name=route+'_all_outputs.pt';atomic_tensor(monitor.path/name,raw);files[name]=sha(monitor.path/name)
        summary[route]=dict(forms=len(codes),mean_selected_f1=sum(f1s)/len(f1s),selected_pixel_exact=exact,
            fraction_selected_f1_at_least095=sum(v>=.95 for v in f1s)/len(f1s),mean_oracle_f1=sum(oracle)/len(oracle),oracle_used_for_selection=False)
        indices=np.linspace(0,len(codes)-1,64,dtype=int).tolist();canvas=Image.new('L',(8*112,8*49),245);draw=ImageDraw.Draw(canvas)
        for i,index in enumerate(indices):
            x=(i%8)*112;y=(i//8)*49;draw.text((x+2,y),str(index),fill=0)
            for j,a in enumerate([gold[index].cpu(),raw['baseline_ink'][index],raw['ink'][index]]):
                canvas.paste(Image.fromarray((255*(1-a[0].numpy().astype(np.uint8))).astype(np.uint8)),(x+2+j*34,y+13))
        name=route+'_gold_baseline_spatial.png';canvas.resize((canvas.width*2,canvas.height*2),Image.Resampling.NEAREST).save(monitor.path/name);boards.append(name)
        print(json.dumps({route:summary[route]}),flush=True);monitor.status('qualifying_spatial_motor',summary=summary)
    g=meta['primitive_gate'];passed=(summary['clean']['mean_selected_f1']>=g['clean_mean_f1'] and
        summary['clean']['fraction_selected_f1_at_least095']>=g['clean_fraction095'] and
        summary['mixture045']['mean_selected_f1']>=g['mixture_mean_f1'] and summary['mixture045']['fraction_selected_f1_at_least095']>=g['mixture_fraction095'])
    save_json(monitor.path/'records.json',records)
    save_json(monitor.path/'report.json',dict(summary=summary,primitive_gate_passed=passed,raw_outputs_sha256=files,boards=boards,
        language_assessed=False,teacher_loaded=False,body_loaded=False,lookup_used_for_selection=False,actually_viewed=False,**monitor.receipt()))


def run(output,stage):
    meta=check(output);folder=output/stage;folder.mkdir(exist_ok=False);monitor=base.Monitor(folder,meta);monitor.status('admitting')
    try:
        with bridge_gpu(meta) as (device,admission):
            language.numerics();monitor.guard();save_json(folder/'admission.json',admission);{'fit':fit,'qualify':qualify}[stage](output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(language.load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','fit','qualify'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
