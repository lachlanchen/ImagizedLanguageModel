#!/usr/bin/env python3
"""Train only the neural glyph decoder against continuous-code ambiguity."""
import argparse
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
from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
from scripts import distill_semantic_glyph_readout as base
from scripts import train_semantic_glyph_language as language
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.score_pixel_language_first_blocks import require,diagnostic_gpu

LANGUAGE=ROOT/'artifacts/semantic_glyph_language_v1'
FILES=('scripts/train_robust_glyph_decoder.py','references/robust_glyph_decoder_protocol.md')


def identity():
    result={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit decoder sources first')
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation')
    require(not output.exists(),'preserve prior decoder attempts')
    original=language.check(LANGUAGE)
    checkpoint=LANGUAGE/'fit/checkpoint_latest.pt'
    require(sha(checkpoint)==language.load(LANGUAGE/'fit/report.json')['checkpoint_sha256'],'baseline changed')
    output.mkdir(parents=True)
    save_json(output/'preparation.json',dict(original,
        decoder_code_sha256=identity(),parent_run=language.relative(LANGUAGE),
        parent_preparation_sha256=sha(LANGUAGE/'preparation.json'),baseline=language.relative(checkpoint),
        baseline_sha256=sha(checkpoint),seed=202611105,fit_steps=6000,batch_size=128,
        learning_rate=.0002,minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,
        maximum_vram_gib=8.,maximum_stage_seconds=1800.,maximum_pressure_wait_seconds=300.,
        decoder_only=True,language_targets_used=False,development_training_allowed=False,
        mixture_maximum=.45,gaussian_maximum=1.,
        primitive_gate=dict(clean_mean_f1=.999,gaussian06_mean_f1=.99,
            gaussian06_fraction095=.95,mixture045_mean_f1=.95,mixture045_fraction095=.8)))


def check(output):
    meta=language.load(output/'preparation.json')
    language.check(ROOT/meta['parent_run'])
    require(meta['decoder_code_sha256']==identity(),'decoder sources changed')
    require(sha(ROOT/meta['parent_run']/'preparation.json')==meta['parent_preparation_sha256'] and
            sha(ROOT/meta['baseline'])==meta['baseline_sha256'],'baseline data changed')
    return meta


def tensors(meta,device):
    data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    codes=((data['semantic']-data['mean'])/data['scale']).to(device)
    unit=F.normalize(codes,dim=-1)*math.sqrt(codes.shape[1])
    return codes,unit,(1-2*data['ink'].float()).to(device)


def fit(output,meta,monitor,device):
    torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    model=base.bridge(meta,device)
    model.decoder_input.requires_grad_(True);model.decoder.requires_grad_(True)
    codes,unit,images=tensors(meta,device);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=meta['learning_rate'],weight_decay=.01)
    for step in range(1,meta['fit_steps']+1):
        if step%20==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        ids=torch.randint(len(codes),(meta['batch_size'],),device=device)
        other=(ids+torch.randint(1,len(codes),(len(ids),),device=device))%len(codes)
        route=torch.rand(len(ids),device=device)
        strength=meta['mixture_maximum']*torch.rand(len(ids),1,device=device)
        gaussian=meta['gaussian_maximum']*torch.rand(len(ids),1,device=device)
        noise=torch.randn_like(codes[ids])
        noisy=unit[ids]+gaussian*noise
        mixed=(1-strength)*unit[ids]+strength*unit[other]+.05*noise
        z=torch.where((route<.6)[:,None],noisy,torch.where((route<.9)[:,None],mixed,codes[ids]))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=reconstruction_loss(model.decode_logits(z),images[ids])
        require(bool(torch.isfinite(loss)),'nonfinite robust-decoder loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        fraction=step/meta['fit_steps']
        for group in opt.param_groups:group['lr']=meta['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
        if step%100==0:
            row=dict(step=step,pixel_loss=float(loss),seconds=time.perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('training_decoder_only');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt'
    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        language_training_complete=True,step=meta['fit_steps'],primitive_only=False,
        decoder_only=True,usable_language_demonstrated=False,preparation_sha256=sha(output/'preparation.json')))
    original=base.bridge(meta,'cpu')
    require(all(torch.equal(original.state_dict()[k],v.cpu()) for k,v in model.state_dict().items()
                if not k.startswith(('decoder.','decoder_input.'))),'reader/encoder/semantic readout/stop changed')
    save_json(monitor.path/'report.json',dict(steps=meta['fit_steps'],checkpoint_sha256=sha(path),
        trained_parameters=sum(p.numel() for p in params),non_decoder_weights_exact=True,
        no_language_examples_used=True,teacher_head_loaded=False,body_loaded=False,**monitor.receipt()))


@torch.no_grad()
def qualify(output,meta,monitor,device):
    require(language.load(output/'fit/status.json')['phase']=='completed','finish decoder fit first')
    path=output/'fit/checkpoint_latest.pt'
    require(sha(path)==language.load(output/'fit/report.json')['checkpoint_sha256'],'candidate changed')
    codes,unit,images=tensors(meta,device);gold=images<0;summary={};bindings={};files={}
    emissions={};records=[]
    for version,checkpoint in [('baseline',ROOT/meta['baseline']),('robust',path)]:
        model=base.bridge(meta,device,checkpoint)
        torch.manual_seed(202611106);torch.cuda.manual_seed_all(202611106)
        for route in ['clean','gaussian06','mixture045']:
            emitted=[];f1s=[];exact=0;inputs=[]
            for start in range(0,len(codes),64):
                monitor.position=dict(model=version,route=route,start=start,total=len(codes));monitor.guard()
                if route=='clean':z=codes[start:start+64]
                elif route=='gaussian06':z=unit[start:start+64]+.6*torch.randn_like(unit[start:start+64])
                else:
                    ids=torch.arange(start,min(start+64,len(codes)),device=device)
                    # Fixed, disjoint from the training RNG. Unit-length forms
                    # make the requested 0.55 target component dominant.
                    z=.55*unit[ids]+.45*unit[(ids+1049)%len(codes)]
                from ilm.visual_lm.semantic_glyph_loop import tensor_hash
                inputs.append(tensor_hash(z));actual=model.decode_logits(z)>=0;expected=gold[start:start+len(z)]
                denom=actual.flatten(1).sum(-1)+expected.flatten(1).sum(-1)
                f1=torch.where(denom==0,torch.ones_like(denom).float(),2*(actual&expected).flatten(1).sum(-1)/denom)
                same=(actual==expected).flatten(1).all(-1);exact+=int(same.sum())
                f1s.extend(f1.cpu().tolist());emitted.append(actual.cpu())
                for i in range(len(z)):records.append(dict(model=version,route=route,index=start+i,f1=float(f1[i]),pixel_exact=bool(same[i])))
            if version=='baseline':bindings[route]=inputs
            else:require(bindings[route]==inputs,'paired robustness inputs differ')
            emissions[version+'/'+route]=torch.cat(emitted)
            name=version+'_'+route+'.pt';atomic_tensor(monitor.path/name,dict(ink=emissions[version+'/'+route]))
            files[name]=sha(monitor.path/name)
            summary[version+'/'+route]=dict(forms=len(codes),mean_f1=sum(f1s)/len(f1s),pixel_exact=exact,
                fraction_f1_at_least_095=sum(v>=.95 for v in f1s)/len(f1s))
        del model
    gate=meta['primitive_gate']
    passed=(summary['robust/clean']['mean_f1']>=gate['clean_mean_f1'] and
            summary['robust/gaussian06']['mean_f1']>=gate['gaussian06_mean_f1'] and
            summary['robust/gaussian06']['fraction_f1_at_least_095']>=gate['gaussian06_fraction095'] and
            summary['robust/mixture045']['mean_f1']>=gate['mixture045_mean_f1'] and
            summary['robust/mixture045']['fraction_f1_at_least_095']>=gate['mixture045_fraction095'])
    boards=[]
    for route in ['clean','gaussian06','mixture045']:
        indices=np.linspace(0,len(codes)-1,64,dtype=int).tolist()
        canvas=Image.new('L',(8*112,8*49),245);draw=ImageDraw.Draw(canvas)
        for i,index in enumerate(indices):
            x=(i%8)*112;y=(i//8)*49;draw.text((x+2,y),str(index),fill=0)
            for j,a in enumerate([gold[index].cpu(),emissions['baseline/'+route][index],emissions['robust/'+route][index]]):
                p=(255*(1-a[0].numpy().astype(np.uint8))).astype(np.uint8)
                canvas.paste(Image.fromarray(p),(x+2+j*34,y+13))
        name=route+'_gold_baseline_robust.png';canvas.resize((canvas.width*2,canvas.height*2),Image.Resampling.NEAREST).save(monitor.path/name)
        boards.append(dict(file=name,sha256=sha(monitor.path/name),indices=indices))
    save_json(monitor.path/'records.json',records)
    save_json(monitor.path/'report.json',dict(summary=summary,primitive_gate_passed=passed,
        all_raw_pixels_retained=files,boards=boards,paired_inputs_sha256=bindings,
        records_sha256=sha(monitor.path/'records.json'),language_assessed=False,
        unseen_glyph_identity_test=False,independent_noise_test=True,actually_viewed=False,**monitor.receipt()))
    print(json.dumps(dict(summary=summary,primitive_gate_passed=passed)),flush=True)


def run(output,stage):
    meta=check(output);folder=output/stage;folder.mkdir(exist_ok=False)
    if stage=='assessment':
        require(language.load(output/'qualify/report.json')['primitive_gate_passed'],'decoder gate failed; do not expand language inference')
        monitor=language.Monitor(folder,meta['assessment_maximum_seconds'])
    else:monitor=base.Monitor(folder,meta)
    monitor.status('admitting')
    try:
        if stage=='assessment':language.assess(output,meta,monitor)
        else:
            with bridge_gpu(meta) as (device,admission):
                language.numerics();monitor.guard();save_json(folder/'admission.json',admission)
                {'fit':fit,'qualify':qualify}[stage](output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(language.load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','fit','qualify','assessment'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
