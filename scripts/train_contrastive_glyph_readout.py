#!/usr/bin/env python3
"""Loss-only readout intervention with an offline contrastive anchor objective."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts import distill_semantic_glyph_readout as parent
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu
from scripts.train_semantic_glyph_language import load,relative,numerics
from scripts.score_pixel_language_first_blocks import require

PARENT=ROOT/'artifacts/semantic_glyph_readout_distillation_v1'
FILES=('scripts/train_contrastive_glyph_readout.py','references/contrastive_glyph_readout_protocol.md')


def identity():
    result={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit contrastive sources first')
    return result


def prepare(output):
    import os
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation')
    meta=parent.check(PARENT);parent.calibrated(PARENT)
    require(not output.exists(),'preserve every prior candidate')
    output.mkdir(parents=True)
    for name in ['genuine.pt','mapping.json','calibrate']:
        (output/name).symlink_to(PARENT/name,target_is_directory=name=='calibrate')
    save_json(output/'preparation.json',dict(meta,variant='contrastive_loss_only',
        parent_run=relative(PARENT),parent_preparation_sha256=sha(PARENT/'preparation.json'),
        parent_calibration_report_sha256=sha(PARENT/'calibrate/report.json'),
        variant_code_sha256=identity(),contrastive_temperature=.07,contrastive_weight=1.,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initialization='same preserved language baseline, not the failed distillation weights',
        reused_calibration_without_new_teacher_calls=True))


def check(output):
    meta=load(output/'preparation.json');original=ROOT/meta['parent_run']
    base=parent.check(original);parent.calibrated(original)
    require(meta['variant_code_sha256']==identity(),'contrastive sources changed')
    require(sha(original/'preparation.json')==meta['parent_preparation_sha256'] and
            sha(original/'calibrate/report.json')==meta['parent_calibration_report_sha256'],'parent changed')
    require(all(meta[k]==v for k,v in base.items() if k!='source_commit'),'baseline recipe changed')
    for name in ['genuine.pt','mapping.json','calibrate']:
        require((output/name).resolve()==(original/name).resolve(),'reuse link changed')
    return meta


def fit(output,meta,monitor,device):
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    model=parent.bridge(meta,device);model.readout.requires_grad_(True)
    data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    images=(1-2*data['ink'].float()).to(device)
    targets=((data['semantic']-data['mean'])/data['scale']).to(device)
    anchors=F.normalize(targets,dim=-1)
    synth=torch.load(output/'calibrate/calibration.pt',weights_only=True,map_location='cpu')
    genuine=torch.load(output/'genuine.pt',weights_only=True,map_location='cpu')
    labels=torch.load(output/'calibrate/genuine_labels.pt',weights_only=True,map_location='cpu')['labels']
    a=synth['split']==0;b=(genuine['split']==0)&(labels>=0)
    pools=[(synth['states'][a].to(device),synth['labels'][a].to(device)),
           (genuine['states'][b].to(device),labels[b].to(device))]
    params=list(model.readout.parameters());opt=torch.optim.AdamW(params,lr=meta['learning_rate'],weight_decay=.01)
    for step in range(1,meta['fit_steps']+1):
        if step%20==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        batch=[];labels=[]
        for x,y in pools:
            ix=torch.randint(len(x),(meta['batch_size']//2,),device=device);batch.append(x[ix]);labels.append(y[ix])
        h=torch.cat(batch);label=torch.cat(labels);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.readout(h).float();code=F.mse_loss(z,targets[label]);cos=1-F.cosine_similarity(z,targets[label]).mean()
            pix=reconstruction_loss(model.decode_logits(z),images[label])
            scores=(F.normalize(z,dim=-1)@anchors.T).float()/meta['contrastive_temperature']
            discrimination=F.cross_entropy(scores,label)
            loss=code+.2*cos+pix+meta['contrastive_weight']*discrimination
        require(bool(torch.isfinite(loss)),'nonfinite contrastive loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        fraction=step/meta['fit_steps']
        for group in opt.param_groups:group['lr']=meta['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
        if step%100==0:
            row=dict(step=step,loss=float(loss),code=float(code),pixel=float(pix),cosine=float(cos),
                discrimination=float(discrimination),batch_anchor_top1=float((scores.argmax(-1)==label).float().mean()),
                seconds=__import__('time').perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('contrastive_readout');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt'
    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        language_training_complete=True,step=meta['fit_steps'],primitive_only=False,
        contrastive_readout_only=True,usable_language_demonstrated=False,
        preparation_sha256=sha(output/'preparation.json')))
    original=parent.bridge(meta,'cpu')
    require(all(torch.equal(original.state_dict()[k],v.cpu()) for k,v in model.state_dict().items() if not k.startswith('readout.')),
            'non-readout weights changed')
    save_json(monitor.path/'report.json',dict(steps=meta['fit_steps'],checkpoint_sha256=sha(path),
        trained_parameters=sum(p.numel() for p in params),synthetic_states=len(pools[0][0]),genuine_states=len(pools[1][0]),
        non_readout_weights_exact=True,teacher_head_loaded=False,body_loaded=False,
        contrastive_anchors_in_deployed_checkpoint=False,**monitor.receipt()))


def run(output,stage):
    meta=check(output);folder=output/stage;folder.mkdir(exist_ok=False)
    monitor=parent.Monitor(folder,meta);monitor.status('admitting')
    try:
        with bridge_gpu(meta) as (device,admission):
            numerics();monitor.guard();save_json(folder/'admission.json',admission)
            if stage=='fit':fit(output,meta,monitor,device)
            else:parent.qualify(output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','fit','qualify'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
