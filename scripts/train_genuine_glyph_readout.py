#!/usr/bin/env python3
"""Broaden the preserved readout with genuine image-copy contexts, not forced states."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
from ilm.visual_lm.semantic_image_memory import native_budget_image
from scripts import train_semantic_glyph_language as language
from scripts import distill_semantic_glyph_readout as base
from scripts.train_semantic_glyph_bridge import atomic_tensor,bridge_gpu,FONT
from scripts.score_pixel_language_first_blocks import require,diagnostic_gpu

LANGUAGE=ROOT/'artifacts/semantic_glyph_language_v1'
FILES=('scripts/train_genuine_glyph_readout.py','references/genuine_glyph_readout_protocol.md',
       'scripts/distill_semantic_glyph_readout.py')
TEMPLATES=[['请抄写方框内的一个字符。','只输出这个字符，不要解释。'],
           ['Copy the single printed character.','Return only that character.'],
           ['下框中的字是什么？只写该字。']]


def identity():
    result={n:sha(ROOT/n) for n in FILES}
    for n in FILES:require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit genuine-readout sources first')
    return result


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only preparation');require(not output.exists(),'preserve all previous runs')
    original=language.check(LANGUAGE);primitive=ROOT/original['primitive']
    checkpoint=LANGUAGE/'fit/checkpoint_latest.pt'
    require(sha(checkpoint)==language.load(LANGUAGE/'fit/report.json')['checkpoint_sha256'],'baseline changed')
    data=torch.load(primitive/'training.pt',weights_only=True,map_location='cpu')
    require(sha(FONT)==language.load(primitive/'preparation.json')['font_sha256'],'instruction font changed')
    font=ImageFont.truetype(str(FONT),26,index=2);backgrounds=[]
    for lines in TEMPLATES:
        page=Image.new('L',(512,256),255);draw=ImageDraw.Draw(page)
        for i,line in enumerate(lines):
            require(draw.textlength(line,font=font)<=464,'instruction exceeds canvas')
            draw.text((24,12+i*36),line,font=font,fill=0)
        draw.rectangle((199,100,311,212),outline=0,width=2);backgrounds.append(page)
    valid=[i for i,a in enumerate(data['ink']) if bool(a.any())]
    empty=[i for i,a in enumerate(data['ink']) if not bool(a.any())]
    held=[valid[i] for i in np.linspace(0,len(valid)-1,256,dtype=int)]
    output.mkdir(parents=True);(output/'prompts').mkdir()
    records=[]
    for template,labels in [(0,valid),(1,valid),(2,held)]:
        for label in labels:
            glyph=Image.fromarray((255*(1-data['ink'][label,0].numpy())).astype(np.uint8))
            page=backgrounds[template].copy();page.paste(glyph.resize((96,96),Image.Resampling.NEAREST),(208,108))
            name=f'prompts/template{template}_form{label:04}.png';page.save(output/name)
            records.append(dict(index=len(records),split='train' if template<2 else 'held_template',
                label=label,template=template,png=name,png_sha256=sha(output/name)))
    samples=[r for r in records if r['template']==0]
    samples=[samples[i] for i in np.linspace(0,len(samples)-1,16,dtype=int)]
    canvas=Image.new('L',(1024,4*150),245);draw=ImageDraw.Draw(canvas)
    for i,row in enumerate(samples):
        x=(i%4)*256;y=(i//4)*150;draw.text((x+4,y+2),f"form{row['label']} (canonical bitmap, not retyped)",fill=0)
        canvas.paste(Image.open(output/row['png']).resize((256,128)),(x,y+18))
    canvas.save(output/'input_preview.png')
    preview_rows=[]
    for template in range(3):
        row=next(r for r in records if r['template']==template and r['label']==held[len(held)//2])
        name=f'input_template{template}.png';Image.open(output/row['png']).save(output/name);preview_rows.append(name)
    save_json(output/'preparation.json',dict(original,genuine_code_sha256=identity(),parent_run=language.relative(LANGUAGE),
        parent_preparation_sha256=sha(LANGUAGE/'preparation.json'),baseline=language.relative(checkpoint),baseline_sha256=sha(checkpoint),
        original_cache_report_sha256=sha(LANGUAGE/'cache/report.json'),font_sha256=sha(FONT),records=records,
        unique_observable_forms=len(valid),empty_forms_excluded=empty,held_template_cases=len(held),
        no_development_training=True,templates=TEMPLATES,input_previews=['input_preview.png',*preview_rows],
        cache_batch=4,maximum_batch_single_state_error=.002,cache_maximum_seconds=14400.,
        fit_steps=18000,batch_size=128,learning_rate=.0002,seed=202611111,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,maximum_vram_gib=8.,maximum_stage_seconds=1800.,
        maximum_pressure_wait_seconds=300.,original_stop_encoder_decoder_frozen=True,
        gate=dict(training_copy_mean_f1=.99,held_copy_mean_f1=.95,held_fraction095=.8,
                  old_cached_target_mean_f1_maximum_loss=.01,held_first_nonstop_fraction=.95)))
    print(json.dumps(dict(records=len(records),observable_forms=len(valid),excluded_empty_forms=len(empty))),flush=True)


def check(output):
    m=language.load(output/'preparation.json');language.check(ROOT/m['parent_run'])
    require(identity()==m['genuine_code_sha256'],'genuine readout sources changed')
    require(sha(ROOT/m['baseline'])==m['baseline_sha256'] and
        sha(ROOT/m['parent_run']/'preparation.json')==m['parent_preparation_sha256'] and
        sha(ROOT/m['parent_run']/'cache/report.json')==m['original_cache_report_sha256'],'original source changed')
    for r in m['records']:require(sha(output/r['png'])==r['png_sha256'],'prepared prompt changed')
    return m


def collate(arguments):
    return dict(inputs_embeds=torch.cat([a['inputs_embeds'] for a in arguments],0),
        position_ids=torch.cat([a['position_ids'] for a in arguments],1),
        visual_pos_masks=torch.cat([a['visual_pos_masks'] for a in arguments],0),
        deepstack_visual_embeds=[torch.cat([a['deepstack_visual_embeds'][i] for a in arguments],0)
            for i in range(len(arguments[0]['deepstack_visual_embeds']))])


@torch.no_grad()
def cache(output,meta,monitor):
    review=language.load(output/'input_review.json')
    require(review['preparation_sha256']==sha(output/'preparation.json') and review['all_four_previews_actually_viewed'],
            'inspect actual prepared pages first')
    with diagnostic_gpu() as (device,admission):
        monitor.admission=admission;language.numerics();monitor.guard()
        interface,provenance=language.reader(device);files=[];maximum_error=None;geometry=None
        for start in range(0,len(meta['records']),meta['cache_batch']):
            monitor.position=dict(start=start,total=len(meta['records']));monitor.guard()
            rows=meta['records'][start:start+meta['cache_batch']];parts=[];geometries=[]
            for r in rows:
                a,g=interface.assemble(native_budget_image(language.image(output/r['png'])));parts.append(a);geometries.append(g)
            require(all(g==geometries[0] for g in geometries),'copy layout geometries differ')
            if geometry is None:geometry=geometries[0]
            require(geometries[0]==geometry,'prepared observer geometry changed')
            if start==0:
                independent=torch.cat([interface.actor.context(**a,use_cache=False).last_hidden_state[:,-1].cpu() for a in parts])
            arguments=collate(parts)
            response=interface.actor.context(**arguments,use_cache=False);states=response.last_hidden_state[:,-1].float().cpu().clone()
            del response,arguments,parts
            if start==0:
                maximum_error=float((states-independent).abs().max())
                require(maximum_error<meta['maximum_batch_single_state_error'],'batched native state differs from single-image execution')
            name=f'batch_{start:05}.pt';value=dict(states=states,labels=torch.tensor([r['label'] for r in rows]),
                indices=[r['index'] for r in rows],split=[r['split'] for r in rows])
            atomic_tensor(monitor.path/name,value)
            files.append(dict(file=name,sha256=sha(monitor.path/name),indices=value['indices']))
            monitor.status('caching_genuine_image_states')
            if start%64==0:print(json.dumps(dict(completed=start+len(rows),total=len(meta['records']))),flush=True)
        save_json(monitor.path/'report.json',dict(files=files,provenance=provenance,geometry=geometry,
            batch_single_maximum_error=maximum_error,source_head_loaded=False,teacher_service=False,
            target_labels_not_given_to_reader=True,**monitor.receipt()))


def pools(output,meta):
    require(language.load(output/'cache/status.json')['phase']=='completed','complete actual-image cache first')
    report=language.load(output/'cache/report.json');states=[];labels=[];split=[]
    for row in report['files']:
        p=output/'cache'/row['file'];require(sha(p)==row['sha256'],'copy hidden cache changed')
        v=torch.load(p,weights_only=True,map_location='cpu');states.append(v['states']);labels.append(v['labels']);split.extend(v['split'])
    h=torch.cat(states);y=torch.cat(labels);mask=torch.tensor([s=='train' for s in split])
    original_h=[];original_y=[]
    for row in language.load(ROOT/meta['parent_run']/'cache/report.json')['files']:
        p=ROOT/meta['parent_run']/'cache'/row['file'];require(sha(p)==row['sha256'],'original causal cache changed')
        v=torch.load(p,weights_only=True,map_location='cpu');keep=v['labels']>=0
        original_h.append(v['states'][keep]);original_y.append(v['labels'][keep])
    return {'copy_train':(h[mask],y[mask]),'copy_held':(h[~mask],y[~mask]),
            'original_histories':(torch.cat(original_h),torch.cat(original_y))}


def fit(output,meta,monitor,device):
    torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    model=base.bridge(meta,device);model.readout.requires_grad_(True)
    data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    targets=((data['semantic']-data['mean'])/data['scale']).to(device);images=(1-2*data['ink'].float()).to(device)
    data_pools=pools(output,meta)
    train=[(h.to(device),y.to(device)) for key,(h,y) in data_pools.items() if key!='copy_held']
    params=list(model.readout.parameters());opt=torch.optim.AdamW(params,lr=meta['learning_rate'],weight_decay=.01)
    for step in range(1,meta['fit_steps']+1):
        if step%20==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        hs=[];ys=[]
        for h,y in train:
            ids=torch.randint(len(h),(meta['batch_size']//2,),device=device);hs.append(h[ids]);ys.append(y[ids])
        h=torch.cat(hs);label=torch.cat(ys);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.readout(h).float();code=F.mse_loss(z,targets[label]);cosine=1-F.cosine_similarity(z,targets[label]).mean()
            pixel=reconstruction_loss(model.decode_logits(z),images[label]);loss=code+.2*cosine+pixel
        require(bool(torch.isfinite(loss)),'nonfinite genuine readout loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        for group in opt.param_groups:group['lr']=meta['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*step/meta['fit_steps'])))
        if step%100==0:
            row=dict(step=step,loss=float(loss),code=float(code),pixel=float(pixel),seconds=time.perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('fitting_genuine_readout');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt'
    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},step=meta['fit_steps'],
        language_training_complete=True,primitive_only=False,usable_language_demonstrated=False,
        preparation_sha256=sha(output/'preparation.json')))
    original=base.bridge(meta,'cpu')
    require(all(torch.equal(original.state_dict()[k],v.cpu()) for k,v in model.state_dict().items() if not k.startswith('readout.')),
            'non-readout weights changed')
    save_json(monitor.path/'report.json',dict(checkpoint_sha256=sha(path),steps=meta['fit_steps'],
        trained_parameters=sum(p.numel() for p in params),copy_training_states=len(data_pools['copy_train'][0]),
        held_template_states_excluded=len(data_pools['copy_held'][0]),old_histories=len(data_pools['original_histories'][0]),
        non_readout_weights_exact=True,**monitor.receipt()))


@torch.no_grad()
def qualify(output,meta,monitor,device):
    require(language.load(output/'fit/status.json')['phase']=='completed','finish readout fit first')
    path=output/'fit/checkpoint_latest.pt';require(sha(path)==language.load(output/'fit/report.json')['checkpoint_sha256'],'candidate changed')
    data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu');gold=data['ink'].to(device).bool()
    groups=pools(output,meta);summary={};records=[];emissions={};files={}
    for version,checkpoint in [('baseline',ROOT/meta['baseline']),('genuine',path)]:
        model=base.bridge(meta,device,checkpoint)
        for route,(states,labels) in groups.items():
            bits=[];f1s=[];exact=0;nonstop=0
            for start in range(0,len(states),64):
                monitor.position=dict(model=version,route=route,start=start,total=len(states));monitor.guard()
                logits,_,stop=model.predict(states[start:start+64].to(device));actual=logits>=0
                target=gold[labels[start:start+64].to(device)];d=actual.flatten(1).sum(-1)+target.flatten(1).sum(-1)
                f1=torch.where(d==0,torch.ones_like(d).float(),2*(actual&target).flatten(1).sum(-1)/d)
                same=(actual==target).flatten(1).all(-1);prob=stop.sigmoid()
                f1s.extend(f1.cpu().tolist());exact+=int(same.sum());nonstop+=int((prob<.5).sum());bits.append(actual.cpu())
                for i in range(len(f1)):records.append(dict(model=version,route=route,index=start+i,label=int(labels[start+i]),
                    f1=float(f1[i]),pixel_exact=bool(same[i]),initial_stop_probability=float(prob[i])))
            key=version+'/'+route;emissions[key]=torch.cat(bits)
            name=version+'_'+route+'.pt';atomic_tensor(monitor.path/name,dict(ink=emissions[key],labels=labels));files[name]=sha(monitor.path/name)
            summary[key]=dict(cases=len(states),mean_f1=sum(f1s)/len(f1s),pixel_exact=exact,
                fraction_f1_at_least095=sum(v>=.95 for v in f1s)/len(f1s),nonstop_fraction=nonstop/len(f1s))
        del model
    boards=[]
    for route,(_,labels) in groups.items():
        indices=np.linspace(0,len(labels)-1,min(64,len(labels)),dtype=int).tolist()
        canvas=Image.new('L',(8*112,8*49),245);draw=ImageDraw.Draw(canvas)
        for i,index in enumerate(indices):
            x=(i%8)*112;y=(i//8)*49;draw.text((x+2,y),str(index),fill=0)
            for j,a in enumerate([gold[labels[index]].cpu(),emissions['baseline/'+route][index],emissions['genuine/'+route][index]]):
                canvas.paste(Image.fromarray((255*(1-a[0].numpy().astype(np.uint8))).astype(np.uint8)),(x+2+j*34,y+13))
        name=route+'_gold_baseline_genuine.png';canvas.resize((canvas.width*2,canvas.height*2),Image.Resampling.NEAREST).save(monitor.path/name);boards.append(name)
    g=meta['gate'];train=summary['genuine/copy_train'];held=summary['genuine/copy_held']
    passed=(train['mean_f1']>=g['training_copy_mean_f1'] and held['mean_f1']>=g['held_copy_mean_f1'] and
        held['fraction_f1_at_least095']>=g['held_fraction095'] and held['nonstop_fraction']>=g['held_first_nonstop_fraction'] and
        summary['genuine/original_histories']['mean_f1']>=summary['baseline/original_histories']['mean_f1']-g['old_cached_target_mean_f1_maximum_loss'])
    save_json(monitor.path/'records.json',records)
    save_json(monitor.path/'report.json',dict(summary=summary,readout_gate_passed=passed,raw_outputs_sha256=files,boards=boards,
        held_template_is_not_unseen_glyph_identity=True,full_language_assessed=False,teacher_loaded=False,body_loaded=False,
        actual_PNG_review_pending=True,**monitor.receipt()))
    print(json.dumps(dict(summary=summary,readout_gate_passed=passed)),flush=True)


def run(output,stage):
    meta=check(output);folder=output/stage
    if stage=='assessment':
        require(language.load(output/'qualify/report.json')['readout_gate_passed'],'head gate failed; preserve previous model')
        review=language.load(output/'qualify/visual_review.json')
        require(review['report_sha256']==sha(output/'qualify/report.json') and review['all_three_boards_actually_viewed'],
                'inspect actual fixed-state outputs before autonomous language assessment')
    folder.mkdir(exist_ok=False)
    if stage in ['cache','assessment']:monitor=language.Monitor(folder,meta[stage+'_maximum_seconds'])
    else:monitor=base.Monitor(folder,meta)
    monitor.status('admitting')
    try:
        if stage=='cache':cache(output,meta,monitor)
        elif stage=='assessment':language.assess(output,meta,monitor)
        else:
            with bridge_gpu(meta) as (device,admission):
                language.numerics();monitor.guard();save_json(folder/'admission.json',admission)
                {'fit':fit,'qualify':qualify}[stage](output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(language.load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','cache','fit','qualify','assessment'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
