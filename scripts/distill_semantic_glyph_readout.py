#!/usr/bin/env python3
"""Offline teacher-function calibration of the continuous glyph readout.

No vocabulary head, source IDs or glyph table enter the resulting runtime.
Calibration states are synthetic function-transfer data, not new language data.
Completed language/primitive source files and checkpoints remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import require, host_resources
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu
from scripts.train_semantic_glyph_language import load, relative, numerics

LANGUAGE=ROOT/'artifacts/semantic_glyph_language_v1'
PRIMITIVE=ROOT/'artifacts/semantic_glyph_bridge_v1'
FILES=('scripts/distill_semantic_glyph_readout.py',
       'references/semantic_readout_distillation_protocol.md',
       'ilm/visual_lm/semantic_glyph_loop.py',
       'scripts/train_semantic_glyph_bridge.py',
       'scripts/train_semantic_glyph_language.py',
       'scripts/check_native_visual_transfer_v71.py',
       'ilm/visual_lm/gguf_visual_language.py',
       'ilm/visual_lm/gguf_language_trunk.py',
       'scripts/score_pixel_language_first_blocks.py')


def identity():
    values={n:sha(ROOT/n) for n in FILES}
    for n,d in values.items():
        require(hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)).hexdigest()==d,
                'commit the fixed distillation implementation first: '+n)
    return values


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    require(not output.exists(),'preserve all previous and partial attempts')
    import torch
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    from ilm.visual_lm.grounded_image_weights import MODEL_SHA
    code=identity();torch.set_num_threads(4)
    require(load(LANGUAGE/'fit/status.json')['phase']=='completed','completed language baseline required')
    require(load(LANGUAGE/'cache/status.json')['phase']=='completed','completed causal cache required')
    inventory=load(PRIMITIVE/'inventory.json')['records']
    glyphs={c:i for i,r in enumerate(inventory) for c in r['chars']}
    archive,tokenizer,_=archive_and_tokenizer()
    vocab=archive.omitted['output.weight'];count=int(vocab.shape[-1])
    # The archive shape is source GGUF ordering, reversed from PyTorch.
    require(count==len(archive.field('tokenizer.ggml.tokens')),'teacher output shape differs')
    eog=set(map(int,archive.field('tokenizer.ggml.eos_token_ids')))
    mapping=[];reasons=Counter()
    for i in range(count):
        text=tokenizer.decode([i],skip_special_tokens=False)
        if i in eog: label=-1;reason='terminal'
        elif not text or '\ufffd' in text: label=-2;reason='empty_or_partial_unicode'
        elif text[0] not in glyphs: label=-2;reason='first_form_outside_training_inventory'
        else: label=glyphs[text[0]];reason='valid_first_complete_form'
        mapping.append(label);reasons[reason]+=1
    requested=[r['source_ids'][0] for r in inventory]
    require(all(mapping[k]==i for i,k in enumerate(requested)), 'single-form anchor does not roundtrip')
    source=load(LANGUAGE/'cache/report.json');parts=[];labels=[];split=[]
    bindings={relative(LANGUAGE/'preparation.json'):sha(LANGUAGE/'preparation.json'),
        relative(LANGUAGE/'cache/report.json'):sha(LANGUAGE/'cache/report.json'),
        relative(LANGUAGE/'fit/checkpoint_latest.pt'):load(LANGUAGE/'fit/report.json')['checkpoint_sha256'],
        relative(PRIMITIVE/'training.pt'):load(PRIMITIVE/'preparation.json')['training_sha256'],
        relative(PRIMITIVE/'inventory.json'):sha(PRIMITIVE/'inventory.json')}
    for position,row in enumerate(source['files']):
        path=LANGUAGE/'cache'/row['file'];require(sha(path)==row['sha256'],'causal cache changed')
        bindings[relative(path)]=row['sha256']
        v=torch.load(path,weights_only=True,map_location='cpu')
        parts.append(v['states']);labels.append(v['labels'])
        # Whole question histories kept together. No development input enters.
        split.extend([int(position%5==0)]*len(v['states']))
    states=torch.cat(parts).float();labels=torch.cat(labels);split=torch.tensor(split)
    require(states.shape==(len(labels),4096) and torch.isfinite(states).all(),'invalid training histories')
    output.mkdir(parents=True)
    atomic_tensor(output/'genuine.pt',dict(states=states,reference_labels=labels,split=split))
    save_json(output/'mapping.json',dict(teacher_to_glyph=mapping,requested_teacher_ids=requested,
        exclusions=dict(reasons),source_model_sha256=MODEL_SHA,training_only=True))
    save_json(output/'preparation.json',dict(code_sha256=code,bindings_sha256=bindings,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        genuine_sha256=sha(output/'genuine.pt'),mapping_sha256=sha(output/'mapping.json'),
        primitive=relative(PRIMITIVE),baseline=relative(LANGUAGE/'fit/checkpoint_latest.pt'),
        selected_from_training_inventory_only=True,development_training_allowed=False,
        seed=202611103,synthetic_train_per_form=4,synthetic_test_per_form=1,
        projection_iterations=24,margin=1.,calibration_batch=32,
        fit_steps=6000,batch_size=128,learning_rate=.0002,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,maximum_vram_gib=8.,
        maximum_stage_seconds=1800.,maximum_pressure_wait_seconds=300.,
        gate=dict(mean_f1_minimum=.95,fraction_f1_at_least_095_minimum=.8,
                  both_synthetic_and_genuine=True),
        vocabulary_head_is_temporary_training_teacher=True,runtime_changed=False))
    print(json.dumps(dict(prepared=True,genuine_states=len(states),forms=len(inventory),exclusions=dict(reasons))),flush=True)


def check(output):
    meta=load(output/'preparation.json')
    require(identity()==meta['code_sha256'],'distillation implementation changed')
    for n,d in meta['bindings_sha256'].items():require(sha(ROOT/n)==d,'bound input changed: '+n)
    for n in ('genuine','mapping'):require(sha(output/(n+('.pt' if n=='genuine' else '.json')))==meta[n+'_sha256'],'prepared data changed')
    return meta


class Monitor:
    def __init__(self,path,meta):
        self.path=path;self.meta=meta;self.began=time.perf_counter();self.waited=0.;self.samples=[];self.position={}
    def status(self,phase,**extra):
        save_json(self.path/'status.json',dict(phase=phase,pid=os.getpid(),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            elapsed_seconds=time.perf_counter()-self.began,position=self.position,**extra))
    def guard(self):
        import torch
        require(time.perf_counter()-self.began<self.meta['maximum_stage_seconds'],'distillation stage time cap')
        require(torch.cuda.max_memory_allocated()/1024**3<=self.meta['maximum_vram_gib'],'distillation VRAM cap')
        while True:
            v=host_resources();self.samples.append(v)
            require(v['available_gib']>=self.meta['minimum_runtime_ram_gib'],'distillation RAM reserve')
            if v['some_pressure_avg10']<=.5 and v['full_pressure_avg10']<=.1:break
            remaining=self.meta['maximum_pressure_wait_seconds']-self.waited
            require(remaining>0,'distillation pressure wait exhausted')
            self.status('waiting_for_memory_pressure',resources=v)
            began=time.perf_counter();time.sleep(min(10.,remaining));self.waited+=time.perf_counter()-began
    def receipt(self):
        import torch
        return dict(elapsed_seconds=time.perf_counter()-self.began,resource_wait_seconds=self.waited,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
            minimum_available_gib=min(v['available_gib'] for v in self.samples))


def bridge(meta,device,checkpoint=None):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
    model=SemanticGlyphBridge().to(device)
    model.load_state_dict(torch.load(checkpoint or ROOT/meta['baseline'],weights_only=True,map_location='cpu')['model'])
    return model.requires_grad_(False).eval()


@__import__('torch').no_grad()
def calibrate(output,meta,monitor,device):
    import torch
    from torch.nn import functional as F
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    archive,_,_=archive_and_tokenizer();source=archive.omitted['output.weight']
    # Reuse the authenticated shared archive. Never save a second head copy.
    array=np.array(archive.gguf.dequantize(source.data,source.tensor_type),dtype=np.float32,copy=True)
    head=torch.from_numpy(array).to(device);del array
    require(head.shape[1]==4096,'teacher head shape differs')
    mapping=load(output/'mapping.json')
    labelmap=torch.tensor(mapping['teacher_to_glyph'],device=device)
    requested=torch.tensor(mapping['requested_teacher_ids'],device=device)
    data=torch.load(output/'genuine.pt',weights_only=True,map_location='cpu')
    states=data['states'].to(device);split=data['split'].to(device)
    variance=states[split==0].var(0).clamp_min(1e-4)
    genuine=[]
    for start in range(0,len(states),meta['calibration_batch']):
        monitor.position=dict(route='genuine_teacher_labels',start=start,total=len(states));monitor.guard()
        winner=F.linear(states[start:start+meta['calibration_batch']],head).argmax(-1)
        genuine.append(labelmap[winner].cpu())
    genuine=torch.cat(genuine)
    atomic_tensor(monitor.path/'genuine_labels.pt',dict(labels=genuine))
    parts=[];labels=[];splits=[];distances=[];accepted_counts=[]
    generator=torch.Generator(device=device).manual_seed(meta['seed'])
    rounds=meta['synthetic_train_per_form']+meta['synthetic_test_per_form']
    for round_index in range(rounds):
        held=round_index>=meta['synthetic_train_per_form']
        pool=states[split==int(held)]
        chosen=torch.randint(len(pool),(len(requested),),generator=generator,device=device)
        accepted=0
        for start in range(0,len(requested),meta['calibration_batch']):
            monitor.position=dict(round=round_index,held_out=held,start=start,total=len(requested));monitor.guard()
            target=requested[start:start+meta['calibration_batch']]
            original=pool[chosen[start:start+len(target)]];h=original.clone()
            for iteration in range(meta['projection_iterations']):
                winner=F.linear(h,head).argmax(-1);active=winner!=target
                if not bool(active.any()):break
                direction=head[target]-head[winner]
                d=variance*direction
                gap=(direction*h).sum(-1)
                length=(direction*d).sum(-1).clamp_min(1e-8)
                step=(meta['margin']-gap).clamp_min(0)/length
                h=h+active[:,None]*step[:,None]*d
            # Validate with the complete teacher again; never trust requested IDs.
            winner=F.linear(h,head).argmax(-1);keep=winner==target
            index=torch.arange(start,start+len(target),device=device)
            if bool(keep.any()):
                parts.append(h[keep].cpu());labels.append(index[keep].cpu())
                splits.append(torch.full((int(keep.sum()),),int(held),dtype=torch.long))
                distances.append((((h-original).square()/variance).mean(-1).sqrt())[keep].cpu())
            accepted+=int(keep.sum())
        accepted_counts.append(dict(round=round_index,held_out=held,accepted=accepted,requested=len(requested)))
        monitor.status('projecting_teacher_function',rounds=accepted_counts)
        print(json.dumps(accepted_counts[-1]),flush=True)
    require(parts,'no counterfactual states accepted')
    value=dict(states=torch.cat(parts),labels=torch.cat(labels),split=torch.cat(splits),
               standardized_rms_displacement=torch.cat(distances))
    path=monitor.path/'calibration.pt';atomic_tensor(path,value)
    d=value['standardized_rms_displacement']
    save_json(monitor.path/'report.json',dict(rounds=accepted_counts,
        calibration_sha256=sha(path),genuine_labels_sha256=sha(monitor.path/'genuine_labels.pt'),
        genuine_supported=int((genuine>=0).sum()),genuine_terminal=int((genuine==-1).sum()),
        genuine_unsupported=int((genuine==-2).sum()),
        accepted_forms_train=len(value['labels'][value['split']==0].unique()),
        accepted_forms_test=len(value['labels'][value['split']==1].unique()),
        standardized_rms_displacement=dict(median=float(d.median()),maximum=float(d.max()),mean=float(d.mean())),
        teacher_head_saved=False,body_loaded=False,new_language_examples=0,optimizer_updates=0,**monitor.receipt()))


def calibrated(output):
    require(load(output/'calibrate/status.json')['phase']=='completed','complete calibration first')
    report=load(output/'calibrate/report.json')
    for name,key in [('calibration.pt','calibration_sha256'),('genuine_labels.pt','genuine_labels_sha256')]:
        require(sha(output/'calibrate'/name)==report[key],'teacher labels changed')


def fit(output,meta,monitor,device):
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    calibrated(output);torch.manual_seed(meta['seed']);torch.cuda.manual_seed_all(meta['seed'])
    model=bridge(meta,device);model.readout.requires_grad_(True)
    data=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    images=(1-2*data['ink'].float()).to(device)
    targets=((data['semantic']-data['mean'])/data['scale']).to(device)
    synth=torch.load(output/'calibrate/calibration.pt',weights_only=True,map_location='cpu')
    genuine=torch.load(output/'genuine.pt',weights_only=True,map_location='cpu')
    real_labels=torch.load(output/'calibrate/genuine_labels.pt',weights_only=True,map_location='cpu')['labels']
    a=synth['split']==0;b=(genuine['split']==0)&(real_labels>=0)
    pools=[(synth['states'][a].to(device),synth['labels'][a].to(device)),
           (genuine['states'][b].to(device),real_labels[b].to(device))]
    require(all(len(x) for x,y in pools),'empty distillation pool')
    params=list(model.readout.parameters());opt=torch.optim.AdamW(params,lr=meta['learning_rate'],weight_decay=.01)
    for step in range(1,meta['fit_steps']+1):
        if step%20==1:monitor.position=dict(step=step,total=meta['fit_steps']);monitor.guard()
        batch=[];labels=[]
        for x,y in pools:
            ix=torch.randint(len(x),(meta['batch_size']//2,),device=device);batch.append(x[ix]);labels.append(y[ix])
        h=torch.cat(batch);label=torch.cat(labels);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.readout(h).float();code=F.mse_loss(z,targets[label]);cos=1-F.cosine_similarity(z,targets[label]).mean()
            pix=reconstruction_loss(model.decode_logits(z),images[label]);loss=code+.2*cos+pix
        require(bool(torch.isfinite(loss)),'nonfinite teacher-function loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        fraction=step/meta['fit_steps']
        for group in opt.param_groups:group['lr']=meta['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
        if step%100==0:
            row=dict(step=step,loss=float(loss),code=float(code),pixel=float(pix),cosine=float(cos),seconds=time.perf_counter()-monitor.began)
            with (monitor.path/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            monitor.status('distilling_readout');print(json.dumps(row),flush=True)
    path=monitor.path/'checkpoint_latest.pt'
    atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
        language_training_complete=True,step=meta['fit_steps'],primitive_only=False,
        readout_distillation_only=True,usable_language_demonstrated=False,
        preparation_sha256=sha(output/'preparation.json')))
    # Stops are retained exactly; this stage does not invent stopping targets.
    original=bridge(meta,'cpu')
    require(all(torch.equal(original.state_dict()[k],v.cpu()) for k,v in model.state_dict().items() if not k.startswith('readout.')),
            'non-readout weights changed')
    save_json(monitor.path/'report.json',dict(steps=meta['fit_steps'],checkpoint_sha256=sha(path),
        trained_parameters=sum(p.numel() for p in params),synthetic_states=len(pools[0][0]),genuine_states=len(pools[1][0]),
        non_readout_weights_exact=True,teacher_head_loaded=False,body_loaded=False,**monitor.receipt()))


@__import__('torch').no_grad()
def qualify(output,meta,monitor,device):
    import torch
    calibrated(output);require(load(output/'fit/status.json')['phase']=='completed','complete distillation first')
    path=output/'fit/checkpoint_latest.pt';require(sha(path)==load(output/'fit/report.json')['checkpoint_sha256'],'candidate changed')
    primitive=torch.load(ROOT/meta['primitive']/'training.pt',weights_only=True,map_location='cpu')
    ink=primitive['ink'].to(device).bool()
    synthetic=torch.load(output/'calibrate/calibration.pt',weights_only=True,map_location='cpu')
    genuine=torch.load(output/'genuine.pt',weights_only=True,map_location='cpu')
    labels=torch.load(output/'calibrate/genuine_labels.pt',weights_only=True,map_location='cpu')['labels']
    a=synthetic['split']==1;b=(genuine['split']==1)&(labels>=0)
    pools={'synthetic':(synthetic['states'][a],synthetic['labels'][a]),
           'genuine':(genuine['states'][b],labels[b])}
    records=[];summary={}
    for version,checkpoint in [('baseline',ROOT/meta['baseline']),('distilled',path)]:
        model=bridge(meta,device,checkpoint)
        for route,(states,targets) in pools.items():
            current=[]
            for start in range(0,len(states),64):
                monitor.position=dict(model=version,route=route,start=start,total=len(states));monitor.guard()
                logits,_,_=model.predict(states[start:start+64].to(device));actual=logits>=0
                gold=ink[targets[start:start+64].to(device)];intersection=(actual&gold).flatten(1).sum(-1)
                denominator=actual.flatten(1).sum(-1)+gold.flatten(1).sum(-1)
                f1=torch.where(denominator==0,torch.ones_like(denominator).float(),2*intersection/denominator)
                exact=(actual==gold).flatten(1).all(-1)
                for i in range(len(f1)):
                    row=dict(model=version,route=route,index=start+i,label=int(targets[start+i]),f1=float(f1[i]),pixel_exact=bool(exact[i]))
                    current.append(row);records.append(row)
            require(current,'empty held-out calibration assessment')
            summary[version+'/'+route]=dict(count=len(current),mean_f1=sum(r['f1'] for r in current)/len(current),
                pixel_exact=sum(r['pixel_exact'] for r in current),fraction_f1_at_least_095=sum(r['f1']>=.95 for r in current)/len(current))
        del model
    gate=all(summary['distilled/'+r]['mean_f1']>=meta['gate']['mean_f1_minimum'] and
             summary['distilled/'+r]['fraction_f1_at_least_095']>=meta['gate']['fraction_f1_at_least_095_minimum'] for r in pools)
    save_json(monitor.path/'records.json',records)
    save_json(monitor.path/'report.json',dict(summary=summary,compiler_gate_passed=gate,
        records_sha256=sha(monitor.path/'records.json'),language_assessed=False,
        heldout_calibration_is_not_language_understanding=True,**monitor.receipt()))
    print(json.dumps(dict(summary=summary,compiler_gate_passed=gate)),flush=True)


def run(output,stage):
    meta=check(output);folder=output/stage;folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,meta);monitor.status('admitting')
    try:
        with bridge_gpu(meta) as (device,admission):
            save_json(folder/'admission.json',admission);numerics();monitor.guard()
            globals()[stage](output,meta,monitor,device)
        monitor.status('completed')
    except BaseException as error:
        monitor.status('failed',reason=repr(error));raise
    finally:
        save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','calibrate','fit','qualify'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.stage=='prepare':prepare(a.output.resolve())
    else:run(a.output.resolve(),a.stage)
