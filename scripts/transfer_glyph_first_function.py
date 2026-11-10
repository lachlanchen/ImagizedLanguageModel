#!/usr/bin/env python3
"""Transfer an offline source decision function to a continuous pixel writer.

Synthetic states test function transfer, not language understanding. The full
source projection and its IDs are used only in calibration, never at inference.
Completed implementations, checkpoints and assessments remain unchanged.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require
from scripts.train_semantic_glyph_bridge import atomic_tensor, bridge_gpu
from scripts.train_semantic_glyph_language import bridge_model, numerics
from scripts.distill_semantic_glyph_readout import Monitor

PARENT = ROOT/'artifacts/balanced_glyph_writer_v1'
PARENT_SHA = 'dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d'
MAPPING = ROOT/'artifacts/semantic_glyph_readout_distillation_v1/mapping.json'
SOURCES = ['scripts/transfer_glyph_first_function.py',
           'references/glyph_first_function_transfer_protocol.md',
           'ilm/visual_lm/semantic_glyph_loop.py',
           'scripts/distill_semantic_glyph_readout.py',
           'scripts/train_semantic_glyph_bridge.py',
           'scripts/train_semantic_glyph_language.py']


def load(path): return json.loads(Path(path).read_text())
def rel(path): return str(Path(path).resolve().relative_to(ROOT))


def identity():
    values = {}
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
                == (ROOT/name).read_bytes(), 'commit function-transfer implementation first')
        values[name] = sha(ROOT/name)
    return values


def verify(run):
    m = load(run/'preparation.json')
    require(identity() == m['code_sha256'], 'function-transfer sources changed')
    for n,d in m['bindings_sha256'].items():
        require(sha(ROOT/n) == d, 'bound training input changed: '+n)
    return m


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation')
    require(not run.exists(), 'preserve completed and partial runs')
    checkpoint = PARENT/'fit/checkpoint_latest.pt'
    require(sha(checkpoint) == PARENT_SHA, 'fixed balanced parent required')
    parent = load(PARENT/'preparation.json')
    require(load(PARENT/'qualify/report.json')['readout_gate_passed'], 'parent physical qualification required')
    mapping = load(MAPPING)
    require(sum(i>=0 for i in mapping['teacher_to_glyph']) == 126213, 'source coverage differs')
    bindings = {}
    def bind(path): bindings[rel(path)] = sha(path)
    for path in [checkpoint, PARENT/'preparation.json', PARENT/'cache/pools.pt',
                 PARENT/'cache/report.json', MAPPING, ROOT/parent['primitive']/'training.pt']:
        bind(path)
    run.mkdir(parents=True)
    save_json(run/'preparation.json', dict(code_sha256=identity(), bindings_sha256=bindings,
        parent=rel(PARENT), initial_checkpoint=rel(checkpoint), inherited_checkpoint_sha256=PARENT_SHA,
        inherited_bridge_updates=36000, pool=rel(PARENT/'cache/pools.pt'), mapping=rel(MAPPING),
        primitive=parent['primitive'], genuine=parent['genuine'], seed=202611122,
        synthetic_training_rounds=2, synthetic_held_rounds=1, projection_iterations=24,
        calibration_batch=128, margin=1., fit_steps=18000, learning_rate=.0001,
        fit_batch=dict(synthetic=128, first=64, later=32, copy=32),
        synthetic_sampling='uniform form, then uniform accepted unit/base within form',
        source_projection='BF16-rounded matrix, FP32 multiplication without TF32',
        maximum_stage_seconds=3600., maximum_pressure_wait_seconds=1800.,
        maximum_vram_gib=14., minimum_entry_ram_gib=48., minimum_runtime_ram_gib=32.,
        gate=dict(synthetic_mean_f1=.95, synthetic_form_macro_f1=.95,
                  synthetic_fraction095=.8, training_mean_f1=.98,
                  cached_copy_mean_f1=.95, cached_copy_fraction095=.8),
        body_frozen=True, encoder_frozen=True, convolutional_decoder_frozen=True,
        stopping_frozen=True, readout_and_decoder_input_trainable=True,
        development_or_reserved_states_used=False, lexical_head_deployed=False,
        usable_language_demonstrated=False))
    print('Prepared full-unit first-form transfer; 126213 requested units per round', flush=True)


def read_shards(run):
    report = load(run/'calibrate/report.json')
    for row in report['shards']:
        require(sha(run/'calibrate'/row['file']) == row['sha256'], 'calibration shard changed')
    return report


def calibrate(run,m,monitor,device):
    import torch
    from torch.nn import functional as F
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    archive,_,_ = archive_and_tokenizer()
    source = archive.omitted['output.weight']
    array = np.array(archive.gguf.dequantize(source.data,source.tensor_type), dtype=np.float32,copy=True)
    head = torch.from_numpy(array).to(device,torch.bfloat16).float(); del array
    mapping = load(ROOT/m['mapping'])
    labels = torch.tensor(mapping['teacher_to_glyph'],device=device)
    requested = torch.where(labels>=0)[0]
    data = torch.load(ROOT/m['pool'],weights_only=True,map_location='cpu')
    # Every fifth TRAIN question is a withheld synthetic base. Both partitions
    # were used by the inherited upstream fit; this is not language holdout.
    first = data['first'].to(device)
    mask = torch.arange(len(first),device=device)%5 == 0
    variance = first[~mask].var(0).clamp_min(1e-4)
    generator = torch.Generator(device=device).manual_seed(m['seed'])
    shards=[];rounds=[]
    with torch.no_grad():
        for round_index in range(3):
            held = round_index == 2
            pool = first[mask if held else ~mask]
            chosen = torch.randint(len(pool),(len(requested),),generator=generator,device=device)
            hs=[];ys=[];units=[];distances=[];failed=[];margins=[]
            for start in range(0,len(requested),m['calibration_batch']):
                monitor.position=dict(round=round_index,held=held,completed=start,total=len(requested));monitor.guard()
                target=requested[start:start+m['calibration_batch']]
                original=pool[chosen[start:start+len(target)]];h=original.clone()
                for iteration in range(m['projection_iterations']):
                    scores=F.linear(h,head);winner=scores.argmax(-1);active=winner!=target
                    if not bool(active.any()):break
                    direction=head[target]-head[winner];d=variance*direction
                    step=(m['margin']-(direction*h).sum(-1)).clamp_min(0)/(direction*d).sum(-1).clamp_min(1e-8)
                    h=h+active[:,None]*step[:,None]*d
                scores=F.linear(h,head);winner=scores.argmax(-1);keep=winner==target
                require(bool(torch.isfinite(h).all()), 'nonfinite projected states')
                failed.extend(target[~keep].cpu().tolist())
                if bool(keep.any()):
                    hs.append(h[keep].cpu());ys.append(labels[target[keep]].cpu());units.append(target[keep].cpu())
                    distances.append((((h-original).square()/variance).mean(-1).sqrt())[keep].cpu())
                    top=scores[keep].topk(2,dim=-1).values;margins.append((top[:,0]-top[:,1]).cpu())
            require(hs, 'no accepted source states')
            value=dict(states=torch.cat(hs),labels=torch.cat(ys),source_units=torch.cat(units),
                       standardized_rms_displacement=torch.cat(distances),top_two_margin=torch.cat(margins))
            path=monitor.path/f'round_{round_index}.pt';atomic_tensor(path,value)
            shards.append(dict(file=path.name,sha256=sha(path),held=held))
            distance=value['standardized_rms_displacement']
            row=dict(round=round_index,held=held,requested=len(requested),accepted=len(value['states']),
                accepted_forms=len(value['labels'].unique()),failed_source_units=failed,
                displacement_median=float(distance.median()),displacement_maximum=float(distance.max()),
                minimum_top_two_margin=float(value['top_two_margin'].min()))
            rounds.append(row);monitor.status('calibrating_full_unit_function',rounds=rounds)
            print(json.dumps({k:v for k,v in row.items() if k!='failed_source_units'}),flush=True)
    save_json(monitor.path/'report.json',dict(shards=shards,rounds=rounds,
        source_model_sha256=mapping['source_model_sha256'],excluded=mapping['exclusions'],
        training_bases=int((~mask).sum()),held_bases=int(mask.sum()),
        no_new_language_examples=True,teacher_head_saved=False,body_loaded=False,
        projection_not_proven_natural_context=True,optimizer_updates=0,**monitor.receipt()))


def copy_pool(m):
    from scripts.train_genuine_glyph_readout import pools
    return pools(ROOT/m['genuine'],load(ROOT/m['genuine']/'preparation.json'))


def fit(run,m,monitor,device):
    import torch
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    report=read_shards(run)
    parts=[torch.load(run/'calibrate'/r['file'],weights_only=True,map_location='cpu') for r in report['shards'] if not r['held']]
    h=torch.cat([p['states'] for p in parts]).to(device);y=torch.cat([p['labels'] for p in parts]).to(device);del parts
    order=y.argsort(stable=True);forms,counts=y[order].unique_consecutive(return_counts=True)
    offsets=torch.cat([counts.new_zeros(1),counts.cumsum(0)[:-1]])
    p={n:v.to(device) for n,v in torch.load(ROOT/m['pool'],weights_only=True,map_location='cpu').items()}
    copied=tuple(v.to(device) for v in copy_pool(m)['copy_train'])
    gold=(1-2*torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'].float()).to(device)
    model=bridge_model({},device,ROOT/m['initial_checkpoint'])
    model.readout.requires_grad_(True);model.decoder_input.requires_grad_(True)
    params=[v for v in model.parameters() if v.requires_grad]
    frozen=[(v,v._version) for v in model.parameters() if not v.requires_grad]
    original=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
    opt=torch.optim.AdamW(params,lr=m['learning_rate'],weight_decay=.01)
    digest=hashlib.sha256();counts_seen={n:0 for n in m['fit_batch']}
    for step in range(m['fit_steps']):
        monitor.position=dict(step=step,total=m['fit_steps']);monitor.guard();opt.zero_grad(set_to_none=True)
        group=torch.randint(len(forms),(128,),device=device)
        ix=order[offsets[group]+(torch.rand(128,device=device)*counts[group]).long()]
        a=torch.randint(len(p['first']),(64,),device=device)
        b=torch.randint(len(p['later']),(32,),device=device)
        c=torch.randint(len(copied[0]),(32,),device=device)
        for name,indices in zip(counts_seen,[ix,a,b,c]):
            digest.update(name.encode());digest.update(indices.cpu().numpy().tobytes());counts_seen[name]+=len(indices)
        states=torch.cat([h[ix],p['first'][a],p['later'][b],copied[0][c]])
        targets=gold[torch.cat([y[ix],p['first_labels'][a],p['later_labels'][b],copied[1][c]])]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_,_=model.predict(states)
            losses=[reconstruction_loss(logits[lo:hi],targets[lo:hi]) for lo,hi in [(0,128),(128,192),(192,224),(224,256)]]
            loss=.5*losses[0]+.25*losses[1]+.125*losses[2]+.125*losses[3]
        require(bool(torch.isfinite(loss)), 'nonfinite function-transfer loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        for group in opt.param_groups:group['lr']=m['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/m['fit_steps'])))
        require(all(v._version==ver for v,ver in frozen), 'frozen read/feedback parameters changed')
        if (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),pixel_groups=[float(v.detach()) for v in losses])
            with (monitor.path/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            monitor.status('training_pixel_function',**row);print(json.dumps(row),flush=True)
        if (step+1)%1000==0:
            path=monitor.path/'checkpoint_latest.pt'
            if path.exists():path.replace(monitor.path/'checkpoint_previous.pt')
            atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                body_tail_adapters=original['body_tail_adapters'],tail_start_layer=32,step=step+1,
                inherited_bridge_updates=36000,new_optimizer_updates=step+1,
                language_training_complete=True,optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                preparation_sha256=sha(run/'preparation.json'),usable_language_demonstrated=False))
    require(counts_seen=={n:v*m['fit_steps'] for n,v in m['fit_batch'].items()},'fixed sample counts differ')
    for name,value in model.state_dict().items():
        if not name.startswith(('readout.','decoder_input.')):
            require(torch.equal(value.cpu(),original['model'][name]), 'non-writer parameter changed')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), 'nonfinite checkpoint')
    save_json(monitor.path/'report.json',dict(actual_optimizer_updates=m['fit_steps'],
        checkpoint_sha256=sha(monitor.path/'checkpoint_latest.pt'),inherited_checkpoint_sha256=PARENT_SHA,
        trained_parameters=sum(v.numel() for v in params),sampled_counts=counts_seen,sampler_sha256=digest.hexdigest(),
        all_frozen_parameters_bit_exact=True,teacher_head_loaded=False,full_body_loaded=False,
        model_architecture_unchanged=True,**monitor.receipt()))


def qualify(run,m,monitor,device):
    import torch
    from PIL import Image,ImageDraw
    report=read_shards(run);require(load(run/'fit/status.json')['phase']=='completed','complete fit first')
    p=torch.load(ROOT/m['pool'],weights_only=True,map_location='cpu')
    copies=copy_pool(m)['copy_held']
    routes=[]
    for row in report['shards']:
        if row['held']:
            d=torch.load(run/'calibrate'/row['file'],weights_only=True,map_location='cpu')
            routes.append(('synthetic',d['states'],d['labels']))
    routes += [('first',p['first'],p['first_labels']),('later',p['later'],p['later_labels']),('cached_copy',*copies)]
    gold=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'].to(device).bool()
    summary={};boards=[]
    with torch.no_grad():
        for version,path in [('parent',ROOT/m['initial_checkpoint']),('transferred',run/'fit/checkpoint_latest.pt')]:
            model=bridge_model({},device,path)
            for name,h,y in routes:
                fs=[];exacts=[];all_ink=[]
                for start in range(0,len(h),128):
                    monitor.position=dict(model=version,route=name,completed=start,total=len(h));monitor.guard()
                    # FP32 matches the actual deployed bridge, not training autocast.
                    logits,_,_=model.predict(h[start:start+128].to(device));actual=logits>=0;target=gold[y[start:start+128].to(device)]
                    intersection=(actual&target).flatten(1).sum(-1);den=(actual.flatten(1).sum(-1)+target.flatten(1).sum(-1))
                    fs.extend(torch.where(den==0,torch.ones_like(den).float(),2*intersection/den).cpu().tolist())
                    exacts.extend((actual==target).flatten(1).all(-1).cpu().tolist());all_ink.append(actual.cpu().to(torch.uint8))
                raw=monitor.path/(version+'_'+name+'.pt');atomic_tensor(raw,dict(ink=torch.cat(all_ink),labels=y,f1=torch.tensor(fs),exact=torch.tensor(exacts)))
                by_form={}
                for label,f in zip(y.tolist(),fs):by_form.setdefault(label,[]).append(f)
                summary[version+'/'+name]=dict(cases=len(fs),mean_f1=float(np.mean(fs)),
                    fraction095=float(np.mean(np.array(fs)>=.95)),pixel_exact=sum(exacts),
                    form_macro_f1=float(np.mean([np.mean(v) for v in by_form.values()])),raw_sha256=sha(raw))
                # Deterministic first 32 rows; show all errors without selection.
                indices=np.linspace(0,len(h)-1,32,dtype=int);canvas=Image.new('L',(420,560),255);draw=ImageDraw.Draw(canvas)
                draw.text((4,3),version+' / '+name+'; target | output',fill=0)
                outputs=torch.cat(all_ink)[:,0].numpy()
                for j,index in enumerate(indices):
                    x=(j%4)*104;y0=26+(j//4)*66;draw.text((x,y0),str(index),fill=0)
                    for k,im in enumerate([gold[y[index],0].cpu().numpy(),outputs[index]]):
                        canvas.paste(Image.fromarray((255*(1-im.astype(np.uint8))).astype(np.uint8)),(x+k*34,y0+15))
                board=monitor.path/(version+'_'+name+'.png');canvas.save(board);boards.append(dict(file=board.name,sha256=sha(board)))
            del model
    s=summary;g=m['gate'];new='transferred/'
    synth=s[new+'synthetic'];real=(s[new+'first']['mean_f1']*3176+s[new+'later']['mean_f1']*20086)/23262
    copy=s[new+'cached_copy']
    passed=(synth['mean_f1']>=g['synthetic_mean_f1'] and synth['form_macro_f1']>=g['synthetic_form_macro_f1']
        and synth['fraction095']>=g['synthetic_fraction095'] and real>=g['training_mean_f1']
        and copy['mean_f1']>=g['cached_copy_mean_f1'] and copy['fraction095']>=g['cached_copy_fraction095'])
    save_json(monitor.path/'report.json',dict(summary=summary,complete_training_mean_f1=real,
        function_gate_passed=bool(passed),boards=boards,checkpoint_sha256=sha(run/'fit/checkpoint_latest.pt'),
        cached_copy_states_from_unadapted_source=True,actual_adapted_full_reader_copy_not_assessed=True,
        stopping_parameters_retained=True,language_accuracy_measured=False,**monitor.receipt()))
    print(json.dumps(dict(summary=summary,function_gate_passed=bool(passed))),flush=True)


def execute(run,stage):
    m=verify(run);folder=run/stage;folder.mkdir(exist_ok=False)
    monitor=Monitor(folder,m);monitor.status('admitting')
    try:
        with bridge_gpu(m) as (device,admission):
            numerics();__import__('torch').manual_seed(m['seed']);__import__('torch').cuda.manual_seed_all(m['seed'])
            save_json(folder/'admission.json',admission);monitor.guard()
            globals()[stage](run,m,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['prepare','calibrate','fit','qualify'])
    p.add_argument('--run',type=Path,default=Path('artifacts/glyph_first_function_transfer_v1'));a=p.parse_args()
    if a.stage=='prepare':prepare(a.run.resolve())
    else:execute(a.run.resolve(),a.stage)
