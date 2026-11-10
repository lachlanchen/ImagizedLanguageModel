#!/usr/bin/env python3
"""Fit and qualify the unchanged continuous writer on real first states."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image,ImageDraw

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/train_natural_first_writer.py'
FIGURE_STEPS=(9000,18000,27000)


def load(p):return json.loads(Path(p).read_text())


def identity():
    names=[OWN,'references/natural_first_image_protocol.md','references/natural_writer_learning_trajectory.md',
           'scripts/cache_natural_first_states.py','ilm/visual_lm/semantic_glyph_loop.py',
           'scripts/train_genuine_glyph_readout.py','scripts/train_semantic_glyph_bridge.py',
           'scripts/train_semantic_glyph_language.py','scripts/distill_semantic_glyph_readout.py']
    values={}
    for n in names:
        require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'commit natural writer recipe first')
        values[n]=sha(ROOT/n)
    return values


def verify(run):
    m=load(run/'writer_preparation.json');require(identity()==m['code_sha256'],'natural writer recipe changed')
    for n,d in m['bindings_sha256'].items():require(sha(ROOT/n)==d,'natural writer input changed: '+n)
    return m


def prepare(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only writer preparation')
    require(not (run/'writer_preparation.json').exists(),'preserve prepared recipe')
    from scripts.cache_natural_first_states import verify as verify_cache
    c=verify_cache(run);cache=load(run/'cache/report.json')
    require(cache['actual_first_states_from_unchanged_adapted_body'],'actual natural image cache required')
    parent=ROOT/'artifacts/balanced_glyph_writer_v1';p=load(parent/'preparation.json')
    bindings={}
    for path in [run/'preparation.json',run/'cache/report.json',run/'cache/index.json',
                 ROOT/c['data']/'records.json',ROOT/c['checkpoint'],parent/'cache/pools.pt',parent/'cache/report.json',
                 ROOT/p['primitive']/'training.pt']:
        bindings[str(path.relative_to(ROOT))]=sha(path)
    save_json(run/'writer_preparation.json',dict(code_sha256=identity(),bindings_sha256=bindings,
        initial_checkpoint=c['checkpoint'],inherited_checkpoint_sha256=c['checkpoint_sha256'],
        natural_data=c['data'],primitive=p['primitive'],genuine=p['genuine'],
        original_pool=str((parent/'cache/pools.pt').relative_to(ROOT)),
        fit_steps=36000,figure_writer_steps=list(FIGURE_STEPS),seed=202611123,learning_rate=.0001,
        maximum_stage_seconds=3600.,maximum_pressure_wait_seconds=1800.,maximum_vram_gib=14.,
        minimum_entry_ram_gib=48.,minimum_runtime_ram_gib=32.,
        held_contexts_not_used_for_fitting=True,body_and_stopping_frozen=True,
        inherited_bridge_updates=36000,new_optimizer_updates=36000))


def natural(run,m):
    import torch
    rows=load(ROOT/m['natural_data']/'records.json');parts=[]
    position=0
    for row in load(run/'cache/report.json')['files']:
        path=run/'cache'/row['file'];require(sha(path)==row['sha256'],'actual natural cache changed')
        v=torch.load(path,weights_only=True,map_location='cpu')
        require(v['record_indices'].tolist()==list(range(position,position+row['records'])),'natural cache order differs')
        require(v['labels'].tolist()==[r['label'] for r in rows[position:position+row['records']]],'natural targets differ')
        parts.append(v['states']);position+=row['records']
    require(position==len(rows),'complete natural cache required')
    return torch.cat(parts).float(),rows


def copy_pool(m):
    from scripts.train_genuine_glyph_readout import pools
    return pools(ROOT/m['genuine'],load(ROOT/m['genuine']/'preparation.json'))


def fit(run,m,monitor,device):
    import torch
    from scripts.train_semantic_glyph_bridge import atomic_tensor
    from scripts.train_semantic_glyph_language import bridge_model
    from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss
    h,rows=natural(run,m);keep=torch.tensor([r['split']=='train' and r['label']>=0 for r in rows])
    labels=torch.tensor([r['label'] for r in rows])[keep].to(device);h=h[keep].to(device)
    require(len(h)>3176,'broader natural training coverage required')
    p={n:v.to(device) for n,v in torch.load(ROOT/m['original_pool'],weights_only=True,map_location='cpu').items()}
    copied=tuple(v.to(device) for v in copy_pool(m)['copy_train'])
    gold=(1-2*torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'].float()).to(device)
    model=bridge_model({},device,ROOT/m['initial_checkpoint']);model.readout.requires_grad_(True)
    params=list(model.readout.parameters());frozen=[(v,v._version) for v in model.parameters() if not v.requires_grad]
    initial=torch.load(ROOT/m['initial_checkpoint'],weights_only=True,map_location='cpu')
    opt=torch.optim.AdamW(params,lr=m['learning_rate'],weight_decay=.01)
    sampler=hashlib.sha256();counts=dict(natural=0,first=0,later=0,copy=0)
    snapshots=[]
    preparation_sha256=sha(run/'writer_preparation.json')
    for step in range(m['fit_steps']):
        monitor.position=dict(step=step,total=m['fit_steps']);monitor.guard();opt.zero_grad(set_to_none=True)
        ix=torch.randint(len(h),(128,),device=device);a=torch.randint(len(p['first']),(64,),device=device)
        b=torch.randint(len(p['later']),(32,),device=device);c=torch.randint(len(copied[0]),(32,),device=device)
        for name,index in zip(counts,[ix,a,b,c]):
            counts[name]+=len(index);sampler.update(name.encode());sampler.update(index.cpu().numpy().tobytes())
        states=torch.cat([h[ix],p['first'][a],p['later'][b],copied[0][c]])
        targets=gold[torch.cat([labels[ix],p['first_labels'][a],p['later_labels'][b],copied[1][c]])]
        logits,_,_=model.predict(states)
        groups=[reconstruction_loss(logits[lo:hi],targets[lo:hi]) for lo,hi in [(0,128),(128,192),(192,224),(224,256)]]
        loss=.5*groups[0]+.25*groups[1]+.125*groups[2]+.125*groups[3]
        require(bool(torch.isfinite(loss)),'nonfinite natural pixel loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
        for group in opt.param_groups:group['lr']=m['learning_rate']*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/m['fit_steps'])))
        require(all(v._version==ver for v,ver in frozen),'frozen image/stop parameters changed')
        if step+1 in m['figure_writer_steps']:
            path=monitor.path/f'writer_step_{step+1:05}.pt'
            atomic_tensor(path,dict(step=step+1,preparation_sha256=preparation_sha256,
                parent_checkpoint_sha256=m['inherited_checkpoint_sha256'],
                readout={name:value.detach().cpu() for name,value in model.readout.state_dict().items()}))
            snapshots.append(dict(step=step+1,file=path.name,sha256=sha(path),sampled_counts=dict(counts)))
        if (step+1)%100==0:
            row=dict(step=step+1,loss=float(loss.detach()),pixel_groups=[float(v.detach()) for v in groups])
            with (monitor.path/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            monitor.status('fitting_natural_first_images',**row);print(json.dumps(row),flush=True)
        if (step+1)%1000==0:
            path=monitor.path/'checkpoint_latest.pt'
            if path.exists():path.replace(monitor.path/'checkpoint_previous.pt')
            atomic_tensor(path,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                body_tail_adapters=initial['body_tail_adapters'],tail_start_layer=32,step=step+1,
                inherited_bridge_updates=36000,new_optimizer_updates=step+1,
                writer_training_family='natural_first_image_v1',
                language_training_complete=step+1==m['fit_steps'],optimizer=opt.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                preparation_sha256=preparation_sha256,usable_language_demonstrated=False))
    require(counts==dict(natural=4608000,first=2304000,later=1152000,copy=1152000),'actual exposure counts differ')
    require([row['step'] for row in snapshots]==m['figure_writer_steps'],'learning-trajectory snapshots missing')
    for name,value in model.state_dict().items():
        if not name.startswith('readout.'):require(torch.equal(value.cpu(),initial['model'][name]),'non-readout model changed')
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()),'nonfinite final writer')
    save_json(monitor.path/'report.json',dict(checkpoint_sha256=sha(monitor.path/'checkpoint_latest.pt'),
        actual_optimizer_updates=m['fit_steps'],sampled_counts=counts,sampler_sha256=sampler.hexdigest(),
        all_checks_passed=True,figure_writer_snapshots=snapshots,
        natural_training_states=len(h),natural_training_forms=len(labels.unique()),
        natural_unsupported_training_excluded=sum(r['split']=='train' and r['label']<0 for r in rows),
        held_contexts_used_for_training=False,all_non_readout_parameters_bit_exact=True,
        trained_parameters=sum(v.numel() for v in params),source_lexical_head_loaded=False,full_body_loaded=False,
        training_arithmetic='FP32, no TF32 or autocast',language_accuracy_measured=False,**monitor.receipt()))


def qualify(run,m,monitor,device):
    import torch
    from scripts.train_semantic_glyph_bridge import atomic_tensor
    from scripts.train_semantic_glyph_language import bridge_model
    from scripts.assess_glyph_tail_adapter import f1
    h,rows=natural(run,m);require(load(run/'fit/status.json')['phase']=='completed','complete natural fit first')
    primitive=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy().astype(np.uint8)
    p=torch.load(ROOT/m['original_pool'],weights_only=True,map_location='cpu')
    copies=copy_pool(m)['copy_held'];routes=[]
    natural_train_forms={r['label'] for r in rows if r['split']=='train' and r['label']>=0}
    for split in ['train','held_context']:
        ix=[i for i,r in enumerate(rows) if r['split']==split]
        routes.append(('natural_'+split,h[ix],[rows[i]['reference_labels'] for i in ix],ix))
    routes += [('inherited_first',p['first'],[[int(i)] for i in p['first_labels']],None),
               ('inherited_later',p['later'],[[int(i)] for i in p['later_labels']],None),
               ('cached_copy',copies[0],[[int(i)] for i in copies[1]],None)]
    summary={};boards=[]
    with torch.no_grad():
        for version,path in [('parent',ROOT/m['initial_checkpoint']),('natural',run/'fit/checkpoint_latest.pt')]:
            model=bridge_model({},device,path)
            for name,states,targets,original_indices in routes:
                inks=[];nonstop=[]
                for start in range(0,len(states),128):
                    monitor.position=dict(model=version,route=name,completed=start,total=len(states));monitor.guard()
                    logits,_,stop=model.predict(states[start:start+128].to(device))
                    inks.append((logits>=0).to(torch.uint8)[:,0].cpu());nonstop.extend((stop<0).cpu().tolist())
                ink=torch.cat(inks).numpy();scores=[];exact=[]
                for actual,ys,emits in zip(ink,targets,nonstop):
                    gold=[primitive[y] for y in ys if y>=0]
                    scores.append(max((f1(actual,g) for g in gold),default=0.) if emits else 0.)
                    exact.append(emits and any(np.array_equal(actual,g) for g in gold))
                raw=monitor.path/(version+'_'+name+'.pt')
                atomic_tensor(raw,dict(ink=torch.from_numpy(ink),emits=torch.tensor(nonstop),
                    f1=torch.tensor(scores),exact=torch.tensor(exact),
                    original_indices=torch.tensor(original_indices if original_indices is not None else list(range(len(states))))))
                summary[version+'/'+name]=dict(cases=len(scores),emitted_first_mean_f1=float(np.mean(scores)),
                    fraction095=float(np.mean(np.array(scores)>=.95)),pixel_exact=sum(exact),
                    nonstop=sum(nonstop),unsupported_reference_cases=sum(not any(y>=0 for y in ys) for ys in targets),
                    raw_sha256=sha(raw))
                if name=='natural_held_context':
                    strata={};questions={};contexts={}
                    for j,index in enumerate(original_indices):
                        row=rows[index];label=row['label']
                        bucket='unsupported_primary' if label<0 else ('seen_natural_first' if label in natural_train_forms else 'absent_natural_first')
                        strata.setdefault(bucket,[]).append(scores[j])
                        questions.setdefault(row['id'],[]).append(scores[j])
                        contexts.setdefault(row['context_hash'],[]).append(scores[j])
                    summary[version+'/'+name].update(
                        primary_first_strata={k:dict(images=len(v),mean_f1=float(np.mean(v)),fraction095=float(np.mean(np.array(v)>=.95))) for k,v in strata.items()},
                        unique_questions=len(questions),unique_contexts=len(contexts),
                        question_macro_f1=float(np.mean([np.mean(v) for v in questions.values()])),
                        context_macro_f1=float(np.mean([np.mean(v) for v in contexts.values()])),
                        repeated_prompt_styles_not_independent=True,
                        absent_natural_first_does_not_mean_unseen_in_copy_or_inherited_training=True)
                canvas=Image.new('L',(600,580),255);draw=ImageDraw.Draw(canvas)
                draw.text((4,4),version+' / '+name+'; reference | actual',fill=0)
                for j,index in enumerate(np.linspace(0,len(ink)-1,32,dtype=int)):
                    x=j%4*149;y0=25+j//4*69;draw.text((x,y0),str(index)+(' STOP' if not nonstop[index] else ''),fill=0)
                    ys=[y for y in targets[index] if y>=0];gold=primitive[ys[0]] if ys else np.zeros((32,32),np.uint8)
                    for k,img in enumerate([gold,ink[index]]):canvas.paste(Image.fromarray(255*(1-img)),(x+35*k,y0+16))
                board=monitor.path/(version+'_'+name+'.png');canvas.save(board);boards.append(dict(file=board.name,sha256=sha(board)))
            del model
    s=summary;held=s['natural/natural_held_context'];copy=s['natural/cached_copy']
    inherited=(3176*s['natural/inherited_first']['emitted_first_mean_f1']+20086*s['natural/inherited_later']['emitted_first_mean_f1'])/23262
    passed=(held['emitted_first_mean_f1']>=.90 and held['fraction095']>=.8 and inherited>=.98
            and copy['emitted_first_mean_f1']>=.95 and copy['fraction095']>=.8)
    save_json(monitor.path/'report.json',dict(summary=summary,boards=boards,natural_first_gate_passed=bool(passed),
        inherited_training_mean_f1=inherited,checkpoint_sha256=sha(run/'fit/checkpoint_latest.pt'),
        all_held_unsupported_cases_remain_in_denominator=True,premature_stop_counts_as_failure=True,
        raw_predictions_retained_even_if_not_emitted=True,cached_copy_source_unadapted=True,
        language_accuracy_measured=False,**monitor.receipt()))
    print(json.dumps(dict(summary=summary,natural_first_gate_passed=bool(passed))),flush=True)


def execute(run,stage):
    import torch
    from scripts.train_semantic_glyph_bridge import bridge_gpu
    from scripts.train_semantic_glyph_language import numerics
    from scripts.distill_semantic_glyph_readout import Monitor
    m=verify(run);folder=run/stage;folder.mkdir(exist_ok=False);monitor=Monitor(folder,m);monitor.status('admitting')
    try:
        with bridge_gpu(m) as (device,admission):
            numerics();torch.manual_seed(m['seed']);torch.cuda.manual_seed_all(m['seed'])
            save_json(folder/'admission.json',admission);monitor.guard();globals()[stage](run,m,monitor,device)
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['prepare','fit','qualify'],required=True)
    p.add_argument('--run',type=Path,default=Path('artifacts/natural_first_image_writer_v1'));a=p.parse_args()
    if a.stage=='prepare':prepare(a.run.resolve())
    else:execute(a.run.resolve(),a.stage)
