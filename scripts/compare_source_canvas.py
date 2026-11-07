#!/usr/bin/env python3
"""Registered matched source-feature versus summary-only writing diagnostic."""
import argparse
from collections import Counter
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.visual_lm.source_canvas import SourceCanvasWriter
from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec, CODEC_SHA
from ilm.visual_lm.semantic_writing_data import ExactEpochSampler
from scripts.cache_semantic_images_v72 import ROOT, CODEC, sha, atomic_json, committed_identity
from scripts.run_parallel_canvas import scores, tree

TRAIN = ROOT/'artifacts/contextual_pixel_learning_v1_attempt02/prepared'
FEATURES = ROOT/'artifacts/contextual_pixel_adaptation_v1/features'
DEV = ROOT/'artifacts/contextual_pixel_spans_v1'
SEED = 2026110802
UPDATES = 3257
SOURCES = ('scripts/compare_source_canvas.py', 'ilm/visual_lm/source_canvas.py',
    'ilm/visual_lm/parallel_canvas.py', 'ilm/visual_lm/semantic_writing_data.py',
    'ilm/visual_lm/codec_block_writer.py', 'ilm/visual_lm/continuous_glyph_codec.py',
    'scripts/run_parallel_canvas.py', 'scripts/cache_semantic_images_v72.py',
    'scripts/train_clean_corpus_v75.py', 'references/source_canvas_protocol.md')


def read(p):
    return json.loads(Path(p).read_text())


def status(run, stage, **fields):
    value = dict(stage=stage, pid=os.getpid(), **fields)
    atomic_json(run/'status.json', value)
    print(json.dumps(value), flush=True)


def checked_image(path, digest):
    if sha(path) != digest:
        raise ValueError('physical image changed: '+str(path))
    a = np.asarray(Image.open(path).convert('L')).copy()
    if a.ndim != 2 or a.shape[0] != 32 or not np.isin(a,(0,255)).all():
        raise ValueError('complete binary 32-high source/reference image required')
    return a


def prepare(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('CPU-only preparation required')
    torch.set_num_threads(4)
    if run.exists() and any(run.iterdir()):
        raise ValueError('preserve existing run; choose a fresh directory')
    run.mkdir(parents=True); (run/'codes').mkdir()
    identity = dict(**committed_identity(SOURCES), seed=SEED, updates=UPDATES,
        purpose='source_canvas_comparison', codec_sha256=CODEC_SHA,
        inputs_sha256=sha(TRAIN/'inputs.json'), references_sha256=sha(TRAIN/'references.json'),
        features_sha256=sha(FEATURES/'report.json'), dev_inputs_sha256=sha(DEV/'prepared/inputs.json'),
        dev_references_sha256=sha(DEV/'prepared/references.json'), dev_features_sha256=sha(DEV/'features/report.json'))
    train = read(TRAIN/'inputs.json'); refs = {r['id']:r for r in read(TRAIN/'references.json')}
    features_report = read(FEATURES/'report.json')
    if not features_report['complete'] or features_report['total'] != 3257:
        raise ValueError('complete original feature inventory required')
    feature_index = {r['id']:r for r in features_report['records']}
    if len(train) != 3257 or len(feature_index) != 3257:
        raise ValueError('all 3257 admitted examples required')
    codec = load_qualified_writing_codec(CODEC,'cpu')
    white = codec.encode(torch.ones(1,1,32,32))[0].detach()
    with torch.no_grad():
        if not bool((codec.decode(white[None])>=0).all()):
            raise ValueError('white padding does not reconstruct exactly')
    torch.save(white,run/'white.pt'); files={'white.pt':sha(run/'white.pt')}
    controls=[]; rows=[]; max_units=0
    for i,row in enumerate(train):
        f = feature_index[row['id']]; fp=ROOT/f['file']
        if sha(fp) != f['sha256'] or f['input_sha256'] != row['input_sha256']:
            raise ValueError('feature input identity changed')
        image=checked_image(ROOT/row['input'],row['input_sha256'])
        value=torch.load(fp,map_location='cpu',weights_only=True)
        if value['id']!=row['id'] or value['input_sha256']!=row['input_sha256']:
            raise ValueError('feature payload binding changed')
        validate_feature(value); max_units=max(max_units,len(value['visual']))
        targets=refs[row['id']]['targets']
        if len(targets)!=1: raise ValueError('registered single-reference training inventory changed')
        target=checked_image(ROOT/targets[0]['file'],targets[0]['sha256'])
        tiles=(target.shape[1]+31)//32
        if not 1<=tiles<=64: raise ValueError('complete reference exceeds fixed capacity')
        padded=np.pad(target,((0,0),(0,tiles*32-target.shape[1])),constant_values=255)
        pixels=torch.from_numpy(padded).float().div(255).reshape(32,tiles,32).permute(1,0,2)[:,None]
        with torch.no_grad(): codes=codec.encode(pixels).cpu()
        p=run/'codes'/f'{i:04d}.pt';torch.save(codes,p);files[str(p.relative_to(run))]=sha(p)
        # Decode every active tile; the separately checked white coordinate
        # makes the remaining identical exterior tiles exactly white.
        with torch.no_grad(): active=(codec.decode(codes)>=0).to(torch.uint8).mul(255).reshape(tiles,32,32).permute(1,0,2).reshape(32,tiles*32).numpy()
        a=np.pad(active,((0,0),(0,2048-active.shape[1])),constant_values=255)
        controls.append(dict(id=row['id'],**scores(a,[target])))
        rows.append(dict(**row,feature=str(fp.relative_to(ROOT)),feature_sha256=f['sha256'],
            codes=str(p.relative_to(run)),tiles=tiles,targets=targets))
        if (i+1)%256==0: status(run,'preparing',completed=i+1,total=3257)
    dev_inputs=read(DEV/'prepared/inputs.json');dev_refs={r['id']:r for r in read(DEV/'prepared/references.json')}
    dev_features={r['id']:r for r in read(DEV/'features/report.json')['records']}; assessment=[]
    for original in dev_inputs:
        if original['split']!='development':continue
        row=dict(original);row['input']=str((DEV/'prepared'/row['input']).relative_to(ROOT))
        checked_image(ROOT/row['input'],row['input_sha256'])
        f=dev_features[row['id']];fp=DEV/'features'/f['file']
        if sha(fp)!=f['sha256']:raise ValueError('development feature changed')
        value=torch.load(fp,map_location='cpu',weights_only=True);validate_feature(value)
        if value['id']!=row['id'] or value['input_sha256']!=row['input_sha256']:
            raise ValueError('development feature/input mismatch')
        targets=[]
        for t in dev_refs[row['id']]['targets']:
            p=DEV/'prepared'/t['file'];checked_image(p,t['sha256'])
            targets.append(dict(file=str(p.relative_to(ROOT)),sha256=t['sha256']))
        assessment.append(dict(**row,feature=str(fp.relative_to(ROOT)),feature_sha256=f['sha256'],targets=targets))
    counts=Counter((r['task'],r.get('condition','normal')) for r in assessment)
    if counts!=Counter({('qa','normal'):96,('qa','question_erased'):96,('copy','normal'):32}):
        raise ValueError('all original development cases required: '+str(counts))
    atomic_json(run/'identity.json',identity)
    atomic_json(run/'prepared.json',dict(identity_sha256=sha(run/'identity.json'),training=rows,
        assessment=assessment,files_sha256=files,maximum_source_units=max_units,
        supplied_code_controls=controls,supplied_code_exact=sum(x['exact'] for x in controls),
        supplied_code_f1=float(np.mean([x['foreground_f1'] for x in controls])),
        previous_exclusions=read(TRAIN/'report.json')['rejected'],actual_source_model_calls=0))
    status(run,'prepared',training=len(rows),development=len(assessment),maximum_source_units=max_units)


def validate_feature(value):
    v,m,g=value['visual'],value['memory'],value['geometry']
    if (v.ndim!=2 or v.shape[1]!=4096 or m.shape!=(52,4096) or g.shape!=(len(v),4)
            or not len(v) or v.dtype!=torch.bfloat16 or m.dtype!=torch.bfloat16
            or not all(torch.isfinite(x).all() for x in (v,m,g))):
        raise ValueError('invalid complete frozen feature payload')


def prepared(run):
    identity=read(run/'identity.json');prep=read(run/'prepared.json')
    if sha(run/'identity.json')!=prep['identity_sha256']:raise ValueError('identity changed')
    for name,digest in identity['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('producer changed: '+name)
    for name,digest in prep['files_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('training coordinate changed: '+name)
    return identity,prep


def load_feature(row):
    p=ROOT/row['feature']
    if sha(p)!=row['feature_sha256']:raise ValueError('source feature changed')
    value=torch.load(p,map_location='cpu',weights_only=True);validate_feature(value)
    if value['id']!=row['id'] or value['input_sha256']!=row['input_sha256']:
        raise ValueError('source input identity changed')
    return {k:value[k] for k in ('memory','visual','geometry')}


def pack(values,device):
    n=len(values);width=max(len(x['visual']) for x in values)
    visual=torch.zeros(n,width,4096,dtype=torch.bfloat16)
    geometry=torch.zeros(n,width,4)
    for i,x in enumerate(values):
        visual[i,:len(x['visual'])]=x['visual'];geometry[i,:len(x['visual'])]=x['geometry']
    return (torch.stack([x['memory'] for x in values]).to(device),visual.to(device),geometry.to(device),
        torch.tensor([len(x['visual']) for x in values],device=device))


def initialize(white,device):
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=SourceCanvasWriter(white).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    return model,optimizer,ExactEpochSampler(3257,SEED+1)


def update(model,optimizer,sampler,data,step,source,*,check=False):
    ids=sampler.take(4);batch=[data[i] for i in ids];inputs=pack(batch,model.device)
    target=model.canvas.white_code.expand(4,64,768).clone()
    lengths=torch.tensor([len(x['codes']) for x in batch],device=model.device)
    for i,x in enumerate(batch):target[i,:len(x['codes'])]=x['codes'].to(model.device)
    factor=(step+1)/100 if step<100 else .1+.9*.5*(1+math.cos(math.pi*(step-100)/(UPDATES-100)))
    optimizer.param_groups[0]['lr']=3e-4*factor
    optimizer.zero_grad(set_to_none=True);loss=model.loss(inputs,target,lengths,source=source)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite writing loss')
    loss.backward();checks={}
    if check:
        groups=['canvas.memory.','canvas.queries','canvas.layers.','canvas.output.','types']
        if source:groups.append('geometry.')
        for group in groups:
            gradients=[p.grad for n,p in model.named_parameters() if n.startswith(group)]
            checks[group]=bool(gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
                and any(bool(g.abs().max()>0) for g in gradients))
        if not all(checks.values()):raise ValueError('gradient check failed: '+str(checks))
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    return dict(update=step+1,indices=ids,loss=float(loss.detach()),gradient_norm=float(norm),gradient_checks=checks)


def snapshot(model,optimizer,sampler,step,identity,elapsed):
    return dict(model=tree(model.state_dict()),optimizer=tree(optimizer.state_dict()),sampler=tree(sampler.state_dict()),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),update=step,identity=identity,elapsed_seconds=elapsed)


def restore(saved,model,optimizer,sampler):
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    sampler.load_state_dict(saved['sampler']);torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])


def preflight(run,white,data,device):
    if (run/'preflight.json').exists():
        if not read(run/'preflight.json')['passed']:raise ValueError('failed preflight')
        return
    results={}
    for arm in ('summary','source'):
        model,opt,sampler=initialize(white,device);source=arm=='source'
        for step in range(3):result=update(model,opt,sampler,data,step,source,check=step==2)
        saved=snapshot(model,opt,sampler,3,{},0.);expected=update(model,opt,sampler,data,3,source)
        tensors=tree(model.state_dict());restore(saved,model,opt,sampler)
        actual=update(model,opt,sampler,data,3,source)
        equal=actual==expected and all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in tensors.items())
        if not equal:raise AssertionError('optimizer/sampler exact resume failed')
        model.eval();inputs=pack([data[0]],device)
        with torch.no_grad():
            z=model(*inputs,source=source)
            if not source:
                altered=(inputs[0],inputs[1]*0+7,inputs[2]*0-3,inputs[3])
                if not torch.equal(z,model(*altered,source=False)):raise ValueError('summary arm used source descriptors')
        largest=sorted(range(len(data)),key=lambda i:len(data[i]['visual']),reverse=True)[:4]
        batch=[data[i] for i in largest];inputs=pack(batch,device)
        target=white.to(device).expand(4,64,768).clone()
        lengths=torch.tensor([len(x['codes']) for x in batch],device=device)
        for i,x in enumerate(batch):target[i,:len(x['codes'])]=x['codes'].to(device)
        model.train();opt.zero_grad(set_to_none=True)
        resource_loss=model.loss(inputs,target,lengths,source=source);resource_loss.backward()
        if not torch.isfinite(resource_loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('largest physical source batch has invalid backward')
        results[arm]=dict(gradients=result['gradient_checks'],exact_resume=equal,optimizer_invocations=5,distinct_positions=4,
            resource_backward_without_optimizer=True,largest_batch_indices=largest,
            largest_source_units=[len(x['visual']) for x in batch])
        del inputs,target,resource_loss,z
        del model,opt,saved,tensors;gc.collect();torch.cuda.empty_cache()
    atomic_json(run/'preflight.json',dict(passed=True,arms=results,reset_before_production=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3))


def save_checkpoint(folder,model,opt,sampler,step,identity,elapsed):
    p=folder/'checkpoint.pt';temp=folder/'checkpoint.partial'
    torch.save(snapshot(model,opt,sampler,step,identity,elapsed),temp)
    if p.exists():p.replace(folder/'checkpoint_previous.pt')
    temp.replace(p)


def train_arm(run,arm,white,data,identity,device):
    folder=run/arm;folder.mkdir(exist_ok=True)
    if (folder/'training.json').exists():
        report=read(folder/'training.json')
        if sha(folder/'checkpoint.pt')!=report['checkpoint_sha256']:raise ValueError('completed checkpoint changed')
        return
    model,opt,sampler=initialize(white,device);step=0;elapsed=0.
    if (folder/'checkpoint.pt').exists():
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise ValueError('resumed experiment identity differs')
        restore(saved,model,opt,sampler);step=saved['update'];elapsed=saved['elapsed_seconds'];del saved
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    journal=folder/f'updates_{step:05d}_{time.time_ns()}.jsonl'
    with journal.open('w') as log:
        while step<UPDATES:
            result=update(model,opt,sampler,data,step,arm=='source');step+=1
            log.write(json.dumps(result)+'\n')
            if step%50==0 or step==UPDATES:
                log.flush();status(run,'training',arm=arm,update=step,total=UPDATES,loss=result['loss'],
                    elapsed_seconds=elapsed+time.monotonic()-started)
            if step%250==0 or step==UPDATES:
                log.flush();save_checkpoint(folder,model,opt,sampler,step,identity,elapsed+time.monotonic()-started)
    if sampler.draws!=13028 or sampler.epoch!=3 or sampler.cursor!=3257:
        raise AssertionError('all four full data passes required')
    atomic_json(folder/'training.json',dict(arm=arm,identity=identity,updates=step,draws=sampler.draws,
        elapsed_seconds=elapsed+time.monotonic()-started,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        optimized_parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(folder/'checkpoint.pt')))
    del model,opt;gc.collect();torch.cuda.empty_cache()


@torch.no_grad()
def assess(run,arm,white,prep,identity,device):
    folder=run/arm/'outputs';folder.mkdir(exist_ok=True)
    cp=run/arm/'checkpoint.pt';training=read(run/arm/'training.json')
    if sha(cp)!=training['checkpoint_sha256']:raise ValueError('final checkpoint changed')
    model,_,_=initialize(white,device);saved=torch.load(cp,map_location='cpu',weights_only=True)
    if saved['identity']!=identity or saved['update']!=UPDATES:raise ValueError('complete final model required')
    model.load_state_dict(saved['model']);model.eval();del saved
    codec=load_qualified_writing_codec(CODEC,device)
    chosen=[dict(row,cohort='development') for row in prep['assessment']]
    chosen += [dict(row,cohort='training') for row in prep['training'][:32]]
    outputs=[];started=time.monotonic()
    for i,row in enumerate(chosen):
        p=folder/f'{i:04d}.png';zfile=p.with_suffix('.pt')
        feature=load_feature(row);images,z=model.generate(pack([feature],device),codec,source=arm=='source')
        a=images[0].numpy();Image.fromarray(a).save(p);torch.save(z[0],zfile)
        outputs.append(dict(id=row['id'],cohort=row['cohort'],task=row['task'],condition=row.get('condition','normal'),
            pair=row['pair'],output=str(p.relative_to(run)),output_sha256=sha(p),coordinates=str(zfile.relative_to(run)),
            coordinates_sha256=sha(zfile),input=row['input'],input_sha256=row['input_sha256']))
        if (i+1)%32==0:status(run,'cached_assessment',arm=arm,calls=i+1,total=len(chosen))
    atomic_json(run/arm/'assessment.json',dict(outputs=outputs,calls=len(outputs),elapsed_seconds=time.monotonic()-started,
        opens_reference_files=False,cached_features=True,native_image_replay=False))
    del model,codec;gc.collect();torch.cuda.empty_cache()


def run_all(run):
    from scripts.train_clean_corpus_v75 import setup
    identity,prep=prepared(run)
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);device=setup()
        torch.backends.mha.set_fastpath_enabled(False)
        status(run,'loading_authenticated_features',total=len(prep['training']))
        data=[]
        for row in prep['training']:
            value=load_feature(row)
            value['codes']=torch.load(run/row['codes'],map_location='cpu',weights_only=True)
            data.append(value)
        white=torch.load(run/'white.pt',map_location='cpu',weights_only=True)
        preflight(run,white,data,device)
        for arm in ('summary','source'):
            train_arm(run,arm,white,data,identity,device)
            assess(run,arm,white,prep,identity,device)
        status(run,'training_and_cached_assessment_complete',native_image_path_qualified=False)


def verify(run):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('CPU-only verification required')
    torch.set_num_threads(4);identity,prep=prepared(run);codec=load_qualified_writing_codec(CODEC,'cpu')
    refrows={r['id']:r for r in prep['assessment']+prep['training']};all_results={}
    for arm in ('summary','source'):
        training=read(run/arm/'training.json');saved=torch.load(run/arm/'checkpoint.pt',map_location='cpu',weights_only=True)
        if training['checkpoint_sha256']!=sha(run/arm/'checkpoint.pt') or saved['update']!=UPDATES or saved['identity']!=identity:
            raise ValueError('completed training identity changed')
        moments=saved['optimizer']['state']
        if not moments or any(int(s['step'])!=UPDATES or not torch.isfinite(s['exp_avg']).all()
            or not torch.isfinite(s['exp_avg_sq']).all() for s in moments.values()):raise ValueError('optimizer completion invalid')
        if arm=='source' and len(moments)!=len(saved['optimizer']['param_groups'][0]['params']):raise ValueError('missing source-branch optimizer state')
        logs=[]
        for p in sorted((run/arm).glob('updates_*.jsonl')):logs.extend(json.loads(s) for s in p.read_text().splitlines())
        if [r['update'] for r in logs]!=list(range(1,UPDATES+1)):
            raise ValueError('replayed or missing production updates; preserve trace and audit before qualification')
        sampler=ExactEpochSampler(3257,SEED+1)
        for r in logs:
            if r['indices']!=sampler.take(4):raise ValueError('sampler draw differs')
        counts=Counter(i for r in logs for i in r['indices'])
        if len(counts)!=3257 or set(counts.values())!={4}:raise ValueError('incomplete data exposure')
        result=[]
        for row in read(run/arm/'assessment.json')['outputs']:
            p=run/row['output'];zp=run/row['coordinates']
            if sha(p)!=row['output_sha256'] or sha(zp)!=row['coordinates_sha256']:raise ValueError('raw output changed')
            z=torch.load(zp,map_location='cpu',weights_only=True)
            with torch.no_grad():a=(codec.decode(z)>=0).to(torch.uint8).mul(255).reshape(64,32,32).permute(1,0,2).reshape(32,2048).numpy()
            if not np.array_equal(a,np.asarray(Image.open(p))):raise ValueError('coordinate/PNG mismatch')
            refs=[checked_image(ROOT/t['file'],t['sha256']) for t in refrows[row['id']]['targets']]
            measured=scores(a,refs)
            # Independent binary-array check of the reference scorer.
            details=[]
            for b in refs:
                width=max(a.shape[1],b.shape[1]);ap=np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=255)<128
                bp=np.pad(b,((0,0),(0,width-b.shape[1])),constant_values=255)<128
                denominator=int(ap.sum()+bp.sum());f1=2*int((ap&bp).sum())/denominator if denominator else 1.
                details.append((bool(np.array_equal(ap,bp)),f1))
            if measured['exact']!=any(x[0] for x in details) or abs(measured['foreground_f1']-max(x[1] for x in details))>1e-12:
                raise ValueError('independent pixel score differs')
            result.append(dict(**row,**measured))
        groups={}
        for cohort,task,condition in [('training','qa','normal'),('development','qa','normal'),
            ('development','qa','question_erased'),('development','copy','normal')]:
            selected=[r for r in result if (r['cohort'],r['task'],r['condition'])==(cohort,task,condition)]
            groups[f'{cohort}/{task}/{condition}']=dict(cases=len(selected),exact=sum(r['exact'] for r in selected),
                foreground_f1=float(np.mean([r['foreground_f1'] for r in selected])))
        pairs={}
        for row in result:
            if row['cohort']=='development' and row['task']=='qa' and row['condition']=='normal':pairs.setdefault(row['pair'],[]).append(row['exact'])
        if len(pairs)!=48 or any(len(x)!=2 for x in pairs.values()):raise ValueError('complete question pairs required')
        all_results[arm]=dict(groups=groups,both_correct_pairs=sum(all(x) for x in pairs.values()),
            question_erasure_f1_gain=groups['development/qa/normal']['foreground_f1']-groups['development/qa/question_erased']['foreground_f1'],
            training=training,rows=result,all_coordinates_replayed=True,all_draws_verified=True)
        del saved
    s=all_results['source'];c=all_results['summary'];g=s['groups']
    gates=dict(copy_exact=g['development/copy/normal']['exact']>=24,qa_exact=g['development/qa/normal']['exact']>=32,
        pairs=s['both_correct_pairs']>=17,question_gain=s['question_erasure_f1_gain']>=.10,
        copy_f1_lead=g['development/copy/normal']['foreground_f1']-c['groups']['development/copy/normal']['foreground_f1']>=.10,
        qa_f1_lead=g['development/qa/normal']['foreground_f1']-c['groups']['development/qa/normal']['foreground_f1']>=.10)
    receipt=dict(identity=identity,arms=all_results,gates=gates,screen_pass=all(gates.values()),
        supplied_code_exact=prep['supplied_code_exact'],supplied_code_f1=prep['supplied_code_f1'],
        additional_preflight_optimizer_invocations=10,cached_feature_diagnostic=True,
        native_image_path_qualified=False,external_pretraining=True,general_ilm_goal_complete=False)
    atomic_json(run/'verification.json',receipt)
    status(run,'verified',groups={arm:r['groups'] for arm,r in all_results.items()},gates=gates,screen_pass=receipt['screen_pass'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run','verify'),required=True)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/source_canvas_v1');args=p.parse_args()
    {'prepare':prepare,'run':run_all,'verify':verify}[args.stage](args.run.resolve())
