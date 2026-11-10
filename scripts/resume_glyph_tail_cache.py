#!/usr/bin/env python3
"""Authenticate interrupted frozen-prefix files and compute only missing cases.

No optimizer state or training recipe changes. Original files, statuses and
logs remain as evidence; the completed cache manifest references their storage.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require, owner_live

OWN = 'scripts/resume_glyph_tail_cache.py'


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def identity():
    require(subprocess.check_output(['git','show','HEAD:'+OWN],cwd=ROOT)==(ROOT/OWN).read_bytes(),
            'commit recovery code first')
    return sha(ROOT/OWN)


def validate(path, job):
    import torch
    v=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    require(v['id']==job['id'] and v['lower'].dtype==torch.float32,'cached source identity/dtype differs')
    labels=torch.tensor(job['labels']+[-1]); n=len(job['labels']); boundary=v['boundary']
    require(torch.equal(v['labels'],labels) and boundary>0,'cached supervision differs')
    require(tuple(v['lower'].shape)==(1,boundary+n,4096)
            and tuple(v['position_ids'].shape)==(4,1,boundary+n)
            and tuple(v['original_final_predictors'].shape)==(1,n+1,4096),'cached sequence shape differs')
    require(bool(torch.isfinite(v['lower']).all()) and bool(torch.isfinite(v['original_final_predictors']).all()),
            'nonfinite frozen prefix')
    require(torch.equal(v['position_ids'][0,0],torch.arange(boundary+n)),'physical positions differ')
    offset=v['geometry']['rotary_offset']
    require(torch.equal(v['position_ids'][1:,0,boundary:],torch.arange(offset,offset+n)[None].expand(3,-1)),
            'glyph rotary positions differ')
    return dict(id=job['id'],file=rel(path),sha256=sha(path),states=n+1,cache_bytes=path.stat().st_size)


def prepare(run, recovery):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU recovery audit requires hidden CUDA')
    require(not recovery.exists(),'preserve prior recovery attempts')
    code=identity()
    from scripts.train_glyph_tail_adapter import verify
    m=verify(run)
    require(not (run/'cache/report.json').exists(),'cache already has a final manifest; inspect it instead')
    old=load(run/'cache/status.json'); pipeline=load(run/'pipeline.json')
    require(not owner_live(old),'old cache owner is still active')
    require(not owner_live(dict(pid=pipeline.get('supervisor_pid'),start_ticks=pipeline.get('supervisor_start_ticks'),
                               boot_id=pipeline.get('boot_id'))),'old coordinator is still active')
    import torch
    torch.set_num_threads(2)
    recovery.mkdir(parents=True); found={}
    preflight=load(run/'preflight_cache/report.json')
    require(load(run/'preflight_cache/status.json')['phase']=='completed','preflight must be complete')
    for index,row in enumerate(preflight['files']):
        require(sha(ROOT/row['file'])==row['sha256'],'sealed preflight changed')
        found[index]=validate(ROOT/row['file'],m['training_jobs'][index])
    paths=list((run/'cache').glob('case_*.pt'))
    paths.extend(p for folder in sorted(run.glob('cache_recovery_*')) if folder!=recovery
                 for p in (folder/'parts').glob('case_*.pt'))
    for position,path in enumerate(sorted(paths)):
        match=re.fullmatch(r'case_(\d{4})\.pt',path.name); require(match is not None,'unexpected cache filename')
        index=int(match[1]); require(index<len(m['training_jobs']),'cached case outside prepared training set')
        row=validate(path,m['training_jobs'][index])
        if index in found:
            a=torch.load(ROOT/found[index]['file'],weights_only=True,map_location='cpu',mmap=True)
            b=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
            require(all(torch.equal(a[k],b[k]) for k in ['lower','position_ids','labels','original_final_predictors'])
                    and a['geometry']==b['geometry'] and a['boundary']==b['boundary'],
                    'conflicting duplicate frozen-prefix records')
        else: found[index]=row
        if position%32==0:
            save_json(recovery/'audit_status.json',dict(phase='authenticating_saved_prefixes',completed=position,total=len(paths)))
    missing=[i for i in range(len(m['training_jobs'])) if i not in found]
    save_json(recovery/'preparation.json',dict(source_sha256=code,original_preparation_sha256=sha(run/'preparation.json'),
        original_cache_status=old,original_cache_status_sha256=sha(run/'cache/status.json'),
        original_preflight_report_sha256=sha(run/'preflight_cache/report.json'),
        original_pipeline=pipeline,original_pipeline_sha256=sha(run/'pipeline.json'),
        adopted={str(i):row for i,row in sorted(found.items())},missing_indices=missing,
        original_last_recorded_elapsed_seconds=old['elapsed_seconds'],
        original_exact_termination_time_unknown=True,original_pressure_wait_total_unknown=True,
        neural_training_recipe_changed=False,optimizer_updates_lost=0,
        recovered_files_hashed_after_interruption=True,partial_writes_ignored_and_retained=True))
    save_json(recovery/'audit_status.json',dict(phase='completed',adopted=len(found),missing=len(missing)))
    print(json.dumps(dict(adopted=len(found),missing=len(missing),optimizer_updates_lost=0)),flush=True)


def run_recovery(run, recovery):
    code=identity(); receipt=load(recovery/'preparation.json')
    require(code==receipt['source_sha256'] and sha(run/'preparation.json')==receipt['original_preparation_sha256'],
            'recovery preparation changed')
    require(sha(run/'preflight_cache/report.json')==receipt['original_preflight_report_sha256'],
            'preflight receipt changed')
    preflight=load(run/'preflight_cache/report.json')
    from scripts.train_glyph_tail_adapter import verify
    m=verify(run)
    require(not (run/'cache/report.json').exists(),'completed cache must not be overwritten')
    for row in receipt['adopted'].values(): require(sha(ROOT/row['file'])==row['sha256'],'adopted prefix changed')
    import torch
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from scripts.train_semantic_glyph_language import reader,bridge_model,image,numerics,Monitor,atomic_tensor
    from scripts.score_pixel_language_first_blocks import diagnostic_gpu
    folder=recovery/'parts'; folder.mkdir(exist_ok=False)
    monitor=Monitor(recovery,m['cache_seconds']); monitor.status('admitting')
    try:
        files={int(i):row for i,row in receipt['adopted'].items()}
        with diagnostic_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission; numerics(); monitor.guard()
            interface,provenance=reader(device); bridge=bridge_model({},device,ROOT/m['bridge'])
            primitive=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')
            glyphs=(1-2*primitive['ink'].float()).to(device)
            embedded=torch.cat([bridge.feedback_embedding(glyphs[i:i+128]) for i in range(0,len(glyphs),128)])
            capture={}
            require(len(interface.actor.context.layers)==36,'source body identity differs')

            def hook(module,args,kwargs): capture['lower']=args[0].detach().cpu().clone()

            handle=interface.actor.context.layers[32].register_forward_pre_hook(hook,with_kwargs=True)

            def forward(job):
                args,geometry=interface.assemble(native_budget_image(image(ROOT/job['prompt'])))
                boundary=args['inputs_embeds'].shape[1]; n=len(job['labels'])
                physical=torch.arange(boundary,boundary+n,device=device)
                rotary=torch.arange(geometry['rotary_offset'],geometry['rotary_offset']+n,device=device)
                positions=torch.stack((physical,rotary,rotary,rotary))[:,None]
                args['inputs_embeds']=torch.cat((args['inputs_embeds'],embedded[job['labels']][None]),1)
                args['position_ids']=torch.cat((args['position_ids'],positions),-1)
                args['visual_pos_masks']=torch.cat((args['visual_pos_masks'],torch.zeros((1,n),dtype=torch.bool,device=device)),1)
                response=interface.actor.context(**args,use_cache=False)
                return dict(lower=capture.pop('lower'),position_ids=args['position_ids'].cpu(),boundary=boundary,
                    labels=torch.tensor(job['labels']+[-1]),original_final_predictors=response.last_hidden_state[:,boundary-1:].float().cpu().clone(),
                    id=job['id'],geometry=geometry)

            # A cold replay checks the resumed neural calculation against an
            # authenticated original case before any new prefix is accepted.
            check=forward(m['training_jobs'][0])
            old=torch.load(ROOT/files[0]['file'],weights_only=True,map_location='cpu',mmap=True)
            error=max(float((check[k]-old[k]).abs().max()) for k in ['lower','original_final_predictors'])
            require(error<m['preflight_state_tolerance'],'recovered calculation differs from original prefix')
            save_json(recovery/'initial_replay.json',dict(maximum_absolute_error=error,
                original_preflight_report_sha256=receipt['original_preflight_report_sha256'],
                adopted_case=files[0],source_sha256=code,neural_recipe_changed=False))
            del check,old
            for position,index in enumerate(receipt['missing_indices']):
                job=m['training_jobs'][index]
                monitor.position=dict(completed=len(files),total=len(m['training_jobs']),id=job['id'],index=index)
                monitor.guard(); value=forward(job)
                path=folder/f'case_{index:04}.pt'; atomic_tensor(path,value); del value
                files[index]=validate(path,job)
                monitor.status('recovering_only_missing_frozen_prefixes')
                if position%16==0: print(json.dumps(monitor.position),flush=True)
            handle.remove()
            require(len(files)==len(m['training_jobs']),'incomplete recovered cache')
            final=dict(preparation_sha256=sha(run/'preparation.json'),files=[files[i] for i in range(len(files))],
                tail_start_layer=32,zero_tail_maximum_absolute_error=preflight['zero_tail_maximum_absolute_error'],
                future_glyph_prior_state_maximum_error=preflight['future_glyph_prior_state_maximum_error'],
                original_preflight_report_sha256=sha(run/'preflight_cache/report.json'),
                recovery_source_sha256=code,recovery_preparation_sha256=sha(recovery/'preparation.json'),
                adopted_prefix_cases=len(receipt['adopted']),new_prefix_cases=len(receipt['missing_indices']),
                cold_replay_maximum_absolute_error=error,original_last_recorded_elapsed_seconds=receipt['original_last_recorded_elapsed_seconds'],
                complete_cross_attempt_wall_time_not_measured=True,complete_cross_attempt_pressure_wait_not_measured=True,
                provenance=provenance,source_vocabulary_parameters=0,lower_body_frozen=True,cache_FP32=True,**monitor.receipt())
            save_json(recovery/'report.json',final)
        monitor.status('completed')
        # Preserve the stale original status inside the immutable recovery
        # receipt, then publish the complete aggregate for the unchanged fit.
        save_json(run/'cache/report.json',final)
        save_json(run/'cache/status.json',dict(load(recovery/'status.json'),phase='completed',pid=None,
            recovery=rel(recovery),original_status_sha256=receipt['original_cache_status_sha256']))
    except BaseException as error:
        monitor.status('failed',reason=repr(error)); raise
    finally:
        save_json(recovery/'status.json',dict(load(recovery/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','run'],required=True)
    p.add_argument('--run',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--recovery-name',default='cache_recovery_01')
    a=p.parse_args(); require(re.fullmatch(r'cache_recovery_\d{2}',a.recovery_name) is not None,'invalid recovery name')
    root=a.run.resolve(); {'prepare':prepare,'run':run_recovery}[a.stage](root,root/a.recovery_name)
