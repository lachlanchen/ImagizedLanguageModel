#!/usr/bin/env python3
"""CPU recount of actual source exposure, cached pixels and final optimizer state."""
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.joint_writer_training import ROOT,PARENT,RUN,PUBLIC,load,sha,require,CachedSequences,previous_seconds
from ilm.visual_lm.glyph_joint_density import unpack_glyphs
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from ilm.visual_lm.continuation_training import canonical_digest
from scripts.prepare_joint_continuation_writer import validate
from scripts.train_visual_continuation import atomic_json


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU audit only')
    torch.set_num_threads(2);r=validate();began=time.perf_counter();minimum=float('inf')
    remaining=r['audit_seconds']-previous_seconds('audit');require(remaining>0,'Cumulative audit budget exhausted')
    folder=RUN/'audit';folder.mkdir(exist_ok=False)
    def guard():
        nonlocal minimum
        mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        minimum=min(minimum,mem['MemAvailable']/1024**2)
        require(minimum>=24 and time.perf_counter()-began<=remaining,'CPU audit resource ceiling reached')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<=8,'CPU audit RSS ceiling reached')
    fit=load(RUN/'fit/report.json');cache=load(RUN/'cache/report.json')
    require(fit['complete'] and cache['complete'],'Complete cache and fit required')
    require(fit['cache_report_sha256']==sha(RUN/'cache/report.json'),'Fit/cache binding differs')
    rows=load(PARENT/'training_manifest.json')['rows'];schedule=np.load(RUN/'schedule.npy',allow_pickle=False)
    events={};executed=0
    for ledger in sorted((RUN/'fit').glob('ledger_*.jsonl')):
        for line in ledger.read_text().splitlines():
            value=json.loads(line);digest=value.pop('event_sha256')
            require(canonical_digest(value)==digest,'Training event hash differs')
            require(digest not in events or events[digest]==value,'Event identity collision')
            events[digest]=value;executed+=1
    chain=[];pointer=fit['ledger_final_sha256'];seen=set()
    while pointer!='0'*64:
        require(pointer in events and pointer not in seen,'Broken or cyclic training chain')
        seen.add(pointer);value=events[pointer];chain.append(value);pointer=value['previous_event_sha256']
    retained=[x for x in reversed(chain) if x['event_kind']=='scheduled']
    require(len(retained)==41298 and [x['step'] for x in retained]==list(range(1,41299)),
            'Complete retained optimizer schedule required')
    exposure=0
    for event,(index,layout) in zip(retained,schedule):
        row=rows[int(index)];key='fixed_cells' if layout==0 else 'proportional'
        require(event['source_index']==int(index) and event['layout']==key and
                event['source_id']==row['id'] and event['glyphs']==row['target_glyphs'],'Source exposure differs')
        exposure+=event['glyphs']
    require(exposure==6108741,'Complete retained glyph exposure required')
    data=CachedSequences();first=np.zeros(4096,np.float64);second=first.copy();weight=0;positions=0
    cached_identities=set()
    for item in cache['records']:
        guard();states,patterns,record=data.get(ROOT/item['directory']);identity=record['identity']
        index=identity['source_index'];row=rows[index]
        require((index,identity['layout']) not in cached_identities,'Duplicate cached source/layout')
        cached_identities.add((index,identity['layout']))
        require(identity['split']=='train' and identity['source_id']==row['id'] and
                len(patterns)==row['target_glyphs'],'Training cache source differs')
        require(tensor_hash(unpack_glyphs(patterns.long()))==row['target_sha256'],
                'Cached physical patterns do not reproduce complete original targets')
        x=states[:-1].numpy().astype(np.float64);w=1 if identity['layout']=='fixed_cells' else 2
        first+=w*np.mean(x,axis=0);second+=w*np.mean(x*x,axis=0);weight+=w;positions+=len(patterns)
    require(len(cache['records'])==27532 and positions==4072494 and weight==41298,'Both-layout cache coverage differs')
    mean=first/weight;scale=np.maximum(np.sqrt(np.maximum(second/weight-mean*mean,0)),1e-4)
    norm=torch.load(RUN/'cache/normalization.pt',weights_only=True,map_location='cpu')
    mean_error=float(np.abs(mean-norm['mean'].numpy()).max());scale_error=float(np.abs(scale-norm['scale'].numpy()).max())
    require(mean_error<=2e-6 and scale_error<=2e-6,'Training-only normalization recount differs')
    snapshots=load(RUN/'fit/snapshots.json');checks=[]
    initial=torch.load(RUN/'initial_parameters.pt',weights_only=True,map_location='cpu')
    require(sorted(map(int,snapshots))==[0,10324,20649,30973,41298],'All fixed checkpoints required')
    for step,snapshot in snapshots.items():
        guard();path=ROOT/snapshot['file'];require(sha(path)==snapshot['sha256'],'Fixed snapshot changed')
        value=torch.load(path,weights_only=True,map_location='cpu')
        require(value['step']==int(step) and value['registration_sha256']==sha(PUBLIC), 'Snapshot binding differs')
        require(all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in value['model'].values()),'FP32 finite head required')
        parameters=sum(v.numel() for k,v in value['model'].items() if k not in ['condition_mean','condition_scale'])
        require(parameters==2642368,'Wrong complete head parameter count')
        if int(step)==0:
            require(all(torch.equal(value['model'][k],v) for k,v in initial.items()),'Initial parameters differ from fresh registered draw')
        if int(step)>0:
            states=value['optimizer']['state'];require(len(states)==len(value['model'])-2,'Every actual parameter needs Adam state')
            require({int(s['step'].item()) for s in states.values()}=={int(step)},'Actual Adam counters differ')
            require(all(s[k].dtype==torch.float32 and torch.isfinite(s[k]).all() for s in states.values()
                        for k in ['exp_avg','exp_avg_sq']),'Finite FP32 optimizer moments required')
        checks.append(dict(step=int(step),parameters=parameters,sha256=snapshot['sha256']))
    resume=load(RUN/'fit/resume_control.json');require(resume['passed'] and resume['exact_model_optimizer_rng'],'Actual resume control required')
    atomic_json(folder/'report.json',dict(complete=True,registration_sha256=sha(PUBLIC),fit_report_sha256=sha(RUN/'fit/report.json'),
        retained_updates=len(retained),executed_events_all_attempts=executed,retained_target_exposure=exposure,
        both_layout_target_states=positions,source_target_inverse_packing_verified=True,
        mean_recount_maximum_error=mean_error,scale_recount_maximum_error=scale_error,snapshots=checks,
        reader_freezing_is_bound_execution_assertion_not_second_neural_replay=True,
        new_model_calls=0,new_optimizer_updates=0,elapsed_seconds=time.perf_counter()-began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,minimum_available_gib=minimum,
        language_quality_unmeasured=True,goal_achieved=False))
    print(json.dumps(dict(audit_complete=True,retained_updates=len(retained),target_exposure=exposure)),flush=True)


if __name__=='__main__':main()
