#!/usr/bin/env python3
"""Train complete visual continuations; evaluate fixed snapshots without references."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,sha
from ilm.visual_lm.continuation_training import (RUN,SEED,load,rel,build_actor,TrainingImages,
    TrainingMonitor,update,trainable_state,restore_trainable,rng_state,set_rng_state,same_nested,canonical_digest)
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.prepare_visual_continuation_training import validate
from scripts.train_semantic_glyph_language import numerics
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.score_pixel_language_first_blocks import require,owner_live
from scripts.infer_glyph_tail_language import prepare_prompt,load_actor as legacy_actor
from scripts.infer_compact_visual_span import save_grid


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('w') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def training_event(result,event_kind,attempt,**identity):
    require(not ({'event_kind','attempt'}|identity.keys())&result.keys(),'Training event fields overlap')
    return dict(result,event_kind=event_kind,attempt=attempt,**identity)


def payload(actor,optimizer,step,r,run,cost):
    return dict(schema='visual-continuation-training-checkpoint-v1',preflight_only=False,
        registration_sha256=sha(run/'preparation.json'),source_sha256=r['source_sha256'],seed=r['seed'],runtime=r['runtime'],
        inference_source_sha256={n:d for n,d in r['source_sha256'].items() if n.startswith('ilm/visual_lm/') or n in (
            'scripts/train_semantic_glyph_language.py','scripts/train_semantic_glyph_bridge.py',
            'scripts/compare_balanced_writer_matched.py','scripts/infer_glyph_tail_language.py','scripts/infer_glyph_context_motor_image.py')},
        step=step,total_steps=r['total_steps'],completed_training=step==r['total_steps'],
        initial_bridge_sha256=r['initial_bridge_sha256'],trainable_weights=trainable_state(actor),
        optimizer=optimizer.state_dict(),rng=rng_state(),cumulative_elapsed_seconds=cost)


def checkpoint(folder,actor,optimizer,step,r,run,cost,attempt):
    path=folder/'checkpoints'/f'step_{step:06}_attempt_{attempt:03}_{time.time_ns()}.pt'
    atomic_tensor(path,payload(actor,optimizer,step,r,run,cost))
    with path.open('rb') as f:os.fsync(f.fileno())
    record=dict(file=rel(path),sha256=sha(path),step=step)
    pointer=folder/'checkpoint.json';old=load(pointer) if pointer.exists() else None
    snapshots=load(folder/'snapshots.json') if (folder/'snapshots.json').exists() else {}
    if step in r['snapshot_steps']:
        key=str(step)
        if key in snapshots:require(snapshots[key]['sha256']==record['sha256'],'Preserve an existing snapshot')
        else:snapshots[key]=record;atomic_json(folder/'snapshots.json',snapshots)
    atomic_json(pointer,dict(latest=record,previous=old['latest'] if old else None))
    keep={x['file'] for x in snapshots.values()}|{record['file']}
    if old:keep.add(old['latest']['file'])
    # Only the previous pointer proves that an optimizer file was superseded.
    # Unreferenced files may be interrupted writes or evidence from another
    # attempt; a directory glob cannot establish that they are disposable.
    superseded=old.get('previous') if old else None
    if superseded and superseded['file'] not in keep:
        candidate=ROOT/superseded['file']
        require(candidate.parent.resolve()==(folder/'checkpoints').resolve(),'Checkpoint outside owned directory')
        require(sha(candidate)==superseded['sha256'],'Superseded checkpoint changed; preserve it')
        candidate.unlink()
    return record


def checked_checkpoint(path,r,run,digest=None):
    if digest is not None:require(sha(path)==digest,'Checkpoint hash differs')
    value=torch.load(path,weights_only=True,map_location='cpu')
    require(value['schema']=='visual-continuation-training-checkpoint-v1' and not value['preflight_only'] and
        value['registration_sha256']==sha(run/'preparation.json') and value['source_sha256']==r['source_sha256'] and
        value['total_steps']==r['total_steps'],'Wrong training checkpoint')
    return value


def train(run,resume):
    r=validate(run,images=True);folder=run/'fit';folder.mkdir(exist_ok=True);(folder/'checkpoints').mkdir(exist_ok=True)
    if (folder/'status.json').exists():require(not owner_live(load(folder/'status.json')),'Original training worker still live')
    if (folder/'report.json').exists() and load(folder/'report.json')['complete']:return
    require(resume==(folder/'checkpoint.json').exists(),'Use --resume exactly when a valid checkpoint exists')
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    numbers=[int(p.stem.split('_')[1]) for p in attempts.glob('attempt_*.json')]
    number=max(numbers,default=0)+1
    previous=None;prior_cost=float(r.get('prior_execution_elapsed_seconds',0.))
    if resume:
        pointer=load(folder/'checkpoint.json')['latest'];previous=checked_checkpoint(ROOT/pointer['file'],r,run,pointer['sha256'])
        prior_cost=previous['cumulative_elapsed_seconds']
        for p in attempts.glob('attempt_*.json'):
            # Every terminated attempt includes loading, waiting and unsaved work.
            prior_cost=max(prior_cost,load(p).get('cumulative_elapsed_seconds',0.))
    monitor=TrainingMonitor(folder,r['training_maximum_seconds']-prior_cost,r['training_pressure_wait_seconds'])
    # Reserve an immutable identity before CUDA admission. SIGKILL can prevent
    # finally from writing an end receipt, but must never reuse that ledger.
    start=dict(attempt=number,pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),began_unix=time.time(),
        prior_accounted_elapsed_seconds=prior_cost,resume=resume,
        resumed_step=previous['step'] if previous else 0)
    with (attempts/f'attempt_{number:03}_start.json').open('x') as f:
        json.dump(start,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    monitor.status('admitting');ledger=None;hooks=[];start_step=previous['step'] if previous else 0
    try:
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard();began=time.perf_counter()
            actor,provenance=build_actor(device,r['seed']);optimizer=torch.optim.AdamW(
                [p for p in actor.parameters() if p.requires_grad],lr=r['peak_learning_rate'],weight_decay=r['weight_decay'])
            if previous:
                restore_trainable(actor,previous['trainable_weights']);optimizer.load_state_dict(previous['optimizer']);set_rng_state(previous['rng'])
            frozen=[(p,p._version) for p in actor.parameters() if not p.requires_grad]
            for layer in actor.interface.actor.context.layers:hooks.append(layer.register_forward_pre_hook(lambda _m,_a:monitor.guard()))
            data=dict(data=TrainingImages(run),schedule=np.load(run/'schedule.npy'));total=r['total_steps']
            require(len(data['schedule'])==total,'Incomplete source exposure schedule')
            loading=time.perf_counter()-began
            atomic_json(folder/f'model_account_attempt_{number:03}.json',dict(
                total_unique_parameters=sum(p.numel() for p in actor.parameters()),
                trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
                inherited_visual_contextual_parameters=provenance['retained_base_parameters'],
                preflight_weights_loaded=False,provenance=provenance,loading_seconds=loading))
            if not previous:checkpoint(folder,actor,optimizer,0,r,run,prior_cost+time.perf_counter()-monitor.began,number)
            ledger=(folder/f'ledger_attempt_{number:03}.jsonl').open('a',buffering=1);chain='0'*64;step=start_step;resume4=None;control6=None
            for position in range(start_step+1,total+1):
                monitor.status('training',attempt=number,cumulative_elapsed_seconds=prior_cost+time.perf_counter()-monitor.began)
                result=update(actor,optimizer,data,position,total,monitor);step=position
                event=training_event(result,'scheduled',number,previous_event_sha256=chain);chain=canonical_digest(event)
                ledger.write(json.dumps(dict(event,event_sha256=chain),ensure_ascii=False)+'\n')
                if position<=6 or position%25==0:print(json.dumps(result,ensure_ascii=False),flush=True)
                if position==4 and not (folder/'resume_roundtrip.json').exists():
                    resume4=folder/f'resume_step4_attempt_{number:03}.pt'
                    atomic_tensor(resume4,payload(actor,optimizer,4,r,run,prior_cost+time.perf_counter()-monitor.began))
                    checkpoint(folder,actor,optimizer,4,r,run,prior_cost+time.perf_counter()-monitor.began,number)
                if position==6 and not (folder/'resume_roundtrip.json').exists():
                    if resume4 is None:
                        # A resumed step4 checkpoint is the same permitted branch.
                        require(start_step==4,'Recover interrupted resume audit before continuing');resume4=ROOT/load(folder/'checkpoint.json')['latest']['file']
                    control6=folder/f'resume_control6_attempt_{number:03}.pt'
                    atomic_tensor(control6,payload(actor,optimizer,6,r,run,prior_cost+time.perf_counter()-monitor.began))
                    expected=torch.load(control6,weights_only=True,map_location='cpu');saved=torch.load(resume4,weights_only=True,map_location='cpu')
                    restore_trainable(actor,saved['trainable_weights']);optimizer.load_state_dict(saved['optimizer']);set_rng_state(saved['rng']);checks=[]
                    for replay_step in (5,6):
                        replay=update(actor,optimizer,data,replay_step,total,monitor);checks.append(replay)
                        ledger.write(json.dumps(training_event(replay,'discarded_resume_verification',number),ensure_ascii=False)+'\n')
                    same_nested(trainable_state(actor),expected['trainable_weights']);same_nested(optimizer.state_dict(),expected['optimizer'])
                    same_nested(rng_state(),expected['rng'])
                    atomic_json(folder/'resume_roundtrip.json',dict(step4_sha256=sha(resume4),control_step6_sha256=sha(control6),
                        real_replayed_steps=checks,all_trainables_optimizer_and_RNG_bit_exact=True,extra_updates_not_added_to_training_step=2))
                    checkpoint(folder,actor,optimizer,6,r,run,prior_cost+time.perf_counter()-monitor.began,number)
                    print('REAL_RESUME_ROUNDTRIP_PASSED',flush=True);del expected,saved
                if position%r['checkpoint_interval']==0 or position in r['snapshot_steps']:
                    require(all(p._version==v and p.grad is None for p,v in frozen),'Frozen body/interface changed')
                    ledger.flush();os.fsync(ledger.fileno())
                    checkpoint(folder,actor,optimizer,position,r,run,prior_cost+time.perf_counter()-monitor.began,number)
                if position%100==0:gc.collect()
            require((folder/'resume_roundtrip.json').exists(),'Actual checkpoint replay not verified')
            require(all(p._version==v and p.grad is None for p,v in frozen),'Frozen model changed')
            for state in optimizer.state.values():require(int(state['step'])==total,'Actual Adam counters incomplete')
            snapshots=load(folder/'snapshots.json');require(sorted(map(int,snapshots))==r['snapshot_steps'],'Missing predetermined snapshots')
            atomic_json(folder/'report.json',dict(complete=True,registration_sha256=sha(run/'preparation.json'),
                actual_optimizer_steps=total,training_cases=r['training_cases'],epochs=3,
                scheduled_target_exposure=r['scheduled_target_exposure'],snapshots=snapshots,
                frozen_parameter_versions_unchanged=True,preflight_updates_carried=False,language_quality_evaluated=False,
                cumulative_elapsed_seconds=prior_cost+time.perf_counter()-monitor.began,**monitor.receipt()))
            monitor.status('complete',step=step);print('COMPLETE_FULL_TRAINING',flush=True)
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        if ledger:ledger.close()
        for hook in hooks:hook.remove()
        status=load(folder/'status.json');status['cumulative_elapsed_seconds']=prior_cost+time.perf_counter()-monitor.began
        atomic_json(attempts/f'attempt_{number:03}.json',status);atomic_json(folder/'status.json',dict(status,pid=None))


def verify_generation(folder,row,registration_sha,model_sha,cap):
    report=load(folder/'report.json')
    require(report['input']==row and report['registration_sha256']==registration_sha and
            report['checkpoint_sha256']==model_sha and report['maximum_glyphs']==cap,'Generation identity differs')
    for name,key in [('output.pt','tensor_sha256'),('ink.npz','raw_sha256'),('answer.png','image_sha256')]:
        require(sha(folder/name)==report[key],'Generated evidence changed')
    value=torch.load(folder/'output.pt',weights_only=True,map_location='cpu')
    with np.load(folder/'ink.npz') as arrays:ink=arrays['ink']
    require(len(ink)==report['emitted_glyphs'] and torch.equal(value['images'],1-2*torch.from_numpy(ink).float()),
            'Saved raster is not the emitted image stream')
    require(len(value['feedback'])==len(ink) and len(value['states'])==len(ink)+1,'Feedback/state trace incomplete')
    page=np.asarray(Image.open(folder/'answer.png').convert('L'))
    if len(ink):
        columns=min(32,len(ink));require(page.shape==(32*((len(ink)+columns-1)//columns),32*columns),'Display dimensions differ')
        for i,tile in enumerate(ink):
            require(np.array_equal(page[(i//columns)*32:(i//columns+1)*32,(i%columns)*32:(i%columns+1)*32],255-255*tile[0]),
                    'Displayed original pixels differ')
    else:require(page.shape==(32,32) and np.all(page==255),'Empty response display differs')
    return report


def generate_cases(actor,rows,folder,registration_sha,model_sha,cap,monitor):
    folder.mkdir(exist_ok=True,parents=True);records=[]
    for i,row in enumerate(rows):
        monitor.position=dict(case=row['id'],completed=i,total=len(rows),cap=cap);monitor.status('generating_registered_images');monitor.guard()
        destination=folder/row['id']
        if destination.exists():records.append(verify_generation(destination,row,registration_sha,model_sha,cap));continue
        require(sha(ROOT/row['image'])==row['sha256'],'Fixed evaluation prompt changed')
        canvas,_=prepare_prompt(ROOT/row['image']);prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
        began=time.perf_counter()
        with torch.no_grad():value=actor.generate(prompt,maximum_glyphs=cap)
        torch.cuda.synchronize();elapsed=time.perf_counter()-began;ink=((1-value['images'])/2).to(torch.uint8)
        require(torch.equal(value['images'],1-2*ink.float()),'Lossy image conversion')
        partial=folder/f'.{row["id"]}.{time.time_ns()}.partial';partial.mkdir()
        atomic_tensor(partial/'output.pt',value);np.savez_compressed(partial/'ink.npz',ink=ink.numpy());save_grid(ink,partial/'answer.png')
        for tile,f in zip(value['images'],value['feedback']):require(tensor_hash(tile[None])==f['image_sha256'],'Feedback did not use emitted pixels')
        record=dict(input=row,registration_sha256=registration_sha,checkpoint_sha256=model_sha,maximum_glyphs=cap,
            emitted_glyphs=len(ink),stopped=value['stopped'],capped=value['capped'],elapsed_seconds=elapsed,
            tensor_sha256=sha(partial/'output.pt'),raw_sha256=sha(partial/'ink.npz'),image_sha256=sha(partial/'answer.png'),
            actual_emitted_pixel_feedback=True,reference_inputs=False,output_repaired=False)
        atomic_json(partial/'report.json',record);partial.rename(destination);records.append(record)
        print(json.dumps(dict(case=row['id'],glyphs=len(ink),stopped=value['stopped'],seconds=elapsed)),flush=True)
    return records


def evaluate(run,parent=False):
    r=validate(run);registration_sha=sha(run/'preparation.json');folder=run/('parent_long_books' if parent else 'evaluation')
    folder.mkdir(exist_ok=True)
    if (folder/'status.json').exists():require(not owner_live(load(folder/'status.json')),'Original evaluator still live')
    if (folder/'report.json').exists() and load(folder/'report.json')['complete']:return
    monitor=TrainingMonitor(folder,r['evaluation_maximum_seconds'],r['evaluation_pressure_wait_seconds']);monitor.status('admitting');hooks=[]
    try:
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard();reports=[]
            if parent:
                from scripts.compare_balanced_writer_matched import CHECKPOINT,CHECKPOINT_SHA
                actor,provenance=legacy_actor(CHECKPOINT,device)
                rows=[row for row in r['evaluation_inputs'] if '__book_' in row['id']]
                hooks.append(actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard()))
                records=generate_cases(actor,rows,folder/'long640',registration_sha,CHECKPOINT_SHA,640,monitor)
                reports.append(dict(arm='balanced_parent_long640',step=18000,records=records))
            else:
                require(load(run/'fit/report.json')['complete'],'Complete registered training before evaluating fixed snapshots')
                actor,provenance=build_actor(device,r['seed']);actor.eval()
                hooks.append(actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard()))
                for layer in actor.interface.actor.context.layers:hooks.append(layer.register_forward_pre_hook(lambda _m,_a:monitor.guard()))
                snapshots=load(run/'fit/snapshots.json')
                for step in r['snapshot_steps']:
                    record=snapshots[str(step)];value=checked_checkpoint(ROOT/record['file'],r,run,record['sha256'])
                    restore_trainable(actor,value['trainable_weights']);del value;versions=[(p,p._version) for p in actor.parameters()]
                    rows=r['evaluation_inputs'] if step in (0,r['total_steps']) else [x for x in r['evaluation_inputs'] if x['condition']=='normal']
                    results=generate_cases(actor,rows,folder/f'step_{step:06}'/'core128',registration_sha,record['sha256'],128,monitor)
                    reports.append(dict(arm='candidate_core128',step=step,records=results))
                    if step in (0,r['total_steps']):
                        books=[x for x in r['evaluation_inputs'] if '__book_' in x['id']]
                        results=generate_cases(actor,books,folder/f'step_{step:06}'/'long640',registration_sha,record['sha256'],640,monitor)
                        reports.append(dict(arm='candidate_long640',step=step,records=results))
                    require(all(p._version==v for p,v in versions),'Evaluation changed a model parameter')
            atomic_json(folder/'report.json',dict(complete=True,registration_sha256=registration_sha,arms=reports,
                total_streams=sum(len(x['records']) for x in reports),new_optimizer_updates=0,provenance=provenance,
                language_quality_unmeasured=True,**monitor.receipt()));monitor.status('complete')
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        for h in hooks:h.remove()
        if (folder/'status.json').exists():atomic_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['train','evaluate','parent-long-books'])
    p.add_argument('--run',type=Path,default=RUN);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='train':train(a.run,a.resume)
    else:evaluate(a.run,a.action=='parent-long-books')
