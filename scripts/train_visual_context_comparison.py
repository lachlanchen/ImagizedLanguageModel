#!/usr/bin/env python3
"""One complete registered corpus pass, with a frozen or fully learned context."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def train(registration,arm):
    import torch
    from ilm.image_edit.baseline import save_json,sha
    from ilm.visual_lm.trainable_visual_context import build_model,configure_optimizer
    from ilm.visual_lm.trainable_context_checkpoint import payload,restore
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.compact_visual_span_training import VisualSpanTrainingExample,initialize_feedback_statistics
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from scripts.cache_visual_sequence_teacher_states import batches,read_sequences
    from scripts.train_full_rank_visual_span import training_source,load_states,source_targets
    from scripts.train_image_native_lm import atomic_torch_save
    from scripts.register_visual_context_comparison import validate

    reg=validate(registration);settings=reg['training'];learn=arm=='trainable'
    run=ROOT/reg['run']/arm/'fit';run.mkdir(parents=True,exist_ok=False)
    teacher=ROOT/reg['teacher'];codec=ROOT/reg['codec']
    corpus,plan,cache,_,_=training_source(teacher)
    identity=dict(registration_sha256=sha(registration),train_context=learn,
                  codec_sha256=sha(codec),model_sources=reg['model_sources'],seed=settings['seed'])
    monitor=GlyphReaderMonitor(run,settings['maximum_seconds_per_arm']);monitor.status('admitting_full_corpus_fit')
    monitor.pressure_wait_allowance_seconds=settings['maximum_pressure_wait_seconds']
    updates=0;history=[]
    try:
        with full_reader_gpu() as (device,admission):
            torch.set_num_threads(2);monitor.admission=admission;monitor.guard()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            model,objective,provenance=build_model(ROOT.parent/'LocalLLM',codec,device)
            archive=open_compact_archive(ROOT.parent/'LocalLLM')
            lookup,references=source_targets(archive,plan['training_native_units'])
            statistics=initialize_feedback_statistics(model,archive,plan['training_native_units']);del archive
            optimizer=configure_optimizer(model,objective,train_context=learn,guard=monitor.guard)
            params=[p for group in optimizer.param_groups for p in group['params']]
            initial_sha=tensor_state_sha(model.state_dict())
            objective_sha=tensor_state_sha(objective.state_dict())
            atomic_torch_save(payload(model,identity,0),run/'model_step_0000000.pt')
            saved=torch.load(run/'model_step_0000000.pt',weights_only=True,map_location='cpu',mmap=True)
            restore(model,saved,identity);del saved
            if tensor_state_sha(model.state_dict())!=initial_sha:
                raise ValueError('Actual full-model checkpoint roundtrip changed initial weights')
            save_json(run/'initial.json',dict(model_sha256=initial_sha,objective_sha256=objective_sha,
                full_model_checkpoint_roundtrip_exact=True,training_parameters=sum(p.numel() for p in params),
                total_model_parameters=sum(p.numel() for p in model.parameters()),feedback_statistics=statistics))
            order=list(range(len(cache)));random.Random(settings['seed']).shuffle(order)
            with (run/'updates.jsonl').open('x') as ledger:
                for ordinal,shard_index in enumerate(order):
                    monitor.position=dict(arm=arm,updates=updates,completed_shards=ordinal,total_shards=len(cache))
                    monitor.status('fitting_complete_visual_records',shard=shard_index);monitor.guard()
                    start=time.perf_counter();shard=plan['shards'][shard_index]
                    records=read_sequences(corpus.directory,shard)
                    states,offsets=load_states(teacher,cache[shard_index],records)
                    groups=list(batches(records,settings['padded_unit_budget']))
                    random.Random(settings['seed']+len(cache)+ordinal).shuffle(groups)
                    sums={}
                    for group in groups:
                        monitor.guard();examples=[];ids=[]
                        for index in group:
                            at=shard['first_record']+index;ids.append(at)
                            record=records[index];row=corpus[at];alignment=corpus.offline_teacher_alignment(at)
                            selected=lookup[alignment['unit_ids'].long()]
                            if bool((selected<0).any()) or not np.array_equal(record.units,alignment['unit_ids'].numpy()):
                                raise ValueError('Original complete training alignment changed')
                            examples.append(VisualSpanTrainingExample(row['ink'].to(device),row['span_glyph_ends'],
                                alignment['span_unit_ends'],row['prompt_spans'],references[selected],
                                states[offsets[index]:offsets[index+1]]))
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            losses=objective(model,examples);loss=torch.stack(list(losses.values())).sum()
                        values={n:float(v.detach()) for n,v in losses.items()}
                        if not all(np.isfinite(list(values.values()))):raise ValueError('Nonfinite complete-record loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                        optimizer.step();updates+=1
                        item=dict(update=updates,shard=shard_index,corpus_indices=ids,losses=values,gradient_norm=float(norm))
                        ledger.write(json.dumps(item)+'\n');ledger.flush()
                        for n,v in values.items():sums[n]=sums.get(n,0.)+v
                        del examples,losses,loss
                    result=dict(shard=shard_index,ordinal=ordinal,records=len(records),target_states=len(states),
                        optimizer_updates=updates,mean_batch_losses={n:v/len(groups) for n,v in sums.items()},
                        seconds=time.perf_counter()-start,moment_bytes=optimizer.moment_bytes())
                    history.append(result);save_json(run/'history.json',history)
                    print(json.dumps(result),flush=True)
                    optimizer.zero_grad(set_to_none=True);del states,records;gc.collect();monitor.guard()
                    fs=os.statvfs(ROOT)
                    if fs.f_bavail*fs.f_frsize/1024**3<settings['minimum_free_disk_gib']:
                        raise RuntimeError('Training disk reserve reached')
                    if (ordinal+1)%settings['checkpoint_every_shards']==0:
                        deploy=payload(model,identity,updates)
                        if ordinal+1 in reg['milestone_completed_shards']:
                            atomic_torch_save(deploy,run/f'model_step_{updates:07d}.pt')
                        atomic_torch_save(dict(deploy=deploy,completed_shards=ordinal+1,history=history,
                            teacher_projection=objective.state_dict(),optimizer=optimizer.state_dict(),
                            cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(device)),run/'checkpoint_latest.pt')
                        del deploy;gc.collect();monitor.guard()
            if updates!=reg['updates_per_arm'] or sum(x['records'] for x in history)!=13785 or sum(x['target_states'] for x in history)!=plan['state_rows']:
                raise ValueError('The complete corpus schedule does not reconcile')
            final=run/f'model_step_{updates:07d}.pt'
            save_json(run/'report.json',dict(identity=identity,updates=updates,training_records=len(corpus),
                causal_state_targets=plan['state_rows'],target_glyph_positions=reg['target_glyph_positions_per_arm'],
                final_checkpoint=final.name,final_checkpoint_sha256=sha(final),history=history,
                training_parameters=sum(p.numel() for p in params),total_model_parameters=sum(p.numel() for p in model.parameters()),
                all_optimizer_counters=[int(v['step']) for v in optimizer.state.values()],
                optimizer_moment_bytes=optimizer.moment_bytes(),resources=monitor.receipt(),
                language_evaluated=False,goal_achieved=False))
            monitor.status('complete_full_corpus_fit',updates=updates)
    except BaseException as error:
        monitor.status('failed_full_corpus_fit',error=repr(error),updates=updates,automatic_retry=False)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--arm',choices=('frozen','trainable'),required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        p.error('Require physical GPU0 and the registered CUDA workspace')
    train(a.registration,a.arm)
