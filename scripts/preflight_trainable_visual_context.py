#!/usr/bin/env python3
"""Register and execute real full-context updates/reset on complete training records."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/visual_sequence_corpus_v1'
TEACHER=ROOT/'artifacts/visual_sequence_teacher_v1'
CODEC=ROOT/'artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt'
REG=ROOT/'references/evidence/trainable_visual_context_preflight_registration_20261116.json'
OUT=ROOT/'references/evidence/trainable_visual_context_preflight_result_20261116.json'
RUN=ROOT/'artifacts/trainable_visual_context_preflight_v1'
PROTOCOL=ROOT/'references/trainable_visual_context_preflight_protocol_20261116.md'


def load(path):return json.loads(path.read_text())


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def registration():
    report=load(DATA/'report.json');prep=load(TEACHER/'preparation.json');cache=load(TEACHER/'cache/report.json')
    source=[Path(__file__),PROTOCOL,DATA/'report.json',DATA/'verification.json',TEACHER/'preparation.json',TEACHER/'cache/report.json',CODEC,
        ROOT/'ilm/visual_lm/trainable_visual_context.py',ROOT/'ilm/visual_lm/offloaded_adamw.py',
        ROOT/'tests/test_offloaded_adamw.py',ROOT/'ilm/visual_lm/compact_visual_span_model.py',
        ROOT/'ilm/visual_lm/compact_visual_span_weights.py',ROOT/'ilm/visual_lm/compact_visual_span_training.py',
        ROOT/'ilm/visual_lm/full_rank_visual_span.py',ROOT/'scripts/train_full_rank_visual_span.py',
        ROOT/'scripts/cache_visual_sequence_teacher_states.py',ROOT/'ilm/visual_lm/glyph_reader_monitor.py',
        ROOT/'ilm/visual_lm/glyph_runtime_resources.py',ROOT/'scripts/train_semantic_glyph_bridge.py',
        ROOT/'scripts/train_semantic_glyph_language.py',ROOT/'scripts/score_pixel_language_first_blocks.py',
        ROOT/'ilm/visual_lm/gguf_language_trunk.py',ROOT/'ilm/visual_lm/grounded_image_weights.py',
        ROOT/'ilm/visual_lm/continuous_glyph_codec.py',ROOT/'ilm/visual_lm/causal_glyph_flow.py',
        ROOT/'ilm/visual_lm/visual_sequence_corpus.py',
        ROOT/'references/evidence/training_glyph_label_result_20261116.json']
    train=[s for s in report['shards'] if s['split']=='train']
    if prep['data_report_sha256']!=sha(DATA/'report.json') or len(train)!=len(cache['shards']):raise ValueError('Original training/cache identity differs')
    selected={};maximum=None;offset=0
    for i,s in enumerate(train):
        pixels=next(n for n in s['files'] if n.endswith('_pixels.npz'))
        teachers=next(n for n in s['files'] if n.endswith('_teacher.npz'))
        origins=next(n for n in s['files'] if n.endswith('_sources.json'))
        for n in (pixels,teachers,origins):
            if sha(DATA/n)!=s['files'][n]:raise ValueError('Training shard changed')
        with np.load(DATA/pixels,allow_pickle=False) as a:
            lengths=np.diff(a['record_glyph_offsets']);prompts=a['prompt_glyphs']
        with np.load(DATA/teachers,allow_pickle=False) as a:units=np.diff(a['record_unit_offsets'])
        metadata=load(DATA/origins)
        for j,row in enumerate(metadata):
            record=dict(corpus_index=offset+j,shard=i,local=j,kind=row['kind'],source_id=row['id'],group=row['group'],
                        glyphs=int(lengths[j]),prompt_glyphs=int(prompts[j]),native_units=int(units[j]))
            if row['kind']=='qa' and 'complete_qa' not in selected:selected['complete_qa']=record
            if row['kind']=='book' and lengths[j]==513 and 'complete_513_book' not in selected:selected['complete_513_book']=record
            if maximum is None or units[j]>maximum['native_units']:maximum=record
        offset+=s['records']
    if offset!=13785 or len(selected)!=2:raise ValueError('Keep original full-corpus support and complete required examples')
    selected['longest_native_record']=maximum
    for row in selected.values():
        shard=train[row['shard']]
        source.extend(DATA/n for n in shard['files'])
        state=TEACHER/'cache'/cache['shards'][row['shard']]['directory']/'states.pt'
        if sha(state)!=cache['shards'][row['shard']]['states_sha256']:raise ValueError('Teacher state source changed')
        source.append(state)
    paths={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(source))}
    return dict(schema='trainable-visual-context-preflight-registration-v1',input_sha256=paths,
        packages={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','Pillow')},
        cases=[dict(role=role,**row) for role,row in selected.items()],training_records=offset,
        treatment='All 1.409B context parameters trainable; full-rank image writer/encoder/objective retained.',
        matched_control='Same FP32 model and offloaded FP32 AdamW, context frozen. Same initial values, examples and reset RNG.',
        resources=dict(entry_host_available_gib=48,runtime_host_available_gib=24,maximum_allocated_gpu_gib=22,
                       maximum_seconds=1800,maximum_pressure_wait_seconds=300,cpu_threads=2,minimum_free_disk_gib=48),
        learning_rates=dict(context=2e-5,image_heads_and_projection=2e-4),weight_decay=.01,gradient_clip=1,
        required=['complete losses and backward finite','all context gradients present/nonzero in treatment and absent in control',
            'actual optimizer step changes every context tensor only in treatment','all active optimizer counters one and moments host-resident',
            'real-gradient sampled update agrees with independent torch AdamW','all initial model/objective parameters exactly restored after every step',
            'codec unchanged and no lexical embedding/head','resource ceilings respected'],
        action_on_failure='Stop and preserve failure. No full fit or automatic retry.',
        action_on_pass='Execution support only. Separately register full matched-corpus fit and complete autonomous evaluation before a learning claim.',
        optimizer_trial_updates_reset=True,full_fit_authorized=False,language_quality_proved=False)


def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Require physical GPU0 and the registered CUDA workspace before Torch import')
    import torch
    from ilm.visual_lm.trainable_visual_context import build_model,configure_optimizer
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.compact_visual_span_training import VisualSpanTrainingExample,initialize_feedback_statistics
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from scripts.train_full_rank_visual_span import training_source,load_states,source_targets
    from scripts.cache_visual_sequence_teacher_states import read_sequences
    reg=load(REG)
    if reg['packages']!={n:importlib.metadata.version(n) for n in reg['packages']}:
        raise ValueError('Registered package versions changed')
    for n,digest in reg['input_sha256'].items():
        if sha(ROOT/n)!=digest:raise ValueError('Bound source changed: '+n)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,timeout=30).split()[0]
    if remote!=head or subprocess.check_output(['git','show','HEAD:'+str(REG.relative_to(ROOT))],cwd=ROOT)!=REG.read_bytes():
        raise ValueError('Commit and push full-context preflight before execution')
    if RUN.exists() or OUT.exists():raise FileExistsError('Preserve first preflight; no automatic retry')
    RUN.mkdir(parents=True);monitor=GlyphReaderMonitor(RUN,1800);monitor.pressure_wait_allowance_seconds=300
    monitor.status('admitting_full_context_preflight')
    rows=[]
    try:
        with full_reader_gpu() as (device,admission):
            torch.set_num_threads(2);monitor.admission=admission;monitor.guard()
            fs=os.statvfs(ROOT)
            if fs.f_bavail*fs.f_frsize/1024**3<48:raise RuntimeError('Disk reserve exhausted')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            model,objective,provenance=build_model(ROOT.parent/'LocalLLM',CODEC,device)
            corpus,plan,cache,_,_=training_source(TEACHER)
            archive=open_compact_archive(ROOT.parent/'LocalLLM')
            lookup,references=source_targets(archive,plan['training_native_units'])
            initialize_feedback_statistics(model,archive,plan['training_native_units']);del archive
            initial={n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
            initial_objective={n:p.detach().cpu().clone() for n,p in objective.state_dict().items()}
            initial_hash=tensor_state_sha(initial);objective_hash=tensor_state_sha(initial_objective)
            cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(device)
            control_losses={}
            for train_context in (False,True):
                for case in reg['cases']:
                    monitor.position=dict(train_context=train_context,role=case['role']);monitor.status('complete_real_record_update')
                    monitor.guard();began=time.perf_counter()
                    shard=plan['shards'][case['shard']];records=read_sequences(corpus.directory,shard)
                    states,offsets=load_states(TEACHER,cache[case['shard']],records)
                    record=records[case['local']];row=corpus[case['corpus_index']]
                    alignment=corpus.offline_teacher_alignment(case['corpus_index']);chosen=lookup[alignment['unit_ids'].long()]
                    if bool((chosen<0).any()) or not np.array_equal(record.units,alignment['unit_ids'].numpy()):raise ValueError('Complete training alignment differs')
                    example=VisualSpanTrainingExample(row['ink'].to(device),row['span_glyph_ends'],alignment['span_unit_ends'],
                        row['prompt_spans'],references[chosen],states[offsets[case['local']]:offsets[case['local']+1]])
                    optimizer=configure_optimizer(model,objective,train_context=train_context,guard=monitor.guard)
                    optimizer.zero_grad(set_to_none=True)
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        losses=objective(model,[example]);total=torch.stack(list(losses.values())).sum()
                    if not bool(torch.isfinite(total)):raise ValueError('Nonfinite complete-image loss')
                    loss_values={n:float(v.detach()) for n,v in losses.items()}
                    if train_context:
                        if max(abs(loss_values[n]-control_losses[case['role']][n]) for n in loss_values)>1e-6:
                            raise ValueError('Matched arms do not begin with the same complete losses')
                    else:control_losses[case['role']]=loss_values
                    total.backward();params=[p for g in optimizer.param_groups for p in g['params']]
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                    context_grad={n:dict(present=p.grad is not None,finite=p.grad is not None and bool(torch.isfinite(p.grad).all()),
                        nonzero=p.grad is not None and bool(p.grad.any())) for n,p in model.context.named_parameters()}
                    if train_context and not all(v['present'] and v['finite'] and v['nonzero'] for v in context_grad.values()):
                        raise ValueError('Full context did not receive finite nonzero gradients')
                    if not train_context and any(v['present'] for v in context_grad.values()):raise ValueError('Frozen control acquired context gradients')
                    ref_groups=[];samples=[]
                    for group in optimizer.param_groups:
                        ps=[]
                        for p in group['params']:
                            if p.grad is None:continue
                            ix=torch.linspace(0,p.numel()-1,min(17,p.numel()),device=p.device).long().unique()
                            q=torch.nn.Parameter(p.detach().reshape(-1)[ix].clone());q.grad=p.grad.reshape(-1)[ix].clone()
                            ps.append(q);samples.append((p,ix,q))
                        ref_groups.append(dict(params=ps,lr=group['lr']))
                    reference=torch.optim.AdamW(ref_groups,weight_decay=.01,foreach=False,fused=False)
                    reference.step();optimizer.step();monitor.guard()
                    maximum_error=max(float((p.detach().reshape(-1)[ix]-q).abs().max()) for p,ix,q in samples)
                    for p,ix,q in samples:
                        torch.testing.assert_close(p.detach().reshape(-1)[ix],q,rtol=1e-6,atol=1e-7)
                    if any(v['step']!=1 or v['exp_avg'].device.type!='cpu' or v['exp_avg_sq'].device.type!='cpu' for v in optimizer.state.values()):
                        raise ValueError('Optimizer counters/moment placement differ')
                    changed=[n for n,p in model.context.named_parameters() if not torch.equal(p.detach().cpu(),initial['context.'+n])]
                    if bool(changed)!=train_context or train_context and len(changed)!=len(list(model.context.parameters())):
                        raise ValueError('Actual context parameter changes differ from selected arm')
                    moment_bytes=optimizer.moment_bytes();steps=len(optimizer.state)
                    if steps!=len(params):raise ValueError('A trainable tensor lacks actual optimizer state')
                    optimizer.zero_grad(set_to_none=True);optimizer.state.clear()
                    del optimizer,reference,ref_groups,samples,losses,total
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng,device)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                        after_losses=objective(model,[example])
                    after_values={n:float(v) for n,v in after_losses.items()}
                    if not all(np.isfinite(list(after_values.values()))):raise ValueError('Nonfinite post-update complete loss')
                    del example,after_losses,states,records
                    with torch.no_grad():
                        for n,p in model.named_parameters():p.copy_(initial[n].to(p.device))
                        for n,b in model.named_buffers():
                            if n in initial:b.copy_(initial[n].to(b.device))
                        for n,p in objective.named_parameters():p.copy_(initial_objective[n].to(p.device))
                    if tensor_state_sha(model.state_dict())!=initial_hash or tensor_state_sha(objective.state_dict())!=objective_hash:
                        raise ValueError('Preflight update was not exactly reset')
                    receipt=dict(**case,train_context=train_context,complete_loss=sum(loss_values.values()),
                        losses=loss_values,post_update_losses=after_values,
                        clipped_gradient_norm=float(norm),context_tensors=len(context_grad),changed_context_tensors=len(changed),
                        optimizer_state_tensors=steps,moment_bytes=moment_bytes,maximum_sampled_adamw_error=maximum_error,
                        all_parameters_restored=True,seconds=time.perf_counter()-began,context_gradients=context_grad)
                    rows.append(receipt);write_new(RUN/f'update_{len(rows):02}.json',receipt)
                    print(json.dumps({k:v for k,v in receipt.items() if k!='context_gradients'}),flush=True)
                    gc.collect();monitor.guard()
            result=dict(schema='trainable-visual-context-preflight-result-v1',registration_sha256=sha(REG),
                model=provenance,updates=rows,all_trial_updates_reset=True,resources=monitor.receipt(),
                preflight_passed=len(rows)==2*len(reg['cases']),full_fit_authorized=False,language_quality_proved=False,
                rejected_reward_used=False,goal_achieved=False)
            write_new(OUT,result);monitor.status('complete_real_context_update_reset',preflight_passed=True)
    except BaseException as error:
        write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),updates_completed=len(rows),
                                        automatic_retry=False,full_fit_authorized=False))
        monitor.status('failed_full_context_preflight',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));args=parser.parse_args()
    if args.action=='register':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only registration required')
        write_new(REG,registration());print('Registered full-context training preflight')
    else:run()
