#!/usr/bin/env python3
"""Bind the complete matched context-learning comparison before any fit."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import shutil
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/'references/evidence/trainable_visual_context_comparison_registration_20261116.json'
PROTOCOL=ROOT/'references/trainable_visual_context_comparison_protocol_20261116.md'
RUN='artifacts/trainable_visual_context_comparison_v1'


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):d.update(block)
    return d.hexdigest()


def load(path):return json.loads(Path(path).read_text())


def validate(path):
    r=load(path)
    if r['schema']!='trainable-visual-context-comparison-v1':raise ValueError('Comparison schema changed')
    for name,digest in r['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Bound source changed: '+name)
    if r['packages']!={n:importlib.metadata.version(n) for n in r['packages']}:raise ValueError('Package version changed')
    return r


def register():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for registration')
    from scripts.train_full_rank_visual_span import training_source
    from scripts.cache_visual_sequence_teacher_states import read_sequences,batches
    preflight=ROOT/'references/evidence/trainable_visual_context_preflight_result_20261116.json'
    p=load(preflight)
    if not p['preflight_passed'] or not p['all_trial_updates_reset'] or len(p['updates'])!=6:
        raise ValueError('Complete actual update/reset preflight required')
    original=load(ROOT/'references/evidence/trainable_visual_context_preflight_registration_20261116.json')
    sources=set(original['input_sha256'])
    sources.update(('scripts/register_visual_context_comparison.py','scripts/train_visual_context_comparison.py',
        'scripts/infer_trainable_visual_context.py','scripts/run_visual_context_comparison.py',
        'scripts/review_visual_context_comparison.py','ilm/visual_lm/trainable_context_checkpoint.py',
        'tests/test_trainable_context_checkpoint.py','references/trainable_visual_context_comparison_protocol_20261116.md',
        'references/evidence/trainable_visual_context_preflight_result_20261116.json',
        'scripts/infer_compact_visual_span.py','ilm/visual_lm/visual_cell_data.py','scripts/train_image_native_lm.py',
        'scripts/prepare_compact_visual_span_evaluation.py','scripts/review_compact_visual_span.py'))
    sources.update(('scripts/verify_trainable_visual_context_cold.py','scripts/verify_full_rank_visual_span_cold.py',
                    'scripts/verify_glyph_tail_cold_inference.py'))
    if shutil.which('strace') is None:raise ValueError('The registered cold-access trace requires strace')
    teacher=ROOT/'artifacts/visual_sequence_teacher_v1';corpus,plan,cache,_,_=training_source(teacher)
    order=[];updates=0;milestone_updates=[]
    for index,shard in enumerate(plan['shards']):
        sources.update(str((corpus.directory/n).relative_to(ROOT)) for n in shard['files'])
        state=teacher/'cache'/cache[index]['directory']/'states.pt'
        if sha(state)!=cache[index]['states_sha256']:raise ValueError('Complete teacher cache changed')
        sources.add(str(state.relative_to(ROOT)))
        records=read_sequences(corpus.directory,shard)
        order.append(len(list(batches(records,1024))))
    import random
    shuffled=list(range(len(cache)));random.Random(202611152).shuffle(shuffled)
    for ordinal,index in enumerate(shuffled,1):
        updates+=order[index]
        if ordinal in (27,54,81,108):milestone_updates.append(updates)
    if len(corpus)!=13785 or len(cache)!=108 or updates!=8172:raise ValueError('Keep the original complete schedule')
    evaluation=ROOT/'artifacts/compact_visual_span_evaluation_v1'
    sources.update(str((evaluation/n).relative_to(ROOT)) for n in ('preparation.json','references.json','inputs/manifest.json'))
    prep=load(evaluation/'preparation.json');manifest=load(evaluation/'inputs/manifest.json')
    if sha(evaluation/'inputs/manifest.json')!=prep['input_manifest_sha256'] or sha(evaluation/'references.json')!=prep['references_sha256']:
        raise ValueError('Original evaluation changed')
    for row in manifest['inputs']:
        path=evaluation/'inputs'/row['image']
        if sha(path)!=row['image_sha256']:raise ValueError('Evaluation image changed')
        sources.add(str(path.relative_to(ROOT)))
    if len(manifest['inputs'])!=120 or sum(x['condition']=='normal' for x in manifest['inputs'])!=44:
        raise ValueError('Keep all paired questions, erasures and book prefixes')
    model_names=('ilm/visual_lm/trainable_visual_context.py','ilm/visual_lm/trainable_context_checkpoint.py',
        'ilm/visual_lm/compact_visual_span_model.py','ilm/visual_lm/compact_visual_span_weights.py',
        'ilm/visual_lm/full_rank_visual_span.py','ilm/visual_lm/continuous_glyph_codec.py',
        'ilm/visual_lm/causal_glyph_flow.py','ilm/visual_lm/gguf_language_trunk.py','ilm/visual_lm/grounded_image_weights.py')
    r=dict(schema='trainable-visual-context-comparison-v1',run=RUN,
        registered_utc=datetime.now(timezone.utc).isoformat(),
        teacher=str(teacher.relative_to(ROOT)),codec='artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt',
        evaluation=str(evaluation.relative_to(ROOT)),model_sources={n:sha(ROOT/n) for n in model_names},
        input_sha256={n:sha(ROOT/n) for n in sorted(sources)},packages=original['packages'],
        treatment='All 1,409,410,048 contextual parameters learn; encoder/full-rank writer/objectives retained.',
        control='Fresh matched frozen context; same FP32 weights, BF16 autocast, source order and host-moment AdamW.',
        training=dict(epochs=1,padded_unit_budget=1024,seed=202611152,context_learning_rate=2e-5,
            image_learning_rate=2e-4,weight_decay=.01,gradient_clip=1,checkpoint_every_shards=9,
            maximum_seconds_per_arm=86400,maximum_pressure_wait_seconds=1800,minimum_free_disk_gib=48),
        training_records_per_arm=13785,causal_state_targets_per_arm=plan['state_rows'],
        target_glyph_positions_per_arm=2576239,updates_per_arm=updates,
        milestone_completed_shards=[27,54,81,108],snapshot_updates=[0,*milestone_updates],
        evaluation_configuration=dict(seed=202611153,flow_steps=8,short_cap=128,long_book_cap=640,
            initial_and_final_cases=120,intermediate_cases=44,long_final_book_cases=24,
            total_streams_per_arm=396,maximum_seconds_per_snapshot=7200),
        resources=dict(entry_host_gib=48,runtime_host_gib=24,maximum_allocated_vram_gib=22,cpu_threads=2,physical_gpu=0),
        primary='Final complete exact QA/32, both-correct pairs/16, readability, erasure dependence, coherent complete books and stopping.',
        pass_for_next_confirmation='At least12/32 exact,4/16 both-correct,24/32 readable,4/12 coherent naturally stopped640-cap books; each QA erasure <=2 exact and normal-minus-best-erased >=8. Retain original successes separately. This single-layout stage cannot satisfy the full goal.',
        action_on_failure='Preserve all results. No intermediate selection, threshold repair, reward tuning or unchanged extra epochs. Diagnose measured failure before another change.',
        action_on_pass='Only development support; separately bind untouched confirmation and remaining proportional/historical/etymology and matched-quality cost requirements.',
        live_teacher_at_inference=False,rejected_rewards_used=False,DRCD_fitting=False,goal_achieved=False)
    with REG.open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'registration':str(REG.relative_to(ROOT)),'sources':len(sources),'updates_per_arm':updates,
                      'snapshots':r['snapshot_updates'],'streams_per_arm':396}))


def require_pushed(path):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,timeout=30).split()[0]
    if head!=remote or subprocess.check_output(['git','show','HEAD:'+str(path.relative_to(ROOT))],cwd=ROOT)!=path.read_bytes():
        raise ValueError('Commit and push complete comparison before execution')


def audit(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Audit with CUDA hidden')
    import math
    import random
    import torch
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from scripts.train_full_rank_visual_span import training_source
    from scripts.cache_visual_sequence_teacher_states import read_sequences,batches
    reg=validate(REG);run=ROOT/reg['run']/arm/'fit';report=load(run/'report.json')
    corpus,plan,cache,_,_=training_source(ROOT/reg['teacher'])
    order=list(range(len(cache)));random.Random(reg['training']['seed']).shuffle(order)
    expected=[]
    for ordinal,index in enumerate(order):
        shard=plan['shards'][index];records=read_sequences(corpus.directory,shard)
        groups=list(batches(records,1024));random.Random(reg['training']['seed']+len(cache)+ordinal).shuffle(groups)
        expected.extend((index,[shard['first_record']+i for i in group]) for group in groups)
    observed=[];seen=[]
    for number,line in enumerate((run/'updates.jsonl').read_text().splitlines(),1):
        row=json.loads(line)
        if row['update']!=number or not math.isfinite(row['gradient_norm']) or len(row['losses'])!=6 or not all(math.isfinite(v) for v in row['losses'].values()):
            raise ValueError('Actual update ledger differs')
        observed.append((row['shard'],row['corpus_indices']));seen.extend(row['corpus_indices'])
    if observed!=expected or len(observed)!=8172 or sorted(seen)!=list(range(13785)):
        raise ValueError('Actual complete source visits/order are not the registered schedule')
    if [r['shard'] for r in report['history']]!=order or sum(r['target_states'] for r in report['history'])!=plan['state_rows']:
        raise ValueError('Complete source/state accounting differs')
    saved=torch.load(run/'checkpoint_latest.pt',weights_only=True,map_location='cpu',mmap=True)
    final_path=run/report['final_checkpoint'];final=torch.load(final_path,weights_only=True,map_location='cpu',mmap=True)
    if saved['completed_shards']!=108 or saved['deploy']['identity']!=final['identity'] or saved['deploy']['updates']!=8172 or final['updates']!=8172:
        raise ValueError('Latest and deployable final cursor/identity differ')
    if final['identity']['registration_sha256']!=sha(REG) or final['identity']['train_context']!=(arm=='trainable'):
        raise ValueError('Actual final model belongs to another arm')
    for name in ('encoder','writer'):
        if tensor_state_sha(saved['deploy']['image_heads'][name])!=tensor_state_sha(final['image_heads'][name]):
            raise ValueError('Final learned image tensors differ')
    if arm=='trainable':
        if len(final['context'])!=309 or tensor_state_sha(final['context'])!=tensor_state_sha(saved['deploy']['context']):
            raise ValueError('Final learned context tensors were lost')
    elif final['context'] is not None:raise ValueError('Frozen control unexpectedly stored a learned context')
    states=saved['optimizer']['state'];count=409 if arm=='trainable' else 100
    if len(states)!=count or any(v['step']!=8172 for v in states.values()):
        raise ValueError('Final optimizer tensors/counters do not reconcile')
    if sum(v[k].numel()*v[k].element_size() for v in states.values() for k in ('exp_avg','exp_avg_sq'))!=8*report['training_parameters']:
        raise ValueError('Final moment storage differs from complete trainables')
    if sha(final_path)!=report['final_checkpoint_sha256']:raise ValueError('Final model file changed')
    snapshots=sorted(p.name for p in run.glob('model_step_*.pt'))
    if snapshots!=[f'model_step_{n:07d}.pt' for n in reg['snapshot_updates']]:raise ValueError('Fixed snapshot set differs')
    result=dict(registration_sha256=sha(REG),arm=arm,updates=8172,records=13785,
        ledger_sha256=sha(run/'updates.jsonl'),final_checkpoint_sha256=sha(final_path),
        expected_source_order_exact=True,latest_matches_full_deploy=True,optimizer_counters_all_exact=True,
        audit_passed=True,cuda_initialized=torch.cuda.is_initialized(),language_quality_proved=False)
    if result['cuda_initialized']:raise ValueError('CPU audit initialized CUDA')
    with (run/'audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',choices=('trainable','frozen'));args=parser.parse_args()
    if args.audit:audit(args.audit)
    else:register()
