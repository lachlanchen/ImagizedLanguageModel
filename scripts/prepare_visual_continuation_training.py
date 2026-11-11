#!/usr/bin/env python3
"""Prepare complete source-bound training images in fixed and natural layouts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.continuation_training import RUN,DATA,SEED,load,rel,layout_schedule
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.prepare_visual_sequence_corpus import documents,text_hash
from scripts.preflight_visual_continuation import SOURCES as MODEL_SOURCES
from scripts.score_pixel_language_first_blocks import require

SOURCES=tuple(dict.fromkeys((*MODEL_SOURCES,
    'ilm/visual_lm/continuation_training.py','ilm/visual_lm/full_text_renderer.py',
    'ilm/visual_lm/causal_visual_prefix.py','scripts/prepare_visual_sequence_corpus.py',
    'scripts/prepare_visual_continuation_training.py','scripts/train_visual_continuation.py',
    'scripts/infer_visual_continuation.py','scripts/infer_glyph_context_motor_image.py','scripts/review_visual_continuation.py',
    'scripts/run_visual_continuation.py','tests/test_continuation_training.py',
    'references/visual_continuation_full_protocol.md')))
PUBLIC=ROOT/'references/evidence/visual_continuation_full_registration_20261112.json'


def sources():
    result={}
    for name in SOURCES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit complete pipeline before preparation: '+name)
        result[name]=sha(ROOT/name)
    return result


def exclusions():
    ceiling=load(ROOT/'artifacts/glyph_feedback_ceiling_v1/preparation.json')
    old=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    ids={Path(old['train'][r['index']]['source']).stem for r in ceiling['jobs'] if r['split']=='train'}
    refs=load(ROOT/'artifacts/compact_visual_span_evaluation_v1/references.json')['reference_rows']
    groups={r['group'] for r in refs}
    return ids,groups


def index(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not run.exists(),'New CPU-only dataset preparation')
    torch.set_num_threads(2);began=time.perf_counter();code=sources();report=load(DATA/'report.json')
    require(load(DATA/'verification.json')['report_sha256']==sha(DATA/'report.json'),'Verified corpus changed')
    bindings={rel(DATA/'report.json'):sha(DATA/'report.json'),rel(DATA/'verification.json'):sha(DATA/'verification.json')}
    for n,v in report['source_sha256'].items():require(sha(ROOT/n)==v,'Original source changed');bindings[n]=v
    for name in ['artifacts/glyph_feedback_ceiling_v1/preparation.json','artifacts/pixel_language_diversity_v1/manifest.json',
                 'artifacts/compact_visual_span_evaluation_v1/references.json']:
        bindings[name]=sha(ROOT/name)
    source={(kind,id):(group,prompt,answer) for split,kind,id,group,prompt,answer in documents() if split=='train'}
    entries=[];offset=0;excluded,groups=exclusions()
    for shard in [s for s in report['shards'] if s['split']=='train']:
        name=next(n for n in shard['files'] if n.endswith('_sources.json'));path=DATA/name
        require(sha(path)==shard['files'][name],'Source index differs')
        for local,row in enumerate(load(path)):entries.append(dict(row,corpus_index=offset+local))
        offset+=shard['records']
    require(offset==13785,'Full source split required');groups|={r['group'] for r in entries if r['id'] in excluded}
    run.mkdir(parents=True);(run/'training_inputs/fixed_cells').mkdir(parents=True);(run/'training_inputs/proportional').mkdir()
    corpus=VisualSequenceCorpus(DATA,split='train',expected_report_sha256=sha(DATA/'report.json'));rows=[];skipped=[];requests=[]
    for row in entries:
        group,prompt,answer=source[row['kind'],row['id']]
        require(group==row['group'] and text_hash(prompt+answer)==row['source_text_sha256'],'Source text/raster identity differs')
        if row['id'] in excluded or group in groups:
            skipped.append(dict(id=row['id'],reason='diagnostic source/group exclusion'));continue
        item=corpus[row['corpus_index']];ink=item['ink'];cut=item['prompt_glyphs']
        require(len(prompt)==cut and len(prompt+answer)==len(ink),'Text/pixel boundary differs')
        if row['kind']=='book':
            if len(ink)<2:skipped.append(dict(id=row['id'],reason='no nonempty book prefix and target'));continue
            cut=min(128,max(1,len(ink)//4));prompt=answer[:cut]
        require(0<cut<len(ink),'Incomplete training sequence')
        gold=ink[cut:];i=len(rows);path=run/'training_inputs/fixed_cells'/f'{i:05}.png'
        page=np.full((32*((cut+31)//32),1024),255,np.uint8)
        for j,tile in enumerate(ink[:cut]):page[(j//32)*32:(j//32+1)*32,(j%32)*32:(j%32+1)*32]=255-255*tile[0].numpy()
        Image.fromarray(page).save(path)
        flat=''.join(' ' if c.isspace() or unicodedata.category(c) in {'Cc','Cf','Cs','Zl','Zp'} else c for c in prompt)
        require(''.join(c for c in prompt if not(c.isspace() or unicodedata.category(c) in {'Cc','Cf','Cs','Zl','Zp'}))==''.join(flat.split()),
                'Visible source content changed')
        rows.append(dict(row,index=i,cut=cut,prompt_glyphs=cut,target_glyphs=len(gold),target_sha256=tensor_hash(gold),
            prompt_text_sha256=text_hash(prompt),natural_text_sha256=text_hash(flat),
            images={'fixed_cells':dict(file=rel(path),sha256=sha(path))}))
        requests.append(dict(index=i,text=flat,sha256=text_hash(flat)))
        if i%500==0:print(json.dumps(dict(indexed=i,source=row['id'])),flush=True)
    save_json(run/'index.json',dict(source_sha256=code,corpus_report_sha256=sha(DATA/'report.json'),bindings_sha256=bindings,
        rows=rows,skipped=skipped,excluded_ids=sorted(excluded),excluded_groups=sorted(groups),
        counts=dict(Counter(r['kind'] for r in rows)),total_targets=sum(r['target_glyphs'] for r in rows),
        maximum_targets=max(r['target_glyphs'] for r in rows),epochs=3,seed=SEED,elapsed_seconds=time.perf_counter()-began))
    save_json(run/'render_requests.json',dict(index_sha256=sha(run/'index.json'),requests=requests))
    print(json.dumps(dict(eligible=len(rows),skipped=len(skipped),targets=sum(r['target_glyphs'] for r in rows))))


def render(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU renderer only');code=sources();index=load(run/'index.json')
    require(code==index['source_sha256'],'Preparation sources changed')
    from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    renderer=CompleteLineRenderer();request=load(run/'render_requests.json')
    require(request['index_sha256']==sha(run/'index.json'),'Render request changed');rows=[];began=time.perf_counter()
    for job in request['requests']:
        i=job['index'];path=run/'training_inputs/proportional'/f'{i:05}.png';receipt=path.with_suffix('.json')
        if receipt.exists():
            r=load(receipt);require(r['text_sha256']==job['sha256'] and sha(path)==r['image_sha256'],'Existing rendering differs');rows.append(r);continue
        require(text_hash(job['text'])==job['sha256'],'Render text differs');fallback=None;meta=None
        try:
            _,meta=renderer._layout(job['text']);strip,check=renderer.render(job['text'],maximum_blocks=meta['blocks'])
            require(meta==check,'Font geometry changed');tensor=torch.from_numpy(strip.copy()).float()[None,None]/127.5-1
            folded=fold_writing_prompt(tensor);require(torch.equal(folded.unfold(),tensor),'Natural page lost source pixels')
            page=((folded.pixels[0,0]+1)*127.5).numpy().astype(np.uint8);ranges=folded.source_ranges
        except (ValueError,OverflowError) as error:
            fallback=repr(error)
            with Image.open(ROOT/index['rows'][i]['images']['fixed_cells']['file']) as image:page=np.asarray(image.convert('L')).copy()
            ranges=None
        Image.fromarray(page).save(path)
        r=dict(index=i,text_sha256=job['sha256'],image_sha256=sha(path),file=rel(path),metadata=meta,
               fixed_raster_fallback=fallback,wrap_ranges=ranges,complete_prompt_preserved=True)
        save_json(receipt,r);rows.append(r)
        if i%250==0:print(json.dumps(dict(rendered=i,fallback=fallback)),flush=True)
    require(sum(r['fixed_raster_fallback'] is not None for r in rows)<=.05*len(rows),'Natural rendering success below95percent')
    save_json(run/'rendering.json',dict(index_sha256=sha(run/'index.json'),source_sha256=code,renderer=renderer.provenance,
        rows=rows,elapsed_seconds=time.perf_counter()-began,all_prompts_preserved=True))


def seal(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not PUBLIC.exists(),'One CPU registration before training')
    from scripts.preflight_visual_continuation import runtime
    from scripts.compare_balanced_writer_matched import CHECKPOINT,CHECKPOINT_SHA
    code=sources();index=load(run/'index.json');rendering=load(run/'rendering.json')
    require(code==index['source_sha256']==rendering['source_sha256'] and rendering['index_sha256']==sha(run/'index.json'), 'Preparation binding differs')
    rows=index['rows'];require(len(rows)==len(rendering['rows']),'Rendering incomplete')
    for row,rendered in zip(rows,rendering['rows']):
        require(row['index']==rendered['index'] and row['natural_text_sha256']==rendered['text_sha256'],'Image source ordering differs')
        row['images']['proportional']=dict(file=rendered['file'],sha256=rendered['image_sha256'],fixed_raster_fallback=rendered['fixed_raster_fallback'])
        for image in row['images'].values():require(sha(ROOT/image['file'])==image['sha256'],'Prepared prompt differs')
    save_json(run/'training_manifest.json',dict(rows=rows,skipped=index['skipped']))
    schedule=layout_schedule(rows,3,SEED);np.save(run/'schedule.npy',schedule)
    n=len(schedule);snapshots=[0,n//4,n//2,3*n//4,n]
    require(len(set(snapshots))==5 and len(rows)>13000,'Substantial complete training and five fixed snapshots required')
    old=load(ROOT/'artifacts/balanced_writer_matched_v1/registration.json');evaluation=old['inputs']
    require(len(evaluation)==216,'Preserve complete current language cohort')
    for row in evaluation:require(sha(ROOT/row['image'])==row['sha256'],'Original evaluation pixels differ')
    value=dict(schema='visual-continuation-full-v1',source_sha256=code,runtime=runtime(),bindings_sha256=index['bindings_sha256'],
        corpus_report_sha256=index['corpus_report_sha256'],index_sha256=sha(run/'index.json'),rendering_sha256=sha(run/'rendering.json'),
        training_manifest_sha256=sha(run/'training_manifest.json'),schedule_sha256=sha(run/'schedule.npy'),
        initial_bridge=rel(CHECKPOINT),initial_bridge_sha256=CHECKPOINT_SHA,seed=SEED,rank=8,alpha=16.,
        training_cases=len(rows),counts=index['counts'],epochs=3,total_steps=n,snapshot_steps=snapshots,
        target_positions_per_epoch=index['total_targets'],scheduled_target_exposure=3*index['total_targets'],
        maximum_complete_target=index['maximum_targets'],natural_layout_fallbacks=sum(r['fixed_raster_fallback'] is not None for r in rendering['rows']),
        natural_to_fixed_exposure_ratio='two proportional and one fixed presentation of each case across three epochs',
        peak_learning_rate=.0001,learning_rate_warmup=500,learning_rate_final_fraction=.1,
        weight_decay=0.,gradient_clip_norm=1.,image_loss='existing balanced reconstruction on complete target',stop_loss_weight=.1,
        trainable_parameters=20212225,trainable_answer_layers=36,new_initial_query=True,discard_preflight_updates=True,
        checkpoint_interval=250,training_maximum_seconds=172800,training_pressure_wait_seconds=7200,
        evaluation_maximum_seconds=21600,evaluation_pressure_wait_seconds=1800,
        minimum_entry_host_ram_gib=48,minimum_runtime_host_ram_gib=24,minimum_free_gpu_gib=20,maximum_allocated_vram_gib=22,
        evaluation_inputs=evaluation,maximum_core_glyphs=128,maximum_long_book_glyphs=640,
        core_endpoint_streams=216,core_interior_streams=76,core_total_streams=660,
        long_book_initial_and_final_streams=48,parent_long_book_streams=24,
        pilot=dict(proportional_exact_minimum=12,proportional_pairs_minimum=4,proportional_readable_minimum=24,
                   long_book_coherent_complete_minimum=4,retain_union_proportional_exact=8,retain_original_fixed_exact=2,
                   each_QA_erasure_exact_maximum=2,normal_minus_best_erased_exact_minimum=8),
        exact_retention=dict(fixed_cells=['qa_09_a_normal','qa_09_b_normal'],
            proportional=['qa_04_a_normal','qa_07_a_normal','qa_09_a_normal','qa_09_b_normal','qa_10_a_normal','qa_14_b_normal','qa_15_b_normal','qa_16_b_normal']),
        full_recipe_continues_independent_of_intermediate_development_scores=True,
        untouched_confirmation_required=True,independent_human_adjudication=False,language_quality_unmeasured=True,goal_achieved=False)
    bindings=value['bindings_sha256']
    for p in [CHECKPOINT,ROOT/'artifacts/balanced_writer_matched_v1/registration.json',
              ROOT/'artifacts/compact_visual_span_evaluation_v1/preparation.json',
              ROOT/'artifacts/compact_visual_span_evaluation_v1/references.json',
              ROOT/'artifacts/glyph_tail_input_presentation_v1/evaluation/preparation.json',
              ROOT/'artifacts/glyph_tail_input_presentation_v1/evaluation/references.json',
              ROOT/'references/evidence/visual_continuation_preflight_20261112/audit.json']:
        bindings[rel(p)]=sha(p)
    save_json(run/'preparation.json',value);save_json(PUBLIC,value)
    print(json.dumps({k:value[k] for k in ['training_cases','counts','total_steps','snapshot_steps','target_positions_per_epoch','scheduled_target_exposure','maximum_complete_target','natural_layout_fallbacks']},indent=2))


def validate(run,images=False):
    r=load(run/'preparation.json');require(r['source_sha256']==sources(),'Registered pipeline source changed')
    from scripts.preflight_visual_continuation import runtime
    require(r['runtime']==runtime(),'Registered numerical runtime changed')
    public=ROOT/r.get('public_registration',rel(PUBLIC))
    require(public.read_bytes()==(run/'preparation.json').read_bytes()==subprocess.check_output(['git','show','HEAD:'+rel(public)],cwd=ROOT),
            'Commit full training registration before neural calls')
    for key,name in [('index_sha256','index.json'),('rendering_sha256','rendering.json'),('training_manifest_sha256','training_manifest.json'),('schedule_sha256','schedule.npy')]:
        require(sha(run/name)==r[key],'Prepared dataset differs: '+name)
    for name,digest in r['bindings_sha256'].items():require(sha(ROOT/name)==digest,'Bound source differs: '+name)
    if images:
        for row in load(run/'training_manifest.json')['rows']:
            for image in row['images'].values():require(sha(ROOT/image['file'])==image['sha256'],'Prepared training image differs')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['index','render','seal']);p.add_argument('--run',type=Path,default=RUN)
    a=p.parse_args();{'index':index,'render':render,'seal':seal}[a.action](a.run)
