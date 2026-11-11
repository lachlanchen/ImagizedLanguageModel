#!/usr/bin/env python3
"""Helper-only native reading on actual glyph-actor prefills; never an ILM decoder."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.compare_glyph_tail_compact_cohort import (
    SOURCES as ACTOR_SOURCES, CHECKPOINT, CHECKPOINT_SHA, EVALUATION, load)
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt

OWN = 'scripts/diagnose_glyph_tail_native_reading.py'
PROTOCOL = 'references/glyph_tail_native_reader_protocol.md'
PRESENTATION = ROOT/'artifacts/glyph_tail_input_presentation_v1'
FIXED = ROOT/'artifacts/glyph_tail_compact_cohort_v1'
SOURCES = tuple(dict.fromkeys((*ACTOR_SOURCES, OWN, PROTOCOL,
    'scripts/check_native_visual_transfer_v71.py', 'scripts/native_visual_baseline_v70.py',
    'scripts/train_semantic_glyph_bridge.py', 'scripts/score_pixel_language_first_blocks.py',
    'scripts/prepare_compact_visual_span_evaluation.py', 'scripts/prepare_visual_sequence_corpus.py',
    'ilm/visual_lm/visual_sequence_corpus.py')))


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sources():
    values = {name: sha(ROOT/name) for name in SOURCES}
    for name, digest in values.items():
        raw = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Commit diagnostic sources before registration: '+name)
    return values


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists():
        raise ValueError('CPU registration into a new directory required')
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    from scripts.prepare_compact_visual_span_evaluation import source_records
    from scripts.train_semantic_glyph_language import CONTROLS, atomic_tensor
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    torch.set_num_threads(2)
    began = time.perf_counter(); code = sources()
    if sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Fixed adapted checkpoint changed')
    prep = load(EVALUATION/'preparation.json')
    if sha(EVALUATION/'references.json') != prep['references_sha256']:
        raise ValueError('Fixed scoring references changed')
    corpus = VisualSequenceCorpus(ROOT/'data/visual_sequence_corpus_v1', split='development',
        expected_report_sha256=prep['visual_corpus_report_sha256'])
    raw = {r.identifier:r for rows in source_records(corpus)['qa'].values() for r in rows}
    refs = [r for r in load(EVALUATION/'references.json')['reference_rows'] if r['kind']=='qa']
    if len(refs)!=32 or len({r['paired_group'] for r in refs})!=16:
        raise ValueError('Keep all current questions')
    bindings = {rel(CHECKPOINT):CHECKPOINT_SHA, rel(CONTROLS):sha(CONTROLS),
        rel(CONTROLS.parent/'compiler.json'):sha(CONTROLS.parent/'compiler.json'),
        rel(EVALUATION/'references.json'):prep['references_sha256']}
    inputs=[]; expected={}; text=[]
    for layout, run, images in [('fixed_cells',FIXED,EVALUATION/'inputs'),
                                ('proportional',PRESENTATION,PRESENTATION/'evaluation/inputs')]:
        reg=load(run/'registration.json'); report=load(run/'generation/report.json')
        if reg['checkpoint_sha256']!=CHECKPOINT_SHA or not report['all_inputs_completed']:
            raise ValueError('Complete fixed actor outputs required')
        bindings[rel(run/'registration.json')]=sha(run/'registration.json')
        bindings[rel(run/'generation/report.json')]=sha(run/'generation/report.json')
        actual={r['id']:r for r in report['records']}
        for row in reg['inputs']:
            if not row['id'].startswith('qa_'):continue
            key=layout+'__'+row['id']; image=images/row['image']; original=actual[row['id']]
            value_path=run/'generation'/row['id']/'legacy_output.pt'
            if (sha(image)!=row['sha256'] or original['input_sha256']!=row['sha256']
                    or sha(value_path)!=original['legacy_output_sha256']):
                raise ValueError('Actual image or glyph result changed')
            value=torch.load(value_path,weights_only=True,map_location='cpu')
            first=value['states'][0].clone()
            if first.shape!=(4096,) or first.dtype!=torch.float32 or not torch.isfinite(first).all():
                raise ValueError('Actual actor first state unavailable')
            if value['observer_geometry']!=original['original_observer_geometry']:
                raise ValueError('Actual geometry changed')
            expected[key]=first
            bindings[rel(image)]=row['sha256'];bindings[rel(value_path)]=original['legacy_output_sha256']
            inputs.append(dict(key=key,layout=layout,case=row['id'],condition=row['condition'],
                image=rel(image),image_sha256=row['sha256'],expected_state_sha256=tensor_hash(first),
                actual_glyph_output_sha256=original['legacy_output_sha256'],
                original_geometry=value['observer_geometry']))
    if len(inputs)!=192 or Counter(x['layout'] for x in inputs)!=Counter(fixed_cells=96,proportional=96):
        raise ValueError('Full two-presentation image coverage required')
    for reference in refs:
        source=raw[reference['source_id']]
        if source.group!=reference['group']:
            raise ValueError('Text source group changed')
        text.append(dict(key='text__'+reference['id'],case=reference['id']+'_normal',layout='text',
            condition='normal',source_id=source.identifier,source_prompt=source.prompt,
            source_prompt_sha256=hashlib.sha256(source.prompt.encode()).hexdigest()))
    output.mkdir(parents=True)
    atomic_tensor(output/'expected_prefills.pt',expected)
    jobs=[]
    for tail in ['adapted','source_zero_output']:
        for row in inputs+text:
            jobs.append(dict(row,tail=tail,id=tail+'__'+row['key']))
    if len(jobs)!=448 or len({r['id'] for r in jobs})!=448:
        raise ValueError('384 image and64 text calls required')
    save_json(output/'registration.json',dict(schema='actual-glyph-native-reading-v1',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources=code,bindings_sha256=bindings,checkpoint=rel(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,
        controls=rel(CONTROLS),compiler=rel(CONTROLS.parent/'compiler.json'),
        expected_prefills_sha256=sha(output/'expected_prefills.pt'),jobs=jobs,
        image_calls=384,text_calls=64,total_calls=448,adapted_image_prefill_checks=192,
        maximum_source_units=128,projection_chunk_rows=4096,
        projection='BF16 source head rows, FP32 chunked linear contraction, ascending source-ID tie break',
        maximum_seconds=3600,pressure_wait_allowance_seconds=600,
        minimum_entry_host_ram_gib=48,minimum_runtime_host_ram_gib=24,
        minimum_free_gpu_gib=20,maximum_allocated_vram_gib=22,
        helper_only=True,ILM_image_answers_generated=0,new_optimizer_updates=0,
        development_training_allowed=False,references_allowed_during_generation=False,
        model_calls_during_preparation=0,preparation_elapsed_seconds=time.perf_counter()-began))
    print(json.dumps(dict(image_calls=384,text_calls=64,first_states=192,model_calls=0)))


def validate(output):
    r=load(output/'registration.json')
    if r['sources']!=sources() or sha(output/'expected_prefills.pt')!=r['expected_prefills_sha256']:
        raise ValueError('Registered diagnostic source or actor states changed')
    for name,digest in r['bindings_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Registered input changed: '+name)
    if len(r['jobs'])!=448 or len({x['id'] for x in r['jobs']})!=448:
        raise ValueError('Registered coverage changed')
    return r


def verify_case(directory,row,registration_sha):
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    report=load(directory/'report.json')
    if report['job']!=row or report['registration_sha256']!=registration_sha:
        raise ValueError('Saved case identity differs')
    if sha(directory/'native_output.pt')!=report['output_sha256']:
        raise ValueError('Saved helper output changed')
    value=torch.load(directory/'native_output.pt',map_location='cpu',weights_only=True)
    for k in ['text','stopped','capped','selected_ids','selected_logits']:
        if value[k]!=report[k]:raise ValueError('Saved helper fields differ')
    if (value['hidden_states'].shape!=(len(value['selected_ids']),4096)
            or not torch.isfinite(value['hidden_states']).all()
            or not np.isfinite(value['selected_logits']).all()):
        raise ValueError('Saved native states invalid')
    if tensor_hash(value['hidden_states'][0])!=report['prefill_state_sha256']:
        raise ValueError('Saved first state differs')
    if row['layout']!='text' and row['tail']=='adapted':
        if report['prefill_state_sha256']!=row['expected_state_sha256'] or not report['actual_actor_prefill_bit_exact']:
            raise ValueError('Actual actor prefill not reproduced')
    return report


def project_source_head(state,head,chunk,guard):
    """Temporary diagnostic projection; no vocabulary module is attached to actor."""
    best=None;token=None
    for start in range(0,len(head),chunk):
        guard()
        logits=F.linear(state[:,-1].float(),head[start:start+chunk].float())
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite helper projection')
        value,index=logits[0].max(0);score=float(value)
        if best is None or score>best:best=score;token=start+int(index)
        del logits,value,index
    return token,best


def run(output):
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from ilm.visual_lm.glyph_tail_adapter import apply_tail_adapters
    from scripts.train_semantic_glyph_language import numerics,atomic_tensor
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer,embed_rows,compile_controls
    from scripts.native_visual_baseline_v70 import TASK
    from scripts.score_pixel_language_first_blocks import owner_live
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise ValueError('Original deterministic GPU0 environment required')
    r=validate(output);registration_sha=sha(output/'registration.json')
    execution=output/'execution';execution.mkdir(exist_ok=True)
    if (execution/'status.json').exists() and owner_live(load(execution/'status.json')):
        raise ValueError('Previous native reader owner still live')
    if (execution/'report.json').exists():
        saved=load(execution/'report.json')
        if saved['all448_completed']:
            for row in r['jobs']:verify_case(execution/row['id'],row,registration_sha)
            return
    attempts=execution/'attempts';attempts.mkdir(exist_ok=True)
    number=len(list(attempts.glob('attempt_*.json')))+1
    monitor=GlyphReaderMonitor(execution,r['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds=r['pressure_wait_allowance_seconds']
    loading=None;hook=None;records=[];monitor.status('admitting')
    try:
        with full_reader_gpu() as (device,admission),torch.inference_mode():
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard()
            began=time.perf_counter();actor,provenance=load_actor(ROOT/r['checkpoint'],device)
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard())
            archive,tokenizer,markers=archive_and_tokenizer()
            controls,compiler=compile_controls(archive,tokenizer,markers)
            if compiler!=load(ROOT/r['compiler']) or any(not torch.equal(v,torch.load(
                    ROOT/r['controls'],weights_only=True,map_location='cpu')[k]) for k,v in controls.items()):
                raise ValueError('Compiled numeric source controls differ')
            monitor.guard();raw=archive.omitted['output.weight']
            head=torch.from_numpy(np.array(archive.gguf.dequantize(raw.data,raw.tensor_type),copy=True)).to(device,torch.bfloat16)
            monitor.guard();loading=time.perf_counter()-began
            expected=torch.load(output/'expected_prefills.pt',weights_only=True,map_location='cpu')
            payload=torch.load(ROOT/r['checkpoint'],weights_only=True,map_location='cpu')['body_tail_adapters']
            eog=set(compiler['eog_ids']);tail=None
            for i,row in enumerate(r['jobs']):
                monitor.position=dict(completed=i,total=448,case=row['id'],decoded_steps=0)
                monitor.guard();monitor.status('native_reading')
                directory=execution/row['id']
                if directory.exists():
                    records.append(verify_case(directory,row,registration_sha));continue
                if row['tail']!=tail:
                    for n,p in actor.interface.actor.context.named_parameters():
                        if n.endswith('.up'):p.zero_()
                    if row['tail']=='adapted':apply_tail_adapters(actor.interface,payload)
                    elif any(bool(p.any()) for n,p in actor.interface.actor.context.named_parameters() if n.endswith('.up')):
                        raise ValueError('Source adapters do not have zero output')
                    tail=row['tail']
                torch.cuda.synchronize();began=time.perf_counter()
                if row['layout']=='text':
                    encode=lambda s:tokenizer.encode(s,add_special_tokens=False).ids
                    ids=[markers['im_start'],*encode('user\n'),*encode(TASK+'\n\n'+row['source_prompt']),
                         markers['im_end'],*encode('\n'),markers['im_start'],*encode('assistant\n')]
                    embedded=embed_rows(archive,ids).to(device)[None]
                    positions=torch.arange(len(ids),device=device)[None,None].expand(4,1,-1)
                    response=actor.interface.actor.context(inputs_embeds=embedded,position_ids=positions,use_cache=True)
                    geometry=dict(sequence_length=len(ids),rotary_offset=len(ids),visual_positions=0)
                    del embedded,positions
                else:
                    canvas,_=prepare_prompt(ROOT/row['image'])
                    prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                    arguments,geometry=actor.interface.assemble(native_budget_image(prompt))
                    if geometry!=row['original_geometry']:raise ValueError('Actual observer geometry differs')
                    response=actor.interface.actor.context(**arguments,use_cache=True)
                    del arguments,prompt
                state,past=response.last_hidden_state[:,-1:],response.past_key_values;del response
                first=state[0,-1].float().cpu();fingerprint=tensor_hash(first)
                matched=None
                if row['layout']!='text' and row['tail']=='adapted':
                    matched=torch.equal(first,expected[row['key']]) and fingerprint==row['expected_state_sha256']
                    if not matched:raise ValueError('Helper prefill differs from actual adapted glyph actor: '+row['id'])
                selected=[];scores=[];states=[];offset=geometry['rotary_offset'];stopped=False
                for step in range(r['maximum_source_units']):
                    states.append(state[0,-1].float().cpu())
                    token,score=project_source_head(state,head,r['projection_chunk_rows'],monitor.guard)
                    selected.append(token);scores.append(score);monitor.position['decoded_steps']=step+1
                    if token in eog:stopped=True;break
                    if step+1<r['maximum_source_units']:
                        embedded=embed_rows(archive,[token]).to(device)[None]
                        positions=torch.tensor([past.get_seq_length(),offset,offset,offset],device=device).reshape(4,1,1)
                        response=actor.interface.actor.context(inputs_embeds=embedded,position_ids=positions,
                            past_key_values=past,use_cache=True)
                        state,past=response.last_hidden_state[:,-1:],response.past_key_values;del response,embedded,positions
                        offset+=1
                answer=tokenizer.decode(selected[:-1] if stopped else selected,skip_special_tokens=False)
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                value=dict(text=answer,selected_ids=selected,selected_logits=scores,hidden_states=torch.stack(states),
                    stopped=stopped,capped=not stopped,geometry=geometry,helper_only=True)
                partial=execution/f'.{row["id"]}.{time.time_ns()}.partial';partial.mkdir()
                atomic_tensor(partial/'native_output.pt',value)
                save_json(partial/'report.json',dict(job=row,registration_sha256=registration_sha,
                    output_sha256=sha(partial/'native_output.pt'),prefill_state_sha256=fingerprint,
                    actual_actor_prefill_bit_exact=matched,elapsed_seconds=elapsed,
                    **{k:value[k] for k in ['text','selected_ids','selected_logits','stopped','capped']}))
                partial.rename(directory);records.append(verify_case(directory,row,registration_sha))
                print(json.dumps(dict(completed=len(records),case=row['id'],stopped=stopped,
                    source_units=len(selected),elapsed_seconds=elapsed),ensure_ascii=False),flush=True)
                del state,past,value,states,first;gc.collect()
            hook.remove();hook=None
            if actor.interface.actor.context.embed_tokens is not None:
                raise ValueError('Helper vocabulary entered ILM actor')
            save_json(execution/'report.json',dict(registration_sha256=registration_sha,records=records,
                all448_completed=len(records)==448,adapted_image_prefills_bit_exact=sum(
                    x['actual_actor_prefill_bit_exact'] is True for x in records),
                helper_vocabulary_not_registered_on_actor=True,ILM_image_answers_generated=0,
                new_optimizer_updates=0,development_training_allowed=False,source_provenance=provenance,
                setup_seconds=loading,attempt=number,**monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:
        if hook is not None:hook.remove()
        status=load(execution/'status.json');status['pid']=None;save_json(execution/'status.json',status)
        save_json(attempts/f'attempt_{number:04d}.json',dict(registration_sha256=registration_sha,
            attempt=number,terminal_status=status,elapsed_seconds=time.perf_counter()-monitor.began,
            resource_wait_seconds=monitor.waited,setup_seconds=loading,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else None,
            complete_case_reports=len(list(execution.glob('*/report.json')))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','run']);p.add_argument('--output',type=Path,
        default=ROOT/'artifacts/glyph_tail_native_reading_v1')
    a=p.parse_args();{'prepare':prepare,'run':run}[a.stage](a.output.resolve())
