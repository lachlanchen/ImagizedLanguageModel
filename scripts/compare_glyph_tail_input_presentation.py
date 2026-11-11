#!/usr/bin/env python3
"""Frozen glyph actor: full proportional prompts versus the fixed-cell control."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.compare_glyph_tail_compact_cohort import (
    SOURCES as ANCESTOR_SOURCES, CHECKPOINT, CHECKPOINT_SHA, EVALUATION, load, verify_case)
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt
from scripts.infer_compact_visual_span import save_grid

OWN='scripts/compare_glyph_tail_input_presentation.py'
PROTOCOL='references/glyph_tail_input_presentation_protocol.md'
SOURCES=tuple(dict.fromkeys((*ANCESTOR_SOURCES,OWN,PROTOCOL,
    'ilm/visual_lm/full_text_renderer.py','ilm/visual_lm/causal_visual_prefix.py',
    'scripts/prepare_compact_visual_span_evaluation.py','scripts/prepare_visual_sequence_corpus.py',
    'ilm/visual_lm/visual_sequence_corpus.py')))


def sources():
    result={name:sha(ROOT/name) for name in SOURCES}
    for name,digest in result.items():
        raw=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Commit registered sources first: '+name)
    return result


def paint(renderer,layout,metadata):
    surface=renderer.cairo.ImageSurface(renderer.cairo.FORMAT_RGB24,metadata['width_px'],32)
    context=renderer.cairo.Context(surface);context.set_source_rgb(1,1,1);context.paint()
    context.set_source_rgb(0,0,0);context.move_to(metadata['draw_x'],metadata['draw_y'])
    renderer.PangoCairo.show_layout(context,layout);surface.flush()
    rgba=np.frombuffer(surface.get_data(),np.uint8).reshape(32,surface.get_stride()//4,4)
    return np.where(rgba[:,:metadata['width_px'],:3].mean(-1)>=128,255,0).astype(np.uint8)


def field_masks(renderer,text,normal,metadata,ranges):
    """Color-isolated Pango fields authenticate masks; only original ink is erased."""
    masks={};length=len(text.encode())
    all_ranges=[ranges['passage'],ranges['question']]
    intervals={'passage':[ranges['passage']],'question':[ranges['question']],
        'structural':[(0,all_ranges[0][0]),(all_ranges[0][1],all_ranges[1][0]),
                      (all_ranges[1][1],length)]}
    bounds={}
    for name,keep in intervals.items():
        layout,current=renderer._layout(text)
        if current!=metadata:raise ValueError('Shaping geometry changed before field masking')
        attrs=renderer.Pango.AttrList();cursor=0
        for start,end in keep+[(length,length)]:
            if cursor<start:
                attr=renderer.Pango.attr_foreground_new(65535,65535,65535)
                attr.start_index=cursor;attr.end_index=start;attrs.insert(attr)
            cursor=end
        layout.set_attributes(attrs)
        if layout.get_pixel_size()!=renderer._layout(text)[0].get_pixel_size():
            raise ValueError('Field attributes changed logical dimensions')
        isolated=paint(renderer,layout,metadata)
        masks[name]=isolated==0
        ys,xs=np.nonzero(masks[name]);bounds[name]=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    total=sum(mask.astype(np.uint8) for mask in masks.values())
    if not np.array_equal(total,(normal==0).astype(np.uint8)):
        raise ValueError('Field masks must disjointly reconstruct every original ink pixel')
    return masks,bounds


def folded_from_ranges(strip,ranges,width):
    page=np.full((32*len(ranges),width),255,np.uint8)
    for i,(start,end) in enumerate(ranges):page[i*32:(i+1)*32,:end-start]=strip[:,start:end]
    recovered=np.concatenate([page[i*32:(i+1)*32,:end-start] for i,(start,end) in enumerate(ranges)],axis=1)
    if not np.array_equal(recovered,strip):raise ValueError('Fold lost source pixels')
    return page


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():raise ValueError('CPU preparation into a new directory only')
    from ilm.visual_lm.full_text_renderer import CompleteLineRenderer
    from ilm.visual_lm.causal_visual_prefix import fold_writing_prompt
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    from scripts.prepare_compact_visual_span_evaluation import source_records, question_parts
    began=time.perf_counter();torch.set_num_threads(2);code=sources()
    original=load(EVALUATION/'preparation.json')
    if sha(EVALUATION/'references.json')!=original['references_sha256']:raise ValueError('Original references changed')
    if sha(EVALUATION/'inputs/manifest.json')!=original['input_manifest_sha256']:raise ValueError('Original inputs changed')
    if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Earlier final actor changed')
    corpus=VisualSequenceCorpus(ROOT/'data/visual_sequence_corpus_v1',split='development',
        expected_report_sha256=original['visual_corpus_report_sha256'])
    groups=source_records(corpus)
    by_id={row.identifier:row for rows in groups['qa'].values() for row in rows}
    refs=[r for r in load(EVALUATION/'references.json')['reference_rows'] if r['kind']=='qa']
    if len(refs)!=32 or len({r['paired_group'] for r in refs})!=16:raise ValueError('Keep all paired questions')
    renderer=CompleteLineRenderer()
    old=load(ROOT/'data/general_instructions/complete_lines_train_20261105/identity.json')
    if renderer.provenance!=old['renderer']:raise ValueError('Reuse the exact earlier input renderer')
    output.mkdir(parents=True);evaluation=output/'evaluation';inputs=evaluation/'inputs';inputs.mkdir(parents=True)
    cases=[];runtime=[];bindings=[]
    for reference in refs:
        record=by_id[reference['source_id']]
        passage,question,question_start=question_parts(record)
        if record.group!=reference['group'] or question!=reference['question']:
            raise ValueError('Source passage/question differs from original registration')
        original_text=record.prompt;text=original_text.replace('\n',' ')
        if ''.join(text.split())!=''.join(original_text.split()):raise ValueError('Non-whitespace content changed')
        # No source-answer text/position/length is used to render or bound a prompt.
        layout,metadata=renderer._layout(text)
        normal,check=renderer.render(text,maximum_blocks=metadata['blocks'])
        if check!=metadata or not np.array_equal(normal,paint(renderer,layout,metadata)):
            raise ValueError('Original renderer pixels differ')
        char_ranges={'passage':(len('資料：'),len(passage)),
                     'question':(question_start,question_start+len(question))}
        ranges={key:[len(text[:a].encode()),len(text[:b].encode())] for key,(a,b) in char_ranges.items()}
        masks,bounds=field_masks(renderer,text,normal,metadata,ranges)
        tensor=torch.from_numpy(normal.copy()).float()[None,None]/127.5-1
        folded=fold_writing_prompt(tensor)
        if not torch.equal(folded.unfold(),tensor):raise ValueError('Original folding did not invert')
        rows=[list(pair) for pair in folded.source_ranges];width=folded.pixels.shape[-1]
        for condition in ['normal','question_blank','passage_blank']:
            strip=normal.copy()
            if condition!='normal':strip[masks[condition.removesuffix('_blank')]]=255
            page=folded_from_ranges(strip,rows,width)
            if condition=='normal' and not np.array_equal(page,((folded.pixels[0,0]+1)*127.5).numpy().astype(np.uint8)):
                raise ValueError('Normal page changed during serialization')
            key=reference['id']+'_'+condition;path=inputs/(key+'.png');Image.fromarray(page).save(path)
            canvas,_=prepare_prompt(path)
            if canvas.size!=(page.shape[1],page.shape[0]) or canvas.tobytes()!=page.tobytes():
                raise ValueError('CLI host preparation changes registered input pixels')
            cases.append(dict(id=key,condition=condition,image=path.name,glyphs=len(original_text),
                              image_sha256=sha(path)))
            runtime.append(dict(id=key,condition=condition,image=path.name,glyphs=len(original_text),sha256=sha(path)))
        bindings.append(dict(case=reference['id'],source_id=record.identifier,source_group=record.group,
            original_prompt_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
            flat_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
            source_characters=len(original_text),renderer=metadata,field_character_ranges=char_ranges,
            field_UTF8_byte_ranges=ranges,field_pixel_bounds=bounds,source_ranges=rows,
            source_ink_pixels=int((normal==0).sum()),field_ink_pixels={k:int(v.sum()) for k,v in masks.items()},
            normal_and_control_size=[page.shape[1],page.shape[0]],full_source_preserved=True,
            masks_disjoint_and_reconstruct_normal=True,controls_erase_only_original_ink=True,
            all_controls_use_original_wrap_ranges=True))
    if len(runtime)!=96:raise ValueError('Preserve all96 fixed QA conditions')
    save_json(inputs/'manifest.json',dict(inputs=cases))
    # Byte-identical original references retain the unchanged scoring definition.
    shutil.copyfile(EVALUATION/'references.json',evaluation/'references.json')
    save_json(evaluation/'preparation.json',dict(original,
        input_manifest_sha256=sha(inputs/'manifest.json'),normal_inputs=32,all_inputs=96,
        whole_held_out_books=[],book_continuations=0,pixel_roundtrips_exact=96,
        original_evaluation_preparation_sha256=sha(EVALUATION/'preparation.json')))
    save_json(output/'rendering_audit.json',dict(records=bindings,renderer=renderer.provenance,
        earlier_renderer_identity_sha256=sha(ROOT/'data/general_instructions/complete_lines_train_20261105/identity.json'),
        source_corpus_report_sha256=corpus.report_sha256,all96_roundtrips_verified=True,
        newline_policy='Each original newline becomes one ordinary space; every non-whitespace codepoint is unchanged',
        model_calls=0,new_optimizer_updates=0,elapsed_seconds=time.perf_counter()-began))
    registration=dict(schema='glyph-tail-input-presentation-v1',checkpoint=str(CHECKPOINT.relative_to(ROOT)),
        checkpoint_sha256=CHECKPOINT_SHA,source_sha256=code,inputs=runtime,
        preparation_sha256=sha(evaluation/'preparation.json'),input_manifest_sha256=sha(inputs/'manifest.json'),
        rendering_audit_sha256=sha(output/'rendering_audit.json'),references_sha256=sha(evaluation/'references.json'),
        original_control_registration_sha256=sha(ROOT/'artifacts/glyph_tail_compact_cohort_v1/registration.json'),
        original_control_analysis_sha256=sha(ROOT/'artifacts/glyph_tail_compact_learning_v1/summary.json'),
        maximum_glyphs=128,maximum_seconds=3600,pressure_wait_allowance_seconds=600,
        minimum_entry_host_ram_gib=48,minimum_runtime_host_ram_gib=24,
        minimum_free_gpu_gib=20,maximum_allocated_vram_gib=22,
        seed=None,flow_steps=None,device='cuda:0',deterministic_writer=True,new_optimizer_updates=0,
        retained_native_parameters=7522463984,references_allowed_during_generation=False,
        fixed_cold_replay_case='qa_01_a_normal',
        decision_rule='Increase exact QA, both-correct pairs and whole readability over2/32,1/16,3/32; retain all three original readable successes and do not increase24 normal QA caps. Development decision only.')
    save_json(output/'registration.json',registration)
    print(json.dumps(dict(registered_inputs=len(runtime),questions=len(refs),
        all_field_masks_and_content_verified=True,model_calls=0,elapsed_seconds=time.perf_counter()-began)))


def validate(output):
    r=load(output/'registration.json');inputs=output/'evaluation/inputs'
    if r['source_sha256']!=sources() or sha(ROOT/r['checkpoint'])!=r['checkpoint_sha256']:
        raise ValueError('Registered actor or source changed')
    if sha(inputs/'manifest.json')!=r['input_manifest_sha256']:raise ValueError('Input manifest changed')
    if len(r['inputs'])!=96 or len({x['id'] for x in r['inputs']})!=96:raise ValueError('Incomplete QA set')
    for row in r['inputs']:
        if Path(row['image']).name!=row['image'] or sha(inputs/row['image'])!=row['sha256']:
            raise ValueError('Input changed')
    return r

def run(output):
    registration = validate(output)
    folder = output / 'generation'
    folder.mkdir(exist_ok=True)
    request = {**registration, 'registration_sha256':sha(output / 'registration.json')}
    if (folder / 'request.json').exists():
        if load(folder / 'request.json') != request:
            raise ValueError('Resume request changed')
    else:
        save_json(folder / 'request.json',request)
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics, atomic_tensor
    from ilm.visual_lm.semantic_glyph_loop import tensor_hash
    from scripts.score_pixel_language_first_blocks import owner_live
    if (folder/'status.json').exists() and owner_live(load(folder/'status.json')):
        raise ValueError('The previous generation owner is still live')
    if (folder/'report.json').exists() and load(folder/'report.json').get('all_inputs_completed'):
        for row in registration['inputs']:
            verify_case(folder/row['id'],row)
        return
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    attempt_index=len(list(attempts.glob('attempt_*.json')))+1
    earlier_attempts={p.name:sha(p) for p in sorted(attempts.glob('attempt_*.json'))}
    loading=None
    monitor = GlyphReaderMonitor(folder,registration['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds=registration['pressure_wait_allowance_seconds']
    monitor.status('admitting')
    try:
        with full_reader_gpu() as (device,admission), torch.inference_mode():
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard()
            began=time.perf_counter()
            actor,provenance=load_actor(ROOT / registration['checkpoint'],device)
            loading=time.perf_counter()-began
            if any(p.requires_grad for p in actor.parameters()):raise ValueError('Frozen actor required')
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard())
            records=[]
            for i,row in enumerate(registration['inputs']):
                monitor.position={'completed':i,'total':len(registration['inputs']),'case':row['id']}
                monitor.guard();monitor.status('generating')
                directory=folder/row['id']
                if directory.exists():
                    records.append(verify_case(directory,row));continue
                path=output/'evaluation/inputs'/row['image']
                canvas,observed=prepare_prompt(path)
                prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                # Retain the complete original native result, including actual
                # pixels, states and feedback hashes. The actor owns generation.
                torch.cuda.synchronize();began=time.perf_counter()
                value=actor.generate(prompt,maximum_glyphs=registration['maximum_glyphs'])
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                signed=value['images'];ink=((1-signed)/2).to(torch.uint8)
                if not torch.equal(signed,1-2*ink.float()):raise ValueError('Lossy output conversion')
                partial=folder/f'.{row["id"]}.{time.time_ns()}.partial';partial.mkdir()
                atomic_tensor(partial/'legacy_output.pt',value)
                np.savez_compressed(partial/'ink.npz',ink=ink.numpy())
                save_grid(ink,partial/'answer.png')
                trace=[]
                for index,event in enumerate(value['feedback']):
                    if event['image_sha256']!=tensor_hash(signed[index:index+1]):raise ValueError('Native feedback did not use emitted pixels')
                    trace.append(dict(glyphs=1,continuous_updates=1,
                        ink_sha256=hashlib.sha256(ink[index:index+1].numpy().tobytes()).hexdigest(),
                        signed_image_sha256=event['image_sha256'],native_embedding_sha256=event['embedding_sha256']))
                result=dict(id=row['id'],input_sha256=row['sha256'],input_glyphs=row['glyphs'],
                    emitted_glyphs=len(ink),stopped=value['stopped'],capped=value['capped'],elapsed_seconds=elapsed,
                    emitted_pixel_feedback=trace,legacy_output_sha256=sha(partial/'legacy_output.pt'),
                    raw_sha256=sha(partial/'ink.npz'),image_sha256=sha(partial/'answer.png'),
                    original_observer_geometry=value['observer_geometry'],host_observation=observed,
                    actual_pixel_feedback_verified=True,reference_inputs=False,output_repaired=False)
                save_json(partial/'report.json',result);partial.rename(directory)
                records.append(verify_case(directory,row))
                print(json.dumps({k:result[k] for k in ['id','emitted_glyphs','stopped','elapsed_seconds']}),flush=True)
                del value,ink,signed;gc.collect()
            hook.remove()
            save_json(folder/'report.json',dict(request_sha256=sha(folder/'request.json'),optimizer_updates=18000,
                new_optimizer_updates=0,records=records,provenance=provenance,checkpoint_load_seconds=loading,
                total_generation_seconds=sum(r['elapsed_seconds'] for r in records),reference_inputs=False,
                output_repaired=False,all_inputs_completed=len(records)==96,
                attempt_index=attempt_index,earlier_attempts_sha256=earlier_attempts,**monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed',error=repr(error));raise
    finally:
        status=load(folder/'status.json');status['pid']=None;save_json(folder/'status.json',status)
        save_json(attempts/f'attempt_{attempt_index:04d}.json',dict(
            registration_sha256=sha(output/'registration.json'),attempt=attempt_index,
            terminal_status=status,elapsed_seconds=time.perf_counter()-monitor.began,
            resource_wait_seconds=monitor.waited,checkpoint_load_seconds=loading,
            peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else None,
            complete_case_reports=len(list(folder.glob('*/report.json'))),
            complete_report_sha256=sha(folder/'report.json') if (folder/'report.json').exists() else None))



def review(output):
    from scripts.review_compact_visual_span import review as prepare_review
    validate(output)
    prepare_review(output/'evaluation',[output/'generation'],output/'review')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','run','prepare-review'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_tail_input_presentation_v1')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'prepare-review':review}[args.phase](args.output.resolve())
