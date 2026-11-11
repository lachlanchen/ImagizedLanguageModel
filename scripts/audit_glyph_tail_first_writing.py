#!/usr/bin/env python3
"""CPU forensics of original first glyphs, coverage and supplied-image reconstruction."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.compare_glyph_tail_compact_cohort import CHECKPOINT,CHECKPOINT_SHA,EVALUATION,verify_case

OWN='scripts/audit_glyph_tail_first_writing.py'
PROTOCOL='references/glyph_tail_first_writing_protocol.md'
PRIMITIVE=ROOT/'artifacts/semantic_glyph_bridge_v1'
SOURCES=(OWN,PROTOCOL,'scripts/compare_glyph_tail_compact_cohort.py',
    'scripts/prepare_compact_visual_span_evaluation.py','scripts/prepare_visual_sequence_corpus.py',
    'ilm/visual_lm/visual_sequence_corpus.py','ilm/visual_lm/semantic_glyph_loop.py',
    'scripts/train_semantic_glyph_bridge.py')


def load(path):return json.loads(Path(path).read_text())
def rel(path):return str(Path(path).resolve().relative_to(ROOT))
def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sources():
    result={name:sha(ROOT/name) for name in SOURCES}
    for name,digest in result.items():
        raw=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Commit audit sources first: '+name)
    return result


class Monitor:
    def __init__(self,folder,seconds):
        self.folder=folder;self.seconds=seconds;self.began=time.perf_counter();self.waited=0.;self.samples=[]
    def guard(self):
        from scripts.score_pixel_language_first_blocks import host_resources
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
            raise ValueError('CPU-only audit requires CUDA hidden and uninitialized')
        while True:
            if time.perf_counter()-self.began>=self.seconds:raise RuntimeError('CPU audit wall budget exhausted')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>4:
                raise RuntimeError('CPU audit exceeds4GiB peak RSS')
            v=host_resources();self.samples.append(v)
            if v['available_gib']<24:raise RuntimeError('CPU audit host reserve reached')
            if v['some_pressure_avg10']<=.5 and v['full_pressure_avg10']<=.1:return
            remaining=min(120-self.waited,self.seconds-(time.perf_counter()-self.began))
            if remaining<=0:raise RuntimeError('CPU audit pressure wait exhausted')
            self.status('waiting_for_memory_pressure',resources=v)
            start=time.perf_counter();time.sleep(min(5.,remaining));self.waited+=time.perf_counter()-start
    def status(self,phase,**extra):
        save_json(self.folder/'status.json',dict(phase=phase,pid=os.getpid(),
            elapsed_seconds=time.perf_counter()-self.began,**extra))
    def receipt(self):
        return dict(elapsed_seconds=time.perf_counter()-self.began,resource_wait_seconds=self.waited,
            peak_RSS_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            minimum_available_gib=min(v['available_gib'] for v in self.samples),CUDA_initialized=torch.cuda.is_initialized())


def pixels_metrics(actual,gold,present=True):
    a=np.asarray(actual,dtype=bool);b=np.asarray(gold,dtype=bool)
    tp=int((a&b).sum());fp=int((a&~b).sum());fn=int((~a&b).sum());denom=2*tp+fp+fn
    return dict(tp=tp,fp=fp,fn=fn,target_ink=int(b.sum()),predicted_ink=int(a.sum()),
        pixel_exact=bool(present and np.array_equal(a,b)),
        foreground_f1=float((2*tp/denom if denom else 1.) if present else 0.))


def aggregate(rows):
    if not rows:return dict(cases=0,exact=0,mean_foreground_f1=None,aggregate_foreground_f1=None)
    tp=sum(x['tp'] for x in rows);fp=sum(x['fp'] for x in rows);fn=sum(x['fn'] for x in rows)
    return dict(cases=len(rows),exact=sum(x['pixel_exact'] for x in rows),
        mean_foreground_f1=float(np.mean([x['foreground_f1'] for x in rows])),
        aggregate_foreground_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,tp=tp,fp=fp,fn=fn)


def prepare(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():raise ValueError('New CPU audit directory required')
    from scripts.score_pixel_language_first_blocks import host_resources
    from scripts.prepare_visual_sequence_corpus import Rasterizer,FONT
    from scripts.prepare_compact_visual_span_evaluation import source_records
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    from scripts.train_semantic_glyph_bridge import atomic_tensor
    torch.set_num_threads(2);entry=host_resources()
    if entry['available_gib']<32:raise RuntimeError('CPU preparation entry reserve unavailable')
    code=sources();output.mkdir(parents=True);monitor=Monitor(output,600);monitor.status('preparing');monitor.guard()
    try:
        primitive=load(PRIMITIVE/'preparation.json');inventory=load(PRIMITIVE/'inventory.json')['records']
        training_path=PRIMITIVE/'training.pt'
        if sha(training_path)!=primitive['training_sha256'] or sha(PRIMITIVE/'inventory.json')!=primitive['inventory_sha256']:
            raise ValueError('Training primitive identity changed')
        data=torch.load(training_path,map_location='cpu',weights_only=True)
        if sha(FONT)!=primitive['font_sha256'] or primitive['font_collection_index']!=2:
            raise ValueError('Primitive font changed')
        renderer=Rasterizer();primitive_map={};primitive_pixels={};compatibility=0
        for i,row in enumerate(inventory):
            if i%128==0:monitor.guard()
            pixel=data['ink'][i].numpy().astype(np.uint8)
            primitive_pixels[hashlib.sha256(pixel.tobytes()).hexdigest()]=i
            for char in row['chars']:
                rendered=np.unpackbits(np.frombuffer(renderer.tile(char),np.uint8),bitorder='big').reshape(1,32,32)
                if not np.array_equal(rendered,pixel):raise ValueError('Source/primitive target style mismatch: '+repr(char))
                primitive_map[char]=i;compatibility+=1
        if compatibility!=primitive['anchored_characters']:raise ValueError('Primitive compatibility coverage changed')
        evaluation=load(EVALUATION/'preparation.json');corpus=VisualSequenceCorpus(ROOT/'data/visual_sequence_corpus_v1',
            split='development',expected_report_sha256=evaluation['visual_corpus_report_sha256'])
        report=load(corpus.directory/'report.json')
        if report['font_sha256']!=primitive['font_sha256'] or report['font_face']!=2:
            raise ValueError('Source corpus font differs')
        original={r.identifier:r for rows in source_records(corpus)['qa'].values() for r in rows}
        refs=[x for x in load(EVALUATION/'references.json')['reference_rows'] if x['kind']=='qa']
        if len(refs)!=32:raise ValueError('Preserve all32 source questions')
        bindings={rel(CHECKPOINT):CHECKPOINT_SHA,rel(PRIMITIVE/'preparation.json'):sha(PRIMITIVE/'preparation.json'),
            rel(PRIMITIVE/'inventory.json'):primitive['inventory_sha256'],rel(training_path):primitive['training_sha256'],
            rel(FONT) if FONT.is_relative_to(ROOT) else str(FONT):sha(FONT),
            rel(EVALUATION/'references.json'):sha(EVALUATION/'references.json'),
            rel(corpus.directory/'report.json'):corpus.report_sha256}
        if sha(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Actor checkpoint changed')
        targets=[];target_rows=[];offset=0
        for reference in refs:
            monitor.guard();source=original[reference['source_id']];record=corpus[source.index]
            if source.answer!=reference['answer'] or source.group!=reference['group']:
                raise ValueError('Canonical source answer changed')
            gold=record['ink'][record['prompt_glyphs']:].clone()
            if len(gold)!=len(source.answer) or not len(gold):raise ValueError('Target raster/text lengths differ')
            char_coverage=[];pixel_coverage=[]
            for char,pixel in zip(source.answer,gold):
                rendered=np.unpackbits(np.frombuffer(renderer.tile(char),np.uint8),bitorder='big').reshape(1,32,32)
                if not np.array_equal(rendered,pixel.numpy()):raise ValueError('Original target raster changed')
                char_coverage.append(char in primitive_map)
                pixel_coverage.append(hashlib.sha256(pixel.numpy().tobytes()).hexdigest() in primitive_pixels)
            target_rows.append(dict(case=reference['id'],source_id=source.identifier,group=source.group,
                answer=source.answer,offset=offset,glyphs=len(gold),first_character=source.answer[0],
                primitive_character_coverage=char_coverage,primitive_bitmap_coverage=pixel_coverage,
                source_glyphs_sha256=tensor_sha(gold)))
            targets.append(gold);offset+=len(gold)
        target_by={x['case']:x for x in target_rows};gold=torch.cat(targets)
        native_path=ROOT/'references/evidence/glyph_tail_native_reader_learning_20261111.json'
        native=load(native_path)
        if not native['all448_cases_authenticated']:raise ValueError('Complete matched native diagnosis required')
        native_by={(x['layout'],x['case']):x for x in native['records'] if x['tail']=='adapted' and x['layout']!='text'}
        bindings[rel(native_path)]=sha(native_path)
        jobs=[];firsts=[];states=[]
        for layout,run in [('fixed_cells',ROOT/'artifacts/glyph_tail_compact_cohort_v1'),
                           ('proportional',ROOT/'artifacts/glyph_tail_input_presentation_v1')]:
            registration=load(run/'registration.json');generation=load(run/'generation/report.json')
            if not generation['all_inputs_completed'] or registration['checkpoint_sha256']!=CHECKPOINT_SHA:
                raise ValueError('Complete unchanged glyph actor required')
            bindings[rel(run/'registration.json')]=sha(run/'registration.json')
            bindings[rel(run/'generation/report.json')]=sha(run/'generation/report.json')
            score_path=run/'review/scores.json';expected=native['glyph_control_bindings'][layout]['original_scores_sha256']
            if sha(score_path)!=expected:raise ValueError('Original complete glyph scores changed')
            bindings[rel(score_path)]=expected;scores={x['case']:x for x in load(score_path)['records'] if x['kind']=='qa'}
            for row in registration['inputs']:
                if not row['id'].startswith('qa_'):continue
                monitor.guard();directory=run/'generation'/row['id'];actual=verify_case(directory,row)
                value_path=directory/'legacy_output.pt';value=torch.load(value_path,map_location='cpu',weights_only=True)
                bindings[rel(value_path)]=actual['legacy_output_sha256']
                state=value['states'][0].clone();n=native_by[layout,row['id']]
                if tensor_sha(state)!=n['prefill_state_sha256']:raise ValueError('Actual native/glyph initial state differs')
                present=bool(len(value['images']));ink=((1-value['images'][0])/2).to(torch.uint8) if present else torch.zeros(1,32,32,dtype=torch.uint8)
                firsts.append(ink);states.append(state)
                base=row['id'].removesuffix('_'+row['condition']);reference=target_by[base];s=scores[row['id']]
                jobs.append(dict(id=layout+'__'+row['id'],case=row['id'],layout=layout,condition=row['condition'],
                    target_case=base,target_first_offset=reference['offset'],first_present=present,
                    original_first_ink_sha256=tensor_sha(ink) if present else None,
                    original_first_state_sha256=tensor_sha(state),initial_stop_probability=value['stop_probabilities'][0],
                    whole_glyph_stopped=value['stopped'],whole_glyph_capped=value['capped'],
                    emitted_glyphs=len(value['images']),native_complete_exact=n['complete_exact'],
                    glyph_complete_exact=s['complete_exact'],glyph_readable=s['readable']))
        if len(jobs)!=192 or sum(x['condition']=='normal' for x in jobs)!=64:
            raise ValueError('Preserve all192 QA streams and64 normal cases')
        atomic_tensor(output/'prepared_pixels.pt',dict(gold_ink=gold,original_first_ink=torch.stack(firsts),
            original_first_states=torch.stack(states)))
        save_json(output/'registration.json',dict(schema='glyph-tail-first-writing-v1',sources=code,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),bindings_sha256=bindings,
            checkpoint_sha256=CHECKPOINT_SHA,prepared_pixels_sha256=sha(output/'prepared_pixels.pt'),jobs=jobs,
            targets=target_rows,compatible_primitive_characters=compatibility,compatible_primitive_images=len(inventory),
            all_original_target_pixels_match_primitive_renderer=True,all_original_first_states_match_native_diagnosis=True,
            CPU_threads=2,CPU_batch_size=32,maximum_seconds=600,pressure_wait_allowance_seconds=120,
            maximum_peak_RSS_gib=4,minimum_entry_available_gib=32,minimum_runtime_available_gib=24,
            first_image_pixel_scores_not_character_or_language_accuracy=True,supplied_reconstruction_not_language=True,
            no_optimizer_updates=True,new_contextual_model_calls=0,new_autonomous_ILM_images=0,
            neural_calls_during_preparation=0,preparation_resources=monitor.receipt()))
        monitor.status('prepared');print(json.dumps(dict(cases=192,normal=64,compatible_primitives=compatibility,
            target_positions=len(gold),neural_calls=0)))
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        status=load(output/'status.json');status['pid']=None;save_json(output/'status.json',status)


def validate(output):
    r=load(output/'registration.json')
    if r['sources']!=sources() or sha(output/'prepared_pixels.pt')!=r['prepared_pixels_sha256']:
        raise ValueError('Registered audit source or pixels changed')
    for name,digest in r['bindings_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path)!=digest:raise ValueError('Bound audit input changed: '+name)
    return r


def boards(output,r,data,reconstructed):
    folder=output/'execution/boards';folder.mkdir()
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    actual={x['id']:i for i,x in enumerate(r['jobs'])};files=[]
    for first in [0,16]:
        canvas=Image.new('L',(840,80+16*84),255);draw=ImageDraw.Draw(canvas)
        for x,title in [(10,'Case'),(155,'Canonical'),(320,'Supplied AE'),(500,'Fixed original'),(710,'Proportional')]:
            draw.text((x,12),title,font=font,fill=0)
        for i,reference in enumerate(r['targets'][first:first+16]):
            key=reference['case'];offset=reference['offset'];y=55+i*84
            draw.text((10,y+20),key,font=font,fill=0)
            arrays=[data['gold_ink'][offset],reconstructed[offset],
                data['original_first_ink'][actual['fixed_cells__'+key+'_normal']],
                data['original_first_ink'][actual['proportional__'+key+'_normal']]]
            for x,array in zip([155,320,500,710],arrays):
                pixels=(255*(1-array.squeeze().numpy())).astype(np.uint8)
                canvas.paste(Image.fromarray(pixels).resize((64,64),Image.Resampling.NEAREST),(x,y))
        path=folder/f'first_glyphs_{first//16+1:02}.png';canvas.save(path);files.append(dict(file=str(path.relative_to(output/'execution')),sha256=sha(path)))
    return files


def run(output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA for CPU reconstruction')
    from ilm.visual_lm.semantic_glyph_loop import SemanticGlyphBridge
    from scripts.score_pixel_language_first_blocks import host_resources
    r=validate(output);execution=output/'execution';execution.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    monitor=Monitor(execution,r['maximum_seconds']);monitor.status('admitting');entry=host_resources()
    try:
        if entry['available_gib']<r['minimum_entry_available_gib']:raise RuntimeError('CPU run entry reserve unavailable')
        monitor.guard();payload=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        bridge=SemanticGlyphBridge().eval().requires_grad_(False);bridge.load_state_dict(payload['model']);del payload
        data=torch.load(output/'prepared_pixels.pt',map_location='cpu',weights_only=True)
        reconstructed=[]
        with torch.inference_mode():
            for first in range(0,len(data['gold_ink']),r['CPU_batch_size']):
                monitor.guard();monitor.status('supplied_image_reconstruction',completed=first,total=len(data['gold_ink']))
                signed=1-2*data['gold_ink'][first:first+r['CPU_batch_size']].float()
                logits=bridge.decode_logits(bridge.encode(signed))
                reconstructed.append((logits.sigmoid()>=.5).to(torch.uint8))
        reconstructed=torch.cat(reconstructed);monitor.guard()
        rows=[]
        for i,job in enumerate(r['jobs']):
            target=data['gold_ink'][job['target_first_offset']].numpy()
            rows.append(dict(job=job,original_first=pixels_metrics(data['original_first_ink'][i].numpy(),target,job['first_present']),
                supplied_first=pixels_metrics(reconstructed[job['target_first_offset']].numpy(),target)))
        full=[]
        for ref in r['targets']:
            for position in range(ref['glyphs']):
                i=ref['offset']+position
                full.append(dict(case=ref['case'],position=position,
                    primitive_character_covered=ref['primitive_character_coverage'][position],
                    primitive_bitmap_covered=ref['primitive_bitmap_coverage'][position],
                    **pixels_metrics(reconstructed[i].numpy(),data['gold_ink'][i].numpy())))
        aggregates=[];subgroups=[]
        for layout in ['fixed_cells','proportional']:
            for condition in ['normal','question_blank','passage_blank']:
                subset=[x for x in rows if x['job']['layout']==layout and x['job']['condition']==condition]
                if len(subset)!=32:raise ValueError('Missing a diagnostic denominator')
                aggregates.append(dict(layout=layout,condition=condition,**aggregate([x['original_first'] for x in subset])))
                if condition!='normal':continue
                for native in [False,True]:
                    for glyph in [False,True]:
                        selected=[x['original_first'] for x in subset if x['job']['native_complete_exact']==native
                            and x['job']['glyph_complete_exact']==glyph]
                        subgroups.append(dict(layout=layout,native_complete_exact=native,glyph_complete_exact=glyph,**aggregate(selected)))
        first_offsets={ref['offset'] for ref in r['targets']}
        coverage=dict(target_questions=32,first_character_covered=sum(x['primitive_character_coverage'][0] for x in r['targets']),
            first_bitmap_covered=sum(x['primitive_bitmap_coverage'][0] for x in r['targets']),
            all_target_positions=len(full),all_target_character_covered=sum(x['primitive_character_covered'] for x in full),
            all_target_bitmap_covered=sum(x['primitive_bitmap_covered'] for x in full),
            complete_answers_character_covered=sum(all(x['primitive_character_coverage']) for x in r['targets']),
            missing_characters=dict(Counter(c for x in r['targets'] for c,covered in zip(x['answer'],x['primitive_character_coverage']) if not covered)),
            continuous_decoder_not_inventory_limited_by_construction=True)
        np.savez_compressed(execution/'pixels.npz',gold_ink=data['gold_ink'].numpy(),
            original_first_ink=data['original_first_ink'].numpy(),supplied_reconstruction_ink=reconstructed.numpy())
        pictures=boards(output,r,data,reconstructed)
        save_json(execution/'report.json',dict(schema='glyph-tail-first-writing-result-v1',registration_sha256=sha(output/'registration.json'),
            raw_pixels_sha256=sha(execution/'pixels.npz'),records=rows,supplied_all_target_positions=full,
            original_first_aggregates=aggregates,outcome_subgroups=subgroups,primitive_coverage=coverage,
            supplied_first_32=aggregate([pixels_metrics(reconstructed[i].numpy(),data['gold_ink'][i].numpy()) for i in sorted(first_offsets)]),
            supplied_all_targets=aggregate(full),supplied_covered_targets=aggregate([x for x in full if x['primitive_character_covered']]),
            supplied_missing_targets=aggregate([x for x in full if not x['primitive_character_covered']]),
            initial_stop_count=sum(x['job']['initial_stop_probability']>=.5 for x in rows),boards=pictures,
            all192_original_streams_preserved=True,full_target_coverage_verified=True,
            original_first_pixels_not_replaced=True,new_contextual_model_calls=0,new_autonomous_ILM_images=0,
            CPU_reconstruction_is_answer_supplied=True,first_pixel_error_not_first_character_semantic_error=True,
            new_optimizer_updates=0,goal_achieved=False,entry_resources=entry,**monitor.receipt()))
        monitor.status('complete');print(json.dumps(dict(primitive_coverage=coverage,
            original_first_aggregates=aggregates,supplied_first_32=load(execution/'report.json')['supplied_first_32']),indent=2))
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        status=load(execution/'status.json');status['pid']=None;save_json(execution/'status.json',status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_tail_first_writing_v1')
    a=p.parse_args();{'prepare':prepare,'run':run}[a.stage](a.output.resolve())
