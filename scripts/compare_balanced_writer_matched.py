#!/usr/bin/env python3
"""Frozen balanced writer on the unchanged current216-stream language cohort."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.compare_glyph_tail_compact_cohort import SOURCES as PARENT_SOURCES,verify_case
from scripts.infer_glyph_tail_language import load_actor,prepare_prompt
from scripts.infer_compact_visual_span import save_grid
from scripts.score_pixel_language_first_blocks import require,owner_live
from scripts.train_glyph_context_motor import load,CHECKPOINT as PARENT_CHECKPOINT,CHECKPOINT_SHA as PARENT_SHA
from scripts.train_semantic_glyph_bridge import atomic_tensor

RUN=ROOT/'artifacts/balanced_writer_matched_v1'
BALANCED=ROOT/'artifacts/balanced_glyph_writer_v1'
CHECKPOINT=BALANCED/'fit/checkpoint_latest.pt'
CHECKPOINT_SHA='dcac4ee207614adca78a8b99cc4fe422b3197204d9028192bef30f111d61586d'
OLD=ROOT/'artifacts/glyph_context_motor_v1'
CAPACITY=ROOT/'artifacts/glyph_context_motor_capacity_v1'
OWN='scripts/compare_balanced_writer_matched.py'
PROTOCOL='references/balanced_writer_matched_decision.md'
SOURCES=tuple(dict.fromkeys([OWN,PROTOCOL,'scripts/assess_balanced_writer_matched.py',
    'scripts/review_balanced_writer_matched.py',*PARENT_SOURCES]))
PUBLIC=ROOT/'references/evidence/balanced_writer_matched_registration_20261111.json'


def sources():
    for name in SOURCES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit the prospective comparison first: '+name)
    return {name:sha(ROOT/name) for name in SOURCES}


def same_nested(a,b):
    if isinstance(a,torch.Tensor):
        require(isinstance(b,torch.Tensor) and torch.equal(a,b),'Frozen contextual adapter tensor differs')
        return 1
    if isinstance(a,dict):
        require(isinstance(b,dict) and a.keys()==b.keys(),'Frozen adapter map differs')
        return sum(same_nested(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        require(type(a)==type(b) and len(a)==len(b),'Frozen adapter list differs')
        return sum(same_nested(x,y) for x,y in zip(a,b))
    require(a==b,'Frozen adapter metadata differs');return 0


def register(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and not run.exists(),
            'Register once on CPU before any new model output')
    torch.set_num_threads(2);code=sources()
    require(sha(CHECKPOINT)==CHECKPOINT_SHA and sha(PARENT_CHECKPOINT)==PARENT_SHA,'Frozen model identity differs')
    fit=load(BALANCED/'fit/report.json');qualified=load(BALANCED/'qualify/report.json')
    audit=load(BALANCED/'final_checkpoint_audit.json')
    require(fit['all_checks_passed'] and fit['actual_optimizer_updates']==18000
        and qualified['readout_gate_passed'] and audit['inherited_body_adapters_bit_exact'],
        'Existing completed fit and physical qualification required')
    require(fit['checkpoint_sha256']==qualified['checkpoint_sha256']==audit['checkpoint_sha256']==CHECKPOINT_SHA,
        'Qualification does not belong to the fixed checkpoint')
    parent=torch.load(PARENT_CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    balanced=torch.load(CHECKPOINT,map_location='cpu',weights_only=True,mmap=True)
    frozen=[];changed=[]
    require(parent['model'].keys()==balanced['model'].keys(),'Bridge parameter map differs')
    for name,value in parent['model'].items():
        equal=torch.equal(value,balanced['model'][name])
        if name.startswith(('readout.','stop.')):
            if not equal:changed.append(name)
        else:
            require(equal,'Previously frozen image/normalization module changed: '+name);frozen.append(name)
    adapter_tensors=same_nested(parent['body_tail_adapters'],balanced['body_tail_adapters'])
    require(changed and all(k.startswith(('readout.','stop.')) for k in changed),'Wrong trained bundle')
    manifest=load(OLD/'evaluation_manifest.json');inputs=manifest['inputs']
    require(len(inputs)==216 and len({r['id'] for r in inputs})==216,'Keep all216 original inputs')
    require(sum(r['layout']=='fixed_cells' for r in inputs)==120 and
        sum(r['layout']=='proportional' for r in inputs)==96,'Complete both layouts')
    paths=[CHECKPOINT,PARENT_CHECKPOINT,BALANCED/'fit/report.json',BALANCED/'qualify/report.json',
        BALANCED/'final_checkpoint_audit.json',OLD/'evaluation_manifest.json',
        CAPACITY/'registration.json',CAPACITY/'prepared.pt',CAPACITY/'execution/raw.pt',CAPACITY/'execution/report.json',
        ROOT/'artifacts/glyph_tail_compact_cohort_v1/registration.json',
        ROOT/'artifacts/glyph_tail_input_presentation_v1/registration.json']
    bindings={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for row in inputs:
        require(sha(ROOT/row['image'])==row['sha256'],'Original physical input changed')
        bindings[row['image']]=row['sha256']
    value=dict(schema='balanced-writer-matched-v1',source_sha256=code,bindings_sha256=bindings,
        checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=CHECKPOINT_SHA,
        parent_checkpoint_sha256=PARENT_SHA,inputs=inputs,maximum_glyphs=128,deterministic_writer=True,
        new_optimizer_updates=0,frozen_bridge_tensor_names=frozen,changed_bundle_tensor_names=changed,
        inherited_contextual_adapter_tensor_count_verified=adapter_tensors,
        earlier_full_physical_qualification_authenticated=True,
        one_glyph_diagnosis_not_language=True,diagnosis_does_not_select_or_cancel_language_trial=True,
        primary_metrics=['complete_exact_qa_32','both_correct_pairs_16','whole_response_readability',
            'book_coherent_complete_12','stops_caps_empty','normal_minus_erased','exact_case_retention'],
        pilot=dict(proportional_exact_minimum=10,proportional_pairs_minimum=2,
            proportional_readable_minimum=16,retain_union_proportional_exact=8,retain_original_fixed_exact=2),
        inherited_visual_contextual_parameters=7522463984,
        reader_encoder_decoder_normalization_and_adapter_tensors_unchanged=True,
        readout_and_stopping_changed_as_bundle=True,source_pretraining_exposure_unknown=True,
        all216_initial_context_states_to_be_compared=True,maximum_seconds=3600,
        pressure_wait_allowance_seconds=600,minimum_entry_host_ram_gib=48,minimum_runtime_host_ram_gib=24,
        maximum_allocated_vram_gib=22,minimum_free_gpu_gib=20,
        prior_development_cohort_familiarity=True,untouched_confirmation_required=True,
        language_quality_unmeasured=True,goal_achieved=False)
    run.mkdir(parents=True);save_json(run/'registration.json',value)
    require(not PUBLIC.exists(),'Preserve earlier registration');save_json(PUBLIC,value)
    print(json.dumps(dict(registration_sha256=sha(PUBLIC),frozen_tensors=len(frozen),
        changed_bundle_tensors=len(changed),new_optimizer_updates=0,streams=216)))


def validate(run):
    r=load(run/'registration.json');require(r['source_sha256']==sources(),'Registered comparison sources changed')
    for name,digest in r['bindings_sha256'].items():require(sha(ROOT/name)==digest,'Bound input changed: '+name)
    require(PUBLIC.read_bytes()==(run/'registration.json').read_bytes()
        and subprocess.check_output(['git','show','HEAD:'+str(PUBLIC.relative_to(ROOT))],cwd=ROOT)==PUBLIC.read_bytes(),
        'Commit registration before any model call')
    return r


def run_generation(run,layout):
    r=validate(run);rows=[x for x in r['inputs'] if x['layout']==layout]
    require(len(rows)==(120 if layout=='fixed_cells' else 96),'Incomplete layout stage')
    folder=run/('generation_'+layout);folder.mkdir(exist_ok=True)
    request=dict(registration_sha256=sha(run/'registration.json'),checkpoint_sha256=CHECKPOINT_SHA,
        layout=layout,inputs=rows,maximum_glyphs=128,source_sha256=r['source_sha256'],deterministic_writer=True)
    if (folder/'request.json').exists():require(load(folder/'request.json')==request,'Resume request changed')
    else:save_json(folder/'request.json',request)
    if (folder/'status.json').exists():require(not owner_live(load(folder/'status.json')),'Generation owner is still live')
    if (folder/'report.json').exists() and load(folder/'report.json')['all_inputs_completed']:
        for row in rows:verify_case(folder/row['id'],row)
        return
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics
    attempts=folder/'attempts';attempts.mkdir(exist_ok=True);attempt=len(list(attempts.glob('attempt_*.json')))+1
    monitor=GlyphReaderMonitor(folder,3600);monitor.pressure_wait_allowance_seconds=600;monitor.status('admitting')
    loading=None
    try:
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();torch.set_num_threads(2);monitor.guard();began=time.perf_counter()
            actor,provenance=load_actor(CHECKPOINT,device)
            require(not any(p.requires_grad for p in actor.parameters()),'Frozen inference required')
            frozen=[(p,p._version) for p in actor.parameters()]
            torch.cuda.synchronize();loading=time.perf_counter()-began
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda _m,_a:monitor.guard());records=[]
            for i,row in enumerate(rows):
                monitor.position=dict(completed=i,total=len(rows),case=row['id']);monitor.guard();monitor.status('generating_balanced_matched')
                directory=folder/row['id']
                if directory.exists():records.append(verify_case(directory,row));continue
                canvas,observed=prepare_prompt(ROOT/row['image'])
                prompt=torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                torch.cuda.synchronize();began=time.perf_counter()
                with torch.inference_mode():value=actor.generate(prompt,maximum_glyphs=128)
                torch.cuda.synchronize();elapsed=time.perf_counter()-began
                signed=value['images'];ink=((1-signed)/2).to(torch.uint8)
                require(torch.equal(signed,1-2*ink.float()),'Lossy output conversion')
                partial=folder/f'.{row["id"]}.{time.time_ns()}.partial';partial.mkdir()
                atomic_tensor(partial/'legacy_output.pt',value);np.savez_compressed(partial/'ink.npz',ink=ink.numpy())
                save_grid(ink,partial/'answer.png')
                trace=[dict(glyphs=1,continuous_updates=1,
                    ink_sha256=hashlib.sha256(ink[j:j+1].numpy().tobytes()).hexdigest(),
                    signed_image_sha256=e['image_sha256'],native_embedding_sha256=e['embedding_sha256'])
                    for j,e in enumerate(value['feedback'])]
                result=dict(id=row['id'],original_case=row['original_case'],layout=layout,
                    input_sha256=row['sha256'],input_glyphs=row['glyphs'],emitted_glyphs=len(ink),
                    stopped=value['stopped'],capped=value['capped'],elapsed_seconds=elapsed,
                    emitted_pixel_feedback=trace,legacy_output_sha256=sha(partial/'legacy_output.pt'),
                    raw_sha256=sha(partial/'ink.npz'),image_sha256=sha(partial/'answer.png'),
                    original_observer_geometry=value['observer_geometry'],host_observation=observed,
                    actual_pixel_feedback_verified=True,reference_inputs=False,output_repaired=False)
                save_json(partial/'report.json',result);partial.rename(directory);records.append(verify_case(directory,row))
                require(all(p._version==v for p,v in frozen),'Frozen inference parameter changed')
                print(json.dumps({k:result[k] for k in ['id','emitted_glyphs','stopped','elapsed_seconds']}),flush=True)
                del value,signed,ink;gc.collect()
            hook.remove()
            save_json(folder/'report.json',dict(request_sha256=sha(folder/'request.json'),records=records,
                checkpoint_sha256=CHECKPOINT_SHA,all_inputs_completed=len(records)==len(rows),
                new_optimizer_updates=0,all_frozen_parameter_versions_unchanged=True,
                provenance=provenance,actor_loading_seconds=loading,
                total_generation_seconds=sum(x['elapsed_seconds'] for x in records),
                actual_pixel_feedback_verified=True,language_quality_unmeasured=True,goal_achieved=False,**monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        status=dict(load(folder/'status.json'),pid=None);save_json(folder/'status.json',status)
        save_json(attempts/f'attempt_{attempt:04}.json',dict(request_sha256=sha(folder/'request.json'),
            terminal_status=status,actor_loading_seconds=loading,**monitor.receipt()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['register','run'])
    parser.add_argument('--run',type=Path,default=RUN);parser.add_argument('--layout',choices=['fixed_cells','proportional'])
    args=parser.parse_args()
    if args.stage=='register':register(args.run.resolve())
    else:
        require(args.layout is not None,'Choose one complete layout');run_generation(args.run.resolve(),args.layout)
