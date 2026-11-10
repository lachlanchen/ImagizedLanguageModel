#!/usr/bin/env python3
"""Bind, physically qualify and test a completed natural writer's full answers.

Uses the reserved cohort and unchanged output-first reviewer/scorer. The first
image gate cannot by itself qualify an autonomous language model.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require

OWN='scripts/assess_natural_first_answers.py'
PROTOCOL='references/natural_answer_assay_protocol.md'
ADMISSION='references/natural_answer_assay_admission_amendment.md'
NAMES=[OWN,PROTOCOL,ADMISSION,'scripts/infer_natural_first_language.py',
       'scripts/review_glyph_tail_answers.py','scripts/assess_glyph_tail_adapter.py',
       'ilm/visual_lm/semantic_glyph_loop.py','ilm/visual_lm/glyph_runtime_resources.py',
       'ilm/visual_lm/glyph_reader_monitor.py']


def load(p):return json.loads(Path(p).read_text())
def rel(p):return str(Path(p).resolve().relative_to(ROOT))


def identity():
    result={}
    for name in NAMES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),'commit full-answer implementation first')
        result[name]=sha(ROOT/name)
    return result


def verify_registration(common):
    m=load(common/'registration.json')
    for n,d in {**m['code_sha256'],**m['bindings_sha256']}.items():
        require(sha(ROOT/n)==d,'reserved source/input changed: '+n)
    require(sha(common/'evaluation_references.json')==m['evaluation_references_sha256'],'reserved references changed')
    r=load(common/'input_review.json')
    require(r['registration_sha256']==sha(common/'registration.json') and r['all16_paired_input_boards_actually_viewed'],
            'actually inspect all reserved input pairs')
    for row in m['input_preview_boards']:require(sha(common/row['file'])==row['sha256'],'viewed input board changed')
    return m


def bind(common):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only checkpoint binding')
    m=verify_registration(common);code=identity();writer=ROOT/m['candidate_run']
    require(all((writer/name).is_file() for name in ['writer_preparation.json','fit/report.json','qualify/report.json']),
            'complete the registered natural writer fit and first-image qualification before binding answers')
    from scripts.train_natural_first_writer import verify
    w=verify(writer);fit=load(writer/'fit/report.json');q=load(writer/'qualify/report.json')
    require(load(writer/'fit/status.json')['phase']=='completed' and fit['actual_optimizer_updates']==36000
        and fit['all_checks_passed'],'complete the registered natural writer fit first')
    require(load(writer/'qualify/status.json')['phase']=='completed',
            'finish measuring the natural writer first-image results')
    review=load(writer/'qualify/visual_review.json')
    require(review['report_sha256']==sha(writer/'qualify/report.json') and review['all10_boards_actually_viewed'],
            'actually review every first-image qualification board')
    for b in q['boards']:require(sha(writer/'qualify'/b['file'])==b['sha256'],'first-image board changed')
    candidate=writer/'fit/checkpoint_latest.pt'
    require(sha(candidate)==fit['checkpoint_sha256']==q['checkpoint_sha256'],
            'natural candidate or first-image checkpoint identity changed')
    parent=ROOT/'artifacts/balanced_glyph_writer_v1';require(sha(parent/'fit/checkpoint_latest.pt')==m['initial_checkpoint_sha256'],'parent changed')
    import torch
    torch.set_num_threads(2)
    a=torch.load(candidate,weights_only=True,map_location='cpu')
    b=torch.load(parent/'fit/checkpoint_latest.pt',weights_only=True,map_location='cpu')
    require(a['writer_training_family']=='natural_first_image_v1' and a['step']==36000
        and a['new_optimizer_updates']==36000 and a['language_training_complete'],'wrong candidate family or completion')
    trained=[n for n in a['model'] if n.startswith('readout.')]
    changed=[n for n in trained if not torch.equal(a['model'][n],b['model'][n])]
    require(changed,'no trained writer tensor changed')
    require(all(torch.equal(v,b['model'][n]) for n,v in a['model'].items() if n not in trained),'frozen bridge changed')
    require(all(torch.equal(v,b['body_tail_adapters'][n]) for n,v in a['body_tail_adapters'].items()),'body adapters changed')
    steps=[int(v['step']) for v in a['optimizer']['state'].values()]
    require(len(steps)==len(trained) and set(steps)=={36000},'actual writer optimizer counters differ')
    audit=dict(checkpoint_sha256=sha(candidate),parent_sha256=sha(parent/'fit/checkpoint_latest.pt'),
        actual_AdamW_tensor_steps=steps,only_readout_changed=True,body_deltas_bit_exact=True,
        natural_first_gate_passed=q['natural_first_gate_passed'],admission_amendment_sha256=code[ADMISSION])
    audit['changed_readout_tensors']=changed
    bindings=dict(m['bindings_sha256'])
    for p in [common/'registration.json',common/'input_review.json',writer/'writer_preparation.json',
              writer/'fit/report.json',writer/'qualify/report.json',writer/'qualify/visual_review.json',
              parent/'preparation.json',parent/'cache/pools.pt',parent/'qualify/report.json',parent/'qualify/visual_review.json']:
        bindings[rel(p)]=sha(p)
    pm=load(parent/'preparation.json')
    for role,source,updates in [('natural_writer',writer,36000),('balanced_parent',parent,18000)]:
        folder=common/role;folder.mkdir(exist_ok=False)
        (folder/'fit').symlink_to((source/'fit').resolve(),target_is_directory=True)
        if role=='balanced_parent':(folder/'qualify').symlink_to((parent/'qualify').resolve(),target_is_directory=True)
        (folder/'evaluation_references.json').symlink_to((common/'evaluation_references.json').resolve())
        save_json(folder/'preparation.json',dict(primitive=pm['primitive'],genuine=pm['genuine'],
            training_jobs=pm['training_jobs'],fresh_cases=m['fresh_cases'],fit_steps=updates,
            code_sha256={**code,**w['code_sha256']},bindings_sha256=bindings,
            evaluation_references_sha256=m['evaluation_references_sha256'],condition=role,
            comparison_common=rel(common),original_pool=rel(parent/'cache/pools.pt'),
            natural_first_report=rel(writer/'qualify/report.json'),
            natural_first_gate_passed=q['natural_first_gate_passed'],
            candidate_weights_selected_by_final_step_only=True,pretraining_exposure_unknown=True))
        other='balanced_parent' if role=='natural_writer' else 'natural_writer'
        save_json(folder/'comparison_registration.json',dict(companion_run=rel(common/other),
            all_outputs_reviewed_before_reference_scoring=True))
    save_json(common/'checkpoint_binding_audit.json',audit)


def candidate(run):
    from scripts.assess_glyph_tail_adapter import candidate as original
    m,path=original(run)
    require(identity()=={n:m['code_sha256'][n] for n in NAMES},'full-answer implementation changed')
    verify_registration(ROOT/m['comparison_common'])
    for n,d in m['bindings_sha256'].items():require(sha(ROOT/n)==d,'bound comparison input changed: '+n)
    return m,path


def qualify(run):
    code=identity();m,checkpoint=candidate(run)
    require(m['condition']=='natural_writer','reuse the unchanged parent qualification instead of rerunning it')
    import torch
    from scripts.infer_natural_first_language import load_actor
    from scripts.train_semantic_glyph_language import numerics,image,atomic_tensor
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    from scripts.assess_glyph_tail_adapter import f1,boards
    folder=run/'qualify';folder.mkdir(exist_ok=False);monitor=GlyphReaderMonitor(folder,3600.);monitor.status('admitting')
    try:
        p=torch.load(ROOT/m['original_pool'],weights_only=True,map_location='cpu')
        require(p['first_labels'].tolist()==[j['labels'][0] for j in m['training_jobs']]
            and p['later_labels'].tolist()==[i for j in m['training_jobs'] for i in j['labels'][1:]],'training-state mapping changed')
        gold=torch.load(ROOT/m['primitive']/'training.pt',weights_only=True,map_location='cpu')['ink'][:,0].numpy().astype(np.uint8)
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard();actor,provenance=load_actor(checkpoint,device)
            predicted={};stops={}
            for name in ['first','later','terminal']:
                values=[];s=[]
                for start in range(0,len(p[name]),128):
                    monitor.position=dict(route=name,completed=start,total=len(p[name]));monitor.guard();h=p[name][start:start+128].to(device)
                    if name=='terminal':stop=actor.bridge.stop(h).flatten()
                    else:
                        logits,_,stop=actor.bridge.predict(h);values.append((logits>=0).to(torch.uint8)[:,0].cpu().numpy())
                    s.extend(stop.cpu().tolist())
                if values:predicted[name]=np.concatenate(values)
                stops[name]=np.array(s)
            train_records=[];train_ink=[];train_y=[];preview=[];offset=0
            selected=set(np.linspace(0,len(m['training_jobs'])-1,16,dtype=int))
            for index,job in enumerate(m['training_jobs']):
                n=len(job['labels']);actual=np.concatenate([predicted['first'][index:index+1],predicted['later'][offset:offset+n-1]])
                offset+=n-1;target=gold[job['labels']]
                train_records.extend(dict(case=index,position=i,label=y,f1=f1(a,g),exact=bool(np.array_equal(a,g))) for i,(a,g,y) in enumerate(zip(actual,target,job['labels'])))
                train_ink.extend(actual);train_y.extend(job['labels'])
                if index in selected:preview.append(dict(index=index,actual=actual,gold=target))
            require(len(train_records)==23262 and offset==20086,'full inherited training check required')
            held=[r for r in load(ROOT/m['genuine']/'preparation.json')['records'] if r['split']=='held_template']
            require(len(held)==256,'full actual reader-copy cohort required')
            copied=[];copy_ink=[];copy_y=[];copy_preview=[];selected=set(np.linspace(0,255,16,dtype=int))
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            for index,row in enumerate(held):
                monitor.position=dict(route='actual_adapted_reader_copy',completed=index,total=256);monitor.guard()
                path=ROOT/m['genuine']/row['png'];require(sha(path)==row['png_sha256'],'held copy input changed')
                args,_=actor.interface.assemble(native_budget_image(image(path)))
                h=actor.interface.actor.context(**args,use_cache=False).last_hidden_state[:,-1]
                logits,_,stop=actor.bridge.predict(h);actual=(logits>=0).to(torch.uint8)[0,0].cpu().numpy();target=gold[row['label']]
                copied.append(dict(index=index,label=row['label'],f1=f1(actual,target),exact=bool(np.array_equal(actual,target)),nonstop=bool(stop[0]<0)))
                copy_ink.append(actual);copy_y.append(row['label'])
                if index in selected:copy_preview.append(dict(index=index,actual=actual[None],gold=target[None]))
            hook.remove();train_f1=float(np.mean([r['f1'] for r in train_records]));copy_f1=float(np.mean([r['f1'] for r in copied]))
            fraction=float(np.mean([r['f1']>=.95 for r in copied]));nonstop=float(np.mean([r['nonstop'] for r in copied]))
            terminal=int((stops['terminal']>=0).sum());continues=sum(int((stops[n]<0).sum()) for n in ['first','later'])
            gate=train_f1>=.98 and copy_f1>=.95 and fraction>=.8 and nonstop>=.95 and terminal/3176>=.95 and continues/23262>=.95
            raw={}
            for name,inks,ys in [('training',train_ink,train_y),('copy_held',copy_ink,copy_y)]:
                path=folder/(name+'_outputs.pt');atomic_tensor(path,dict(ink=torch.from_numpy(np.stack(inks)[:,None]),labels=torch.tensor(ys)));raw[path.name]=sha(path)
            names=boards(folder,'training',preview)+boards(folder,'copy_held',copy_preview)
            save_json(folder/'records.json',dict(training=train_records,copy_held=copied))
            save_json(folder/'report.json',dict(code_sha256=code,checkpoint_sha256=sha(checkpoint),boards=names,
                training_mean_f1=train_f1,copy_held_mean_f1=copy_f1,copy_held_fraction095=fraction,copy_nonstop_fraction=nonstop,
                terminal_correct=terminal,terminal_cases=3176,continue_correct=continues,continue_cases=23262,
                readout_gate_passed=bool(gate),natural_first_gate_passed=m['natural_first_gate_passed'],
                raw_outputs_sha256=raw,copy_inputs_run_through_actual_adapted_full_body=True,
                training_states_cached_from_unchanged_adapted_body=True,autonomous_language_accuracy=False,
                provenance=provenance,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


def assess(run):
    code=identity();m,checkpoint=candidate(run)
    report=load(run/'qualify/report.json');review=load(run/'qualify/visual_review.json')
    require(review['report_sha256']==sha(run/'qualify/report.json') and review['all8_qualification_boards_actually_viewed'],
            'actually inspect all full physical qualification boards')
    for name in report['boards']:require(sha(run/'qualify'/name)==review['boards_sha256'][name],'viewed full qualification board changed')
    import torch
    from scripts.train_semantic_glyph_language import image,numerics,atomic_tensor
    from scripts.infer_natural_first_language import load_actor,output_png
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    old=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    controls=load(ROOT/'artifacts/semantic_glyph_language_v1/preparation.json')['jobs'][:8]
    jobs=[dict(key=f'train/{i:04}/normal',prompt=r['prompt']) for i,r in enumerate(controls)]
    for i,row in enumerate(old['development'][:16]):
        for route in ['normal','blank']:jobs.append(dict(key=f'development/{i:04}/{route}',prompt='artifacts/pixel_language_diversity_v1/'+row[route]))
    jobs.extend(dict(key=f"fresh/{r['index']:04}/normal",prompt=r['prompt']) for r in m['fresh_cases'])
    require(len(jobs)==72,'complete reserved stream cohort required')
    folder=run/'assessment';folder.mkdir(exist_ok=False);monitor=GlyphReaderMonitor(folder,1800.);monitor.status('admitting')
    try:
        records=[];monitor.guard()
        with full_reader_gpu() as (device,admission),torch.no_grad():
            monitor.admission=admission;numerics();monitor.guard();began=time.perf_counter()
            actor,provenance=load_actor(checkpoint,device);torch.cuda.synchronize();load_seconds=time.perf_counter()-began
            hook=actor.interface.actor.context.register_forward_pre_hook(lambda module,args:monitor.guard())
            for index,job in enumerate(jobs):
                monitor.position=dict(completed=index,total=72,key=job['key']);monitor.guard();path=ROOT/job['prompt'];pixels=image(path)
                torch.cuda.synchronize();began=time.perf_counter();waited=monitor.waited
                value=actor.generate(pixels,maximum_glyphs=64);torch.cuda.synchronize();elapsed=time.perf_counter()-began;waiting=monitor.waited-waited
                stem=job['key'].replace('/','_');atomic_tensor(folder/(stem+'.pt'),value);output_png(value,folder/(stem+'.png'))
                records.append(dict(**job,input_sha256=sha(path),file=stem+'.pt',sha256=sha(folder/(stem+'.pt')),
                    png=stem+'.png',png_sha256=sha(folder/(stem+'.png')),glyphs=len(value['images']),stopped=value['stopped'],capped=value['capped'],
                    generation_wall_seconds=elapsed,recorded_generation_pressure_wait_seconds=waiting,
                    generation_wall_excluding_recorded_pressure_wait_seconds=elapsed-waiting))
                monitor.status('generating_actual_complete_answers');print(json.dumps(records[-1]),flush=True)
            hook.remove();common=ROOT/m['comparison_common']
            sources=[ROOT/'artifacts/pixel_language_diversity_v1/manifest.json',ROOT/'artifacts/semantic_glyph_language_v1/preparation.json',
                     run/'preparation.json',common/'registration.json',common/'checkpoint_binding_audit.json',run/'qualify/report.json']
            save_json(folder/'report.json',dict(records=records,code_sha256=code,checkpoint_sha256=sha(checkpoint),
                case_index_sources_sha256={rel(p):sha(p) for p in sources},provenance=provenance,references_read=False,
                fixed_physical_gate_passed=report['readout_gate_passed'],assessment_does_not_override_failed_component_gate=True,
                source_lexical_head=False,tokenizer=False,font=False,glyph_bank=False,actual_emitted_pixel_feedback=True,
                quality_not_yet_reviewed=True,actor_loading_wall_seconds=load_seconds,
                generation_timing_includes_image_prefill_pixel_feedback_and_stop=True,
                generation_timing_excludes_prompt_file_read_output_serialization_and_actor_loading=True,
                matched_text_LLM_efficiency_demonstrated=False,**monitor.receipt()))
        monitor.status('completed')
    except BaseException as error:monitor.status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['bind','qualify','assess'],required=True)
    p.add_argument('--run',type=Path,default=Path('artifacts/natural_first_answer_assay_v1'))
    a=p.parse_args();globals()[a.stage](a.run.resolve())
