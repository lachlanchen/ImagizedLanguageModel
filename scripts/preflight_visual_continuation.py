#!/usr/bin/env python3
"""Real full-reader gradient/memory preflight for continuation-only visual adapters."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from ilm.visual_lm.semantic_glyph_loop import reconstruction_loss,tensor_hash
from ilm.visual_lm.visual_continuation import VisualContinuationActor,prefill_visual_prompt
from scripts.compare_glyph_tail_compact_cohort import SOURCES as READER_SOURCES
from scripts.compare_balanced_writer_matched import CHECKPOINT,CHECKPOINT_SHA
from scripts.infer_glyph_tail_language import prepare_prompt
from scripts.train_semantic_glyph_language import CONTROLS,reader,bridge_model,numerics
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.score_pixel_language_first_blocks import require,owner_live

RUN=ROOT/'artifacts/visual_continuation_preflight_v1'
DATA=ROOT/'data/visual_sequence_corpus_v1'
PUBLIC=ROOT/'references/evidence/visual_continuation_preflight_registration_20261112.json'
SOURCES=tuple(dict.fromkeys((*READER_SOURCES,'ilm/visual_lm/visual_continuation.py',
    'scripts/compare_balanced_writer_matched.py','scripts/train_semantic_glyph_bridge.py',
    'ilm/visual_lm/visual_sequence_corpus.py','scripts/preflight_visual_continuation.py',
    'tests/test_visual_continuation.py','references/visual_continuation_preflight_protocol.md')))
SEED=2026111217


def load(path):return json.loads(Path(path).read_text())
def rel(path):return str(Path(path).resolve().relative_to(ROOT))


def sources():
    values={}
    for name in SOURCES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit preflight source first: '+name)
        values[name]=sha(ROOT/name)
    return values


def runtime():
    import transformers
    from transformers.models.qwen3_vl import modeling_qwen3_vl as native
    from transformers import cache_utils
    from torch.utils import checkpoint as cp
    return dict(torch=torch.__version__,transformers=transformers.__version__,source_sha256={
        'qwen3_vl':sha(inspect.getfile(native)), 'cache_utils':sha(inspect.getfile(cache_utils)),
        'checkpoint':sha(inspect.getfile(cp))})


def register(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),'CPU registration')
    require(not run.exists() and not PUBLIC.exists(),'Preserve prior preflight')
    torch.set_num_threads(2);code=sources();bindings={}
    def bind(p):bindings[rel(p)]=sha(p)
    for p in [DATA/'report.json',DATA/'verification.json',CHECKPOINT,CONTROLS,
        ROOT/'artifacts/glyph_feedback_ceiling_v1/preparation.json',
        ROOT/'artifacts/pixel_language_diversity_v1/manifest.json',
        ROOT/'artifacts/compact_visual_span_evaluation_v1/references.json']:
        bind(p)
    report=load(DATA/'report.json');require(load(DATA/'verification.json')['report_sha256']==sha(DATA/'report.json'),
                                            'Corpus verification differs')
    require(sha(CHECKPOINT)==CHECKPOINT_SHA,'Initial balanced bridge differs')
    ceiling=load(ROOT/'artifacts/glyph_feedback_ceiling_v1/preparation.json')
    old=load(ROOT/'artifacts/pixel_language_diversity_v1/manifest.json')
    excluded={Path(old['train'][j['index']]['source']).stem for j in ceiling['jobs'] if j['split']=='train'}
    excluded_groups={r['group'] for r in load(ROOT/'artifacts/compact_visual_span_evaluation_v1/references.json')['reference_rows']}
    entries=[];offset=0
    for shard in [s for s in report['shards'] if s['split']=='train']:
        src=DATA/next(n for n in shard['files'] if n.endswith('_sources.json'))
        pixels=DATA/next(n for n in shard['files'] if n.endswith('_pixels.npz'))
        require(sha(src)==shard['files'][src.name] and sha(pixels)==shard['files'][pixels.name], 'Training shard changed')
        rows=load(src)
        with np.load(pixels) as a:
            lengths=np.diff(a['record_glyph_offsets']);prompts=a['prompt_glyphs']
            for i,row in enumerate(rows):
                entries.append(dict(row,index=offset+i,total=int(lengths[i]),prompt=int(prompts[i]),
                                    source_file=rel(src),pixel_file=rel(pixels)))
        offset+=shard['records']
    require(offset==13785,'Complete training split required')
    excluded_groups|={x['group'] for x in entries if x['id'] in excluded}
    eligible=[x for x in entries if x['id'] not in excluded and x['group'] not in excluded_groups]
    qa=[x for x in eligible if x['kind']=='qa'];books=[x for x in eligible if x['kind']=='book' and x['total']>128]
    chosen=[('shortest_qa',max(qa,key=lambda x:(-(x['total']-x['prompt']),x['id']))),
            ('largest_qa_prompt',max(qa,key=lambda x:(x['prompt'],x['id']))),
            ('longest_qa',max(qa,key=lambda x:(x['total']-x['prompt'],x['id']))),
            ('longest_book',max(books,key=lambda x:(x['total'],x['id'])))]
    require(len({x['id'] for _,x in chosen})==4,'Four distinct complete training cases required')
    run.mkdir(parents=True);(run/'inputs').mkdir();corpus=VisualSequenceCorpus(DATA,split='train',expected_report_sha256=sha(DATA/'report.json'))
    cases=[];targets={}
    for name,row in chosen:
        item=corpus[row['index']];n=128 if row['kind']=='book' else row['prompt']
        ink=item['ink'];gold=ink[n:].clone();prompt=ink[:n]
        require(n>0 and len(gold)>0,'Nonempty complete prompt and target required')
        canvas=np.full((32*((n+31)//32),1024),255,np.uint8)
        for i,tile in enumerate(prompt):canvas[(i//32)*32:(i//32+1)*32,(i%32)*32:(i%32+1)*32]=255-255*tile[0].numpy()
        path=run/'inputs'/(name+'.png');Image.fromarray(canvas).save(path);bind(path)
        for source in [row['source_file'],row['pixel_file']]:bind(ROOT/source)
        targets[name]=gold
        cases.append(dict(name=name,source=row,prompt_image=rel(path),prompt_glyphs=n,target_glyphs=len(gold),
                          target_sha256=tensor_hash(gold),book_prefix_rule=128 if row['kind']=='book' else None))
    atomic_tensor(run/'targets.pt',targets)
    value=dict(schema='visual-continuation-preflight-v1',source_sha256=code,runtime=runtime(),bindings_sha256=bindings,
        initial_bridge=rel(CHECKPOINT),initial_bridge_sha256=CHECKPOINT_SHA,source_context_adapter_outputs_zero=True,
        targets_sha256=sha(run/'targets.pt'),cases=cases,training_records_scanned=13785,
        excluded_diagnostic_ids=sorted(excluded),excluded_source_groups=sorted(excluded_groups),
        seed=SEED,rank=8,alpha=16.,layers=36,updates=4,learning_rate=.0001,weight_decay=0.,
        native_state_tolerance=.002,maximum_seconds=1800,pressure_wait_allowance_seconds=300,
        minimum_entry_host_ram_gib=48,minimum_runtime_host_ram_gib=24,minimum_free_gpu_gib=20,
        maximum_allocated_vram_gib=22,preflight_only=True,carry_updates_into_full_fit=False,
        autonomous_language_evaluation=False,goal_achieved=False)
    save_json(run/'registration.json',value);save_json(PUBLIC,value)
    print(json.dumps(dict(registration_sha256=sha(PUBLIC),cases=[(j['name'],j['prompt_glyphs'],j['target_glyphs']) for j in cases])))


def validate(run):
    r=load(run/'registration.json');require(r['source_sha256']==sources() and r['runtime']==runtime(),'Registered code/runtime differs')
    for n,v in r['bindings_sha256'].items():require(sha(ROOT/n)==v,'Bound preflight input changed: '+n)
    require(sha(run/'targets.pt')==r['targets_sha256'],'Targets differ')
    require(PUBLIC.read_bytes()==(run/'registration.json').read_bytes()==
            subprocess.check_output(['git','show','HEAD:'+rel(PUBLIC)],cwd=ROOT),'Commit registration before GPU execution')
    return r


def prompt_tensor(path):
    canvas,_=prepare_prompt(path)
    return torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1


def fingerprints(memory,states):
    return dict(states=tensor_hash(states),keys=[tensor_hash(k) for k in memory.keys],values=[tensor_hash(v) for v in memory.values])


def native_combined(actor,prompt,embeddings):
    from ilm.visual_lm.semantic_image_memory import native_budget_image
    args,geo=actor.interface.assemble(native_budget_image(prompt));n=embeddings.shape[1];p=args['inputs_embeds'].shape[1]
    physical=torch.arange(p,p+n,device=embeddings.device);rotary=torch.arange(geo['rotary_offset'],geo['rotary_offset']+n,device=embeddings.device)
    args['inputs_embeds']=torch.cat((args['inputs_embeds'],embeddings),1)
    args['position_ids']=torch.cat((args['position_ids'],torch.stack((physical,rotary,rotary,rotary))[:,None]),-1)
    args['visual_pos_masks']=F.pad(args['visual_pos_masks'],(0,n),value=False)
    return actor.interface.actor.context(**args,use_cache=False).last_hidden_state[:,-n:]


def execute(run):
    r=validate(run);folder=run/'execution';folder.mkdir(exist_ok=False)
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    monitor=GlyphReaderMonitor(folder,r['maximum_seconds']);monitor.pressure_wait_allowance_seconds=r['pressure_wait_allowance_seconds'];monitor.status('admitting')
    body_hooks=[];layer_calls=[0]*36
    try:
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;numerics();torch.set_num_threads(2);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
            monitor.guard();began=time.perf_counter();interface,provenance=reader(device)
            bridge=bridge_model({},device,CHECKPOINT)
            actor=VisualContinuationActor(interface,bridge,rank=r['rank'],alpha=r['alpha']).train()
            bridge.readout.requires_grad_(True);bridge.stop.requires_grad_(True)
            require(len(actor.dynamics.adapters)==36 and interface.actor.context.embed_tokens is None,'Wrong complete model')
            frozen=[(p,p._version) for p in actor.parameters() if not p.requires_grad]
            initial={n:p.detach().cpu().clone() for n,p in actor.named_parameters() if p.requires_grad}
            parameters=[p for p in actor.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=r['learning_rate'],weight_decay=r['weight_decay'])
            for i,layer in enumerate(interface.actor.context.layers):
                def hook(_m,_a,i=i):layer_calls[i]+=1;monitor.guard()
                body_hooks.append(layer.register_forward_pre_hook(hook))
            torch.cuda.synchronize();loading=time.perf_counter()-began
            targets=torch.load(run/'targets.pt',weights_only=True,map_location='cpu');records=[];identity=None;zero_error=None
            for step,job in enumerate(r['cases'],1):
                monitor.position=dict(step=step,total=4,case=job['name']);monitor.guard();monitor.status('training_preflight')
                began=time.perf_counter();prompt=prompt_tensor(ROOT/job['prompt_image']);gold=1-2*targets[job['name']].float().to(device)
                require(tensor_hash(targets[job['name']])==job['target_sha256'],'Target identity differs')
                memory,source_states,geometry=prefill_visual_prompt(interface,prompt)
                if step==1:identity=fingerprints(memory,source_states)
                del source_states
                with torch.no_grad():embedded=bridge.feedback_embedding(gold)[None]
                if step==1:
                    with torch.no_grad():
                        x=actor.dynamics.embeddings(embedded)
                        split=actor.dynamics(x,memory,checkpoint_layers=False)
                        expected=native_combined(actor,prompt,x)
                        zero_error=float((split-expected).abs().max());require(zero_error<=r['native_state_tolerance'],'Native/split zero adapter differs')
                        del expected,split,x
                before_calls=list(layer_calls);optimizer.zero_grad(set_to_none=True)
                h=actor.dynamics(actor.dynamics.embeddings(embedded),memory,checkpoint_layers=True)
                logits,_,stops=bridge.predict(h[0].float())
                image_loss=reconstruction_loss(logits[:-1],gold)
                stop_loss=.5*(F.softplus(stops[:-1]).mean()+F.softplus(-stops[-1]))
                loss=image_loss+.1*stop_loss
                require(torch.isfinite(loss),'Nonfinite full preflight loss');loss.backward();monitor.guard()
                gradients={n:float(p.grad.float().norm()) for n,p in actor.named_parameters() if p.requires_grad and p.grad is not None}
                require(len(gradients)==len(initial) and all(np.isfinite(v) for v in gradients.values()),'Missing/nonfinite trainable gradient')
                for i in range(36):
                    require(gradients[f'dynamics.adapters.{i}.up']>0,'Disconnected residual up map')
                    if step>1:require(gradients[f'dynamics.adapters.{i}.down']>0,'Disconnected residual down map')
                require(gradients['dynamics.query']>0 and all(v>0 for n,v in gradients.items() if n.startswith(('bridge.readout.','bridge.stop.'))),
                        'Query or writing/stopping gradient disconnected')
                require(all(p.grad is None and p._version==version for p,version in frozen),'Frozen model changed or received gradients')
                optimizer.step();monitor.guard();torch.cuda.synchronize()
                record=dict(step=step,case=job['name'],target_glyphs=len(gold),prompt_geometry=geometry,
                    image_loss=float(image_loss),stop_loss=float(stop_loss),loss=float(loss),gradients=gradients,
                    forward_and_recomputed_layer_calls=[a-b for a,b in zip(layer_calls,before_calls)],seconds=time.perf_counter()-began)
                records.append(record);save_json(folder/'progress.json',dict(records=records,loading_seconds=loading))
                print(json.dumps({k:record[k] for k in ['step','case','target_glyphs','loss','seconds']}),flush=True)
                del h,logits,stops,loss,image_loss,stop_loss,embedded,memory,gold;gc.collect()
            changed={n:int((p.detach().cpu()!=initial[n]).sum()) for n,p in actor.named_parameters() if p.requires_grad}
            require(all(v>0 for v in changed.values()),'Some trainable tensors never changed')
            checks={};first=r['cases'][0];prompt=prompt_tensor(ROOT/first['prompt_image'])
            with torch.no_grad():
                memory,states,_=prefill_visual_prompt(interface,prompt)
                require(fingerprints(memory,states)==identity,'Source prompt changed after answer adaptation')
                del states
                signed=1-2*targets[first['name']].float().to(device);embedded=bridge.feedback_embedding(signed)[None]
                x=actor.dynamics.embeddings(embedded);parallel=actor.dynamics(x,memory,checkpoint_layers=False)
                history=None;serial=[]
                for i in range(x.shape[1]):
                    value,history=actor.dynamics(x[:,i:i+1],memory,checkpoint_layers=False,previous=history,return_history=True);serial.append(value)
                serial_error=float((torch.cat(serial,1)-parallel).abs().max())
                require(serial_error<=r['native_state_tolerance'],'Serial/parallel answer computation differs')
                checks.update(prompt_state_and_all_prompt_KV_bit_exact_after_updates=True,serial_parallel_max_abs=serial_error)
                del memory,x,parallel,history,serial,embedded,signed
                book=r['cases'][-1];prompt=prompt_tensor(ROOT/book['prompt_image']);memory,states,_=prefill_visual_prompt(interface,prompt);del states
                signed=1-2*targets[book['name']].float().to(device);embed=bridge.feedback_embedding(signed)[None]
                x=actor.dynamics.embeddings(embed);correct=actor.dynamics(x,memory,checkpoint_layers=False)
                last=x.clone();last[:,-1]=bridge.feedback_embedding(-signed[-1:])
                changed_last=actor.dynamics(last,memory,checkpoint_layers=False)
                require(torch.equal(correct[:,:-1],changed_last[:,:-1]) and not torch.equal(correct[:,-1],changed_last[:,-1]),'Last target leaked or disconnected')
                first_x=x.clone();first_x[:,1]=bridge.feedback_embedding(-signed[:1])
                changed_first=actor.dynamics(first_x,memory,checkpoint_layers=False)
                require(torch.equal(correct[:,:1],changed_first[:,:1]) and not torch.equal(correct[:,1:],changed_first[:,1:]),'Current target leaked or disconnected')
                checks.update(last_target_causal_bit_exact=True,first_target_excluded_from_first_query_bit_exact=True,
                              altered_targets_change_later_states=True)
                del memory,signed,embed,x,correct,last,changed_last,first_x,changed_first
                output=actor.generate(prompt_tensor(ROOT/first['prompt_image']),maximum_glyphs=8)
                atomic_tensor(folder/'unqualified_inference.pt',output)
            require(all(p._version==v and p.grad is None for p,v in frozen),'Frozen weights changed')
            monitor.guard();atomic_tensor(folder/'checkpoint.pt',dict(preflight_only=True,optimizer_updates=4,
                initial_bridge_sha256=CHECKPOINT_SHA,trainable_weights={n:p.detach().cpu() for n,p in actor.named_parameters() if p.requires_grad},
                optimizer=optimizer.state_dict(),seed=SEED))
            for handle in body_hooks:handle.remove()
            body_hooks=[]
            save_json(folder/'report.json',dict(schema='visual-continuation-preflight-result-v1',registration_sha256=sha(run/'registration.json'),
                source_sha256=r['source_sha256'],runtime=r['runtime'],records=records,checks=checks,
                zero_adapter_native_split_max_abs=zero_error,changed_coordinates=changed,
                trainable_parameters=sum(p.numel() for p in parameters),inherited_visual_contextual_parameters=provenance['retained_base_parameters'],
                new_residual_parameters=sum(p.numel() for p in actor.dynamics.adapters.parameters()),query_parameters=actor.dynamics.query.numel(),
                actual_optimizer_updates=4,all36_answer_layers_have_real_gradients=True,frozen_parameter_versions_unchanged=True,
                checkpoint_sha256=sha(folder/'checkpoint.pt'),unqualified_inference_sha256=sha(folder/'unqualified_inference.pt'),
                unqualified_inference_glyphs=len(output['images']),unqualified_inference_stopped=output['stopped'],
                language_quality_evaluated=False,carry_preflight_updates_into_full_fit=False,all_checks_passed=True,loading_seconds=loading,
                provenance=provenance,**monitor.receipt()))
            monitor.status('complete');print(json.dumps(dict(preflight_passed=True,checks=checks)),flush=True)
    except BaseException as error:monitor.status('failed',error=repr(error));raise
    finally:
        for handle in body_hooks:handle.remove()
        s=load(folder/'status.json');save_json(folder/'attempt.json',s);save_json(folder/'status.json',dict(s,pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','run']);p.add_argument('--run',type=Path,default=RUN)
    a=p.parse_args();(register if a.action=='register' else execute)(a.run)
