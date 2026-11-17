#!/usr/bin/env python3
"""Full-support, head-only calibration of the compact reader's coordinates."""
import argparse
import copy
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/nonlinear_feedback_alignment_v1'
EXEC=RUN/'execution'
REG=ROOT/'references/evidence/nonlinear_feedback_alignment_registration_20261118.json'
PROTOCOL=ROOT/'references/nonlinear_feedback_alignment_protocol_20261118.md'
CPU=ROOT/'references/evidence/nonlinear_feedback_alignment_cpu_20261118.json'
PREVIOUS=ROOT/'references/evidence/visual_context_gradient_registration_20261117.json'
FILES=(str(Path(__file__).relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT)),
       'ilm/visual_lm/nonlinear_visual_feedback.py','tests/test_nonlinear_visual_feedback.py',str(CPU.relative_to(ROOT)))


def cpu_guard(began):
    original.require(time.perf_counter()-began<900,'Preparation deadline')
    original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Preparation RSS ceiling')


def prepare():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Prepare without CUDA')
    original.require(not RUN.exists(),'Preserve every previous attempt')
    for n in FILES:
        original.require(subprocess.check_output(['git','show','HEAD:'+n],cwd=ROOT)==(ROOT/n).read_bytes(),'Commit preparation sources first')
    import numpy as np
    import torch
    from scripts.audit_visual_feedback_identifiability import arrays
    from scripts.cache_visual_sequence_teacher_states import read_sequences,batches
    torch.set_num_threads(2)
    began=time.perf_counter();previous=original.load(PREVIOUS)
    plan_path=ROOT/previous['teacher']/'preparation.json';plan=original.load(plan_path);directory=Path(plan['data'])
    original.require(original.sha(directory/'report.json')==plan['data_report_sha256'],'Full training source changed')
    appearances=Counter();mass=defaultdict(float);images={};case_sequences={}
    cases={c['corpus_index']:c for c in previous['cases']}
    batches_total=spans_total=units_total=records_total=0
    bindings={str(plan_path.relative_to(ROOT)):original.sha(plan_path),str((directory/'report.json').relative_to(ROOT)):plan['data_report_sha256']}
    for shard in plan['shards']:
        cpu_guard(began)
        pixels=arrays(directory,shard,'_pixels.npz');alignment=arrays(directory,shard,'_teacher.npz')
        records=read_sequences(directory,shard)
        for n,d in shard['files'].items():bindings[str((directory/n).relative_to(ROOT))]=d
        for group in batches(records,1024):
            counts=Counter();units=spans=0
            for local in group:
                first,end=pixels['record_span_offsets'][local:local+2]
                glyph_start=int(pixels['record_glyph_offsets'][local]);unit_start=int(alignment['record_unit_offsets'][local])
                record_sources=[]
                for index in range(first,end):
                    glyph_end=int(pixels['span_glyph_ends'][index]);unit_end=int(alignment['span_unit_ends'][index])
                    source=tuple(map(int,alignment['unit_ids'][unit_start:unit_end]))
                    image=pixels['ink'][glyph_start:glyph_end].tobytes()
                    original.require(0<len(source)<=3 and 0<len(image)<=16*128,'Complete span outside reader support')
                    original.require(images.setdefault(source,image)==image,'Native sequence has different pixels')
                    counts[source]+=1;units+=len(source);spans+=1;record_sources.append(source)
                    glyph_start,unit_start=glyph_end,unit_end
                global_index=shard['first_record']+local
                if global_index in cases:
                    case=cases[global_index]
                    original.require(len(records[local].units)==case['native_units']
                        and int(pixels['record_glyph_offsets'][local+1]-pixels['record_glyph_offsets'][local])==case['glyphs'],
                        'Prescribed complete record changed')
                    case_sequences[global_index]=record_sources
            batches_total+=1;spans_total+=spans;units_total+=units
            for source,n in counts.items():appearances[source]+=n;mass[source]+=n/units
        records_total+=len(records)
    physical=sorted(set(images.values()),key=lambda x:(len(x),hashlib.sha256(x).digest()))
    original.require((records_total,batches_total,spans_total,units_total,len(images),len(physical))
        ==(13785,8172,5749578,5844979,29755,29740),'Complete support counts changed')
    image_index={image:i for i,image in enumerate(physical)}
    row_image=[];row_slot=[];row_unit=[];row_weight=[];source_rows={}
    for source in sorted(images):
        start=len(row_image)
        for slot,unit in enumerate(source):
            row_image.append(image_index[images[source]]);row_slot.append(slot);row_unit.append(unit)
            row_weight.append(mass[source]/batches_total)
        source_rows[source]=list(range(start,len(row_image)))
    original.require(abs(sum(row_weight)-1)<1e-10,'Full-source native mass must sum to one')
    lengths=np.array([len(v)//128 for v in physical],dtype=np.int64)
    values=dict(packed_ink=torch.from_numpy(np.frombuffer(b''.join(physical),dtype=np.uint8).copy().reshape(-1,128)),
        image_glyph_ends=torch.from_numpy(lengths.cumsum()),row_image=torch.tensor(row_image),row_slot=torch.tensor(row_slot),
        row_native_id=torch.tensor(row_unit),row_weight=torch.tensor(row_weight,dtype=torch.float64),
        cases=[dict(case=c,rows=torch.tensor([i for s in case_sequences[c['corpus_index']] for i in source_rows[s]])) for c in previous['cases']])
    RUN.mkdir();data_path=RUN/'prepared.pt';torch.save(values,data_path)
    result=dict(schema='ilm-nonlinear-feedback-preparation-v1',records=records_total,scheduled_batches=batches_total,
        span_appearances=spans_total,native_feedback_positions=units_total,native_sequences=len(images),physical_spans=len(physical),
        distinct_sequence_slot_rows=len(row_image),weight_sum=sum(row_weight),data_sha256=original.sha(data_path),bindings=bindings,
        source_code_sha256={n:original.sha(ROOT/n) for n in FILES},elapsed_seconds=time.perf_counter()-began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,neural_calls=0,cuda_initialized=torch.cuda.is_initialized())
    cpu_guard(began);original.write_new(RUN/'preparation.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['bindings','source_code_sha256']}),flush=True)


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Register with CUDA hidden')
    old=original.load(PREVIOUS);prep=original.load(RUN/'preparation.json');cpu=original.load(CPU)
    original.require(cpu['tests']==4 and cpu['exit_code']==0,'Require CPU numerical checks')
    for n,d in cpu['implementation_sha256'].items():original.require(original.sha(ROOT/n)==d,'CPU checked source changed')
    original.require(prep['source_code_sha256']=={n:original.sha(ROOT/n) for n in FILES},'Prepared sources changed')
    bound=dict(old['input_sha256']);bound.update(prep['bindings'])
    for p in [PREVIOUS,RUN/'preparation.json',RUN/'prepared.pt',*map(lambda n:ROOT/n,FILES),
              ROOT/'references/evidence/visual_feedback_identifiability_20261111.json']:
        bound[str(p.relative_to(ROOT))]=original.sha(p)
    for n,d in bound.items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    result=dict(schema='ilm-nonlinear-feedback-alignment-v1',input_sha256=bound,packages=old['packages'],
        initial=old['arms']['trainable'],codec=old['codec'],teacher=old['teacher'],cases=old['cases'],
        run=str(EXEC.relative_to(ROOT)),hidden_width=4096,branch_seed=202611184,permutation_seed=202611185,
        cache_batch=64,fit_batch=256,epochs=20,learning_rate=.0002,weight_decay=.01,clip=1.,
        order=['affine_control','nonlinear_residual'],singleton_relative_rms_tolerance=.01,
        preflight_updates_per_complete_record=2,preflight_updates_reset=True,
        resources=dict(maximum_seconds=1800,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,
            minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        reader_support=dict(final_error_maximum=.10,absolute_gain_over_control_minimum=.10,all_three_training_cases_maximum=.10),
        language_measured=False,full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False)
    original.write_new(REG,result)
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG),bindings=len(bound))),flush=True)


def raw_queries(encoder,codec,packed,ends,indices,device):
    import numpy as np
    import torch
    spans=[]
    for i in indices:
        first=0 if i==0 else int(ends[i-1]);last=int(ends[i])
        bits=np.unpackbits(packed[first:last].numpy(),bitorder='big').reshape(-1,1,32,32)
        spans.append(torch.from_numpy(bits.copy()).to(device))
    codes=codec.encode(1-torch.cat(spans).float()).float().split([len(s) for s in spans])
    captured=[];handle=encoder.feedback.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach()))
    try:encoder(codes)
    finally:handle.remove()
    original.require(len(captured)==1,'One frozen query capture required')
    return captured[0]


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Require registered GPU0 environment')
    reg=original.load(REG);original.require(not EXEC.exists(),'Preserve every execution; no automatic retry')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push sources first')
    for n in [str(REG.relative_to(ROOT)),*FILES]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Uncommitted execution source')
    import torch
    from ilm.visual_lm.compact_visual_span_model import VisualSpanEncoder
    from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec,ContinuousGlyphCodecConfig
    from ilm.visual_lm.causal_glyph_flow import load_v34_ema_codec
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.nonlinear_visual_feedback import NonlinearVisualFeedback,relative_feedback_error
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_full_rank_visual_span import source_targets
    EXEC.mkdir();monitor=GlyphReaderMonitor(EXEC,1800);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External calibration deadline')))
    def guard():
        monitor.guard();original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Head-only allocation cap')
    results=[]
    try:
        monitor.status('verifying_registered_alignment_sources')
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Source changed: '+n)
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            guard();data=torch.load(RUN/'prepared.pt',weights_only=True,map_location='cpu')
            final=torch.load(ROOT/reg['initial']['deploy'],weights_only=True,map_location='cpu',mmap=True)
            encoder=VisualSpanEncoder(768,2048,16,3).to(device).eval().requires_grad_(False)
            encoder.load_state_dict(final['image_heads']['encoder'],strict=True)
            codec=ContinuousGlyphCodec(ContinuousGlyphCodecConfig())
            load_v34_ema_codec(codec,ROOT/reg['codec']);codec=codec.to(device).eval().requires_grad_(False)
            before_encoder=tensor_state_sha(encoder.state_dict());before_codec=tensor_state_sha(codec.state_dict())
            count=len(data['image_glyph_ends']);features=torch.empty((count,3,384),dtype=torch.bfloat16)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for first in range(0,count,reg['cache_batch']):
                    monitor.position=dict(cache_first=first,total_images=count);monitor.status('caching_frozen_reader_queries');guard()
                    indices=list(range(first,min(first+reg['cache_batch'],count)))
                    features[indices]=raw_queries(encoder,codec,data['packed_ink'],data['image_glyph_ends'],indices,device).cpu()
                differences=[]
                for index in range(8):
                    singleton=raw_queries(encoder,codec,data['packed_ink'],data['image_glyph_ends'],[index],device).cpu().float()
                    cached=features[index:index+1].float()
                    difference=float((singleton-cached).square().mean().sqrt()/singleton.square().mean().sqrt().clamp_min(1e-8))
                    original.require(difference<=reg['singleton_relative_rms_tolerance'],'Singleton feature equivalence failed')
                    differences.append(difference)
            torch.save(features,EXEC/'frozen_queries.pt')
            original.require(tensor_state_sha(encoder.state_dict())==before_encoder and tensor_state_sha(codec.state_dict())==before_codec,'Frozen frontend changed')
            plan=original.load(ROOT/reg['teacher']/'preparation.json');archive=open_compact_archive(ROOT.parent/'LocalLLM')
            lookup,references=source_targets(archive,plan['training_native_units']);del archive
            select=lookup[data['row_native_id']];original.require(bool((select>=0).all()),'Unseen native source row')
            target=references[select].to(device).float();x=features[data['row_image'],data['row_slot']].to(device)
            weights=data['row_weight'].to(device).float();mean=encoder.feedback_mean.detach();scale=encoder.feedback_scale.detach()
            base=copy.deepcopy(encoder.feedback).float().requires_grad_(True)
            def make(nonlinear):
                h=copy.deepcopy(base)
                return (NonlinearVisualFeedback(h,hidden_width=4096,seed=reg['branch_seed']) if nonlinear else h).requires_grad_(True)
            def predict(head,values):return head(values)*scale+mean
            def evaluate(head):
                values=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    total=0.
                    for first in range(0,len(x),256):
                        error=relative_feedback_error(predict(head,x[first:first+256]),target[first:first+256])
                        total+=float((error*weights[first:first+256]).sum())
                    for row in data['cases']:
                        indices=row['rows'].to(device)
                        error=relative_feedback_error(predict(head,x[indices]),target[indices])
                        values.append(dict(case=row['case'],native_positions=len(indices),relative_error=float(error.mean())))
                return dict(weighted_error=total,complete_training_cases=values)
            new=make(True)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for first in range(0,len(x),256):
                    guard();original.require(torch.equal(base(x[first:first+256]),new(x[first:first+256])),'Initial function differs')
            initial=evaluate(base)
            # New branch support uses all native positions of each complete real
            # record. The actual trial weights/moments are discarded before fit.
            preflight=[]
            for row in data['cases']:
                guard();head=make(True);saved={n:v.detach().clone() for n,v in head.state_dict().items()}
                opt=torch.optim.AdamW(head.parameters(),lr=reg['learning_rate'],weight_decay=.01)
                indices=row['rows'].to(device);steps=[]
                for step in range(2):
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=relative_feedback_error(predict(head,x[indices]),target[indices]).mean()
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                    output_norm=float(head.project.weight.grad.norm());expand_norm=float(head.expand.weight.grad.norm())
                    original.require(output_norm>0 and (step==0 or expand_norm>0),'New branch did not open')
                    opt.step();steps.append(dict(update=step+1,loss=float(loss.detach()),gradient_norm=float(norm),output_gradient=output_norm,expansion_gradient=expand_norm))
                head.load_state_dict(saved,strict=True);original.require(all(torch.equal(v,saved[n]) for n,v in head.state_dict().items()),'Preflight reset failed')
                preflight.append(dict(case=row['case'],complete_native_positions=len(indices),steps=steps,exact_reset=True))
                del head,opt,saved
            original.write_new(EXEC/'preflight.json',dict(initial_functions_exact=True,singleton_relative_rms=differences,
                complete_record_trials=preflight,encoder_unchanged=True,codec_unchanged=True))
            del new;gc.collect();torch.cuda.empty_cache()
            for arm in reg['order']:
                head=make(arm=='nonlinear_residual');opt=torch.optim.AdamW(head.parameters(),lr=reg['learning_rate'],weight_decay=.01)
                history=[];updates=0
                for epoch in range(20):
                    permutation=torch.randperm(len(x),generator=torch.Generator(device='cpu').manual_seed(reg['permutation_seed']+epoch))
                    seen=0
                    for first in range(0,len(x),256):
                        if updates%8==0:
                            monitor.position=dict(arm=arm,epoch=epoch+1,rows=seen,total_rows=len(x),updates=updates)
                            monitor.status('fitting_feedback_coordinates');guard()
                        indices=permutation[first:first+256].to(device);opt.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            error=relative_feedback_error(predict(head,x[indices]),target[indices])
                            loss=(error*weights[indices]).sum()*len(x)/len(indices)
                        original.require(bool(torch.isfinite(loss)),'Nonfinite feedback objective')
                        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
                        seen+=len(indices);updates+=1
                    original.require(seen==len(x),'Epoch omitted source rows')
                    assessment=evaluate(head)
                    item=dict(epoch=epoch+1,seen_rows=seen,updates=updates,
                        permutation_sha256=hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),**assessment)
                    history.append(item);original.write_new(EXEC/f'{arm}_epoch_{epoch+1:02d}.json',item)
                    print(json.dumps(dict(arm=arm,epoch=epoch+1,weighted_error=assessment['weighted_error'],updates=updates)),flush=True)
                original.require(all(int(s['step'])==updates for s in opt.state.values()),'Actual Adam counters differ')
                path=EXEC/f'{arm}_final.pt';torch.save(dict(kind=arm,head=head.state_dict(),hidden_width=4096,
                    updates=updates,initial_reader_sha256=before_encoder,registration_sha256=original.sha(REG)),path)
                results.append(dict(arm=arm,trainable_parameters=sum(p.numel() for p in head.parameters()),epochs=20,updates=updates,
                    history=history,final=history[-1],head_sha256=original.sha(path),head_file=str(path.relative_to(ROOT))))
                del head,opt;gc.collect();torch.cuda.empty_cache();guard()
            control,candidate=results
            error=candidate['final']['weighted_error'];gain=control['final']['weighted_error']-error
            cases_pass=all(r['relative_error']<=.10 for r in candidate['final']['complete_training_cases'])
            support=error<=.10 and gain>=.10 and cases_pass
            original.require(tensor_state_sha(encoder.state_dict())==before_encoder and tensor_state_sha(codec.state_dict())==before_codec,'Original frontend changed')
            result=dict(schema='ilm-nonlinear-feedback-alignment-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                initial=initial,arms=results,preflight_sha256=original.sha(EXEC/'preflight.json'),features_sha256=original.sha(EXEC/'frozen_queries.pt'),
                unique_visual_inputs=count,source_update_rows=len(x),reader_support=support,
                support_checks=dict(final_error_at_most_010=error<=.10,gain_over_affine_at_least_010=gain>=.10,all_complete_training_cases_at_most_010=cases_pass),
                resources=monitor.receipt(),original_encoder_unchanged=True,original_codec_unchanged=True,
                original_encoder_sha256=before_encoder,original_codec_sha256=before_codec,
                language_measured=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(EXEC/'result.json',result);monitor.status('complete_feedback_alignment',reader_support=support)
    except BaseException as error:
        original.write_new(EXEC/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(results),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_feedback_alignment',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','register','run'))
    action=parser.parse_args().action
    {'prepare':prepare,'register':register,'run':run}[action]()
