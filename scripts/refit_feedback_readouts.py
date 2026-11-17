#!/usr/bin/env python3
"""Direct weighted readout refitting; frozen visual features, no language fit."""
import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
SOURCE=ROOT/'artifacts/nonlinear_feedback_alignment_v1'
RUN=ROOT/'artifacts/feedback_ridge_refit_v1'
REG=ROOT/'references/evidence/feedback_ridge_refit_registration_20261118.json'
OLD=ROOT/'references/evidence/nonlinear_feedback_alignment_registration_20261118.json'
PROTOCOL=ROOT/'references/feedback_ridge_refit_protocol_20261118.md'
CPU=ROOT/'references/evidence/weighted_feedback_ridge_cpu_20261118.json'
FILES=(str(Path(__file__).relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT)),str(CPU.relative_to(ROOT)),
       'ilm/visual_lm/weighted_feedback_ridge.py','tests/test_weighted_feedback_ridge.py')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Register without CUDA')
    old=original.load(OLD);result=original.load(SOURCE/'execution/result.json');cpu=original.load(CPU)
    original.require(not result['reader_support'] and cpu['tests']==3 and cpu['exit_code']==0,'Require completed failure and CPU checks')
    for n,d in cpu['implementation_sha256'].items():original.require(original.sha(ROOT/n)==d,'CPU checked source changed')
    bound=dict(old['input_sha256'])
    paths=[OLD,SOURCE/'execution/result.json',SOURCE/'execution/frozen_queries.pt',SOURCE/'execution/preflight.json',
           ROOT/'references/evidence/nonlinear_feedback_alignment_result_20261118.json',*map(lambda n:ROOT/n,FILES)]
    paths.extend(ROOT/a['head_file'] for a in result['arms'])
    for p in paths:bound[str(p.relative_to(ROOT))]=original.sha(p)
    for n,d in bound.items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    original.write_new(REG,dict(schema='ilm-feedback-ridge-refit-v1',input_sha256=bound,packages=old['packages'],
        source=str(SOURCE.relative_to(ROOT)),initial=old['initial'],teacher=old['teacher'],
        arms=result['arms'],cases=old['cases'],baseline_tolerance=1e-6,ridge_scale=1e-4,block_rows=1024,
        normal_residual_tolerance=1e-7,order=['affine_control','nonlinear_residual'],
        resources=dict(maximum_seconds=1800,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,
                       minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,cpu_threads=2),
        reader_support=dict(final_error_maximum=.10,absolute_gain_over_refit_affine_minimum=.10,all_complete_cases_maximum=.10),
        optimizer_updates=0,features_frozen=True,full_language_fit_authorized=False,language_measured=False,
        automatic_retry=False,goal_achieved=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG),bindings=len(bound))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Require registered GPU0 environment')
    reg=original.load(REG);original.require(not RUN.exists(),'Preserve every attempt')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push sources first')
    for n in [str(REG.relative_to(ROOT)),*FILES]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Uncommitted source')
    import torch
    from torch import nn
    from torch.nn import functional as F
    from ilm.visual_lm.nonlinear_visual_feedback import NonlinearVisualFeedback,relative_feedback_error
    from ilm.visual_lm.weighted_feedback_ridge import WeightedFeedbackMoments
    from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_full_rank_visual_span import source_targets
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,1800);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External ridge deadline')))
    def guard():
        monitor.guard();original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Ridge GPU cap')
    results=[]
    try:
        monitor.status('verifying_ridge_sources')
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Package versions changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            guard();data=torch.load(SOURCE/'prepared.pt',weights_only=True,map_location='cpu')
            queries=torch.load(SOURCE/'execution/frozen_queries.pt',weights_only=True,map_location='cpu')
            x=queries[data['row_image'],data['row_slot']].to(device)
            plan=original.load(ROOT/reg['teacher']/'preparation.json');archive=open_compact_archive(ROOT.parent/'LocalLLM')
            lookup,references=source_targets(archive,plan['training_native_units']);del archive
            target=references[lookup[data['row_native_id']]].to(device).float();weights=data['row_weight'].to(device).float()
            snapshot=torch.load(ROOT/reg['initial']['deploy'],weights_only=True,map_location='cpu',mmap=True)
            mean=snapshot['image_heads']['encoder']['feedback_mean'].to(device)
            scale=snapshot['image_heads']['encoder']['feedback_scale'].to(device)
            original.require(bool((scale>0).all()),'Native affine scaling must be invertible')
            def make(arm):
                base=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,2048)).to(device)
                head=NonlinearVisualFeedback(base,hidden_width=4096) if arm['arm']=='nonlinear_residual' else base
                saved=torch.load(ROOT/arm['head_file'],weights_only=True,map_location='cpu')
                head.load_state_dict(saved['head'],strict=True)
                return head.eval().requires_grad_(False)
            def phi(head,values):
                base=head.original if isinstance(head,NonlinearVisualFeedback) else head
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    norm=base[0](values)
                    parts=[torch.ones((len(values),1),device=device,dtype=norm.dtype),norm]
                    if isinstance(head,NonlinearVisualFeedback):parts.append(F.gelu(head.expand(norm)))
                    return torch.cat(parts,dim=-1)
            def actual(head,indices=None):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    if indices is not None:
                        return float(relative_feedback_error(head(x[indices])*scale+mean,target[indices]).mean())
                    total=0.
                    for first in range(0,len(x),256):
                        errors=relative_feedback_error(head(x[first:first+256])*scale+mean,target[first:first+256])
                        total+=float((errors*weights[first:first+256]).sum())
                    return total
            def solve(head,indices,tag):
                monitor.position=dict(stage=tag,rows=len(indices));monitor.status('accumulating_weighted_normal_equations');guard()
                dimension=4481 if isinstance(head,NonlinearVisualFeedback) else 385
                moments=WeightedFeedbackMoments(dimension,2048,device)
                whole=len(indices)==len(x)
                for first in range(0,len(indices),1024):
                    guard();ix=indices[first:first+1024]
                    mass=weights[ix].double() if whole else torch.full((len(ix),),1/len(indices),device=device,dtype=torch.float64)
                    coefficient=mass/target[ix].double().square().sum(-1).clamp_min(1e-8)
                    moments.add(phi(head,x[ix]),target[ix].double()-mean.double(),coefficient)
                guard();monitor.status('solving_weighted_readout')
                coefficient,receipt=moments.solve(reg['ridge_scale']);guard()
                ideal=0.
                for first in range(0,len(indices),1024):
                    ix=indices[first:first+1024];prediction=phi(head,x[ix]).double()@coefficient+mean.double()
                    err=(prediction-target[ix].double()).square().sum(-1)/target[ix].double().square().sum(-1).clamp_min(1e-8)
                    mass=weights[ix].double() if whole else torch.full((len(ix),),1/len(indices),device=device,dtype=torch.float64)
                    ideal+=float((mass*err).sum())
                before={n:v.detach().clone() for n,v in head.state_dict().items()}
                normalized=coefficient/scale.double()[None]
                base=head.original if isinstance(head,NonlinearVisualFeedback) else head
                with torch.no_grad():
                    base[1].bias.copy_(normalized[0].float());base[1].weight.copy_(normalized[1:385].T.float())
                    if isinstance(head,NonlinearVisualFeedback):head.project.weight.copy_(normalized[385:].T.float())
                changed={'original.1.weight','original.1.bias','project.weight'} if isinstance(head,NonlinearVisualFeedback) else {'1.weight','1.bias'}
                original.require(all(torch.equal(v,before[n]) for n,v in head.state_dict().items() if n not in changed),'Feature parameters changed')
                return dict(**receipt,ideal_weighted_native_error=ideal,native_coefficient_norm=float(coefficient.norm()),features_exactly_unchanged=True)
            all_rows=torch.arange(len(x),device=device);preflight=[]
            for arm in reg['arms']:
                head=make(arm);baseline=actual(head)
                original.require(abs(baseline-arm['final']['weighted_error'])<=reg['baseline_tolerance'],'Baseline native error changed')
                saved={n:v.detach().clone() for n,v in head.state_dict().items()};before_hash=tensor_state_sha(head.state_dict())
                for row in data['cases']:
                    ix=row['rows'].to(device);old=actual(head,ix)
                    receipt=solve(head,ix,arm['arm']+'/'+row['case']['role']);new=actual(head,ix)
                    original.require(new<old,'Actual BF16 preflight did not improve')
                    head.load_state_dict(saved,strict=True);original.require(tensor_state_sha(head.state_dict())==before_hash,'Trial reset failed')
                    preflight.append(dict(arm=arm['arm'],case=row['case'],complete_native_positions=len(ix),
                        original_error=old,refit_error=new,solve=receipt,exact_reset=True))
                del head,saved;gc.collect();torch.cuda.empty_cache()
            original.write_new(RUN/'preflight.json',dict(trials=preflight,all_six_pass=True))
            for arm in reg['arms']:
                head=make(arm);receipt=solve(head,all_rows,arm['arm']+'/full_support')
                error=actual(head);cases=[dict(case=row['case'],relative_error=actual(head,row['rows'].to(device))) for row in data['cases']]
                path=RUN/(arm['arm']+'_ridge.pt');torch.save(dict(kind=arm['arm'],head=head.state_dict(),
                    hidden_width=4096,registration_sha256=original.sha(REG),readout_only_refit=True),path)
                result=dict(arm=arm['arm'],old_weighted_error=arm['final']['weighted_error'],actual_weighted_error=error,
                    complete_training_cases=cases,solve=receipt,head_file=str(path.relative_to(ROOT)),head_sha256=original.sha(path))
                results.append(result);original.write_new(RUN/(arm['arm']+'.json'),result)
                print(json.dumps(dict(arm=arm['arm'],actual_error=error,ideal_error=receipt['ideal_weighted_native_error'])),flush=True)
                del head;gc.collect();torch.cuda.empty_cache();guard()
            a,b=results;checks=dict(final_error_at_most_010=b['actual_weighted_error']<=.10,
                gain_over_refit_affine_at_least_010=a['actual_weighted_error']-b['actual_weighted_error']>=.10,
                all_complete_training_cases_at_most_010=all(v['relative_error']<=.10 for v in b['complete_training_cases']))
            result=dict(schema='ilm-feedback-ridge-refit-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                arms=results,reader_support=all(checks.values()),support_checks=checks,resources=monitor.receipt(),
                complete_rows=len(x),optimizer_updates=0,features_unchanged=True,language_measured=False,
                full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_feedback_ridge_refit',reader_support=result['reader_support'])
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(results),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_feedback_ridge_refit',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
