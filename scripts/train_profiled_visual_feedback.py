#!/usr/bin/env python3
"""Registered complete-record preflight and full-support variable projection."""
import argparse
import copy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
SOURCE=ROOT/'artifacts/nonlinear_feedback_alignment_v1'
RIDGE=ROOT/'artifacts/feedback_ridge_refit_v1'
OLDREG=ROOT/'references/evidence/feedback_ridge_refit_registration_20261118.json'
RUNS={s:ROOT/f'artifacts/profiled_feedback_{s}_v1' for s in ('preflight','fit')}
REGS={s:ROOT/f'references/evidence/profiled_feedback_{s}_registration_20261118.json' for s in RUNS}
CPU=ROOT/'references/evidence/profiled_feedback_cpu_20261118.json'
FILES=(str(Path(__file__).relative_to(ROOT)), 'ilm/visual_lm/profiled_visual_feedback.py',
    'tests/test_profiled_visual_feedback.py',str(CPU.relative_to(ROOT)),
    'references/profiled_feedback_preflight_protocol_20261118.md',
    'references/profiled_feedback_fit_protocol_20261118.md')


def register(stage):
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Register without CUDA')
    old=original.load(OLDREG);source_result=original.load(RIDGE/'result.json');cpu=original.load(CPU)
    original.require(cpu['tests']==4 and cpu['exit_code']==0,'Require gradient and execution CPU checks')
    for n,d in cpu['implementation_sha256'].items():original.require(original.sha(ROOT/n)==d,'CPU-qualified source changed')
    arm=next(v for v in source_result['arms'] if v['arm']=='nonlinear_residual')
    paths=[OLDREG,RIDGE/'result.json',ROOT/arm['head_file'],
        ROOT/'references/evidence/feedback_ridge_refit_result_20261118.json',
        ROOT/'references/evidence/frozen_query_identifiability_20261118.json',*map(lambda n:ROOT/n,FILES)]
    bound=dict(old['input_sha256'])
    if stage=='fit':
        pre=original.load(RUNS['preflight']/'result.json')
        original.require(pre['preflight_passed'] and pre['all_trial_states_reset'],'Passing actual preflight required')
        pre_reg=original.load(REGS['preflight'])
        for n,d in pre_reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Preflight source changed')
        paths.extend([REGS['preflight'],RUNS['preflight']/'result.json'])
    for p in paths:bound[str(p.relative_to(ROOT))]=original.sha(p)
    for n,d in bound.items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    original.write_new(REGS[stage],dict(schema='ilm-profiled-feedback-'+stage+'-v1',stage=stage,
        input_sha256=bound,packages=old['packages'],initial=old['initial'],teacher=old['teacher'],
        source_head=arm['head_file'],source_head_sha256=arm['head_sha256'],
        legacy_bf16_error=arm['actual_weighted_error'],legacy_replay_tolerance=1e-6,cases=old['cases'],
        precision='FP32 feedback features/weights; FP64 native output solve and objective',
        initial_ridge_scale=1e-4,fixed_ridge_after_initialization=True,outer_steps=12,
        relative_proposal=.02,armijo=1e-4,maximum_search_trials=6,gradient_block_rows=256,solve_block_rows=1024,
        finite_difference_relative_displacement=.001,finite_difference_relative_tolerance=.05,
        finite_difference_absolute_tolerance=1e-8,preflight_actual_error_allowance=.0001,
        resources=dict(maximum_seconds=1800,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,
            minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        reader_support=dict(final_actual_error_maximum=.10,gain_over_control_minimum=.10,each_complete_record_maximum=.10),
        feature_parameters=['expand.weight','expand.bias'],all_other_feature_parameters_frozen=True,
        source_query_slot_representation_unchanged=True,language_measured=False,
        full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(stage=stage,registration=str(REGS[stage].relative_to(ROOT)),
        sha256=original.sha(REGS[stage]),bindings=len(bound))),flush=True)


class Experiment:
    def __init__(self,reg,device,monitor,directory):
        import torch
        from ilm.visual_lm.compact_visual_span_weights import open_compact_archive
        from scripts.train_full_rank_visual_span import source_targets
        self.reg,self.device,self.monitor,self.directory=reg,device,monitor,directory
        self.data=torch.load(SOURCE/'prepared.pt',weights_only=True,map_location='cpu')
        queries=torch.load(SOURCE/'execution/frozen_queries.pt',weights_only=True,map_location='cpu')
        self.x=queries[self.data['row_image'],self.data['row_slot']].to(device)
        plan=original.load(ROOT/reg['teacher']/'preparation.json');archive=open_compact_archive(ROOT.parent/'LocalLLM')
        lookup,references=source_targets(archive,plan['training_native_units']);del archive
        select=lookup[self.data['row_native_id']];original.require(bool((select>=0).all()),'Native target support changed')
        self.target=references[select].to(device).float();self.weights=self.data['row_weight'].to(device).float()
        snapshot=torch.load(ROOT/reg['initial']['deploy'],weights_only=True,map_location='cpu',mmap=True)
        self.mean=snapshot['image_heads']['encoder']['feedback_mean'].to(device)
        self.scale=snapshot['image_heads']['encoder']['feedback_scale'].to(device)
        original.require(bool((self.scale>0).all()),'Native scale must be invertible')
        self.source=torch.load(ROOT/reg['source_head'],weights_only=True,map_location='cpu')['head']
        self.rows=torch.arange(len(self.x),device=device)

    def guard(self):
        import torch
        self.monitor.guard()
        original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Profiled feedback allocation cap')

    def make(self):
        import torch
        from torch import nn
        from ilm.visual_lm.profiled_visual_feedback import Float32VisualFeedback
        base=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,2048)).to(self.device)
        head=Float32VisualFeedback(base,hidden_width=4096)
        head.load_state_dict(self.source,strict=True)
        return head.eval().requires_grad_(False)

    def actual(self,head,indices,mass,legacy=False):
        import torch
        from ilm.visual_lm.nonlinear_visual_feedback import NonlinearVisualFeedback,relative_feedback_error
        total=0.
        with torch.no_grad():
            for first in range(0,len(indices),256):
                ix=indices[first:first+256]
                if legacy:
                    with torch.autocast('cuda',dtype=torch.bfloat16):prediction=NonlinearVisualFeedback.forward(head,self.x[ix])
                else:prediction=head(self.x[ix])
                error=relative_feedback_error(prediction*self.scale+self.mean,self.target[ix])
                total+=float((error*mass[first:first+len(ix)]).sum())
        return total

    def solve(self,head,indices,mass,ridge=None,label='solve'):
        import torch
        from ilm.visual_lm.profiled_visual_feedback import FixedRidgeMoments,feedback_features,install_native_readout,native_partial_loss
        self.monitor.position=dict(stage=label,rows=len(indices));self.monitor.status('accumulating_profiled_moments');self.guard()
        moments=FixedRidgeMoments(4481,2048,self.device)
        with torch.no_grad():
            for first in range(0,len(indices),1024):
                self.guard();ix=indices[first:first+1024]
                c=mass[first:first+len(ix)].double()/self.target[ix].double().square().sum(-1).clamp_min(1e-8)
                moments.add(feedback_features(head,self.x[ix]),self.target[ix].double()-self.mean.double(),c)
            self.guard();self.monitor.status('solving_profiled_readout')
            coefficient,receipt=(moments.solve(self.reg['initial_ridge_scale']) if ridge is None else moments.solve_fixed(ridge))
            ridge=receipt['ridge'];error=0.
            for first in range(0,len(indices),1024):
                ix=indices[first:first+1024]
                c=mass[first:first+len(ix)].double()/self.target[ix].double().square().sum(-1).clamp_min(1e-8)
                error+=float(native_partial_loss(feedback_features(head,self.x[ix]),coefficient,
                    self.target[ix].double()-self.mean.double(),c))
            penalty=ridge*float(coefficient.square().sum())
            install_native_readout(head,coefficient,self.scale)
        actual=self.actual(head,indices,mass)
        return coefficient,dict(**receipt,ideal_native_error=error,ridge_penalty=penalty,
            profiled_objective=error+penalty,actual_native_error=actual)

    def gradient(self,head,coefficient,indices,mass,label):
        import torch
        from ilm.visual_lm.profiled_visual_feedback import feedback_features,native_partial_loss
        head.zero_grad(set_to_none=True);head.expand.requires_grad_(True)
        for first in range(0,len(indices),256):
            if first%2048==0:
                self.monitor.position=dict(stage=label,gradient_rows=first,total_rows=len(indices));self.monitor.status('accumulating_full_feature_gradient');self.guard()
            ix=indices[first:first+256]
            c=mass[first:first+len(ix)].double()/self.target[ix].double().square().sum(-1).clamp_min(1e-8)
            loss=native_partial_loss(feedback_features(head,self.x[ix]),coefficient,
                self.target[ix].double()-self.mean.double(),c)
            loss.backward()
        gradients=[p.grad.detach().clone() for p in head.expand.parameters()]
        original.require(all(bool(torch.isfinite(g).all()) for g in gradients),'Nonfinite full feature gradient')
        norm=math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
        original.require(norm>0,'Zero full feature gradient')
        original.require(all(p.grad is None for p in head.original.parameters()) and head.project.weight.grad is None,'Frozen coefficients received gradients')
        head.expand.requires_grad_(False);head.zero_grad(set_to_none=True)
        return gradients,norm

    def step(self,head,coefficient,current,indices,mass,label,finite_difference=False):
        import torch
        from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
        saved={n:v.detach().clone() for n,v in head.state_dict().items()}
        parameters=list(head.expand.parameters());start=[p.detach().clone() for p in parameters]
        gradients,norm=self.gradient(head,coefficient,indices,mass,label)
        parameter_norm=math.sqrt(sum(float(v.double().square().sum()) for v in start))
        fd=None
        if finite_difference:
            epsilon=self.reg['finite_difference_relative_displacement']*max(parameter_norm,1.)
            objectives=[]
            for sign in (1.,-1.):
                with torch.no_grad():
                    for p,a,g in zip(parameters,start,gradients):p.copy_(a-sign*epsilon*g/norm)
                _,trial=self.solve(head,indices,mass,current['ridge'],label+'/finite_difference')
                objectives.append(trial['profiled_objective']);head.load_state_dict(saved,strict=True)
            measured=(objectives[0]-objectives[1])/(2*epsilon);expected=-norm
            error=abs(measured-expected)
            fd=dict(displacement=epsilon,measured_derivative=measured,expected_derivative=expected,
                absolute_error=error,relative_error=error/max(norm,1e-30))
            original.require(error<=self.reg['finite_difference_relative_tolerance']*norm+self.reg['finite_difference_absolute_tolerance'],
                'Complete-record finite-difference gradient check failed')
        initial_step=self.reg['relative_proposal']*max(parameter_norm,1.)/norm
        trials=[]
        for attempt in range(self.reg['maximum_search_trials']):
            self.guard();head.load_state_dict(saved,strict=True)
            eta=initial_step*(.5**attempt)
            with torch.no_grad():
                for p,a,g in zip(parameters,start,gradients):p.copy_(a-eta*g)
            candidate,receipt=self.solve(head,indices,mass,current['ridge'],label+'/trial_'+str(attempt))
            threshold=current['profiled_objective']-self.reg['armijo']*eta*norm**2
            accepted=receipt['profiled_objective']<=threshold
            row=dict(attempt=attempt+1,step_size=eta,armijo_threshold=threshold,accepted=accepted,solve=receipt)
            trials.append(row);original.write_new(self.directory/(label+'_trial_'+str(attempt+1)+'.json'),row)
            if accepted:
                changed={n for n,v in head.state_dict().items() if not torch.equal(v,saved[n])}
                original.require(changed<= {'expand.weight','expand.bias','original.1.weight','original.1.bias','project.weight'},'Unexpected feature change')
                return candidate,receipt,dict(gradient_norm=norm,parameter_norm=parameter_norm,
                    finite_difference=fd,trials=trials,accepted_attempt=attempt+1,changed_parameters=sorted(changed))
        head.load_state_dict(saved,strict=True)
        original.require(tensor_state_sha(head.state_dict())==tensor_state_sha(saved),'Rejected-step restore failed')
        raise RuntimeError('Bounded profiled line search exhausted')


def run(stage):
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Require registered GPU0 environment')
    reg=original.load(REGS[stage]);directory=RUNS[stage];original.require(not directory.exists(),'Preserve all executions')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push sources first')
    for n in [str(REGS[stage].relative_to(ROOT)),*FILES]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Uncommitted source')
    import torch
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    directory.mkdir();monitor=GlyphReaderMonitor(directory,1800);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External profiled feedback deadline')))
    completed=[]
    try:
        monitor.status('verifying_profiled_feedback_sources')
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            experiment=Experiment(reg,device,monitor,directory);experiment.guard();head=experiment.make()
            source_hash=tensor_state_sha(head.state_dict())
            legacy=experiment.actual(head,experiment.rows,experiment.weights,legacy=True)
            original.require(abs(legacy-reg['legacy_bf16_error'])<=reg['legacy_replay_tolerance'],'Legacy BF16 source error changed')
            same_weight_fp32=experiment.actual(head,experiment.rows,experiment.weights)
            if stage=='preflight':
                for index,row in enumerate(experiment.data['cases']):
                    head=experiment.make();saved={n:v.detach().clone() for n,v in head.state_dict().items()}
                    ix=row['rows'].to(device);mass=torch.full((len(ix),),1/len(ix),device=device)
                    coefficient,initial=experiment.solve(head,ix,mass,label='preflight_'+row['case']['role'])
                    _,final,step=experiment.step(head,coefficient,initial,ix,mass,'preflight_'+row['case']['role'],finite_difference=index==0)
                    original.require(final['actual_native_error']<=initial['actual_native_error']+reg['preflight_actual_error_allowance'],
                        'Actual complete-record error regressed beyond preflight allowance')
                    head.load_state_dict(saved,strict=True);original.require(tensor_state_sha(head.state_dict())==source_hash,'Preflight source reset failed')
                    result=dict(case=row['case'],complete_native_positions=len(ix),initial=initial,final=final,
                        step=step,exact_reset=True)
                    completed.append(result);original.write_new(directory/(row['case']['role']+'.json'),result)
                    del saved;gc.collect();torch.cuda.empty_cache();experiment.guard()
                result=dict(schema='ilm-profiled-feedback-preflight-result-v1',preflight_passed=True,all_trial_states_reset=True,
                    trials=completed,legacy_bf16_error=legacy,same_weight_fp32_error=same_weight_fp32,
                    source_head_tensor_sha256=source_hash,retained_successor_head=False)
            else:
                coefficient,initial=experiment.solve(head,experiment.rows,experiment.weights,label='common_fp32_baseline')
                control_cases=[]
                for row in experiment.data['cases']:
                    ix=row['rows'].to(device);mass=torch.full((len(ix),),1/len(ix),device=device)
                    control_cases.append(dict(case=row['case'],actual_error=experiment.actual(head,ix,mass)))
                control_path=directory/'fixed_feature_control.pt'
                torch.save(dict(head=head.state_dict(),head_precision='float32_feedback',
                    registration_sha256=original.sha(REGS[stage]),nonlinear_features_changed=False),control_path)
                for iteration in range(reg['outer_steps']):
                    coefficient,point,step=experiment.step(head,coefficient,initial if iteration==0 else point,
                        experiment.rows,experiment.weights,'outer_'+str(iteration+1).zfill(2))
                    row=dict(outer_step=iteration+1,solve=point,step=step,full_source_rows=len(experiment.rows))
                    completed.append(row);original.write_new(directory/('outer_'+str(iteration+1).zfill(2)+'.json'),row)
                    print(json.dumps(dict(outer_step=iteration+1,actual_error=point['actual_native_error'],
                        objective=point['profiled_objective'],attempt=step['accepted_attempt'])),flush=True)
                cases=[]
                for row in experiment.data['cases']:
                    ix=row['rows'].to(device);mass=torch.full((len(ix),),1/len(ix),device=device)
                    cases.append(dict(case=row['case'],actual_error=experiment.actual(head,ix,mass)))
                path=directory/'learned_features_final.pt'
                torch.save(dict(head=head.state_dict(),head_precision='float32_feedback',
                    registration_sha256=original.sha(REGS[stage]),outer_steps=len(completed)),path)
                checks=dict(final_actual_error_at_most_010=point['actual_native_error']<=.10,
                    gain_over_control_at_least_010=initial['actual_native_error']-point['actual_native_error']>=.10,
                    all_complete_cases_at_most_010=all(v['actual_error']<=.10 for v in cases))
                result=dict(schema='ilm-profiled-feedback-fit-result-v1',legacy_bf16_error=legacy,
                    same_weight_fp32_error=same_weight_fp32,common_fp32_control=initial,control_cases=control_cases,
                    outer_steps=completed,final=point,final_cases=cases,reader_support=all(checks.values()),support_checks=checks,
                    control_head_sha256=original.sha(control_path),final_head_sha256=original.sha(path),
                    fitted_feature_parameters=sum(p.numel() for p in head.expand.parameters()),full_source_rows=len(experiment.rows))
            result.update(registration_sha256=original.sha(REGS[stage]),source_commit=commit,resources=monitor.receipt(),
                language_measured=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(directory/'result.json',result);monitor.status('complete_profiled_feedback_'+stage)
    except BaseException as error:
        original.write_new(directory/'failure.json',dict(error_type=type(error).__name__,error=str(error),
            completed_trials_or_steps=len(completed),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_profiled_feedback_'+stage,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('register-preflight','run-preflight','register-fit','run-fit'))
    verb,stage=parser.parse_args().action.split('-',1)
    register(stage) if verb=='register' else run(stage)
