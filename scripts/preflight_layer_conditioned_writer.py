#!/usr/bin/env python3
"""Real complete-record fitting/reset test of language-conditioned image layers."""
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
RUN=ROOT/'artifacts/layer_conditioned_writer_preflight_v1'
REG=ROOT/'references/evidence/layer_conditioned_writer_preflight_registration_20261118.json'
PARENT=ROOT/'references/evidence/reader_writing_consequence_registration_20261118.json'
FILES=('scripts/preflight_layer_conditioned_writer.py','ilm/visual_lm/layer_conditioned_writer.py',
       'tests/test_layer_conditioned_writer.py','references/writer_layer_conditioning_direction_20261118.md',
       'references/writer_layer_conditioning_preflight_20261118.md')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU registration only')
    parent=original.load(PARENT);bound=dict(parent['input_sha256'])
    paths=[PARENT,ROOT/'references/evidence/reader_writing_consequence_result_20261118.json',
           ROOT/'references/evidence/layer_conditioned_writer_cpu_20261118.json',*[ROOT/n for n in FILES]]
    for p in paths:bound[str(p.relative_to(ROOT))]=original.sha(p)
    for n,d in bound.items():original.require(original.sha(ROOT/n)==d,'Changed bound source: '+n)
    original.write_new(REG,dict(schema='ilm-layer-conditioned-writer-preflight-v1',input_sha256=bound,
        packages=parent['packages'],initial=parent['initial'],teacher=parent['teacher'],cases=parent['cases'],codec=parent['codec'],
        arms=['unchanged_conditioning','layer_conditioning'],updates_per_arm=12,microbatch_spans=16,
        learning_rate=.0002,weight_decay=.01,gradient_clip=1.,training_seed=202611187,
        sampling_seed=202611188,flow_steps=8,complete_targets_per_arm_per_evaluation=552,
        reference_history=True,supplied_lengths=True,all_trial_updates_reset=True,
        support=dict(gain_over_initial=.05,gain_over_control=.02,no_case_deterioration=True,no_additional_whitespace_errors=True),
        resources=dict(maximum_seconds=1200,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=22,
                       minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        full_language_fit_authorized=False,autonomous_language_measured=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG),bindings=len(bound))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
                     and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic GPU0 only')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before execution')
    for n in [str(REG.relative_to(ROOT)),*FILES]:original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Uncommitted source')
    import numpy as np
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.layer_conditioned_writer import LayerConditionedWriter
    from ilm.visual_lm.trainable_context_checkpoint import load_deploy
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_full_rank_visual_span import training_source
    from scripts.measure_reader_writing_consequence import pixels,pool
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,1200);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External writer-conditioning deadline')))
    finished=[]
    try:
        monitor.status('verifying_layer_conditioning_sources')
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Changed source: '+n)
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            corpus,*_=training_source(ROOT/reg['teacher']);monitor.guard()
            model,updates,_=load_deploy(ROOT/reg['initial']['deploy'],ROOT.parent/'LocalLLM',ROOT/reg['codec'],device)
            original.require(updates==8172,'Use prescribed original final actor')
            frozen=lambda:tensor_state_sha({n:v for n,v in model.state_dict().items() if not n.startswith('writer.')})
            frozen_before=frozen();old=model.writer
            old_hash=tensor_state_sha(old.state_dict());length_hash=tensor_state_sha(old.length.state_dict())
            cache=[]
            for case in reg['cases']:
                monitor.position=dict(case=case['role']);monitor.status('caching_whole_training_record');monitor.guard()
                row=corpus[case['corpus_index']];alignment=corpus.offline_teacher_alignment(case['corpus_index'])
                original.require(len(row['ink'])==case['glyphs'],'Complete source changed')
                ends=row['span_glyph_ends'].tolist();sizes=[b-a for a,b in zip([0,*ends[:-1]],ends)]
                spans=list(row['ink'].to(device).split(sizes));counts=torch.diff(alignment['span_unit_ends'],prepend=alignment['span_unit_ends'].new_zeros(1)).to(device)
                original.require(int(counts.sum())==case['native_units'],'Native count metadata changed')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):observation=model.observe(spans,counts,[len(spans)])
                first=max(row['prompt_spans']-1,0)
                # Clone outside inference mode so the frozen values can serve
                # as ordinary inputs to a differentiable writer computation.
                context=observation.contexts[first:-1].detach().clone()
                target=[v.detach().clone() for v in observation.glyph_latents[first+1:]]
                ink=torch.cat(spans[first+1:]).cpu()
                original.require(len(context)==len(target) and len(ink) in [512,5,35],'All next-span targets required')
                cache.append(dict(case=case,context=context,target=target,ink=ink))
                del observation;gc.collect();torch.cuda.empty_cache()
            original.require(sum(len(v['ink']) for v in cache)==552,'Complete support changed')
            cachepath=RUN/'conditional_training_cache.pt'
            torch.save([dict(case=v['case'],context=v['context'].cpu(),target=[t.cpu() for t in v['target']],ink=v['ink']) for v in cache],cachepath)

            def mse(writer,record,seed,backward=False):
                targets=record['target'];gold=pad_sequence(targets,batch_first=True).float()
                lengths=torch.tensor([len(t) for t in targets],device=device)
                valid=torch.arange(gold.shape[1],device=device)[None]<lengths[:,None]
                generator=torch.Generator(device=device).manual_seed(seed)
                noise=torch.randn(gold.shape,device=device,generator=generator)
                time=torch.rand(len(gold),device=device,generator=generator)
                noise_hash=hashlib.sha256(noise.cpu().numpy().tobytes()+time.cpu().numpy().tobytes()).hexdigest()
                loss=0.;denominator=int(valid.sum())
                for start in range(0,len(gold),reg['microbatch_spans']):
                    monitor.guard();stop=start+reg['microbatch_spans'];g=gold[start:stop];n=noise[start:stop];t=time[start:stop];ok=valid[start:stop]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        prediction=writer.velocity(n.lerp(g,t[:,None,None]),t,record['context'][start:stop],ok)
                        term=(prediction.float()-(g-n)).square().mean(-1)[ok].sum()/denominator
                    original.require(bool(torch.isfinite(term)),'Nonfinite complete-record loss')
                    if backward:term.backward()
                    loss+=float(term.detach())
                return dict(loss=loss,noise_time_sha256=noise_hash,positions=denominator)

            def evaluate(writer,arm,stage):
                rows=[]
                for index,record in enumerate(cache):
                    monitor.position=dict(arm=arm,stage=stage,case=record['case']['role']);monitor.status('measuring_every_writer_target')
                    target=record['target'];lengths=[len(t) for t in target];output=[None]*len(target);latents=[None]*len(target);noise_hash=hashlib.sha256()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        for location,size in enumerate(lengths):
                            generator=torch.Generator(device=device).manual_seed(reg['sampling_seed']+100000*index+location)
                            noise=torch.randn((1,size,writer.latent_width),device=device,generator=generator)
                            noise_hash.update(noise.cpu().numpy().tobytes())
                            # Batch by fixed consecutive microbatch below would
                            # change numerical execution; keep the declared
                            # one-span sampling schedule for every arm/stage.
                            value=noise;ok=torch.ones((1,size),device=device,dtype=torch.bool)
                            for step in range(reg['flow_steps']):
                                t=torch.full((1,),step/reg['flow_steps'],device=device)
                                value=value+writer.velocity(value,t,record['context'][location:location+1],ok).float()/reg['flow_steps']
                            output[location]=(model.codec.decode(value[0])<0).to(torch.uint8).cpu();latents[location]=value[0].cpu()
                            if location%16==0:monitor.guard()
                    generated=torch.cat(output);path=RUN/f'{arm}_{stage}_{record["case"]["role"]}.npz'
                    np.savez_compressed(path,ink=generated.numpy(),latent=torch.cat(latents).numpy(),span_lengths=np.array(lengths,dtype=np.int32))
                    rows.append(dict(case=record['case'],pixels=pixels(generated,record['ink']),raw_file=str(path.relative_to(ROOT)),raw_sha256=original.sha(path),noise_sha256=noise_hash.hexdigest()))
                return dict(cases=rows,pooled_pixels=pool(rows))

            initial_velocity=None;initial_grads=None;initial_pixels=None;initial_noise=None
            for arm in reg['arms']:
                monitor.position=dict(arm=arm);monitor.status('initializing_identity_writer');monitor.guard()
                writer=LayerConditionedWriter(copy.deepcopy(old),modulated=arm=='layer_conditioning').to(device).eval()
                writer.requires_grad_(True);writer.length.requires_grad_(False)
                state=copy.deepcopy(writer.state_dict());before=tensor_state_sha(state)
                r=cache[0];probe=pad_sequence(r['target'][:16],batch_first=True).float();time=torch.full((len(probe),),.5,device=device)
                valid=torch.arange(probe.shape[1],device=device)[None]<torch.tensor([len(t) for t in r['target'][:16]],device=device)[:,None]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    source=old.velocity(probe,time,r['context'][:16],valid)
                    velocity=writer.velocity(probe,time,r['context'][:16],valid)
                drift=float((velocity.float()-source.float()).square().mean().sqrt()/source.float().square().mean().sqrt().clamp_min(1e-8))
                original.require(drift<=1e-5,'Common execution diverges from original writer')
                if initial_velocity is None:initial_velocity=velocity.cpu()
                else:original.require(torch.equal(initial_velocity,velocity.cpu()),'Identity velocity changed')
                writer.zero_grad(set_to_none=True);gradient_probe=mse(writer,r,reg['training_seed']-1,True)
                grads={n:p.grad.detach().clone() for n,p in writer.original.named_parameters() if p.requires_grad}
                original.require(all(bool(torch.isfinite(v).all()) for v in grads.values()),'Nonfinite base gradient')
                if initial_grads is None:initial_grads=grads
                else:
                    squared=sum(float((grads[n].float()-v.float()).square().sum()) for n,v in initial_grads.items())
                    norm=sum(float(v.float().square().sum()) for v in initial_grads.values())
                    original.require((squared/max(norm,1e-30))**.5<=1e-5,'Initial base gradients differ')
                    original.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm())>0 for p in writer.modulations.parameters()),'Modulation gradient unsupported')
                writer.zero_grad(set_to_none=True)
                start=evaluate(writer,arm,'initial')
                if initial_pixels is None:
                    initial_pixels=[np.load(ROOT/v['raw_file'])['ink'] for v in start['cases']]
                    initial_noise=[v['noise_sha256'] for v in start['cases']]
                else:
                    original.require(all(np.array_equal(a,np.load(ROOT/b['raw_file'])['ink']) for a,b in zip(initial_pixels,start['cases'])),'Initial generated pixels differ')
                    original.require(initial_noise==[v['noise_sha256'] for v in start['cases']],'Initial sampling differs')
                optimizer=torch.optim.AdamW([p for p in writer.parameters() if p.requires_grad],lr=reg['learning_rate'],weight_decay=reg['weight_decay'])
                ledger=[]
                for update in range(reg['updates_per_arm']):
                    record=cache[update%len(cache)];monitor.position=dict(arm=arm,update=update+1,case=record['case']['role']);monitor.status('fitting_complete_writer_record')
                    optimizer.zero_grad(set_to_none=True);measurement=mse(writer,record,reg['training_seed']+update,True)
                    norm=torch.nn.utils.clip_grad_norm_([p for p in writer.parameters() if p.requires_grad],reg['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();ledger.append(dict(update=update+1,case=record['case'],gradient_norm=float(norm),**measurement))
                final=evaluate(writer,arm,'final');fitted_hash=tensor_state_sha(writer.state_dict())
                original.require(tensor_state_sha(writer.length.state_dict())==length_hash,'Length head changed')
                writer.load_state_dict(state);original.require(tensor_state_sha(writer.state_dict())==before,'Trial reset failed')
                original.require(tensor_state_sha(old.state_dict())==old_hash and frozen()==frozen_before,'Original actor changed')
                result=dict(arm=arm,initial=start,final=final,ledger=ledger,initial_function_hash=before,fitted_hash=fitted_hash,
                            exact_reset=True,source_writer_unchanged=True,frozen_actor_unchanged=True,length_head_unchanged=True,
                            source_loop_relative_rms=drift,gradient_probe=gradient_probe,added_parameters=writer.added_parameters)
                finished.append(result);original.write_new(RUN/f'{arm}.json',result)
                del writer,optimizer,state,grads;gc.collect();torch.cuda.empty_cache()
            a,b=finished
            original.require([v['noise_time_sha256'] for v in a['ledger']]==[v['noise_time_sha256'] for v in b['ledger']],'Training noise differs')
            original.require(all(x['noise_sha256']==y['noise_sha256'] for x,y in zip(a['final']['cases'],b['final']['cases'])),'Final sampling differs')
            baseline=a['initial']['pooled_pixels'];control=a['final']['pooled_pixels'];candidate=b['final']['pooled_pixels']
            checks=dict(gain_over_initial=candidate['f1']-baseline['f1']>=.05,gain_over_control=candidate['f1']-control['f1']>=.02,
                        no_case_deterioration=all(x['pixels']['f1']>=y['pixels']['f1'] for x,y in zip(b['final']['cases'],a['final']['cases'])),
                        no_additional_whitespace_errors=all(candidate[k]<=min(baseline[k],control[k]) for k in ['false_blank','false_ink']))
            result=dict(schema='ilm-layer-conditioned-writer-preflight-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                        arms=finished,support_checks=checks,support=all(checks.values()),conditional_cache_sha256=original.sha(cachepath),
                        frozen_model_sha256=frozen_before,original_writer_sha256=old_hash,all_updates_reset=True,
                        complete_actor_parameters_before_intervention=sum(p.numel() for p in model.parameters()),
                        resources=monitor.receipt(),full_language_fit_authorized=False,autonomous_language_measured=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_layer_conditioning_preflight')
            print(json.dumps(dict(support=result['support'],checks=checks,initial=baseline,control=control,candidate=candidate)),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(finished),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_layer_conditioning_preflight',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
