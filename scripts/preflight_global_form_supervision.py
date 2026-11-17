#!/usr/bin/env python3
"""Fixed-budget physical writer trial with training-only global form supervision."""
import argparse
import copy
import importlib.metadata
import json
import os
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
REG=ROOT/'references/evidence/global_form_preflight_registration_20261118.json'
RUN=ROOT/'artifacts/global_form_preflight_v1'
CACHE=ROOT/'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt'
PREPARATION=ROOT/'artifacts/global_form_preparation_v1/result.json'
CONTROL=ROOT/'artifacts/occupancy_density_preflight_v1/tree.json'
FILES=('scripts/preflight_global_form_supervision.py','ilm/visual_lm/global_form_supervision.py',
       'tests/test_global_form_supervision.py','ilm/visual_lm/occupancy_glyph_density.py',
       'ilm/visual_lm/conditional_pixel_density.py','ilm/visual_lm/glyph_reader_monitor.py',
       'ilm/visual_lm/visual_context_deployment_resources.py','ilm/visual_lm/full_rank_visual_span.py',
       'scripts/measure_reader_writing_consequence.py','scripts/diagnose_visual_context_gradients.py',
       'references/global_form_preflight_protocol_20261118.md',
       'references/evidence/global_form_supervision_cpu_20261118.json',
       'references/evidence/global_form_preparation_registration_20261118.json',
       'references/evidence/global_form_preparation_result_20261118.json',
       'references/evidence/occupancy_density_preflight_result_20261118.json',
       'references/evidence/occupancy_density_preflight_registration_20261118.json',
       'artifacts/global_form_preparation_v1/result.json','artifacts/global_form_preparation_v1/labels.pt',
       'artifacts/occupancy_density_preflight_v1/tree.json',str(CACHE.relative_to(ROOT)))


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only registration')
    preparation=original.load(PREPARATION);control=original.load(CONTROL)
    original.require(preparation['all_targets_exactly_supported'] and preparation['complete_target_positions']==552,'Complete preparation required')
    paths=list(FILES)
    for stage in ['initial','final']:
        for case in control[stage]['cases']:paths.extend([case['raw_file'],case['trace_file']])
    original.write_new(REG,dict(schema='ilm-global-form-preflight-v1',input_sha256={n:original.sha(ROOT/n) for n in paths},
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},classes=preparation['classes'],seed=202611190,
        helper_seed=202611191,helper_hidden_width=192,helper_loss_coefficient=1.,helper_normalization='log(classes)',
        initial_tensor_sha256=control['initial_tensor_sha256'],updates=96,complete_cases=3,targets_per_evaluation=552,
        microbatch_images=8,generation_batch_images=16,learning_rate=.0002,weight_decay=.01,gradient_clip=1.,
        maximum_replay_logp_error=.0002,unchanged_control=str(CONTROL.relative_to(ROOT)),
        support=dict(pooled_f1=.75,every_case_f1=.60,gain_over_flat=.05,gain_over_tree=.05,false_blank=0,false_ink=0,
                     rotated_minus_correct_nll=.5,source_form_accuracy=.90,every_case_source_form_accuracy=.80),
        reference_history=True,supplied_span_lengths=True,glyphs_independent_given_span_state=True,
        resources=dict(maximum_seconds=1800,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,
                       minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        all_trial_updates_reset=True,helper_absent_from_generation=True,autonomous_language_measured=False,
        full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
                     and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Deterministic physical GPU0 required')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in [str(REG.relative_to(ROOT)),*[n for n in reg['input_sha256'] if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+name])==(ROOT/name).read_bytes(),'Commit every bound source')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before execution')
    import numpy as np
    import torch
    from ilm.visual_lm.global_form_supervision import GlobalFormSupervisor
    from ilm.visual_lm.occupancy_glyph_density import OccupancyGlyphDensity,OccupancySequence,encode_occupancy,decode_occupancy
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.measure_reader_writing_consequence import pixels,pool
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,1800);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External global-form preflight deadline')))
    try:
        monitor.status('verifying_global_form_preflight_sources')
        for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Bound source changed: '+name)
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            def guard():
                monitor.guard();original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Head-only eight-GiB ceiling')
            cached=torch.load(CACHE,weights_only=True,map_location='cpu')
            saved=torch.load(ROOT/'artifacts/global_form_preparation_v1/labels.pt',weights_only=True,map_location='cpu');records=[]
            original.require(len(cached)==len(saved['labels'])==3 and len(saved['class_mask_sha256'])==reg['classes'],'Whole source label coverage differs')
            for row,labels in zip(cached,saved['labels']):
                sizes=[len(v) for v in row['target']]
                original.require(len(sizes)==len(row['context']) and sum(sizes)==len(row['ink'])==len(labels),'Complete source/span support differs')
                state=torch.repeat_interleave(row['context'].float(),torch.tensor(sizes),dim=0).to(device)
                index=torch.cat([torch.arange(s) for s in sizes]).to(device)
                length=torch.cat([torch.full((s,),s) for s in sizes]).to(device)
                records.append(dict(case=row['case'],states=state,index=index,length=length,ink=row['ink'].to(device),labels=labels.to(device)))
            original.require([len(v['ink']) for v in records]==[512,5,35],'Complete target coverage differs')
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(reg['seed']).get_state())
            density=OccupancyGlyphDensity(mode='tree').to(device).eval();initial=copy.deepcopy(density.state_dict());initial_hash=tensor_state_sha(initial)
            original.require(initial_hash==reg['initial_tensor_sha256'],'Fresh writer differs from unchanged tree control')
            helper=GlobalFormSupervisor(192,reg['classes'],seed=reg['helper_seed']).to(device).eval()
            helper_initial=copy.deepcopy(helper.state_dict());helper_initial_hash=tensor_state_sha(helper_initial)
            original.require(helper not in list(density.modules()),'Training-only helper is attached to inference')
            for record in records:record['sequences']={'tree':encode_occupancy(record['ink'],'tree')}

            def sequence_slice(sequence,start,stop):
                return OccupancySequence(**{n:v[start:stop] for n,v in vars(sequence).items()})

            def likelihood(density,record,*,backward=False,rotate=False,context_probe=False):
                states=torch.roll(record['states'],1,0) if rotate else record['states']
                if context_probe:states=states.detach().clone().requires_grad_(True)
                sequence=record['sequences']['tree'];total=0.;auxiliary=0.
                for start in range(0,len(states),reg['microbatch_images']):
                    guard();stop=start+reg['microbatch_images'];chunk=sequence_slice(sequence,start,stop)
                    value=-density.log_prob(states[start:stop],chunk,record['index'][start:stop],record['length'][start:stop]).sum()/len(states)
                    original.require(bool(torch.isfinite(value)),'Nonfinite complete-image likelihood')
                    if backward:
                        ce,_=helper.losses(density,states[start:stop],record['labels'][start:stop],record['index'][start:stop],record['length'][start:stop])
                        ce=ce.sum()/len(states);(value/1024+ce/helper.loss_scale).backward();auxiliary+=float(ce.detach())
                    total+=float(value.detach())
                result=dict(nats_per_image=total,positions=len(states))
                if backward:result.update(auxiliary_nats_per_image=auxiliary,combined_objective=total/1024+auxiliary/helper.loss_scale)
                if context_probe:
                    original.require(states.grad is not None and bool(torch.isfinite(states.grad).all()) and float(states.grad.norm())>0,'Contextual support gradient missing')
                    result['context_gradient_norm']=float(states.grad.norm())
                return result

            def classify(stage):
                rows=[]
                for record in records:
                    guard();predictions=[];ce_total=0.
                    with torch.no_grad():
                        for start in range(0,len(record['ink']),8):
                            stop=start+8;ce,logits=helper.losses(density,record['states'][start:stop],record['labels'][start:stop],record['index'][start:stop],record['length'][start:stop])
                            predictions.append(logits.argmax(-1).cpu());ce_total+=float(ce.sum())
                    predictions=torch.cat(predictions);labels=record['labels'].cpu();correct=int((predictions==labels).sum())
                    path=RUN/f'{stage}_{record["case"]["role"]}_helper_predictions.pt'
                    torch.save(dict(predictions=predictions,source_labels=labels),path)
                    rows.append(dict(case=record['case'],positions=len(labels),correct=correct,accuracy=correct/len(labels),
                        mean_cross_entropy=ce_total/len(labels),predictions_file=str(path.relative_to(ROOT)),predictions_sha256=original.sha(path)))
                return dict(cases=rows,positions=552,correct=sum(v['correct'] for v in rows),accuracy=sum(v['correct'] for v in rows)/552)

            def evaluate(density,stage):
                rows=[];allroots=[]
                for record in records:
                    monitor.position=dict(mode=density.mode,stage=stage,case=record['case']['role']);monitor.status('generating_every_occupancy_image')
                    arrays=[];replays=[];maximum_error=0.;traces=[]
                    for start in range(0,len(record['ink']),reg['generation_batch_images']):
                        guard();stop=start+reg['generation_batch_images'];states=record['states'][start:stop]
                        captured=[];method=density.step
                        def observe(*args,**kwargs):
                            logits,cache=method(*args,**kwargs);captured.append(logits.detach().cpu());return logits,cache
                        density.step=observe
                        try:
                            ink,trace=density.generate(states,glyph_index=record['index'][start:stop],span_length=record['length'][start:stop])
                        finally:del density.step
                        seq=encode_occupancy(ink,density.mode)
                        original.require(torch.equal(decode_occupancy(seq,32,density.mode),ink.cpu()),'Generated physical trace not lossless')
                        trace_actions=torch.tensor([v['action'] for v in trace]).T
                        trace_active=torch.tensor([v['active'] for v in trace]).T
                        trace_meta=torch.tensor([v['meta'] for v in trace]).permute(1,0,2)
                        original.require(torch.equal(trace_active,seq.valid.cpu()) and torch.equal(trace_actions[trace_active],seq.actions.cpu()[trace_active]),'Actual generation and replay action coverage differ')
                        meta=torch.stack((seq.depths,seq.rows,seq.columns),-1).cpu()
                        original.require(torch.equal(trace_meta[trace_active],meta[trace_active]),'Supplied rather than generated topology')
                        with torch.no_grad():parallel=density(states,seq,record['index'][start:stop],record['length'][start:stop]).cpu()
                        cached_logits=torch.stack(captured,1)
                        cp=cached_logits.log_softmax(-1).gather(-1,trace_actions[...,None]).squeeze(-1)
                        pp=parallel.log_softmax(-1).gather(-1,seq.actions.cpu()[...,None]).squeeze(-1)
                        error=float((cp[trace_active]-pp[trace_active]).abs().max())
                        original.require(error<=reg['maximum_replay_logp_error'],'Cached/parallel physical probability replay failed')
                        maximum_error=max(maximum_error,error)
                        arrays.append(ink.cpu());replays.append(dict(actions=seq.actions.cpu(),depths=seq.depths.cpu(),rows=seq.rows.cpu(),columns=seq.columns.cpu(),valid=seq.valid.cpu(),allow_zero=seq.allow_zero.cpu(),cached_logp=cp,parallel_logp=pp))
                        traces.append(dict(actions=trace_actions,active=trace_active,meta=trace_meta))
                    generated=torch.cat(arrays);allroots.append(generated.bool().flatten(1).any(1))
                    path=RUN/f'{density.mode}_{stage}_{record["case"]["role"]}.npz';np.savez_compressed(path,ink=generated.numpy())
                    tpath=RUN/f'{density.mode}_{stage}_{record["case"]["role"]}_physical_trace.pt';torch.save(dict(batches=replays,actual_trace=traces),tpath)
                    with torch.no_grad():
                        natural=likelihood(density,record);rotated=likelihood(density,record,rotate=True)
                    rows.append(dict(case=record['case'],pixels=pixels(generated,record['ink'].cpu()),correct_state_nll=natural,
                        rotated_state_nll=rotated,changed_state_rows=int((record['states']!=torch.roll(record['states'],1,0)).any(1).sum()),
                        raw_file=str(path.relative_to(ROOT)),raw_sha256=original.sha(path),trace_file=str(tpath.relative_to(ROOT)),trace_sha256=original.sha(tpath),maximum_replay_logp_error=maximum_error))
                advantage=sum((v['rotated_state_nll']['nats_per_image']-v['correct_state_nll']['nats_per_image'])*v['pixels']['positions'] for v in rows)/552
                return dict(cases=rows,pooled_pixels=pool(rows),pooled_rotated_minus_correct_nll=advantage),torch.cat(allroots)

            probes=[]
            for record in records:
                monitor.position=dict(case=record['case']['role']);monitor.status('probing_whole_record_shared_condition')
                density.zero_grad(set_to_none=True);helper.zero_grad(set_to_none=True)
                states=record['states'].detach().clone().requires_grad_(True)
                for start in range(0,len(states),8):
                    guard();stop=start+8
                    ce,_=helper.losses(density,states[start:stop],record['labels'][start:stop],record['index'][start:stop],record['length'][start:stop])
                    (ce.sum()/len(states)/helper.loss_scale).backward()
                auxiliary_norms={}
                for name in ['condition','glyph_position','span_length']:
                    parameters=list(getattr(density,name).parameters())
                    original.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),'Auxiliary shared gradient missing')
                    auxiliary_norms[name]=sum(float(p.grad.norm()) for p in parameters)
                    original.require(auxiliary_norms[name]>0,'Zero auxiliary shared gradient')
                for module in [density.blocks,density.bits,density.root_output,density.mask_output]:
                    original.require(all(p.grad is None for p in module.parameters()),'Helper reached a physical-prefix path')
                original.require(states.grad is not None and float(states.grad.norm())>0 and bool(torch.isfinite(states.grad).all()),'Auxiliary context gradient missing')
                density.zero_grad(set_to_none=True);helper.zero_grad(set_to_none=True)
                probe=likelihood(density,record,backward=True,context_probe=True)
                original.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in [*density.parameters(),*helper.parameters()]),'Incomplete combined gradient')
                probes.append(dict(case=record['case'],auxiliary_shared_gradient_norms=auxiliary_norms,**probe))
            density.zero_grad(set_to_none=True);helper.zero_grad(set_to_none=True)
            before,_=evaluate(density,'initial');before_classification=classify('initial');control=original.load(CONTROL)
            for case in before['cases']:
                oldcase=next(v for v in control['initial']['cases'] if v['case']==case['case'])
                original.require(np.array_equal(np.load(ROOT/case['raw_file'])['ink'],np.load(ROOT/oldcase['raw_file'])['ink']),'Initial pixels differ from immutable control')
            original.require(tensor_state_sha(density.state_dict())==initial_hash and tensor_state_sha(helper.state_dict())==helper_initial_hash,'Preflight probes changed tensors')
            parameters=[*density.parameters(),*helper.parameters()]
            optimizer=torch.optim.AdamW(parameters,lr=reg['learning_rate'],weight_decay=reg['weight_decay']);ledger=[]
            for update in range(reg['updates']):
                record=records[update%3];monitor.position=dict(update=update+1,case=record['case']['role']);monitor.status('fitting_global_form_whole_record')
                optimizer.zero_grad(set_to_none=True);value=likelihood(density,record,backward=True)
                norm=torch.nn.utils.clip_grad_norm_(parameters,reg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();ledger.append(dict(update=update+1,case=record['case'],gradient_norm=float(norm),**value))
            after,_=evaluate(density,'final');after_classification=classify('final')
            trained_hash=tensor_state_sha(density.state_dict());helper_trained_hash=tensor_state_sha(helper.state_dict())
            original.require(trained_hash!=initial_hash and helper_trained_hash!=helper_initial_hash,'Actual training update missing')
            density.load_state_dict(initial);helper.load_state_dict(helper_initial)
            original.require(tensor_state_sha(density.state_dict())==initial_hash and tensor_state_sha(helper.state_dict())==helper_initial_hash,'Exact trial reset failed')
            for name,digest in reg['input_sha256'].items():original.require(original.sha(ROOT/name)==digest,'Frozen source changed during trial: '+name)
            p=after['pooled_pixels'];c=control['final']['pooled_pixels'];bound=reg['support']
            old=original.load(ROOT/'references/evidence/occupancy_density_preflight_result_20261118.json')
            flat=next(v for v in old['arms'] if v['mode']=='flat')['final']['pooled_pixels']
            checks=dict(pooled_f1=p['f1']>=bound['pooled_f1'],every_case_f1=all(v['pixels']['f1']>=bound['every_case_f1'] for v in after['cases']),
                gain_over_flat=p['f1']-flat['f1']>=bound['gain_over_flat'],gain_over_tree=p['f1']-c['f1']>=bound['gain_over_tree'],
                false_blank=p['false_blank']==0,false_ink=p['false_ink']==0,
                conditional_dependence=after['pooled_rotated_minus_correct_nll']>=bound['rotated_minus_correct_nll'],
                source_form_accuracy=after_classification['accuracy']>=bound['source_form_accuracy'],
                every_case_source_form_accuracy=all(v['accuracy']>=bound['every_case_source_form_accuracy'] for v in after_classification['cases']))
            result=dict(schema='ilm-global-form-preflight-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                initial=before,final=after,initial_classification=before_classification,final_classification=after_classification,
                gradient_probes=probes,ledger=ledger,physical_writer_parameters=sum(p.numel() for p in density.parameters()),
                training_only_helper_parameters=sum(p.numel() for p in helper.parameters()),classes=reg['classes'],
                initial_tensor_sha256=initial_hash,trained_tensor_sha256=trained_hash,helper_initial_tensor_sha256=helper_initial_hash,
                helper_trained_tensor_sha256=helper_trained_hash,all_96_trial_updates_reset=True,helper_reset=True,source_cache_unchanged=True,
                initial_pixels_match_old_tree=True,helper_absent_from_generation=True,support_checks=checks,support=all(checks.values()),
                control_result=str(CONTROL.relative_to(ROOT)),control_sha256=original.sha(CONTROL),control_reexecuted=False,
                control_final_pixels=c,resources=monitor.receipt(),source_actor_parameters=1465381142,source_actor_loaded=False,
                scope='Training-only cached actual histories, supplied lengths and independent within-span glyphs; no autonomous language',
                autonomous_language_measured=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_global_form_preflight')
            print(json.dumps(dict(support=result['support'],checks=checks,pixels=p,source_form_accuracy=after_classification['accuracy'])),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_global_form_preflight',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
