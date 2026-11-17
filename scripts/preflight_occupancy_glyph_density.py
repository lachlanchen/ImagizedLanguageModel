#!/usr/bin/env python3
"""Prospectively fixed real-image comparison of flat and tree writing laws."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
REG=ROOT/'references/evidence/occupancy_density_preflight_registration_20261118.json'
RUN=ROOT/'artifacts/occupancy_density_preflight_v1'
CACHE=ROOT/'artifacts/layer_conditioned_writer_preflight_v1/conditional_training_cache.pt'
FILES=('scripts/preflight_occupancy_glyph_density.py','ilm/visual_lm/occupancy_glyph_density.py',
       'tests/test_occupancy_glyph_density.py','scripts/audit_occupancy_density_structure.py',
       'ilm/visual_lm/conditional_pixel_density.py','ilm/visual_lm/glyph_reader_monitor.py',
       'ilm/visual_lm/visual_context_deployment_resources.py','scripts/measure_reader_writing_consequence.py',
       'references/occupancy_density_preflight_protocol_20261118.md',
       'references/evidence/occupancy_density_cpu_20261118.json',
       'references/evidence/occupancy_density_structure_registration_20261118.json',
       'references/evidence/occupancy_density_structure_result_20261118.json',
       'references/evidence/layer_conditioned_writer_preflight_registration_20261118.json',
       'references/evidence/layer_conditioned_writer_preflight_result_20261118.json',
       'artifacts/layer_conditioned_writer_preflight_v1/result.json',str(CACHE.relative_to(ROOT)))


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU registration only')
    previous=original.load(ROOT/'artifacts/layer_conditioned_writer_preflight_v1/result.json')
    original.require(original.sha(CACHE)==previous['conditional_cache_sha256'],'Immutable original conditional cache differs')
    structure=original.load(ROOT/'references/evidence/occupancy_density_structure_result_20261118.json')
    original.require(structure['all_68948_saved_cells_roundtrip'] and structure['all_original_packed_bytes_preserved'],'Structure qualification failed')
    original.write_new(REG,dict(schema='ilm-occupancy-density-preflight-v1',input_sha256={n:original.sha(ROOT/n) for n in FILES},
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy']},modes=['flat','tree'],seed=202611190,
        updates_per_arm=96,complete_cases=3,targets_per_evaluation=552,microbatch_images=8,generation_batch_images=16,
        learning_rate=.0002,weight_decay=.01,gradient_clip=1.,maximum_replay_logp_error=.0002,
        support=dict(pooled_f1=.75,every_case_f1=.60,gain_over_flat=.05,false_blank=0,false_ink=0,rotated_minus_correct_nll=.5),
        reference_history=True,supplied_span_lengths=True,glyphs_independent_given_span_state=True,
        resources=dict(maximum_seconds=1800,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=8,
                       minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        all_trial_updates_reset=True,autonomous_language_measured=False,full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG))),flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
                     and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Require deterministic physical GPU0')
    original.require(not RUN.exists(),'Preserve every execution')
    reg=original.load(REG);commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for n in [str(REG.relative_to(ROOT)),*[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Commit every bound source')
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push before execution')
    import numpy as np
    import torch
    from ilm.visual_lm.occupancy_glyph_density import OccupancyGlyphDensity,OccupancySequence,encode_occupancy,decode_occupancy
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.measure_reader_writing_consequence import pixels,pool
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,1800);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External occupancy preflight deadline')))
    completed=[]
    try:
        monitor.status('verifying_occupancy_preflight_sources')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Package versions changed')
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            def guard():
                monitor.guard();original.require(torch.cuda.max_memory_allocated()/1024**3<=8,'Head-only eight-GiB ceiling')
            cached=torch.load(CACHE,weights_only=True,map_location='cpu');records=[]
            for row in cached:
                sizes=[len(v) for v in row['target']]
                original.require(len(sizes)==len(row['context']) and sum(sizes)==len(row['ink']),'Whole source/span cache differs')
                state=torch.repeat_interleave(row['context'].float(),torch.tensor(sizes),dim=0).to(device)
                index=torch.cat([torch.arange(s) for s in sizes]).to(device)
                length=torch.cat([torch.full((s,),s) for s in sizes]).to(device)
                records.append(dict(case=row['case'],states=state,index=index,length=length,ink=row['ink'].to(device)))
            original.require(len(records)==3 and [len(v['ink']) for v in records]==[512,5,35],'Complete target coverage differs')
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(reg['seed']).get_state())
            prototype=OccupancyGlyphDensity(mode='flat');initial=copy.deepcopy(prototype.state_dict());initial_hash=tensor_state_sha(initial)
            initial_roots=None

            def sequence_slice(sequence,start,stop):
                return OccupancySequence(**{n:v[start:stop] for n,v in vars(sequence).items()})

            def likelihood(density,record,*,backward=False,rotate=False,context_probe=False):
                states=torch.roll(record['states'],1,0) if rotate else record['states']
                if context_probe:states=states.detach().clone().requires_grad_(True)
                sequence=record['sequences'][density.mode];total=0.
                for start in range(0,len(states),reg['microbatch_images']):
                    guard();stop=start+reg['microbatch_images'];chunk=sequence_slice(sequence,start,stop)
                    value=-density.log_prob(states[start:stop],chunk,record['index'][start:stop],record['length'][start:stop]).sum()/len(states)
                    original.require(bool(torch.isfinite(value)),'Nonfinite complete-image likelihood')
                    if backward:(value/1024).backward()
                    total+=float(value.detach())
                result=dict(nats_per_image=total,positions=len(states))
                if context_probe:
                    original.require(states.grad is not None and bool(torch.isfinite(states.grad).all()) and float(states.grad.norm())>0,'Contextual support gradient missing')
                    result['context_gradient_norm']=float(states.grad.norm())
                return result

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

            for record in records:record['sequences']={m:encode_occupancy(record['ink'],m) for m in reg['modes']}
            for mode in reg['modes']:
                guard();density=copy.deepcopy(prototype).to(device);density.mode=mode;density.eval()
                original.require(tensor_state_sha(density.state_dict())==initial_hash,'Common initial tensors differ')
                probes=[]
                for record in records:
                    monitor.position=dict(mode=mode,case=record['case']['role']);monitor.status('probing_complete_occupancy_gradients')
                    density.zero_grad(set_to_none=True);probe=likelihood(density,record,backward=True,context_probe=True)
                    original.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in density.parameters()),'Incomplete/invalid physical gradient')
                    for name in ['root_output','mask_output']:
                        gradient=getattr(density,name).weight.grad
                        original.require(float(gradient.norm())>0,'Root/mask head receives no gradient')
                        probe[name+'_gradient_norm']=float(gradient.norm())
                    probes.append(dict(case=record['case'],**probe))
                density.zero_grad(set_to_none=True);before,roots=evaluate(density,'initial')
                if initial_roots is None:initial_roots=roots
                else:original.require(torch.equal(initial_roots,roots),'Initial learned root choices differ')
                optimizer=torch.optim.AdamW(density.parameters(),lr=reg['learning_rate'],weight_decay=reg['weight_decay']);ledger=[]
                for update in range(reg['updates_per_arm']):
                    record=records[update%3];monitor.position=dict(mode=mode,update=update+1,case=record['case']['role']);monitor.status('fitting_complete_occupancy_record')
                    optimizer.zero_grad(set_to_none=True);value=likelihood(density,record,backward=True)
                    norm=torch.nn.utils.clip_grad_norm_(density.parameters(),reg['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();ledger.append(dict(update=update+1,case=record['case'],gradient_norm=float(norm),**value))
                after,_=evaluate(density,'final');trained_hash=tensor_state_sha(density.state_dict())
                density.load_state_dict(initial);original.require(tensor_state_sha(density.state_dict())==initial_hash,'Occupancy trial reset failed')
                original.require(original.sha(CACHE)==reg['input_sha256'][str(CACHE.relative_to(ROOT))],'Frozen conditional cache changed')
                arm=dict(mode=mode,initial=before,final=after,gradient_probes=probes,ledger=ledger,parameter_count=sum(p.numel() for p in density.parameters()),
                         initial_tensor_sha256=initial_hash,trained_tensor_sha256=trained_hash,exact_reset=True,source_cache_unchanged=True)
                completed.append(arm);original.write_new(RUN/f'{mode}.json',arm)
                del density,optimizer;torch.cuda.empty_cache()
            flat,tree=completed;p=tree['final']['pooled_pixels'];c=flat['final']['pooled_pixels'];bound=reg['support']
            checks=dict(pooled_f1=p['f1']>=bound['pooled_f1'],every_case_f1=all(v['pixels']['f1']>=bound['every_case_f1'] for v in tree['final']['cases']),
                        gain_over_flat=p['f1']-c['f1']>=bound['gain_over_flat'],false_blank=p['false_blank']==0,false_ink=p['false_ink']==0,
                        conditional_dependence=tree['final']['pooled_rotated_minus_correct_nll']>=bound['rotated_minus_correct_nll'])
            result=dict(schema='ilm-occupancy-density-preflight-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                        arms=completed,support_checks=checks,support=all(checks.values()),all_96_trial_updates_per_arm_reset=True,
                        resources=monitor.receipt(),source_actor_parameters=1465381142,source_actor_loaded=False,
                        scope='Cached actual source-image histories, known counts/lengths, independent glyphs within each span; training-only head support',
                        autonomous_language_measured=False,full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_occupancy_density_preflight')
            print(json.dumps(dict(support=result['support'],checks=checks,flat=c,tree=p)),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(completed),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_occupancy_density_preflight',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
