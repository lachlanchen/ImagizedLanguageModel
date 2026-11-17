#!/usr/bin/env python3
"""Paired generated-pixel consequence of the two final FP32 reader heads."""
import argparse
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
SOURCE=ROOT/'artifacts/profiled_feedback_fit_v1'
PREPARED=ROOT/'artifacts/nonlinear_feedback_alignment_v1'
RUN=ROOT/'artifacts/reader_writing_consequence_v1'
REG=ROOT/'references/evidence/reader_writing_consequence_registration_20261118.json'
OLDREG=ROOT/'references/evidence/profiled_feedback_fit_registration_20261118.json'
PROTOCOL=ROOT/'references/reader_writing_consequence_protocol_20261118.md'
FILES=(str(Path(__file__).relative_to(ROOT)),str(PROTOCOL.relative_to(ROOT)))


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Register without CUDA')
    old=original.load(OLDREG);fit=original.load(SOURCE/'result.json')
    original.require(len(fit['outer_steps'])==12 and not fit['reader_support'],'Use the prescribed failed final reader')
    heads=dict(fixed_feature='fixed_feature_control.pt',learned_feature='learned_features_final.pt')
    bound=dict(old['input_sha256'])
    paths=[OLDREG,SOURCE/'result.json',ROOT/'references/evidence/profiled_feedback_fit_result_20261118.json',
        ROOT/'ilm/visual_lm/trainable_context_checkpoint.py',ROOT/'ilm/visual_lm/visual_sequence_corpus.py',
        *map(lambda n:ROOT/n,FILES),*[SOURCE/n for n in heads.values()]]
    for p in paths:bound[str(p.relative_to(ROOT))]=original.sha(p)
    for n,d in bound.items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
    original.write_new(REG,dict(schema='ilm-reader-writing-consequence-v1',input_sha256=bound,
        packages=old['packages'],initial=old['initial'],teacher=old['teacher'],cases=old['cases'],
        codec='artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt',
        heads={k:str((SOURCE/n).relative_to(ROOT)) for k,n in heads.items()},order=list(heads),
        flow_steps=8,seed=202611182,reference_history=True,supplied_update_counts=True,
        supplied_next_span_lengths=True,all_next_span_targets=True,
        reader_gate_remains_failed=True,optimizer_updates_permitted=False,autonomous_language_measured=False,
        resources=dict(maximum_seconds=1200,maximum_pressure_wait_seconds=300,maximum_allocated_gpu_gib=22,
            minimum_entry_host_gib=48,minimum_runtime_host_gib=24,minimum_free_gpu_gib=20,minimum_free_disk_gib=48,cpu_threads=2),
        full_language_fit_authorized=False,goal_achieved=False,automatic_retry=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)),sha256=original.sha(REG),bindings=len(bound))),flush=True)


def pixels(ink,gold):
    a,b=ink.bool(),gold.bool()
    original.require(a.shape==b.shape and a.ndim==4 and a.shape[1:]==(1,32,32),'Complete pixel support changed')
    pa,pb=a.flatten(1).any(1),b.flatten(1).any(1)
    out=dict(positions=len(a),tp=int((a&b).sum()),fp=int((a&~b).sum()),fn=int((~a&b).sum()),
        exact=int((a==b).flatten(1).all(1).sum()),source_blank=int((~pb).sum()),
        false_blank=int((~pa&pb).sum()),false_ink=int((pa&~pb).sum()))
    out['f1']=2*out['tp']/max(1,2*out['tp']+out['fp']+out['fn'])
    return out


def pool(rows):
    names=('positions','tp','fp','fn','exact','source_blank','false_blank','false_ink')
    values={k:sum(v['pixels'][k] for v in rows) for k in names}
    values['f1']=2*values['tp']/max(1,2*values['tp']+values['fp']+values['fn'])
    return values


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Require deterministic GPU0 environment')
    reg=original.load(REG);original.require(not RUN.exists(),'Preserve every execution')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    original.require(subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,timeout=30).split()[0]==commit,'Push sources before execution')
    for n in [str(REG.relative_to(ROOT)),*FILES]:
        original.require(subprocess.check_output(['git','show','HEAD:'+n])==(ROOT/n).read_bytes(),'Uncommitted source')
    import numpy as np
    import torch
    from torch import nn
    from ilm.visual_lm.profiled_visual_feedback import Float32VisualFeedback
    from ilm.visual_lm.trainable_context_checkpoint import load_deploy
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_full_rank_visual_span import training_source
    RUN.mkdir();monitor=GlyphReaderMonitor(RUN,1200);monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda _s,_f:(_ for _ in ()).throw(TimeoutError('External writing-consequence deadline')))
    completed=[]
    try:
        monitor.status('verifying_writing_consequence_sources')
        original.require(reg['packages']=={n:importlib.metadata.version(n) for n in reg['packages']},'Packages changed')
        for n,d in reg['input_sha256'].items():original.require(original.sha(ROOT/n)==d,'Bound source changed: '+n)
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission;torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;monitor.guard()
            corpus,*_=training_source(ROOT/reg['teacher'])
            data=torch.load(PREPARED/'prepared.pt',weights_only=True,map_location='cpu')
            cache=torch.load(PREPARED/'execution/frozen_queries.pt',weights_only=True,map_location='cpu')
            model,updates,provenance=load_deploy(ROOT/reg['initial']['deploy'],ROOT.parent/'LocalLLM',ROOT/reg['codec'],device)
            original.require(updates==8172,'Require the final preserved actor')
            frozen=lambda:tensor_state_sha({n:v for n,v in model.state_dict().items() if not n.startswith('encoder.feedback.')})
            original_frozen=frozen()
            for arm in reg['order']:
                monitor.position=dict(arm=arm);monitor.status('installing_fixed_reader_head');monitor.guard()
                base=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,2048)).to(device)
                head=Float32VisualFeedback(base,hidden_width=4096)
                saved=torch.load(ROOT/reg['heads'][arm],weights_only=True,map_location='cpu')
                original.require(saved['head_precision']=='float32_feedback','Reader precision changed')
                head.load_state_dict(saved['head'],strict=True)
                model.encoder.feedback=head.eval().requires_grad_(False);model.requires_grad_(False).eval()
                head_before=tensor_state_sha(head.state_dict());rows=[]
                for index,case in enumerate(reg['cases']):
                    monitor.position=dict(arm=arm,role=case['role']);monitor.status('measuring_complete_conditional_writing');monitor.guard()
                    row=corpus[case['corpus_index']];alignment=corpus.offline_teacher_alignment(case['corpus_index'])
                    original.require(len(row['ink'])==case['glyphs'],'Complete input changed')
                    ends=row['span_glyph_ends'].tolist();lengths=[b-a for a,b in zip([0,*ends[:-1]],ends)]
                    ink=row['ink'].to(device);spans=list(ink.split(lengths))
                    counts=torch.diff(alignment['span_unit_ends'],prepend=alignment['span_unit_ends'].new_zeros(1)).to(device)
                    original.require(int(counts.sum())==case['native_units'],'Complete source counts changed')
                    captured=[];hook=head.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach()))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        try:observation=model.observe(spans,counts,[len(spans)])
                        finally:hook.remove()
                        original.require(len(captured)==1,'One live query capture required')
                        valid=torch.arange(3,device=device)[None]<counts[:,None]
                        live=observation.feedback_vectors[valid].float()
                        prepared=next(v for v in data['cases'] if v['case']==case)['rows']
                        cached_query=cache[data['row_image'][prepared],data['row_slot'][prepared]].to(device)
                        cached=head(cached_query)*model.encoder.feedback_scale+model.encoder.feedback_mean
                        drift=float((live-cached).square().mean().sqrt()/live.square().mean().sqrt().clamp_min(1e-8))
                        first=max(0,row['prompt_spans']-1);context=observation.contexts[first:];targets=spans[first+1:]
                        target_lengths=[len(t) for t in targets];length_guess=model.writer.length(context.float()).argmax(-1)
                        output=[None]*len(targets);latents=[None]*len(targets);noises=[None]*len(targets)
                        for size in sorted(set(target_lengths)):
                            monitor.guard();where=[j for j,v in enumerate(target_lengths) if v==size];noise=[]
                            for location in where:
                                generator=torch.Generator(device=device).manual_seed(reg['seed']+100000*index+location)
                                noise.append(torch.randn((size,model.writer.latent_width),device=device,generator=generator))
                            value=torch.stack(noise);ok=torch.ones(value.shape[:2],dtype=torch.bool,device=device)
                            for location,n in zip(where,noise):noises[location]=n.cpu()
                            for step in range(reg['flow_steps']):
                                phase=torch.full((len(value),),step/reg['flow_steps'],device=device)
                                value=value+model.writer.velocity(value,phase,context[where],ok).float()/reg['flow_steps']
                            generated=(model.codec.decode(value.flatten(0,1))<0).to(torch.uint8).split(size)
                            for location,bits,z in zip(where,generated,value):output[location]=bits.cpu();latents[location]=z.cpu()
                    original.require(all(v is not None for v in output),'A target span was omitted')
                    generated=torch.cat(output);gold=torch.cat(targets).cpu();z=torch.cat(latents);noise=torch.cat(noises)
                    path=RUN/f'{arm}_{case["role"]}.npz'
                    np.savez_compressed(path,ink=generated.numpy(),latent=z.numpy(),span_lengths=np.array(target_lengths,dtype=np.int32),
                        length_argmax=length_guess.cpu().numpy())
                    qpath=RUN/f'{arm}_{case["role"]}_queries.pt';torch.save(dict(queries=captured[0].cpu(),update_counts=counts.cpu()),qpath)
                    result=dict(arm=arm,case=case,pixels=pixels(generated,gold),live_cached_feedback_relative_rms=drift,
                        initial_noise_sha256=hashlib.sha256(noise.numpy().tobytes()).hexdigest(),raw_file=str(path.relative_to(ROOT)),
                        raw_sha256=original.sha(path),live_queries_file=str(qpath.relative_to(ROOT)),live_queries_sha256=original.sha(qpath),
                        length_decisions=len(target_lengths)+1,length_correct=int((length_guess.cpu()==torch.tensor([*target_lengths,0])).sum()),
                        stop_predicted_at_terminal=bool(length_guess[-1]==0),reference_history=True,supplied_lengths=True,
                        autonomous_language_measured=False)
                    rows.append(result);original.write_new(RUN/f'{arm}_{case["role"]}.json',result)
                    del observation,captured,context,live,cached,output,latents,noises;gc.collect();torch.cuda.empty_cache()
                original.require(frozen()==original_frozen and tensor_state_sha(head.state_dict())==head_before,'Frozen actor/head changed')
                completed.append(dict(arm=arm,cases=rows,pooled_pixels=pool(rows),head_unchanged=True,all_other_model_tensors_unchanged=True))
            for a,b in zip(completed[0]['cases'],completed[1]['cases']):
                original.require(a['case']==b['case'] and a['initial_noise_sha256']==b['initial_noise_sha256'],'Paired sampling changed')
            result=dict(schema='ilm-reader-writing-consequence-result-v1',registration_sha256=original.sha(REG),source_commit=commit,
                arms=completed,pooled_f1_difference=completed[1]['pooled_pixels']['f1']-completed[0]['pooled_pixels']['f1'],
                complete_model_parameters=sum(p.numel() for p in model.parameters()),frozen_model_tensor_sha256=original_frozen,
                all_sampling_noise_matched=True,resources=monitor.receipt(),reader_gate_remains_failed=True,
                reference_history=True,supplied_lengths=True,autonomous_language_measured=False,
                full_language_fit_authorized=False,goal_achieved=False)
            original.write_new(RUN/'result.json',result);monitor.status('complete_reader_writing_consequence')
            print(json.dumps(dict(pooled_f1_difference=result['pooled_f1_difference'],arms=[(a['arm'],a['pooled_pixels']) for a in completed])),flush=True)
    except BaseException as error:
        original.write_new(RUN/'failure.json',dict(error_type=type(error).__name__,error=str(error),completed_arms=len(completed),automatic_retry=False,full_language_fit_authorized=False))
        monitor.status('failed_reader_writing_consequence',error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'))
    register() if parser.parse_args().action=='register' else run()
