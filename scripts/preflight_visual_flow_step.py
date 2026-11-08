#!/usr/bin/env python3
"""Admit one real flow-learning step after complete native candidate replay."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import types

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.image_edit import visual_semantic as vs

CANDIDATE = ROOT / "artifacts/visual_semantic_native_preflight_v2"
DATA = ROOT / "artifacts/visual_flow_pages_v1"
CONDITION = ROOT / "artifacts/visual_task_condition_1024_v1/runtime"
FILES = ("scripts/preflight_visual_flow_step.py", "ilm/image_edit/native_flow_step.py",
    "ilm/image_edit/visual_flow_learning.py", "ilm/image_edit/visual_semantic.py",
    "ilm/image_edit/compiled_visual_condition.py", "ilm/image_edit/baseline.py",
    "references/visual_flow_step_preflight_protocol.md")


def identity():
    if subprocess.check_output(["git", "status", "--porcelain", "--", *FILES], cwd=ROOT, text=True):
        raise ValueError("commit the exact flow preflight code and protocol first")
    return {p: sha(ROOT / p) for p in FILES}


def owner(pid):
    p=Path('/proc')/str(pid)
    try:return dict(pid=pid,command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),
        start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def prepare(output):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise ValueError("hide CUDA for CPU preparation")
    code=identity();output.mkdir(parents=True,exist_ok=False)
    import torch
    from safetensors import safe_open
    from torch import nn
    from diffusers import AutoencoderKL, SD3Transformer2DModel
    from ilm.image_edit.native_flow_step import NativeLowRankLinear, add_adapters, trainable
    module=vs.source_module()
    with torch.device('meta'):
        t=SD3Transformer2DModel.from_config(json.loads((vs.SNAPSHOT/'transformer/config.json').read_text()))
        v=AutoencoderKL.from_config(json.loads((vs.SNAPSHOT/'vae/config.json').read_text()))
        pipe=module.DeepGenPipeline(transformer=t,vae=v,scheduler=module.FlowMatchEulerDiscreteScheduler.from_config(
            json.loads((vs.SNAPSHOT/'scheduler/scheduler_config.json').read_text())))
        pipe.connector_module=vs.connector(module,meta=True)
    pipe.transformer.to(dtype=torch.bfloat16);pipe.connector_module.to(dtype=torch.bfloat16)
    layers=add_adapters(pipe,16);params=trainable(pipe)
    pipe.transformer.enable_gradient_checkpointing();pipe.connector_module.connector.gradient_checkpointing=True
    if not pipe.transformer.gradient_checkpointing or not all(p.dtype==torch.float32 for p in params.values()):
        raise ValueError('native checkpointing or FP32 adapter masters unavailable')
    counts={name:sum(p.numel() for n,p in params.items() if n.startswith(name+'.')) for name in layers}
    layer_checks=[]
    with safe_open(str(vs.SNAPSHOT/'transformer/diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as handle:
        for i,name in enumerate(('context_embedder','proj_out')):
            w=handle.get_tensor(name+'.weight');b=handle.get_tensor(name+'.bias')
            base=nn.Linear(w.shape[1],w.shape[0],bias=True,dtype=w.dtype)
            with torch.no_grad():base.weight.copy_(w);base.bias.copy_(b)
            frozen=base.weight.detach().clone()
            torch.manual_seed(20269301+i)
            adapter=NativeLowRankLinear(base,16)
            x=torch.randn((2,w.shape[1]),generator=torch.Generator().manual_seed(20269303+i),dtype=w.dtype)
            expected=base(x);actual=adapter(x)
            if not torch.equal(expected,actual):raise AssertionError('zero adapter changes real native projection')
            actual.float().square().mean().backward()
            if adapter.up.grad is None or not torch.isfinite(adapter.up.grad).all() or not adapter.up.grad.any():
                raise AssertionError('actual projection did not reach finite adapter gradient')
            opt=torch.optim.AdamW([adapter.down,adapter.up],lr=5e-5,foreach=False);opt.step()
            if not torch.equal(frozen,base.weight) or not adapter.up.detach().any():raise AssertionError('projection update invariant failed')
            layer_checks.append(dict(source_projection=name,actual_released_weights_used=True,
                zero_adapter_output_exact=True,finite_nonzero_up_gradient=True,adapter_weights_changed=True,
                frozen_source_weight_unchanged=True,native_sampling_dtype=str(w.dtype),autocast_required=False))
    data=json.loads((DATA/'report.json').read_text())
    if sha(DATA/'image_pairs.jsonl')!=data['image_manifest_sha256']:raise ValueError('actual image manifest changed')
    rows=[json.loads(x) for x in (DATA/'image_pairs.jsonl').read_text().splitlines()]
    qa=[r for r in rows if r['split']=='train' and r['task']=='qa'][:2]
    copy=[next(r for r in rows if r['split']=='train' and r['task']=='copy' and r['language']==lang) for lang in ('zh','en')]
    chosen=[*qa,*copy]
    if len(chosen)!=4:raise ValueError('complete fixed codec cases required')
    for row in chosen:
        for key in ('input','target'):
            if sha(DATA/row[key])!=row[key+'_sha256']:raise ValueError('registered input/target pixels changed')
    baseline=json.loads((CANDIDATE/'inputs.json').read_text())[0]
    if baseline['input_sha256']!=qa[0]['input_sha256']:raise ValueError('zero-adapter replay must use original first native case')
    result=dict(code_sha256=code,image_manifest_sha256=data['image_manifest_sha256'],
        selected_cases=chosen,native_baseline_case=baseline,rank=16,adapter_layer_names=layers,
        adapter_parameter_counts=counts,adapter_trainable_parameters=sum(counts.values()),
        native_checkpointing_enabled_on_meta=True,real_projection_checks=layer_checks,
        complete_generation_model_loaded=False,actual_source_projection_weights_loaded=True,full_model_forward_exercised=False,
        CPU_projection_optimizer_steps=2,model_optimizer_steps=0,
        cuda_initialized=torch.cuda.is_initialized(),complete_native_replay_required=True,
        planned_codec_decodes=8,planned_autonomous_generations=2,planned_model_optimizer_steps=1,
        learning_rate=5e-5,betas=[.9,.95],weight_decay=.05,clip_norm=1.,
        condition_manifest_sha256=sha(CONDITION/'manifest.json'),condition_payload_sha256=sha(CONDITION/'condition.safetensors'),
        language_learning_established=False,general_ilm_goal_complete=False)
    if identity()!=code:raise ValueError('preparation sources changed')
    save_json(output/'preparation.json',result)
    print(json.dumps(dict(adapter_parameter_counts=counts,projection_checks=layer_checks,full_model_forward_exercised=False),indent=2))


def run(output,wait_pid):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('admit only GPU0')
    prepared=json.loads((output/'preparation.json').read_text());code=identity()
    if prepared['code_sha256']!=code:raise ValueError('registered preparation implementation changed')
    runtime=output/'execution';runtime.mkdir(exist_ok=False)
    expected=owner(wait_pid) if wait_pid else None
    if wait_pid and (expected is None or 'scripts/preflight_visual_semantic.py' not in expected['command'] or CANDIDATE.name not in expected['command']):
        raise ValueError('exact currently live native candidate owner required')
    phase='waiting_candidate';started=time.time()
    def unchanged():
        if identity()!=code or sha(DATA/'image_pairs.jsonl')!=prepared['image_manifest_sha256']:
            raise ValueError('registered learning source or image data changed')
    def status(**extra):save_json(runtime/'runtime.json',dict(phase=phase,
        pid=None if phase in ('failed','ready_for_raw_review') else os.getpid(),wait_owner=expected,
        code_sha256=code,gpu=None if phase=='waiting_candidate' else 0,novnc_url=None,
        general_ilm_goal_complete=False,**extra))
    def interrupted(signum,frame):raise InterruptedError('flow preflight signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupted)
    try:
        status()
        while expected is not None and owner(wait_pid)==expected:unchanged();time.sleep(15)
        terminal=json.loads((CANDIDATE/'generation/runtime.json').read_text())
        baseline=json.loads((CANDIDATE/'generation/report.json').read_text()) if (CANDIDATE/'generation/report.json').exists() else {}
        if terminal['phase']!='ready_for_raw_review' or baseline.get('complete_original_generations')!=12:
            raise ValueError('native candidate ended without complete actual replay and ten pages')
        for name in ('native_compiled_replay','native_pruned_replay'):
            r=baseline[name]
            if not r['decoded_tensor_exact'] or not r['png_pixels_exact'] or not all(r['forward_fields_exact'].values()):
                raise ValueError('complete native numerical equivalence required')
        unchanged();phase='verifying_source';status()
        if vs.verify_release()!=baseline['source']:raise ValueError('source release changed after native replay')
        for name,key in (('manifest.json','condition_manifest_sha256'),('condition.safetensors','condition_payload_sha256')):
            if sha(CONDITION/name)!=prepared[key]:raise ValueError('fixed condition changed')
        with gpu_zero() as device:
            import torch
            import numpy as np
            from PIL import Image
            from diffusers import FlowMatchEulerDiscreteScheduler
            from ilm.image_edit.native_flow_step import add_adapters,adapter_state,trainable,pixels,frozen_input_features,encode_target
            from ilm.image_edit.visual_flow_learning import sample_flow,denoising_loss
            torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            phase='loading';status();pipe,loader=vs.load_pipeline(device)
            vs.install_compiled(pipe,CONDITION);removal=vs.remove_lexical_modules(pipe)
            torch.manual_seed(20269301);layers=add_adapters(pipe,16)
            if layers!=prepared['adapter_layer_names']:raise ValueError('actual native adapter layers differ from meta check')
            selected=trainable(pipe)
            if sum(p.numel() for p in selected.values())!=prepared['adapter_trainable_parameters']:
                raise ValueError('actual trainable parameter count differs')
            save_json(runtime/'loader.json',dict(loader=loader,removal=removal,adapter_layers=layers))
            first=prepared['selected_cases'][0];seed=prepared['native_baseline_case']['seed']
            def checked_path(row,key):
                p=DATA/row[key]
                if sha(p)!=row[key+'_sha256']:raise ValueError('actual registered image changed')
                return p
            phase='zero_adapter_replay';status()
            before=vs.generate(pipe,checked_path(first,'input'),seed=seed,width=1024,height=1024)
            vs.save_generation(before,runtime/'before')
            saved=torch.load(CANDIDATE/'generation/000/actual_tensors.pt',map_location='cpu',weights_only=True)
            with Image.open(CANDIDATE/'generation/000/answer.png') as image:native=dict(image=image.copy(),**saved)
            replay=vs.exact_replay(native,before);save_json(runtime/'zero_adapter_replay.json',replay)
            if not replay['decoded_tensor_exact'] or not replay['png_pixels_exact'] or not all(replay['forward_fields_exact'].values()):
                raise ValueError('zero adapters change actual native generation')
            del before,native,saved
            phase='codec';status();codec=[]
            for i,row in enumerate(prepared['selected_cases']):
                target=checked_path(row,'target');pv=pixels(pipe,target)
                with torch.no_grad():distribution=pipe.vae.encode(pv[None]).latent_dist
                gold=np.array(Image.open(target).convert('RGB'));mask=gold.mean(2)<128
                for kind in ('mode','sample'):
                    torch.manual_seed(20269320+i);torch.cuda.manual_seed(20269320+i)
                    with torch.no_grad():
                        z=distribution.mode() if kind=='mode' else distribution.sample()
                        z=(z-pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor
                        decoded=pipe.latents_to_pixels(z)
                    raw=torch.clamp(127.5*decoded[0].permute(1,2,0)+128.,0,255).to('cpu',dtype=torch.uint8).numpy()
                    value=dict(image=Image.fromarray(raw),pixels=decoded.detach().cpu().contiguous(),forward_inputs={})
                    saved=vs.save_generation(value,runtime/'codec'/f'{i:02}_{kind}')
                    m=raw.mean(2)<128;tp=int((mask&m).sum());fp=int((~mask&m).sum());fn=int((mask&~m).sum())
                    if not np.array_equal(raw,torch.clamp(127.5*value['pixels'][0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()):
                        raise ValueError('codec PNG differs from real decoder tensor')
                    codec.append(dict(id=row['id'],kind=kind,target_sha256=row['target_sha256'],
                        foreground_F1=2*tp/max(1,2*tp+fp+fn),predicted_to_target_ink_ratio=int(m.sum())/max(1,int(mask.sum())),
                        target_black_pixels=int(mask.sum()),saved=saved))
                    save_json(runtime/'codec.json',codec)
                del distribution,pv,z,decoded,value
            phase='one_forward_learning';status();unchanged()
            for name in ('transformer','connector_module'):getattr(pipe,name).train()
            pipe.transformer.enable_gradient_checkpointing();pipe.connector_module.connector.gradient_checkpointing=True
            frozen_versions={component+'.'+n:p._version for component in ('transformer','connector_module','lmm','vae')
                for n,p in getattr(pipe,component).named_parameters() if not p.requires_grad}
            initial=adapter_state(pipe)
            optimizer=torch.optim.AdamW(list(selected.values()),lr=5e-5,betas=(.9,.95),weight_decay=.05,foreach=False)
            torch.manual_seed(20269310);torch.cuda.manual_seed(20269310)
            tick=time.perf_counter();merged,reference,feature_meta=frozen_input_features(pipe,checked_path(first,'input'))
            target=encode_target(pipe,checked_path(first,'target'))
            pooled,sequence=pipe.connector_module.llm2dit(merged)
            schedule=FlowMatchEulerDiscreteScheduler.from_pretrained(str(vs.SNAPSHOT/'scheduler'),local_files_only=True)
            flow=sample_flow(target,schedule,time_generator=torch.Generator(device='cpu').manual_seed(20269311),
                noise_generator=torch.Generator(device=device).manual_seed(20269312))
            loss,prediction=denoising_loss(pipe.transformer,flow,sequence_condition=sequence[:1],
                pooled_condition=pooled[:1],reference_latents=[[reference]])
            if not loss.requires_grad:raise ValueError('actual native denoising loss has no trainable path')
            loss.backward()
            grad_counts={component:sum(p.grad is not None and bool(p.grad.abs().max()>0) for n,p in selected.items() if n.startswith(component+'.'))
                for component in ('transformer','connector_module')}
            if not all(grad_counts.values()) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in selected.values()):
                raise FloatingPointError('actual native learning lacks finite nonzero gradients in both components')
            norm=torch.nn.utils.clip_grad_norm_(list(selected.values()),1.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            changed=[n for n,p in selected.items() if not torch.equal(initial[n],p.detach().cpu())]
            if not changed:raise ValueError('real optimizer step did not change any adapter')
            current_versions={component+'.'+n:p._version for component in ('transformer','connector_module','lmm','vae')
                for n,p in getattr(pipe,component).named_parameters() if not p.requires_grad}
            if frozen_versions!=current_versions:raise ValueError('frozen parameter version changed')
            torch.save(dict(clean_target_latents=target.detach().cpu(),noisy_latents=flow.noisy_latents.detach().cpu(),
                target_velocity=flow.velocity.detach().cpu(),predicted_velocity=prediction.detach().cpu(),
                timesteps=flow.timesteps.detach().cpu(),sigmas=flow.sigmas.detach().cpu(),reference_image_latents=reference.detach().cpu()),runtime/'actual_training_tensors.pt')
            torch.save(dict(adapters=adapter_state(pipe),optimizer=optimizer.state_dict(),optimizer_step=1,
                metadata=prepared,torch_cpu_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state()),runtime/'checkpoint.pt')
            learning=dict(actual_optimizer_updates=1,actual_logical_denoiser_forwards=1,loss=float(loss.detach()),
                gradient_norm_before_clip=float(norm),finite_nonzero_gradient_tensors_by_component=grad_counts,
                changed_adapter_tensors=len(changed),changed_adapter_names=changed,frozen_parameter_versions_unchanged=True,
                input_sha256=first['input_sha256'],target_sha256=first['target_sha256'],feature_metadata=feature_meta,
                timestep_indices=flow.indices.tolist(),timesteps=flow.timesteps.detach().cpu().tolist(),
                sigmas=flow.sigmas.detach().cpu().float().flatten().tolist(),elapsed_seconds=elapsed,
                checkpoint_sha256=sha(runtime/'checkpoint.pt'),language_learning_established=False)
            save_json(runtime/'learning.json',learning)
            del target,reference,merged,pooled,sequence,flow,prediction,loss,optimizer,initial
            for p in selected.values():p.grad=None
            for name in ('transformer','connector_module'):getattr(pipe,name).eval()
            phase='autonomous_after_update';status()
            after=vs.generate(pipe,checked_path(first,'input'),seed=seed,width=1024,height=1024)
            vs.save_generation(after,runtime/'after')
            result=dict(preparation_sha256=sha(output/'preparation.json'),candidate_report_sha256=sha(CANDIDATE/'generation/report.json'),
                zero_adapter_replay=replay,actual_codec_decodes=len(codec),codec_scores=codec,learning=learning,
                actual_autonomous_generations=2,all_generated_PNGs_authenticated_against_real_decoder_tensors=True,
                actual_trainable_parameters=sum(p.numel() for p in selected.values()),
                peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_vram_gib=torch.cuda.max_memory_reserved()/1024**3,elapsed_seconds=time.time()-started,
                retained_language_body_pretraining=True,runtime_lexical_vocabulary_or_OCR=False,
                training_step_is_feasibility_only=True,raw_review_completed=False,long_training_quality_qualified=False,
                language_learning_established=False,general_ilm_goal_complete=False)
            save_json(runtime/'report.json',result);unchanged()
        phase='ready_for_raw_review';status(owned_GPU_job_finished=True)
    except BaseException as error:
        phase='failed';status(exception=type(error).__name__,reason=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-for-candidate-pid',type=int)
    args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    prepare(args.output.resolve()) if args.stage=='prepare' else run(args.output.resolve(),args.wait_for_candidate_pid)
