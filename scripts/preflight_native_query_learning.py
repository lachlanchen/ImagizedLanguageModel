#!/usr/bin/env python3
"""Run two real continuous-query updates and two native image answers."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha

FILES = ('ilm/image_edit/native_query_learning.py','scripts/preflight_native_query_learning.py',
         'references/native_query_learning_preflight_protocol.md')
TRAINING = ROOT/'artifacts/native_visual_flow_learning_v1'
PROBE = ROOT/'artifacts/native_flow_condition_probe_v1'
SAMPLING = ROOT/'artifacts/native_flow_positive_sampling_probe_v1'


def require(value,message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def identity():
    require(not subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT,text=True),
            'commit the native query preflight sources first')
    return {name:sha(ROOT/name) for name in FILES}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for preparation')
    code = identity()
    require(load(PROBE/'runtime.json')['phase']=='completed'
            and load(SAMPLING/'runtime.json')['phase']=='completed','complete actual predecessors required')
    require(load(PROBE/'result.json')['actual_denoiser_forwards']==144
            and load(SAMPLING/'result.json')['actual_autonomous_generations']==10,'complete predecessor calls required')
    cases = load(PROBE/'cases.json')[:2]
    require([r['id'] for r in cases]==['TRAIN_3802_QUERY_0','TRAIN_3802_QUERY_1']
            and all(r['split']=='train' and r['variant']=='normal' for r in cases),'fixed first seen pair required')
    prior = load(SAMPLING/'final/outputs.json')[:2]
    require(all(a['id']==b['id'] and a['own_sha256']==b['input_sha256'] for a,b in zip(cases,prior)),
            'matched original pre-update sampling cases differ')
    original = load(SAMPLING/'preparation.json')
    sources = {}
    for key in ['training_code_sha256','assessment_code_sha256','condition_probe_code_sha256','code_sha256']:
        sources.update(original[key])
    require(len(sources)==19,'all19 preserved source bindings required')
    for name,expected in sources.items():
        require(sha(ROOT/name)==expected,'preserved predecessor source changed')
    deployment = TRAINING/'deployment/deployment.json'
    info = load(deployment)
    require(sha(TRAINING/'deployment/adapters.safetensors')==info['weights_sha256'],'final adapter export changed')
    bound = dict(sources)
    for path in [deployment,PROBE/'preparation.json',PROBE/'cases.json',PROBE/'final/outputs.json',
                 PROBE/'result.json',SAMPLING/'preparation.json',SAMPLING/'final/outputs.json',SAMPLING/'result.json']:
        bound[str(path.relative_to(ROOT))]=sha(path)
    field_rows = load(PROBE/'final/outputs.json')[:2]
    for a,b in zip(cases,field_rows):
        require(all(a[k]==b[k] for k in a),'matched original field case differs')
        require(sha(PROBE/'final'/b['actual_tensor_file'])==b['actual_tensor_file_sha256'],'original field tensors changed')
    for row in cases:
        for key in ('own','other','target'):
            require(sha(row[key])==row[key+'_sha256'],'actual fixed image changed')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'cases.json',cases)
    save_json(output/'preparation.json',dict(code_sha256=code,bound_predecessors_sha256=bound,
        deployment_sha256=sha(deployment),cases_sha256=sha(output/'cases.json'),
        final_weights_sha256=info['weights_sha256'],updates=2,learning_rate=5e-5,betas=[.9,.95],
        query_shape=[128,2048],query_trainable_parameters=262144,weight_decay=0.,clip_norm=1.,
        native_grid_index=0,noise_level_exactly_one=True,scheduled_autonomous_images=2,
        guidance=1.,steps=50,canvas=[1024,1024],maximum_allocated_vram_GiB=23.,
        retained_pretrained_reader=True,external_teacher_or_OCR_in_training=False,
        model_loaded=False,general_ilm_goal_complete=False))
    print('Prepared2 real shared-query updates and2 autonomous native image answers; no model loaded.')


def unchanged(output):
    meta = load(output/'preparation.json')
    require(identity()==meta['code_sha256'],'query preflight source changed')
    require(sha(output/'cases.json')==meta['cases_sha256'],'fixed cases changed')
    for name,expected in meta['bound_predecessors_sha256'].items():
        require(sha(ROOT/name)==expected,'bound predecessor changed: '+name)
    require(sha(TRAINING/'deployment/adapters.safetensors')==meta['final_weights_sha256'],'final weights changed')
    return meta


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','use GPU0 only')
    meta = unchanged(output)
    execution = output/'execution'
    execution.mkdir(exist_ok=False)
    phase = 'loading'
    started = time.perf_counter()

    def status(**extra):
        save_json(execution/'runtime.json',dict(phase=phase,pid=None if phase in ('failed','completed') else os.getpid(),
            gpu=0,novnc_url=None,general_ilm_goal_complete=False,**extra))

    def stop(signum,frame):
        raise InterruptedError('query learning signal '+str(signum))

    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        status()
        with gpu_zero() as device:
            import torch
            from safetensors.torch import load_file
            from diffusers import FlowMatchEulerDiscreteScheduler
            from ilm.image_edit import visual_semantic as vs
            from ilm.image_edit.native_flow_run import CONDITION, restore_adapters
            from ilm.image_edit.native_flow_step import add_adapters, frozen_input_features, encode_target
            from ilm.image_edit.native_query_learning import input_features
            from ilm.image_edit.visual_flow_learning import interpolate, denoising_loss
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            source = vs.verify_release()
            pipe, loader = vs.load_pipeline(device)
            vs.install_compiled(pipe,CONDITION)
            removal = vs.remove_lexical_modules(pipe)
            info = load(TRAINING/'deployment/deployment.json')
            require(source==info['source_sha256'],'exact native model release required')
            torch.manual_seed(20270001)
            add_adapters(pipe,info['rank'])
            restore_adapters(pipe,load_file(str(TRAINING/'deployment/adapters.safetensors'),device='cpu'))
            for part in ('lmm','vae','transformer','connector_module'):
                getattr(pipe,part).eval().requires_grad_(False)
            master = torch.nn.Parameter(pipe.connector_module.meta_queries.detach().float().clone())
            require(list(master.shape)==meta['query_shape'] and master.numel()==meta['query_trainable_parameters'],
                    'actual native query shape differs')
            initial = master.detach().cpu().clone()
            cases = load(output/'cases.json')
            first = cases[0]
            torch.manual_seed(first['feature_seed']);torch.cuda.manual_seed(first['feature_seed'])
            original_features, original_reference, _ = frozen_input_features(pipe,Path(first['own']))
            torch.manual_seed(first['feature_seed']);torch.cuda.manual_seed(first['feature_seed'])
            with torch.no_grad():
                equivalent, reference = input_features(pipe,Path(first['own']),master)
            require(torch.equal(original_features,equivalent) and torch.equal(original_reference,reference),
                    'zero-change query helper does not reproduce native features exactly')
            del equivalent, reference, original_reference
            pipe.llm.train()
            pipe.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            pipe.transformer.train();pipe.transformer.enable_gradient_checkpointing()
            pipe.connector_module.train();pipe.connector_module.connector.gradient_checkpointing=True
            versions={part+'.'+name:p._version for part in ('lmm','vae','transformer','connector_module')
                for name,p in getattr(pipe,part).named_parameters()}
            optimizer = torch.optim.AdamW([master],lr=meta['learning_rate'],betas=tuple(meta['betas']),
                                         weight_decay=0.,foreach=False)
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(vs.SNAPSHOT/'scheduler'),local_files_only=True)
            require(float(scheduler.sigmas[0])==1.,'native sigma-one grid required')
            history=[];phase='training';status()
            for step,row in enumerate(cases,1):
                unchanged(output)
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(row['feature_seed']);torch.cuda.manual_seed(row['feature_seed'])
                merged, reference = input_features(pipe,Path(row['own']),master)
                require(merged.requires_grad,'reader features are detached from query masters')
                if step==1:
                    require(torch.equal(merged.detach(),original_features),'differentiated/checkpointed feature forward differs')
                    del original_features
                pooled,sequence = pipe.connector_module.llm2dit(merged)
                torch.manual_seed(row['codec_seed']);torch.cuda.manual_seed(row['codec_seed'])
                clean = encode_target(pipe,Path(row['target']))
                noise = torch.randn(clean.shape,device=device,dtype=clean.dtype,
                                    generator=torch.Generator(device=device).manual_seed(row['noise_seed']))
                sigma = scheduler.sigmas[:1].to(device=device,dtype=clean.dtype).reshape(1,1,1,1)
                timestep = scheduler.timesteps[:1].to(device)
                flow = interpolate(clean,noise,sigma,timestep,torch.tensor([0]))
                require(torch.equal(flow.noisy_latents,noise),'full-noise writer input contains a target')
                loss,prediction = denoising_loss(pipe.transformer,flow,sequence_condition=sequence[:1],
                    pooled_condition=pooled[:1],reference_latents=[[reference]])
                old = torch.load(PROBE/'final'/f'{step-1:03}.pt',map_location='cpu',weights_only=True)
                require(torch.equal(clean.cpu(),old['clean']) and torch.equal(noise.cpu(),old['noise']),
                        'registered codec/common-noise tensors do not reproduce')
                if step==1:
                    require(torch.equal(prediction.detach().cpu(),old['point0_own']),
                            'zero-query-change native field replay differs')
                loss.backward()
                require(master.grad is not None and torch.isfinite(master.grad).all() and bool(master.grad.abs().max()>0),
                        'real native loss does not reach a finite nonzero query gradient')
                gradient = master.grad.detach().cpu().clone()
                norm = torch.nn.utils.clip_grad_norm_([master],meta['clip_norm'],error_if_nonfinite=True)
                optimizer.step()
                require(torch.isfinite(master).all(),'query optimizer produced nonfinite parameters')
                require(torch.cuda.max_memory_allocated()/1024**3 < meta['maximum_allocated_vram_GiB'],'VRAM admission exceeded')
                tensor_path=execution/f'update_{step:03}.pt'
                torch.save(dict(query_gradient=gradient,queries_after=master.detach().cpu(),clean=clean.cpu(),noise=noise.cpu(),
                    noisy=flow.noisy_latents.cpu(),target_velocity=flow.velocity.cpu(),predicted_velocity=prediction.detach().cpu()),tensor_path)
                history.append(dict(step=step,id=row['id'],loss=float(loss.detach()),gradient_norm=float(norm),
                    finite_nonzero_query_gradient=True,gradient_max=float(gradient.abs().max()),
                    actual_tensor_file=tensor_path.name,actual_tensor_sha256=sha(tensor_path)))
                save_json(execution/'updates.json',history)
                status(completed_updates=step)
                del old,merged,reference,pooled,sequence,clean,noise,flow,prediction,loss
                print(f'Completed query update{step}/2',flush=True)
            require(not torch.equal(initial,master.detach().cpu()),'query masters did not change')
            actual_versions={part+'.'+name:p._version for part in ('lmm','vae','transformer','connector_module')
                for name,p in getattr(pipe,part).named_parameters()}
            require(versions==actual_versions,'a frozen model weight changed during query optimization')
            torch.save(dict(queries=master.detach().cpu(),initial_queries=initial,optimizer=optimizer.state_dict(),
                completed_updates=2,metadata=meta),execution/'checkpoint.pt')
            with torch.no_grad():
                pipe.connector_module.meta_queries.copy_(master.to(dtype=pipe.transformer.dtype))
            for part in ('lmm','vae','transformer','connector_module'):
                getattr(pipe,part).eval()
            pipe.llm.gradient_checkpointing_disable()
            prior=load(SAMPLING/'final/outputs.json')[:2]
            generated=[];phase='sampling';status()
            for i,(row,before) in enumerate(zip(cases,prior)):
                unchanged(output)
                value=vs.generate(pipe,Path(row['own']),seed=before['seed'],width=1024,height=1024,steps=50,guidance=1.)
                result=vs.save_generation(value,execution/'answers'/f'{i:03}')
                generated.append(dict(id=row['id'],input_sha256=row['own_sha256'],seed=before['seed'],
                    path=f'{i:03}/answer.png',sha256=result['sha256'],result=result))
                save_json(execution/'answers/outputs.json',generated)
            require(torch.cuda.max_memory_allocated()/1024**3 < meta['maximum_allocated_vram_GiB'],'sampling VRAM admission exceeded')
            save_json(execution/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                loader=loader,removal=removal,source=source,query_parameters=master.numel(),actual_optimizer_updates=2,
                native_helper_features_exact=True,differentiated_features_exact=True,initial_native_field_exact=True,
                all_frozen_weight_versions_unchanged_during_updates=True,query_checkpoint_sha256=sha(execution/'checkpoint.pt'),
                actual_autonomous_generations=2,outputs_sha256=sha(execution/'answers/outputs.json'),history=history,
                elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
                retained_runtime_parameters=vs.parameters(pipe),semantic_learning_validated=False,general_ilm_goal_complete=False))
        phase='completed';status()
    except BaseException as error:
        phase='failed';status(reason=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','run'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'run':run}[args.stage](args.output.resolve())
