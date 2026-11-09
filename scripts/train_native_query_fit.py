#!/usr/bin/env python3
"""Bounded shared-reader-query learning and autonomous native image assessment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from scripts.preflight_native_query_learning import PROBE, SAMPLING, TRAINING, require, load

FILES = ('scripts/train_native_query_fit.py', 'references/native_query_fit_protocol.md')
PREFLIGHT = ROOT/'artifacts/native_query_learning_preflight_v1'


def identity():
    require(not subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT),
            'commit the query-fit implementation and protocol first')
    return {name:sha(ROOT/name) for name in FILES}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for preparation')
    code = identity()
    previous = load(PREFLIGHT/'preparation.json')
    report = load(PREFLIGHT/'execution/report.json')
    require(load(PREFLIGHT/'execution/runtime.json')['phase']=='completed'
            and report['actual_optimizer_updates']==2, 'completed query preflight required')
    bound = {**previous['code_sha256'], **previous['bound_predecessors_sha256']}
    for path in [PREFLIGHT/'preparation.json', PREFLIGHT/'execution/report.json']:
        bound[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in bound.items():
        require(sha(ROOT/name)==digest, 'bound predecessor changed: '+name)
    train = load(PREFLIGHT/'cases.json')
    source_rows = {}
    for variant in ('normal','question_erased','passage_erased','copy'):
        path = TRAINING/'evaluation'/(variant+'_references.jsonl')
        bound[str(path.relative_to(ROOT))] = sha(path)
        source_rows[variant] = [dict(row, **{key:str((path.parent/row[key]).resolve())
            for key in ('input','target')}) for row in map(json.loads,path.read_text().splitlines())]
    normal = [source_rows['normal'][i] for i in (0,1,32,33)]
    require([r['id'] for r in normal[:2]]==[r['id'] for r in train], 'first seen pair differs')
    require([r['split'] for r in normal]==['train','train','validation','validation'], 'matched splits differ')
    copies = [next(r for r in source_rows['copy'] if r['split']==split and r['language']==language)
        for split in ('train','validation') for language in ('zh','en')]
    prior = load(SAMPLING/'final/outputs.json')
    require([r['id'] for r in prior]==[r['id'] for r in normal], 'original normal baselines differ')
    rng = random.Random(20270101)
    schedule = []
    for cycle in range(64):
        index = 0 if cycle%8==0 else rng.randrange(1,1000)
        for i in (0,1):
            schedule.append(dict(step=len(schedule)+1,case=i,cycle=cycle,grid_index=index,
                noise_seed=train[0]['noise_seed'] if cycle==0 else 20270200+cycle))
    require(sum(r['grid_index']==0 for r in schedule)==16, 'full-noise anchor count differs')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'training_cases.json',train)
    save_json(output/'schedule.json',schedule)
    plans, manifests = {}, {}
    def cohort(name, rows, seeds):
        plans[name] = [dict(id=row['id'],input=row['input'],input_sha256=row['input_sha256'],seed=seed)
            for row,seed in zip(rows,seeds)]
        manifests[name] = rows
    seeds = [r['seed'] for r in prior]
    cohort('initial_copy',copies,[20270301,20270302,20270303,20270304])
    cohort('final_normal',normal,seeds)
    cohort('final_copy',copies,[20270301,20270302,20270303,20270304])
    cohort('final_new_noise',normal[:2],[20270401,20270401])
    # Reproduce the original input-only paragraph geometry before masking.
    import numpy as np
    from PIL import Image, ImageDraw
    from ilm.image_edit.data import display_text, font_and_coverage, page
    pages=ROOT/'artifacts/image_edit_learning/pages_v2'
    source_path=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    for path in (pages/'pairs.jsonl',pages/'preparation.json',source_path,ROOT/'ilm/image_edit/data.py'):
        bound[str(path.relative_to(ROOT))]=sha(path)
    page_prep=load(pages/'preparation.json')
    require(sha(pages/'pairs.jsonl')==page_prep['manifest_sha256'], 'original input manifest changed')
    require(sha(source_path)==load(PROBE/'preparation.json')['source_QA_sha256'], 'source QA changed')
    original={r['id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    prompts=[]
    for row in normal[:2]:
        source=original[row['id']]
        context,question=display_text(source['context']),display_text(source['question'])
        prompts.append(dict(prompt=f'资料：{context}\n问题：{question}\n回答：',reference='',context=context,question=question))
    font, provenance=font_and_coverage(prompts)
    require(all(provenance[k]==page_prep['font'][k] for k in
        ('font_file_sha256','font_face_index','font_size','charset_sha256')), 'original input font changed')
    arrays=[];question_tops=[];question_lines=[]
    for row,source in zip(normal[:2],prompts):
        rendered,_=page(source['prompt'],(1024,1024),font)
        _,context_geometry=page('资料：'+source['context'],(1024,1024),font)
        _,question_geometry=page('问题：'+source['question'],(1024,1024),font)
        with Image.open(row['input']) as image:actual=np.asarray(image.convert('RGB')).copy()
        require(np.array_equal(np.asarray(rendered),actual), 'complete original seen page does not reproduce')
        arrays.append(actual);question_tops.append(32+48*context_geometry['lines'])
        question_lines.append(question_geometry['lines'])
    require(question_tops[0]==question_tops[1] and prompts[0]['context']==prompts[1]['context'],
        'complete same-passage pair required')
    top=question_tops[0];answer_y=top+48*max(question_lines)
    controls=output/'controls';controls.mkdir()
    erased=Image.fromarray(arrays[0]);erased.paste('white',(0,top,1024,1024))
    draw=ImageDraw.Draw(erased)
    draw.text((32,top),'问题：',font=font,anchor='lt',fill='black')
    draw.text((32,answer_y),'回答：',font=font,anchor='lt',fill='black')
    question_path=controls/'question_erased.png';erased.save(question_path)
    require(all(np.array_equal(np.asarray(erased)[:top],a[:top]) for a in arrays), 'question erasure changed passage pixels')
    for kind in ('question_erased','passage_erased'):
        rows = []
        for i, row in enumerate(normal[:2]):
            path=question_path
            if kind=='passage_erased':
                image=Image.fromarray(arrays[i]);image.paste('white',(0,0,1024,top))
                ImageDraw.Draw(image).text((32,32),'资料：',font=font,anchor='lt',fill='black')
                require(np.array_equal(np.asarray(image)[top:],arrays[i][top:]), 'passage erasure changed question pixels')
                path=controls/f'passage_erased_{i}.png';image.save(path)
            rows.append(dict(row,input=str(path),input_sha256=sha(path)))
        cohort('final_'+kind,rows,seeds[:2])
    require(plans['final_question_erased'][0]['input_sha256']==plans['final_question_erased'][1]['input_sha256']
        and seeds[0]==seeds[1], 'erased question pair or common seed differs')
    save_json(output/'inference.json',plans)
    for name, rows in manifests.items():
        (output/(name+'_references.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    baseline = [dict(r,path=str(SAMPLING/'final'/r['path'])) for r in prior]
    save_json(output/'initial_normal_outputs.json',baseline)
    registered = ['training_cases.json','schedule.json','inference.json','initial_normal_outputs.json',
        *(name+'_references.jsonl' for name in manifests),
        *(str(p.relative_to(output)) for p in sorted(controls.glob('*.png')))]
    for row in train:
        for key in ('own','other','target'):
            require(sha(row[key])==row[key+'_sha256'], 'actual training page changed')
    for rows in plans.values():
        for row in rows:
            require(sha(row['input'])==row['input_sha256'], 'actual evaluation page changed')
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sources_sha256=bound,
        files_sha256={p:sha(output/p) for p in registered},updates=128,learning_rate=5e-5,
        betas=[.9,.95],weight_decay=0.,clip_norm=1.,query_shape=[128,2048],query_parameters=262144,
        full_noise_updates=16,native_other_noise_updates=112,scheduled_new_generations=18,
        reused_original_baseline_QA_images=4,maximum_allocated_vram_GiB=23.,
        maximum_execution_seconds=2700,maximum_wait_seconds=7200,poll_seconds=30,
        source_final_weights_sha256=previous['final_weights_sha256'],
        seen_controls_question_y=top,question_erased_common_answer_y=answer_y,
        original_seen_pages_reproduced_exactly=True,erasure_outside_pixels_preserved=True,
        reused_development_data=True,semantic_learning_validated=False,general_ilm_goal_complete=False))
    print('Prepared128 query updates and18 native image generations; no model loaded.')


def unchanged(output):
    meta = load(output/'preparation.json')
    require(identity()==meta['code_sha256'], 'query-fit sources changed')
    for name,digest in meta['bound_sources_sha256'].items():
        require(sha(ROOT/name)==digest, 'predecessor binding changed: '+name)
    for name,digest in meta['files_sha256'].items():
        require(sha(output/name)==digest, 'prepared file changed: '+name)
    require(sha(TRAINING/'deployment/adapters.safetensors')==meta['source_final_weights_sha256'],
            'source final adapters changed')
    return meta


def resources():
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    swap=(memory['SwapTotal']-memory['SwapFree'])/memory['SwapTotal'] if memory['SwapTotal'] else 0.
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=10).strip())
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);available=True
        except BlockingIOError:
            available=False
    return dict(ready=memory['MemAvailable']>=24*1024**2 and swap<=.75 and free>=20*1024 and available,
        available_RAM_GiB=memory['MemAvailable']/1024**2,swap_used_fraction=swap,
        GPU0_free_MiB=free,project_lock_available=available)


def wait(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'use GPU0 only')
    meta=unchanged(output);started=time.monotonic()
    require(not (output/'execution').exists(), 'an execution attempt already exists')
    with (output/'wait.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            unchanged(output);observed=resources()
            if observed['ready']:
                save_json(output/'wait.json',dict(phase='admitted',pid=None,resources=observed))
                child=None
                def stop(signum,frame):
                    raise InterruptedError('query-fit coordinator signal '+str(signum))
                for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
                try:
                    child=subprocess.Popen([sys.executable,__file__,'--stage','run','--output',str(output)],
                        cwd=ROOT,start_new_session=True)
                    save_json(output/'wait.json',dict(phase='model_running',pid=os.getpid(),child_pid=child.pid))
                    child.wait()
                    require(child.returncode==0, 'query-fit model child failed; no automatic retry')
                    save_json(output/'wait.json',dict(phase='offline_scoring',pid=os.getpid(),child_pid=None))
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
                    subprocess.run([sys.executable,__file__,'--stage','score','--output',str(output)],
                        cwd=ROOT,env=env,check=True)
                    save_json(output/'wait.json',dict(phase='completed',pid=None,child_pid=None,
                        raw_image_review_required=True,general_ilm_goal_complete=False))
                    return
                except BaseException as error:
                    save_json(output/'wait.json',dict(phase='failed',pid=None,reason=str(error)))
                    raise
                finally:
                    if child is not None and child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            elapsed=time.monotonic()-started
            timed_out=elapsed>=meta['maximum_wait_seconds']
            save_json(output/'wait.json',dict(phase='wait_timeout' if timed_out else 'waiting_resources',
                pid=None if timed_out else os.getpid(),elapsed_seconds=elapsed,resources=observed))
            if timed_out:
                print('Resource wait expired; no training attempt or model loading occurred.',flush=True)
                return
            print('Waiting for existing resource thresholds: '+json.dumps(observed),flush=True)
            time.sleep(meta['poll_seconds'])


def run(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','use GPU0 only')
    meta=unchanged(output);execution=output/'execution';execution.mkdir(exist_ok=False)
    started=time.perf_counter();phase='loading';completed=0
    def status(**extra):
        save_json(execution/'runtime.json',dict(phase=phase,pid=None if phase in ('completed','failed') else os.getpid(),
            completed_updates=completed,gpu=0,general_ilm_goal_complete=False,**extra))
    def budget():
        require(time.perf_counter()-started<meta['maximum_execution_seconds'], 'bounded execution time exceeded')
    def stop(signum,frame):
        if signum==signal.SIGALRM:
            raise TimeoutError('bounded query-fit execution time exceeded')
        raise InterruptedError('query fit signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    signal.signal(signal.SIGALRM,stop);signal.alarm(meta['maximum_execution_seconds'])
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
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            source=vs.verify_release();pipe,loader=vs.load_pipeline(device)
            vs.install_compiled(pipe,CONDITION);removal=vs.remove_lexical_modules(pipe)
            info=load(TRAINING/'deployment/deployment.json')
            require(source==info['source_sha256'], 'native source release differs')
            torch.manual_seed(20270001);add_adapters(pipe,info['rank'])
            restore_adapters(pipe,load_file(str(TRAINING/'deployment/adapters.safetensors'),device='cpu'))
            for part in ('lmm','vae','transformer','connector_module'):
                getattr(pipe,part).eval().requires_grad_(False)
            master=torch.nn.Parameter(pipe.connector_module.meta_queries.detach().float().clone())
            initial=master.detach().cpu().clone()
            require(list(master.shape)==meta['query_shape'], 'native query shape differs')
            plans=load(output/'inference.json');cases=load(output/'training_cases.json')
            def generate(name):
                rows=[];folder=execution/name;folder.mkdir(exist_ok=False)
                for i,row in enumerate(plans[name]):
                    budget();unchanged(output)
                    require(sha(row['input'])==row['input_sha256'], 'inference input changed')
                    value=vs.generate(pipe,Path(row['input']),seed=row['seed'],width=1024,height=1024,steps=50,guidance=1.)
                    saved=vs.save_generation(value,folder/f'{i:03}')
                    budget()
                    rows.append(dict(id=row['id'],input_sha256=row['input_sha256'],seed=row['seed'],
                        path=f'{i:03}/answer.png',sha256=saved['sha256'],result=saved))
                    save_json(folder/'outputs.json',rows)
                    require(torch.cuda.max_memory_allocated()/1024**3<meta['maximum_allocated_vram_GiB'],'VRAM bound exceeded')
                    print('Generated '+name+' '+str(i+1)+'/'+str(len(plans[name])),flush=True)
                return rows
            phase='initial_copy';status();generate('initial_copy')
            first=cases[0]
            torch.manual_seed(first['feature_seed']);torch.cuda.manual_seed(first['feature_seed'])
            ordinary,reference,_=frozen_input_features(pipe,Path(first['own']))
            torch.manual_seed(first['feature_seed']);torch.cuda.manual_seed(first['feature_seed'])
            with torch.no_grad():equivalent,ref=input_features(pipe,Path(first['own']),master)
            require(torch.equal(ordinary,equivalent) and torch.equal(reference,ref), 'native helper feature replay differs')
            del equivalent,reference,ref
            clean_cache=[]
            for row in cases:
                require(sha(row['own'])==row['own_sha256'] and sha(row['target'])==row['target_sha256'],
                    'actual training input or target changed')
                torch.manual_seed(row['codec_seed']);torch.cuda.manual_seed(row['codec_seed'])
                clean_cache.append(encode_target(pipe,Path(row['target'])))
            pipe.llm.train();pipe.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            pipe.transformer.train();pipe.transformer.enable_gradient_checkpointing()
            pipe.connector_module.train();pipe.connector_module.connector.gradient_checkpointing=True
            versions={part+'.'+name:p._version for part in ('lmm','vae','transformer','connector_module')
                for name,p in getattr(pipe,part).named_parameters()}
            optimizer=torch.optim.AdamW([master],lr=meta['learning_rate'],betas=tuple(meta['betas']),weight_decay=0.,foreach=False)
            scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(str(vs.SNAPSHOT/'scheduler'),local_files_only=True)
            require(len(scheduler.sigmas)==1000 and float(scheduler.sigmas[0])==1., 'fresh native training schedule required')
            history=[];phase='training';status()
            for entry in load(output/'schedule.json'):
                budget();unchanged(output);optimizer.zero_grad(set_to_none=True)
                row=cases[entry['case']];index=entry['grid_index'];step=entry['step'];clean=clean_cache[entry['case']]
                require(sha(row['own'])==row['own_sha256'], 'actual training page changed')
                torch.manual_seed(row['feature_seed']);torch.cuda.manual_seed(row['feature_seed'])
                merged,reference=input_features(pipe,Path(row['own']),master)
                require(merged.requires_grad, 'reader condition detached from learned queries')
                if step==1:
                    require(torch.equal(merged.detach(),ordinary), 'checkpointed feature replay differs');del ordinary
                pooled,sequence=pipe.connector_module.llm2dit(merged)
                noise=torch.randn(clean.shape,device=device,dtype=clean.dtype,
                    generator=torch.Generator(device=device).manual_seed(entry['noise_seed']))
                sigma=scheduler.sigmas[index:index+1].to(device=device,dtype=clean.dtype).reshape(1,1,1,1)
                flow=interpolate(clean,noise,sigma,scheduler.timesteps[index:index+1].to(device),torch.tensor([index]))
                if index==0:require(torch.equal(flow.noisy_latents,noise), 'full-noise anchor contains target pixels')
                loss,prediction=denoising_loss(pipe.transformer,flow,sequence_condition=sequence[:1],
                    pooled_condition=pooled[:1],reference_latents=[[reference]])
                if step==1:
                    old=torch.load(PROBE/'final/000.pt',map_location='cpu',weights_only=True)
                    require(torch.equal(clean.cpu(),old['clean']) and torch.equal(noise.cpu(),old['noise'])
                        and torch.equal(prediction.detach().cpu(),old['point0_own']), 'initial native field replay differs')
                    del old
                loss.backward()
                require(master.grad is not None and torch.isfinite(master.grad).all() and bool(master.grad.abs().max()>0),
                    'query gradient missing, zero or nonfinite')
                gradient=master.grad.detach().cpu().clone() if step in (1,2,32,64,96,128) else None
                norm=torch.nn.utils.clip_grad_norm_([master],meta['clip_norm'],error_if_nonfinite=True)
                optimizer.step();require(torch.isfinite(master).all(), 'nonfinite learned queries')
                require(torch.cuda.max_memory_allocated()/1024**3<meta['maximum_allocated_vram_GiB'], 'VRAM bound exceeded')
                observed=dict(entry,id=row['id'],loss=float(loss.detach()),gradient_norm=float(norm),
                    sigma=float(sigma.item()),finite_nonzero_query_gradient=True)
                if gradient is not None:
                    path=execution/f'update_{step:03}.pt'
                    torch.save(dict(query_gradient=gradient,queries_after=master.detach().cpu(),clean=clean.cpu(),noise=noise.cpu(),
                        noisy=flow.noisy_latents.cpu(),target_velocity=flow.velocity.cpu(),predicted_velocity=prediction.detach().cpu()),path)
                    observed.update(actual_tensor_file=path.name,actual_tensor_sha256=sha(path))
                history.append(observed);save_json(execution/'updates.json',history);completed=step;status()
                if step%32==0:
                    path=execution/f'checkpoint_{step:03}.pt'
                    torch.save(dict(queries=master.detach().cpu(),initial_queries=initial,optimizer=optimizer.state_dict(),
                        completed_updates=step,preparation_sha256=sha(output/'preparation.json')),path)
                    print(f'Completed{step}/128 updates; native loss={observed["loss"]:.8f}',flush=True)
                del merged,reference,pooled,sequence,noise,flow,loss,prediction,gradient
            actual={part+'.'+name:p._version for part in ('lmm','vae','transformer','connector_module')
                for name,p in getattr(pipe,part).named_parameters()}
            require(actual==versions, 'frozen weight changed during query updates')
            require(not torch.equal(master.detach().cpu(),initial), 'no query coordinate changed')
            with torch.no_grad():pipe.connector_module.meta_queries.copy_(master.to(pipe.transformer.dtype))
            for part in ('lmm','vae','transformer','connector_module'):getattr(pipe,part).eval()
            pipe.llm.gradient_checkpointing_disable()
            # Matched full-noise endpoint fields use the original diagnostic seeds.
            phase='final_fields';status();fields=[]
            with torch.no_grad():
                for i,row in enumerate(cases):
                    budget();torch.manual_seed(row['feature_seed']);torch.cuda.manual_seed(row['feature_seed'])
                    merged,reference=input_features(pipe,Path(row['own']),master)
                    pooled,sequence=pipe.connector_module.llm2dit(merged);clean=clean_cache[i]
                    noise=torch.randn(clean.shape,device=device,dtype=clean.dtype,
                        generator=torch.Generator(device=device).manual_seed(row['noise_seed']))
                    sigma=scheduler.sigmas[:1].to(device=device,dtype=clean.dtype).reshape(1,1,1,1)
                    flow=interpolate(clean,noise,sigma,scheduler.timesteps[:1].to(device),torch.tensor([0]))
                    loss,prediction=denoising_loss(pipe.transformer,flow,sequence_condition=sequence[:1],
                        pooled_condition=pooled[:1],reference_latents=[[reference]])
                    path=execution/f'final_field_{i:03}.pt'
                    torch.save(dict(predicted_velocity=prediction.cpu(),target_velocity=flow.velocity.cpu(),noise=noise.cpu()),path)
                    fields.append(dict(id=row['id'],loss=float(loss),actual_tensor_file=path.name,sha256=sha(path)))
            phase='final_sampling';status()
            for name in plans:
                if name!='initial_copy':generate(name)
            require(sum(len(load(execution/name/'outputs.json')) for name in plans)==18, 'complete18 new outputs required')
            save_json(execution/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                actual_optimizer_updates=completed,query_parameters=master.numel(),
                native_helper_features_exact=True,checkpointed_features_exact=True,initial_native_field_exact=True,
                all_frozen_weight_versions_unchanged_during_training=True,fields=fields,
                checkpoint_sha256=sha(execution/'checkpoint_128.pt'),actual_new_generations=18,
                updates_sha256=sha(execution/'updates.json'),
                elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
                retained_runtime_parameters=vs.parameters(pipe),loader=loader,removal=removal,
                semantic_learning_validated=False,general_ilm_goal_complete=False))
        phase='completed';status()
    except BaseException as error:
        phase='failed';status(reason=str(error));raise
    finally:
        signal.alarm(0)


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'hide CUDA for offline evaluation')
    meta=unchanged(output);execution=output/'execution'
    report=load(execution/'report.json')
    require(load(execution/'runtime.json')['phase']=='completed' and report['actual_optimizer_updates']==128
        and report['actual_new_generations']==18, 'complete registered model execution required')
    require(sha(execution/'checkpoint_128.pt')==report['checkpoint_sha256']
        and sha(execution/'updates.json')==report['updates_sha256'], 'saved execution changed')
    from ilm.image_edit.native_flow_assessment import authenticate_native_png
    destination=output/'assessment';destination.mkdir(exist_ok=False)
    plans=load(output/'inference.json');cohorts={};authenticated=[]
    for name in ('initial_normal',*plans):
        manifest=output/(('final_normal' if name=='initial_normal' else name)+'_references.jsonl')
        generations=output/'initial_normal_outputs.json' if name=='initial_normal' else execution/name/'outputs.json'
        results=load(generations)
        expected=load(SAMPLING/'inputs.json') if name=='initial_normal' else plans[name]
        require(len(results)==len(expected), 'complete fixed evaluation cohort required')
        for row,case in zip(results,expected):
            require(row['id']==case['id'] and row['input_sha256']==case['input_sha256'] and row['seed']==case['seed'],
                'original generation identity differs')
            path=(generations.parent/row['path']).resolve()
            signature=authenticate_native_png(path,row['sha256'],row['result']['tensor_sha256'])
            authenticated.append(dict(cohort=name,id=row['id'],path=str(path.relative_to(ROOT)),
                PNG_sha256=row['sha256'],actual_tensor_file_sha256=row['result']['tensor_sha256'],pixels=signature))
        for stage in ('calibrate','score'):
            command=[str(ROOT.parent/'LocalLLM/.local/image-generation/venv/bin/python'),
                'scripts/score_paired_writing.py','--stage',stage,'--task','copy' if 'copy' in name else 'qa',
                '--train-limit','2','--manifest',str(manifest),'--output',str(destination/name/stage)]
            if stage=='score':command.extend(['--calibration',str(destination/name/'calibrate/report.json'),
                '--generations',str(generations)])
            subprocess.run(command,cwd=ROOT,check=True)
        scored=load(destination/name/'score/report.json')
        require(scored['complete_cohort'], 'offline score cohort incomplete')
        cohorts[name]=dict(summary=scored['summary'],report_sha256=sha(destination/name/'score/report.json'),
            calibration_sha256=sha(destination/name/'calibrate/report.json'))
    total=lambda name,split:cohorts[name]['summary'][split]['ocr_exact']
    copies=lambda name:sum(cohorts[name]['summary'][s]['ocr_exact'] for s in ('train','validation'))
    import torch
    a,b=[torch.load(execution/'final_question_erased'/str(i).zfill(3)/'actual_tensors.pt',map_location='cpu',weights_only=True)['pixels'] for i in (0,1)]
    erased_exact=torch.equal(a,b)
    require(erased_exact, 'identical question-erased input/seed decoded differently')
    save_json(destination/'result.json',dict(preparation_sha256=sha(output/'preparation.json'),
        execution_report_sha256=sha(execution/'report.json'),cohorts=cohorts,authenticated_original_pages=authenticated,
        total_original_pages=22,newly_generated_pages=18,reused_original_baseline_pages=4,
        question_erased_decoded_pixels_bit_exact=erased_exact,
        developmental_gates=dict(seen_fit=total('final_normal','train')==2,
            second_noise_seen_fit=total('final_new_noise','train')==2,
            held_pair=total('final_normal','validation')==2,
            question_dependence=total('final_normal','train')>total('final_question_erased','train'),
            passage_dependence=total('final_normal','train')>total('final_passage_erased','train'),
            copying_retention=copies('final_copy')>=copies('initial_copy')),
        fixed_canonical_OCR_only=True,raw_image_review_completed=False,semantic_learning_validated=False,
        general_ilm_goal_complete=False))
    print('Scored and authenticated all22 original pages; actual raw-image review remains required.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','wait','run','score'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'wait':wait,'run':run,'score':score}[args.stage](args.output.resolve())
