#!/usr/bin/env python3
"""Bounded native supplied-content writing diagnostic, never an ILM deployment."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT, TASK, gpu_zero, save_json, sha
from scripts.train_native_query_fit import resources

FILES = ('scripts/run_native_writer_content_control.py',
         'references/native_writer_content_control_protocol.md')
FIT = ROOT/'artifacts/native_query_fit_v1'
TRAIN = ROOT/'artifacts/native_visual_flow_learning_v1'
SAMPLING = ROOT/'artifacts/native_flow_positive_sampling_probe_v1'
TEMPLATE = ('Replace the reference image with a clean white page. Write exactly the following text in clear black print, '
    'preserving every character, word and its order:\n{reference_text}\nOnly this text should appear. '
    'Do not add labels, headings, quotation marks, explanations or decoration.')


def load(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def identity():
    for name in FILES:
        subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    require(not subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT),
            'commit the complete control implementation and protocol first')
    return {name:sha(ROOT/name) for name in FILES}


def rows(name):
    return [json.loads(line) for line in (FIT/name).read_text().splitlines() if line.strip()]


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    code=identity();old=load(FIT/'preparation.json')
    require(load(FIT/'wait.json')['phase']=='completed'
        and load(FIT/'execution/report.json')['actual_optimizer_updates']==128,'complete predecessor required')
    bound={**old['bound_sources_sha256'],**old['code_sha256']}
    for name in ('preparation.json','execution/report.json','final_normal_references.jsonl',
                 'final_copy_references.jsonl'):
        path=FIT/name;bound[str(path.relative_to(ROOT))]=sha(path)
    selected=rows('final_normal_references.jsonl')[:2]
    copies=rows('final_copy_references.jsonl')
    selected += [next(r for r in copies if r['language']=='en' and r['split']==split)
                 for split in ('train','validation')]
    require(len({r['id'] for r in selected})==4 and [r['language'] for r in selected]==['zh','zh','en','en'],
            'fixed two-Chinese/two-English cases required')
    cases=[]
    from PIL import Image
    from scripts.evaluate_image_edit_baseline import engine,normalize,recognize
    binary,engine_info=engine();calibration=[]
    for i,row in enumerate(selected):
        for key in ('input','target'):
            require(Path(row[key]).is_absolute() and sha(row[key])==row[key+'_sha256'],'original page differs')
            with Image.open(row[key]) as image:
                require(image.size==(1024,1024),'complete native geometry required')
        text=row['reference_text'];prompt=TEMPLATE.format(reference_text=text)
        require(text and prompt.count(text)==1,'supply the exact reference once')
        cases.append(dict(row,content_prompt=prompt,content_prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
                          content_seed=20270501+i))
        raw,stderr=recognize(binary,row['target'])
        calibration.append(dict(id=row['id'],language=row['language'],expected=text,raw_transcript=raw,
            ocr_stderr=stderr,normalized=normalize(raw,row['language']),png_sha256=row['target_sha256'],
            ocr_exact=normalize(raw,row['language'])==normalize(text,row['language'])))
    replays={}
    for arm,name in (('source','initial'),('adapted','final')):
        path=SAMPLING/name/'outputs.json';bound[str(path.relative_to(ROOT))]=sha(path)
        original=load(path)[0]
        require(original['id']==cases[0]['id'] and original['input_sha256']==cases[0]['input_sha256'],
                'same first native execution counterpart required')
        png=path.parent/original['path'];tensor=png.with_name('actual_tensors.pt')
        require(sha(png)==original['sha256'] and sha(tensor)==original['result']['tensor_sha256'],
                'original replay files changed')
        replays[arm]=dict(id=original['id'],seed=original['seed'],png=str(png),png_sha256=sha(png),
            tensors=str(tensor),tensors_sha256=sha(tensor))
    deployment=TRAIN/'deployment/deployment.json';model=load(deployment)
    require(model['completed_updates']==1536,'exact completed adapter export required')
    for path in (deployment,deployment.parent/'adapters.safetensors'):
        bound[str(path.relative_to(ROOT))]=sha(path)
    for name,digest in bound.items():require(sha(ROOT/name)==digest,'predecessor binding changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'cases.json',cases);save_json(output/'replays.json',replays)
    save_json(output/'calibration.json',dict(engine=engine_info,rows=calibration,
        cases=4,ocr_exact=sum(r['ocr_exact'] for r in calibration),generator_loaded=False))
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sources_sha256=bound,
        prepared_files_sha256={name:sha(output/name) for name in ('cases.json','replays.json','calibration.json')},
        adapter_deployment=str(deployment),adapter_weights_sha256=model['weights_sha256'],
        original_source_revision=model['source_revision'],arm_order=['source','adapted'],guidance_order=[1.,4.],
        supplied_content_draws=16,replay_draws=2,denoiser_calls=900,denoiser_batch_elements=1300,
        steps=50,canvas=[1024,1024],maximum_wait_seconds=3600,poll_seconds=30,
        maximum_execution_seconds=1800,maximum_allocated_vram_GiB=23.,
        supplied_answer_text=True,tokenizer_and_word_table_retained=True,original_continuous_queries=True,
        reference_answer_images_used_by_generator=False,optimizer_updates=0,
        all_predecessor_bindings_verified=True,semantic_learning_validated=False,general_ilm_goal_complete=False))
    print('Prepared4 exact supplied strings and16 native writing draws plus2 execution replays; no model loaded.',flush=True)


def unchanged(output):
    meta=load(output/'preparation.json')
    require(identity()==meta['code_sha256'],'registered control code changed')
    for name,digest in meta['bound_sources_sha256'].items():
        require(sha(ROOT/name)==digest,'predecessor binding changed: '+name)
    for name,digest in meta['prepared_files_sha256'].items():
        require(sha(output/name)==digest,'prepared control file changed: '+name)
    return meta


def execute(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','use physical GPU0 only')
    meta=unchanged(output);folder=output/'execution';folder.mkdir(exist_ok=False)
    started=time.perf_counter();phase='loading';completed=0
    def status(**extra):
        save_json(folder/'runtime.json',dict(phase=phase,pid=None if phase in ('completed','failed') else os.getpid(),
            completed_draws=completed,supplied_answer_text=True,semantic_learning_validated=False,
            general_ilm_goal_complete=False,**extra))
    def interrupted(signum,frame):
        if signum==signal.SIGALRM:raise TimeoutError('bounded writer control exceeded30minutes')
        raise InterruptedError('writer control signal '+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP,signal.SIGALRM):signal.signal(sig,interrupted)
    signal.alarm(meta['maximum_execution_seconds'])
    try:
        status()
        with gpu_zero() as device:
            import torch
            import numpy as np
            from PIL import Image
            from safetensors.torch import load_file
            from ilm.image_edit import visual_semantic as vs
            from ilm.image_edit.native_flow_step import add_adapters
            from ilm.image_edit.native_flow_run import restore_adapters
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            source=vs.verify_release();pipe,loader=vs.load_pipeline(device)
            deployment=load(meta['adapter_deployment'])
            require(source==deployment['source_sha256'],'same complete pretrained source required')
            require(add_adapters(pipe,deployment['rank'])==deployment['adapter_layer_names'],'exact adapter graph required')
            for name in ('lmm','transformer','connector_module','vae'):
                getattr(pipe,name).eval()
            calls=[];head_calls=[]
            hook=pipe.transformer.register_forward_pre_hook(
                lambda module,args,kwargs:calls.append(int(kwargs['hidden_states'].shape[0])),with_kwargs=True)
            head_hook=pipe.lmm.lm_head.register_forward_hook(lambda *args:head_calls.append(True))
            initial_queries=pipe.connector_module.meta_queries.detach().cpu().clone()
            cases,replays=load(output/'cases.json'),load(output/'replays.json');records=[]
            def generate(row,prompt,seed,guidance,destination):
                require(sha(row['input'])==row['input_sha256'],'complete original input changed')
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                calls.clear();pipe._ilm_capture.clear()
                with Image.open(row['input']) as image:
                    result=pipe(prompt,image=image.convert('RGB'),negative_prompt='',height=1024,width=1024,
                        num_inference_steps=50,guidance_scale=guidance,seed=seed,num_images_per_prompt=1)
                pixels=pipe._ilm_capture['pixels'];fields=pipe._ilm_capture['forward_inputs']
                require(pixels.shape==(1,3,1024,1024) and torch.isfinite(pixels).all() and len(result.images)==1,
                        'finite complete actual decoded page required')
                expected=torch.clamp(127.5*pixels[0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
                require(np.array_equal(expected,np.array(result.images[0])),'native PNG/decoder pixels differ')
                require(calls==[1 if guidance==1. else 2]*50,'native sampler call/branch count differs')
                require(torch.cuda.max_memory_allocated()/1024**3<=meta['maximum_allocated_vram_GiB'],
                        'registered peak allocation bound exceeded')
                value=dict(image=result.images[0],pixels=pixels,forward_inputs=fields)
                saved=vs.save_generation(value,destination)
                return value,saved,list(calls)
            phase='generating';status()
            for arm in meta['arm_order']:
                if arm=='adapted':restore_adapters(pipe,load_file(str(Path(meta['adapter_deployment']).parent/'adapters.safetensors'),device='cpu'))
                require(torch.equal(initial_queries,pipe.connector_module.meta_queries.detach().cpu()),'original reader queries changed')
                old=replays[arm]
                value,saved,batches=generate(cases[0],TASK,old['seed'],1.,folder/(arm+'_replay'))
                baseline=torch.load(old['tensors'],map_location='cpu',weights_only=True)
                with Image.open(old['png']) as image:baseline['image']=image.copy()
                equality=vs.exact_replay(baseline,value)
                records.append(dict(arm=arm,kind='replay',id=cases[0]['id'],guidance=1.,seed=old['seed'],
                    path=arm+'_replay/answer.png',sha256=saved['sha256'],result=saved,
                    original=old,equality=equality,denoiser_batch_sizes=batches))
                completed+=1;save_json(folder/'outputs.json',records);status(arm=arm)
                require(equality['decoded_tensor_exact'] and equality['png_pixels_exact']
                    and all(equality['forward_fields_exact'].values()),'fixed-task native replay differs; preserve and stop')
                del baseline,value
                for guidance in meta['guidance_order']:
                    for i,row in enumerate(cases):
                        name=f'{arm}_g{int(guidance)}_{i:03}'
                        value,saved,batches=generate(row,row['content_prompt'],row['content_seed'],guidance,folder/name)
                        records.append(dict(arm=arm,kind='supplied_content',id=row['id'],guidance=guidance,
                            seed=row['content_seed'],path=name+'/answer.png',sha256=saved['sha256'],result=saved,
                            content_prompt_sha256=row['content_prompt_sha256'],input_sha256=row['input_sha256'],
                            denoiser_batch_sizes=batches,supplied_answer_text=True))
                        completed+=1;save_json(folder/'outputs.json',records);status(arm=arm,guidance=guidance)
                        print(f'Completed{completed}/18: {name}',flush=True);del value
            hook.remove();head_hook.remove()
            require(not head_calls,'unused vocabulary output head was called')
            require(all(p.grad is None for name in ('lmm','transformer','connector_module','vae')
                for p in getattr(pipe,name).parameters()),'diagnostic unexpectedly accumulated gradients')
            require(len(records)==18 and sum(r['kind']=='supplied_content' for r in records)==16,'complete draws required')
            require(sum(len(r['denoiser_batch_sizes']) for r in records)==900
                and sum(sum(r['denoiser_batch_sizes']) for r in records)==1300,'complete sampler accounting differs')
            unchanged(output)
            save_json(folder/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),
                outputs_sha256=sha(folder/'outputs.json'),complete_generations=18,supplied_content_generations=16,
                execution_replays=2,all_replays_bit_exact=True,denoiser_calls=900,denoiser_batch_elements=1300,
                loader=loader,runtime_parameters=vs.parameters(pipe),original_queries_unchanged=True,
                unused_vocabulary_output_head_called=False,tokenizer_and_input_word_table_retained=True,
                reference_answer_images_used_by_generator=False,supplied_answer_text=True,optimizer_updates=0,
                elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
                semantic_learning_validated=False,general_ilm_goal_complete=False))
            phase='completed';status()
    except BaseException as error:
        phase='failed';status(reason=str(error));raise
    finally:signal.alarm(0)


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for separate CPU scoring')
    unchanged(output)
    folder=output/'execution';report=load(folder/'report.json')
    require(load(folder/'runtime.json')['phase']=='completed' and report['complete_generations']==18,
            'complete model child must exit before scoring')
    require(report['outputs_sha256']==sha(folder/'outputs.json'),'model output index changed')
    import torch
    import numpy as np
    from PIL import Image
    from scripts.evaluate_image_edit_baseline import engine,normalize,recognize
    binary,current=engine();calibration=load(output/'calibration.json')
    require(current==calibration['engine'],'reference OCR engine changed')
    cases={r['id']:r for r in load(output/'cases.json')};observed=[]
    for row in load(folder/'outputs.json'):
        path=folder/row['path'];tensor=path.with_name('actual_tensors.pt')
        require(sha(path)==row['sha256'] and sha(tensor)==row['result']['tensor_sha256'],'original generation changed')
        actual=torch.load(tensor,map_location='cpu',weights_only=True)
        pixels=torch.clamp(127.5*actual['pixels'][0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
        with Image.open(path) as image:require(np.array_equal(np.array(image),pixels),'actual decoded pixels differ')
        if row['kind']=='replay':
            old=row['original'];require(sha(old['tensors'])==old['tensors_sha256'] and sha(old['png'])==old['png_sha256'],'original replay changed')
            original=torch.load(old['tensors'],map_location='cpu',weights_only=True)
            require(torch.equal(actual['pixels'],original['pixels']),'independent decoded replay differs')
            require(set(actual['forward_inputs'])==set(original['forward_inputs']),'replay input keys differ')
            for key,a in actual['forward_inputs'].items():
                b=original['forward_inputs'][key]
                require(a is None and b is None if a is None else torch.equal(a,b),'independent forward replay differs')
        case=cases[row['id']];raw,stderr=recognize(binary,path)
        observed.append(dict(arm=row['arm'],kind=row['kind'],id=row['id'],guidance=row['guidance'],
            language=case['language'],expected=case['reference_text'],raw_transcript=raw,ocr_stderr=stderr,
            normalized=normalize(raw,case['language']),ocr_exact=normalize(raw,case['language'])==normalize(case['reference_text'],case['language']),
            path=row['path'],png_sha256=row['sha256'],actual_decoded_pixels_authenticated=True))
    summary={}
    for arm in ('source','adapted'):
        for guidance in (1.,4.):
            selected=[r for r in observed if r['kind']=='supplied_content' and r['arm']==arm and r['guidance']==guidance]
            require(len(selected)==4 and len({r['id'] for r in selected})==4,'complete fixed cell required')
            summary[f'{arm}_g{int(guidance)}']=dict(cases=4,ocr_exact=sum(r['ocr_exact'] for r in selected),
                per_language={lang:dict(cases=2,ocr_exact=sum(r['ocr_exact'] for r in selected if r['language']==lang)) for lang in ('zh','en')})
    save_json(output/'assessment.json',dict(model_report_sha256=sha(folder/'report.json'),rows=observed,summary=summary,
        calibration_sha256=sha(output/'calibration.json'),calibration_exact=calibration['ocr_exact'],calibration_cases=4,
        all18_original_PNGs_tensor_authenticated=True,all2_replays_independently_verified=True,
        model_loaded=False,raw_image_review_completed=False,supplied_answer_text=True,
        autonomous_QA_validated=False,semantic_learning_validated=False,general_ilm_goal_complete=False))
    print(json.dumps(summary),flush=True)


def coordinate(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','physical GPU0 only')
    meta=unchanged(output);require(not (output/'execution').exists(),'preserve the existing execution attempt')
    import fcntl
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic();child=None
        def interrupted(signum,frame):raise InterruptedError('coordinator signal '+str(signum))
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
        try:
            while True:
                observed=resources();elapsed=time.monotonic()-started
                if observed['ready']:break
                timeout=elapsed>=meta['maximum_wait_seconds']
                save_json(output/'coordinator.json',dict(phase='wait_timeout' if timeout else 'waiting_resources',
                    pid=None if timeout else os.getpid(),child_pid=None,elapsed_seconds=elapsed,resources=observed))
                if timeout:return
                print('Waiting for existing thresholds: '+json.dumps(observed),flush=True)
                time.sleep(meta['poll_seconds'])
            child=subprocess.Popen([sys.executable,__file__,'--stage','execute','--output',str(output)],cwd=ROOT,start_new_session=True)
            save_json(output/'coordinator.json',dict(phase='model_running',pid=os.getpid(),child_pid=child.pid))
            child.wait(timeout=meta['maximum_execution_seconds']+120)
            require(child.returncode==0,'writer control failed; retain attempt, no automatic retry')
            save_json(output/'coordinator.json',dict(phase='CPU_scoring',pid=os.getpid(),child_pid=None))
            child=subprocess.Popen([sys.executable,__file__,'--stage','score','--output',str(output)],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
            save_json(output/'coordinator.json',dict(phase='CPU_scoring',pid=os.getpid(),child_pid=child.pid))
            child.wait(timeout=600);require(child.returncode==0,'CPU score failed; preserve all original model draws')
            save_json(output/'coordinator.json',dict(phase='completed',pid=None,child_pid=None,
                raw_image_review_required=True,general_ilm_goal_complete=False))
        except BaseException as error:
            save_json(output/'coordinator.json',dict(phase='failed',pid=None,child_pid=None,reason=str(error)));raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','coordinate','execute','score'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'coordinate':coordinate,'execute':execute,'score':score}[args.stage](args.output.resolve())
