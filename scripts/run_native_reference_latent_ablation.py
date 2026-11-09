#!/usr/bin/env python3
"""Preserve visual reading while removing only the denoiser's VAE reference."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT,gpu_zero,save_json,sha
from scripts.train_native_query_fit import resources
from scripts.run_native_writer_content_control import unchanged as old_unchanged

FILES=('scripts/run_native_reference_latent_ablation.py',
       'references/native_reference_latent_ablation_protocol.md')
PRIOR=ROOT/'artifacts/native_writer_content_control_v1'


def load(path):return json.loads(Path(path).read_text())


def require(value,message):
    if not value:raise ValueError(message)


def identity():
    for name in FILES:
        subprocess.run(['git','ls-files','--error-unmatch',name],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    require(not subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT),
            'commit complete ablation runner and protocol first')
    return {name:sha(ROOT/name) for name in FILES}


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for preparation')
    code=identity();previous=old_unchanged(PRIOR)
    require(load(PRIOR/'coordinator.json')['phase']=='completed'
        and load(PRIOR/'execution/report.json')['complete_generations']==18,'complete previous writer control required')
    archive=ROOT/'references/evidence/native_writer_content_control_result'
    reviewed=load(archive/'raw_review.json')
    require(reviewed['all18_original_generated_pages_actually_inspected']
        and reviewed['all4_original_references_actually_inspected'],'complete original review required')
    bound={**previous['bound_sources_sha256'],**previous['code_sha256']}
    for path in [PRIOR/name for name in ('preparation.json','cases.json','calibration.json',
        'execution/report.json','execution/outputs.json','assessment.json')]+[
        ROOT/'scripts/audit_native_writer_content_control.py',
        ROOT/'artifacts/native_writer_content_control_audit_v1/report.json',archive/'raw_review.json',archive/'provenance.json']:
        bound[str(path.relative_to(ROOT))]=sha(path)
    cases=load(PRIOR/'cases.json');all_outputs=load(PRIOR/'execution/outputs.json')
    require(len(cases)==4,'exact four fixed cases required');counterparts={}
    for arm in ('source','adapted'):
        selected=[r for r in all_outputs if r['arm']==arm and r['kind']=='supplied_content' and r['guidance']==4.]
        require([r['id'] for r in selected]==[r['id'] for r in cases],'complete original guidance-four cell required')
        counterparts[arm]=[]
        for row in selected:
            png=PRIOR/'execution'/row['path'];tensor=png.with_name('actual_tensors.pt')
            require(sha(png)==row['sha256'] and sha(tensor)==row['result']['tensor_sha256'],'original counterpart differs')
            counterparts[arm].append(dict(row,png=str(png),tensors=str(tensor)))
    calibration=load(PRIOR/'calibration.json')
    from scripts.evaluate_image_edit_baseline import engine
    _,current=engine();require(current==calibration['engine'] and calibration['ocr_exact']==4,'unchanged four-target calibration required')
    for row in cases:
        for key in ('input','target'):require(sha(row[key])==row[key+'_sha256'],'original case page changed')
    for name,digest in bound.items():require(sha(ROOT/name)==digest,'bound predecessor changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'cases.json',cases);save_json(output/'counterparts.json',counterparts)
    save_json(output/'calibration.json',calibration)
    save_json(output/'preparation.json',dict(code_sha256=code,bound_sources_sha256=bound,
        prepared_files_sha256={name:sha(output/name) for name in ('cases.json','counterparts.json','calibration.json')},
        adapter_deployment=previous['adapter_deployment'],adapter_weights_sha256=previous['adapter_weights_sha256'],
        source_revision=previous['original_source_revision'],arm_order=['source','adapted'],
        supplied_content_removed_reference_draws=8,replay_draws=2,steps=50,guidance=4.,canvas=[1024,1024],
        actual_expected_denoiser_calls=500,actual_expected_batch_elements=1000,
        removed_reference_calls=400,preserved_reference_calls=100,maximum_wait_seconds=3600,poll_seconds=30,
        maximum_execution_seconds=1800,maximum_allocated_vram_GiB=23.,
        supplied_answer_text=True,visual_reader_preserved=True,tokenizer_and_word_table_retained=True,
        target_PNG_used_by_generator=False,optimizer_updates=0,semantic_learning_validated=False,general_ilm_goal_complete=False))
    print('Prepared8 reference removals and2 exact execution replays; same four content prompts, inputs and seeds.',flush=True)


def unchanged(output):
    meta=load(output/'preparation.json');require(identity()==meta['code_sha256'],'ablation code changed')
    for name,digest in meta['bound_sources_sha256'].items():require(sha(ROOT/name)==digest,'predecessor changed: '+name)
    for name,digest in meta['prepared_files_sha256'].items():require(sha(output/name)==digest,'prepared file changed: '+name)
    return meta


def execute(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','physical GPU0 only')
    meta=unchanged(output);folder=output/'execution';folder.mkdir(exist_ok=False)
    started=time.perf_counter();phase='loading';completed=0
    def status(**extra):save_json(folder/'runtime.json',dict(phase=phase,pid=None if phase in ('failed','completed') else os.getpid(),
        completed_draws=completed,semantic_learning_validated=False,general_ilm_goal_complete=False,**extra))
    def interrupted(signum,frame):
        if signum==signal.SIGALRM:raise TimeoutError('ablation exceeded30minute bound')
        raise InterruptedError('ablation signal '+str(signum))
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,interrupted)
    signal.alarm(meta['maximum_execution_seconds'])
    try:
        status()
        with gpu_zero() as device:
            import torch
            import numpy as np
            from PIL import Image
            from safetensors.torch import load_file
            from ilm.image_edit import visual_semantic as vs
            from ilm.image_edit.native_flow_run import restore_adapters
            from ilm.image_edit.native_flow_step import add_adapters
            from scripts.audit_native_writer_content_control import equal_fields
            torch.use_deterministic_algorithms(True)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            source=vs.verify_release();pipe,loader=vs.load_pipeline(device);deployment=load(meta['adapter_deployment'])
            require(source==deployment['source_sha256'],'same complete source required')
            require(add_adapters(pipe,deployment['rank'])==deployment['adapter_layer_names'],'same adapter graph required')
            for name in ('lmm','connector_module','transformer','vae'):getattr(pipe,name).eval()
            queries=pipe.connector_module.meta_queries.detach().cpu().clone()
            state={};head_calls=[]
            def intervention(module,args,kwargs):
                reference=kwargs.get('cond_hidden_states')
                require(isinstance(reference,list) and len(reference)==2 and all(len(r)==1 for r in reference),
                        'actual two-branch native reference input required before intervention')
                if not state['steps']:
                    state['reference_object']=reference
                    state['first']=dict(before_reference=[[t.detach().cpu().clone() for t in r] for r in reference],
                        **{key:kwargs[key].detach().cpu().clone() for key in
                           ('hidden_states','timestep','encoder_hidden_states','pooled_projections')})
                require(reference is state['reference_object'],'source reference object changed during native sampling')
                state['other_args']={k:v for k,v in kwargs.items() if k!='cond_hidden_states'}
                return args,dict(kwargs,cond_hidden_states=None) if state['remove'] else kwargs
            def observe(module,args,kwargs):
                require(all(kwargs[k] is v for k,v in state['other_args'].items()),'intervention changed a non-reference argument')
                is_none=kwargs['cond_hidden_states'] is None
                require(is_none==state['remove'],'actual denoiser reference did not match intervention')
                state['steps'].append(dict(batch=int(kwargs['hidden_states'].shape[0]),before_reference_present=True,
                    after_reference_none=is_none,other_argument_objects_unchanged=True))
            change_hook=pipe.transformer.register_forward_pre_hook(intervention,with_kwargs=True)
            observe_hook=pipe.transformer.register_forward_pre_hook(observe,with_kwargs=True)
            head_hook=pipe.lmm.lm_head.register_forward_hook(lambda *args:head_calls.append(True))
            records=[];cases=load(output/'cases.json');counterparts=load(output/'counterparts.json')
            def draw(arm,i,remove,kind):
                nonlocal completed
                case=cases[i];old=counterparts[arm][i];name=arm+('_replay' if kind=='replay' else f'_removed_{i:03}')
                require(sha(case['input'])==case['input_sha256'],'complete source input changed')
                seed=case['content_seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed(seed)
                pipe._ilm_capture.clear();state.clear();state.update(remove=remove,steps=[])
                with Image.open(case['input']) as image:
                    result=pipe(case['content_prompt'],image=image.convert('RGB'),negative_prompt='',height=1024,width=1024,
                        num_inference_steps=50,guidance_scale=4.,seed=seed,num_images_per_prompt=1)
                pixels=pipe._ilm_capture['pixels'];fields=pipe._ilm_capture['forward_inputs']
                require(pixels.shape==(1,3,1024,1024) and torch.isfinite(pixels).all() and len(result.images)==1,'complete finite decoder page required')
                expected=torch.clamp(127.5*pixels[0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
                require(np.array_equal(expected,np.asarray(result.images[0])),'native PNG pixels differ from decoder')
                require(len(state['steps'])==50 and all(s['batch']==2 for s in state['steps']),'fifty real two-branch denoiser calls required')
                require(sha(old['png'])==old['sha256'] and sha(old['tensors'])==old['result']['tensor_sha256'],'original counterpart changed')
                original=torch.load(old['tensors'],map_location='cpu',weights_only=True)
                require(equal_fields(original['forward_inputs'],fields),'image-reader inputs differ from original counterpart')
                if kind=='replay':
                    require(torch.equal(original['pixels'],pixels),'full-reference decoder replay differs')
                    with Image.open(old['png']) as image:require(np.array_equal(np.asarray(image),expected),'full-reference PNG replay differs')
                elif i==0:
                    first=torch.load(folder/(arm+'_replay')/'intervention.pt',map_location='cpu',weights_only=True)['first_forward']
                    for key in ('hidden_states','timestep','encoder_hidden_states','pooled_projections'):
                        require(torch.equal(first[key],state['first'][key]),'first non-reference denoiser tensor differs: '+key)
                require(torch.cuda.max_memory_allocated()/1024**3<=23.,'registered allocation bound exceeded')
                value=dict(image=result.images[0],pixels=pixels,forward_inputs=fields)
                saved=vs.save_generation(value,folder/name)
                torch.save(dict(first_forward=state['first'],steps=state['steps'],reference_removed=remove),folder/name/'intervention.pt')
                records.append(dict(arm=arm,kind=kind,id=case['id'],case_index=i,guidance=4.,seed=seed,path=name+'/answer.png',
                    sha256=saved['sha256'],result=saved,intervention_sha256=sha(folder/name/'intervention.pt'),
                    original_png_sha256=old['sha256'],original_tensors_sha256=old['result']['tensor_sha256'],
                    reader_inputs_bit_exact=True,reference_removed=remove,actual_denoiser_calls=50,
                    actual_denoiser_batch_elements=100,supplied_answer_text=True))
                completed+=1;save_json(folder/'outputs.json',records);status(arm=arm)
                print(f'Completed{completed}/10: {name}',flush=True)
            phase='generating';status()
            for arm in meta['arm_order']:
                if arm=='adapted':restore_adapters(pipe,load_file(str(Path(meta['adapter_deployment']).parent/'adapters.safetensors'),device='cpu'))
                require(torch.equal(queries,pipe.connector_module.meta_queries.detach().cpu()),'native reader queries changed')
                draw(arm,0,False,'replay')
                for i in range(4):draw(arm,i,True,'reference_removed')
            change_hook.remove();observe_hook.remove();head_hook.remove()
            require(not head_calls,'unused word head was called')
            require(all(p.grad is None for name in ('lmm','connector_module','transformer','vae')
                for p in getattr(pipe,name).parameters()),'unexpected diagnostic gradient')
            require(len(records)==10 and sum(r['reference_removed'] for r in records)==8,'complete ablation coverage required')
            unchanged(output)
            save_json(folder/'report.json',dict(preparation_sha256=sha(output/'preparation.json'),outputs_sha256=sha(folder/'outputs.json'),
                complete_generations=10,reference_removed_draws=8,execution_replays=2,denoiser_calls=500,denoiser_batch_elements=1000,
                removed_reference_calls=400,preserved_reference_calls=100,all_reader_inputs_bit_exact_to_counterparts=True,
                first_case_first_step_nonreference_tensors_bit_exact=True,loader=loader,runtime_parameters=vs.parameters(pipe),
                elapsed_seconds=time.perf_counter()-started,peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
                original_queries_unchanged=True,supplied_answer_text=True,tokenizer_and_word_table_retained=True,
                target_PNG_used_by_generator=False,optimizer_updates=0,semantic_learning_validated=False,general_ilm_goal_complete=False))
            phase='completed';status()
    except BaseException as error:phase='failed';status(reason=str(error));raise
    finally:signal.alarm(0)


def score(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for separate CPU verification/scoring')
    meta=unchanged(output);folder=output/'execution';report=load(folder/'report.json')
    require(load(folder/'runtime.json')['phase']=='completed' and report['complete_generations']==10,'complete GPU child required')
    require(report['preparation_sha256']==sha(output/'preparation.json') and report['outputs_sha256']==sha(folder/'outputs.json'),
            'model report bindings differ')
    require(report['optimizer_updates']==0 and report['supplied_answer_text']
        and report['tokenizer_and_word_table_retained'] and not report['target_PNG_used_by_generator']
        and report['elapsed_seconds']<1800 and report['peak_allocated_vram_GiB']<=23.,'registered diagnostic boundary differs')
    import torch
    import numpy as np
    from PIL import Image
    from scripts.evaluate_image_edit_baseline import engine,recognize
    from scripts.audit_native_writer_content_control import equal_fields,normalize
    torch.set_num_threads(4);binary,current=engine();calibration=load(output/'calibration.json')
    require(current==calibration['engine'],'reference OCR engine changed')
    cases=load(output/'cases.json');counterparts=load(output/'counterparts.json');rows=load(folder/'outputs.json')
    for case,reference in zip(cases,calibration['rows']):
        for key in ('input','target'):require(sha(case[key])==case[key+'_sha256'],'original case image changed')
        require(reference['id']==case['id'] and reference['expected']==case['reference_text']
            and reference['png_sha256']==case['target_sha256'] and reference['ocr_exact']==(
                normalize(reference['raw_transcript'],case['language'])==normalize(case['reference_text'],case['language'])),
                'owning reference calibration differs')
    require(calibration['cases']==calibration['ocr_exact']==4,'complete four-target reference calibration required')
    expected=[(a,k,i) for a in ('source','adapted') for k,i in [('replay',0),*[('reference_removed',j) for j in range(4)]]]
    require([(r['arm'],r['kind'],r['case_index']) for r in rows]==expected,'complete fixed draw order required')
    observed=[];removed=preserved=0
    for row in rows:
        case=cases[row['case_index']];old=counterparts[row['arm']][row['case_index']]
        require(row['id']==case['id']==old['id'] and row['seed']==case['content_seed']==old['seed']
            and row['guidance']==old['guidance']==4.,'registered paired case, seed or guidance differs')
        path=folder/row['path'];tensor=path.with_name('actual_tensors.pt');intervention=path.with_name('intervention.pt')
        require(sha(path)==row['sha256'] and sha(tensor)==row['result']['tensor_sha256']
            and sha(intervention)==row['intervention_sha256'],'original PNG/tensor/intervention binding differs')
        actual=torch.load(tensor,map_location='cpu',weights_only=True);probe=torch.load(intervention,map_location='cpu',weights_only=True)
        require(actual['pixels'].shape==(1,3,1024,1024) and torch.isfinite(actual['pixels']).all(),'finite complete actual page required')
        pixels=torch.clamp(127.5*actual['pixels'][0].permute(1,2,0)+128.,0,255).to(torch.uint8).numpy()
        with Image.open(path) as image:require(np.array_equal(np.asarray(image),pixels),'unrepaired actual pixels differ')
        require(sha(old['tensors'])==old['result']['tensor_sha256'] and sha(old['png'])==old['sha256'],'old counterpart changed')
        original=torch.load(old['tensors'],map_location='cpu',weights_only=True)
        require(equal_fields(original['forward_inputs'],actual['forward_inputs']),'independent reader comparison differs')
        is_removed=row['kind']=='reference_removed';require(probe['reference_removed']==is_removed,'intervention mode differs')
        steps=probe['steps'];require(len(steps)==50 and all(s==dict(batch=2,before_reference_present=True,
            after_reference_none=is_removed,other_argument_objects_unchanged=True) for s in steps),'actual hook observations differ')
        refs=probe['first_forward']['before_reference']
        require(len(refs)==2 and all(len(r)==1 and r[0].ndim==3 and torch.isfinite(r[0]).all() for r in refs),
                'actual source reference latents must exist before removal')
        removed+=50*is_removed;preserved+=50*(not is_removed)
        if not is_removed:require(torch.equal(original['pixels'],actual['pixels']),'independent full-reference replay differs')
        elif row['case_index']==0:
            prior=torch.load(folder/(row['arm']+'_replay')/'intervention.pt',map_location='cpu',weights_only=True)['first_forward']
            for key in ('hidden_states','timestep','encoder_hidden_states','pooled_projections'):
                require(torch.equal(prior[key],probe['first_forward'][key]),'independent first-step isolation differs: '+key)
        raw,stderr=recognize(binary,path);text=normalize(raw,case['language'])
        observed.append(dict(arm=row['arm'],kind=row['kind'],id=case['id'],language=case['language'],path=row['path'],
            png_sha256=row['sha256'],actual_decoded_pixels_authenticated=True,expected=case['reference_text'],
            raw_transcript=raw,ocr_stderr=stderr,normalized=text,ocr_exact=text==normalize(case['reference_text'],case['language'])))
    require(removed==report['removed_reference_calls']==400 and preserved==report['preserved_reference_calls']==100,
            'complete actual intervention counts differ')
    require(report['denoiser_calls']==removed+preserved==500
        and report['denoiser_batch_elements']==2*(removed+preserved)==1000,'sampler totals differ')
    old_assessment=load(PRIOR/'assessment.json');summary={}
    for arm in ('source','adapted'):
        old_rows=[r for r in old_assessment['rows'] if r['arm']==arm and r['kind']=='supplied_content' and r['guidance']==4.]
        new_rows=[r for r in observed if r['arm']==arm and r['kind']=='reference_removed']
        require([r['id'] for r in old_rows]==[r['id'] for r in new_rows]==[r['id'] for r in cases],'paired complete case sets differ')
        for r in old_rows:require(r['ocr_exact']==(normalize(r['raw_transcript'],r['language'])==normalize(r['expected'],r['language'])),
                                 'original exact-score recomputation differs')
        summary[arm]=dict(cases=4,full_reference_exact=sum(r['ocr_exact'] for r in old_rows),
            removed_reference_exact=sum(r['ocr_exact'] for r in new_rows),per_language={lang:dict(cases=2,
                full_reference_exact=sum(r['ocr_exact'] for r in old_rows if r['language']==lang),
                removed_reference_exact=sum(r['ocr_exact'] for r in new_rows if r['language']==lang)) for lang in ('zh','en')})
    save_json(output/'assessment.json',dict(model_report_sha256=sha(folder/'report.json'),rows=observed,summary=summary,
        calibration_sha256=sha(output/'calibration.json'),reference_OCR_exact=calibration['ocr_exact'],reference_cases=4,
        all10_actual_PNGs_tensor_authenticated=True,all2_full_reference_replays_bit_exact=True,
        all8_reader_input_comparisons_bit_exact=True,both_first_step_isolations_bit_exact=True,
        actual_reference_removed_calls=removed,actual_reference_preserved_calls=preserved,
        actual_denoiser_calls=removed+preserved,actual_denoiser_batch_elements=2*(removed+preserved),
        model_loaded=False,raw_image_review_completed=False,supplied_answer_text=True,
        semantic_learning_validated=False,general_ilm_goal_complete=False))
    print(json.dumps(summary),flush=True)


def coordinate(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','physical GPU0 only')
    meta=unchanged(output);require(not (output/'execution').exists(),'preserve existing execution attempt')
    import fcntl
    with (output/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);started=time.monotonic();child=None
        def stop(signum,frame):raise InterruptedError('ablation coordinator signal '+str(signum))
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
        try:
            while True:
                observed=resources();elapsed=time.monotonic()-started
                if observed['ready']:break
                timeout=elapsed>=meta['maximum_wait_seconds']
                save_json(output/'coordinator.json',dict(phase='wait_timeout' if timeout else 'waiting_resources',
                    pid=None if timeout else os.getpid(),child_pid=None,resources=observed,elapsed_seconds=elapsed))
                if timeout:return
                print('Waiting for existing resource thresholds: '+json.dumps(observed),flush=True);time.sleep(meta['poll_seconds'])
            for stage,timeout in [('execute',1920),('score',600)]:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if stage=='execute' else '')
                child=subprocess.Popen([sys.executable,__file__,'--stage',stage,'--output',str(output)],cwd=ROOT,env=env,start_new_session=True)
                save_json(output/'coordinator.json',dict(phase='model_running' if stage=='execute' else 'CPU_scoring',pid=os.getpid(),child_pid=child.pid))
                child.wait(timeout=timeout);require(child.returncode==0,stage+' failed; no automatic retry')
            save_json(output/'coordinator.json',dict(phase='completed',pid=None,child_pid=None,raw_image_review_required=True,general_ilm_goal_complete=False))
        except BaseException as error:save_json(output/'coordinator.json',dict(phase='failed',pid=None,child_pid=None,reason=str(error)));raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','coordinate','execute','score'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'prepare':prepare,'coordinate':coordinate,'execute':execute,'score':score}[args.stage](args.output.resolve())
