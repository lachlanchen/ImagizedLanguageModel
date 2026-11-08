#!/usr/bin/env python3
"""Probe the trained positive field after the complete autonomous assessment.

This is a paired, partly target-corrupted field diagnostic, not autonomous
answer generation. Full-noise fields receive independent noise only.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.image_edit.native_flow_run import CONDITION, identity, restore_adapters
from ilm.image_edit import visual_semantic as vs

FILES = ('scripts/run_native_flow_condition_probe.py', 'references/native_flow_condition_probe_protocol.md')
INDICES = (0, 499)


def code_identity():
    if subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT,text=True):
        raise ValueError('commit the probe implementation/protocol before preparation')
    return {p:sha(ROOT/p) for p in FILES}


def metric(u0, u1, p0, p1):
    """A=1 perfect,0 identical,-1 swapped. Float64 offline coordinate means."""
    import torch
    a,b,c,d = (x.detach().cpu().double() for x in (u0,u1,p0,p1))
    if any(x.shape != a.shape or not torch.isfinite(x).all() for x in (a,b,c,d)):
        raise ValueError('finite matching field tensors required')
    mse = lambda x,y: float((x-y).square().mean())
    distance = mse(a,b)
    if distance <= 0: raise ValueError('distinct native target fields required')
    advantages = [mse(d,a)-mse(c,a), mse(c,b)-mse(d,b)]
    value = sum(advantages)/(2*distance)
    dot = float(((b-a)*(d-c)).mean())/distance
    if abs(value-dot) > 1e-10*max(1,abs(value),abs(dot)):
        raise ValueError('paired field algebra differs')
    return dict(normalized_alignment=value,loss_advantages=advantages,target_field_MSE_distance=distance,
        dot_product_alignment=dot)


def math_check(output):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('hide CUDA for arithmetic checks')
    source = ROOT/'artifacts/visual_flow_step_preflight_v2/execution/actual_training_tensors.pt'
    saved = torch.load(source,map_location='cpu',weights_only=True)
    a = saved['target_velocity']; b = a.roll(1,-1); predicted = saved['predicted_velocity']
    checks = dict(perfect=metric(a,b,a,b),ignored=metric(a,b,predicted,predicted),swapped=metric(a,b,b,a))
    for name,expected in [('perfect',1.),('ignored',0.),('swapped',-1.)]:
        if abs(checks[name]['normalized_alignment']-expected) > 1e-12: raise ValueError('arithmetic control failed')
    if torch.cuda.is_initialized(): raise ValueError('CPU check initialized CUDA')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'report.json',dict(implementation_sha256=sha(Path(__file__)),source_saved_tensors_sha256=sha(source),
        source_geometry=list(a.shape),constructed_second_target='spatial roll of actual preflight target velocity',
        constructed_perfect_ignored_and_swapped_fields=True,actual_conditional_model_observations=False,
        checks=checks,neural_models_loaded=0,neural_calls=0,general_ilm_goal_complete=False))
    print('Actual saved tensor geometry: perfect/ignored/swapped arithmetic gives1/0/-1. No model called.')


def prepare(training, assessment, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('hide CUDA for preparation')
    code = code_identity(); meta = json.loads((training/'preparation.json').read_text())
    if identity() != meta['code_sha256']: raise ValueError('training sources changed')
    deployment = training/'deployment/deployment.json'; info = json.loads(deployment.read_text())
    if info['completed_updates'] != 1536: raise ValueError('fixed final model required')
    source_path = ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    assessment_prep = json.loads((assessment/'preparation.json').read_text())
    if sha(source_path) != assessment_prep['source_QA_sha256']: raise ValueError('owning source QA changed')
    accepted = {r['id']:{a['text'] for a in r['references']} for r in map(json.loads,source_path.read_text().splitlines())}
    rows = []
    for variant in ('normal','question_erased'):
        name = variant+'_references.jsonl'; path = training/'evaluation'/name
        if sha(path) != meta['evaluation']['files_sha256'][name]: raise ValueError('reference manifest changed')
        source = [json.loads(x) for x in path.read_text().splitlines()]
        selected = ([r for r in source if r['split']=='train'][:8]+[r for r in source if r['split']=='validation'][:8]) if variant=='normal' else source[:2]
        if len(selected) != (16 if variant=='normal' else 2): raise ValueError('complete fixed probe subset required')
        for i in range(0,len(selected),2):
            pair = selected[i:i+2]
            if pair[0]['pair'] != pair[1]['pair'] or pair[0]['id'] == pair[1]['id']:
                raise ValueError('complete two-question passage pair required')
            if accepted[pair[0]['id']] & accepted[pair[1]['id']]: raise ValueError('source-accepted answer sets overlap')
            group = len(rows)//2
            for own,other in (pair,pair[::-1]):
                case = dict(id=own['id'],variant=variant,split=own['split'],pair=own['pair'],
                    own=str((path.parent/own['input']).resolve()),other=str((path.parent/other['input']).resolve()),
                    target=str((path.parent/own['target']).resolve()),own_sha256=own['input_sha256'],
                    other_sha256=other['input_sha256'],target_sha256=own['target_sha256'],
                    noise_seed=20269900+group,feature_seed=20269930+group,codec_seed=20269960+group)
                for key in ('own','other','target'):
                    if sha(case[key]) != case[key+'_sha256']: raise ValueError('bound complete image changed')
                if variant=='question_erased' and case['own_sha256'] != case['other_sha256']:
                    raise ValueError('question-erased pair must have identical pixels')
                rows.append(case)
    if Counter((r['variant'],r['split']) for r in rows) != Counter({('normal','train'):8,('normal','validation'):8,('question_erased','validation'):2}):
        raise ValueError('fixed18-case diagnostic required')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'cases.json',rows)
    save_json(output/'preparation.json',dict(code_sha256=code,training_code_sha256=meta['code_sha256'],
        deployment=str(deployment),deployment_sha256=sha(deployment),weights_sha256=info['weights_sha256'],
        complete_assessment=str(assessment),cases_sha256=sha(output/'cases.json'),indices=list(INDICES),
        source_QA_sha256=sha(source_path),accepted_answer_sets_disjoint=True,
        arm_order=['final','initial'],source_revision=vs.REVISION,scheduled_denoiser_forwards=144,
        positive_fixed_task_field_only=True,autonomous_generation=False,optimizer_updates=0,
        complete_assessment_must_finish_before_probe=True,general_ilm_goal_complete=False))
    print('Prepared18 image cases and144 future denoiser forwards; no model loaded.')


def unchanged(output):
    meta = json.loads((output/'preparation.json').read_text())
    if code_identity() != meta['code_sha256'] or identity() != meta['training_code_sha256']:
        raise ValueError('registered probe/training implementation changed')
    if sha(output/'cases.json') != meta['cases_sha256'] or sha(meta['deployment']) != meta['deployment_sha256']:
        raise ValueError('registered cases or deployment changed')
    if sha(Path(meta['deployment']).parent/'adapters.safetensors') != meta['weights_sha256']:
        raise ValueError('fixed final weights changed')
    return meta


def require_complete_assessment(meta):
    assessment = Path(meta['complete_assessment'])
    if (json.loads((assessment/'runtime.json').read_text())['phase'] != 'ready_for_raw_review' or
        json.loads((assessment/'assessment.json').read_text())['actual_new_generations'] != 384):
        raise ValueError('complete384-page assessment must finish before a field probe')


def arm(output, name):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0': raise ValueError('use GPU0 only')
    meta = unchanged(output); require_complete_assessment(meta)
    rows = json.loads((output/'cases.json').read_text())
    folder = output/name;folder.mkdir(exist_ok=False);started = time.perf_counter()
    with gpu_zero() as device:
        import torch
        from safetensors.torch import load_file
        from diffusers import FlowMatchEulerDiscreteScheduler
        from ilm.image_edit.native_flow_step import add_adapters, frozen_input_features, encode_target
        from ilm.image_edit.visual_flow_learning import interpolate, denoising_loss
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        source = vs.verify_release();pipe,loader = vs.load_pipeline(device)
        vs.install_compiled(pipe,CONDITION);removal = vs.remove_lexical_modules(pipe)
        info = json.loads(Path(meta['deployment']).read_text())
        if source != info['source_sha256']: raise ValueError('fixed source release changed')
        torch.manual_seed(20269880); add_adapters(pipe,info['rank'])
        if name=='final': restore_adapters(pipe,load_file(str(Path(meta['deployment']).parent/'adapters.safetensors'),device='cpu'))
        for part in ('transformer','connector_module'): getattr(pipe,part).eval()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(vs.SNAPSHOT/'scheduler'),local_files_only=True)
        if len(scheduler.sigmas) != 1000 or float(scheduler.sigmas[0]) != 1: raise ValueError('fresh native training grid required')
        results=[]
        with torch.no_grad():
            for i,row in enumerate(rows):
                unchanged(output)
                conditions=[]
                for which in ('own','other'):
                    if sha(row[which]) != row[which+'_sha256']: raise ValueError('input image changed')
                    torch.manual_seed(row['feature_seed']);torch.cuda.manual_seed(row['feature_seed'])
                    merged,reference,feature_meta = frozen_input_features(pipe,Path(row[which]))
                    pooled,sequence = pipe.connector_module.llm2dit(merged)
                    conditions.append((pooled[:1],sequence[:1],reference));del merged
                if sha(row['target']) != row['target_sha256']: raise ValueError('reference image changed')
                torch.manual_seed(row['codec_seed']);torch.cuda.manual_seed(row['codec_seed'])
                clean = encode_target(pipe,Path(row['target']))
                generator = torch.Generator(device=device).manual_seed(row['noise_seed'])
                noise = torch.randn(clean.shape,dtype=clean.dtype,device=device,generator=generator)
                tensors=dict(clean=clean.cpu(),noise=noise.cpu());points=[]
                for index in INDICES:
                    sigma = scheduler.sigmas[index:index+1].to(device=device,dtype=clean.dtype).reshape(1,1,1,1)
                    timestep = scheduler.timesteps[index:index+1].to(device)
                    flow = interpolate(clean,noise,sigma,timestep,torch.tensor([index]))
                    if index==0 and (float(sigma.item())!=1 or not torch.equal(flow.noisy_latents,noise)):
                        raise ValueError('full-noise forward contains target information')
                    prefix='point'+str(index)+'_';tensors[prefix+'noisy']=flow.noisy_latents.cpu();tensors[prefix+'target']=flow.velocity.cpu()
                    losses=[]
                    for which,(pooled,sequence,reference) in zip(('own','other'),conditions):
                        loss,predicted = denoising_loss(pipe.transformer,flow,sequence_condition=sequence,
                            pooled_condition=pooled,reference_latents=[[reference]])
                        tensors[prefix+which]=predicted.cpu();losses.append(float(loss))
                    if row['variant']=='question_erased' and not torch.equal(tensors[prefix+'own'],tensors[prefix+'other']):
                        raise ValueError('identical erased conditions yield different fields')
                    points.append(dict(index=index,sigma=float(sigma.item()),timestep=float(timestep.item()),
                        native_FP32_MSE_own=losses[0],native_FP32_MSE_other=losses[1],
                        native_FP32_loss_advantage=losses[1]-losses[0],target_free_noisy_frame=index==0))
                path=folder/f'{i:03}.pt';torch.save(tensors,path)
                results.append(dict(**row,points=points,actual_tensor_file=path.name,actual_tensor_file_sha256=sha(path)))
                save_json(folder/'outputs.json',results)
                save_json(folder/'status.json',dict(phase='probing',pid=os.getpid(),completed=i+1,total=18))
                del tensors,conditions,clean,noise,flow,predicted,pooled,sequence,reference
                print(f'Probed{i+1}/18',flush=True)
        unchanged(output)
        save_json(folder/'report.json',dict(arm=name,preparation_sha256=sha(output/'preparation.json'),
            actual_denoiser_forwards=72,actual_cases=18,learned_weights_loaded=name=='final',source=source,removal=removal,
            runtime_parameters=vs.parameters(pipe),elapsed_seconds=time.perf_counter()-started,
            peak_allocated_vram_GiB=torch.cuda.max_memory_allocated()/1024**3,
            positive_fixed_task_only=True,autonomous_generations=0,optimizer_updates=0,semantic_learning_validated=False))
    save_json(folder/'status.json',dict(phase='completed',pid=None))


def collect(output):
    import torch
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('hide CUDA for collection')
    meta=unchanged(output);require_complete_assessment(meta);results={};bindings={}
    cases=json.loads((output/'cases.json').read_text())
    for name in meta['arm_order']:
        folder=output/name;report=json.loads((folder/'report.json').read_text());rows=json.loads((folder/'outputs.json').read_text())
        if report['actual_cases']!=18 or report['actual_denoiser_forwards']!=72 or len(rows)!=18:
            raise ValueError('complete two-arm field diagnostic required')
        if report['preparation_sha256']!=sha(output/'preparation.json') or report['learned_weights_loaded']!=(name=='final'):
            raise ValueError('owning model arm or preparation differs')
        for row,wanted in zip(rows,cases):
            if any(row[k]!=v for k,v in wanted.items()) or [p['index'] for p in row['points']]!=list(INDICES):
                raise ValueError('original complete case order or grid differs')
        scores=[]
        for i in range(0,18,2):
            a,b=rows[i:i+2];actual=[]
            for row in (a,b):
                path=folder/row['actual_tensor_file']
                if sha(path)!=row['actual_tensor_file_sha256']: raise ValueError('actual field tensors changed')
                fields=torch.load(path,map_location='cpu',weights_only=True)
                if not torch.equal(fields['point0_noisy'],fields['noise']): raise ValueError('full-noise frame contains target')
                for point in row['points']:
                    prefix='point'+str(point['index'])+'_'
                    for which in ('own','other'):
                        mse=float((fields[prefix+which].float()-fields[prefix+'target'].float()).square().mean())
                        if not math.isclose(mse,point['native_FP32_MSE_'+which],rel_tol=1e-6,abs_tol=1e-8):
                            raise ValueError('actual native point MSE does not reproduce')
                actual.append(fields)
            x,y=actual
            if a['pair']!=b['pair'] or not torch.equal(x['noise'],y['noise']): raise ValueError('paired common noise differs')
            if not (torch.equal(x['point0_own'],y['point0_other']) and torch.equal(x['point0_other'],y['point0_own'])):
                raise ValueError('full-noise field depends on reference scoring target')
            score=metric(x['point0_target'],y['point0_target'],x['point0_own'],y['point0_own'])
            if a['variant']=='question_erased' and score['normalized_alignment']!=0: raise ValueError('erased control alignment is nonzero')
            scores.append(dict(ids=[a['id'],b['id']],variant=a['variant'],split=a['split'],**score))
        results[name]=scores;bindings[name]=dict(report_sha256=sha(folder/'report.json'),outputs_sha256=sha(folder/'outputs.json'))
    if torch.cuda.is_initialized(): raise ValueError('offline collection initialized CUDA')
    save_json(output/'result.json',dict(preparation_sha256=sha(output/'preparation.json'),rows=results,bindings=bindings,
        actual_denoiser_forwards=144,all_full_noise_predictions_swap_exactly=True,all_erased_control_fields_exact=True,
        autonomous_generations=0,optimizer_updates=0,normalized_alignment_is_not_accuracy=True,
        semantic_learning_validated=False,general_ilm_goal_complete=False))
    print(json.dumps(results,indent=2))


def owner(pid):
    try:
        path=Path('/proc')/str(pid);stat=(path/'stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z': return None
        return dict(pid=pid,start_ticks=stat[19],command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError: return None


def run(output, wait_pid):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('hide CUDA for coordinator')
    meta=unchanged(output);waiting=owner(wait_pid)
    if not waiting or 'scripts/run_native_visual_flow_assessment.py' not in waiting['command'] or meta['complete_assessment'] not in waiting['command']:
        raise ValueError('wait for the exact live complete-assessment coordinator')
    lock=(output/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'runtime.json').exists(): raise ValueError('preserve existing probe attempt; no automatic rerun')
    child=None;phase='waiting_complete_assessment'
    def status(**extra): save_json(output/'runtime.json',dict(phase=phase,pid=os.getpid() if phase not in ('failed','completed') else None,
        child_pid=child.pid if child is not None and child.poll() is None else None,wait_owner=waiting,
        gpu=None,novnc_url=None,general_ilm_goal_complete=False,**extra))
    def stop(signum,frame): raise InterruptedError('probe coordinator signal'+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
    try:
        status()
        while owner(wait_pid)==waiting:
            unchanged(output);time.sleep(30)
        if owner(wait_pid) is not None: raise ValueError('assessment PID was reused')
        require_complete_assessment(meta)
        for name in (*meta['arm_order'],'collect'):
            unchanged(output);phase=name
            command=[str(Path.home()/'.local/bin/with-current-nvidia'),'env','CUDA_DEVICE_ORDER=PCI_BUS_ID',
                'CUDA_VISIBLE_DEVICES='+('' if name=='collect' else '0'),'PYTHONNOUSERSITE=1','OMP_NUM_THREADS=4',
                'OPENBLAS_NUM_THREADS=2','MKL_NUM_THREADS=4','PYTHONPATH=.',str(ROOT.parent/'LocalLLM/.local/image-generation/venv/bin/python'),
                'scripts/run_native_flow_condition_probe.py','--stage','collect' if name=='collect' else 'arm','--output',str(output)]
            if name!='collect':command+=['--arm',name]
            with (output/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);status()
                try:
                    if child.wait():raise RuntimeError('field probe stage failed: '+name)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        phase='completed';status()
    except BaseException as error:
        phase='failed';status(reason=str(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('math','prepare','run','arm','collect'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--training-run',type=Path);p.add_argument('--assessment',type=Path)
    p.add_argument('--wait-for-assessment-pid',type=int);p.add_argument('--arm',choices=('final','initial'));args=p.parse_args()
    output=args.output.resolve()
    if args.stage=='math':math_check(output)
    elif args.stage=='prepare':prepare(args.training_run.resolve(),args.assessment.resolve(),output)
    elif args.stage=='run':run(output,args.wait_for_assessment_pid)
    elif args.stage=='arm':arm(output,args.arm)
    else:collect(output)
