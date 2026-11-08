#!/usr/bin/env python3
"""Wait for actual final training, then run all eight fixed image cohorts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.image_edit.native_flow_run import identity

FILES=('scripts/run_native_visual_flow_assessment.py','scripts/assess_native_visual_flow.py',
    'ilm/image_edit/native_flow_assessment.py','scripts/score_paired_writing.py',
    'scripts/evaluate_image_edit_baseline.py','references/native_visual_flow_assessment_protocol.md')
VARIANTS=('normal','question_erased','passage_erased','copy')
CALIBRATIONS=ROOT/'artifacts/native_visual_flow_reference_calibration_v1'


def assessment_identity():
    if subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=ROOT,text=True):
        raise ValueError('commit the complete assessment implementation and protocol first')
    return {name:sha(ROOT/name) for name in FILES}


def owner(pid):
    path=Path('/proc')/str(pid)
    try:return dict(pid=pid,start_ticks=(path/'stat').read_text().rsplit(')',1)[1].split()[19],
        command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None


def prepare(training,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for CPU preparation')
    code=assessment_identity();meta=json.loads((training/'preparation.json').read_text())
    if identity()!=meta['code_sha256']:raise ValueError('training sources changed')
    calibrations={}
    for name,digest in meta['evaluation']['files_sha256'].items():
        if sha(training/'evaluation'/name)!=digest:raise ValueError('registered evaluation images/manifest changed')
    for variant in VARIANTS:
        index=training/'evaluation'/(variant+'_inputs.json');manifest=training/'evaluation'/(variant+'_references.jsonl')
        rows=json.loads(index.read_text());refs=[json.loads(x) for x in manifest.read_text().splitlines()]
        calibration=CALIBRATIONS/variant/'report.json';record=json.loads(calibration.read_text())
        if [r['id'] for r in rows]!=[r['id'] for r in refs] or record['cohort']!=[r['id'] for r in refs]:
            raise ValueError('complete fixed assessment cohort/calibration required')
        if record['manifest_sha256']!=sha(manifest) or not record['complete_cohort'] or record['stage']!='calibrate':
            raise ValueError('owning original reference calibration required')
        for name,digest in record['implementation'].items():
            if sha(ROOT/name)!=digest:raise ValueError('calibration evaluator implementation changed')
        for r in rows:
            if set(r)!={'id','input','input_sha256','seed'} or sha(index.parent/r['input'])!=r['input_sha256']:
                raise ValueError('input-only index binding changed')
        calibrations[variant]=dict(path=str(calibration),sha256=sha(calibration))
    source=ROOT/'data/visual_qa/cmrc2018_short/train.jsonl'
    page_prep=json.loads((ROOT/'artifacts/visual_flow_pages_v1/report.json').read_text())
    if sha(source)!=page_prep['original_QA_source_sha256']:raise ValueError('original QA source changed')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'preparation.json',dict(training_run=str(training),training_preparation_sha256=sha(training/'preparation.json'),
        training_code_sha256=meta['code_sha256'],assessment_code_sha256=code,source_QA_sha256=sha(source),
        evaluation_files_sha256=meta['evaluation']['files_sha256'],calibrations=calibrations,
        arm_order=['final','initial'],variant_order=list(VARIANTS),scheduled_new_generations=384,
        selected_final_update=1536,intermediate_checkpoint_selection=False,model_loaded=False,general_ilm_goal_complete=False))
    print('Prepared all384 fixed initial/final generation calls; no model loaded.')


def run(training,output,wait_pid,resume):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('hide CUDA for the lightweight coordinator')
    code=assessment_identity();prepared=json.loads((output/'preparation.json').read_text())
    if str(training)!=prepared['training_run'] or code!=prepared['assessment_code_sha256']:
        raise ValueError('exact registered training run and assessment sources required')
    lock=(output/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (output/'runtime.json').exists() and not resume:raise ValueError('explicit unchanged --resume required')
    completed=json.loads((output/'completed_stages.json').read_text()) if (output/'completed_stages.json').exists() else {}
    waiting=owner(wait_pid) if wait_pid else None
    if waiting and ('scripts/train_native_visual_flow.py' not in waiting['command'] or str(training) not in waiting['command']):
        raise ValueError('wait owner is not the exact native training process')
    phase='starting';child=None;started=time.time()

    def unchanged():
        if (assessment_identity()!=code or identity()!=prepared['training_code_sha256'] or
            sha(training/'preparation.json')!=prepared['training_preparation_sha256']):
            raise ValueError('registered live training/assessment implementation changed')
        for name,digest in prepared['evaluation_files_sha256'].items():
            if sha(training/'evaluation'/name)!=digest:raise ValueError('fixed assessment input or reference changed')
        for item in prepared['calibrations'].values():
            if sha(item['path'])!=item['sha256']:raise ValueError('original reference calibration changed')

    def status(**extra):
        save_json(output/'runtime.json',dict(phase=phase,coordinator_pid=None if phase in ('failed','ready_for_raw_review') else os.getpid(),
            child_pid=child.pid if child is not None and child.poll() is None else None,started_unix=started,
            training_owner=waiting,assessment_code_sha256=code,gpu=None,novnc_url=None,general_ilm_goal_complete=False,**extra))

    def interrupt(signum,frame):raise InterruptedError('assessment signal '+str(signum))
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(signum,interrupt)

    def stage(name,script,arguments,gpu,primary):
        nonlocal phase,child
        unchanged();phase=name
        if name in completed:
            if sha(primary)!=completed[name]['primary_sha256']:raise ValueError('completed stage binding changed')
            return
        command=[str(Path.home()/'.local/bin/with-current-nvidia'),'env','CUDA_DEVICE_ORDER=PCI_BUS_ID',
            'CUDA_VISIBLE_DEVICES='+('0' if gpu else ''),'PYTHONNOUSERSITE=1','OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=2',
            'MKL_NUM_THREADS=4','PYTHONPATH=.',str(ROOT.parent/'LocalLLM/.local/image-generation/venv/bin/python'),script,*map(str,arguments)]
        attempt=1+len(list(output.glob(name+'.attempt_*.log')))
        with (output/f'{name}.attempt_{attempt:03}.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(child_command=command,child_gpu=0 if gpu else None)
            try:
                if child.wait():raise RuntimeError(name+' failed; preserve outputs, with no automatic restart or quality reroll')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        unchanged();completed[name]=dict(primary=str(primary),primary_sha256=sha(primary),command=command)
        save_json(output/'completed_stages.json',completed)

    try:
        phase='waiting_training';status()
        while waiting and owner(waiting['pid'])==waiting:
            unchanged();time.sleep(30)
        if waiting and owner(waiting['pid']) is not None:raise ValueError('training PID was reused')
        unchanged();terminal=json.loads((training/'status.json').read_text());trained=json.loads((training/'report.json').read_text())
        if terminal['phase']!='completed' or trained['updates']!=1536 or trained['checkpoint_sha256']!=sha(training/'checkpoint.pt'):
            raise ValueError('owner ended without complete final1536-update training')
        stage('export','scripts/train_native_visual_flow.py',['--stage','export','--output',training],False,training/'deployment/deployment.json')
        for arm in prepared['arm_order']:
            for variant in prepared['variant_order']:
                name=arm+'_'+variant;folder=output/(name+'_inference')
                arguments=['--deployment',training/'deployment','--inputs',training/'evaluation'/(variant+'_inputs.json'),'--output',folder]
                if arm=='initial':arguments.append('--initial-source')
                stage('infer_'+name,'scripts/infer_native_visual_flow.py',arguments,True,folder/'report.json')
                stage('score_'+name,'scripts/score_paired_writing.py',['--stage','score','--task','copy' if variant=='copy' else 'qa',
                    '--train-limit','32','--manifest',training/'evaluation'/(variant+'_references.jsonl'),
                    '--calibration',prepared['calibrations'][variant]['path'],'--generations',folder/'outputs.json',
                    '--output',output/(name+'_ocr')],False,output/(name+'_ocr/report.json'))
        stage('assessment','scripts/assess_native_visual_flow.py',['--training-run',training,'--output',output],False,output/'assessment.json')
        phase='ready_for_raw_review';status(owned_child_exited=True)
    except BaseException as error:
        phase='failed';status(reason=str(error),owned_child_exited=child is None or child.poll() is not None);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run'),required=True)
    p.add_argument('--training-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-for-training-pid',type=int);p.add_argument('--resume',action='store_true');args=p.parse_args()
    training=args.training_run.resolve();output=args.output.resolve()
    prepare(training,output) if args.stage=='prepare' else run(training,output,args.wait_for_training_pid,args.resume)
