#!/usr/bin/env python3
"""Bound admission-only deployment repair and the remaining matched comparison."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.register_visual_context_comparison import ROOT,REG,load,sha,validate,require_pushed
from scripts.resume_visual_context_comparison import RECOVERY,resource_sample,admission_ready
from scripts.verify_visual_context_deployment_sources import verify_sources

EXECUTION=ROOT/'references/evidence/visual_context_deployment_execution_registration_20261117.json'
PROTOCOL=ROOT/'references/visual_context_deployment_execution_protocol_20261117.md'
PREFLIGHT=ROOT/'references/evidence/visual_context_deployment_cpu_preflight_20261117.json'
NEW_NAMES={'continue_visual_context_deployment.py','infer_visual_context_deployment.py',
    'verify_visual_context_deployment_cold.py','run_visual_context_comparison.py',
    'train_visual_context_comparison.py','infer_trainable_visual_context.py',
    'verify_trainable_visual_context_cold.py','resume_visual_context_comparison.py'}


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')


def model_owners():
    owners=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cwd=(proc/'cwd').resolve();argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (OSError,RuntimeError):continue
        if state=='Z' or not (cwd==ROOT or ROOT in cwd.parents):continue
        if any(Path(a).name in NEW_NAMES for a in argv) or (
                any(Path(a).name=='register_visual_context_comparison.py' for a in argv) and '--audit' in argv):
            owners.append(dict(pid=int(proc.name),argv=argv))
    return owners


def treatment_folders(reg):
    base=reg['run']+'/trainable/'
    return [base+'generated_step_0000000_admission_recovery1',
        *[base+f'generated_step_{s:07d}' for s in reg['snapshot_updates'][1:]],
        base+'generated_final_long_books']


def stages(reg):
    base=reg['run'];fit=base+'/frozen/fit';result=[]
    def add(name,args,output,gpu=True):result.append(dict(name=name,arguments=args,output=output,gpu=gpu))
    add('trainable_cold',['scripts/verify_visual_context_deployment_cold.py','--arm','trainable'],
        base+'/trainable/cold_deployment_v1',False)
    add('frozen_fit',['scripts/train_visual_context_comparison.py','--registration',str(REG.relative_to(ROOT)),
        '--arm','frozen'],fit)
    add('frozen_audit',['scripts/register_visual_context_comparison.py','--audit','frozen'],None,False)
    for index,step in enumerate(reg['snapshot_updates']):
        output=base+f'/frozen/generated_step_{step:07d}'
        add('frozen_generate_'+str(index),['scripts/infer_visual_context_deployment.py','--manifest',
            reg['evaluation']+'/inputs/manifest.json','--checkpoint',fit+f'/model_step_{step:07d}.pt',
            '--output',output,'--profile','all' if index in (0,4) else 'normal'],output)
    output=base+'/frozen/generated_final_long_books'
    add('frozen_long_books',['scripts/infer_visual_context_deployment.py','--manifest',reg['evaluation']+'/inputs/manifest.json',
        '--checkpoint',fit+'/model_step_0008172.pt','--output',output,'--profile','long_books','--maximum-glyphs','640'],output)
    add('frozen_cold',['scripts/verify_visual_context_deployment_cold.py','--arm','frozen'],base+'/frozen/cold_deployment_v1',False)
    add('prepare_review',['scripts/review_visual_context_comparison.py','prepare'],base+'/review',False)
    return result


def terminal_and_complete(reg):
    run=ROOT/reg['run'];old=load(run/'status.json');previous=load(run/'admission_recovery1/status.json')
    if (not old['owner_exited'] or old['phase']!='failed_serial_comparison'
            or not previous['owner_exited'] or previous['phase']!='failed_serial_recovery'
            or previous['stage']!='trainable_cold_replay'
            or (ROOT/'artifacts/trainable_visual_context_comparison_admission_recovery1.exit').read_text().strip()!='1'):
        raise ValueError('Preserve both terminal executions and the failed strict cold stage')
    if model_owners():raise ValueError('A comparison model owner is still live')
    counts=[]
    for name in treatment_folders(reg):
        folder=ROOT/name;report=load(folder/'report.json')
        if report['request_sha256']!=sha(folder/'request.json') or report['reference_inputs'] or report['output_repaired']:
            raise ValueError('Preserve actual treatment outputs')
        counts.append(len(report['records']))
    if counts!=[120,44,44,44,120,24]:raise ValueError('Keep all396 treatment streams')
    for item in stages(reg):
        if item['output'] and (ROOT/item['output']).exists():raise ValueError('Do not overwrite a remaining stage: '+item['output'])
    return run


def verify_previous_bindings(previous):
    for name,digest in previous['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Original recovery binding changed: '+name)


def register():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA while binding execution')
    equivalence=verify_sources();reg=validate(REG);previous=load(RECOVERY);verify_previous_bindings(previous)
    run=terminal_and_complete(reg);preflight=load(PREFLIGHT)
    if not preflight['passed'] or preflight['new_model_calls']!=0:raise ValueError('CPU source/guard checks must pass')
    for name,digest in preflight['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('CPU checks predate a source change: '+name)
    paths=[Path(__file__).resolve(),PROTOCOL,PREFLIGHT,ROOT/'tests/test_visual_context_deployment.py',
        ROOT/'scripts/verify_visual_context_deployment_cold.py',ROOT/'scripts/verify_visual_context_deployment_sources.py',
        ROOT/'scripts/infer_visual_context_deployment.py',ROOT/'ilm/visual_lm/visual_context_deployment_resources.py',
        run/'admission_recovery1/status.json',run/'admission_recovery1/stages.jsonl',run/'admission_recovery1/admission.jsonl',
        run/'admission_recovery1/trainable_cold_replay.log',ROOT/'artifacts/trainable_visual_context_comparison_admission_recovery1.exit',
        ROOT/'references/evidence/visual_context_cold_guard_failure_20261117.json']
    for folder in treatment_folders(reg):paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    cold=run/'trainable/cold'
    paths.extend(p for p in cold.rglob('*') if p.is_file() and not p.name.startswith('access.'))
    digests={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if set(digests)&set(previous['input_sha256']):raise ValueError('Do not duplicate or replace prior bindings')
    binding=dict(schema='visual-context-deployment-execution-v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        original_registration_sha256=sha(REG),previous_recovery_registration_sha256=sha(RECOVERY),
        input_sha256=digests,retained_failed_trace_sha256=sorted(sha(p) for p in cold.glob('access.*')),
        source_equivalence=equivalence,execution_directory=reg['run']+'/deployment_execution_v1',stages=stages(reg),
        retained_treatment_runs=treatment_folders(reg),retained_treatment_streams=396,
        remaining_control_streams=396,total_language_streams=792,
        admission=dict(host_available_gib=48.,gpu0_free_mib=20480,disk_free_gib=48.,max_some_pressure_avg10=.5,
            max_full_pressure_avg10=.1,quiet_samples=2,interval_seconds=10.,maximum_total_queue_seconds=1800.),
        original_599_bindings_preserved=True,original_21_recovery_bindings_preserved=True,
        original_strict_artifact_audit_unchanged=True,old_metadata_exemptions=False,
        treatment_fit_and_generation_repeated=False,frozen_fit_runs=1,automatic_retry=False,
        fresh_cold_replays=2,prior_failed_cold_call_preserved_and_additional=True,
        language_quality_measured=False,goal_achieved=False)
    write_new(EXECUTION,binding)
    print(json.dumps(dict(registration=str(EXECUTION.relative_to(ROOT)),stages=len(binding['stages']),
        additional_bindings=len(digests),retained_treatment_streams=396,remaining_control_streams=396)))


def run():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('Keep physical GPU0 and registered workspace')
    reg=validate(REG);binding=load(EXECUTION);previous=load(RECOVERY)
    if binding['original_registration_sha256']!=sha(REG) or binding['previous_recovery_registration_sha256']!=sha(RECOVERY):
        raise ValueError('Prior registrations changed')
    verify_previous_bindings(previous);terminal_and_complete(reg)
    if binding['stages']!=stages(reg) or verify_sources()!=binding['source_equivalence']:raise ValueError('Execution source or plan changed')
    for name,digest in binding['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('New execution binding changed: '+name)
    cold=ROOT/reg['run']/'trainable/cold'
    if sorted(sha(p) for p in cold.glob('access.*'))!=binding['retained_failed_trace_sha256']:raise ValueError('Preserve failed trace')
    require_pushed(EXECUTION)
    out=ROOT/binding['execution_directory'];out.mkdir(exist_ok=False);child=None;stage='starting';queued=0.
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage=stage,
            child_pid=child.pid if child and child.poll() is None else None,admission_queue_seconds=queued,automatic_retry=False,**extra)
        temp=out/'status.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(out/'status.json')
    def stop(_signal,_frame):raise KeyboardInterrupt('Deployment execution termination requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def admission():
        nonlocal queued
        settings=binding['admission'];quiet=0;began=time.monotonic()
        try:
            while True:
                sample=resource_sample()
                with (out/'admission.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,sample=sample,unix=time.time()))+'\n')
                quiet=quiet+1 if admission_ready(sample,settings) else 0
                elapsed=queued+time.monotonic()-began
                status('waiting_for_unchanged_resource_limits',consecutive_quiet_samples=quiet,current_stage_queue_seconds=time.monotonic()-began)
                if elapsed>settings['maximum_total_queue_seconds']:raise RuntimeError('Admission queue exhausted')
                if quiet>=settings['quiet_samples']:return
                if elapsed+settings['interval_seconds']>settings['maximum_total_queue_seconds']:raise RuntimeError('Admission queue exhausted')
                time.sleep(settings['interval_seconds'])
        finally:queued+=time.monotonic()-began
    try:
        with (ROOT/reg['run']/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (out/'registration_sha256.txt').write_text(sha(EXECUTION)+'\n')
            write_new(out/'resource_inventory.json',dict(processes=model_owners(),memory=subprocess.check_output(['free','-h'],text=True),
                tmux=subprocess.run(['tmux','list-sessions','-F','#S'],capture_output=True,text=True).stdout,
                listening_ports=subprocess.check_output(['ss','-ltn'],text=True),foreign_processes_changed=False))
            generated=list(binding['retained_treatment_runs'])
            for item in binding['stages']:
                stage=item['name']
                if item['output'] and (ROOT/item['output']).exists():raise ValueError('Stage destination already exists: '+item['output'])
                if stage=='frozen_fit':
                    receipt=load(ROOT/reg['run']/'trainable/cold_deployment_v1/verification.json')
                    if (not receipt['actual_pixels_stopping_and_feedback_exact'] or receipt['other_project_artifacts_read']
                            or receipt['execution_registration_sha256']!=sha(EXECUTION)):
                        raise ValueError('Strict treatment cold verification required before control training')
                if item['gpu'] or stage.endswith('_cold'):admission()
                if stage=='prepare_review':
                    left=load(ROOT/reg['run']/'trainable/fit/initial.json');right=load(ROOT/reg['run']/'frozen/fit/initial.json')
                    if left['model_sha256']!=right['model_sha256'] or left['objective_sha256']!=right['objective_sha256']:
                        raise ValueError('Initial full model/objectives differ across arms')
                    write_new(ROOT/reg['run']/'generation_manifest.json',dict(runs=generated,expected_streams=792,
                        registration_sha256=sha(REG),deployment_execution_registration_sha256=sha(EXECUTION),
                        matched_initial_state_exact=True,previous_failed_execution_receipts_preserved=True))
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                if not item['gpu']:env['CUDA_VISIBLE_DEVICES']=''
                began=time.monotonic()
                with (out/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',*item['arguments']],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    status('running_serial_stage');code=child.wait()
                with (out/'stages.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,exit_code=code,elapsed_seconds=time.monotonic()-began))+'\n')
                if code:raise RuntimeError(f'Stage {stage} exited {code}; no automatic retry')
                child=None
                if '_generate_' in stage or stage.endswith('_long_books'):generated.append(item['output'])
                status('serial_stage_complete')
            status('awaiting_frozen_visual_readings',streams=792,language_scored=False)
    except BaseException as error:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status('failed_deployment_execution',error=repr(error));raise
    finally:
        if (out/'status.json').exists():
            value=load(out/'status.json');value.update(pid=None,child_pid=None,owner_exited=True)
            (out/'status.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));args=parser.parse_args()
    if args.action=='register':register()
    else:run()
