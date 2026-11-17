#!/usr/bin/env python3
"""One bound continuation after the pre-model resource-admission failure."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.register_visual_context_comparison import ROOT, REG, load, sha, validate, require_pushed

RECOVERY=ROOT/'references/evidence/visual_context_admission_recovery_registration_20261117.json'
PROTOCOL=ROOT/'references/visual_context_admission_recovery_protocol_20261117.md'
SCRIPT=Path(__file__).resolve()
FAILURE="RuntimeError('bridge memory pressure entry limit')"


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')


def failed_admission_evidence(folder):
    s=load(folder/'status.json')
    if (s['phase']!='failed_visual_context_inference' or s['error']!=FAILURE
            or s['position'] or s['automatic_retry']
            or {p.name for p in folder.iterdir()}!={'status.json','request.json'}):
        raise ValueError('Only the preserved zero-output, pre-model admission failure is eligible')
    return s


def stages(reg,failed_seconds):
    run=reg['run'];result=[]
    def add(name,args,output,gpu=True):result.append(dict(name=name,arguments=args,output=output,gpu=gpu))
    for arm in ('trainable','frozen'):
        fit=f'{run}/{arm}/fit'
        if arm=='frozen':
            add('frozen_fit',['scripts/train_visual_context_comparison.py','--registration',str(REG.relative_to(ROOT)),'--arm',arm],fit)
            add('frozen_audit',['scripts/register_visual_context_comparison.py','--audit',arm],None,False)
        for index,step in enumerate(reg['snapshot_updates']):
            output=f'{run}/{arm}/generated_step_{step:07d}'
            if arm=='trainable' and index==0:output+='_admission_recovery1'
            args=['scripts/infer_trainable_visual_context.py','--manifest',reg['evaluation']+'/inputs/manifest.json',
                  '--checkpoint',fit+f'/model_step_{step:07d}.pt','--output',output,
                  '--profile','all' if index in (0,4) else 'normal']
            if arm=='trainable' and index==0:args+=['--seconds',str(7200.-failed_seconds)]
            add(arm+'_generate_'+str(index),args,output)
        output=f'{run}/{arm}/generated_final_long_books'
        add(arm+'_long_books',['scripts/infer_trainable_visual_context.py','--manifest',reg['evaluation']+'/inputs/manifest.json',
            '--checkpoint',fit+'/model_step_0008172.pt','--output',output,'--profile','long_books','--maximum-glyphs','640'],output)
        add(arm+'_cold_replay',['scripts/verify_trainable_visual_context_cold.py','--arm',arm],f'{run}/{arm}/cold',False)
    add('prepare_review',['scripts/review_visual_context_comparison.py','prepare'],run+'/review',False)
    return result


def active_model_owners():
    names={'run_visual_context_comparison.py','train_visual_context_comparison.py','infer_trainable_visual_context.py',
           'verify_trainable_visual_context_cold.py','resume_visual_context_comparison.py'}
    owners=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            cwd=(proc/'cwd').resolve()
            argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        except (OSError,RuntimeError):continue
        if state=='Z' or not (cwd==ROOT or ROOT in cwd.parents):continue
        if any(Path(a).name in names for a in argv) or (
                any(Path(a).name=='register_visual_context_comparison.py' for a in argv) and '--audit' in argv):
            owners.append(dict(pid=int(proc.name),argv=argv))
    return owners


def eligible(reg):
    run=ROOT/reg['run'];s=load(run/'status.json')
    if (s['phase']!='failed_serial_comparison' or not s['owner_exited'] or s['pid'] is not None
            or s['child_pid'] is not None or s['stage']!='trainable_generate_0'
            or (ROOT/(reg['run']+'.exit')).read_text().strip()!='1'):
        raise ValueError('Require the terminal original controller and exact first-stage failure')
    owners=active_model_owners()
    if owners:raise ValueError('A model owner remains live: '+repr(owners))
    audit=load(run/'trainable/fit/audit.json');report=load(run/'trainable/fit/report.json')
    if (not audit['audit_passed'] or audit['cuda_initialized'] or not audit['expected_source_order_exact']
            or not audit['latest_matches_full_deploy'] or not audit['optimizer_counters_all_exact']
            or audit['language_quality_proved'] or audit['updates']!=8172 or audit['records']!=13785
            or audit['registration_sha256']!=sha(REG)
            or report['final_checkpoint_sha256']!=audit['final_checkpoint_sha256']):
        raise ValueError('Require the completed audited treatment; never repeat its fit or audit')
    failure=failed_admission_evidence(run/'trainable/generated_step_0000000')
    if not 0<failure['elapsed_seconds']<7200:raise ValueError('Keep the cumulative first-generation budget')
    return failure


def register():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA while binding recovery')
    reg=validate(REG);failure=eligible(reg);run=ROOT/reg['run'];plan=stages(reg,failure['elapsed_seconds'])
    for item in plan:
        if item['output'] and (ROOT/item['output']).exists():raise ValueError('Destination exists: '+item['output'])
    paths=[SCRIPT,PROTOCOL,ROOT/'tests/test_visual_context_admission_recovery.py',run/'status.json',
        ROOT/(reg['run']+'.exit'),ROOT/(reg['run']+'.log'),run/'trainable_generate_0.log',run/'trainable_audit.log']
    paths += [run/'trainable/generated_step_0000000'/n for n in ('status.json','request.json')]
    paths += [run/'trainable/fit'/n for n in ('initial.json','report.json','audit.json','updates.jsonl','history.json','checkpoint_latest.pt')]
    paths += sorted((run/'trainable/fit').glob('model_step_*.pt'))
    if sorted(p.name for p in (run/'trainable/fit').glob('model_step_*.pt')) != [
            f'model_step_{n:07d}.pt' for n in reg['snapshot_updates']]:
        raise ValueError('Preserve exactly the five original fixed checkpoints')
    digests={str(p.relative_to(ROOT)):sha(p) for p in paths}
    final=str((run/'trainable/fit/model_step_0008172.pt').relative_to(ROOT))
    initial=str((run/'trainable/fit/model_step_0000000.pt').relative_to(ROOT))
    if (digests[final]!=load(run/'trainable/fit/audit.json')['final_checkpoint_sha256']
            or digests[initial]!=load(run/'trainable/generated_step_0000000/request.json')['checkpoint_sha256']):
        raise ValueError('An audited final or requested initial checkpoint changed')
    binding=dict(schema='visual-context-admission-recovery-v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        original_registration=str(REG.relative_to(ROOT)),original_registration_sha256=sha(REG),
        recovery_directory=reg['run']+'/admission_recovery1',input_sha256=digests,
        original_bound_sources=len(reg['input_sha256']),failed_generation_seconds=failure['elapsed_seconds'],stages=plan,
        admission=dict(host_available_gib=48.,gpu0_free_mib=20480,disk_free_gib=48.,max_some_pressure_avg10=.5,
            max_full_pressure_avg10=.1,quiet_samples=2,interval_seconds=10.,maximum_total_queue_seconds=1800.),
        skip_completed_treatment_fit_and_audit=True,unchanged_training_and_inference_sources=True,
        original_failed_directory_preserved=True,automatic_retry=False,streams=792,cold_replays_outside_language_streams=2,
        language_quality_proved=False,goal_achieved=False)
    write_new(RECOVERY,binding)
    print(json.dumps(dict(registration=str(RECOVERY.relative_to(ROOT)),stages=len(plan),
        original_bound_sources=binding['original_bound_sources'],new_bindings=len(paths))))


def admission_ready(sample,settings):
    return (sample['available_gib']>=settings['host_available_gib'] and sample['gpu0_free_mib']>=settings['gpu0_free_mib']
        and sample['disk_free_gib']>=settings['disk_free_gib']
        and sample['some_pressure_avg10']<=settings['max_some_pressure_avg10']
        and sample['full_pressure_avg10']<=settings['max_full_pressure_avg10'] and sample['mutex_available'])


def resource_sample():
    import shutil
    from scripts.score_pixel_language_first_blocks import host_resources
    sample=host_resources()
    sample['gpu0_free_mib']=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    sample['disk_free_gib']=shutil.disk_usage(ROOT).free/2**30;sample['mutex_available']=False
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);sample['mutex_available']=True
        except BlockingIOError:pass
    return sample


def run():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID'
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):
        raise ValueError('Keep the registered physical GPU0 and CUDA workspace')
    recovery=load(RECOVERY);reg=validate(REG);failure=eligible(reg)
    if recovery['original_registration_sha256']!=sha(REG) or recovery['stages']!=stages(reg,failure['elapsed_seconds']):
        raise ValueError('Original comparison or continuation plan changed')
    for name,digest in recovery['input_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Recovery-bound artifact changed: '+name)
    require_pushed(RECOVERY)
    out=ROOT/recovery['recovery_directory'];out.mkdir(exist_ok=False)
    child=None;stage='starting';queued=0.
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage=stage,
            child_pid=child.pid if child and child.poll() is None else None,admission_queue_seconds=queued,automatic_retry=False,**extra)
        temporary=out/'status.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(out/'status.json')
    def stop(_signal,_frame):raise KeyboardInterrupt('Recovery termination requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def admission():
        nonlocal queued
        settings=recovery['admission'];quiet=0;began=time.monotonic()
        try:
            while True:
                sample=resource_sample()
                with (out/'admission.jsonl').open('a') as f:f.write(json.dumps(dict(stage=stage,sample=sample,unix=time.time()))+'\n')
                quiet=quiet+1 if admission_ready(sample,settings) else 0
                status('waiting_for_unchanged_resource_limits',consecutive_quiet_samples=quiet,
                       current_stage_queue_seconds=time.monotonic()-began)
                if queued+time.monotonic()-began>settings['maximum_total_queue_seconds']:
                    raise RuntimeError('Recovery admission queue exhausted; no stage retry')
                if quiet>=settings['quiet_samples']:return
                if queued+time.monotonic()-began+settings['interval_seconds']>settings['maximum_total_queue_seconds']:
                    raise RuntimeError('Recovery admission queue exhausted; no stage retry')
                time.sleep(settings['interval_seconds'])
        finally:queued+=time.monotonic()-began
    try:
        with (ROOT/reg['run']/'owner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (out/'registration_sha256.txt').write_text(sha(RECOVERY)+'\n')
            write_new(out/'resource_inventory.json',dict(processes=active_model_owners(),
                tmux=subprocess.run(['tmux','list-sessions','-F','#S'],capture_output=True,text=True).stdout,
                listening_ports=subprocess.run(['ss','-ltn'],capture_output=True,text=True).stdout,
                memory=subprocess.check_output(['free','-h'],text=True),foreign_processes_changed=False))
            generated=[]
            for item in recovery['stages']:
                stage=item['name']
                if item['output'] and (ROOT/item['output']).exists():raise ValueError('Never overwrite a stage: '+item['output'])
                if item['gpu'] or stage.endswith('_cold_replay'):admission()
                if stage=='prepare_review':
                    left=load(ROOT/reg['run']/'trainable/fit/initial.json');right=load(ROOT/reg['run']/'frozen/fit/initial.json')
                    if left['model_sha256']!=right['model_sha256'] or left['objective_sha256']!=right['objective_sha256']:
                        raise ValueError('Matched arms did not start identically')
                    write_new(ROOT/reg['run']/'generation_manifest.json',dict(runs=generated,expected_streams=792,
                        registration_sha256=sha(REG),recovery_registration_sha256=sha(RECOVERY),matched_initial_state_exact=True,
                        preserved_failed_initial_attempt_seconds=recovery['failed_generation_seconds']))
                env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
                    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                if not item['gpu']:env['CUDA_VISIBLE_DEVICES']=''
                began=time.monotonic()
                with (out/(stage+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-u',*item['arguments']],cwd=ROOT,env=env,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    status('running_serial_stage');code=child.wait()
                with (out/'stages.jsonl').open('a') as f:
                    f.write(json.dumps(dict(stage=stage,exit_code=code,elapsed_seconds=time.monotonic()-began))+'\n')
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
        status('failed_serial_recovery',error=repr(error));raise
    finally:
        p=out/'status.json'
        if p.exists():
            value=load(p);value.update(pid=None,child_pid=None,owner_exited=True);p.write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));args=parser.parse_args()
    if args.action=='register':register()
    else:run()
