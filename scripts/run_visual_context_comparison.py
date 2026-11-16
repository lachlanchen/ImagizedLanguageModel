#!/usr/bin/env python3
"""One serial, non-retrying owner for both full fits and all prescribed streams."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.register_visual_context_comparison import ROOT,REG,load,sha,validate,require_pushed


def main(registration):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('CUDA_DEVICE_ORDER')!='PCI_BUS_ID' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Use physical GPU0 with the registered workspace')
    reg=validate(registration);require_pushed(registration)
    run=ROOT/reg['run'];run.mkdir(parents=True,exist_ok=False)
    child=None;stage='starting'
    def status(phase,**extra):
        value=dict(phase=phase,pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),stage=stage,
            child_pid=child.pid if child and child.poll() is None else None,automatic_retry=False,**extra)
        temporary=run/'status.tmp';temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(run/'status.json')
    def stop(_signal,_frame):raise KeyboardInterrupt('Comparison termination requested')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(name,arguments,gpu=True):
        nonlocal child,stage
        stage=name;status('launching_serial_stage')
        env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1',
                 HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        with (run/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',*map(str,arguments)],cwd=ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status('running_serial_stage')
            code=child.wait()
        if code:raise RuntimeError(f'Stage {name} exited {code}; preserve its log and partial files')
        child=None;status('serial_stage_complete')
    try:
        with (run/'owner.lock').open('x') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (run/'registration_sha256.txt').write_text(sha(registration)+'\n')
            generated=[]
            for arm in ('trainable','frozen'):
                execute(arm+'_fit',[ROOT/'scripts/train_visual_context_comparison.py','--registration',registration,'--arm',arm])
                execute(arm+'_audit',[ROOT/'scripts/register_visual_context_comparison.py','--audit',arm],gpu=False)
                fit=run/arm/'fit';report=load(fit/'report.json')
                if report['updates']!=reg['updates_per_arm']:raise ValueError('Full fit did not complete')
                snapshots=[fit/f'model_step_{n:07d}.pt' for n in reg['snapshot_updates']]
                if sorted(fit.glob('model_step_*.pt'))!=snapshots:raise ValueError('Preserve exactly five fixed snapshots')
                for index,checkpoint in enumerate(snapshots):
                    profile='all' if index in (0,4) else 'normal'
                    output=run/arm/f'generated_step_{reg["snapshot_updates"][index]:07d}'
                    command=[ROOT/'scripts/infer_trainable_visual_context.py','--manifest',
                        ROOT/reg['evaluation']/'inputs/manifest.json','--checkpoint',checkpoint,'--output',output,'--profile',profile]
                    execute(arm+'_generate_'+str(index),command);generated.append(str(output.relative_to(ROOT)))
                output=run/arm/'generated_final_long_books'
                execute(arm+'_long_books',[ROOT/'scripts/infer_trainable_visual_context.py','--manifest',
                    ROOT/reg['evaluation']/'inputs/manifest.json','--checkpoint',snapshots[-1],'--output',output,
                    '--profile','long_books','--maximum-glyphs','640'])
                generated.append(str(output.relative_to(ROOT)))
                execute(arm+'_cold_replay',[ROOT/'scripts/verify_trainable_visual_context_cold.py','--arm',arm],gpu=False)
            left=load(run/'trainable/fit/initial.json');right=load(run/'frozen/fit/initial.json')
            if left['model_sha256']!=right['model_sha256'] or left['objective_sha256']!=right['objective_sha256']:
                raise ValueError('Matched complete models did not start identically')
            (run/'generation_manifest.json').write_text(json.dumps(dict(runs=generated,expected_streams=792,
                registration_sha256=sha(registration),matched_initial_state_exact=True),indent=2)+'\n')
            execute('prepare_review',[ROOT/'scripts/review_visual_context_comparison.py','prepare'],gpu=False)
            status('awaiting_frozen_visual_readings',streams=792,language_scored=False)
    except BaseException as error:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status('failed_serial_comparison',error=repr(error));raise
    finally:
        p=run/'status.json'
        if p.exists():
            s=load(p);s.update(pid=None,child_pid=None,owner_exited=True);p.write_text(json.dumps(s,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,default=REG)
    main(p.parse_args().registration.resolve())
