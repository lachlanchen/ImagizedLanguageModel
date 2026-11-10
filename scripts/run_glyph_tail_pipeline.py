#!/usr/bin/env python3
"""Run one cache/fit/qualification sequence, then await actual image review."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import host_resources, require, owner_live


def load(path):
    return json.loads(Path(path).read_text())


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def main(run, after_completed_cache=False):
    own='scripts/run_glyph_tail_pipeline.py'
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),
            'commit supervisor first')
    previous=None
    if after_completed_cache:
        previous=load(run/'pipeline.json')
        for prefix in ['supervisor','child']:
            require(not owner_live(dict(pid=previous.get(prefix+'_pid'),
                start_ticks=previous.get(prefix+'_start_ticks'),boot_id=previous.get('boot_id'))),
                'previous coordinator or child is still active')
        require(load(run/'cache/status.json')['phase']=='completed' and not (run/'fit').exists(),
                'resume only after the complete cache and before main fitting')
        number=0
        while (run/f'pipeline_attempt_{number:02}.json').exists(): number+=1
        save_json(run/f'pipeline_attempt_{number:02}.json',previous)
    else:
        require(not (run/'pipeline.json').exists(),'preserve previous coordinator state')
    meta=load(run/'preparation.json')
    for stage in ['preflight_cache','preflight_fit']:
        require(load(run/stage/'status.json')['phase']=='completed','finish preflight first')
    require(load(run/'preflight_fit/report.json')['all_checks_passed'],'actual update checks failed')
    require(load(run/'input_review.json')['all32_fresh_input_images_actually_viewed'],'review inputs first')
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',
             CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT))
    began=time.time(); child=None
    state=dict(supervisor_pid=os.getpid(),supervisor_start_ticks=ticks(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        source_sha256=sha(ROOT/own),preparation_sha256=sha(run/'preparation.json'),
        continued_after_completed_cache=after_completed_cache,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        stages=[],child_pid=None,child_start_ticks=None,entry_wait_seconds=0.)

    def save(phase,**extra):
        state.update(phase=phase,recorded_unix=time.time(),elapsed_seconds=time.time()-began,**extra)
        save_json(run/'pipeline.json',state)

    def interrupt(signum,frame):
        if child is not None and child.poll() is None:
            child.terminate()
            try: child.wait(timeout=30)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        save('interrupted',signal=signum,child_pid=None,child_start_ticks=None)
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
    save('preparing_serial_stages')
    try:
        stages=[('fit','scripts/train_glyph_tail_adapter.py'),('qualify','scripts/assess_glyph_tail_adapter.py')]
        if not after_completed_cache: stages.insert(0,('cache','scripts/train_glyph_tail_adapter.py'))
        for stage,script in stages:
            require(not (run/stage).exists(),'preserve prior or partial stage: '+stage)
            waited=0.
            while True:
                observed=host_resources()
                minimum=48. if stage=='fit' else (64. if observed['swap_fraction']>.75 else 24.)
                ready=(observed['available_gib']>=minimum and observed['some_pressure_avg10']<=.5
                       and observed['full_pressure_avg10']<=.1)
                if ready: break
                require(waited<300.,'entry reserve did not recover; no child launched')
                save('waiting_for_stage_entry',current_stage=stage,resources=observed)
                start=time.monotonic(); time.sleep(5); elapsed=time.monotonic()-start
                waited+=elapsed; state['entry_wait_seconds']+=elapsed
            flag='--run' if stage=='qualify' else '--output'
            command=['python','-u',script,'--stage',stage,flag,str(run)]
            log=run/(stage+'_process.log')
            with log.open('w') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                save('running_'+stage,current_stage=stage,child_pid=child.pid,child_start_ticks=ticks(child.pid),
                     current_log=str(log.relative_to(ROOT)))
                while child.poll() is None:
                    status=run/stage/'status.json'
                    if status.exists():
                        try: state['child_status']=load(status)
                        except json.JSONDecodeError: pass
                    save('running_'+stage)
                    time.sleep(5)
                code=child.returncode
            status=load(run/stage/'status.json')
            state['stages'].append(dict(stage=stage,returncode=code,status=status,
                report_sha256=sha(run/stage/'report.json') if (run/stage/'report.json').exists() else None))
            save('finished_'+stage,child_pid=None,child_start_ticks=None)
            # A nonzero process exit is retained for an agent audit, even if a
            # complete receipt was written before interpreter teardown.
            require(code==0 and status['phase']=='completed','inspect failed or nonzero child exit before continuing')
        gate=load(run/'qualify/report.json')['readout_gate_passed']
        save('waiting_for_agent_visual_review',physical_gate_passed=gate,autonomous_answers_generated=False,
             usable_language_demonstrated=False,supervisor_pid=None)
    except BaseException as error:
        if isinstance(error,SystemExit): raise
        save('waiting_for_agent_audit',reason=repr(error),child_pid=None,child_start_ticks=None,supervisor_pid=None)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('artifacts/glyph_tail_adaptation_v2'))
    parser.add_argument('--after-completed-cache',action='store_true')
    args=parser.parse_args(); main(args.run.resolve(),args.after_completed_cache)
