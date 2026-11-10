#!/usr/bin/env python3
"""Continue a completed natural cache into the fixed fit and qualification.

Waits without a model allocation; never overlaps two heavy project jobs and
never changes a failed recipe. Actual result/image review remains a later step.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources


def load(path):return json.loads(Path(path).read_text())


def main(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0','physical GPU0 worker environment required')
    folder=run/'pipeline';folder.mkdir(exist_ok=False);began=time.monotonic();active=None
    def status(phase,**extra):
        save_json(folder/'status.json',dict(phase=phase,pid=os.getpid(),
            elapsed_seconds=time.monotonic()-began,**extra))
    try:
        while True:
            require(time.monotonic()-began<14400,'pipeline waiting deadline')
            attempts=sorted((run/'cache/attempts').glob('*/status.json'))
            require(attempts,'start the registered real-cache job first')
            cache_status=load(attempts[-1])
            if cache_status['phase']=='failed':raise RuntimeError('cache failed; preserve attempt and inspect: '+cache_status.get('reason',''))
            if cache_status['phase']=='completed' and cache_status['pid'] is None:
                require((run/'cache/report.json').exists(),'cache completed without final report');break
            status('waiting_for_existing_cache',cache_attempt=str(attempts[-1].relative_to(run)),
                   cache_position=cache_status.get('position'))
            time.sleep(5)
        for stage in ['prepare','fit','qualify']:
            admitted=False;stable=0;waiting=time.monotonic()
            while not admitted:
                require(time.monotonic()-waiting<1800,'pipeline pre-admission deadline')
                r=host_resources();stable=stable+1 if (r['available_gib']>=48 and r['some_pressure_avg10']<=.5 and r['full_pressure_avg10']<=.1) else 0
                admitted=stable>=3
                status('waiting_for_stage_admission',stage=stage,stable_samples=stable,resources=r)
                if not admitted:time.sleep(5)
            status('running_stage',stage=stage)
            env=dict(os.environ)
            if stage=='prepare':env['CUDA_VISIBLE_DEVICES']=''
            command=[sys.executable,str(ROOT/'scripts/train_natural_first_writer.py'),'--stage',stage,'--run',str(run)]
            with (folder/(stage+'.log')).open('x') as stream:
                active=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                while active.poll() is None:
                    # Child owns the precise memory/time limits and mutex.
                    status('running_stage',stage=stage,child_pid=active.pid)
                    time.sleep(2)
                require(active.returncode==0,'natural writer stage failed: '+stage)
                active=None
        report=load(run/'qualify/report.json')
        status('completed_awaiting_actual_image_review',function_gate_passed=report['natural_first_gate_passed'],
            full_language_assessed=False,deployed_model_changed=False)
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.terminate()
            try:active.wait(timeout=15)
            except subprocess.TimeoutExpired:active.kill();active.wait()
        status('failed',reason=repr(error));raise
    finally:save_json(folder/'status.json',dict(load(folder/'status.json'),pid=None))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=Path('artifacts/natural_first_image_writer_v1'))
    main(p.parse_args().run.resolve())
