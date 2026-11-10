#!/usr/bin/env python3
"""After the exact cache owner exits, run the authorized fit and qualification.

This CPU-only supervisor stops before the required actual image review. It
never retries failed stages, relaxes a protocol, or starts autonomous QA.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(path.read_text())
def require(value,message):
    if not value:raise RuntimeError(message)

def live(owner):
    try:
        fields=Path('/proc',str(owner['pid']),'stat').read_text().rsplit(')',1)[1].split()
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except (FileNotFoundError,ProcessLookupError):return False
    return owner['boot_id']==boot and owner['start_ticks']==fields[19]


def main(output):
    own='scripts/continue_genuine_glyph_pipeline.py'
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','supervisor itself must remain CPU-only')
    require(subprocess.check_output(['git','show','HEAD:'+own],cwd=ROOT)==(ROOT/own).read_bytes(),'commit supervisor first')
    receipt=output/'pipeline.json';require(not receipt.exists(),'preserve prior supervisors')
    original=load(output/'cache/status.json')
    require(original.get('pid') and live(original),'attach only to an authenticated live cache owner')
    prepared=load(output/'preparation.json');began=time.monotonic()
    deadline=max(0,prepared['cache_maximum_seconds']-original['elapsed_seconds'])+300
    state=dict(pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],cache_owner=original,
        supervisor_source_sha256=hashlib.sha256((ROOT/own).read_bytes()).hexdigest(),
        preparation_sha256=hashlib.sha256((output/'preparation.json').read_bytes()).hexdigest(),
        started_fit=False,started_qualification=False,autonomous_QA_started=False)
    def status(phase,**extra):
        state.update(phase=phase,elapsed_seconds=time.monotonic()-began,**extra)
        temporary=receipt.with_suffix('.partial');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(receipt)
    try:
        status('waiting_for_exact_cache_owner_exit')
        while live(original):
            require(time.monotonic()-began<deadline,'cache owner wait exceeded declared deadline')
            require(hashlib.sha256((output/'preparation.json').read_bytes()).hexdigest()==state['preparation_sha256'],'preparation changed')
            time.sleep(20)
        current=load(output/'cache/status.json')
        require(current['phase']=='completed' and current['pid'] is None,'cache owner exited without a complete cache')
        for stage in ['fit','qualify']:
            require(not (output/stage).exists(),'another owner already created stage '+stage)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(ROOT))
            command=[sys.executable,'-u',str(ROOT/'scripts/train_genuine_glyph_readout.py'),'--stage',stage,'--output',str(output)]
            log=output.parent/(output.name+'_'+stage+'.log')
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                status('running_'+stage,child_pid=child.pid,**{'started_fit' if stage=='fit' else 'started_qualification':True})
                code=child.wait()
            require(code==0 and load(output/stage/'status.json')['phase']=='completed','stage failed: '+stage)
            status('completed_'+stage,child_pid=None)
        result=load(output/'qualify/report.json')
        status('waiting_for_agent_visual_review',readout_gate_passed=result['readout_gate_passed'],
            no_user_permission_required_for_authorized_review=True,language_goal_complete=False)
    except BaseException as error:
        status('stopped',reason=repr(error));raise
    finally:status(state['phase'],pid=None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
