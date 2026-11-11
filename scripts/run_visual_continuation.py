#!/usr/bin/env python3
"""One serial controller: full fit, fixed snapshots, matched parent books, review pages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ilm.visual_lm.continuation_training import RUN,load
from ilm.image_edit.baseline import sha
from scripts.score_pixel_language_first_blocks import require,owner_live
from scripts.train_visual_continuation import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=RUN);a=p.parse_args();run=a.run
    status=run/'controller_status.json'
    if status.exists():require(not owner_live(load(status)),'Original controller is live')
    base=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),began_unix=time.time())
    atomic_json(status,dict(base,phase='starting',active_child_pid=None))
    try:
        for action,directory in [('train','fit'),('evaluate','evaluation'),('parent-long-books','parent_long_books')]:
            report=run/directory/'report.json'
            if report.exists() and load(report)['complete']:continue
            command=[sys.executable,'-u','scripts/train_visual_continuation.py',action,'--run',str(run)]
            if action=='train' and (run/'fit/checkpoint.json').exists():command.append('--resume')
            with (run/(action+'_console.log')).open('a') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=Path.cwd())
                atomic_json(status,dict(base,phase=action,active_child_pid=child.pid))
                result=child.wait()
            require(result==0 and report.exists() and load(report)['complete'],'Child incomplete; preserve evidence before any retry')
        if not (run/'review/mapping.json').exists():
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            subprocess.run([sys.executable,'scripts/review_visual_continuation.py','prepare','--run',str(run)],env=env,check=True)
        atomic_json(status,dict(base,phase='ready_for_complete_output_review',active_child_pid=None,
            review_mapping_sha256=sha(run/'review/mapping.json'),complete_training_and_generation=True,language_scoring_complete=False))
    except BaseException as error:
        atomic_json(status,dict(base,phase='failed',active_child_pid=None,error=repr(error)));raise
    finally:
        value=load(status);value['pid']=None;atomic_json(status,value)


if __name__=='__main__':main()
