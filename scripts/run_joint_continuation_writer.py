#!/usr/bin/env python3
"""One durable serial owner; a failed stage never retries itself."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ilm.visual_lm.joint_writer_training import ROOT,RUN,load,require,sha
from scripts.score_pixel_language_first_blocks import owner_live
from scripts.train_visual_continuation import atomic_json
from scripts.prepare_joint_continuation_writer import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');a=p.parse_args()
    validate();status=RUN/'controller_status.json'
    if status.exists():
        require(not owner_live(load(status)),'Original controller remains live')
    identity=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),began_unix=time.time())
    atomic_json(status,dict(identity,phase='starting',child_pid=None))
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(ROOT))
    try:
        for action,folder in [('cache','cache'),('train','fit'),('audit','audit'),('qualify','qualify')]:
            report=RUN/folder/'report.json'
            if report.exists() and load(report)['complete']:
                continue
            child_status=RUN/folder/'status.json'
            if child_status.exists():
                require(not owner_live(load(child_status)),'Prior stage worker remains live')
                require(a.resume,'Preserve failed stage; inspect before explicit manual resume')
            command=([sys.executable,'-u','scripts/audit_joint_continuation_writer.py'] if action=='audit'
                     else [sys.executable,'-u','scripts/train_joint_continuation_writer.py',action])
            if a.resume and child_status.exists() and action!='audit':
                command.append('--resume')
            with (RUN/(action+'_console.log')).open('a') as log, (RUN/'controller_journal.jsonl').open('a',buffering=1) as journal:
                child_environment=dict(environment,CUDA_VISIBLE_DEVICES='') if action=='audit' else environment
                began=time.perf_counter();child=subprocess.Popen(command,cwd=ROOT,env=child_environment,stdout=log,stderr=subprocess.STDOUT)
                atomic_json(status,dict(identity,phase=action,child_pid=child.pid))
                journal.write(json.dumps(dict(event='started',action=action,child_pid=child.pid,unix=time.time()))+'\n')
                journal.flush();os.fsync(journal.fileno());result=child.wait()
                journal.write(json.dumps(dict(event='ended',action=action,child_pid=child.pid,unix=time.time(),
                    elapsed_seconds=time.perf_counter()-began,returncode=result,
                    report_sha256=sha(report) if report.exists() else None))+'\n');journal.flush();os.fsync(journal.fileno())
            require(result==0 and report.exists() and load(report)['complete'],'Stage incomplete; preserve before any manual recovery')
        atomic_json(status,dict(identity,phase='awaiting_opaque_physical_review',child_pid=None,
            full_corpus_fit_complete=True,physical_gate_unmeasured=True,language_quality_unmeasured=True))
    except BaseException as e:
        atomic_json(status,dict(identity,phase='failed',child_pid=None,error=repr(e)));raise
    finally:
        value=load(status);value['pid']=None;atomic_json(status,value)


if __name__=='__main__':main()
