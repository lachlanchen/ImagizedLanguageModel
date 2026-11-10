"""Serial CPU publication admission with a large reserve under occupied global swap."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from native_flow_manuscript import build_slot as original_build_slot
from glyph_step_data import ROOT, require
from scripts.score_pixel_language_first_blocks import host_resources


def audit_surface():
    # Inspect ownership without recording unrelated command lines or touching
    # another project's processes. Runtime inventory remains private.
    owned=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name)==os.getpid():continue
        try:
            cwd=(path/'cwd').resolve(strict=True)
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if cwd==ROOT or ROOT in cwd.parents:
            if any(word in command.split()[:2] for word in ['soffice','latexmk','pdflatex','xelatex','lualatex','blender']):
                owned.append(int(path.name))
    require(not owned,'preserve an already running project publication/GUI job')
    def probe(command):
        p=subprocess.run(command,capture_output=True,text=True,timeout=10)
        return dict(returncode=p.returncode,output=p.stdout)
    return dict(project_heavy_processes=owned,memory=probe(['free','-h']),
                sessions=probe(['tmux','list-sessions']),listeners=probe(['ss','-ltnp']))


def record(kind,values):
    path=ROOT/'artifacts/publication_resource_events.jsonl'
    with path.open('a') as stream:
        stream.write(json.dumps(dict(kind=kind,pid=os.getpid(),unix=time.time(),**values))+'\n')


@contextmanager
def build_slot():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU publication requires hidden CUDA')
    initial=host_resources()
    if initial['swap_fraction']<=.75:
        with original_build_slot() as admission:
            surface=audit_surface();record('entry',dict(mode='original',admission=admission,surface=surface))
            yield admission
        return
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observed=host_resources();surface=audit_surface()
        require(observed['available_gib']>=64.,'CPU publication needs64 GiB reserve under high global swap')
        require(observed['some_pressure_avg10']<=.5 and observed['full_pressure_avg10']<=.1,
                'CPU publication needs low actual memory pressure')
        admission=dict(mode='post_terminal_CPU_publication_64GiB_reserve_and_PSI',entry=observed,
            minimum_entry_available_gib=64.,minimum_live_available_gib=24.,same_project_GPU_lock_held=True,
            original_RAM_swap_helper_unchanged=True,other_project_processes_modified=False,CUDA_hidden=True)
        record('entry',dict(admission=admission,surface=surface));yield admission


def owned_command(command,cwd,log,timeout=180):
    """Bound and clean up only this exact command's new process group."""
    with Path(log).open('x') as stream:
        process=subprocess.Popen(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        began=time.monotonic();minimum=float('inf')
        try:
            while process.poll() is None:
                observed=host_resources();minimum=min(minimum,observed['available_gib'])
                require(minimum>=24.,'publication child reached24 GiB live RAM reserve')
                require(time.monotonic()-began<timeout,'bounded publication child timed out')
                time.sleep(1)
            require(process.returncode==0,'owned publication command failed: '+str(log))
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            record('owned_child_finished',dict(child_pid=process.pid,returncode=process.returncode,
                elapsed_seconds=time.monotonic()-began,minimum_available_gib=minimum,
                log=str(Path(log).resolve().relative_to(ROOT))))
