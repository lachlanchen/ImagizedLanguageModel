"""Two independent checkpoint deployments for the same typed and pictured question."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.infer_continuous_image_language import image_cells
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_continuous_image_language import ROOT

PROMPT = '請說明「言」字的演變。'
OUT = ROOT/'artifacts/native_yan_cold_inference_v1'
FIT = ROOT/'references/evidence/continuous_image_training_completed_20261118.json'
CLI = ROOT/'scripts/infer_continuous_image_language.py'


def inspect_trace(path, allowed, workspace):
    """Inspect actual syscall arguments; the child runs from the empty workspace."""
    accesses = set()
    external_connects = []
    traced = path.read_text().splitlines()
    for line in traced:
        if 'connect(' in line and ('AF_INET' in line or 'AF_INET6' in line):
            external_connects.append(line)
        match = re.search(r'openat\([^,]*, "([^"]+)"', line)
        if not match:
            continue
        entry = Path(match[1])
        if not entry.is_absolute():
            entry = workspace/entry
        accesses.add(entry.resolve())
    require(not external_connects, 'Network connection attempted')
    require(allowed['checkpoint'] in accesses and len(traced) > 0, 'Traced checkpoint open required')
    # The system libraries, project Python sources and optional input font are
    # available. Private experiment/corpus files are restricted to this case.
    unexpected = []
    for entry in accesses:
        if any(entry.is_relative_to(ROOT/k) for k in ('artifacts','data','publication','references')):
            if entry not in allowed.values() and not entry.is_relative_to(allowed['output']) and entry != workspace:
                unexpected.append(str(entry))
        elif entry.suffix.lower() in ('.pt','.gguf','.safetensors','.onnx','.pth') and entry != allowed['checkpoint']:
            # Python site .pth files configure imports; they are plain startup text.
            if entry.suffix != '.pth' or 'site-packages' not in entry.parts:
                unexpected.append(str(entry))
    require(not unexpected, 'Unexpected data/weight access: '+str(sorted(unexpected)[:5]))
    return dict(trace_sha256=sha(path), unique_open_paths=len(accesses), external_connect_attempts=0,
                checkpoint_opened=True, additional_dataset_or_model_files_opened=False)


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only deployment')
    record, = [a for a in load(FIT)['arms'] if a['components']==8]
    snap = record['snapshots'][-1]
    checkpoint = ROOT/snap['file']
    require(snap['update']==13785 and sha(checkpoint)==snap['sha256'], 'Final model identity')
    OUT.mkdir(exist_ok=False)
    raster = Rasterizer()
    packed = b''.join(raster.tile(c) for c in PROMPT)
    cells = np.unpackbits(np.frombuffer(packed,dtype=np.uint8)).reshape(-1,1,32,32)
    source = OUT/'prompt.png'
    Image.fromarray(255*(1-np.concatenate(cells[:,0],axis=1))).save(source)
    input_hash=hashlib.sha256(cells.tobytes()).hexdigest()
    routes={}
    env=dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='', MKL_NUM_THREADS='2',
             OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    for key, option in [('typed',['--text',PROMPT]),('pictured',['--image',str(source)])]:
        target=OUT/key
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            trace=OUT/(key+'.trace')
            start=time.monotonic()
            args=['strace','-f','-s','2048','-e','trace=openat,connect','-o',str(trace),
                  sys.executable,str(CLI),'--checkpoint',str(checkpoint),*option,
                  '--output',str(target),'--device','cpu','--maximum-new','128']
            with (OUT/(key+'.log')).open('x') as log:
                child=subprocess.run(['timeout','--kill-after=5','600s',*args],env=env,cwd=tmp,stdout=log,stderr=subprocess.STDOUT)
            require(child.returncode==0, f'{key} exited {child.returncode}')
            elapsed=time.monotonic()-start
            allowed=dict(checkpoint=checkpoint,output=target)
            if key=='pictured':
                allowed['prompt']=source
            access=inspect_trace(trace,allowed,Path(tmp))
        r=load(target/'receipt.json')
        require(r['prompt_ink_sha256']==input_hash and r['checkpoint_sha256']==snap['sha256']
                and r['typed_input_rendering']==(key=='typed') and r['actual_emitted_pixel_feedback'], 'Wrong route')
        with np.load(target/'actual_pixels.npz', allow_pickle=False) as archive:
            ink=archive['ink']
        n=r['glyphs']
        require(ink.shape==(n,1,32,32) and ink.dtype==np.uint8 and (ink<=1).all(),'Invalid pixels')
        page=image_cells(target/'answer.png')
        require(np.array_equal(page[:n],ink) and not page[n:].any() and r['capped']==(n==128)
                and r['stopped']==(n<128), 'Invalid answer/termination')
        routes[key]=dict(trace=access, elapsed_seconds=elapsed, receipt=r, output_png_sha256=sha(target/'answer.png'),
                         raw_ink_sha256=hashlib.sha256(ink.tobytes()).hexdigest(),
                         array_sha256=sha(target/'actual_pixels.npz'), ink_pixels=int(ink.sum()))
        write_new(OUT/(key+'_verified.json'),routes[key])
    a,b=routes.values()
    same=all(a[k]==b[k] for k in ('output_png_sha256','raw_ink_sha256','ink_pixels')) and all(
             a['receipt'][k]==b['receipt'][k] for k in ('stopped','capped','glyphs'))
    report=dict(prompt=PROMPT,prompt_sha256=sha(source),checkpoint_sha256=snap['sha256'],cli_sha256=sha(CLI),
                protocol_sha256=sha(ROOT/'references/native_yan_inference_protocol.md'),routes=routes,
                identical_pixels_and_stop=same,matched_quality_efficiency_measured=False)
    write_new(OUT/'report.json',report)
    print(json.dumps(dict(identical=same, glyphs=a['receipt']['glyphs'], stopped=a['receipt']['stopped'])))
    require(same,'Preserved divergent cold outputs')


if __name__=='__main__':
    main()
