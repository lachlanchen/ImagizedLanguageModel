#!/usr/bin/env python3
"""Replay the fixed first reserved image with the completed natural writer.

This verifies the standalone execution boundary, not answer correctness. The
original 18,000-update verifier and its published receipts remain unchanged.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require
from scripts.verify_glyph_tail_cold_inference import trace_audit as original_trace_audit

OWN='scripts/verify_natural_first_cold_inference.py'
PROTOCOL='references/natural_first_cold_inference_protocol.md'


def load(path):return json.loads(Path(path).read_text())


def identity():
    names=[OWN,PROTOCOL,'scripts/verify_glyph_tail_cold_inference.py',
           'scripts/infer_natural_first_language.py','scripts/assess_natural_first_answers.py',
           'ilm/visual_lm/semantic_glyph_loop.py','ilm/visual_lm/glyph_tail_adapter.py',
           'ilm/visual_lm/glyph_runtime_resources.py','ilm/visual_lm/glyph_reader_monitor.py']
    for name in names:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'commit natural cold verification sources first')
    return {name:sha(ROOT/name) for name in names}


def trace_audit(folder,working,prompt,checkpoint,output):
    """Apply the old access check and a narrower project-artifact allowlist."""
    receipt=original_trace_audit(folder,working,prompt)
    from scripts.train_semantic_glyph_language import CONTROLS
    allowed={p.resolve() for p in [prompt,checkpoint,CONTROLS,ROOT/'artifacts/ilm_gpu0.lock',
        ROOT/'artifacts/pixel_language_diversity_training_v2/execution/status.json',
        ROOT/'artifacts/pixel_language_diversity_training_v2/execution/events.jsonl']}
    denied=[];observed=set()
    for path in sorted(folder.glob('access.*')):
        for line in path.read_text(errors='replace').splitlines():
            match=re.search(r'openat2?\(([^,]+),\s*("(?:\\.|[^"\\])*")',line)
            if not match:continue
            name=Path(ast.literal_eval(match[2]))
            if not name.is_absolute():
                directory=re.search(r'<([^>]+)>',match[1])
                name=(Path(directory[1]) if directory else working)/name
            name=name.resolve();observed.add(name)
            if name.is_relative_to(ROOT/'artifacts') and not (
                    name in allowed or name.is_relative_to(output) or name.is_dir()):
                denied.append(str(name))
    require(not denied,'cold inference accessed an undeclared project artifact: '+repr(sorted(set(denied))))
    require(prompt in observed and checkpoint.resolve() in observed,'trace lacks actual prompt/checkpoint file opens')
    return dict(**receipt,undeclared_project_artifacts=[],
        project_artifact_allowlist=[str(p.relative_to(ROOT)) for p in sorted(allowed) if p.is_relative_to(ROOT)],
        actual_prompt_and_checkpoint_file_opens_verified=True,
        owner_status_and_event_ledger_allowed_for_resource_admission_only=True)


def main(run,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','verifier stays on CPU; only its child uses GPU0')
    require((run/'preparation.json').is_file() and (run/'assessment/report.json').is_file(),
            'complete the bound natural writer and all 72 assessment streams first')
    from scripts.assess_natural_first_answers import candidate
    m,checkpoint=candidate(run)
    require(m['condition']=='natural_writer' and m['fit_steps']==36000,'fixed natural-writer condition required')
    require(load(run/'assessment/status.json')['phase']=='completed','finish all actual answers first')
    assessed=load(run/'assessment/report.json')
    require(len(assessed['records'])==72 and assessed['checkpoint_sha256']==sha(checkpoint),'assessment identity differs')
    row,=[r for r in assessed['records'] if r['key']=='fresh/0000/normal']
    require(row['prompt']==m['fresh_cases'][0]['prompt'],'fixed first reserved input changed')
    prompt=(ROOT/row['prompt']).resolve()
    require(sha(prompt)==row['input_sha256'],'assessed prompt changed')
    code=identity()
    for text in subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'],text=True).splitlines():
        if not text.strip().isdigit():continue
        try:cwd=(Path('/proc')/text.strip()/'cwd').resolve(strict=True)
        except FileNotFoundError:continue
        require(not (cwd==ROOT or ROOT in cwd.parents),'wait for the current project GPU owner to exit')
    output.mkdir(parents=True,exist_ok=False);working=output/'empty_working_directory';working.mkdir()
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=str(ROOT),HF_HOME=str(output/'empty_hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=['strace','-ff','-yy','-s','4096','-e','trace=openat,openat2,connect','-o',str(output/'access'),
        sys.executable,'-u',str(ROOT/'scripts/infer_natural_first_language.py'),'--checkpoint',str(checkpoint),
        '--image',str(prompt),'--output',str(output/'inference'),'--maximum-glyphs','64']
    save_json(output/'invocation.json',dict(code_sha256=code,checkpoint_sha256=sha(checkpoint),
        fixed_assessment_key=row['key'],expected_assessment_record=row,
        assessment_report_sha256=sha(run/'assessment/report.json'),references_opened=False,
        empty_working_directory=True,empty_model_cache=True))
    child=None
    try:
        with (output/'inference.log').open('x') as log:
            child=subprocess.Popen(command,cwd=working,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_json(output/'status.json',dict(phase='running_actual_cold_child',pid=os.getpid(),child_pid=child.pid))
            child.wait(timeout=2000)
        require(child.returncode==0 and load(output/'inference/status.json')['phase']=='completed',
                'inspect failed cold child before verification')
        access=trace_audit(output,working,prompt,checkpoint,output)
        import torch
        torch.set_num_threads(2)
        expected_path=run/'assessment'/row['file'];require(sha(expected_path)==row['sha256'],'assessment tensor changed')
        expected=torch.load(expected_path,weights_only=True,map_location='cpu')
        actual=torch.load(output/'inference/output.pt',weights_only=True,map_location='cpu')
        require(set(actual)==set(expected),'cold output fields differ')
        for key in expected:
            if torch.is_tensor(expected[key]):require(torch.equal(actual[key],expected[key]),'cold tensor differs: '+key)
            else:require(actual[key]==expected[key],'cold event field differs: '+key)
        require(sha(output/'inference/output.png')==row['png_sha256'],'cold PNG bytes differ')
        inference=load(output/'inference/report.json')
        require(inference['checkpoint_sha256']==sha(checkpoint) and inference['input_sha256']==row['input_sha256'],
                'cold model/input identity differs')
        require(inference['observed_prompt_sha256']==sha(output/'inference/observed_prompt.png'),'observed raster changed')
        require(not torch.cuda.is_initialized(),'CPU verifier initialized CUDA')
        save_json(output/'verification.json',dict(**access,code_sha256=code,checkpoint_sha256=sha(checkpoint),
            fixed_assessment_key=row['key'],actual_pixels_states_stopping_and_feedback_bit_exact=True,
            actual_PNG_bytes_exact=True,inference_report_sha256=sha(output/'inference/report.json'),
            references_opened=False,runtime_output_repaired=False,independent_human_grading=False,
            general_language_accuracy_proven=False,one_tested_cold_input_only=True))
        save_json(output/'status.json',dict(phase='completed',pid=None,child_pid=None))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save_json(output/'status.json',dict(phase='failed',pid=None,child_pid=None,reason=repr(error)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/natural_first_answer_assay_v1/natural_writer'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
