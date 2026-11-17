#!/usr/bin/env python3
"""Fresh strict cold replay after the admission-only source change."""
import argparse
import json
import os
import signal
import subprocess
import sys

from scripts.register_visual_context_comparison import ROOT,REG,load,sha
from scripts.verify_visual_context_deployment_sources import verify_sources

EXECUTION=ROOT/'references/evidence/visual_context_deployment_execution_registration_20261117.json'


def verify(arm):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide CUDA; only the traced child uses GPU0')
    from scripts.verify_full_rank_visual_span_cold import audit_access,compare_outputs
    equivalence=verify_sources();binding=load(EXECUTION)
    for name,digest in equivalence['source_sha256'].items():
        expected=binding['input_sha256'].get(name)
        if expected is None:
            original=load(REG)['input_sha256'];expected=original.get(name)
        if expected!=digest:raise ValueError('Deployment equivalence source is not bound: '+name)
    reg=load(REG);run=ROOT/reg['run']/arm;fit=load(run/'fit/report.json')
    checkpoint=(run/'fit'/fit['final_checkpoint']).resolve();generated=run/'generated_step_0008172'
    expected,=[r for r in load(generated/'report.json')['records'] if r['id']=='book_01_normal']
    request=load(generated/'request.json');prompt=(ROOT/reg['evaluation']/'inputs/book_01_normal.png').resolve()
    if sha(checkpoint)!=fit['final_checkpoint_sha256'] or sha(prompt)!=expected['input_sha256']:
        raise ValueError('Preserve the final checkpoint and first fixed book prompt')
    output=run/'cold_deployment_v1';output.mkdir(exist_ok=False);working=output/'empty';working.mkdir()
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=str(ROOT),HF_HOME=str(output/'empty_hf_cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    command=['strace','-ff','-yy','-s','4096','-e','trace=openat,openat2,connect','-o',str(output/'access'),
        sys.executable,'-u',str(ROOT/'scripts/infer_visual_context_deployment.py'),'--image',str(prompt),
        '--glyph-count',str(expected['input_glyphs']),'--checkpoint',str(checkpoint),'--output',str(output/'inference'),
        '--maximum-glyphs',str(request['maximum_glyphs']),'--flow-steps',str(request['flow_steps']),'--seed',str(request['seed'])]
    child=None
    try:
        with (output/'inference.log').open('x') as log:
            child=subprocess.Popen(command,cwd=working,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            if child.wait(timeout=1800):raise RuntimeError('Strict cold child failed; preserve its log')
    except BaseException:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
    access=audit_access(output,working,prompt,checkpoint)
    actual,=load(output/'inference/report.json')['records']
    compare_outputs(generated/'book_01_normal',expected,output/'inference/answer',actual)
    result=dict(arm=arm,checkpoint_sha256=sha(checkpoint),fixed_case='book_01_normal',
        execution_registration_sha256=sha(EXECUTION),source_equivalence=equivalence,
        actual_pixels_stopping_and_feedback_exact=True,**access,
        obsolete_metadata_exemption_added=False,original_treatment_failed_cold_preserved=True,
        reference_inputs=False,language_quality_proved=False,only_one_cold_case=True)
    with (output/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=('trainable','frozen'),required=True)
    verify(parser.parse_args().arm)
