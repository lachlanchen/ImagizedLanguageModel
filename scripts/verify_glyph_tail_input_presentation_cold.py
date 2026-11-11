#!/usr/bin/env python3
"""Replay the fixed proportional-presentation question image with the unchanged legacy actor."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.compare_glyph_tail_input_presentation import EVALUATION, validate, verify_case
from scripts.verify_glyph_tail_cold_inference import trace_audit


def load(path):
    return json.loads(path.read_text())


def main(run, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Hide CUDA from the CPU verifier; only the cold child uses GPU 0')
    request = validate(run)
    report = load(run/'generation/report.json')
    if not report['all_inputs_completed'] or len(report['records']) != 96:
        raise ValueError('Complete all registered inputs before the cold replay')
    row, = [r for r in request['inputs'] if r['id'] == 'qa_01_a_normal']
    expected_dir = run/'generation'/row['id']
    expected_row = verify_case(expected_dir, row)
    checkpoint = ROOT/request['checkpoint']
    prompt = (run/'evaluation/inputs'/row['image']).resolve()
    if sha(prompt) != row['sha256'] or sha(checkpoint) != request['checkpoint_sha256']:
        raise ValueError('Registered prompt or checkpoint changed')
    names = ['scripts/verify_glyph_tail_input_presentation_cold.py', 'scripts/infer_glyph_tail_language.py',
             'scripts/compare_glyph_tail_input_presentation.py', 'scripts/verify_glyph_tail_cold_inference.py',
             'references/glyph_tail_input_presentation_protocol.md']
    for name in names:
        if subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('Commit cold verification sources first: '+name)
    code = {name: sha(ROOT/name) for name in names}
    status = load(run/'generation/status.json')
    if status['pid'] is not None or status['phase'] != 'complete':
        raise ValueError('Wait for the generation owner to exit and release its GPU context')
    output.mkdir(parents=True, exist_ok=False)
    working = output/'empty_working_directory'; working.mkdir()
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', PYTHONPATH=str(ROOT),
        HF_HOME=str(output/'empty_hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = ['strace', '-ff', '-yy', '-s', '4096', '-e', 'trace=openat,openat2,connect',
        '-o', str(output/'access'), 'python', '-u', str(ROOT/'scripts/infer_glyph_tail_language.py'),
        '--checkpoint', str(checkpoint), '--image', str(prompt),
        '--output', str(output/'inference'), '--maximum-glyphs', '128']
    save_json(output/'invocation.json', dict(code_sha256=code, checkpoint_sha256=sha(checkpoint),
        case=row['id'], input_sha256=sha(prompt), expected_case_report=expected_row,
        generation_report_sha256=sha(run/'generation/report.json'),
        empty_working_directory=True, empty_model_cache=True, maximum_glyphs=128,
        unchanged_image_only_runtime=True, reference_text_opened=False))
    child = None
    try:
        with (output/'inference.log').open('x') as log:
            child = subprocess.Popen(command, cwd=working, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save_json(output/'status.json', dict(phase='running_actual_cold_child',
                pid=os.getpid(), child_pid=child.pid))
            child.wait(timeout=2000)
        if child.returncode or load(output/'inference/status.json')['phase'] != 'completed':
            raise ValueError('Inspect the failed cold child before verification')
        access = trace_audit(output, working, prompt)
        # The earlier audit predates this cohort. Explicitly ban its references,
        # previous generated answers and the diagnostic/training corpus roots.
        forbidden_roots = [EVALUATION.resolve(), ROOT/'artifacts/glyph_tail_compact_cohort_v1', run, ROOT/'references',
            ROOT/'artifacts/visual_sequence_corpus_v1', ROOT/'artifacts/visual_sequence_teacher_v1',
            ROOT/'artifacts/compact_visual_span_diagnosis_v2']
        opened = set()
        for trace in output.glob('access.*'):
            for line in trace.read_text(errors='replace').splitlines():
                match = re.search(r'openat2?\(([^,]+),\s*("(?:\\.|[^"\\])*")', line)
                if not match: continue
                name = ast.literal_eval(match[2]); p = Path(name)
                if not p.is_absolute():
                    directory = re.search(r'<([^>]+)>', match[1])
                    p = (Path(directory[1]) if directory else working)/p
                opened.add(p.resolve())
        forbidden = [str(p) for p in opened if p != prompt and any(p.is_relative_to(r) for r in forbidden_roots)]
        required = [prompt, checkpoint.resolve()]
        if forbidden or any(p not in opened for p in required):
            raise ValueError('Forbidden reference open or missing required input: '+str(forbidden))
        import torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(2)
        expected = torch.load(expected_dir/'legacy_output.pt', map_location='cpu', weights_only=True)
        actual = torch.load(output/'inference/output.pt', map_location='cpu', weights_only=True)
        if set(expected) != set(actual): raise ValueError('Cold output fields differ')
        for key in expected:
            same = torch.equal(expected[key],actual[key]) if torch.is_tensor(expected[key]) else expected[key] == actual[key]
            if not same: raise ValueError('Cold actual pixel/state/feedback field differs: '+key)
        # The unchanged CLI writes one long row. The cohort wraps rows. Compare
        # the exact raw signed tiles and the long-row display, not PNG file bytes.
        a = np.full((32,max(1,len(actual['images']))*32),255,np.uint8)
        for i,tile in enumerate(actual['images']):
            a[:,i*32:(i+1)*32]=((tile[0].numpy()+1)*127.5).astype(np.uint8)
        if not np.array_equal(a,np.asarray(Image.open(output/'inference/output.png'))):
            raise ValueError('CLI display differs from original emitted pixels')
        inference = load(output/'inference/report.json')
        if inference['input_sha256'] != row['sha256'] or inference['checkpoint_sha256'] != request['checkpoint_sha256']:
            raise ValueError('Cold runtime identity changed')
        if torch.cuda.is_initialized(): raise ValueError('CPU verifier initialized CUDA')
        save_json(output/'verification.json', dict(**access, code_sha256=code,
            fixed_case=row['id'], generation_report_sha256=sha(run/'generation/report.json'),
            expected_case_report_sha256=sha(expected_dir/'report.json'),
            inference_report_sha256=sha(output/'inference/report.json'),
            checkpoint_sha256=sha(checkpoint), actual_pixels_states_stopping_and_feedback_bit_exact=True,
            required_input_and_checkpoint_opens_verified=True, additional_reference_roots_checked=True,
            CLI_long_row_pixels_verified=True, PNG_file_layout_differs_intentionally=True,
            reference_text_opened=False, output_repaired=False, general_language_accuracy_proven=False,
            one_tested_input_only=True))
        save_json(output/'status.json', dict(phase='completed',pid=None,child_pid=None))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
        save_json(output/'status.json',dict(phase='failed',pid=None,child_pid=None,reason=repr(error)))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_tail_input_presentation_v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
