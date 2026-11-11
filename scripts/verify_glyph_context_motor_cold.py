#!/usr/bin/env python3
"""Replay the prospectively fixed book input through standalone motor inference."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.score_pixel_language_first_blocks import require

CASE = 'fixed_cells__book_01_normal'
SOURCES = (
    'scripts/verify_glyph_context_motor_cold.py',
    'scripts/infer_glyph_context_motor_image.py',
    'references/glyph_context_motor_cold_decision.md',
    'scripts/verify_glyph_tail_cold_inference.py',
    'ilm/visual_lm/grounded_image_weights.py',
    'ilm/visual_lm/gguf_visual_language.py',
)


def load(path):
    return json.loads(Path(path).read_text())


def equal_tree(expected, actual, path='output'):
    """Compare every nested tensor/event, including the unused final cap sample."""
    import torch
    require(type(expected) is type(actual), 'Cold value type differs: '+path)
    if torch.is_tensor(expected):
        require(expected.dtype == actual.dtype and expected.shape == actual.shape
                and torch.equal(expected, actual), 'Cold tensor differs: '+path)
    elif isinstance(expected, dict):
        require(expected.keys() == actual.keys(), 'Cold fields differ: '+path)
        for key in expected:
            equal_tree(expected[key], actual[key], path+'.'+str(key))
    elif isinstance(expected, (list, tuple)):
        require(len(expected) == len(actual), 'Cold list length differs: '+path)
        for i, (left, right) in enumerate(zip(expected, actual)):
            equal_tree(left, right, path+f'[{i}]')
    else:
        require(expected == actual, 'Cold scalar/event differs: '+path)


def opened_paths(output, working):
    opened = set()
    for trace in output.glob('access.*'):
        for line in trace.read_text(errors='replace').splitlines():
            match = re.search(r'openat2?\(([^,]+),\s*("(?:\\.|[^"\\])*")', line)
            if not match:
                continue
            name = ast.literal_eval(match[2]); path = Path(name)
            if not path.is_absolute():
                directory = re.search(r'<([^>]+)>', match[1])
                path = (Path(directory[1]) if directory else working)/path
            opened.add(path.resolve())
    return opened


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU verifier must hide CUDA')
    from scripts.infer_glyph_context_motor import study, verified_case
    from scripts.train_glyph_context_motor import CHECKPOINT, CHECKPOINT_SHA
    from scripts.train_semantic_glyph_language import CONTROLS
    from scripts.verify_glyph_tail_cold_inference import trace_audit
    study(run)
    reports = {}
    for layout, count in [('fixed_cells', 120), ('proportional', 96)]:
        folder = run/('generation_'+layout)
        report = load(folder/'report.json'); status = load(folder/'status.json')
        require(report['all_inputs_completed'] and len(report['records']) == count
                and status['phase'] == 'complete' and status['pid'] is None,
                'Finish both generation owners before the cold replay')
        require(report['request_sha256'] == sha(folder/'request.json'), 'Generation request changed')
        reports[layout] = report
    pids = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                  '--format=csv,noheader,nounits'], text=True).splitlines()
    for pid in pids:
        if not pid.strip().isdigit():
            continue
        try:
            cwd = (Path('/proc')/pid.strip()/'cwd').resolve(strict=True)
        except FileNotFoundError:
            continue
        require(cwd != ROOT and ROOT not in cwd.parents, 'Project GPU owner has not exited')
    manifest = load(run/'evaluation_manifest.json')
    row, = [r for r in manifest['inputs'] if r['id'] == CASE]
    expected_dir = run/'generation_fixed_cells'/CASE
    expected_row = verified_case(expected_dir, row)
    require(expected_row in reports['fixed_cells']['records'], 'Fixed cold case absent')
    prompt = (ROOT/row['image']).resolve(); motor = run/'fit/checkpoint_latest.pt'
    fit = load(run/'fit/report.json'); motor_sha = sha(motor)
    require(sha(prompt) == row['sha256'] and sha(CHECKPOINT) == CHECKPOINT_SHA
            and motor_sha == fit['checkpoint_sha256']
            and all(r['motor_checkpoint_sha256'] == motor_sha for r in reports.values()),
            'Input or model identity changed')
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) == (ROOT/name).read_bytes(),
                'Commit replay sources first: '+name)
    code = {name:sha(ROOT/name) for name in SOURCES}
    output.mkdir(parents=True, exist_ok=False)
    working = output/'empty_working_directory'; working.mkdir()
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', PYTHONPATH=str(ROOT),
        HF_HOME=str(output/'empty_hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = ['strace', '-ff', '-yy', '-s', '4096', '-e', 'trace=openat,openat2,connect',
        '-o', str(output/'access'), sys.executable, '-u', str(ROOT/'scripts/infer_glyph_context_motor_image.py'),
        '--parent-checkpoint', str(CHECKPOINT), '--motor-checkpoint', str(motor),
        '--motor-sha256', motor_sha, '--image', str(prompt), '--seed', str(row['seed']),
        '--maximum-glyphs', '128', '--output', str(output/'inference')]
    save_json(output/'invocation.json', dict(code_sha256=code, fixed_case=CASE, seed=row['seed'],
        input_sha256=row['sha256'], parent_checkpoint_sha256=CHECKPOINT_SHA, motor_checkpoint_sha256=motor_sha,
        expected_case_report_sha256=sha(expected_dir/'report.json'),
        generation_report_sha256={k:sha(run/('generation_'+k)/'report.json') for k in reports},
        maximum_glyphs=128, empty_working_directory=True, empty_model_cache=True,
        image_only_standalone_child=True, references_opened_by_child=False))
    child = None
    try:
        with (output/'inference.log').open('x') as log:
            child = subprocess.Popen(command, cwd=working, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            save_json(output/'status.json', dict(phase='running_actual_cold_child',
                pid=os.getpid(), child_pid=child.pid))
            child.wait(timeout=2000)
        require(child.returncode == 0 and load(output/'inference/status.json')['phase'] == 'complete',
                'Inspect the retained failed replay before retrying')
        access = trace_audit(output, working, prompt)
        opened = opened_paths(output, working)
        allowed = {prompt, CHECKPOINT.resolve(), motor.resolve(), CONTROLS.resolve(), ROOT/'artifacts/ilm_gpu0.lock'}
        forbidden = [str(p) for p in sorted(opened) if p not in allowed
            and not p.is_relative_to(output)
            and (p.is_relative_to(ROOT/'artifacts') or p.is_relative_to(ROOT/'references'))]
        require(not forbidden, 'Cold child opened training/reference artifacts: '+str(forbidden))
        require({prompt, CHECKPOINT.resolve(), motor.resolve(), CONTROLS.resolve()} <= opened,
                'Required standalone input/weight/control opens absent')
        import torch
        import numpy as np
        from PIL import Image
        torch.set_num_threads(2)
        expected = torch.load(expected_dir/'legacy_output.pt', map_location='cpu', weights_only=True)
        actual = torch.load(output/'inference/output.pt', map_location='cpu', weights_only=True)
        equal_tree(expected, actual)
        with Image.open(expected_dir/'answer.png') as left, Image.open(output/'inference/output.png') as right:
            require(left.size == right.size and np.array_equal(np.asarray(left), np.asarray(right)),
                    'Cold displayed pixels differ')
        inference = load(output/'inference/report.json')
        require(inference['input_sha256'] == row['sha256'] and inference['seed'] == row['seed']
                and inference['parent_checkpoint_sha256'] == CHECKPOINT_SHA
                and inference['motor_checkpoint_sha256'] == motor_sha
                and inference['source']['vocabulary_parameters_materialized'] == 0
                and inference['all_frozen_parameter_versions_unchanged'], 'Cold runtime identity changed')
        require(not torch.cuda.is_initialized(), 'CPU verifier initialized CUDA')
        save_json(output/'verification.json', dict(**access, code_sha256=code, fixed_case=CASE,
            invocation_sha256=sha(output/'invocation.json'),
            inference_report_sha256=sha(output/'inference/report.json'),
            all_nested_pixels_states_stopping_feedback_noise_and_motor_values_bit_exact=True,
            displayed_pixels_exact=True, original_case_stopped=actual['stopped'],
            original_case_capped=actual['capped'], required_input_weight_control_opens_verified=True,
            all_other_project_training_reference_artifact_opens_rejected=True,
            source_GGUF_lexical_payload_not_materialized_by_bound_loader=True,
            file_trace_alone_does_not_prove_tensor_level_absence=True,
            output_repaired=False, one_tested_input_only=True, language_quality_proven=False,
            goal_achieved=False))
        save_json(output/'status.json', dict(phase='complete', pid=None, child_pid=None))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        save_json(output/'status.json', dict(phase='failed', pid=None, child_pid=None, error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_context_motor_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); main(args.run.resolve(), args.output.resolve())
