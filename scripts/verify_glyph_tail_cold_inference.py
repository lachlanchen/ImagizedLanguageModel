#!/usr/bin/env python3
"""Cold image-only replay on the first prospectively reserved input, never a selected success."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess

from ilm.image_edit.baseline import ROOT, sha, save_json
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/verify_glyph_tail_cold_inference.py'
PROTOCOL = 'references/glyph_tail_cold_inference_protocol.md'


def load(path):
    return json.loads(Path(path).read_text())


def identity():
    names = [OWN, PROTOCOL, 'scripts/infer_glyph_tail_language.py',
             'ilm/visual_lm/glyph_tail_adapter.py', 'ilm/visual_lm/semantic_glyph_loop.py',
             'ilm/visual_lm/glyph_reader_monitor.py', 'ilm/visual_lm/glyph_runtime_resources.py']
    for name in names:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT) == (ROOT/name).read_bytes(),
                'commit cold verification sources first')
    return {name: sha(ROOT/name) for name in names}


def trace_audit(folder, working, allowed_input):
    opened, network, traces = set(), [], {}
    for path in sorted(folder.glob('access.*')):
        traces[path.name] = sha(path)
        for line in path.read_text(errors='replace').splitlines():
            if 'connect(' in line and re.search(r'sa_family=AF_INET6?[,}]', line): network.append(line)
            match = re.search(r'openat2?\(([^,]+),\s*("(?:\\.|[^"\\])*")', line)
            if not match: continue
            try: name = ast.literal_eval(match[2])
            except (ValueError, SyntaxError): raise RuntimeError('unparsed trace filename')
            p = Path(name)
            if not p.is_absolute():
                directory = re.search(r'<([^>]+)>', match[1])
                p = (Path(directory[1]) if directory else working) / p
            opened.add(str(p.resolve()))
    require(traces, 'no system-call trace captured')
    forbidden = []
    artifact_roots = ['semantic_glyph_bridge_v1', 'genuine_glyph_readout_v1',
                      'direct_pixel_readout_v1', 'semantic_glyph_language_v1', 'pixel_language_diversity_v1']
    for name in sorted(opened):
        p = Path(name)
        if p == allowed_input: continue
        bank = p.name in ['inventory.json', 'evaluation_references.json', 'tokenizer.json', 'vocab.json', 'merges.txt']
        font = p.suffix.lower() in ['.ttf', '.ttc', '.otf', '.woff', '.woff2']
        corpus = p.is_relative_to(ROOT/'data') or any(p.is_relative_to(ROOT/'artifacts'/r) for r in artifact_roots)
        if bank or font or corpus: forbidden.append(name)
    require(not network and not forbidden, 'cold inference accessed forbidden input or attempted an IP connection')
    return dict(opened_path_count=len(opened), forbidden_paths=forbidden, IP_connection_attempts=network,
                trace_files_sha256=traces, exact_input_file_exempted=True,
                source_GGUF_metadata_may_contain_unused_vocabulary=True)


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'run verifier with CUDA hidden; only its child uses GPU0')
    from scripts.assess_glyph_tail_adapter import candidate
    m, checkpoint = candidate(run)
    require(load(run/'assessment/status.json')['phase'] == 'completed', 'complete all 72 actual answers first')
    assessed = load(run/'assessment/report.json')
    require(len(assessed['records']) == 72 and assessed['checkpoint_sha256'] == sha(checkpoint), 'assessment identity differs')
    row, = [r for r in assessed['records'] if r['key'] == 'fresh/0000/normal']
    require(row['prompt'] == m['fresh_cases'][0]['prompt'], 'fixed first reserved input changed')
    prompt = (ROOT/row['prompt']).resolve()
    require(sha(prompt) == row['input_sha256'], 'actual assessed input changed')
    code = identity()
    pids = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    for text in pids:
        if not text.strip().isdigit(): continue
        try: cwd = (Path('/proc')/text.strip()/'cwd').resolve(strict=True)
        except FileNotFoundError: continue
        require(not (cwd == ROOT or ROOT in cwd.parents), 'wait for actual project GPU process exit')
    output.mkdir(parents=True, exist_ok=False); working = output/'empty_working_directory'; working.mkdir()
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
        PYTHONPATH=str(ROOT), HF_HOME=str(output/'empty_hf_cache'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command = ['strace', '-ff', '-yy', '-s', '4096', '-e', 'trace=openat,openat2,connect', '-o', str(output/'access'),
        'python', '-u', str(ROOT/'scripts/infer_glyph_tail_language.py'), '--checkpoint', str(checkpoint),
        '--image', str(prompt), '--output', str(output/'inference'), '--maximum-glyphs', '64']
    save_json(output/'invocation.json', dict(code_sha256=code, checkpoint_sha256=sha(checkpoint),
        fixed_assessment_key=row['key'], expected_assessment_record=row,
        assessment_report_sha256=sha(run/'assessment/report.json'), references_opened=False,
        empty_model_cache=True, empty_working_directory=True))
    child = None
    try:
        with (output/'inference.log').open('x') as log:
            child = subprocess.Popen(command, cwd=working, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            save_json(output/'status.json', dict(phase='running_actual_cold_child', pid=os.getpid(), child_pid=child.pid))
            child.wait(timeout=2000)
        require(child.returncode == 0 and load(output/'inference/status.json')['phase'] == 'completed',
                'inspect failed cold child before verification')
        access = trace_audit(output, working, prompt)
        import torch
        torch.set_num_threads(2)
        expected_path = run/'assessment'/row['file']
        require(sha(expected_path) == row['sha256'], 'actual assessment tensor changed')
        expected = torch.load(expected_path, weights_only=True, map_location='cpu')
        actual = torch.load(output/'inference/output.pt', weights_only=True, map_location='cpu')
        require(set(actual) == set(expected), 'cold output fields differ')
        for key in expected:
            if torch.is_tensor(expected[key]): require(torch.equal(actual[key], expected[key]), 'cold tensor differs: '+key)
            else: require(actual[key] == expected[key], 'cold feedback/event field differs: '+key)
        require(sha(output/'inference/output.png') == row['png_sha256'], 'cold PNG differs')
        inference = load(output/'inference/report.json')
        require(inference['checkpoint_sha256'] == sha(checkpoint) and inference['input_sha256'] == row['input_sha256'],
                'cold runtime identity differs')
        require(not torch.cuda.is_initialized(), 'CPU verifier unexpectedly initialized CUDA')
        save_json(output/'verification.json', dict(**access, code_sha256=code,
            checkpoint_sha256=sha(checkpoint), fixed_assessment_key=row['key'],
            actual_pixels_states_stopping_and_feedback_bit_exact=True, actual_PNG_bytes_exact=True,
            inference_report_sha256=sha(output/'inference/report.json'),
            references_opened=False, runtime_output_repaired=False, independent_human_grading=False,
            general_language_accuracy_proven=False, one_tested_cold_input_only=True))
        save_json(output/'status.json', dict(phase='completed', pid=None, child_pid=None))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        save_json(output/'status.json', dict(phase='failed', pid=None, child_pid=None, reason=repr(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('artifacts/glyph_tail_adaptation_v2'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.run.resolve(), a.output.resolve())
