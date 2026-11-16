#!/usr/bin/env python3
"""Inspect retained reader logs and pinned source without model execution."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'artifacts/reader_preprocessing_audit_v1'
OUT = ROOT / 'references/evidence/isolated_reader_preprocessing_audit_20261116.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    log_path = ROOT / 'artifacts/isolated_glyph_teacher_gpu0_v2/transcribe_server.log'
    text = log_path.read_text()
    token_counts = list(map(int, re.findall(r'decoding image batch 1/1, n_tokens_batch = (\d+)', text)))
    prompt_counts = list(map(int, re.findall(r'init sampler,.*tokens: text = (\d+), total = 179', text)))
    if len(token_counts) != 402 or set(token_counts) != {81} or len(prompt_counts) != 402 or set(prompt_counts) != {98}:
        raise ValueError('Retained complete request-prefix geometry differs')
    manifest = json.loads((CACHE / 'upstream_manifest.json').read_text())
    sources = {row['file']: dict(url=row['url'], sha256=sha(ROOT / row['file']),
        git_blob_sha1=subprocess.check_output(['git', 'hash-object', str(ROOT / row['file'])], text=True).strip())
        for row in manifest['files']}
    source = (CACHE / 'ollama_runner.go').read_text()
    body = source.split('func qwenVLServerArgs(modelArch string)', 1)[1].split('\n}\n', 1)[0]
    arches = re.findall(r'"([a-z0-9]+)"', body.split('return []string', 1)[0])
    if 'qwen35' in arches or '--image-min-tokens' not in body or '1024' not in body:
        raise ValueError('Pinned upstream architecture-specific budget rule differs')
    launch, = [line for line in text.splitlines() if 'starting llama-server' in line]
    if '--image-min-tokens' in launch or '--mmproj' not in launch:
        raise ValueError('Saved launch geometry differs')
    for fragment in ('patch_size:         16', 'n_merge:            2',
                     'image_min_pixels:   8192', 'image_max_pixels:   4194304'):
        if fragment not in text: raise ValueError('Saved vision preprocessing differs')
    runner = ROOT.parent / 'LocalLLM/.local/ollama/lib/ollama/llama-server'
    reported = subprocess.check_output([str(runner), '--version'], stderr=subprocess.STDOUT, text=True)
    if '96278e39f' not in reported: raise ValueError('Installed runner identity changed')
    value = dict(schema='saved-reader-preprocessing-audit-v1', audit_source_sha256=sha(Path(__file__)),
        retained_log_sha256=sha(log_path), installed_runner_sha256=sha(runner),
        installed_ollama_sha256=sha(ROOT.parent / 'LocalLLM/.local/ollama/bin/ollama'),
        runner_version_output=reported.strip(), declared_upstream=manifest,
        installed_go_binary_exact_source_unverified=True, reproducible_binary_build_verified=False,
        pinned_sources=sources, encoded_requests=len(token_counts), vision_positions_per_request=81,
        text_positions_per_request=98, total_prompt_positions=179,
        input_png_size=[288,288], patch_size=16, spatial_merge=2,
        expected_merged_positions=(288 // (16 * 2))**2,
        image_min_pixels=8192, image_max_pixels=4194304, original_size_within_limits=True,
        grounding_warning_present='at minimum 1024 image tokens' in text,
        upstream_minimum_1024_architectures=arches, qwen35_covered_by_that_switch=False,
        actual_command_has_image_min_tokens=False,
        full_prompt_reprocess_messages=text.count('forcing full prompt re-processing'),
        nonconsecutive_position_messages=text.count('non-consecutive token position'),
        interpretation='Images were encoded. The logged 81-position budget agrees with dynamic 288-pixel geometry. Pinned Ollama code applies a larger minimum to other Qwen-VL families but omits qwen35. Low budget is a candidate cause, not a demonstrated cause of OCR failure. No output or old review is repaired.',
        causal_resolution_experiment_performed=False, new_model_calls=0, optimizer_updates=0,
        teacher_qualified=False, student_language_improvement=False, goal_achieved=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({k:value[k] for k in ['encoded_requests','vision_positions_per_request',
        'qwen35_covered_by_that_switch','full_prompt_reprocess_messages','new_model_calls']}))


if __name__ == '__main__': main()
