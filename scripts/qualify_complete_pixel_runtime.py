#!/usr/bin/env python3
"""Cold-load the deployable pixel actor and replay fixed original PNG inputs.

This qualifies deployment reconstruction, not unseen-language quality. The
first four cases and the first longest answer are selected without scores.
Expected output tensors are read only after each actual generation is saved.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.visual_lm.codec_image_interface import prepare_image_prompt
from ilm.visual_lm.complete_pixel_runtime import RUNTIME_CORE, read_pixel_checkpoint, load_pixel_actor
from ilm.visual_lm.native_prompt_interface import validate_native_answer
from scripts.infer_complete_pixel_language import save_answer
from scripts.preflight_native_pixel_actor import CODEC, CONTROLS


def load(path):
    return json.loads(path.read_text())


def terminal(run):
    status = load(run / 'execution/status.json')
    if status['phase'] != 'completed' or status['pid'] is not None:
        raise ValueError('the original training and assessment process must finish')


def prepare(run, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('hide CUDA for preparation')
    terminal(run)
    audit = load(run / 'audit.json')
    if audit['report_sha256'] != sha(run / 'execution/report.json'):
        raise ValueError('complete the independent final audit first')
    cases = load(run / 'cases.json')
    selected = list(dict.fromkeys([0, 1, 2, 3, max(range(32), key=lambda i: cases[i]['blocks'])]))
    files = (*RUNTIME_CORE, 'ilm/visual_lm/complete_pixel_runtime.py',
             'ilm/visual_lm/codec_image_interface.py', 'ilm/visual_lm/native_prompt_interface.py',
             'scripts/infer_complete_pixel_language.py', 'scripts/qualify_complete_pixel_runtime.py')
    code = {name: sha(ROOT / name) for name in files}
    for name, digest in code.items():
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)).hexdigest() != digest:
            raise ValueError('commit the deployable runtime first: ' + name)
    bound = {name: sha(run / name) for name in
             ('preparation.json', 'cases.json', 'audit.json', 'execution/report.json', 'execution/checkpoint_latest.pt')}
    for index in selected:
        bound[cases[index]['normal']] = sha(run / cases[index]['normal'])
        for name in ('receipt.json', 'result.pt', 'answer.png'):
            relative = f'execution/final_train/{index:02}/{name}'
            bound[relative] = sha(run / relative)
        receipt = load(run / f'execution/final_train/{index:02}/receipt.json')
        if not receipt['identity']['actual_PNG_prefill']:
            raise ValueError('the owning baseline must have invoked the real PNG reader')
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'preparation.json', dict(code_sha256=code, bound_sha256=bound, indices=selected,
        training_run=str(run.relative_to(ROOT)), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        selection='first four training cases plus first longest complete answer, without outcome selection',
        language_quality_claim=False))
    print(json.dumps(dict(indices=selected, model_loaded=False, CUDA_initialized=torch.cuda.is_initialized()), indent=2))


def execute(run, output):
    terminal(run)
    meta = load(output / 'preparation.json')
    for name, digest in meta['code_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('prepared runtime source changed: ' + name)
    for name, digest in meta['bound_sha256'].items():
        if sha(run / name) != digest:
            raise ValueError('prepared input or owning output changed: ' + name)
    if (output / 'status.json').exists():
        raise FileExistsError('preserve the existing qualification and prepare a new directory')
    torch.set_num_threads(4)
    started = time.perf_counter()
    save_json(output / 'status.json', dict(phase='loading_cold_actor', pid=os.getpid()))
    results = []
    try:
        with gpu_zero() as device:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            state, raster, identity = read_pixel_checkpoint(run / 'execution/checkpoint_latest.pt', run)
            local = ROOT.parent / 'LocalLLM/.local'
            actor, provenance = load_pixel_actor(state, raster, controls=CONTROLS, codec=CODEC,
                ollama_root=local / 'models/ollama', gguf_source=local / 'tools/llama.cpp/gguf-py', device=device)
            load_seconds = time.perf_counter() - started
            (output / 'cases').mkdir()
            cases = load(run / 'cases.json')
            for index in meta['indices']:
                prepared = prepare_image_prompt(run / cases[index]['normal'], layout='page')
                begun = time.perf_counter()
                actual = actor.generate(prepared.pixels, maximum_blocks=128)
                validate_native_answer(actual, 128)
                saved = save_answer(actual, output / 'cases' / f'{index:02}', page_width=1024)
                # No expected output or answer length enters the actor.
                expected = torch.load(run / f'execution/final_train/{index:02}/result.pt', map_location='cpu', weights_only=True)
                checks = dict(images_bit_exact=torch.equal(actual['images'], expected['images']),
                    stop_probabilities_bit_exact=actual['stop_probabilities'] == expected['stop_probabilities'],
                    stop_and_cap_equal=(actual['stopped'], actual['capped']) == (expected['stopped'], expected['capped']),
                    history_feature_hashes_equal=actual['feedback_sha256'] == expected['feedback_sha256'],
                    reader_memory_hash_equal=actual['reader_memory_sha256'] == expected['reader_memory_sha256'])
                record = dict(index=index, checks=checks, output=saved, generation_seconds=time.perf_counter() - begun)
                results.append(record)
                save_json(output / f'case_{index:02}.json', record)
                save_json(output / 'status.json', dict(phase='actual_PNG_replay', pid=os.getpid(), completed=len(results), total=len(meta['indices'])))
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            passed = all(all(row['checks'].values()) for row in results) and peak <= 22
            save_json(output / 'report.json', dict(checkpoint=identity, source=provenance, cases=results,
                all_actual_PNG_replays_bit_exact=passed, actual_PNG_inputs=len(results),
                peak_allocated_vram_gib=peak, model_load_seconds=load_seconds,
                elapsed_seconds=time.perf_counter() - started, reference_answers_supplied=False,
                legacy_training_checkpoint_loaded=False, teacher_service=False, runtime_OCR=False,
                font_answer_renderer=False, language_generalization_qualified=False))
            if not passed:
                raise ValueError('cold deployment reconstruction differs; preserve the complete raw evidence')
        save_json(output / 'status.json', dict(phase='completed', pid=None, cases=len(results)))
        print(json.dumps(dict(cases=len(results), all_replays_bit_exact=True, peak_allocated_vram_gib=peak), indent=2))
    except BaseException as error:
        save_json(output / 'status.json', dict(phase='failed', pid=None, reason=repr(error), completed=len(results)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else execute)(args.training_run.resolve(), args.output.resolve())
