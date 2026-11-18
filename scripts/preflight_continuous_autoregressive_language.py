#!/usr/bin/env python3
"""Register/execute a complete-record native continuous-density update/reset."""
import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from scripts import diagnose_visual_context_gradients as common

ROOT = common.ROOT
REG = ROOT / 'references/evidence/continuous_autoregressive_preflight_registration_20261118.json'
OUT = ROOT / 'references/evidence/continuous_autoregressive_preflight_result_20261118.json'
RUN = ROOT / 'artifacts/continuous_autoregressive_preflight_v1'
PARENT = ROOT / 'references/evidence/trainable_visual_context_preflight_registration_20261116.json'
DATA = ROOT / 'data/visual_sequence_corpus_v1'
CODEC = ROOT / 'artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt'
FILES = [
    'scripts/preflight_continuous_autoregressive_language.py',
    'ilm/visual_lm/continuous_autoregressive_language.py',
    'tests/test_continuous_autoregressive_language.py',
    'references/continuous_autoregressive_language_direction_20261118.md',
    'ilm/visual_lm/continuous_glyph_codec.py', 'ilm/visual_lm/codec_block_writer.py',
    'ilm/visual_lm/conditional_pixel_density.py', 'ilm/visual_lm/visual_cell_stream.py',
    'ilm/visual_lm/saccade_lm.py', 'ilm/visual_lm/visual_sequence_corpus.py',
    'ilm/visual_lm/full_rank_visual_span.py', 'ilm/visual_lm/glyph_reader_monitor.py',
    'ilm/visual_lm/visual_context_deployment_resources.py',
    'scripts/diagnose_visual_context_gradients.py',
    'scripts/train_semantic_glyph_language.py', 'scripts/train_semantic_glyph_bridge.py',
    'scripts/score_pixel_language_first_blocks.py', 'ilm/image_edit/baseline.py',
    'references/evidence/codec_output_v67_result.json',
    'references/continuous_glyph_codec_v34_result.md',
    'references/evidence/residual_plan_motor_comparison_result_20261118.json',
]


def register():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Register with CUDA hidden')
    cases = common.load(PARENT)['cases']
    report = common.load(DATA / 'report.json')
    shards = [s for s in report['shards'] if s['split'] == 'train']
    paths = {ROOT / n for n in FILES} | {PARENT, CODEC, DATA / 'report.json', DATA / 'verification.json'}
    for case in cases:
        for name, digest in shards[case['shard']]['files'].items():
            if name.endswith(('_pixels.npz', '_sources.json')):
                common.require(common.sha(DATA / name) == digest, 'Original training source changed')
                paths.add(DATA / name)
    common.write_new(REG, dict(schema='ilm-continuous-autoregressive-preflight-v1',
        input_sha256={str(p.relative_to(ROOT)): common.sha(p) for p in sorted(paths)},
        packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy')}, cases=cases,
        data_report_sha256=common.sha(DATA / 'report.json'), seed=202611198,
        components=[1, 8], model_dim=384, layers=8, heads=6, maximum_cells=2048,
        group_width=16, visual_groups=48, minimum_scale=.02,
        precision='FP32 with global strict deterministic algorithms; TF32 disabled',
        optimizer=dict(lr=.0001, betas=[.9, .95], weight_decay=.01, gradient_clip=1.),
        resources=dict(maximum_seconds=900, maximum_pressure_wait_seconds=180,
            minimum_entry_host_gib=48, minimum_runtime_host_gib=24, minimum_free_gpu_gib=20,
            maximum_allocated_gpu_gib=22, maximum_process_rss_gib=8, minimum_free_disk_gib=48, cpu_threads=2),
        actual_trial_updates=6, retained_updates=0, complete_target_positions_per_arm=552,
        support=['finite loss and all selected gradients nonzero', 'same complete-record total objective decreases after one actual step',
            'all learned tensors reset exactly', 'frozen codec unchanged', 'identical common initial tensors',
            'all actual decoded initial/updated conditional images preserved', 'resource limits pass'],
        action_on_pass='Separately register complete-corpus training and autonomous language evaluation; no full fit admitted here.',
        action_on_failure='Preserve failure without automatic retry or threshold change.',
        inherited_language_parameters=0, teacher_states_or_embeddings_used=False,
        supplied_history_outputs_are_not_autonomous_language=True,
        latent_density_is_not_pixel_likelihood=True, language_quality_measured=False, goal_achieved=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=common.sha(REG))), flush=True)


def run():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Registered physical GPU0 execution required')
    common.require(not RUN.exists() and not OUT.exists(), 'Preserve every first execution')
    reg = common.load(REG)
    for name, digest in reg['input_sha256'].items():
        common.require(common.sha(ROOT / name) == digest, 'Bound file changed: ' + name)
        if name.startswith(('ilm/', 'scripts/', 'tests/', 'references/')):
            common.require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT) == (ROOT / name).read_bytes(), 'Commit all bound sources')
    common.require(subprocess.check_output(['git', 'show', 'HEAD:' + str(REG.relative_to(ROOT))], cwd=ROOT) == REG.read_bytes(), 'Commit registration')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    common.require(subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True, timeout=30).split()[0] == head, 'Push before execution')
    common.require(reg['packages'] == {p: importlib.metadata.version(p) for p in reg['packages']}, 'Packages changed')
    import numpy as np
    import torch
    from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
    from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
    from scripts.measure_reader_writing_consequence import pixels
    RUN.mkdir()
    monitor = GlyphReaderMonitor(RUN, 900)
    monitor.pressure_wait_allowance_seconds = 180
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('Continuous preflight external deadline')))
    rows, actual_updates = [], 0
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            def guard():
                monitor.guard()
                common.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3, 'Process RSS ceiling')
            corpus = VisualSequenceCorpus(DATA, split='train', expected_report_sha256=reg['data_report_sha256'])
            codec = load_qualified_writing_codec(CODEC, device)
            frozen_hash = tensor_state_sha(codec.state_dict())
            common_initial = None
            parameters = {}
            for components in reg['components']:
                guard()
                torch.manual_seed(reg['seed'])
                model = ContinuousImageLanguage(codec, components=components, model_dim=reg['model_dim'],
                    layers=reg['layers'], heads=reg['heads'], maximum_cells=reg['maximum_cells']).to(device).eval()
                parameters[str(components)] = dict(whole=sum(p.numel() for p in model.parameters()),
                    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), codec=7423361)
                initial = {n: p.detach().cpu().clone() for n, p in model.state_dict().items() if not n.startswith('codec.')}
                common_hash = tensor_state_sha({n: p for n, p in initial.items() if not n.startswith('writer.output.')})
                if common_initial is None:
                    common_initial = common_hash
                common.require(common_initial == common_hash, 'Common body/local initialization differs')
                initial_hash = tensor_state_sha(initial)
                for case in reg['cases']:
                    monitor.position = dict(components=components, case=case['role'])
                    monitor.status('complete_native_visual_record_update_and_reset')
                    guard()
                    row = corpus[case['corpus_index']]
                    ink = row['ink'].to(device)
                    common.require(len(ink) == case['glyphs'] and row['prompt_glyphs'] == case['prompt_glyphs'], 'Keep complete authenticated record')
                    prompt = row['prompt_glyphs']
                    first = max(1, prompt)
                    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=reg['optimizer']['lr'], betas=tuple(reg['optimizer']['betas']), weight_decay=reg['optimizer']['weight_decay'])
                    def assess(stage):
                        guard()
                        with torch.no_grad():
                            terms = model.training_terms(ink, prompt)
                            state = model.history(ink)[0, first - 1:-1]
                            generated = []
                            for offset in range(0, len(state), 32):
                                guard()
                                images, _ = model.draw(state[offset:offset + 32])
                                generated.append(images.cpu())
                            images = torch.cat(generated)
                        path = RUN / f'k{components}_{case["role"]}_{stage}.npz'
                        np.savez_compressed(path, ink=images.numpy())
                        return dict(code_nll=float(terms['code_nll']), stop_bce=float(terms['stop_bce']),
                            total=float(terms['code_nll'] + terms['stop_bce']),
                            target_positions=terms['target_positions'], stop_positions=terms['stop_positions'],
                            pixels=pixels(images, ink[first:].cpu()),
                            raw_file=str(path.relative_to(ROOT)), raw_sha256=common.sha(path))
                    before = assess('initial')
                    optimizer.zero_grad(set_to_none=True)
                    terms = model.training_terms(ink, prompt)
                    loss = terms['code_nll'] + terms['stop_bce']
                    common.require(bool(torch.isfinite(loss)), 'Nonfinite real-source loss')
                    loss.backward()
                    selected = ['input_projection.weight', 'blocks.0.attention.qkv.weight',
                        'blocks.7.attention.qkv.weight', 'writer.condition.weight',
                        'writer.blocks.0.qkv.weight', 'writer.output.weight', 'stop.weight']
                    named = dict(model.named_parameters())
                    norms = {n: float(named[n].grad.norm()) if named[n].grad is not None else 0. for n in selected}
                    common.require(all(np.isfinite(v) and v > 0 for v in norms.values()), 'Missing actual native/density/stop gradients')
                    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                    optimizer.step()
                    actual_updates += 1
                    changed = tensor_state_sha({n: p for n, p in model.state_dict().items() if not n.startswith('codec.')})
                    common.require(changed != initial_hash, 'Actual optimizer did not change model')
                    after = assess('updated')
                    common.require(after['total'] < before['total'], 'Complete same-record objective did not decrease')
                    common.require(tensor_state_sha(codec.state_dict()) == frozen_hash, 'Frozen motor changed')
                    for name, parameter in model.state_dict().items():
                        if name in initial:
                            parameter.copy_(initial[name].to(parameter))
                    reset_hash = tensor_state_sha({n: p for n, p in model.state_dict().items() if not n.startswith('codec.')})
                    common.require(reset_hash == initial_hash, 'Trial model reset differs')
                    rows.append(dict(components=components, case=case, before=before, after=after,
                        gradient_norms=norms, total_unclipped_gradient_norm=float(norm),
                        initial_sha256=initial_hash, updated_sha256=changed, reset_sha256=reset_hash,
                        frozen_codec_unchanged=True, complete_record=True))
                    common.write_new(RUN / f'k{components}_{case["role"]}_receipt.json', rows[-1])
                    del optimizer, terms, loss, ink
                    model.zero_grad(set_to_none=True)
                    guard()
                del model, initial
                gc.collect()
                torch.cuda.empty_cache()
            common.require(actual_updates == 6 and sum(v['before']['target_positions'] for v in rows) == 1104, 'Complete six-case support required')
            result = dict(schema='ilm-continuous-autoregressive-preflight-result-v1',
                registration_sha256=common.sha(REG), source_commit=head,
                source_hashes=reg['input_sha256'], parameters=parameters, cases=rows,
                actual_trial_updates=actual_updates, retained_updates=0,
                actual_decoded_images=2208, common_initial_sha256=common_initial,
                all_trial_updates_reset=True, execution_support=True,
                language_quality_measured=False, autonomous_generation_run=False,
                current_direction_proved=False, full_fit_authorized=False, goal_achieved=False,
                resources=dict(monitor.receipt(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024))
            common.write_new(OUT, result)
            monitor.status('complete_reset_support_only')
            print(json.dumps(dict(output=str(OUT.relative_to(ROOT)), sha256=common.sha(OUT), parameters=parameters)), flush=True)
    except BaseException as error:
        common.write_new(RUN / 'failure.json', dict(error=repr(error), completed_cases=rows,
            actual_trial_updates=actual_updates, resources=monitor.receipt(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['register', 'run'], required=True)
    (register if parser.parse_args().stage == 'register' else run)()
