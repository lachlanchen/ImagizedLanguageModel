#!/usr/bin/env python3
"""Frozen complete-response comparison for the actual-context glyph motor."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow, ContextMotorBridge
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.compare_glyph_tail_compact_cohort import verify_case
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt
from scripts.infer_compact_visual_span import save_grid
from scripts.score_pixel_language_first_blocks import require, owner_live
from scripts.train_glyph_context_motor import validate, load, CHECKPOINT, CHECKPOINT_SHA
from scripts.train_semantic_glyph_bridge import atomic_tensor

OWN = 'scripts/infer_glyph_context_motor.py'
SOURCES = (OWN, 'scripts/infer_glyph_tail_language.py', 'scripts/infer_compact_visual_span.py',
    'scripts/compare_glyph_tail_compact_cohort.py', 'ilm/visual_lm/glyph_reader_monitor.py',
    'ilm/visual_lm/glyph_runtime_resources.py', 'ilm/visual_lm/context_motor_flow.py',
    'ilm/visual_lm/semantic_glyph_loop.py')


def identity():
    result = {n:sha(ROOT/n) for n in SOURCES}
    for name, digest in result.items():
        raw = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        require(hashlib.sha256(raw).hexdigest() == digest, 'Commit inference source first: '+name)
    return result


def register(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration')
    require(not (run/'study_registration.json').exists() and not (run/'fit').exists(), 'Register once before fitting')
    r = validate(run)
    value = dict(schema='glyph-context-motor-study-v1', training_registration_sha256=sha(run/'registration.json'),
        evaluation_manifest_sha256=r['evaluation_manifest_sha256'], runtime_source_sha256=identity(),
        final_EMA_only=True, evaluation_streams=216, maximum_glyphs=128,
        one_sample_32Heun_last_Euler=True, parent_checkpoint_sha256=CHECKPOINT_SHA,
        pilot=dict(proportional_exact_minimum=8, proportional_pairs_minimum=2,
                   proportional_readable_minimum=11, retain_original_proportional_exact=6, retain_original_fixed_exact=2),
        complete_original_controls_preserved=True, untouched_confirmation_required=True,
        language_accuracy_not_measured=True, goal_achieved=False)
    save_json(run/'study_registration.json', value)
    public = ROOT/'references/evidence/glyph_context_motor_registration_20261111.json'
    require(not public.exists(), 'Preserve earlier public registration')
    save_json(public, dict(training=r, study=value))
    print(json.dumps(dict(study_registration_sha256=sha(run/'study_registration.json'),
                         public_registration_sha256=sha(public), registered_before_fitting=True)))


def study(run):
    r = validate(run); s = load(run/'study_registration.json')
    require(s['training_registration_sha256'] == sha(run/'registration.json'), 'Training registration changed')
    current = identity()
    if s['runtime_source_sha256'] != current:
        amendment = load(run/'execution_amendment.json')
        require(amendment['original_study_sha256'] == sha(run/'study_registration.json')
                and amendment['original_runtime_source_sha256'] == s['runtime_source_sha256']
                and amendment['updated_runtime_source_sha256'] == current
                and amendment['changed_runtime_sources'] == [OWN]
                and amendment['answers_emitted_before_amendment'] == 0,
                'Unregistered runtime change')
        s = dict(s, runtime_source_sha256=current, execution_amendment_sha256=sha(run/'execution_amendment.json'))
    return r, s


def amend_loading(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only recovery registration')
    validate(run); original = load(run/'study_registration.json'); current = identity()
    changed = [n for n in current if current[n] != original['runtime_source_sha256'][n]]
    require(changed == [OWN] and not (run/'execution_amendment.json').exists(), 'One explicit loading correction only')
    folder = run/'generation_fixed_cells'; status = load(folder/'status.json')
    require(status['phase'] == 'failed' and status['pid'] is None and not owner_live(status)
            and status['error'] == "RuntimeError('Inference tensors do not track version counter.')"
            and not list(folder.rglob('legacy_output.pt')) and not (run/'generation_proportional').exists(),
            'Only a terminal, zero-answer loading failure can use this amendment')
    attempt = load(folder/'attempts/attempt_0001.json')
    archive = run/'generation_attempts/fixed_cells_0001_setup_failure'
    archive.parent.mkdir(exist_ok=True); require(not archive.exists(), 'Preserve previous recovery')
    folder.rename(archive)
    (run/'generation_fixed_console.log').rename(archive/'console.log')
    amendment = dict(schema='glyph-context-motor-execution-recovery-v1',
        original_study_sha256=sha(run/'study_registration.json'),
        original_runtime_source_sha256=original['runtime_source_sha256'], updated_runtime_source_sha256=current,
        changed_runtime_sources=changed, answers_emitted_before_amendment=0,
        failed_attempt_receipt_sha256=sha(archive/'attempts/attempt_0001.json'),
        failed_console_sha256=sha(archive/'console.log'), failed_attempt_elapsed_seconds=attempt['elapsed_seconds'],
        failed_attempt_peak_allocated_vram_gib=attempt['peak_allocated_vram_gib'],
        failed_attempt_retained=str(archive.relative_to(ROOT)),
        correction='Construct frozen parameter modules under no_grad, then generate under inference_mode; parameter version counters remain available.',
        same_model_weights_training_sampler_inputs_stop_cap_and_pilot=True, optimizer_updates=0,
        prefill_or_answer_quality_observed_before_correction=False)
    save_json(run/'execution_amendment.json', amendment)
    public = ROOT/'references/evidence/glyph_context_motor_execution_recovery_20261111.json'
    require(not public.exists(), 'Preserve earlier public recovery')
    save_json(public, amendment)
    print(json.dumps(dict(execution_amendment_sha256=sha(run/'execution_amendment.json'),
                         answers_emitted_before_correction=0, model_recipe_unchanged=True)))


def verified_case(directory, row):
    result = verify_case(directory, row)
    value = torch.load(directory/'legacy_output.pt', map_location='cpu', weights_only=True)
    require(value['case_seed'] == row['seed'] and value['context_motor_flow'], 'Sample configuration changed')
    traces = value['motor_trace']; n = len(value['images'])
    require(len(traces) == n+int(value['capped']), 'Missing emitted or unused cap sample')
    for i, event in enumerate(traces):
        require(event['noise'].shape == (1, 1024) and event['normalized_motor'].shape == (1, 1024)
                and tensor_hash(event['noise']) == event['noise_sha256'], 'Motor/noise trace differs')
        require(torch.isfinite(event['normalized_motor']).all(), 'Nonfinite sampled motor')
        if i < n:
            require(event['image_sha256'] == tensor_hash(value['images'][i:i+1]), 'Motor output differs from actual image')
    require(result['case_seed'] == row['seed'] and result['unused_final_cap_samples'] == int(value['capped']),
            'Sample count or seed changed')
    return result


def run_generation(run, layout):
    r, s = study(run)
    fit = load(run/'fit/report.json'); checkpoint = run/'fit/checkpoint_latest.pt'
    require(fit['training_complete'] and fit['actual_optimizer_updates'] == 12000
            and sha(checkpoint) == fit['checkpoint_sha256'], 'Completed fixed-exposure writer required')
    rows = [x for x in load(run/'evaluation_manifest.json')['inputs'] if x['layout'] == layout]
    require(len(rows) == (120 if layout == 'fixed_cells' else 96), 'Incomplete evaluation arm')
    folder = run/('generation_'+layout); folder.mkdir(exist_ok=True)
    request = dict(study_registration_sha256=sha(run/'study_registration.json'),
        fit_checkpoint_sha256=fit['checkpoint_sha256'], layout=layout, inputs=rows,
        maximum_glyphs=128, source_sha256=s['runtime_source_sha256'],
        execution_amendment_sha256=s.get('execution_amendment_sha256'))
    if (folder/'request.json').exists():
        require(load(folder/'request.json') == request, 'Resume request changed')
    else:
        save_json(folder/'request.json', request)
    if (folder/'status.json').exists():
        require(not owner_live(load(folder/'status.json')), 'Previous generation owner is still live')
    if (folder/'report.json').exists() and load(folder/'report.json')['all_inputs_completed']:
        for row in rows:
            verified_case(folder/row['id'], row)
        return
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics
    attempts = folder/'attempts'; attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob('attempt_*.json')))+1
    monitor = GlyphReaderMonitor(folder, 3600)
    monitor.pressure_wait_allowance_seconds = 600
    monitor.status('admitting')
    loading = None
    try:
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            began = time.perf_counter()
            actor, provenance = load_actor(CHECKPOINT, device)
            payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
            flow = ContextMotorFlow().to(device).eval().requires_grad_(False)
            flow.load_state_dict(payload['ema'])
            actor.bridge = ContextMotorBridge(actor.bridge, flow, payload['motor_mean'].to(device),
                                             payload['motor_scale'].to(device)).eval().requires_grad_(False)
            del payload
            require(not any(p.requires_grad for p in actor.parameters()), 'Frozen inference required')
            frozen = [(p, p._version) for p in actor.parameters()]
            torch.cuda.synchronize(); loading = time.perf_counter()-began
            hook = actor.interface.actor.context.register_forward_pre_hook(lambda _m, _a:monitor.guard())
            records = []
            for index, row in enumerate(rows):
                monitor.position = dict(completed=index, total=len(rows), case=row['id'])
                monitor.guard(); monitor.status('generating_actual_context_motor')
                directory = folder/row['id']
                if directory.exists():
                    records.append(verified_case(directory, row)); continue
                canvas, observed = prepare_prompt(ROOT/row['image'])
                prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None, None]/127.5-1
                actor.bridge.begin_case(row['seed'])
                torch.cuda.synchronize(); began = time.perf_counter()
                with torch.inference_mode():
                    value = actor.generate(prompt, maximum_glyphs=128)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-began
                value.update(context_motor_flow=True, motor_trace=actor.bridge.motor_trace,
                             case_seed=row['seed'], sampling_candidates=1, sampling_intervals=32)
                signed = value['images']; ink = ((1-signed)/2).to(torch.uint8)
                require(torch.equal(signed, 1-2*ink.float()), 'Lossy image conversion')
                partial = folder/f'.{row["id"]}.{time.time_ns()}.partial'; partial.mkdir()
                atomic_tensor(partial/'legacy_output.pt', value)
                np.savez_compressed(partial/'ink.npz', ink=ink.numpy()); save_grid(ink, partial/'answer.png')
                feedback = [dict(glyphs=1, continuous_updates=1,
                    ink_sha256=hashlib.sha256(ink[i:i+1].numpy().tobytes()).hexdigest(),
                    signed_image_sha256=e['image_sha256'], native_embedding_sha256=e['embedding_sha256'])
                    for i, e in enumerate(value['feedback'])]
                result = dict(id=row['id'], original_case=row['original_case'], layout=layout,
                    input_sha256=row['sha256'], input_glyphs=row['glyphs'], emitted_glyphs=len(ink),
                    stopped=value['stopped'], capped=value['capped'], elapsed_seconds=elapsed,
                    case_seed=row['seed'], unused_final_cap_samples=int(value['capped']),
                    emitted_pixel_feedback=feedback, legacy_output_sha256=sha(partial/'legacy_output.pt'),
                    raw_sha256=sha(partial/'ink.npz'), image_sha256=sha(partial/'answer.png'),
                    original_observer_geometry=value['observer_geometry'], host_observation=observed,
                    actual_pixel_feedback_verified=True, reference_inputs=False, output_repaired=False)
                save_json(partial/'report.json', result); partial.rename(directory)
                records.append(verified_case(directory, row))
                require(all(p._version == v for p, v in frozen), 'Frozen inference parameter mutated')
                print(json.dumps({k:result[k] for k in ['id', 'emitted_glyphs', 'stopped', 'elapsed_seconds']}), flush=True)
                del value, signed, ink; gc.collect()
            hook.remove()
            save_json(folder/'report.json', dict(request_sha256=sha(folder/'request.json'), records=records,
                all_inputs_completed=len(records) == len(rows), new_optimizer_updates=0,
                parent_checkpoint_sha256=CHECKPOINT_SHA, motor_checkpoint_sha256=fit['checkpoint_sha256'],
                provenance=provenance, actor_loading_seconds=loading,
                total_generation_seconds=sum(x['elapsed_seconds'] for x in records),
                actual_sampled_pixels_verified=True, all_frozen_parameter_versions_unchanged=True,
                one_sample_no_reranking=True, language_accuracy_not_measured=True, goal_achieved=False,
                **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        status = dict(load(folder/'status.json'), pid=None); save_json(folder/'status.json', status)
        save_json(attempts/f'attempt_{attempt:04}.json', dict(request_sha256=sha(folder/'request.json'),
            terminal_status=status, actor_loading_seconds=loading, **monitor.receipt()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['register', 'amend-loading', 'run'])
    parser.add_argument('--layout', choices=['fixed_cells', 'proportional'])
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_context_motor_v1')
    args = parser.parse_args()
    if args.stage == 'register':
        register(args.run.resolve())
    elif args.stage == 'amend-loading':
        amend_loading(args.run.resolve())
    else:
        require(args.layout is not None, 'Choose one complete presentation stage')
        run_generation(args.run.resolve(), args.layout)
