#!/usr/bin/env python3
"""Gated complete-response evaluation of the two registered rollout-fit arms."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from ilm.visual_lm.context_motor_flow import ContextMotorFlow,ContextMotorBridge
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.infer_glyph_context_motor import SOURCES as BASE_SOURCES,verified_case
from scripts.infer_glyph_tail_language import load_actor,prepare_prompt
from scripts.infer_compact_visual_span import save_grid
from scripts.score_pixel_language_first_blocks import require,owner_live
from scripts.train_rollout_endpoint_motor import validate,load,CHECKPOINT,CHECKPOINT_SHA,BASE,RUN
from scripts.train_semantic_glyph_bridge import atomic_tensor

OWN='scripts/infer_rollout_endpoint_motor.py'
SOURCES=(OWN,*BASE_SOURCES,'scripts/train_rollout_endpoint_motor.py',
         'scripts/assess_rollout_endpoint_motor.py','ilm/visual_lm/rollout_endpoint.py')
PUBLIC=ROOT/'references/evidence/rollout_endpoint_language_registration_20261111.json'


def identity():
    for name in SOURCES:
        require(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)==(ROOT/name).read_bytes(),
                'Commit the complete runtime before language registration: '+name)
    return {name:sha(ROOT/name) for name in SOURCES}


def register(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),
            'CPU-only registration before language generation')
    require(not (run/'language_registration.json').exists() and not PUBLIC.exists(),'Register once')
    r=validate(run);assessment=load(run/'assessment/report.json');audit=load(run/'fit_audit.json')
    require(load(run/'assessment/status.json')['phase']=='complete'
        and assessment['full216_language_generation_eligible']
        and all(assessment['training_writing_gates']['candidate'].values()),
        'Candidate must pass both fixed physical gates before either language trial')
    require(audit['registration_sha256']==sha(run/'registration.json')
        and audit['paired_indices_and_final_CPU_CUDA_RNG_bit_exact'],'Finish the independent fit audit')
    for arm in ['control','rollout']:
        path=run/('fit_'+arm)/'checkpoint_latest.pt'
        require(sha(path)==assessment['checkpoint_sha256'][arm]==audit['arms'][arm]['checkpoint_sha256'],
                'Final paired model binding changed')
        require(not (run/('generation_'+arm+'_fixed_cells')).exists()
            and not (run/('generation_'+arm+'_proportional')).exists(),'Register before any new language output')
    value=dict(schema='rollout-endpoint-language-v1',training_registration_sha256=sha(run/'registration.json'),
        assessment_sha256=sha(run/'assessment/report.json'),fit_audit_sha256=sha(run/'fit_audit.json'),
        final_checkpoint_sha256=assessment['checkpoint_sha256'],runtime_source_sha256=identity(),
        evaluation_manifest_sha256=sha(BASE/'evaluation_manifest.json'),
        final_EMA_only=True,arms=['control','rollout'],streams_per_arm=216,maximum_glyphs=128,
        one_sample_32Heun_last_Euler=True,complete_language_pilot=r['complete_language_pilot'],
        same_original_inputs_seeds_stopping_and_pixel_feedback=True,
        runtime_registered_after_headfit_before_language_generation=True,
        no_candidate_language_outputs_or_scores_seen=True,untouched_confirmation_required=True,
        language_accuracy_not_measured=True,goal_achieved=False)
    save_json(run/'language_registration.json',value);save_json(PUBLIC,value)
    print(json.dumps(dict(language_registration_sha256=sha(PUBLIC),outputs_not_started=True)))


def study(run):
    r=validate(run);s=load(run/'language_registration.json')
    require(s['training_registration_sha256']==sha(run/'registration.json')
        and s['assessment_sha256']==sha(run/'assessment/report.json')
        and s['fit_audit_sha256']==sha(run/'fit_audit.json')
        and s['runtime_source_sha256']==identity(),'Language registration changed')
    require(PUBLIC.read_bytes()==(run/'language_registration.json').read_bytes()
        and subprocess.check_output(['git','show','HEAD:'+str(PUBLIC.relative_to(ROOT))],cwd=ROOT)==PUBLIC.read_bytes(),
        'Commit language registration before generation')
    for arm in ['control','rollout']:
        require(sha(run/('fit_'+arm)/'checkpoint_latest.pt')==s['final_checkpoint_sha256'][arm],
                'Registered final model changed')
    return r,s


def run_generation(run, layout, arm):
    r, s = study(run)
    fit = load(run/('fit_'+arm)/'report.json'); checkpoint = run/('fit_'+arm)/'checkpoint_latest.pt'
    require(fit['training_complete'] and fit['actual_optimizer_updates'] == 2000
            and sha(checkpoint) == fit['checkpoint_sha256'], 'Completed fixed-exposure writer required')
    rows = [x for x in load(BASE/'evaluation_manifest.json')['inputs'] if x['layout'] == layout]
    require(len(rows) == (120 if layout == 'fixed_cells' else 96), 'Incomplete evaluation arm')
    folder = run/('generation_'+arm+'_'+layout); folder.mkdir(exist_ok=True)
    request = dict(study_registration_sha256=sha(run/'language_registration.json'),
        fit_checkpoint_sha256=fit['checkpoint_sha256'], layout=layout, arm=arm, inputs=rows,
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


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['register','run']);parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--layout',choices=['fixed_cells','proportional'])
    parser.add_argument('--arm',choices=['control','rollout']);args=parser.parse_args()
    if args.stage=='register':register(args.run.resolve())
    else:
        require(args.arm is not None and args.layout is not None,'Choose one registered arm and layout')
        run_generation(args.run.resolve(),args.layout,args.arm)
