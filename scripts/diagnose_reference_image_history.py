#!/usr/bin/env python3
"""Register/run the frozen reference-history instrument; no training or answer scoring."""
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

from ilm.image_edit.baseline import ROOT, save_json, sha
from ilm.visual_lm.semantic_glyph_loop import tensor_hash
from scripts.compare_balanced_writer_matched import RUN as BALANCED, CHECKPOINT, CHECKPOINT_SHA
from scripts.compare_glyph_tail_compact_cohort import SOURCES as ACTOR_SOURCES
from scripts.infer_glyph_tail_language import load_actor, prepare_prompt
from scripts.score_pixel_language_first_blocks import require, owner_live
from scripts.train_semantic_glyph_bridge import atomic_tensor
from scripts.audit_glyph_tail_first_writing import pixels_metrics

RUN = ROOT/'artifacts/reference_image_history_v1'
FIRST = ROOT/'artifacts/glyph_tail_first_writing_v1'
NATIVE = ROOT/'references/evidence/glyph_tail_native_reader_learning_20261111.json'
PUBLIC = ROOT/'references/evidence/reference_image_history_registration_20261112.json'
SOURCES = tuple(dict.fromkeys((*ACTOR_SOURCES,
    'scripts/diagnose_reference_image_history.py',
    'scripts/summarize_reference_image_history.py',
    'scripts/compare_balanced_writer_matched.py',
    'ilm/visual_lm/reference_image_history.py',
    'tests/test_reference_image_history.py',
    'references/reference_image_history_decision.md',
    'ilm/visual_lm/glyph_reader_monitor.py', 'ilm/visual_lm/glyph_runtime_resources.py',
    'scripts/audit_glyph_tail_first_writing.py')))


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sources():
    result = {}
    for name in SOURCES:
        require(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
                == (ROOT/name).read_bytes(), 'Commit diagnostic sources first: '+name)
        result[name] = sha(ROOT/name)
    return result


def register(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(),
            'CPU registration requires hidden CUDA')
    require(not run.exists() and not PUBLIC.exists(), 'Preserve existing registration')
    torch.set_num_threads(2)
    code = sources()
    require(sha(CHECKPOINT) == CHECKPOINT_SHA, 'Frozen balanced model differs')
    first = load(FIRST/'registration.json')
    require(sha(FIRST/'prepared_pixels.pt') == first['prepared_pixels_sha256'], 'Reference archive changed')
    prepared = torch.load(FIRST/'prepared_pixels.pt', weights_only=True, map_location='cpu')
    targets = first['targets']; gold = prepared['gold_ink']
    require(len(targets) == 32 and sum(t['glyphs'] for t in targets) == len(gold) == 745,
            'Keep complete canonical references')
    for target in targets:
        image = gold[target['offset']:target['offset']+target['glyphs']]
        require(tensor_hash(image) == target['source_glyphs_sha256'], 'Original reference pixels differ')
    by_case = {t['case']:t for t in targets}
    previous = load(BALANCED/'registration.json')
    require(previous['checkpoint_sha256'] == CHECKPOINT_SHA, 'Prior actor differs')
    bindings = {}
    def bind(path):
        bindings[rel(path)] = sha(path)
        return rel(path)
    for path in [CHECKPOINT, FIRST/'registration.json', FIRST/'prepared_pixels.pt',
                 BALANCED/'registration.json', NATIVE,
                 ROOT/'references/evidence/balanced_writer_matched_learning_20261111.json',
                 ROOT/'references/evidence/glyph_context_motor_learning_20261111.json']:
        bind(path)
    native = load(NATIVE)
    require(native['all448_cases_authenticated'], 'Use completed native diagnostic')
    natives = {(r['layout'], r['case']):r for r in native['records']
               if r['tail'] == 'adapted' and r['layout'] != 'text'}
    jobs = []; first_states = {}
    for layout, parent in [('fixed_cells', ROOT/'artifacts/glyph_tail_compact_cohort_v1'),
                           ('proportional', ROOT/'artifacts/glyph_tail_input_presentation_v1')]:
        paths = {'balanced':BALANCED/'review'/layout/'scores.json',
                 'flow':ROOT/'artifacts/glyph_context_motor_v1/review'/layout/'scores.json',
                 'parent':parent/'review/scores.json'}
        scores = {}
        for name, path in paths.items():
            bind(path); scores[name] = {r['case']:r for r in load(path)['records']}
        folder = BALANCED/('generation_'+layout)
        report = load(folder/'report.json'); bind(folder/'report.json')
        require(report['all_inputs_completed'], 'Prior generation incomplete')
        actuals = {r['id']:r for r in report['records']}
        for row in previous['inputs']:
            if row['layout'] != layout or row['condition'] != 'normal' or '__qa_' not in row['id']:
                continue
            key = row['original_case'].removesuffix('_normal'); target = by_case[key]
            require(sha(ROOT/row['image']) == row['sha256'], 'Prompt image differs')
            bind(ROOT/row['image']); old_path = folder/row['id']/'legacy_output.pt'
            require(sha(old_path) == actuals[row['id']]['legacy_output_sha256'], 'Saved trajectory differs')
            bind(old_path); value = torch.load(old_path, weights_only=True, map_location='cpu')
            require(len(value['states']) == len(value['images'])+1 == len(value['stop_probabilities']),
                    'Incomplete autonomous trace')
            first_states[row['id']] = value['states'][0].clone()
            require(tensor_hash(first_states[row['id']]) == natives[layout, row['original_case']]['prefill_state_sha256'],
                    'Existing native/actual initial state mismatch')
            jobs.append(dict(id=row['id'], layout=layout, case=key, image=row['image'],
                image_sha256=row['sha256'], original_output=rel(old_path),
                expected_geometry=value['observer_geometry'], reference_offset=target['offset'],
                reference_glyphs=target['glyphs'], native_complete_exact=natives[layout, row['original_case']]['complete_exact'],
                original_scores={name:{k:values[row['original_case']][k] for k in
                    ['complete_exact', 'readable', 'stopped', 'emitted_glyphs']} for name, values in scores.items()}))
    require(len(jobs) == 64 and len({j['id'] for j in jobs}) == 64, 'Keep all normal cases in both layouts')
    run.mkdir(parents=True)
    atomic_tensor(run/'prepared.pt', dict(gold_ink=gold, initial_states=first_states))
    value = dict(schema='reference-image-history-v1', source_sha256=code, bindings_sha256=bindings,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        checkpoint=rel(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA, jobs=jobs, targets=targets,
        prepared_sha256=sha(run/'prepared.pt'), correct_history_target_positions=1490,
        terminal_boundaries=64, new_optimizer_updates=0, new_autonomous_generations=0,
        maximum_seconds=3600, pressure_wait_allowance_seconds=600, minimum_entry_host_ram_gib=48,
        minimum_runtime_host_ram_gib=24, minimum_free_gpu_gib=20, maximum_allocated_vram_gib=22,
        later_physical_gate=dict(median_f1=.99, fraction_at_least_095=.9),
        stop_diagnostic=dict(maximum_nonterminal_false_stop_fraction=.05, minimum_terminal_recognition_fraction=.8),
        supplied_reference_images=True, supplied_target_length=True, autonomous_language_evidence=False,
        prior_development_familiarity=True, independent_human_adjudication=False, goal_achieved=False)
    save_json(run/'registration.json', value); save_json(PUBLIC, value)
    print(json.dumps(dict(registration_sha256=sha(PUBLIC), cases=64, glyph_predictions=1490, terminal_boundaries=64)))


def validate(run):
    r = load(run/'registration.json')
    require(r['source_sha256'] == sources(), 'Registered sources differ')
    for name, digest in r['bindings_sha256'].items():
        require(sha(ROOT/name) == digest, 'Bound input changed: '+name)
    require(sha(run/'prepared.pt') == r['prepared_sha256'], 'Prepared images/states differ')
    require(PUBLIC.read_bytes() == (run/'registration.json').read_bytes() ==
        subprocess.check_output(['git','show','HEAD:'+rel(PUBLIC)], cwd=ROOT), 'Commit registration before model calls')
    return r


def verify_case(folder, job, registration_sha):
    report = load(folder/'report.json')
    require(report['job'] == job and report['registration_sha256'] == registration_sha, 'Case binding differs')
    require(sha(folder/'trace.pt') == report['trace_sha256'] and
            sha(folder/'pixels.npz') == report['pixels_sha256'], 'Saved diagnostic trace differs')
    require(report['initial_state_bit_exact'] and report['autonomous_pixels_reproduced'] and
            report['autonomous_stop_probabilities_reproduced'], 'Instrument identity failed')
    return report


def run_model(run):
    r = validate(run); registration_sha = sha(run/'registration.json')
    execution = run/'execution'; execution.mkdir(exist_ok=True)
    if (execution/'status.json').exists():
        require(not owner_live(load(execution/'status.json')), 'Prior owner still live')
    if (execution/'report.json').exists() and load(execution/'report.json')['all64_complete']:
        for job in r['jobs']:verify_case(execution/job['id'], job, registration_sha)
        return
    from ilm.visual_lm.reference_image_history import trace_reference_history
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics
    attempts = execution/'attempts'; attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob('attempt_*.json')))+1
    monitor = GlyphReaderMonitor(execution, r['maximum_seconds'])
    monitor.pressure_wait_allowance_seconds = r['pressure_wait_allowance_seconds']
    monitor.status('admitting'); completed = []; hook = None; body_calls = 0
    try:
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            began = time.perf_counter(); actor, provenance = load_actor(CHECKPOINT, device)
            require(not any(p.requires_grad for p in actor.parameters()), 'Frozen actor required')
            versions = [(p, p._version) for p in actor.parameters()]
            torch.cuda.synchronize(); loading = time.perf_counter()-began
            def before_body(_m, _a):
                nonlocal body_calls
                body_calls += 1; monitor.guard()
            hook = actor.interface.actor.context.register_forward_pre_hook(before_body)
            data = torch.load(run/'prepared.pt', weights_only=True, map_location='cpu')
            for i, job in enumerate(r['jobs']):
                monitor.position = dict(completed=i, total=64, case=job['id'])
                monitor.guard(); monitor.status('reference_image_history')
                destination = execution/job['id']
                if destination.exists():
                    completed.append(verify_case(destination, job, registration_sha)); continue
                canvas, observed = prepare_prompt(ROOT/job['image'])
                prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None,None]/127.5-1
                gold = data['gold_ink'][job['reference_offset']:job['reference_offset']+job['reference_glyphs']]
                original = torch.load(ROOT/job['original_output'], weights_only=True, map_location='cpu')
                before = body_calls; torch.cuda.synchronize(); began = time.perf_counter()
                value = trace_reference_history(actor, prompt, 1-2*gold.float(), monitor.guard)
                require(torch.equal(value['states'][0], data['initial_states'][job['id']]), 'Initial state is not bit exact')
                require(value['observer_geometry'] == job['expected_geometry'], 'Observer geometry differs')
                require(body_calls-before == len(gold)+1, 'Incorrect correct-history contextual call count')
                replay = []; replay_stops = []
                for h in original['states']:
                    monitor.guard(); logits, semantic, stop = actor.bridge.predict(h[None].to(device))
                    replay.append((logits.sigmoid() >= .5).to(torch.uint8).cpu()[0])
                    replay_stops.append(float(stop.sigmoid()[0])); del logits, semantic, stop
                replay = torch.stack(replay)
                old_ink = ((1-original['images'])/2).to(torch.uint8)
                require(torch.equal(replay[:len(old_ink)], old_ink), 'Autonomous pixels not reproduced from saved states')
                require(replay_stops == original['stop_probabilities'], 'Autonomous stopping not reproduced')
                require(value['stop_probabilities'][0] == replay_stops[0] and
                        torch.equal(value['predicted_ink'][0], replay[0]), 'First writer prediction differs across histories')
                torch.cuda.synchronize(); elapsed = time.perf_counter()-began
                require(all(p._version == version for p, version in versions), 'Frozen parameter changed')
                metrics = [pixels_metrics(a.numpy(), b.numpy()) for a, b in zip(value['predicted_ink'], gold)]
                partial = execution/f'.{job["id"]}.{time.time_ns()}.partial'; partial.mkdir()
                atomic_tensor(partial/'trace.pt', value)
                np.savez_compressed(partial/'pixels.npz', gold_ink=gold.numpy(),
                    forced_ink=value['predicted_ink'].numpy(), autonomous_ink=old_ink.numpy(),
                    forced_stops=np.asarray(value['stop_probabilities'], dtype=np.float64),
                    autonomous_stops=np.asarray(replay_stops, dtype=np.float64))
                record = dict(job=job, registration_sha256=registration_sha,
                    trace_sha256=sha(partial/'trace.pt'), pixels_sha256=sha(partial/'pixels.npz'),
                    initial_state_bit_exact=True, autonomous_pixels_reproduced=True,
                    autonomous_stop_probabilities_reproduced=True, correct_history_body_calls=body_calls-before,
                    elapsed_seconds=elapsed, per_position_canonical_pixels=metrics,
                    premature_stop_positions=[j for j,p in enumerate(value['stop_probabilities'][:-1]) if p>=.5],
                    terminal_stop=value['stop_probabilities'][-1]>=.5,
                    reference_conditioned=True, autonomous_language_evidence=False, host_observation=observed)
                save_json(partial/'report.json', record); partial.rename(destination)
                completed.append(verify_case(destination, job, registration_sha))
                print(json.dumps(dict(case=job['id'], glyphs=len(gold), initial_bit_exact=True,
                    premature_stops=len(record['premature_stop_positions']), terminal_stop=record['terminal_stop'],
                    seconds=elapsed)), flush=True)
                del original, value, replay, old_ink, prompt; gc.collect()
            hook.remove(); hook = None
            save_json(execution/'report.json', dict(schema='reference-image-history-execution-v1',
                registration_sha256=registration_sha, records=completed, all64_complete=len(completed)==64,
                source_sha256=r['source_sha256'], checkpoint_sha256=CHECKPOINT_SHA,
                new_optimizer_updates=0, new_autonomous_generations=0, helper_native_calls=0,
                actual_contextual_calls_this_attempt=body_calls, all_parameter_versions_unchanged=True,
                loading_seconds=loading, provenance=provenance, **monitor.receipt()))
            monitor.status('complete')
    except BaseException as error:
        monitor.status('failed', error=repr(error)); raise
    finally:
        if hook is not None:hook.remove()
        status = load(execution/'status.json')
        save_json(attempts/f'attempt_{attempt:04}.json', status)
        save_json(execution/'status.json', dict(status, pid=None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['register','run'])
    p.add_argument('--run', type=Path, default=RUN)
    args = p.parse_args()
    (register if args.action == 'register' else run_model)(args.run)


if __name__ == '__main__':main()
