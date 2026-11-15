#!/usr/bin/env python3
"""Bind, measure and recount existing physical paths of the ranked writer.

No optimizer, glyph generation, new reading or autonomous language evaluation.
Registration and failed admission do not import the neural runtime.
"""
import time
BEGAN = time.perf_counter()

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess

from scripts import evaluate_global_glyph_ranking_language as actor_io
from scripts.analyze_joint_context_blank_modes import score_density

ROOT, ORIGINAL, TRAINING = actor_io.ROOT, actor_io.RUN, actor_io.TRAINING_PUBLIC
RESULT = ROOT / 'references/evidence/global_glyph_ranking_physical_result_20261115.json'
PUBLIC = ROOT / 'references/evidence/ranked_writer_whitespace_registration_20261115.json'
DEST = ORIGINAL / 'whitespace_density'
EXPORT = ROOT / 'references/evidence/ranked_writer_whitespace_result_20261115.json'
load, sha, require, write_new = actor_io.load, actor_io.sha, actor_io.require, actor_io.write_new
NEW_SOURCES = (
    'scripts/analyze_ranked_writer_whitespace.py', 'tests/test_ranked_writer_whitespace.py',
    'scripts/summarize_ranked_writer_whitespace.py',
    'references/global_glyph_ranking_whitespace_decision.md',
    'references/global_glyph_ranking_physical_result.md',
    'scripts/summarize_global_glyph_ranking_result.py',
)


def case_counts(source_blank, emitted_blank, blank_logp, emitted_logp, selected, credited):
    """One fixed density rule; source labels assess costs, never choose output."""
    import numpy as np
    vectors = [np.asarray(v) for v in [source_blank, emitted_blank, selected, credited]]
    source_blank, emitted_blank, selected, credited = vectors
    require(all(v.dtype == np.bool_ and v.shape == source_blank.shape for v in vectors)
            and source_blank.ndim == 1 and len(source_blank) > 0, 'Complete Boolean position vectors required')
    b, e = np.asarray(blank_logp), np.asarray(emitted_logp)
    require(b.shape == e.shape == source_blank.shape and np.isfinite(b).all() and np.isfinite(e).all(),
            'Complete finite image log probabilities required')
    require(not (credited & ~selected).any() and not (credited & emitted_blank).any(), 'Credit must be selected and contain ink')
    # An already blank image remains blank, even at a floating-point tie.
    switched = (b > e + .001) & ~emitted_blank
    after = emitted_blank | switched
    return dict(positions=len(b), source_whitespace=int(source_blank.sum()),
        original_false_ink=int((source_blank & ~emitted_blank).sum()),
        original_false_blanks=int((~source_blank & emitted_blank).sum()),
        switched_to_blank=int(switched.sum()), fixed_false_ink=int((source_blank & switched).sum()),
        added_false_blanks=int((~source_blank & switched).sum()),
        predicted_false_ink=int((source_blank & ~after).sum()),
        predicted_false_blanks=int((~source_blank & after).sum()),
        selected_positions=int(selected.sum()), original_readable=int(credited.sum()),
        lost_selected_credits=int((credited & switched).sum()),
        lost_nonempty_source_credits=int((credited & switched & ~source_blank).sum()),
        predicted_readable=int((credited & ~switched).sum()))


def decision(groups):
    require(len(groups) == 7, 'All seven original strata required')
    total = Counter()
    for row in groups.values():
        require(all(type(n) is int and n >= 0 for n in row.values()), 'Nonnegative integer counts required')
        require(row['selected_positions'] > 0 and row['predicted_readable'] <= row['original_readable'] <= row['selected_positions'],
                'Invalid selected readable counts')
        require(row['predicted_false_ink'] == row['original_false_ink'] - row['fixed_false_ink']
                and row['predicted_false_blanks'] == row['original_false_blanks'] + row['added_false_blanks']
                and row['predicted_readable'] == row['original_readable'] - row['lost_selected_credits'],
                'Counterfactual accounting differs')
        total.update(row)
    require([total[k] for k in ['positions', 'source_whitespace', 'original_false_ink', 'original_false_blanks',
                               'selected_positions', 'original_readable']] == [12597, 252, 25, 37, 1040, 1002],
            'Original ranked cohort changed')
    rules = dict(whitespace=total['predicted_false_ink'] <= 11,
        no_added_false_blanks=total['added_false_blanks'] == 0,
        credited_nonempty_sources_retained=total['lost_nonempty_source_credits'] == 0,
        all_seven_readable=all(r['predicted_readable'] / r['selected_positions'] >= .9 for r in groups.values()))
    return dict(rules=rules, separately_registered_decoding_comparison_eligible=all(rules.values()),
                totals=dict(total), old_gate_changed=False, autonomous_language_admitted=False)


def sources(training):
    result = actor_io.committed_sources(training)
    for name in NEW_SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit diagnostic implementation before binding: ' + name)
        result[name] = sha(ROOT / name)
    return result


def registration_payload():
    training, result = load(TRAINING), load(RESULT)
    require(result['registration_sha256'] == sha(TRAINING) and result['physical_gate_passed'] is False
            and result['readable_groups_passed'] and result['credited_candidate_ink_passed']
            and not result['whitespace_passed'], 'Completed ranked actor whitespace failure required')
    require(result['totals']['candidate']['false_ink'] == 25 and result['totals']['candidate']['false_blanks'] == 37,
            'Keep actual new failure counts')
    snapshot = load(ORIGINAL / 'fit/snapshots.json')['41298']
    bindings = dict(training['bindings_sha256'])
    def bind(path):
        name, digest = str(path.relative_to(ROOT)), sha(path)
        require(name not in bindings or bindings[name] == digest, 'Conflicting original input: ' + name)
        bindings[name] = digest
    for path in [TRAINING, RESULT, ROOT / snapshot['file']]:
        bind(path)
    for name in ['fit/report.json', 'fit/snapshots.json', 'audit/report.json', 'qualify/report.json',
                 'qualify/gate.json', 'qualify/frozen_readings.json', 'qualify/mapping.json',
                 'qualify/boards.json', 'qualify/review/session.json', 'qualify/review/freeze_receipt.json']:
        bind(ORIGINAL / name)
    report = load(ORIGINAL / 'qualify/report.json')
    for record in report['records']:
        bind(ROOT / record['job']['image'])
        path = ORIGINAL / 'qualify/cases' / record['job']['id'] / 'pixels.npz'
        require(sha(path) == record['pixels_sha256'], 'Fixed qualification pixels differ')
        bind(path)
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Bound original evidence changed: ' + name)
    return dict(schema='ranked-writer-whitespace-density-v1', source_sha256=sources(training),
        bindings_sha256=bindings, training_registration_sha256=sha(TRAINING), snapshot=snapshot,
        qualification_jobs=training['qualification_jobs'], model_account=training['model_account'],
        runtime=training['runtime'], external_program_sha256=training['external_program_sha256'],
        cases=124, positions=12597, source_whitespace=252, original_false_ink=25, original_false_blanks=37,
        numerical_margin_nats=.001, source_nll_mean_max_error=2e-5, required_saved_argmax_disagreements=0,
        maximum_seconds=900, external_term_seconds=880, external_kill_grace_seconds=20,
        pressure_wait_seconds=120, minimum_entry_ram_gib=48, minimum_free_vram_gib=20,
        minimum_runtime_ram_gib=24, maximum_allocated_vram_gib=22, minimum_free_disk_gib=48,
        maximum_output_mib=256, cpu_threads=2, batch_glyphs=64, automatic_retry=False,
        source_history_supplied=True, all131_actual_tensors_required=True, generated_glyphs=0,
        optimizer_updates=0, autonomous_answers=0, original_frozen_readings_and_gate_immutable=True,
        counterfactual_rule='blank iff blank_logp > saved_image_logp + 0.001; otherwise retain saved pixels',
        fixed_eligibility='falseink<=11/252; no added falseblanks over37; preserve every credited nonempty-source form; all7 predicted readability fractions>=.90',
        action='Eligibility permits only a separately registered physical decoding comparison with new actual review. Rejection redirects whitespace learning after a fresh reset preflight. Neither admits language generation.')


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration required')
    require(not PUBLIC.exists() and not DEST.exists(), 'Preserve first diagnostic binding')
    value = registration_payload()
    write_new(DEST / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), cases=124, positions=12597,
        bound_sources=len(value['source_sha256']), bound_inputs=len(value['bindings_sha256']), new_neural_calls=0)


def validate():
    require(PUBLIC.exists() and (DEST / 'registration.json').exists(), 'Bind diagnostic before execution')
    require(PUBLIC.read_bytes() == (DEST / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact diagnostic binding first')
    value = load(PUBLIC)
    require(value == registration_payload(), 'Diagnostic source, evidence or fixed rule changed')
    return value


def selected_flags(job, n, mapping, readings):
    import numpy as np
    selected, credited = np.zeros(n, bool), np.zeros(n, bool)
    for key, owner in mapping.items():
        if owner['role'] != 'candidate' or owner['case'] != job['id']:
            continue
        pos = owner['position']; require(0 <= pos < n and not selected[pos], 'Unique original selected position required')
        selected[pos] = True
        row = readings[key]
        credited[pos] = bool(row['text'].strip()) and not row['blank'] and not row['uncertain']
    return selected, credited


def ensure_first_execution():
    require(not (DEST / 'status.json').exists() and not (DEST / 'attempts').exists(),
            'No automatic retry; preserve the first diagnostic attempt')


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'GPU0 and deterministic CUBLAS required')
    value = validate(); ensure_first_execution()
    with (DEST / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); ensure_first_execution()
        identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())
        write_new(DEST / 'attempts/start.json', dict(identity, registration_sha256=sha(PUBLIC),
            prelaunch_validation_seconds=time.perf_counter() - BEGAN))
        hooks, monitor, error = [], None, None
        try:
            import numpy as np
            import torch
            from ilm.visual_lm.glyph_joint_density import pack_glyphs
            from ilm.visual_lm.joint_context_model import build_joint_context_actor
            from ilm.visual_lm.joint_context_training import require_process_runtime
            from ilm.visual_lm.joint_writer_training import Monitor, PARENT
            from ilm.visual_lm.continuation_training import TrainingImages
            from ilm.visual_lm.visual_continuation import prefill_visual_prompt
            from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
            from scripts.prepare_global_glyph_ranking import validate as validate_training
            from scripts.recover_visual_continuation_evaluation import resources, admitted
            from scripts.train_semantic_glyph_language import numerics
            from scripts.infer_glyph_tail_language import prepare_prompt
            training = validate_training(); require_process_runtime(training)
            monitor = Monitor(DEST, value['maximum_seconds'], value['pressure_wait_seconds'], value['maximum_allocated_vram_gib'])
            monitor.began = BEGAN; monitor.status('admitting')
            while True:
                observed = resources()
                if admitted(observed, 48, 20 * 1024):
                    break
                remaining = min(120 - monitor.waited, 900 - (time.perf_counter() - BEGAN))
                require(remaining > 0, 'Admission/pressure budget exhausted')
                monitor.status('waiting_for_admission', resources=observed)
                start = time.perf_counter(); time.sleep(min(10., remaining)); monitor.waited += time.perf_counter() - start
            with full_reader_gpu() as (device, admission), torch.no_grad():
                monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
                actor, provenance, account = build_joint_context_actor(device)
                require(account == value['model_account'], 'Actual complete model differs')
                saved = torch.load(ROOT / value['snapshot']['file'], weights_only=True, map_location='cpu')
                restored = actor_io.restore_snapshot(actor, saved, 41298, training); del saved
                versions = [(p, p._version) for p in [*actor.parameters(), *actor.buffers()]]
                hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
                data = TrainingImages(PARENT)
                records = {r['job']['id']: r for r in load(ORIGINAL / 'qualify/report.json')['records']}
                mapping = load(ORIGINAL / 'qualify/mapping.json')['rows']
                readings = load(ORIGINAL / 'qualify/frozen_readings.json')['readings']
                details, groups, max_nll_error = [], {}, 0.
                for i, job in enumerate(value['qualification_jobs']):
                    monitor.position = dict(completed=i, total=124, case=job['id']); monitor.status('scoring_saved_images'); monitor.guard()
                    record = records[job['id']]
                    with np.load(ORIGINAL / 'qualify/cases' / job['id'] / 'pixels.npz', allow_pickle=False) as arrays:
                        gold = torch.from_numpy(arrays['gold_ink'].copy())
                        emitted = pack_glyphs(torch.from_numpy(arrays['candidate_ink'].copy()))
                    if job['split'] == 'train':
                        prompt, target, _, layout = data.get(int(job['id'].split('_')[1]), int(job['layout'] == 'proportional'))
                        require(layout == job['layout'] and torch.equal(gold, target), 'Complete source differs')
                    else:
                        canvas, _ = prepare_prompt(ROOT / job['image'])
                        prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None, None] / 127.5 - 1
                    memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
                    require(geometry == record['geometry'], 'Original prompt geometry differs')
                    feedback = actor.bridge.feedback_embedding((1 - 2 * gold.float()).to(device))[None]
                    states = actor.dynamics(actor.dynamics.embeddings(feedback), memory, checkpoint_layers=False)[0].float()
                    require(states.shape == (len(gold) + 1, 4096), 'Actual target-preceding states required')
                    patterns = pack_glyphs(gold); batches = []
                    for start in range(0, len(gold), 64):
                        monitor.guard(); end = min(start + 64, len(gold))
                        batches.append(score_density(actor.bridge.density, states[start:end], patterns[start:end].to(device), emitted[start:end].to(device)))
                    output = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
                    require(not output['saved_argmax_disagreements'].any(), 'Saved greedy image does not reproduce exactly')
                    error_nll = abs(float((-output['source_logp'] / 128).mean()) - record['pattern_nll_mean'])
                    require(error_nll <= value['source_nll_mean_max_error'], 'Source NLL differs from qualification')
                    max_nll_error = max(max_nll_error, error_nll)
                    white, empty = (patterns.numpy() == 0).all(1), (emitted.numpy() == 0).all(1)
                    require(np.allclose(output['blank_logp'][empty], output['emitted_logp'][empty], atol=1e-5, rtol=0), 'Identical blank paths differ')
                    selected, credited = selected_flags(job, len(gold), mapping, readings)
                    count = case_counts(white, empty, output['blank_logp'], output['emitted_logp'], selected, credited)
                    certified = output['blank_logp'] > output['nonempty_upper_bound'] + .001
                    count.update(certified_blank_global_mode=int(certified.sum()),
                        certified_blank_global_mode_source_white=int((certified & white).sum()),
                        certified_blank_global_mode_false_ink=int((certified & white & ~empty).sum()))
                    key = '/'.join(job[k] for k in ['split', 'kind', 'layout']); groups.setdefault(key, Counter()).update(count)
                    folder = DEST / 'cases' / job['id']; folder.mkdir(parents=True)
                    path = folder / 'probabilities.npz'
                    np.savez_compressed(path, **output, source_blank=white, emitted_blank=empty)
                    details.append(dict(case=job['id'], positions=len(gold), file=str(path.relative_to(ROOT)),
                        sha256=sha(path), source_nll_mean_error=error_nll))
                    require(sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()) <= 256 * 1024**2, 'Diagnostic output ceiling reached')
                    print(json.dumps(dict(completed=i + 1, total=124, case=job['id'])), flush=True)
                    del states, memory, feedback, prompt, gold, emitted, patterns, batches
                require(len(details) == 124 and all(p._version == v for p, v in versions), 'Incomplete probe or mutated weights')
                result = dict(complete=True, registration_sha256=sha(PUBLIC), cases=details,
                    groups={k: dict(v) for k, v in groups.items()}, decision=decision(groups),
                    actual_restored_inventory=restored, all131_actual_parameters_restored=True, model_account=account,
                    provenance=provenance, maximum_source_nll_mean_error=max_nll_error, actual_context_recomputed=True,
                    stale_predictor_cache_used=False, model_tensors_unchanged=True, generated_glyphs=0,
                    new_optimizer_updates=0, new_autonomous_streams=0, original_gate_changed=False,
                    full_language_unmeasured=True, goal_achieved=False, **monitor.receipt())
                write_new(DEST / 'report.json', result); monitor.status('complete')
        except BaseException as caught:
            error = caught
            if monitor is not None:
                monitor.status('failed', error=repr(caught))
            raise
        finally:
            for hook in hooks:
                hook.remove()
            cost = monitor.receipt() if monitor else dict(elapsed_seconds=time.perf_counter() - BEGAN, resource_wait_seconds=0.)
            write_new(DEST / 'attempts/end.json', dict(identity, error=repr(error) if error else None,
                complete=error is None and (DEST / 'report.json').exists(), **cost))
            if monitor is not None:
                from scripts.train_visual_continuation import atomic_json
                atomic_json(DEST / 'status.json', dict(load(DEST / 'status.json'), pid=None))
    return dict(complete=True, decision=result['decision'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action', choices=['register', 'validate', 'run'])
    args = p.parse_args()
    print(json.dumps(register() if args.action == 'register' else dict(valid=bool(validate())) if args.action == 'validate' else run()))
