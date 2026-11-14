#!/usr/bin/env python3
"""Score existing images on the final learned-context actor; never train or write.

Registration is CPU-only and imports no neural runtime. The failed actor's
autonomous evaluator stays closed. This is a separately bound probability probe.
"""
import time
BEGAN = time.perf_counter()

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess

from scripts import evaluate_joint_context_language as context
from scripts.analyze_joint_writer_blank_density import blank_path_bounds

ROOT = context.ROOT
ORIGINAL = context.RUN
TRAINING_PUBLIC = context.TRAINING_PUBLIC
RESULT = ROOT / 'references/evidence/joint_context_physical_result_20261114.json'
PUBLIC = ROOT / 'references/evidence/joint_context_blank_modes_registration_20261114.json'
DEST = ORIGINAL / 'blank_modes'
load, sha, require, write_new = context.load, context.sha, context.require, context.write_new
NEW_SOURCES = ('scripts/analyze_joint_context_blank_modes.py', 'tests/test_joint_context_blank_modes.py',
    'references/joint_context_blank_modes_protocol.md', 'references/joint_context_failure_decision.md',
    'scripts/analyze_joint_writer_blank_density.py', 'tests/test_joint_writer_blank_density.py',
    'scripts/summarize_joint_context_result.py')
BOOK_RULES = {'development/book/fixed_cells': (93, 47),
              'train/book/fixed_cells': (307, 154), 'train/book/proportional': (289, 145)}


def decision(groups):
    """Require every prospective stratum; a pooled majority is insufficient."""
    rules = {}
    for key, (count, minimum) in BOOK_RULES.items():
        require(key in groups and groups[key]['false_blank'] == count, 'Original book false-blank cohort changed')
        observed = groups[key]['source_preferred_false_blank']
        require(type(observed) is int and 0 <= observed <= count, 'Invalid source-preference count')
        rules[key] = dict(false_blanks=count, source_preferred=observed, required=minimum, passed=observed >= minimum)
    return dict(groups=rules, search_comparison_eligible=all(row['passed'] for row in rules.values()))


def sources(training):
    result = context.committed_sources(training)
    for name in NEW_SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit probe source before registration: ' + name)
        result[name] = sha(ROOT / name)
    return result


def registration_payload():
    training = load(TRAINING_PUBLIC); result = load(RESULT)
    require(result['registration_sha256'] == sha(TRAINING_PUBLIC) and not result['physical_gate_passed'],
            'Diagnose the completed failed actor only')
    require(result['qualification_cases'] == 124 and result['reviewed_images'] == 2080
            and result['review']['readings_frozen_before_owner_unmask'], 'Complete frozen result required')
    groups = {'/'.join(row[k] for k in ['split', 'kind', 'layout']): row
              for row in result['complete_supplied_history_pixel_diagnostics']}
    require(sum(int(row['candidate_false_blank']) for row in groups.values()) == 711
            and sum(int(row['positions']) for row in groups.values()) == 12597, 'Keep every physical position')
    require(all(groups[k]['candidate_false_blank'] == n for k, (n, _) in BOOK_RULES.items()),
            'Fixed book failure strata differ')
    snapshot = load(ORIGINAL / 'fit/snapshots.json')['41298']
    bindings = dict(training['bindings_sha256'])
    bindings[str(TRAINING_PUBLIC.relative_to(ROOT))] = sha(TRAINING_PUBLIC)
    bindings[str(RESULT.relative_to(ROOT))] = sha(RESULT)
    for name, digest in result['input_sha256'].items():
        bindings[str((ORIGINAL / name).relative_to(ROOT))] = digest
    bindings[snapshot['file']] = snapshot['sha256']
    qualification = load(ORIGINAL / 'qualify/report.json')
    for record in qualification['records']:
        job = record['job']; path = ORIGINAL / 'qualify/cases' / job['id'] / 'pixels.npz'
        bindings[str(path.relative_to(ROOT))] = record['pixels_sha256']
        path = ROOT / job['image']; bindings[job['image']] = sha(path)
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Bound evidence changed: ' + name)
    return dict(schema='joint-context-blank-modes-probe-v1', source_sha256=sources(training),
        bindings_sha256=bindings, training_registration_sha256=sha(TRAINING_PUBLIC), snapshot=snapshot,
        qualification_jobs=training['qualification_jobs'], model_account=training['model_account'],
        runtime=training['runtime'], external_program_sha256=training['external_program_sha256'],
        cases=124, positions=12597, false_blanks=711, book_false_blanks=689,
        book_rules={k: dict(false_blanks=n, required=m) for k, (n, m) in BOOK_RULES.items()},
        numerical_margin_nats=.001, source_nll_mean_max_error=2e-5,
        maximum_saved_greedy_logp_deficit_nats=1e-4, batch_glyphs=64,
        maximum_seconds=1200, external_term_seconds=1180, external_kill_grace_seconds=20,
        pressure_wait_seconds=180, minimum_entry_ram_gib=48,
        admission_headroom_gib=2, minimum_free_vram_gib=20, minimum_runtime_ram_gib=24,
        maximum_allocated_vram_gib=22, minimum_free_disk_gib=48, cpu_threads=2,
        all131_context_and_head_tensors_restored=True, cached_predictor_states_used=False,
        generated_glyphs=0, new_optimizer_updates=0, new_autonomous_streams=0,
        original_frozen_readings_and_scores_immutable=True,
        action='All three source-preference thresholds passing permits considering a separately registered whole-glyph decoding comparison. Otherwise redirect the conditional density/representation objective. No result authorizes retrying the old fit or autonomous evaluation of the failed actor.')


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration required')
    require(not PUBLIC.exists() and not DEST.exists(), 'Preserve the first probe registration')
    value = registration_payload()
    write_new(DEST / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), cases=value['cases'], positions=value['positions'],
                bound_sources=len(value['source_sha256']), new_model_calls=0)


def validate():
    require(PUBLIC.exists() and (DEST / 'registration.json').exists(), 'Register probe before execution')
    require(PUBLIC.read_bytes() == (DEST / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact probe binding first')
    value = load(PUBLIC)
    require(value == registration_payload(), 'Probe source, inputs or fixed rule changed')
    return value


def score_density(model, states, source, emitted):
    """Teacher-force only the existing physical paths, with double log sums."""
    import numpy as np
    import torch
    require(source.shape == emitted.shape == (len(states), 128), 'Complete physical patterns required')
    zero = torch.zeros_like(source)
    blank_lp = model(states, zero).float().log_softmax(-1)
    blank, upper = blank_path_bounds(blank_lp.detach().cpu().numpy())
    result = dict(blank_logp=blank, nonempty_upper_bound=upper,
        blank_argmax_disagreements=(blank_lp.argmax(-1) != 0).sum(1).cpu().numpy())
    for name, path in [('source_logp', source), ('emitted_logp', emitted)]:
        lp = model(states, path).float().log_softmax(-1)
        chosen = lp.gather(-1, path[..., None]).squeeze(-1)
        result[name] = chosen.double().sum(1).cpu().numpy()
        if name == 'emitted_logp':
            result['saved_argmax_disagreements'] = (lp.argmax(-1) != path).sum(1).cpu().numpy()
            result['saved_greedy_logp_deficit'] = (lp.max(-1).values - chosen).max(1).values.cpu().numpy()
    require(all(np.isfinite(v).all() for v in result.values()), 'Nonfinite physical path scores')
    return result


def ensure_first_execution():
    require(not (DEST / 'status.json').exists() and not (DEST / 'attempts').exists(),
            'No automatic probe retry; preserve and reconcile the first attempt')


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'GPU0 and deterministic CUBLAS required')
    value = validate(); ensure_first_execution()
    with (DEST / 'owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_first_execution()
        identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())
        write_new(DEST / 'attempts/start.json', dict(identity, registration_sha256=sha(PUBLIC),
            prelaunch_validation_seconds=time.perf_counter() - BEGAN))
        hooks = []; monitor = None; error = None
        def interrupted(signum, _frame):
            raise KeyboardInterrupt('Probe interrupted by signal ' + str(signum))
        for signum in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
            signal.signal(signum, interrupted)
        try:
            import numpy as np
            import torch
            from PIL import Image
            from ilm.visual_lm.glyph_joint_density import pack_glyphs
            from ilm.visual_lm.joint_context_model import build_joint_context_actor
            from ilm.visual_lm.joint_context_training import require_process_runtime
            from ilm.visual_lm.joint_writer_training import Monitor, PARENT
            from ilm.visual_lm.continuation_training import TrainingImages
            from ilm.visual_lm.visual_continuation import prefill_visual_prompt
            from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
            from scripts.prepare_joint_context_training import validate as validate_training
            from scripts.recover_visual_continuation_evaluation import resources, admitted
            from scripts.train_semantic_glyph_language import numerics
            from scripts.train_visual_continuation import atomic_json
            from scripts.infer_glyph_tail_language import prepare_prompt
            training_reg = validate_training(); require_process_runtime(training_reg)
            monitor = Monitor(DEST, value['maximum_seconds'], value['pressure_wait_seconds'], value['maximum_allocated_vram_gib'])
            monitor.began = BEGAN; monitor.status('admitting')
            while True:
                observed = resources()
                if admitted(observed, value['minimum_entry_ram_gib'], value['minimum_free_vram_gib'] * 1024):
                    break
                remaining = min(value['pressure_wait_seconds'] - monitor.waited,
                                value['maximum_seconds'] - (time.perf_counter() - BEGAN))
                require(remaining > 0, 'Admission/pressure budget exhausted')
                monitor.status('waiting_for_admission', resources=observed)
                start = time.perf_counter(); time.sleep(min(10., remaining)); monitor.waited += time.perf_counter() - start
            with full_reader_gpu() as (device, admission), torch.no_grad():
                monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
                actor, provenance, account = build_joint_context_actor(device)
                require(account == value['model_account'], 'Actual complete model differs')
                saved = torch.load(ROOT / value['snapshot']['file'], weights_only=True, map_location='cpu')
                restored = context.restore_snapshot(actor, saved, 41298, training_reg); del saved
                versions = [(p, p._version) for p in [*actor.parameters(), *actor.buffers()]]
                hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
                data = TrainingImages(PARENT); qualification = load(ORIGINAL / 'qualify/report.json')
                records = {row['job']['id']: row for row in qualification['records']}
                detail = []; groups = {}; maximum_nll_error = 0.; total = 0
                for i, job in enumerate(value['qualification_jobs']):
                    monitor.position = dict(completed=i, total=124, case=job['id']); monitor.status('scoring_existing_physical_paths'); monitor.guard()
                    record = records[job['id']]; p = ORIGINAL / 'qualify/cases' / job['id'] / 'pixels.npz'
                    require(sha(p) == record['pixels_sha256'], 'Original case changed')
                    with np.load(p, allow_pickle=False) as arrays:
                        gold = torch.from_numpy(arrays['gold_ink'].copy()); emitted = pack_glyphs(torch.from_numpy(arrays['candidate_ink'].copy()))
                    if job['split'] == 'train':
                        prompt, target, _, layout = data.get(int(job['id'].split('_')[1]), int(job['layout'] == 'proportional'))
                        require(layout == job['layout'] and torch.equal(gold, target), 'Complete training source differs')
                    else:
                        canvas, _ = prepare_prompt(ROOT / job['image'])
                        prompt = torch.from_numpy(np.asarray(canvas).copy()).float()[None, None] / 127.5 - 1
                    memory, _, geometry = prefill_visual_prompt(actor.interface, prompt)
                    require(geometry == record['geometry'], 'Prompt geometry differs from original qualification')
                    feedback = actor.bridge.feedback_embedding((1 - 2 * gold.float()).to(device))[None]
                    states = actor.dynamics(actor.dynamics.embeddings(feedback), memory, checkpoint_layers=False)[0].float()
                    require(states.shape == (len(gold) + 1, 4096), 'Actual complete learned-context states required')
                    patterns = pack_glyphs(gold); batches = []
                    for start in range(0, len(gold), value['batch_glyphs']):
                        monitor.guard(); end = min(start + value['batch_glyphs'], len(gold))
                        batches.append(score_density(actor.bridge.density, states[start:end], patterns[start:end].to(device), emitted[start:end].to(device)))
                    output = {k: np.concatenate([row[k] for row in batches]) for k in batches[0]}
                    source_blank = (patterns.numpy() == 0).all(1); emitted_blank = (emitted.numpy() == 0).all(1)
                    false_blank = emitted_blank & ~source_blank
                    require(np.all(output['blank_argmax_disagreements'][emitted_blank] == 0), 'Saved blank is inconsistent with restored physical argmax')
                    require(float(output['saved_greedy_logp_deficit'].max()) <= value['maximum_saved_greedy_logp_deficit_nats'], 'Saved physical greedy path differs materially')
                    require(np.allclose(output['emitted_logp'][emitted_blank], output['blank_logp'][emitted_blank], atol=1e-5, rtol=0), 'Identical blank paths differ')
                    nll_error = abs(float((-output['source_logp'] / 128).mean()) - record['pattern_nll_mean'])
                    maximum_nll_error = max(maximum_nll_error, nll_error)
                    require(nll_error <= value['source_nll_mean_max_error'], 'Restored source likelihood differs from qualification')
                    better = output['source_logp'] >= output['blank_logp'] + value['numerical_margin_nats']
                    certified = output['nonempty_upper_bound'] < output['blank_logp'] - value['numerical_margin_nats']
                    require(not (false_blank & better & certified).any(), 'Source preference contradicts blank global-mode bound')
                    key = '/'.join(job[k] for k in ['split', 'kind', 'layout']); group = groups.setdefault(key, Counter())
                    group.update(positions=len(gold), source_blank=int(source_blank.sum()), emitted_blank=int(emitted_blank.sum()),
                        false_blank=int(false_blank.sum()), source_preferred_false_blank=int((false_blank & better).sum()),
                        certified_blank_global_mode_false_blank=int((false_blank & certified).sum()),
                        saved_blank_argmax_disagreements=int(output['blank_argmax_disagreements'][emitted_blank].sum()),
                        all_saved_path_argmax_disagreements=int(output['saved_argmax_disagreements'].sum()))
                    folder = DEST / 'cases' / job['id']; folder.mkdir(parents=True)
                    np.savez_compressed(folder / 'probabilities.npz', **output, source_blank=source_blank, emitted_blank=emitted_blank)
                    detail.append(dict(case=job['id'], positions=len(gold), file=str((folder / 'probabilities.npz').relative_to(ROOT)),
                        sha256=sha(folder / 'probabilities.npz'), source_nll_mean_error=nll_error))
                    total += len(gold)
                    print(json.dumps(dict(completed=i + 1, total=124, case=job['id'], positions=len(gold))), flush=True)
                    del states, memory, feedback, prompt, gold, emitted, patterns, batches
                require(len(detail) == 124 and total == 12597 and sum(x['false_blank'] for x in groups.values()) == 711, 'Complete original failure cohort required')
                require(all(p._version == version for p, version in versions), 'Probability probe mutated model tensors')
                result = dict(complete=True, registration_sha256=sha(PUBLIC), cases=detail, positions=total,
                    groups={k: dict(v) for k, v in groups.items()}, decision=decision(groups),
                    actual_restored_inventory=restored, all131_actual_parameters_restored=True, model_account=account,
                    provenance=provenance, maximum_source_nll_mean_error=maximum_nll_error,
                    actual_context_recomputed=True, stale_predictor_cache_used=False, model_tensors_unchanged=True,
                    reader_calls=124, generated_glyphs=0, new_optimizer_updates=0, new_autonomous_streams=0,
                    original_readings_and_scores_changed=False, full_language_unmeasured=True, goal_achieved=False,
                    **monitor.receipt())
                write_new(DEST / 'report.json', result); monitor.status('complete')
        except BaseException as caught:
            error = caught
            if monitor is not None:
                monitor.status('failed', error=repr(caught))
            raise
        finally:
            for hook in hooks:
                hook.remove()
            cost = monitor.receipt() if monitor else dict(elapsed_seconds=time.perf_counter() - BEGAN,
                resource_wait_seconds=0., peak_allocated_vram_gib=None)
            write_new(DEST / 'attempts/end.json', dict(identity, error=repr(error) if error else None,
                complete=error is None and (DEST / 'report.json').exists(), **cost))
            if monitor is not None:
                from scripts.train_visual_continuation import atomic_json
                atomic_json(DEST / 'status.json', dict(load(DEST / 'status.json'), pid=None))
    return dict(complete=True, positions=total, decision=result['decision'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['register', 'validate', 'run'])
    args = parser.parse_args()
    result = register() if args.action == 'register' else dict(valid=bool(validate())) if args.action == 'validate' else run()
    print(json.dumps(result))
