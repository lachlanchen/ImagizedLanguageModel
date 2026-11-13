#!/usr/bin/env python3
"""Gate, bind and run the complete joint-context autonomous comparison.

Planning and failed-gate admission import no Torch and create no generation
directory. Successful generation requires a separate committed post-gate binding.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from scripts import review_joint_writer_physical as physical
from scripts.evaluate_joint_writer_language import plan, remaining_budget, NEW_SOURCES as ORIGINAL_REVIEW_SOURCES

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/joint_context_learning_v1'
TRAINING_PUBLIC = ROOT / 'references/evidence/joint_context_learning_registration_20261114.json'
PUBLIC = ROOT / 'references/evidence/joint_context_language_registration_v1.json'
DEST = RUN / 'language'
load, sha, require, write_new = physical.load, physical.sha, physical.require, physical.write_new
NEW_SOURCES = ('scripts/evaluate_joint_context_language.py', 'scripts/review_joint_context_language.py',
               'tests/test_joint_context_language_evaluation.py', 'references/joint_context_language_execution.md')


@contextmanager
def review_scope():
    """Temporarily scope unchanged physical-review functions; no import-time mutation."""
    old = physical.RUN, physical.PUBLIC
    physical.RUN, physical.PUBLIC = RUN, TRAINING_PUBLIC
    try:
        yield physical
    finally:
        physical.RUN, physical.PUBLIC = old


def ink_flag(reading, ink_pixels):
    require(type(ink_pixels) is int and 0 <= ink_pixels <= 1024, 'Physical ink count required')
    credited = bool(reading['text'].strip()) and not reading['uncertain'] and not reading['blank']
    actual_blank = ink_pixels == 0
    return dict(actual_blank=actual_blank, credited=credited,
                blank_flag_mismatch=actual_blank != reading['blank'], credited_empty=credited and actual_blank)


def ink_audit(qualify, mapping, frozen, bindings, guard):
    """Check original pixels after the sealed review; never repair a reading."""
    from PIL import Image
    errors = []; candidate_empty = control_empty = 0
    for key, owner in mapping.items():
        guard(); path = qualify / 'pages' / (key + '.png')
        require(sha(path) == owner['image_sha256'] == frozen['image_sha256'][key], 'Reviewed physical page changed')
        with Image.open(path) as image:
            require(image.mode == 'L' and image.size == (32, 32), 'Original32x32gray image required')
            values = list(image.getdata())
        require(all(v in (0, 255) for v in values), 'Original binary physical pixels required')
        result = ink_flag(frozen['readings'][key], sum(v == 0 for v in values))
        if result['credited_empty']:
            require(owner['role'] in {'candidate', 'parent'}, 'Unexpected physical actor')
            candidate_empty += int(owner['role'] == 'candidate')
            control_empty += int(owner['role'] == 'parent')
        if result['blank_flag_mismatch']:
            errors.append(dict(id=key, role=owner['role'], case=owner['case'], position=owner['position'],
                original_reading=frozen['readings'][key], image_sha256=owner['image_sha256'], **result))
    return dict(schema='joint-context-credited-ink-audit-v1', bindings=bindings, images=len(mapping),
        candidate_credited_empty=candidate_empty, control_credited_empty=control_empty,
        disagreements=errors, blank_flag_mismatches=len(errors),
        candidate_credits_contain_ink=candidate_empty == 0, original_readings_changed=False,
        original_scores_changed=False, semantic_regrading=False, new_model_calls=0, new_optimizer_updates=0)


def physical_gate(*, persist_audit=False):
    # No neural import or generation output directory precedes this decision.
    path = RUN / 'qualify/gate.json'
    require(path.exists(), 'Physical writing review is not yet complete')
    gate = load(path)
    require(gate.get('passed') is True, 'Physical gate failed; preserve this actor without language generation')
    began = time.perf_counter()
    def guard():
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(mem['MemAvailable'].split()[0]) >= 24 * 1024**2, 'Retain24GiBhostRAM')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2, 'CPU gate RSS ceiling reached')
        require(time.perf_counter() - began <= 900, 'CPU gate audit ceiling reached')
    with review_scope() as review:
        values, session = review.session_context()
        qualify, registration, report, manifest, ids, bindings = values
        frozen = review.read_frozen_bundle(qualify, manifest, ids, bindings)
        require(gate.get('schema') == 'joint-writer-physical-gate-v2' and gate['bindings'] == bindings,
                'Source-bound freeze-before-unmask physical review required')
        require(gate['frozen_readings_sha256'] == sha(qualify / 'frozen_readings.json')
                and gate['session_sha256'] == sha(qualify / 'review/session.json'), 'Frozen review changed')
        require(gate['complete_registered_positions_recounted'] and gate['original_page_and_board_pixels_verified']
                and gate['readings_frozen_before_workflow_owner_map_read'], 'Complete physical provenance required')
        mapping = load(qualify / 'mapping.json')
        require(sha(qualify / 'mapping.json') == bindings['mapping_sha256'], 'Owner mapping changed')
        groups, passed = review.aggregate(registration['qualification_jobs'], mapping['rows'], frozen['readings'],
            bindings['registration_sha256'], registration['physical_gate_readable_nonblank_fraction'])
        require(passed and gate['groups'] == groups, 'Physical group recount differs')
        review.verify_pixels(qualify, report, manifest, mapping['rows'], frozen, guard)
        audit = ink_audit(qualify, mapping['rows'], frozen, bindings, guard)
    audit_path = RUN / 'qualify/credited_ink_audit.json'
    if persist_audit and not audit_path.exists():
        write_new(audit_path, audit)  # Preserve a failed ink audit too.
    if audit_path.exists():
        require(load(audit_path) == audit, 'Preserve original ink audit; do not rewrite readings')
    require(audit['candidate_credits_contain_ink'], 'Credited candidate contains an empty image; preserve contradictory frozen review')
    return registration, session, audit


def committed_sources(registration):
    result = dict(registration['source_sha256'])
    for name in sorted(set(NEW_SOURCES) | set(ORIGINAL_REVIEW_SOURCES)):
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit evaluation implementation before binding: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Registered source changed: ' + name)
    return result


def check_snapshot(saved, step, registration):
    from ilm.visual_lm.joint_context_training import check_checkpoint
    check_checkpoint(saved, registration, step)
    require(saved['parent_checkpoint'] == registration['parent_checkpoint']
            and saved['normalization_sha256'] == registration['bindings_sha256'][
                'artifacts/continuation_joint_writer_v1/cache/normalization.pt']
            and saved['only_context_and_density_trainable'] is True, 'Wrong context/head initialization provenance')


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only post-gate registration required')
    registration, session, audit = physical_gate(persist_audit=True)
    require(not DEST.exists() and not PUBLIC.exists(), 'Preserve existing evaluation preparation')
    arms = plan(registration); sources = committed_sources(registration)
    # Neural-runtime import occurs only after the complete physical/ink gate.
    import torch
    torch.set_num_threads(2); require(not torch.cuda.is_initialized(), 'CPU preparation must not initialize CUDA')
    bindings = dict(registration['bindings_sha256'])
    def bind(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Original input/control changed: ' + name)
    for name in ['registration.json', 'fit/report.json', 'fit/snapshots.json', 'audit/report.json',
                 'qualify/report.json', 'qualify/boards.json', 'qualify/mapping.json',
                 'qualify/frozen_readings.json', 'qualify/gate.json', 'qualify/credited_ink_audit.json',
                 'qualify/review/session.json', 'qualify/review/freeze_receipt.json']:
        bind(RUN / name)
    fit = load(RUN / 'fit/report.json'); snapshots = load(RUN / 'fit/snapshots.json')
    require(fit['complete'] and fit['snapshots'] == snapshots and sorted(map(int, snapshots)) == registration['snapshot_steps'],
            'Every actual complete context/head snapshot required')
    for key, item in snapshots.items():
        path = ROOT / item['file']; require(sha(path) == item['sha256'] and item['step'] == int(key), 'Snapshot changed')
        saved = torch.load(path, weights_only=True, map_location='cpu'); check_snapshot(saved, int(key), registration)
        bind(path); del saved
    for row in registration['evaluation_inputs']:
        require(sha(ROOT / row['image']) == row['sha256'], 'Fixed evaluation prompt changed')
        bindings[row['image']] = row['sha256']
    for directory in [ROOT / 'artifacts/compact_visual_span_evaluation_v1',
                      ROOT / 'artifacts/glyph_tail_input_presentation_v1/evaluation']:
        preparation = load(directory / 'preparation.json')
        require(sha(directory / 'inputs/manifest.json') == preparation['input_manifest_sha256']
                and sha(directory / 'references.json') == preparation['references_sha256'], 'Original scoring material changed')
        # Hash reference files without opening their answer content during preparation.
        for name in ['preparation.json', 'inputs/manifest.json', 'references.json']:
            bind(directory / name)
    value = dict(schema='joint-context-autonomous-evaluation-v1', source_sha256=sources, bindings_sha256=bindings,
        joint_registration_sha256=sha(TRAINING_PUBLIC), runtime=registration['runtime'],
        parent_checkpoint=registration['parent_checkpoint'], snapshots=snapshots, arms=arms,
        model_account=fit['model_account'], total_candidate_streams=708, total_core_streams=660, total_long_book_streams=48,
        all131_context_and_head_tensors_restored_per_snapshot=True, greedy_physical_masks=True,
        actual_emitted_pixel_feedback=True, stop_probability_threshold=.5, reference_inputs=False,
        reference_output_lengths=False, live_teacher=False, lexical_output_head=False, font_lookup=False, OCR=False,
        maximum_seconds=registration['generation_seconds'], pressure_wait_seconds=1800,
        minimum_entry_ram_gib=48, prelaunch_extra_ram_gib=2, minimum_runtime_ram_gib=24,
        minimum_free_vram_gib=20, maximum_allocated_vram_gib=22, minimum_free_disk_gib=48,
        pilot=registration['pilot'], exact_retention=registration['exact_retention'],
        unchanged_control='Original point-writer final streams and frozen scores, explicitly reused; new actual step0 is the within-trial control',
        head_only_control_language='unmeasured; failed physical gate, not regenerated', original_control_reexecuted=False,
        physical_reviewer_kind=session['reviewer_kind'], candidate_credited_empty=audit['candidate_credited_empty'],
        prior_development_familiarity=True, independent_human_adjudication=False, untouched_confirmation_required=True,
        complete_output_review_before_new_references=True, language_quality_unmeasured=True, goal_achieved=False)
    DEST.mkdir(); write_new(DEST / 'preparation.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), candidate_streams=708, generated_streams=0)


def validate():
    registration, _, _ = physical_gate()
    require(PUBLIC.exists() and (DEST / 'preparation.json').exists(), 'Bind post-gate evaluation before generation')
    value = load(PUBLIC)
    require(PUBLIC.read_bytes() == (DEST / 'preparation.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact post-gate registration first')
    require(value['source_sha256'] == committed_sources(registration) and value['arms'] == plan(registration)
            and value['joint_registration_sha256'] == sha(TRAINING_PUBLIC), 'Source, snapshot plan or parent trial changed')
    for name, digest in value['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound evidence changed: ' + name)
    return value


def restore_snapshot(actor, saved, step, registration):
    """Restore every registered context/head tensor even after inference freezing."""
    import torch
    from ilm.visual_lm.joint_context_training import tensor_inventory
    check_snapshot(saved, step, registration)
    parameters = dict(actor.named_parameters()); names = registration['initial_trainable_inventory']
    require(set(saved['trainable_weights']) == set(names) <= set(parameters), 'Missing actual context/head parameters')
    with torch.no_grad():
        for name, weight in saved['trainable_weights'].items():
            require(parameters[name].shape == weight.shape and parameters[name].dtype == torch.float32, 'Wrong actual trainable tensor')
            parameters[name].copy_(weight.to(parameters[name].device))
    actual = tensor_inventory({name: parameters[name].detach().cpu() for name in names})
    require(actual == tensor_inventory(saved['trainable_weights']), 'Actual restored context/head differs from checkpoint')
    actor.requires_grad_(False).eval()
    require(not any(p.requires_grad for p in actor.parameters()), 'Inference must freeze all parameters')
    return actual


def execute(resume=False):
    began = time.perf_counter(); value = validate()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'PhysicalGPU0 and deterministic CUBLAS required')
    from scripts.score_pixel_language_first_blocks import owner_live
    with (RUN / 'language_evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = DEST / 'status.json'; previous = bool(list((DEST / 'attempts').glob('*/start.json')))
        if status.exists():
            require(not owner_live(load(status)), 'Original evaluator still live')
        require(resume == previous, 'Explicit manual resume required exactly for a prior attempt')
        require(not (DEST / 'report.json').exists(), 'Completed evaluation must not repeat')
        seconds, pressure = remaining_budget(DEST, value['maximum_seconds'], value['pressure_wait_seconds'])
        execute_locked(value, seconds, pressure, began)


def execute_locked(value, seconds, pressure, began):
    number = len(list((DEST / 'attempts').glob('*/start.json'))) + 1
    attempt = DEST / 'attempts' / f'{number:03}'
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())
    write_new(attempt / 'start.json', dict(identity, registration_sha256=sha(PUBLIC), remaining_seconds=seconds,
        remaining_pressure_seconds=pressure, prelaunch_validation_seconds=time.perf_counter() - began))
    hooks = []; error = None; monitor = None
    try:
        import torch
        from ilm.visual_lm.joint_context_model import build_joint_context_actor
        from ilm.visual_lm.joint_context_training import require_process_runtime
        from ilm.visual_lm.joint_writer_training import Monitor
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        from scripts.recover_visual_continuation_evaluation import resources, admitted
        from scripts.train_semantic_glyph_language import numerics
        from scripts.train_visual_continuation import atomic_json, generate_cases
        registration = load(TRAINING_PUBLIC); require_process_runtime(registration)
        monitor = Monitor(DEST, seconds, pressure, value['maximum_allocated_vram_gib']); monitor.began = began
        monitor.status('admitting')
        while True:
            observed = resources()
            if admitted(observed, value['minimum_entry_ram_gib'], value['minimum_free_vram_gib'] * 1024):
                break
            remaining = min(pressure - monitor.waited, seconds - (time.perf_counter() - began))
            require(remaining > 0, 'Cumulative admission/evaluation budget exhausted')
            monitor.status('waiting_for_admission', resources=observed)
            start = time.perf_counter(); time.sleep(min(10., remaining)); monitor.waited += time.perf_counter() - start
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance, account = build_joint_context_actor(device)
            require(account == value['model_account'], 'Whole deployed model differs')
            names = set(registration['initial_trainable_inventory'])
            frozen = [(p, p._version) for name, p in actor.named_parameters() if name not in names]
            buffers = [(p, p._version) for p in actor.buffers()]
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
            arms = []
            for planned in value['arms']:
                step = planned['step']; snapshot = value['snapshots'][str(step)]
                require(sha(ROOT / snapshot['file']) == snapshot['sha256'], 'Actual snapshot changed')
                saved = torch.load(ROOT / snapshot['file'], weights_only=True, map_location='cpu')
                restored = restore_snapshot(actor, saved, step, registration); del saved
                require(all(p._version == v for p, v in frozen + buffers), 'Snapshot restoration changed frozen body/buffers')
                versions = [(p, p._version) for p in [*actor.parameters(), *actor.buffers()]]
                cap = planned['maximum_glyphs']; folder = DEST / f'step_{step:06}' / ('core128' if cap == 128 else 'long640')
                records = generate_cases(actor, planned['inputs'], folder, sha(PUBLIC), snapshot['sha256'], cap, monitor)
                require(all(p._version == v for p, v in versions), 'Generation changed model tensors')
                arms.append(dict(arm=planned['arm'], step=step, records=records, restored_context_and_head_inventory=restored))
            require(sum(len(a['records']) for a in arms) == 708, 'All708complete streams required')
            cost = monitor.receipt()
            write_new(DEST / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), arms=arms, total_streams=708,
                model_account=account, provenance=provenance, whole_deployed_parameter_elements=sum(p.numel() for p in actor.parameters()),
                all131_context_and_head_tensors_restored_per_snapshot=True, model_tensors_unchanged=True,
                new_optimizer_updates=0, actual_emitted_pixel_feedback=True, reference_inputs=False,
                reference_output_lengths=False, original_control_reexecuted=False, language_quality_unmeasured=True,
                complete_output_review_pending=True, goal_achieved=False,
                cumulative_elapsed_seconds=value['maximum_seconds'] - seconds + cost['elapsed_seconds'],
                cumulative_pressure_wait_seconds=value['pressure_wait_seconds'] - pressure + cost['resource_wait_seconds'], **cost))
            monitor.status('awaiting_opaque_complete_output_review')
    except BaseException as caught:
        error = caught
        if monitor is not None:
            monitor.status('failed', error=repr(caught))
        raise
    finally:
        for hook in hooks:
            hook.remove()
        cost = monitor.receipt() if monitor else dict(elapsed_seconds=time.perf_counter() - began,
            resource_wait_seconds=0., peak_allocated_vram_gib=None, neural_import_failed=True)
        if 'minimum_available_gib' in cost and not math.isfinite(cost['minimum_available_gib'] or 0):
            cost['minimum_available_gib'] = None
        write_new(attempt / 'end.json', dict(identity, complete=error is None and (DEST / 'report.json').exists(),
            error=repr(error) if error else None, **cost))
        if monitor is not None:
            atomic_json(DEST / 'status.json', dict(load(DEST / 'status.json'), pid=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['plan', 'register', 'generate'])
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    if args.action == 'plan':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only plan required')
        arms = plan(load(TRAINING_PUBLIC)); result = dict(arms=[dict(step=a['step'], arm=a['arm'], cap=a['maximum_glyphs'],
            cases=len(a['inputs'])) for a in arms], total_streams=708, new_model_calls=0, language_quality_unmeasured=True)
    elif args.action == 'register':
        result = register()
    else:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'Evaluation interrupted by signal{signum}; preserve attempt')
        for signum in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT]:
            signal.signal(signum, interrupted)
        execute(args.resume); result = dict(generated=True, language_quality_unmeasured=True)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
