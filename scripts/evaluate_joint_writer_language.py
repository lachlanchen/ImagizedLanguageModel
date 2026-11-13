#!/usr/bin/env python3
"""Plan, bind and execute the registered autonomous joint-writer comparison.

Planning imports no neural runtime. Registration and generation require the
completed fit, sealed physical review and a passing physical gate. No score is
assigned by this generator.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from scripts import review_joint_writer_physical as review

ROOT, RUN = review.ROOT, review.RUN
PUBLIC = ROOT / 'references/evidence/continuation_joint_writer_language_registration_v1.json'
DEST = RUN / 'language'
NEW_SOURCES = (
    'scripts/evaluate_joint_writer_language.py', 'scripts/review_joint_writer_physical.py',
    'scripts/recover_visual_continuation_evaluation.py',
    'tests/test_joint_writer_language_evaluation.py',
    'references/joint_writer_language_execution.md',
    'scripts/review_joint_writer_language.py', 'tests/test_joint_writer_complete_review.py',
    'scripts/review_compact_visual_span.py', 'scripts/prepare_compact_visual_span_evaluation.py',
    'scripts/infer_compact_visual_span.py', 'scripts/audit_visual_continuation_protocol.py',
    'references/joint_writer_complete_review.md',
)
EXPECTED_INPUTS = {
    ('fixed_cells', 'normal'): 44, ('fixed_cells', 'question_blank'): 32,
    ('fixed_cells', 'passage_blank'): 32, ('fixed_cells', 'prefix_blank'): 12,
    ('proportional', 'normal'): 32, ('proportional', 'question_blank'): 32,
    ('proportional', 'passage_blank'): 32,
}
load, sha, require, write_new = review.load, review.sha, review.require, review.write_new


def plan(registration):
    rows = registration['evaluation_inputs']
    require(len(rows) == 216 and len({x['id'] for x in rows}) == 216, 'Retain all 216 distinct inputs')
    require(Counter((x['layout'], x['condition']) for x in rows) == EXPECTED_INPUTS,
            'All layouts, question/passage erasures and book-prefix controls required')
    require(registration['snapshot_steps'] == [0, 10324, 20649, 30973, 41298], 'All five fixed snapshots required')
    books = [row for row in rows if '__book_' in row['id']]
    require(len(books) == 24 and Counter(x['condition'] for x in books) == {'normal': 12, 'prefix_blank': 12},
            'Retain both book-prefix conditions')
    for layout in ['fixed_cells', 'proportional']:
        for condition in ['normal', 'question_blank', 'passage_blank']:
            questions = [x['original_case'] for x in rows if x['layout'] == layout
                         and x['condition'] == condition and '__qa_' in x['id']]
            pairs = Counter(x.split('_')[1] for x in questions)
            require(len(questions) == 32 and len(pairs) == 16 and set(pairs.values()) == {2},
                    'Retain both questions in all sixteen passage groups')
    arms = []
    for step in registration['snapshot_steps']:
        endpoint = step in [0, 41298]
        chosen = rows if endpoint else [x for x in rows if x['condition'] == 'normal']
        arms.append(dict(arm='candidate_core128', step=step, maximum_glyphs=128, inputs=chosen))
        if endpoint:
            arms.append(dict(arm='candidate_long640', step=step, maximum_glyphs=640, inputs=books))
    require(sum(len(x['inputs']) for x in arms) == registration['new_candidate_streams_after_physical_gate'] == 708,
            'The complete 708-stream comparison is required')
    return arms


def physical_gate():
    # Fail before any neural imports or output-directory creation on an unqualified run.
    require((RUN / 'qualify/gate.json').exists(), 'Physical writing review is not yet complete')
    gate = load(RUN / 'qualify/gate.json')
    require(gate.get('passed') is True, 'Physical gate failed; do not generate unqualified language outputs')
    values, session = review.session_context()
    qualify, registration, _, manifest, ids, bindings = values
    frozen = review.read_frozen_bundle(qualify, manifest, ids, bindings)
    require(gate.get('schema') == 'joint-writer-physical-gate-v2' and gate['bindings'] == bindings,
            'Source-bound freeze-before-unmask review required')
    require(gate['frozen_readings_sha256'] == sha(qualify / 'frozen_readings.json')
            and gate['session_sha256'] == sha(qualify / 'review/session.json'), 'Physical review changed')
    require(gate['complete_registered_positions_recounted'] is True
            and gate['original_page_and_board_pixels_verified'] is True
            and gate['readings_frozen_before_workflow_owner_map_read'] is True,
            'Complete physical gate provenance required')
    mapping = load(qualify / 'mapping.json')
    require(sha(qualify / 'mapping.json') == bindings['mapping_sha256'], 'Physical mapping changed')
    groups, passed = review.aggregate(registration['qualification_jobs'], mapping['rows'], frozen['readings'],
                                    bindings['registration_sha256'], registration['physical_gate_readable_nonblank_fraction'])
    require(passed and gate['groups'] == groups, 'Physical group recount differs')
    return registration, session


def committed_sources(registration):
    result = dict(registration['source_sha256'])
    for name in NEW_SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit evaluation implementation before binding: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Registered source changed: ' + name)
    return result


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only evaluation registration required')
    registration, session = physical_gate()
    require(not DEST.exists() and not PUBLIC.exists(), 'Preserve existing evaluation preparation')
    arms = plan(registration)
    sources = committed_sources(registration)
    bindings = dict(registration['bindings_sha256'])
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Original bound model, corpus or control changed: ' + name)
    def bind(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for name in ['registration.json', 'cache/report.json', 'cache/normalization.pt', 'fit/report.json',
                 'fit/snapshots.json', 'audit/report.json', 'qualify/report.json', 'qualify/boards.json',
                 'qualify/mapping.json', 'qualify/frozen_readings.json', 'qualify/gate.json',
                 'qualify/review/session.json', 'qualify/review/freeze_receipt.json']:
        bind(RUN / name)
    snapshots = load(RUN / 'fit/snapshots.json')
    fit = load(RUN / 'fit/report.json')
    require(snapshots == fit['snapshots'], 'Snapshot index differs from the completed fitted model')
    require(fit['cache_report_sha256'] == sha(RUN / 'cache/report.json'), 'Fitted cache report changed')
    require(sorted(map(int, snapshots)) == registration['snapshot_steps'], 'Fixed head snapshots incomplete')
    for step, value in snapshots.items():
        path = ROOT / value['file']
        require(sha(path) == value['sha256'] and value['step'] == int(step), 'Head snapshot changed')
        bind(path)
    for row in registration['evaluation_inputs']:
        require(sha(ROOT / row['image']) == row['sha256'], 'Complete evaluation prompt changed')
        bindings[row['image']] = row['sha256']
    # The full point-writer control and its frozen scores are reused, never regenerated.
    parent = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
    for name in ['evaluation/report.json', 'review/frozen_readings.json', 'review/summary.json',
                 'review/protocol_pilot_audit.json']:
        bind(parent / name)
    frozen = load(RUN / 'qualify/frozen_readings.json')
    for key, digest in frozen['image_sha256'].items():
        path = RUN / 'qualify/pages' / (key + '.png')
        require(sha(path) == digest, 'Reviewed physical page changed')
    value = dict(
        schema='joint-writer-autonomous-evaluation-v1', source_sha256=sources, bindings_sha256=bindings,
        joint_registration_sha256=sha(review.PUBLIC), runtime=registration['runtime'],
        parent_checkpoint=registration['parent_checkpoint'], snapshots=snapshots, arms=arms,
        total_candidate_streams=708, total_core_streams=660, total_long_book_streams=48,
        greedy_physical_masks=True, actual_emitted_pixel_feedback=True, stop_probability_threshold=.5,
        reference_inputs=False, reference_output_lengths=False, live_teacher=False, lexical_output_head=False,
        font_lookup=False, OCR=False, maximum_seconds=registration['generation_seconds'],
        pressure_wait_seconds=1800, minimum_entry_ram_gib=48, prelaunch_extra_ram_gib=2,
        minimum_runtime_ram_gib=24, minimum_free_vram_gib=20, maximum_allocated_vram_gib=22,
        pilot=registration['pilot'], exact_retention=registration['exact_retention'],
        unchanged_control='Original point-writer final streams and frozen scores, explicitly reused',
        original_control_reexecuted=False, physical_reviewer_kind=session['reviewer_kind'],
        prior_development_familiarity=True, independent_human_adjudication=False,
        untouched_confirmation_required=True, complete_output_review_before_new_references=True,
        language_quality_unmeasured=True, goal_achieved=False,
    )
    DEST.mkdir()
    write_new(DEST / 'preparation.json', value)
    write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), candidate_streams=708, generated_streams=0)


def validate():
    registration, _ = physical_gate()
    require(PUBLIC.exists() and (DEST / 'preparation.json').exists(), 'Bind evaluation before generation')
    value = load(PUBLIC)
    require(PUBLIC.read_bytes() == (DEST / 'preparation.json').read_bytes()
            == subprocess.check_output(['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT),
            'Commit the exact post-gate evaluation registration before generation')
    require(value['source_sha256'] == committed_sources(registration) and value['arms'] == plan(registration),
            'Evaluation code or complete-case plan changed')
    for name, digest in value['bindings_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound model, prompt, review or unchanged control changed: ' + name)
    return value


def remaining_budget(destination, total, pressure):
    spent = waited = 0.
    attempts = sorted((destination / 'attempts').glob('*/start.json'))
    for path in attempts:
        terminal = path.with_name('end.json')
        require(terminal.exists(), 'Reconcile the preserved unterminated attempt before manual recovery')
        value = load(terminal)
        seconds, idle = value['elapsed_seconds'], value['resource_wait_seconds']
        require(all(math.isfinite(x) and x >= 0 for x in [seconds, idle]) and idle <= seconds,
                'Invalid cumulative evaluation cost')
        spent += seconds
        waited += idle
    require(total > spent and pressure >= waited, 'Cumulative evaluation budget exhausted')
    return total - spent, pressure - waited


def check_head(value, step, registration, evaluation):
    require(value['schema'] == 'joint-continuation-head-checkpoint-v1' and value['step'] == step,
            'Wrong registered head snapshot')
    require(value['registration_sha256'] == evaluation['joint_registration_sha256']
            and value['parent_checkpoint'] == evaluation['parent_checkpoint']
            and value['runtime'] == evaluation['runtime']
            and value['source_sha256'] == registration['source_sha256'], 'Wrong head provenance')
    require(value['cache_report_sha256'] == sha(RUN / 'cache/report.json'), 'Head cache binding differs')
    require(value['completed_training'] is (step == 41298) and value['only_head_trainable'] is True,
            'Wrong snapshot completion metadata')


def execute(resume=False):
    began_stage = time.perf_counter()
    value = validate()  # Before neural imports, CUDA initialization or stage creation.
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Physical GPU0 and deterministic CUDA required')
    from scripts.score_pixel_language_first_blocks import owner_live
    with (RUN / 'language_evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = DEST / 'status.json'
        if status.exists():
            require(not owner_live(load(status)), 'Original evaluator remains live')
            require(resume, 'Preserve prior attempt; explicit manual resume required')
        else:
            previous = bool(list((DEST / 'attempts').glob('*/start.json')))
            require(resume == previous, 'Explicit resume is required exactly when a prior attempt exists')
        require(not (DEST / 'report.json').exists(), 'Completed evaluation must not run again')
        seconds, pressure = remaining_budget(DEST, value['maximum_seconds'], value['pressure_wait_seconds'])
        _execute_locked(value, seconds, pressure, began_stage)


def _execute_locked(value, seconds, pressure, began_stage):
    number = len(list((DEST / 'attempts').glob('*/start.json'))) + 1
    attempt = DEST / 'attempts' / f'{number:03}'
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), began_unix=time.time())
    write_new(attempt / 'start.json', dict(identity, registration_sha256=sha(PUBLIC), remaining_seconds=seconds,
        remaining_pressure_seconds=pressure, prelaunch_validation_seconds=time.perf_counter() - began_stage))
    hooks, error, monitor = [], None, None
    try:
        import torch
        from ilm.visual_lm.continuation_training import TrainingMonitor
        from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, JointGlyphBridge
        from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
        from scripts.recover_visual_continuation_evaluation import resources, admitted
        from scripts.train_joint_continuation_writer import frozen_parent
        from scripts.train_semantic_glyph_language import numerics
        from scripts.train_visual_continuation import atomic_json, generate_cases
        from scripts.prepare_joint_continuation_writer import validate as validate_joint
        monitor = TrainingMonitor(DEST, seconds, pressure)
        monitor.began = began_stage  # Include neural imports and setup in the same ceiling.
        monitor.status('admitting')
        registration = validate_joint()
        while True:
            observed = resources()
            if admitted(observed, value['minimum_entry_ram_gib'], value['minimum_free_vram_gib'] * 1024):
                break
            require(monitor.waited < pressure and time.perf_counter() - monitor.began < seconds,
                    'Cumulative admission/evaluation budget exhausted')
            monitor.status('waiting_for_admission', resources=observed)
            began = time.perf_counter()
            time.sleep(min(10., pressure - monitor.waited, seconds - (began - monitor.began)))
            monitor.waited += time.perf_counter() - began
        with full_reader_gpu() as (device, admission), torch.no_grad():
            monitor.admission = admission
            numerics()
            torch.set_num_threads(2)
            monitor.guard()
            actor, provenance = frozen_parent(device, registration)
            normalization = torch.load(RUN / 'cache/normalization.pt', weights_only=True, map_location='cpu')
            density = GlyphJointDensity(normalization['mean'], normalization['scale']).to(device).eval()
            actor.bridge = JointGlyphBridge(actor.bridge, density)
            actor.requires_grad_(False).eval()
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard())
                     for layer in actor.interface.actor.context.layers]
            arms = []
            for planned in value['arms']:
                step = planned['step']
                snapshot = value['snapshots'][str(step)]
                require(sha(ROOT / snapshot['file']) == snapshot['sha256'], 'Head changed before loading')
                saved = torch.load(ROOT / snapshot['file'], weights_only=True, map_location='cpu')
                check_head(saved, step, registration, value)
                require(all(x.dtype == torch.float32 and bool(torch.isfinite(x).all()) for x in saved['model'].values()),
                        'Finite FP32 joint-writer tensors required')
                density.load_state_dict(saved['model'], strict=True)
                del saved
                versions = [(x, x._version) for x in [*actor.parameters(), *actor.buffers()]]
                cap = planned['maximum_glyphs']
                folder = DEST / f'step_{step:06}' / ('core128' if cap == 128 else 'long640')
                records = generate_cases(actor, planned['inputs'], folder, sha(PUBLIC), snapshot['sha256'], cap, monitor)
                require(all(x._version == version for x, version in versions), 'Evaluation changed model tensors')
                arms.append(dict(arm=planned['arm'], step=step, records=records))
            require(sum(len(x['records']) for x in arms) == 708, 'Incomplete autonomous comparison')
            parameter_elements = sum(x.numel() for x in actor.parameters())
            cost = monitor.receipt()
            report = dict(
                complete=True, registration_sha256=sha(PUBLIC), arms=arms, total_streams=708,
                provenance=provenance, whole_deployed_parameter_elements=parameter_elements,
                joint_head_parameter_elements=sum(x.numel() for x in density.parameters()),
                new_optimizer_updates=0, actual_emitted_pixel_feedback=True,
                reference_inputs=False, reference_output_lengths=False,
                original_control_reexecuted=False, model_tensors_unchanged=True,
                language_quality_unmeasured=True, complete_output_review_pending=True, goal_achieved=False,
                cumulative_elapsed_seconds=value['maximum_seconds'] - seconds + cost['elapsed_seconds'],
                cumulative_pressure_wait_seconds=value['pressure_wait_seconds'] - pressure + cost['resource_wait_seconds'],
                **cost)
            write_new(DEST / 'report.json', report)
            monitor.status('awaiting_opaque_complete_output_review')
    except BaseException as caught:
        error = caught
        if monitor is not None:
            monitor.status('failed', error=repr(caught))
        raise
    finally:
        for hook in hooks:
            hook.remove()
        cost = monitor.receipt() if monitor is not None else dict(
            elapsed_seconds=time.perf_counter() - began_stage, resource_wait_seconds=0,
            peak_allocated_vram_gib=None, neural_import_failed=True)
        if 'minimum_available_gib' in cost and not math.isfinite(cost['minimum_available_gib']):
            cost['minimum_available_gib'] = None
        receipt = dict(identity, complete=error is None and (DEST / 'report.json').exists(),
                       error=repr(error) if error else None, **cost)
        write_new(attempt / 'end.json', receipt)
        status = DEST / 'status.json'
        if monitor is not None:
            atomic_json(status, dict(load(status), pid=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'register', 'generate'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.action == 'plan':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only planning required')
        arms = plan(load(review.PUBLIC))
        result = dict(arms=[dict(step=x['step'], arm=x['arm'], cap=x['maximum_glyphs'], cases=len(x['inputs']))
                            for x in arms], total_streams=708, new_model_calls=0, language_quality_unmeasured=True)
    elif args.action == 'register':
        result = register()
    else:
        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f'Evaluation interrupted by signal {signum}; preserve this attempt')
        for signum in [signal.SIGTERM, signal.SIGHUP]:
            signal.signal(signum, interrupted)
        execute(args.resume)
        result = dict(generated=True, language_quality_unmeasured=True)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
