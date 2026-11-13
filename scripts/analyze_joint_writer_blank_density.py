#!/usr/bin/env python3
"""Source-bound probability diagnosis after the joint writer's frozen review.

No reader replay, optimization, new glyph generation or language grading.
Registration and original-result export do not import Torch.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'artifacts/continuation_joint_writer_v1'
RUN = OLD / 'blank_density'
ORIGINAL = ROOT / 'references/evidence/joint_writer_physical_result_20261113.json'
PUBLIC = ROOT / 'references/evidence/joint_writer_blank_density_registration_20261113.json'
SOURCES = ['scripts/analyze_joint_writer_blank_density.py',
           'tests/test_joint_writer_blank_density.py',
           'references/joint_writer_blank_density_protocol.md']


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write_new(path, value):
    from scripts.review_joint_writer_physical import write_new as publish
    publish(path, value)


def blank_path_bounds(log_probabilities):
    """Upper-bound every nonempty completion by its first nonzero prefix.

    Input is [batch, physical positions, alphabet], scored on the all-zero
    history. Later conditional probabilities cannot exceed one. This bound
    can be loose; failing to certify the blank does not prove another mode.
    """
    import numpy as np
    x = np.asarray(log_probabilities, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1] > 0 and x.shape[2] > 1
            and np.isfinite(x).all() and (x <= 1e-7).all(), 'Finite log probabilities required')
    z = x[:, :, 0]
    previous = np.concatenate([np.zeros((len(x), 1)), np.cumsum(z, axis=1)[:, :-1]], axis=1)
    return z.sum(axis=1), (previous + x[:, :, 1:].max(axis=2)).max(axis=1)


def original_export():
    from scripts.review_joint_writer_physical import (session_context,
        read_frozen_bundle, aggregate, verify_pixels)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU export only')
    (q, reg, qualification, boards, ids, bindings), session = session_context()
    frozen = read_frozen_bundle(q, boards, ids, bindings)
    mapping = load(q / 'mapping.json')['rows']
    groups, passed = aggregate(reg['qualification_jobs'], mapping, frozen['readings'],
                              bindings['registration_sha256'], .9)
    gate = load(q / 'gate.json')
    require(groups == gate['groups'] and passed == gate['passed'], 'Original gate recount differs')
    verify_pixels(q, qualification, boards, mapping, frozen)
    fit, audit = [load(OLD / name / 'report.json') for name in ['fit', 'audit']]
    journal = [json.loads(x) for x in (OLD / 'controller_journal.jsonl').read_text().splitlines()]
    costs = Counter()
    for event in journal:
        if event['event'] == 'ended':
            costs[event['action']] += event['elapsed_seconds']
    inventory = load(q / 'post_freeze_error_inventory.json')
    require(len(inventory['candidate_unclear_selected']) ==
            sum(x['count'] - x['readable_nonblank'] for x in groups if x['role'] == 'candidate'),
            'Post-freeze inventory does not match frozen physical failure count')
    sources = ['registration.json', 'cache/report.json', 'fit/report.json', 'fit/snapshots.json',
        'fit/resume_control.json', 'audit/report.json', 'qualify/report.json', 'qualify/gate.json',
        'qualify/frozen_readings.json', 'qualify/review/freeze_receipt.json',
        'qualify/boards.json', 'qualify/mapping.json', 'controller_journal.jsonl',
        'qualify/post_freeze_error_inventory.json']
    return dict(schema='joint-writer-completed-physical-result-v1',
        export_source_sha256=sha(Path(__file__)),
        input_sha256={name: sha(OLD / name) for name in sources},
        registration_sha256=bindings['registration_sha256'],
        training={k: fit[k] for k in ['retained_updates', 'extra_executed_resume_updates',
            'training_cases', 'epochs', 'target_exposure', 'head_parameters', 'snapshots',
            'elapsed_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib']},
        audit={k: audit[k] for k in ['complete', 'retained_updates', 'executed_events_all_attempts',
            'retained_target_exposure', 'source_target_inverse_packing_verified',
            'mean_recount_maximum_error', 'scale_recount_maximum_error', 'elapsed_seconds', 'peak_rss_gib']},
        qualification_cases=len(qualification['records']), reviewed_boards=len(boards['boards']),
        reviewed_images=len(ids), groups=groups, physical_gate_passed=passed,
        cumulative_stage_child_seconds=dict(costs),
        qualification_resources={k: qualification[k] for k in ['elapsed_seconds',
            'resource_wait_seconds', 'peak_allocated_vram_gib']},
        post_freeze_pixel_statistics=inventory['group_statistics'],
        post_freeze_inventory_is_descriptive_not_regrading=True,
        review=dict(reviewer=session['reviewer'], reviewer_kind=session['reviewer_kind'],
            prior_development_familiarity=True, independent_human_adjudication=False,
            frozen_readings_sha256=sha(q / 'frozen_readings.json'),
            readings_frozen_before_owner_unmask=True),
        autonomous_language_quality='unmeasured; physical prerequisite failed',
        new_autonomous_streams=0, full_deployed_parameter_recount_available=False,
        inherited_visual_contextual_parameters=7522463984,
        cost_scope='Head-only fit/qualification peaks are not complete-model footprints. Cache cost includes the failed first attempt and CPU recovery verification; inherited pretraining and prior reader training remain separate.',
        new_model_calls=0, new_optimizer_updates=0, goal_achieved=False)


def source_bindings():
    values = load(OLD / 'registration.json')['source_sha256'].copy()
    for name in SOURCES:
        require((ROOT / name).read_bytes() == subprocess.check_output(
            ['git', 'show', 'HEAD:' + name], cwd=ROOT), 'Commit source before registration: ' + name)
        values[name] = sha(ROOT / name)
    for name, digest in values.items():
        require(sha(ROOT / name) == digest, 'Original or diagnostic source changed: ' + name)
    return values


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve the existing diagnostic')
    require(not load(OLD / 'qualify/gate.json')['passed'], 'This is a diagnosis of the actual failed actor')
    files = ['registration.json', 'fit/report.json', 'fit/snapshots.json', 'cache/report.json',
             'cache/normalization.pt', 'qualify/report.json', 'qualify/gate.json',
             'qualify/frozen_readings.json', 'qualify/review/freeze_receipt.json']
    head = load(OLD / 'fit/snapshots.json')['41298']
    inputs = {str((OLD / name).relative_to(ROOT)): sha(OLD / name) for name in files}
    inputs[head['file']] = head['sha256']
    inputs[str(ORIGINAL.relative_to(ROOT))] = sha(ORIGINAL)
    value = dict(schema='joint-writer-blank-density-diagnosis-v1', source_sha256=source_bindings(),
        input_sha256=inputs, head=head, cases=124, complete_target_positions=12597,
        batch_glyphs=32, maximum_seconds=900, pressure_seconds=120,
        maximum_allocated_vram_gib=2, minimum_available_host_gib=24,
        numerical_margin_nats=.001, new_optimizer_updates=0,
        generated_glyphs=0, reader_calls=0,
        action='Test exact source, blank and saved-output probabilities on all124 complete cached histories. A source-over-blank margin on at least7 development-book and1 train-fixed-QA selected false blanks permits considering a separately registered search-only intervention. Otherwise do not assume search repairs the fitted density. No result authorizes changing the old gate or running its blocked autonomous evaluation.',
        old_frozen_readings_immutable=True, autonomous_language_quality_unmeasured=True)
    RUN.mkdir()
    write_new(RUN / 'registration.json', value)
    write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), cases=124, target_positions=12597)


def run():
    began = time.perf_counter()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Use the single registered GPU0')
    r = load(PUBLIC)
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() ==
            subprocess.check_output(['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT),
            'Commit and push registration before neural scoring')
    require(r['source_sha256'] == source_bindings(), 'Diagnostic source differs')
    for name, digest in r['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Diagnostic input changed: ' + name)
    require(not (RUN / 'status.json').exists(), 'No automatic diagnostic retry')
    import numpy as np
    import torch
    from ilm.visual_lm.glyph_joint_density import GlyphJointDensity, pack_glyphs
    from ilm.visual_lm.joint_writer_training import CachedSequences, Monitor
    from scripts.train_joint_continuation_writer import finish
    from scripts.train_semantic_glyph_bridge import bridge_gpu
    from scripts.train_semantic_glyph_language import numerics
    from scripts.train_visual_continuation import atomic_json
    torch.set_num_threads(2)
    monitor = Monitor(RUN, r['maximum_seconds'] - (time.perf_counter() - began),
                      r['pressure_seconds'], r['maximum_allocated_vram_gib'])
    monitor.status('admitting'); error = None
    try:
        meta = load(OLD / 'cache/report.json'); q = load(OLD / 'qualify/report.json')
        records = {x['job']['id']: x for x in q['records']}; data = CachedSequences()
        groups = {}; detail = []; total = 0; needed_book = 0; needed_qa = 0
        with bridge_gpu(dict(minimum_entry_ram_gib=24.)) as (device, admission), torch.no_grad():
            monitor.admission = admission; numerics(); monitor.guard()
            norm = torch.load(OLD / 'cache/normalization.pt', weights_only=True, map_location='cpu')
            model = GlyphJointDensity(norm['mean'], norm['scale']).to(device).eval().requires_grad_(False)
            model.load_state_dict(torch.load(ROOT / r['head']['file'], weights_only=True, map_location='cpu')['model'])
            versions = [x._version for x in list(model.parameters()) + list(model.buffers())]
            for i, item in enumerate(meta['qualification']):
                job = item['job']; monitor.position = dict(completed=i, total=124, case=job['id'])
                monitor.status('scoring_saved_density'); monitor.guard()
                states, patterns, _ = data.get(ROOT / item['directory']); n = len(patterns)
                pixels = OLD / 'qualify/cases' / job['id'] / 'pixels.npz'
                require(sha(pixels) == records[job['id']]['pixels_sha256'], 'Saved predictions changed')
                with np.load(pixels, allow_pickle=False) as arrays:
                    emitted = pack_glyphs(torch.from_numpy(arrays['candidate_ink']))
                output = {k: [] for k in ['source_logp', 'blank_logp', 'emitted_logp', 'nonempty_upper_bound', 'blank_argmax_disagreements']}
                for start in range(0, n, r['batch_glyphs']):
                    monitor.guard(); h = states[start:start + min(r['batch_glyphs'], n-start)].to(device)
                    gold = patterns[start:start + len(h)].to(device=device, dtype=torch.long)
                    old = emitted[start:start + len(h)].to(device); zero = torch.zeros_like(gold)
                    lp = model(h, zero).float().log_softmax(-1)
                    z, upper = blank_path_bounds(lp.cpu().numpy())
                    output['blank_logp'].append(z); output['nonempty_upper_bound'].append(upper)
                    output['blank_argmax_disagreements'].append((lp.argmax(-1) != 0).sum(1).cpu().numpy())
                    for name, target in [('source_logp', gold), ('emitted_logp', old)]:
                        values = model(h, target).float().log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
                        output[name].append(values.double().sum(1).cpu().numpy())
                output = {k: np.concatenate(v) for k, v in output.items()}
                require(all(np.isfinite(v).all() for v in output.values()), 'Nonfinite diagnostic scores')
                gold_blank = (patterns.numpy() == 0).all(1); old_blank = (emitted.numpy() == 0).all(1)
                require(np.allclose(output['blank_logp'][old_blank], output['emitted_logp'][old_blank], atol=1e-5, rtol=0), 'Identical blank images must have identical likelihood')
                false_blank = old_blank & ~gold_blank
                better = output['source_logp'] > output['blank_logp'] + r['numerical_margin_nats']
                certified = output['nonempty_upper_bound'] < output['blank_logp'] - r['numerical_margin_nats']
                k = '/'.join([job['split'], job['kind'], job['layout']]); s = groups.setdefault(k, Counter())
                s.update(positions=n, gold_blank=int(gold_blank.sum()), emitted_blank=int(old_blank.sum()),
                    false_blank=int(false_blank.sum()), source_preferred_false_blank=int((false_blank & better).sum()),
                    blank_upper_bound_holds_false_blank=int((false_blank & certified).sum()),
                    saved_blank_parallel_argmax_disagreements=int(output['blank_argmax_disagreements'][old_blank].sum()))
                selected = np.zeros(n, dtype=bool); selected[job['positions']] = True
                count = int((selected & false_blank & better).sum())
                if k == 'development/book/fixed_cells': needed_book += count
                if k == 'train/qa/fixed_cells': needed_qa += count
                directory = RUN / 'cases' / job['id']; directory.mkdir(parents=True)
                np.savez_compressed(directory / 'probabilities.npz', **output,
                    gold_blank=gold_blank, emitted_blank=old_blank, selected=selected)
                detail.append(dict(case=job['id'], positions=n,
                    file=str((directory / 'probabilities.npz').relative_to(ROOT)),
                    sha256=sha(directory / 'probabilities.npz'))); total += n
            require(versions == [x._version for x in list(model.parameters()) + list(model.buffers())], 'Writer mutated during scoring')
            require(len(detail) == 124 and total == 12597, 'Full diagnostic cohort required')
            atomic_json(RUN / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC),
                cases=detail, positions=total, groups={k: dict(v) for k, v in groups.items()},
                selected_false_blanks_with_source_preferred=dict(development_book=needed_book, train_fixed_qa=needed_qa),
                search_only_hypothesis_merits_separate_registration=needed_book >= 7 and needed_qa >= 1,
                inference_parameters_unchanged=True, reader_calls=0, new_optimizer_updates=0,
                generated_glyphs=0, full_language_unmeasured=True,
                total_child_seconds=time.perf_counter()-began, **monitor.receipt()))
    except BaseException as exc:
        error = exc
        raise
    finally:
        finish(RUN, monitor, error)
    return dict(complete=True, positions=total, selected_source_preferred_book=needed_book,
                selected_source_preferred_fixed_qa=needed_qa)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['export-original', 'verify-original', 'register', 'run'])
    args = p.parse_args()
    if args.action in {'export-original', 'verify-original'}:
        value = original_export()
        if args.action == 'verify-original':
            require(load(ORIGINAL) == value, 'Original export does not reproduce')
        else:
            write_new(ORIGINAL, value)
        result = dict(physical_gate_passed=value['physical_gate_passed'],
                      reviewed_images=value['reviewed_images'], new_model_calls=0)
    else:
        result = register() if args.action == 'register' else run()
    print(json.dumps(result))
