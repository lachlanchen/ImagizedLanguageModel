#!/usr/bin/env python3
"""Verify the white-target objective on complete actual sources; reset all steps."""
import time
BEGAN = time.perf_counter()

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess

from scripts import review_joint_writer_physical as evidence

ROOT = evidence.ROOT
PARENT = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
CONTROL = ROOT / 'artifacts/global_glyph_ranking_v1'
RUN = ROOT / 'artifacts/white_target_ranking_preflight_v1'
PUBLIC = ROOT / 'references/evidence/white_target_ranking_preflight_registration_20261115.json'
TRAINING = ROOT / 'references/evidence/global_glyph_ranking_registration_20261114.json'
PROBE = ROOT / 'references/evidence/ranked_writer_whitespace_registration_20261115.json'
NEW = ('ilm/visual_lm/white_target_ranking.py', 'tests/test_white_target_ranking.py',
       'scripts/preflight_white_target_ranking.py', 'scripts/summarize_white_target_preflight.py',
       'references/white_target_ranking_decision.md', 'references/white_target_preflight_protocol.md')
load, sha, require, write_new = evidence.load, evidence.sha, evidence.require, evidence.write_new


def sources():
    result = dict(load(PROBE)['source_sha256'])
    for name in NEW:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit new objective/preflight source: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Preserve completed source: ' + name)
    return result


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not RUN.exists() and not PUBLIC.exists(),
            'CPU-only first registration required')
    import numpy as np
    import torch
    from scripts.preflight_visual_continuation import runtime
    from ilm.visual_lm.joint_context_training import expected_initial, tensor_inventory
    require(not torch.cuda.is_initialized(), 'CPU registration must not initialize CUDA'); torch.set_num_threads(2)
    training = load(TRAINING); rows = load(PARENT / 'training_manifest.json')['rows']
    old_path = ROOT / 'references/evidence/global_glyph_ranking_preflight_registration_20261114.json'
    old = load(old_path); old_report = load(ROOT / 'artifacts/global_glyph_ranking_preflight_v1/execution/report.json')
    require(len(rows) == 13766 and old_report['complete'] and old_report['all_updates_reset'], 'Original full-source proof required')
    cases = [dict(row) for row in old['cases']]
    previous = {r['case']: r for r in old_report['cases']}
    for case in cases:
        case['expected_white_targets'] = case['source']['target_glyphs'] - previous[case['id']]['nonempty_targets']
    report = load(CONTROL / 'qualify/report.json')
    require([r['job'] for r in report['records']] == training['qualification_jobs'], 'Original qualification order required')
    first = None; selected_pixels = []
    bindings = dict(training['bindings_sha256'])
    def bind(path):
        name, digest = str(path.relative_to(ROOT)), sha(path)
        require(name not in bindings or bindings[name] == digest, 'Conflicting original evidence: ' + name)
        bindings[name] = digest
    for record in report['records']:
        job = record['job']; path = CONTROL / 'qualify/cases' / job['id'] / 'pixels.npz'
        require(sha(path) == record['pixels_sha256'], 'Fixed source pixels differ'); bind(path)
        if job['split'] == 'train' and job['kind'] == 'book' and first is None:
            with np.load(path, allow_pickle=False) as arrays:
                gold = arrays['gold_ink']; white = ~gold.reshape(len(gold), -1).any(1)
            if white.any() and (~white).any():
                first = dict(index=int(job['id'].split('_')[1]), qualification_case=job['id'],
                    targets=len(white), white_targets=int(white.sum()), source_id=job['source_id'])
    require(first is not None and first['targets'] == rows[first['index']]['target_glyphs'], 'Real complete mixed-white source required')
    for layout in ['fixed_cells', 'proportional']:
        job_id = f"train_{first['index']:05}_{layout}"
        record = next(r for r in report['records'] if r['job']['id'] == job_id)
        path = CONTROL / 'qualify/cases' / job_id / 'pixels.npz'
        with np.load(path, allow_pickle=False) as arrays:
            gold = arrays['gold_ink']; white = ~gold.reshape(len(gold), -1).any(1)
        require(len(gold) == first['targets'] and int(white.sum()) == first['white_targets'], 'Both complete source layouts required')
        selected_pixels.append(str(path.relative_to(ROOT)))
        cases.append(dict(id='white_book_' + layout, index=first['index'], layout=layout,
            source=rows[first['index']], expected_white_targets=first['white_targets'],
            qualification_pixels=str(path.relative_to(ROOT))))
    for path in [TRAINING, PROBE, old_path, ROOT / 'artifacts/global_glyph_ranking_preflight_v1/execution/report.json',
                 ROOT / 'references/evidence/global_glyph_ranking_physical_result_20261115.json',
                 ROOT / 'references/evidence/ranked_writer_whitespace_result_20261115.json',
                 CONTROL / 'qualify/report.json', CONTROL / 'qualify/gate.json',
                 CONTROL / 'whitespace_density/report.json', CONTROL / 'whitespace_density/attempts/end.json']:
        bind(path)
    for case in cases:
        row = case['source']['images'][case['layout']]; require(sha(ROOT / row['file']) == row['sha256'], 'Original prompt changed'); bind(ROOT / row['file'])
    for name, digest in bindings.items():
        require(sha(ROOT / name) == digest, 'Bound source/control changed: ' + name)
    value = dict(schema='white-target-ranking-real-source-preflight-v1', source_sha256=sources(),
        input_sha256=bindings, runtime=runtime(), external_program_sha256=training['external_program_sha256'],
        cases=cases, mixed_white_selection=first, selected_source_pixel_arrays=selected_pixels,
        initial_trainable_inventory=tensor_inventory(expected_initial()),
        ranking_strength=1., white_strength=1., margin_nats=1., qa_microbatch_glyphs=2, book_microbatch_glyphs=64,
        maximum_seconds=1200, external_term_seconds=1180, external_kill_grace_seconds=20,
        pressure_seconds=180, maximum_allocated_vram_gib=22, minimum_entry_host_gib=48,
        minimum_free_gpu_gib=20, minimum_runtime_host_gib=24, minimum_free_disk_gib=48, cpu_threads=2,
        gradient_relative_l2_tolerance=.01, gradient_absolute_tolerance=1e-5,
        gradient_maximum_scaled_tolerance=.01, loss_absolute_tolerance=2e-5,
        learning_rate=4e-7, weight_decay=.01, gradient_clip=1., expected_trainable_parameters=5005760,
        optimizer_updates=4, updates_carried_into_fit=False, new_autonomous_images=0,
        language_evaluation=False, goal_achieved=False,
        pass_action='Only all6 complete cases with actual131 finite gradients, both QA unsplit comparisons, four real book Adam steps, unchanged frozen state and exact resets permits a separate full-corpus registration. No language claim or automatic fit.',
        fail_action='Preserve first attempt and costs; diagnose the failed math, source or resource condition before any separately justified recovery. No fit or changed gate.')
    write_new(RUN / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), sources=len(value['source_sha256']), inputs=len(bindings),
        mixed_white_selection=first, cases=[(x['id'], x['source']['target_glyphs'], x['expected_white_targets']) for x in cases])


def validate():
    r = load(PUBLIC)
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit exact new registration first')
    require(r['source_sha256'] == sources(), 'New preflight source differs')
    for name, digest in r['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound preflight input changed: ' + name)
    return r


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
            'Physical GPU0 and deterministic CUBLAS required')
    r = validate(); folder = RUN / 'execution'; folder.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from scripts.preflight_visual_continuation import runtime
    from ilm.visual_lm.joint_context_training import tensor_inventory, require_process_runtime
    from ilm.visual_lm.continuation_training import TrainingImages, trainable_state, restore_trainable
    from ilm.visual_lm.joint_context_model import build_joint_context_actor, differentiable_states
    from ilm.visual_lm.white_target_ranking import backward_white_target_ranking
    from ilm.visual_lm.glyph_joint_density import pack_glyphs
    from ilm.visual_lm.joint_writer_training import Monitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_semantic_glyph_language import numerics
    from scripts.train_visual_continuation import atomic_json
    require(runtime() == r['runtime'], 'Registered neural runtime changed'); require_process_runtime(r)
    monitor = Monitor(folder, r['maximum_seconds'], r['pressure_seconds'], r['maximum_allocated_vram_gib'])
    monitor.began = BEGAN; monitor.status('admitting'); hooks = []; error = None
    def interrupted(signum, _frame):
        raise KeyboardInterrupt('White-target preflight interrupted: ' + str(signum))
    for signum in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]: signal.signal(signum, interrupted)
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance, account = build_joint_context_actor(device)
            initial = trainable_state(actor); parameters = {n: p for n, p in actor.named_parameters() if p.requires_grad}
            require(tensor_inventory(initial) == r['initial_trainable_inventory']
                and len(parameters) == 131 and account['trainable_parameters'] == 5005760, 'Original fresh initialization differs')
            frozen = [(p, p._version) for p in actor.parameters() if not p.requires_grad]
            affines = {n: b.detach().clone() for n, b in actor.bridge.named_buffers()
                if n in {'semantic_mean', 'semantic_scale', 'density.condition_mean', 'density.condition_scale'}}
            require(len(affines) == 4, 'Four frozen affine buffers required')
            hooks = [layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()) for layer in actor.dynamics.context.layers]
            data = TrainingImages(PARENT); results = []
            def frozen_ok():
                require(all(p._version == v and p.grad is None for p, v in frozen), 'Frozen parameter changed or received gradient')
                buffers = dict(actor.bridge.named_buffers())
                require(all(torch.equal(buffers[n], b) for n, b in affines.items()), 'Frozen affine changed')
            def gradients():
                out = {}
                for name, p in parameters.items():
                    require(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0,
                        'Missing, nonfinite or disconnected actual gradient: ' + name)
                    out[name] = p.grad.detach().float().cpu().clone()
                return out
            for i, job in enumerate(r['cases']):
                monitor.position = dict(completed=i, total=6, case=job['id']); monitor.status('complete_source_white_objective')
                prompt, ink, row, layout = data.get(job['index'], int(job['layout'] == 'proportional'))
                require(row == job['source'] and layout == job['layout'], 'Complete source changed')
                white_count = int((ink.reshape(len(ink), -1).sum(1) == 0).sum())
                require(white_count == job['expected_white_targets'], 'Real source whitespace differs')
                if 'qualification_pixels' in job:
                    with np.load(ROOT / job['qualification_pixels'], allow_pickle=False) as arrays:
                        require(np.array_equal(ink.numpy(), arrays['gold_ink']), 'Actual full source differs from selected original pixels')
                require(tensor_inventory(trainable_state(actor)) == r['initial_trainable_inventory'], 'An earlier update carried into this case')
                actor.zero_grad(set_to_none=True); h, memory, geometry = differentiable_states(actor, prompt, ink)
                if job['id'].startswith('qa_'):
                    target = pack_glyphs(ink).to(device)
                    lp = actor.bridge.density(h[:-1], target).float().log_softmax(-1)
                    source = lp.gather(-1, target[..., None]).squeeze(-1).sum(1)
                    zlp = actor.bridge.density(h[:-1], torch.zeros_like(target)).float().log_softmax(-1)
                    blank = zlp[..., 0].sum(1); active = (target != 0).any(1)
                    upper = torch.stack([zlp[:, :j, 0].sum(1) + zlp[:, j, 1:].max(1).values for j in range(128)], 1).max(1).values
                    image = -source.mean() / 128
                    rank = (F.softplus(r['margin_nats'] + blank - source) * active).mean() / 128
                    wrank = (F.softplus(r['margin_nats'] + upper - blank) * ~active).mean() / 128
                    s = actor.bridge.stop(h).flatten(); eos = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
                    loss = image + rank + wrank + .1 * eos; expected_loss = float(loss.detach()); loss.backward()
                    reference = gradients(); frozen_ok()
                    del h, lp, zlp, source, blank, upper, image, rank, wrank, s, eos, loss
                    actor.zero_grad(set_to_none=True); h, _, _ = differentiable_states(actor, prompt, ink, memory=memory)
                    result = backward_white_target_ranking(h, actor.bridge.density, actor.bridge.stop, ink,
                        ranking_strength=r['ranking_strength'], white_strength=r['white_strength'], margin_nats=r['margin_nats'],
                        microbatch_glyphs=r['qa_microbatch_glyphs'], guard=monitor.guard)
                    actual = gradients(); errors = {}
                    for name, ref in reference.items():
                        diff = (actual[name] - ref).double(); norm = float(ref.double().norm())
                        relative, absolute = float(diff.norm()) / max(norm, 1e-12), float(diff.abs().max())
                        require(relative <= r['gradient_relative_l2_tolerance'] or absolute <= r['gradient_absolute_tolerance'], 'Actual gradient L2 mismatch: ' + name)
                        require(absolute <= r['gradient_absolute_tolerance'] + r['gradient_maximum_scaled_tolerance'] * float(ref.abs().max()), 'Actual gradient maximum mismatch: ' + name)
                        errors[name] = dict(relative_l2=relative, maximum_absolute=absolute,
                            reference_norm=norm, reference_maximum=float(ref.abs().max()))
                    require(abs(result['loss'] - expected_loss) <= r['loss_absolute_tolerance'], 'Actual complete-source objective differs')
                    result.update(gradient_comparison=errors, reference_loss=expected_loss, optimizer_updates=0)
                    del reference, actual
                else:
                    result = backward_white_target_ranking(h, actor.bridge.density, actor.bridge.stop, ink,
                        ranking_strength=r['ranking_strength'], white_strength=r['white_strength'], margin_nats=r['margin_nats'],
                        microbatch_glyphs=r['book_microbatch_glyphs'], guard=monitor.guard)
                    g = gradients(); result['gradient_norms'] = {n: float(v.norm()) for n, v in g.items()}; del g
                    torch.nn.utils.clip_grad_norm_(list(parameters.values()), r['gradient_clip'], error_if_nonfinite=True)
                    optimizer = torch.optim.AdamW(list(parameters.values()), lr=r['learning_rate'], weight_decay=r['weight_decay'])
                    optimizer.step(); monitor.guard()
                    require(len(optimizer.state) == 131 and {int(v['step']) for v in optimizer.state.values()} == {1}, 'Every actual Adam counter required')
                    changed = {n: int((p.detach().cpu() != initial[n]).sum()) for n, p in parameters.items()}
                    require(all(v > 0 for v in changed.values()), 'An actual learned tensor did not update')
                    result.update(optimizer_updates=1, changed_elements=changed)
                    restore_trainable(actor, initial); del optimizer
                frozen_ok(); actor.zero_grad(set_to_none=True)
                require(tensor_inventory(trainable_state(actor)) == r['initial_trainable_inventory'], 'Preflight carried trained weights')
                require(result['white_targets'] == white_count and (result['white_ranking_loss'] > 0) == bool(white_count), 'New term must act on real whitespace only')
                result.update(case=job['id'], target_glyphs=len(ink), geometry=geometry,
                    frozen_parameters_and_affines_unchanged=True, updates_carried=False)
                write_new(folder / (job['id'] + '.json'), result); results.append(result)
                del h, memory, prompt, ink
            write_new(folder / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC), cases=results,
                model_account=account, provenance=provenance, actual_trainable_tensors=131,
                all_updates_reset=True, actual_optimizer_updates=4, new_autonomous_images=0,
                language_quality_unmeasured=True, goal_achieved=False, **monitor.receipt()))
            monitor.status('complete')
    except BaseException as caught:
        error = caught; monitor.status('failed', error=repr(caught)); raise
    finally:
        for hook in hooks: hook.remove()
        write_new(folder / 'end.json', dict(complete=error is None and (folder / 'report.json').exists(),
            error=repr(error) if error else None, **monitor.receipt()))
        atomic_json(folder / 'status.json', dict(load(folder / 'status.json'), pid=None))
    return dict(complete=True, actual_optimizer_updates=4, updates_carried=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['register', 'validate', 'run']); args = parser.parse_args()
    print(json.dumps(register() if args.action == 'register' else dict(valid=bool(validate())) if args.action == 'validate' else run()))
