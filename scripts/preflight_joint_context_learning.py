#!/usr/bin/env python3
"""Real complete-source GPU gradient and memory proof, not language grading."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

BEGAN = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / 'artifacts/continuation_joint_writer_v1'
PARENT = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
RUN = ROOT / 'artifacts/joint_context_preflight_v1'
PUBLIC = ROOT / 'references/evidence/joint_context_preflight_registration_20261113.json'
NEW = ['ilm/visual_lm/joint_context_model.py', 'ilm/visual_lm/joint_continuation_learning.py',
       'scripts/preflight_joint_context_learning.py', 'tests/test_joint_continuation_learning.py',
       'references/joint_context_preflight_protocol.md']


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources():
    result = load(WRITER / 'registration.json')['source_sha256'].copy()
    for name in NEW:
        require((ROOT / name).read_bytes() == subprocess.check_output(
            ['git', 'show', 'HEAD:' + name], cwd=ROOT), 'Commit source first: ' + name)
        result[name] = sha(ROOT / name)
    for name, digest in result.items():
        require(sha(ROOT / name) == digest, 'Bound source changed: ' + name)
    return result


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only registration')
    require(not RUN.exists() and not PUBLIC.exists(), 'Preserve existing preflight')
    from scripts.preflight_visual_continuation import runtime
    from scripts.review_joint_writer_physical import write_new
    import torch
    require(not torch.cuda.is_initialized(), 'Registration must not initialize CUDA')
    rows = load(PARENT / 'training_manifest.json')['rows']
    require(len(rows) == 13766, 'Complete original eligible source manifest required')
    qa = next(i for i, r in enumerate(rows) if r['kind'] == 'qa'
              and 5 <= r['target_glyphs'] <= 63 and r['target_glyphs'] % 2)
    length = max(r['target_glyphs'] for r in rows if r['kind'] == 'book')
    book = next(i for i, r in enumerate(rows) if r['kind'] == 'book' and r['target_glyphs'] == length)
    cases = [dict(id=f'{kind}_{layout}', index=index, layout=layout, source=rows[index])
             for kind, index in [('qa', qa), ('long_book', book)]
             for layout in ['fixed_cells', 'proportional']]
    old = load(WRITER / 'registration.json'); bindings = old['bindings_sha256'].copy()
    def bind(path):
        bindings[str(path.relative_to(ROOT))] = sha(path)
    for path in [WRITER / 'registration.json', WRITER / 'initial_parameters.pt',
                 WRITER / 'cache/normalization.pt', WRITER / 'qualify/gate.json',
                 ROOT / 'references/evidence/joint_writer_blank_density_result_20261113.json']:
        bind(path)
    for case in cases:
        bind(ROOT / case['source']['images'][case['layout']]['file'])
    value = dict(schema='joint-context-real-source-preflight-v1', source_sha256=sources(),
        runtime=runtime(), input_sha256=bindings, cases=cases, qa_selection='first odd-length complete QA, 5..63 glyphs',
        book_selection='first complete book with maximal target length',
        maximum_seconds=1800, pressure_seconds=300, maximum_allocated_vram_gib=22,
        minimum_entry_host_gib=48, minimum_free_gpu_gib=20, minimum_runtime_host_gib=24,
        gradient_relative_l2_tolerance=.01, gradient_absolute_tolerance=1e-5,
        gradient_maximum_scaled_tolerance=.01, loss_absolute_tolerance=2e-5,
        qa_microbatch_glyphs=2, book_microbatch_glyphs=64,
        learning_rate=4e-7, weight_decay=.01, gradient_clip=1.,
        expected_trainable_parameters=5005760, stop_head_parameters_frozen=True,
        preflight_only=True, optimizer_updates=2, updates_carried_into_fit=False,
        new_autonomous_images=0, language_evaluation=False, goal_achieved=False)
    RUN.mkdir()
    write_new(RUN / 'registration.json', value); write_new(PUBLIC, value)
    return dict(registration_sha256=sha(PUBLIC), cases=[(x['id'], x['index'], x['source']['target_glyphs']) for x in cases])


def validate():
    from scripts.preflight_visual_continuation import runtime
    r = load(PUBLIC)
    require(r['source_sha256'] == sources() and r['runtime'] == runtime(), 'Registered source/runtime differs')
    require(PUBLIC.read_bytes() == (RUN / 'registration.json').read_bytes() == subprocess.check_output(
        ['git', 'show', 'HEAD:' + str(PUBLIC.relative_to(ROOT))], cwd=ROOT), 'Commit/push registration before GPU work')
    for name, digest in r['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound input changed: ' + name)
    return r


def execute():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Use only registered GPU0')
    r = validate()
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.continuation_training import TrainingImages, trainable_state, restore_trainable
    from ilm.visual_lm.glyph_joint_density import pack_glyphs
    from ilm.visual_lm.joint_context_model import build_joint_context_actor, differentiable_states
    from ilm.visual_lm.joint_continuation_learning import backward_complete_image_likelihood
    from ilm.visual_lm.joint_writer_training import Monitor
    from ilm.visual_lm.glyph_runtime_resources import full_reader_gpu
    from scripts.train_joint_continuation_writer import finish
    from scripts.train_semantic_glyph_language import numerics
    from scripts.train_visual_continuation import atomic_json
    folder = RUN / 'execution'; folder.mkdir(exist_ok=False)
    monitor = Monitor(folder, r['maximum_seconds'] - (time.perf_counter() - BEGAN),
                      r['pressure_seconds'], r['maximum_allocated_vram_gib'])
    monitor.status('admitting'); hooks = []; error = None
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission; numerics(); torch.set_num_threads(2); monitor.guard()
            actor, provenance, account = build_joint_context_actor(device)
            initial = trainable_state(actor); parameters = {n: p for n, p in actor.named_parameters() if p.requires_grad}
            frozen = [(p, p._version) for p in actor.parameters() if not p.requires_grad]
            affine = {n: b.detach().clone() for n, b in actor.bridge.named_buffers()
                      if n in {'semantic_mean', 'semantic_scale', 'density.condition_mean', 'density.condition_scale'}}
            require(len(affine) == 4, 'Four frozen feedback/condition affine buffers required')
            require(account['trainable_parameters'] == r['expected_trainable_parameters'], 'Wrong actual model')
            for layer in actor.dynamics.context.layers:
                hooks.append(layer.register_forward_pre_hook(lambda _m, _a: monitor.guard()))
            data = TrainingImages(PARENT); results = []
            def frozen_ok():
                require(all(p._version == v and p.grad is None for p, v in frozen), 'Frozen parameter changed or got a gradient')
                current = dict(actor.bridge.named_buffers())
                require(all(torch.equal(current[n], b) for n, b in affine.items()), 'Frozen affine buffer changed')
            def gradients():
                out = {}
                for n, p in parameters.items():
                    require(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'Missing/nonfinite gradient: ' + n)
                    out[n] = p.grad.detach().float().cpu().clone()
                require(all(float(v.norm()) > 0 for v in out.values()), 'Disconnected trainable tensor')
                return out
            for number, job in enumerate(r['cases']):
                monitor.position = dict(completed=number, total=4, case=job['id']); monitor.status('real_source_preflight')
                prompt, ink, row, layout = data.get(job['index'], int(job['layout'] == 'proportional'))
                require(row == job['source'] and layout == job['layout'], 'Preflight case differs')
                require(all(torch.equal(p.detach().cpu(), initial[n]) for n, p in parameters.items()), 'Carry-over before case')
                actor.zero_grad(set_to_none=True); began = time.perf_counter()
                h, memory, geometry = differentiable_states(actor, prompt, ink)
                if job['id'].startswith('qa_'):
                    gold = pack_glyphs(ink).to(device)
                    lp = actor.bridge.density(h[:-1], gold).float().log_softmax(-1)
                    nll = -lp.gather(-1, gold[..., None]).mean()
                    s = actor.bridge.stop(h).flatten()
                    eos = .5 * (F.softplus(s[:-1]).mean() + F.softplus(-s[-1]))
                    loss = nll + .1 * eos; expected_loss = float(loss.detach()); loss.backward()
                    reference = gradients(); frozen_ok()
                    del h, lp, nll, s, eos, loss
                    actor.zero_grad(set_to_none=True)
                    h, _, _ = differentiable_states(actor, prompt, ink, memory=memory)
                    result = backward_complete_image_likelihood(h, actor.bridge.density, actor.bridge.stop, ink,
                        microbatch_glyphs=r['qa_microbatch_glyphs'], guard=monitor.guard)
                    actual = gradients(); errors = {}
                    for name, ref in reference.items():
                        delta = (actual[name] - ref).double(); magnitude = float(ref.double().norm())
                        relative = float(delta.norm()) / max(magnitude, 1e-12)
                        absolute = float(delta.abs().max()); scale = float(ref.abs().max())
                        require(relative <= r['gradient_relative_l2_tolerance'] or absolute <= r['gradient_absolute_tolerance'],
                                'Gradient relative mismatch: ' + name)
                        require(absolute <= r['gradient_absolute_tolerance'] + r['gradient_maximum_scaled_tolerance'] * scale,
                                'Gradient maximum mismatch: ' + name)
                        errors[name] = dict(relative_l2=relative, maximum_absolute=absolute, reference_norm=magnitude)
                    require(abs(result['loss'] - expected_loss) <= r['loss_absolute_tolerance'], 'Complete QA loss differs')
                    # Actual target-prefix alignment in this new constructor.
                    with torch.no_grad():
                        feedback = actor.bridge.feedback_embedding((1-2*ink.float()).to(device))[None]
                        x = actor.dynamics.embeddings(feedback)
                        original = actor.dynamics(x, memory, checkpoint_layers=False)
                        cut = len(ink)//2; changed = x.clone()
                        changed[:, cut+1] = actor.bridge.feedback_embedding((2*ink[cut:cut+1].float()-1).to(device))
                        other = actor.dynamics(changed, memory, checkpoint_layers=False)
                        require(torch.equal(original[:, :cut+1], other[:, :cut+1])
                                and not torch.equal(original[:, cut+1:], other[:, cut+1:]), 'Target-prefix causal alignment failed')
                    result.update(gradient_comparison=errors, target_precedes_own_history_verified=True,
                                  reference_loss=expected_loss, optimizer_updates=0)
                    del reference, actual, feedback, x, original, changed, other
                else:
                    result = backward_complete_image_likelihood(h, actor.bridge.density, actor.bridge.stop, ink,
                        microbatch_glyphs=r['book_microbatch_glyphs'], guard=monitor.guard)
                    grads = gradients(); result['gradient_norms'] = {n: float(v.norm()) for n, v in grads.items()}
                    del grads
                    norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), r['gradient_clip'], error_if_nonfinite=True)
                    optimizer = torch.optim.AdamW(list(parameters.values()), lr=r['learning_rate'], weight_decay=r['weight_decay'])
                    optimizer.step(); monitor.guard()
                    require({int(s['step']) for s in optimizer.state.values()} == {1}
                            and len(optimizer.state) == len(parameters), 'Actual complete-source optimizer step missing')
                    changed = {n: int((p.detach().cpu() != initial[n]).sum()) for n, p in parameters.items()}
                    require(all(v > 0 for v in changed.values()), 'A trainable tensor did not update')
                    require(all(bool(torch.isfinite(p).all()) for p in parameters.values()), 'Nonfinite updated parameter')
                    result.update(gradient_norm_before_clip=float(norm), changed_elements=changed,
                                  optimizer_updates=1, adam_states=len(optimizer.state))
                    del optimizer
                    restore_trainable(actor, initial)
                frozen_ok(); actor.zero_grad(set_to_none=True)
                require(all(torch.equal(p.detach().cpu(), initial[n]) for n, p in parameters.items()), 'Initial state not restored')
                torch.cuda.synchronize(); result.update(case=job['id'], source_id=row['id'],
                    target_glyphs=len(ink), layout=layout, geometry=geometry, seconds=time.perf_counter()-began)
                results.append(result); atomic_json(folder / 'progress.json', dict(records=results, model_account=account))
                print(json.dumps({k: result[k] for k in ['case', 'target_glyphs', 'loss', 'seconds', 'optimizer_updates']}), flush=True)
                del h, memory; gc.collect()
            require(len(results) == 4 and sum(x['optimizer_updates'] for x in results) == 2, 'Incomplete actual preflight')
            frozen_ok(); monitor.guard()
            atomic_json(folder / 'report.json', dict(complete=True, registration_sha256=sha(PUBLIC),
                model_account=account, provenance=provenance, records=results,
                gradients_into_actual_continuation_and_head=True, frozen_parameters_unchanged=True,
                feedback_and_condition_affine_buffers_bit_exact=True,
                all_trainables_restored_to_registered_initialization=True, updates_carried_into_fit=False,
                optimizer_updates=2, new_autonomous_images=0, language_evaluated=False,
                total_process_instrumented_seconds=time.perf_counter()-BEGAN, **monitor.receipt()))
    except BaseException as exc:
        error = exc; raise
    finally:
        for hook in hooks:
            hook.remove()
        finish(folder, monitor, error)
    return dict(complete=True, cases=4, actual_optimizer_updates=2, language_unmeasured=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    args = parser.parse_args()
    print(json.dumps(register() if args.action == 'register' else execute()))
