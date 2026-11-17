#!/usr/bin/env python3
"""Fit and qualify the complete frozen-plan physical correction motor."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import resource
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
RUN = ROOT / 'artifacts/residual_plan_motor_comparison_v1'
REG = ROOT / 'references/evidence/residual_motor_comparison_registration_20261118.json'
FILES = ('scripts/train_residual_plan_motor.py', 'references/residual_motor_comparison_protocol_20261118.md',
         'ilm/visual_lm/residual_plan_motor.py', 'ilm/visual_lm/residual_motor_metrics.py',
         'tests/test_residual_plan_motor.py', 'references/evidence/residual_plan_motor_cpu_20261118.json',
         'references/residual_plan_motor_direction_20261118.md',
         'references/evidence/residual_motor_preparation_registration_20261118.json',
         'references/evidence/residual_motor_preparation_result_20261118.json',
         'artifacts/residual_plan_motor_preparation_v1/physical_targets.pt',
         'artifacts/residual_plan_motor_preparation_v1/result.json',
         'artifacts/spatial_capacity_comparison_v1/binary_final.pt',
         'references/evidence/spatial_capacity_comparison_result_20261118.json',
         'artifacts/binary_codec_preparation_v1/partition.pt',
         'artifacts/spatial_capacity_comparison_v1/source_order.pt',
         'ilm/visual_lm/spatial_plan_codec.py', 'ilm/visual_lm/binary_plan_glyph_density.py',
         'ilm/visual_lm/occupancy_glyph_density.py', 'ilm/visual_lm/conditional_pixel_density.py',
         'ilm/visual_lm/binary_codec_metrics.py', 'ilm/visual_lm/form_balanced_codec_support.py',
         'ilm/visual_lm/full_rank_visual_span.py', 'ilm/visual_lm/glyph_reader_monitor.py',
         'ilm/visual_lm/visual_context_deployment_resources.py',
         'scripts/train_spatial_capacity_codec_comparison.py', 'scripts/diagnose_visual_context_gradients.py')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    prepared = original.load(ROOT / 'references/evidence/residual_motor_preparation_result_20261118.json')
    original.require(prepared['all_original_target_bytes_recover'] and prepared['exact_original_physical_partition_retained'], 'Prepared complete sources required')
    original.write_new(REG, dict(schema='ilm-residual-motor-comparison-v1', input_sha256={n: original.sha(ROOT / n) for n in FILES},
        packages={n: importlib.metadata.version(n) for n in ['torch', 'numpy']},
        seed=202611214, order_seed=202611196, width=128, layers=2, heads=4, code_bits=256,
        frozen_codec_parameters=1770449, trainable_refinement_parameters=615314, physical_motor_parameters=2385763,
        source_images=68948, train_images=64041, held_images=4907, uniform_training_forms=8629,
        updates=6000, batch_images=128, evaluation_batch_images=128, retained_training_draws=768000,
        learning_rate=.0002, minimum_learning_rate=.00002, warmup_updates=250,
        adam_betas=[.9, .95], weight_decay=.01, gradient_clip=1., replay_update=3,
        objective='Exact conditional image NLL given supplied plan, per image divided by 1024; equal-form training mass',
        quality=dict(train_f1=.98, held_f1=.98, held_macro_f1=.98, held_macro_exact=.80,
                     maximum_f1_drop_from_continuous=.005, continuous_held_f1=.9964238498113617,
                     minimum_gain_over_base=.02, base_held_f1=.9436900709752929,
                     false_blank=0, false_ink=0, maximum_generated_event_ratio=.60,
                     old_direct_tree_mean_events=97.30248883216336, maximum_event_probability_error=5e-5),
        resources=dict(maximum_seconds=2400, maximum_pressure_wait_seconds=300, maximum_allocated_gpu_gib=8,
                       maximum_process_rss_gib=8, minimum_entry_host_gib=48, minimum_runtime_host_gib=24,
                       minimum_free_gpu_gib=20, minimum_free_disk_gib=48, cpu_threads=2),
        unchanged_base_retrained=False, only_final_checkpoint_qualifies=True, automatic_retry=False,
        autonomous_language_measured=False, full_language_fit_authorized=False, goal_achieved=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
                     and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Deterministic physical GPU0 required')
    original.require(not RUN.exists(), 'Preserve every execution')
    reg = original.load(REG)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in [str(REG.relative_to(ROOT)), *[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git', 'show', 'HEAD:' + name]) == (ROOT / name).read_bytes(), 'Commit bound sources before fit')
    original.require(subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], text=True, timeout=30).split()[0] == commit, 'Push before fit')
    import numpy as np
    import torch
    from ilm.visual_lm.binary_plan_glyph_density import unpack_plan, xor_correction
    from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.occupancy_glyph_density import encode_occupancy
    from ilm.visual_lm.residual_motor_metrics import residual_motor_metrics
    from ilm.visual_lm.residual_plan_motor import ResidualPlanMotor
    from ilm.visual_lm.spatial_plan_codec import SpatialPlanCodec
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_spatial_capacity_codec_comparison import nested_sha
    RUN.mkdir()
    monitor = GlyphReaderMonitor(RUN, 2400)
    monitor.pressure_wait_allowance_seconds = 300
    updates_done = 0
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('External residual motor deadline')))
    try:
        for name, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT / name) == digest, 'Bound source changed: ' + name)
        original.require(reg['packages'] == {n: importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            def guard():
                monitor.guard()
                original.require(torch.cuda.max_memory_allocated() / 1024**3 <= 8, 'Motor GPU ceiling')
                original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3, 'Motor RSS ceiling')
            prepared = original.load(ROOT / 'artifacts/residual_plan_motor_preparation_v1/result.json')
            original.require(original.sha(ROOT / prepared['cache_file']) == prepared['cache_sha256'], 'Prepared correction cache changed')
            data = torch.load(ROOT / prepared['cache_file'], weights_only=True, map_location='cpu')
            partition = torch.load(ROOT / 'artifacts/binary_codec_preparation_v1/partition.pt', weights_only=True, map_location='cpu')
            def unpack_images(packed):
                return torch.from_numpy(np.unpackbits(packed.numpy(), axis=1, bitorder='big').reshape(-1, 1, 32, 32).copy())
            base_cpu = unpack_images(data['base_packed'])
            residual_cpu = unpack_images(data['residual_packed'])
            target_cpu = base_cpu ^ residual_cpu
            patterns = data['plan_patterns']
            bits_cpu = unpack_plan(patterns)
            bits, bases, residuals = bits_cpu.to(device), base_cpu.to(device), residual_cpu.to(device)
            source = balanced_codec_support(partition, reg['order_seed'])
            order = source['order']
            original.require(torch.equal(order, torch.load(ROOT / 'artifacts/spatial_capacity_comparison_v1/source_order.pt', weights_only=True, map_location='cpu')), 'Original source order differs')
            torch.save(order, RUN / 'source_order.pt')
            frozen_path = ROOT / 'artifacts/spatial_capacity_comparison_v1/binary_final.pt'
            frozen = torch.load(frozen_path, weights_only=True, map_location='cpu')
            codec = SpatialPlanCodec(16)
            codec.load_state_dict(frozen['model'])
            fixed_hash = tensor_state_sha(codec.state_dict())
            torch.manual_seed(reg['seed'])
            motor = ResidualPlanMotor(codec).to(device).train()
            original.require(sum(p.numel() for p in motor.parameters()) == 2385763 and sum(p.numel() for p in motor.refinement.parameters()) == 615314, 'Actual motor dimensions differ')
            initial = copy.deepcopy(motor.refinement.state_dict())
            initial_hash = tensor_state_sha(initial)
            torch.save(initial, RUN / 'initial_refinement.pt')
            def objective(indices, backward=False, balanced=False):
                total = 0.
                for start in range(0, len(indices), 128):
                    guard()
                    idx = indices[start:start + 128]
                    selected = idx.to(device)
                    states = motor.features(bits[selected], bases[selected])
                    nll = -motor.refinement.log_prob(states, encode_occupancy(residuals[selected], 'tree')) / 1024
                    if balanced:
                        nll = nll * source['complete_source_weights'][idx].to(device=device, dtype=torch.float32)
                    value = nll.sum() / len(indices)
                    if backward:
                        value.backward()
                    total += float(value.detach())
                return dict(images=len(indices), image_nll_per_pixel=total)
            def evaluate(stage):
                monitor.position = dict(stage=stage)
                monitor.status('generating_every_residual_motor_image')
                outputs = np.zeros((68948, 1, 32, 32), dtype=np.uint8)
                generated = np.zeros_like(outputs)
                actions = np.zeros((68948, 342), dtype=np.uint8)
                depths, rows, columns = np.zeros_like(actions), np.zeros_like(actions), np.zeros_like(actions)
                lengths = np.zeros(68948, dtype=np.uint16)
                event_logp = np.zeros((68948, 342), dtype=np.float32)
                remaining = []
                maximum_error = 0.
                motor.eval()
                with torch.no_grad():
                    for start in range(0, len(bits), 128):
                        guard()
                        end = min(start + 128, len(bits))
                        emitted, trace = motor.generate(bits[start:end])
                        original.require(torch.equal(trace['base'], bases[start:end]), 'Public motor base differs from authenticated saved D(B)')
                        correction = trace['residual']
                        sequence = encode_occupancy(correction, 'tree')
                        original.require(len(trace['correction_trace']) == sequence.actions.shape[1], 'Emitted tree length differs')
                        for pos, event in enumerate(trace['correction_trace']):
                            valid = torch.tensor(event['active'], device=device)
                            meta = torch.tensor(event['meta'], device=device)
                            actual = torch.tensor(event['action'], device=device)
                            original.require(torch.equal(valid, sequence.valid[:, pos]), 'Actual activity differs from emitted pixels')
                            for value, expected in [(actual, sequence.actions[:, pos]), (meta[:, 0], sequence.depths[:, pos]), (meta[:, 1], sequence.rows[:, pos]), (meta[:, 2], sequence.columns[:, pos])]:
                                original.require(torch.equal(value[valid], expected[valid]), 'Actual causal action/region differs from emitted correction')
                        states = motor.features(bits[start:end], trace['base'])
                        parallel = motor.refinement(states, sequence).log_softmax(-1).gather(-1, sequence.actions[..., None]).squeeze(-1)
                        features = motor.refinement.condition_features(states)
                        previous = None
                        caches = None
                        cached = torch.zeros_like(parallel)
                        for pos in range(sequence.actions.shape[1]):
                            logits, caches = motor.refinement.step(features, previous, sequence.depths[:, pos], sequence.rows[:, pos], sequence.columns[:, pos], sequence.allow_zero[:, pos], pos, caches)
                            valid = sequence.valid[:, pos]
                            previous = sequence.actions[:, pos]
                            original.require(torch.equal(logits.argmax(-1)[valid], previous[valid]), 'Actual greedy generation differs from cached replay')
                            cached[:, pos] = logits.log_softmax(-1).gather(-1, previous[:, None]).squeeze(-1)
                        error = float((cached - parallel).abs()[sequence.valid].max())
                        maximum_error = max(maximum_error, error)
                        original.require(error <= 5e-5, 'Actual emitted-prefix probability replay differs')
                        count = sequence.actions.shape[1]
                        for array, value in [(actions, sequence.actions), (depths, sequence.depths), (rows, sequence.rows), (columns, sequence.columns)]:
                            array[start:end, :count] = torch.where(sequence.valid, value, torch.zeros_like(value)).to(torch.uint8).cpu().numpy()
                        lengths[start:end] = sequence.valid.sum(1).cpu().numpy()
                        event_logp[start:end, :count] = torch.where(sequence.valid, cached, torch.zeros_like(cached)).cpu().numpy()
                        outputs[start:end] = emitted.cpu().numpy()
                        generated[start:end] = correction.cpu().numpy()
                        remaining.append(encode_occupancy(xor_correction(emitted, target_cpu[start:end].to(device)), 'tree').valid.sum(1).cpu())
                remaining = torch.cat(remaining)
                if stage == 'initial':
                    original.require(np.array_equal(outputs, base_cpu.numpy()) and not generated.any(), 'Initial motor must equal every unchanged base')
                metrics = residual_motor_metrics(torch.from_numpy(outputs), target_cpu, partition, remaining, torch.from_numpy(lengths.astype(np.int64)))
                path = RUN / f'{stage}_outputs.npz'
                np.savez_compressed(path, ink=outputs, correction=generated, actions=actions, depths=depths, rows=rows, columns=columns,
                                    lengths=lengths, executed_prefix_log_probability=event_logp, remaining_error_events=remaining.numpy())
                motor.train()
                original.require(tensor_state_sha(motor.codec.state_dict()) == fixed_hash, 'Frozen codec tensors changed')
                return dict(metrics=metrics, raw_file=str(path.relative_to(ROOT)), raw_sha256=original.sha(path),
                            all_68948_images_and_actual_traces=True, maximum_event_probability_error=maximum_error,
                            actual_generation_not_true_residual_replay=True)
            monitor.status('probing_complete_residual_motor_support')
            motor.zero_grad(set_to_none=True)
            before = objective(partition['train_index'], backward=True, balanced=True)
            original.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in motor.refinement.parameters()), 'Missing/nonfinite motor gradient')
            norms = {name: float(module.weight.grad.norm()) for name, module in [('condition', motor.refinement.condition[1]), ('root', motor.refinement.root_output), ('mask', motor.refinement.mask_output)]}
            original.require(all(v > 0 for v in norms.values()), 'Zero condition/root/mask gradient')
            trial = torch.optim.AdamW(motor.refinement.parameters(), lr=.0002, betas=(.9, .95), weight_decay=.01)
            norm = torch.nn.utils.clip_grad_norm_(motor.refinement.parameters(), 1., error_if_nonfinite=True)
            trial.step()
            original.require(tensor_state_sha(motor.refinement.state_dict()) != initial_hash, 'No actual refinement update')
            with torch.no_grad():
                after = objective(partition['train_index'], balanced=True)
            motor.refinement.load_state_dict(initial)
            motor.zero_grad(set_to_none=True)
            del trial
            original.require(tensor_state_sha(motor.refinement.state_dict()) == initial_hash and tensor_state_sha(motor.codec.state_dict()) == fixed_hash, 'Preflight reset/frozen codec differs')
            preflight = dict(before=before, after=after, gradient_norms=norms, total_gradient_norm=float(norm), exact_reset=True, retained_updates=0)
            initial_evaluation = evaluate('initial')
            optimizer = torch.optim.AdamW(motor.refinement.parameters(), lr=.0002, betas=(.9, .95), weight_decay=.01)
            ledger_path = RUN / 'ledger.jsonl'
            curve = []
            with ledger_path.open('x') as ledger:
                for update in range(1, 6001):
                    guard()
                    monitor.position = dict(update=update)
                    if update == 1 or update % 100 == 0:
                        monitor.status('fitting_joint_physical_motor')
                    indices = order[torch.arange((update - 1) * 128, update * 128) % len(order)]
                    lr = .0002 * update / 250 if update <= 250 else .00002 + (.0002 - .00002) * .5 * (1 + math.cos(math.pi * (update - 250) / 5750))
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    model_copy = copy.deepcopy(motor.refinement.state_dict()) if update == 3 else None
                    optimizer_copy = copy.deepcopy(optimizer.state_dict()) if update == 3 else None
                    def step():
                        optimizer.zero_grad(set_to_none=True)
                        loss = objective(indices, backward=True)
                        total_norm = torch.nn.utils.clip_grad_norm_(motor.refinement.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        return dict(**loss, gradient_norm=float(total_norm))
                    loss = step()
                    if update == 3:
                        first_model = tensor_state_sha(motor.refinement.state_dict())
                        first_optimizer = nested_sha(optimizer.state_dict())
                        motor.refinement.load_state_dict(model_copy)
                        optimizer.load_state_dict(optimizer_copy)
                        other = step()
                        original.require(loss == other and first_model == tensor_state_sha(motor.refinement.state_dict()) and first_optimizer == nested_sha(optimizer.state_dict()), 'Optimizer replay differs')
                        del model_copy, optimizer_copy
                    row = dict(update=update, learning_rate=lr, batch_index_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(), **loss)
                    ledger.write(json.dumps(row) + '\n')
                    updates_done = update
                    if update == 1 or update % 100 == 0:
                        ledger.flush()
                        curve.append(row)
            final_evaluation = evaluate('final')
            checkpoint = RUN / 'final_refinement.pt'
            torch.save(dict(architecture='residual-plan-motor-v1', width=128, layers=2, heads=4, code_bits=256,
                            frozen_codec_file=str(frozen_path.relative_to(ROOT)), frozen_codec_sha256=original.sha(frozen_path),
                            refinement=motor.refinement.state_dict(), updates=6000, initial_tensor_sha256=initial_hash), checkpoint)
            b = final_evaluation['metrics']
            q = reg['quality']
            checks = dict(train_f1=b['train']['pixels']['f1'] >= .98, held_f1=b['held']['pixels']['f1'] >= .98,
                          held_macro_f1=b['held']['macro_form_f1'] >= .98, held_macro_exact=b['held']['macro_form_exact_rate'] >= .80,
                          retention=b['held']['pixels']['f1'] >= q['continuous_held_f1'] - .005,
                          gain=b['held']['pixels']['f1'] - q['base_held_f1'] >= .02,
                          false_blank=b['all']['pixels']['false_blank'] == 0, false_ink=b['all']['pixels']['false_ink'] == 0,
                          generated_event_work=b['all']['mean_plan_and_generated_events'] <= .60 * q['old_direct_tree_mean_events'])
            for name, digest in reg['input_sha256'].items():
                original.require(original.sha(ROOT / name) == digest, 'Bound source changed during fit: ' + name)
            result = dict(schema='ilm-residual-motor-comparison-result-v1', registration_sha256=original.sha(REG), source_commit=commit,
                          updates=6000, retained_training_draws=768000, optimizer_replay=dict(update=3, exact=True, extra_trial_images=128),
                          preflight=preflight, initial=initial_evaluation, final=final_evaluation, training_curve=curve,
                          initial_tensor_sha256=initial_hash, final_tensor_sha256=tensor_state_sha(motor.refinement.state_dict()),
                          frozen_codec_tensor_sha256=fixed_hash, frozen_codec_unchanged=True, unchanged_control_retrained=False,
                          checkpoint_file=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=original.sha(checkpoint),
                          ledger_file=str(ledger_path.relative_to(ROOT)), ledger_sha256=original.sha(ledger_path),
                          source_order_file=str((RUN / 'source_order.pt').relative_to(ROOT)), source_order_sha256=original.sha(RUN / 'source_order.pt'),
                          physical_motor_parameters=2385763, trainable_refinement_parameters=615314,
                          support_checks=checks, support=all(checks.values()), resources=monitor.receipt(),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          autonomous_language_measured=False, full_language_fit_authorized=False, goal_achieved=False)
            original.write_new(RUN / 'result.json', result)
            monitor.status('complete_residual_plan_motor')
            print(json.dumps(dict(support=result['support'], checks=checks, held=b['held'])), flush=True)
    except BaseException as error:
        if 'motor' in locals():
            torch.save(dict(refinement=motor.refinement.state_dict(), retained_updates=updates_done), RUN / 'failure_refinement.pt')
        original.write_new(RUN / 'failure.json', dict(error_type=type(error).__name__, error=str(error), retained_updates=updates_done, automatic_retry=False))
        monitor.status('failed_residual_plan_motor', error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
