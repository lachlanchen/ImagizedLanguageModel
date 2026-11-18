#!/usr/bin/env python3
"""Compare strict actual-source optimizer replay with one global numeric flag."""
import argparse
import copy
import importlib.metadata
import json
import os
import resource
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
RUN = ROOT / 'artifacts/residual_motor_replay_diagnostic_v1'
REG = ROOT / 'references/evidence/residual_motor_replay_diagnostic_registration_20261118.json'
FILES = ('scripts/diagnose_residual_motor_replay.py', 'references/residual_motor_replay_diagnostic_protocol_20261118.md',
         'references/evidence/residual_motor_replay_failure_20261118.json',
         'references/evidence/residual_motor_loaded_comparison_registration_20261118.json',
         'artifacts/residual_plan_motor_loaded_v1/initial_refinement.pt',
         'artifacts/residual_plan_motor_loaded_v1/ledger.jsonl',
         'artifacts/residual_plan_motor_loaded_v1/failure_refinement.pt',
         'artifacts/residual_plan_motor_preparation_v1/physical_targets.pt',
         'artifacts/binary_codec_preparation_v1/partition.pt',
         'artifacts/spatial_capacity_comparison_v1/source_order.pt',
         'artifacts/spatial_capacity_comparison_v1/binary_final.pt',
         'ilm/visual_lm/residual_motor_data.py', 'ilm/visual_lm/residual_plan_motor.py',
         'ilm/visual_lm/residual_motor_resources.py', 'ilm/visual_lm/occupancy_glyph_density.py',
         'ilm/visual_lm/conditional_pixel_density.py', 'ilm/visual_lm/binary_plan_glyph_density.py',
         'ilm/visual_lm/spatial_plan_codec.py', 'ilm/visual_lm/full_rank_visual_span.py',
         'ilm/visual_lm/glyph_reader_monitor.py', 'ilm/visual_lm/visual_context_deployment_resources.py',
         'scripts/train_spatial_capacity_codec_comparison.py', 'scripts/diagnose_visual_context_gradients.py')


def register():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU registration only')
    original.write_new(REG, dict(schema='ilm-residual-motor-replay-diagnostic-v1',
        input_sha256={n: original.sha(ROOT / n) for n in FILES}, packages={n: importlib.metadata.version(n) for n in ['torch', 'numpy']},
        modes=[False, True], intervention='torch.use_deterministic_algorithms(mode,warn_only=False) only',
        warmup_reconstruction_updates=2, repetitions_of_update_three=4, batch_images=128,
        total_discarded_trial_updates=12, total_trial_image_exposures=1536,
        unchanged_original_cudnn_and_cublas_flags=True, original_model_and_loss_unchanged=True,
        resources=dict(maximum_seconds=240, maximum_pressure_wait_seconds=60, minimum_entry_host_gib=40,
                       minimum_runtime_host_gib=24, maximum_allocated_gpu_gib=8, maximum_process_rss_gib=8, cpu_threads=2),
        retained_updates=0, no_image_generation=True, automatic_retry=False, full_language_fit_authorized=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=original.sha(REG))), flush=True)


def run():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
                     and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Physical deterministic-environment GPU0 required')
    original.require(not RUN.exists(), 'Preserve every diagnostic')
    reg = original.load(REG)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    for name in [str(REG.relative_to(ROOT)), *[n for n in FILES if not n.startswith('artifacts/')]]:
        original.require(subprocess.check_output(['git', 'show', 'HEAD:' + name]) == (ROOT / name).read_bytes(), 'Commit diagnostic bindings')
    original.require(subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], text=True, timeout=30).split()[0] == commit, 'Push before numerical diagnostic')
    import numpy as np
    import torch
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.occupancy_glyph_density import encode_occupancy
    from ilm.visual_lm.residual_motor_data import decode_motor_patterns
    from ilm.visual_lm.residual_motor_resources import physical_motor_gpu
    from ilm.visual_lm.residual_plan_motor import ResidualPlanMotor
    from ilm.visual_lm.spatial_plan_codec import SpatialPlanCodec
    from scripts.train_spatial_capacity_codec_comparison import nested_sha
    RUN.mkdir()
    monitor = GlyphReaderMonitor(RUN, 240)
    monitor.pressure_wait_allowance_seconds = 60
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('Replay diagnostic deadline')))
    previous_global = torch.are_deterministic_algorithms_enabled()
    results = []
    try:
        for name, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT / name) == digest, 'Diagnostic source changed: ' + name)
        original.require(reg['packages'] == {n: importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
        with physical_motor_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            def guard():
                monitor.guard()
                original.require(torch.cuda.max_memory_allocated() / 1024**3 <= 8 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3, 'Replay diagnostic memory cap')
            data = torch.load(ROOT / 'artifacts/residual_plan_motor_preparation_v1/physical_targets.pt', weights_only=True, map_location='cpu')
            order = torch.load(ROOT / 'artifacts/spatial_capacity_comparison_v1/source_order.pt', weights_only=True, map_location='cpu')
            partition = torch.load(ROOT / 'artifacts/binary_codec_preparation_v1/partition.pt', weights_only=True, map_location='cpu')
            selected = order[:384]
            original.require(not bool(partition['held'][selected].any()), 'Only actual training images')
            patterns = data['plan_patterns'][selected]
            bits = decode_motor_patterns(patterns).to(device)
            def unpack(name):
                packed = data[name][selected].numpy()
                return torch.from_numpy(np.unpackbits(packed, axis=1, bitorder='big').reshape(-1, 1, 32, 32).copy()).to(device)
            bases, residuals = unpack('base_packed'), unpack('residual_packed')
            initial = torch.load(ROOT / 'artifacts/residual_plan_motor_loaded_v1/initial_refinement.pt', weights_only=True, map_location='cpu')
            initial_hash = tensor_state_sha(initial)
            frozen = torch.load(ROOT / 'artifacts/spatial_capacity_comparison_v1/binary_final.pt', weights_only=True, map_location='cpu')
            codec = SpatialPlanCodec(16)
            codec.load_state_dict(frozen['model'])
            fixed_hash = tensor_state_sha(codec.state_dict())
            motor = ResidualPlanMotor(codec).to(device).train()
            def cpu_tensors(values):
                return {n: v.detach().cpu().clone() for n, v in values.items()}
            def optimizer_tensors(value, prefix=''):
                result = {}
                if isinstance(value, torch.Tensor):
                    result[prefix] = value.detach().cpu().clone()
                elif isinstance(value, dict):
                    for key, child in value.items():
                        result.update(optimizer_tensors(child, prefix + '/' + str(key)))
                elif isinstance(value, (list, tuple)):
                    for key, child in enumerate(value):
                        result.update(optimizer_tensors(child, prefix + '/' + str(key)))
                return result
            def differences(first, later):
                original.require(first.keys() == later.keys(), 'Replay tensor coverage differs')
                changed = []
                for name, a in first.items():
                    b = later[name]
                    original.require(a.shape == b.shape and a.dtype == b.dtype, 'Replay shape/dtype differs')
                    if not torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)):
                        changed.append(dict(name=name, maximum_absolute_difference=float((a.double() - b.double()).abs().max()),
                                            unequal_values=int((a != b).sum()), elements=a.numel()))
                return changed
            def rng_digest():
                return nested_sha(dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
            for enabled in reg['modes']:
                guard()
                torch.use_deterministic_algorithms(enabled, warn_only=False)
                motor.refinement.load_state_dict(initial)
                original.require(tensor_state_sha(motor.refinement.state_dict()) == initial_hash, 'Common actual initial state differs')
                optimizer = torch.optim.AdamW(motor.refinement.parameters(), lr=.0002, betas=(.9, .95), weight_decay=.01)
                def step(update, capture=False):
                    guard()
                    monitor.position = dict(global_deterministic=enabled, update=update)
                    monitor.status('diagnosing_actual_optimizer_replay')
                    begin = (update - 1) * 128
                    end = update * 128
                    for group in optimizer.param_groups:
                        group['lr'] = .0002 * update / 250
                    optimizer.zero_grad(set_to_none=True)
                    before_rng = rng_digest()
                    states = motor.features(bits[begin:end], bases[begin:end])
                    loss = -motor.refinement.log_prob(states, encode_occupancy(residuals[begin:end], 'tree')).mean() / 1024
                    loss.backward()
                    original.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in motor.refinement.parameters()), 'Finite actual gradients required')
                    norm = torch.nn.utils.clip_grad_norm_(motor.refinement.parameters(), 1., error_if_nonfinite=True)
                    gradients = cpu_tensors({n: p.grad for n, p in motor.refinement.named_parameters()}) if capture else None
                    optimizer.step()
                    result = dict(loss=float(loss.detach()), gradient_norm=float(norm), rng_before=before_rng, rng_after=rng_digest(),
                                  model_sha256=tensor_state_sha(motor.refinement.state_dict()), optimizer_sha256=nested_sha(optimizer.state_dict()))
                    if capture:
                        result['gradient_sha256'] = tensor_state_sha(gradients)
                        return result, gradients, cpu_tensors(motor.refinement.state_dict()), optimizer_tensors(optimizer.state_dict())
                    return result
                warmup = [step(1), step(2)]
                saved_model = copy.deepcopy(motor.refinement.state_dict())
                saved_optimizer = copy.deepcopy(optimizer.state_dict())
                records = []
                baseline = None
                for repeat in range(4):
                    motor.refinement.load_state_dict(saved_model)
                    optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
                    actual = step(3, capture=True)
                    if baseline is None:
                        baseline = actual
                    row = dict(repetition=repeat, **actual[0])
                    row['gradient_differences'] = differences(baseline[1], actual[1])
                    row['model_differences'] = differences(baseline[2], actual[2])
                    row['optimizer_differences'] = differences(baseline[3], actual[3])
                    row['exact'] = actual[0] == baseline[0]
                    records.append(row)
                motor.refinement.load_state_dict(initial)
                original.require(tensor_state_sha(motor.refinement.state_dict()) == initial_hash and tensor_state_sha(motor.codec.state_dict()) == fixed_hash, 'Discarded trials must reset and preserve frozen codec')
                result = dict(global_deterministic=enabled, warmup=warmup, repetitions=records,
                              all_replays_exact=all(r['exact'] for r in records),
                              all_rng_unchanged=all(r['rng_before'] == r['rng_after'] for r in [*warmup, *records]),
                              reset_exact=True, retained_updates=0, discarded_trial_updates=6, trial_image_exposures=768)
                results.append(result)
                original.write_new(RUN / ('deterministic.json' if enabled else 'original.json'), result)
                del optimizer, saved_model, saved_optimizer, baseline
            for name, digest in reg['input_sha256'].items():
                original.require(original.sha(ROOT / name) == digest, 'Frozen diagnostic source changed: ' + name)
            qualified = not results[0]['all_replays_exact'] and results[1]['all_replays_exact'] and all(r['all_rng_unchanged'] and r['reset_exact'] for r in results)
            output = dict(schema='ilm-residual-motor-replay-diagnostic-result-v1', source_commit=commit,
                          registration_sha256=original.sha(REG), modes=results, deterministic_setting_qualified=qualified,
                          initial_tensor_sha256=initial_hash, frozen_codec_tensor_sha256=fixed_hash,
                          all_trial_updates_reset=True, retained_updates=0, total_trial_updates=12, total_trial_image_exposures=1536,
                          source_images=384, no_image_generation=True, resources=monitor.receipt(),
                          peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                          autonomous_language_measured=False, full_language_fit_authorized=False, automatic_retry=False)
            original.write_new(RUN / 'result.json', output)
            monitor.status('complete_residual_motor_replay_diagnostic')
            print(json.dumps(dict(qualified=qualified, original_exact=results[0]['all_replays_exact'], deterministic_exact=results[1]['all_replays_exact'])), flush=True)
    except BaseException as error:
        reset = None
        if 'motor' in locals() and 'initial' in locals():
            motor.refinement.load_state_dict(initial)
            reset = tensor_state_sha(motor.refinement.state_dict()) == initial_hash
        original.write_new(RUN / 'failure.json', dict(error_type=type(error).__name__, error=str(error), cleanup_reset_exact=reset, completed_modes=len(results), automatic_retry=False))
        monitor.status('failed_residual_motor_replay_diagnostic', error=str(error))
        raise
    finally:
        torch.use_deterministic_algorithms(previous_global, warn_only=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['register', 'run'])
    register() if parser.parse_args().action == 'register' else run()
