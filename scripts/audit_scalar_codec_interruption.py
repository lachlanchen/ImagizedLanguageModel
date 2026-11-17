#!/usr/bin/env python3
"""Audit the saved completed control and untouched interrupted hard-code arm."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

from scripts import diagnose_visual_context_gradients as original

ROOT = original.ROOT
RUN = ROOT / 'artifacts/scalar_codec_comparison_v1'
REG = ROOT / 'references/evidence/scalar_codec_comparison_registration_20261118.json'
PUBLIC = ROOT / 'references/evidence/scalar_codec_interrupted_result_20261118.json'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit only')
    original.require(not PUBLIC.exists(), 'Preserve every audit')
    import numpy as np
    import torch
    from ilm.visual_lm.binary_plan_glyph_density import xor_correction
    from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.occupancy_glyph_density import encode_occupancy
    from ilm.visual_lm.spatial_plan_metrics import plan_codec_metrics
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2)
    began = time.perf_counter()
    samples = []
    def guard():
        host = host_resources()
        samples.append(host)
        original.require(host['available_gib'] >= 24 and time.perf_counter() - began < 180, 'CPU reserve/deadline')
        original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 4 * 1024**3, 'CPU RSS ceiling')
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = host_resources()
        original.require(host['available_gib'] >= 48 and host['some_pressure_avg10'] <= .5 and host['full_pressure_avg10'] <= .1, 'CPU admission')
        terminal = original.load(ROOT / 'artifacts/scalar_codec_comparison_v1.exit.json')
        status = original.load(RUN / 'status.json')
        original.require(terminal['phase'] == 'terminal' and terminal['exit_code'] == 1, 'Retain original interrupted execution')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        for pid, ticks in [(status['pid'], status['start_ticks']), (terminal['supervisor_pid'], terminal['supervisor_start_ticks']), (terminal['timeout_pid'], terminal['timeout_start_ticks'])]:
            stat = Path(f'/proc/{pid}/stat')
            original.require(boot != terminal['boot_id'] or not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[19] != ticks, 'Original owner still live')
        reg = original.load(REG)
        for name, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT / name) == digest, 'Bound source changed: ' + name)
        paths = [REG, RUN / 'status.json', RUN / 'failure.json', RUN / 'continuous.json',
                 RUN / 'common_initial.pt', RUN / 'source_order.pt', RUN / 'failure_state.pt',
                 ROOT / 'artifacts/scalar_codec_comparison_v1.exit.json', ROOT / 'artifacts/scalar_codec_comparison_v1.log']
        inputs = {str(p.relative_to(ROOT)): original.sha(p) for p in paths}
        failure = original.load(RUN / 'failure.json')
        original.require(failure['completed_arms'] == 1 and failure['error'] == 'Zero bottleneck/drawing gradient', 'Different interruption')
        initial = torch.load(RUN / 'common_initial.pt', weights_only=True, map_location='cpu')
        failed = torch.load(RUN / 'failure_state.pt', weights_only=True, map_location='cpu')
        original.require(failed['mode'] == 'binary' and tensor_state_sha(failed['model']) == tensor_state_sha(initial), 'Hard-code trial weights changed')
        original.require(all(not (RUN / name).exists() for name in ['binary_ledger.jsonl', 'binary.json', 'binary_final.pt', 'binary_initial.npz', 'binary_final.npz', 'result.json']), 'Unexpected hard-code retained result')
        arm = original.load(RUN / 'continuous.json')
        original.require(arm['mode'] == 'continuous' and arm['updates'] == 6000 and arm['retained_training_draws'] == 768000, 'Incomplete continuous arm')
        original.require(arm['preflight']['exact_reset'] and arm['preflight']['before']['positions'] == arm['preflight']['after']['positions'] == 64041, 'Complete real control preflight required')
        original.require(arm['optimizer_replay'] == dict(update=3, exact_loss_gradient_model_optimizer=True, extra_trial_images=128, retained_extra_updates=0), 'Optimizer replay differs')
        original.require(arm['initial_tensor_sha256'] == tensor_state_sha(initial), 'Common initial weights differ')
        checkpoint = ROOT / arm['checkpoint_file']
        original.require(original.sha(checkpoint) == arm['checkpoint_sha256'], 'Final control changed')
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        original.require(saved['updates'] == 6000 and saved['mode'] == 'continuous' and saved['scalar_features'] == 128 and saved['code_bits'] == 256 and saved['scalar_levels'] == 4, 'Final metadata differs')
        original.require(tensor_state_sha(saved['model']) == arm['final_tensor_sha256'] and sum(v.numel() for v in saved['model'].values()) == 1768393, 'Final control tensors differ')
        inputs[str(checkpoint.relative_to(ROOT))] = original.sha(checkpoint)
        partition = torch.load(ROOT / 'artifacts/binary_codec_preparation_v1/partition.pt', weights_only=True, map_location='cpu')
        source = torch.load(ROOT / 'artifacts/nonlinear_feedback_alignment_v1/prepared.pt', weights_only=True, map_location='cpu')
        target = torch.from_numpy(np.unpackbits(source['packed_ink'].numpy(), axis=1, bitorder='big').reshape(-1, 1, 32, 32).copy())
        order = torch.load(RUN / 'source_order.pt', weights_only=True, map_location='cpu')
        original.require(torch.equal(order, balanced_codec_support(partition, reg['order_seed'])['order']), 'Actual source order differs')
        old_initial = torch.load(ROOT / 'artifacts/spatial_capacity_comparison_v1/common_initial.pt', weights_only=True, map_location='cpu')
        shared = [n for n in old_initial if n.startswith('decoder.') or (n.startswith('encoder.') and int(n.split('.')[1]) < 8)]
        original.require(all(torch.equal(initial[n], old_initial[n]) for n in shared) and sum(initial[n].numel() for n in shared) == 1766209, 'Common CNN changed')
        ledger = ROOT / arm['ledger_file']
        original.require(original.sha(ledger) == arm['ledger_sha256'], 'Control ledger changed')
        rows = [json.loads(line) for line in ledger.read_text().splitlines()]
        original.require(len(rows) == 6000, 'Incomplete retained updates')
        for update, row in enumerate(rows, 1):
            indices = order[torch.arange((update - 1) * 128, update * 128) % len(order)]
            lr = .0002 * update / 250 if update <= 250 else .00002 + (.0002 - .00002) * .5 * (1 + math.cos(math.pi * (update - 250) / 5750))
            original.require(row['update'] == update and row['positions'] == 128 and row['learning_rate'] == lr and row['batch_index_sha256'] == hashlib.sha256(indices.numpy().tobytes()).hexdigest(), 'Source/update schedule differs')
            original.require(all(math.isfinite(row[k]) for k in ['loss', 'bce', 'edge', 'dice', 'gradient_norm']), 'Nonfinite retained update')
        original.require(arm['training_curve'] == [r for r in rows if r['update'] == 1 or r['update'] % 100 == 0], 'Curve differs')
        inputs[str(ledger.relative_to(ROOT))] = original.sha(ledger)
        audits = []
        for stage in ['initial', 'final']:
            guard()
            report = arm[stage]
            path = ROOT / report['raw_file']
            original.require(original.sha(path) == report['raw_sha256'], 'Saved reconstruction changed')
            inputs[str(path.relative_to(ROOT))] = original.sha(path)
            with np.load(path) as arrays:
                ink, values, events = arrays['ink'], arrays['plan'], arrays['correction_events']
            original.require(ink.dtype == np.uint8 and ink.shape == (68948, 1, 32, 32) and np.isin(ink, [0, 1]).all(), 'Complete real pixels required')
            original.require(values.dtype == np.float32 and values.shape == (68948, 128) and np.isfinite(values).all() and ((values >= -1) & (values <= .5)).all(), 'Complete scalar features required')
            if stage == 'initial':
                quantized = np.rint(2 * values) / 2
                original.require((quantized == 0).all(), 'Initial hard quantizer not identically zero')
            recounted = []
            for start in range(0, 68948, 256):
                guard()
                output = torch.from_numpy(ink[start:start + 256])
                expected = target[start:start + 256]
                residual = xor_correction(output, expected)
                original.require(torch.equal(xor_correction(output, residual), expected), 'XOR recovery differs')
                recounted.append(encode_occupancy(residual, 'tree').valid.sum(1))
            recounted = torch.cat(recounted)
            original.require(np.array_equal(recounted.numpy(), events), 'Every correction event required')
            metrics = plan_codec_metrics(torch.from_numpy(ink), target, partition, recounted, 'continuous', 256)
            original.require(metrics == report['metrics'], 'All-image metric recount differs')
            audits.append(dict(stage=stage, images=68948, all_pixels_and_correction_events_recounted=True, metrics=metrics))
        result = dict(schema='ilm-scalar-codec-interrupted-result-v1', registration_sha256=original.sha(REG),
                      source_commit='0494b2018128b30040e2704155517f5968370589', input_sha256=inputs,
                      comparison_complete=False, retained_hard_updates=0, retained_continuous_updates=6000,
                      hard_weights_exact_initial=True, hard_generation_or_fidelity_measured=False,
                      original_failure_state_metadata_updates=failed['updates'],
                      failure_metadata_interpretation='Stale prior-arm counter; binary weights equal initial and no binary ledger/update exists. Preserve original file.',
                      all_initial_8825344_quantized_features_zero=True,
                      zero_expansion_weight_gradient_explanation='Expansion-weight gradient is zero when every supplied quantized feature is zero. This does not imply zero encoder or total decoder gradient; those norms were not retained.',
                      continuous_control=arm, audits=audits, continuous_ceiling_pass=arm['final']['metrics']['held']['pixels']['f1'] >= reg['quality']['continuous_held_f1'],
                      no_retry_admitted=True, full_language_fit_authorized=False, autonomous_language_measured=False,
                      support=False, goal_achieved=False, auditor_sha256=original.sha(Path(__file__)),
                      resources=dict(elapsed_seconds=time.perf_counter() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                     minimum_available_host_gib=min(v['available_gib'] for v in samples), cpu_threads=2))
        original.write_new(PUBLIC, result)
        print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)), sha256=original.sha(PUBLIC), continuous_held_f1=arm['final']['metrics']['held']['pixels']['f1'], hard_result='unmeasured', comparison_complete=False)), flush=True)


if __name__ == '__main__':
    main()
