#!/usr/bin/env python3
"""Independently recount every actual image and causal physical correction."""
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
RUN = ROOT / 'artifacts/residual_plan_motor_loaded_v1'
REG = ROOT / 'references/evidence/residual_motor_loaded_comparison_registration_20261118.json'
PUBLIC = ROOT / 'references/evidence/residual_plan_motor_comparison_result_20261118.json'
OUT = ROOT / 'references/evidence/residual_plan_motor_comparison_20261118'


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU audit only')
    original.require(not PUBLIC.exists() and not OUT.exists(), 'Preserve every audit and plate')
    terminal = original.load(ROOT / 'artifacts/residual_plan_motor_loaded_v1.exit.json')
    original.require(terminal['phase'] == 'terminal' and terminal['exit_code'] == 0, 'Actual comparison must complete normally')
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from ilm.visual_lm.binary_plan_glyph_density import xor_correction
    from ilm.visual_lm.form_balanced_codec_support import balanced_codec_support
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.occupancy_glyph_density import OccupancySequence, decode_occupancy, encode_occupancy
    from ilm.visual_lm.residual_motor_metrics import residual_motor_metrics
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2)
    began = time.perf_counter()
    samples = []
    def guard():
        value = host_resources()
        samples.append(value)
        original.require(value['available_gib'] >= 24 and time.perf_counter() - began < 240, 'CPU audit reserve/deadline')
        original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 4 * 1024**3, 'CPU audit RSS ceiling')
    with (ROOT / 'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        host = host_resources()
        original.require(host['available_gib'] >= 48 and host['some_pressure_avg10'] <= .5 and host['full_pressure_avg10'] <= .1, 'CPU audit admission')
        reg = original.load(REG)
        raw = original.load(RUN / 'result.json')
        original.require(raw['registration_sha256'] == original.sha(REG), 'Registration changed')
        for name, digest in reg['input_sha256'].items():
            original.require(original.sha(ROOT / name) == digest, 'Bound source changed: ' + name)
        paths = [REG, RUN / 'result.json', ROOT / 'artifacts/residual_plan_motor_loaded_v1.exit.json',
                 RUN / 'initial_refinement.pt']
        inputs = {str(p.relative_to(ROOT)): original.sha(p) for p in paths}
        prepared = original.load(ROOT / 'artifacts/residual_plan_motor_preparation_v1/result.json')
        original.require(original.sha(ROOT / prepared['cache_file']) == prepared['cache_sha256'], 'Correction targets changed')
        data = torch.load(ROOT / prepared['cache_file'], weights_only=True, map_location='cpu')
        def unpack_images(packed):
            return torch.from_numpy(np.unpackbits(packed.numpy(), axis=1, bitorder='big').reshape(-1, 1, 32, 32).copy())
        base = unpack_images(data['base_packed'])
        source = base ^ unpack_images(data['residual_packed'])
        partition = torch.load(ROOT / 'artifacts/binary_codec_preparation_v1/partition.pt', weights_only=True, map_location='cpu')
        initial = torch.load(RUN / 'initial_refinement.pt', weights_only=True, map_location='cpu')
        original.require(tensor_state_sha(initial) == raw['initial_tensor_sha256'], 'Initial motor tensors changed')
        original.require(torch.count_nonzero(initial['root_output.weight']) == 0 and torch.equal(initial['root_output.bias'], torch.tensor([0., -2.])), 'Identity root initialization differs')
        original.require(raw['preflight']['exact_reset'] and raw['preflight']['retained_updates'] == 0 and raw['preflight']['before']['images'] == raw['preflight']['after']['images'] == 64041, 'Complete-source preflight/reset required')
        original.require(all(v > 0 and math.isfinite(v) for v in raw['preflight']['gradient_norms'].values()), 'Missing actual condition/root/mask gradients')
        original.require(raw['optimizer_replay'] == dict(update=3, exact=True, extra_trial_images=128), 'Actual optimizer replay differs')
        checkpoint = ROOT / raw['checkpoint_file']
        original.require(original.sha(checkpoint) == raw['checkpoint_sha256'], 'Final motor checkpoint changed')
        inputs[str(checkpoint.relative_to(ROOT))] = original.sha(checkpoint)
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu')
        original.require(saved['architecture'] == 'residual-plan-motor-v1' and saved['updates'] == 6000 and saved['width'] == 128 and saved['layers'] == 2 and saved['heads'] == 4 and saved['code_bits'] == 256, 'Final motor configuration differs')
        original.require(sum(v.numel() for v in saved['refinement'].values()) == raw['trainable_refinement_parameters'] == 615314 and tensor_state_sha(saved['refinement']) == raw['final_tensor_sha256'], 'Actual trainable count/tensors differ')
        original.require(original.sha(ROOT / saved['frozen_codec_file']) == saved['frozen_codec_sha256'], 'Frozen codec file changed')
        frozen = torch.load(ROOT / saved['frozen_codec_file'], weights_only=True, map_location='cpu')
        original.require(tensor_state_sha(frozen['model']) == raw['frozen_codec_tensor_sha256'] and sum(v.numel() for v in frozen['model'].values()) == 1770449, 'Frozen model tensor identity differs')
        original.require(raw['physical_motor_parameters'] == 2385763 and raw['frozen_codec_unchanged'] and not raw['unchanged_control_retrained'], 'Whole physical motor scope differs')
        order_path = ROOT / raw['source_order_file']
        original.require(original.sha(order_path) == raw['source_order_sha256'], 'Source order file changed')
        inputs[str(order_path.relative_to(ROOT))] = original.sha(order_path)
        order = torch.load(order_path, weights_only=True, map_location='cpu')
        original.require(torch.equal(order, balanced_codec_support(partition, reg['order_seed'])['order']), 'Complete sampling order differs')
        ledger_path = ROOT / raw['ledger_file']
        original.require(original.sha(ledger_path) == raw['ledger_sha256'], 'Training ledger changed')
        inputs[str(ledger_path.relative_to(ROOT))] = original.sha(ledger_path)
        ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
        original.require(len(ledger) == raw['updates'] == 6000 and raw['retained_training_draws'] == 768000, 'Incomplete retained fit')
        for update, row in enumerate(ledger, 1):
            indices = order[torch.arange((update - 1) * 128, update * 128) % len(order)]
            lr = .0002 * update / 250 if update <= 250 else .00002 + (.0002 - .00002) * .5 * (1 + math.cos(math.pi * (update - 250) / 5750))
            original.require(row['update'] == update and row['images'] == 128 and row['learning_rate'] == lr and row['batch_index_sha256'] == hashlib.sha256(indices.numpy().tobytes()).hexdigest(), 'Actual data/update schedule differs')
            original.require(math.isfinite(row['image_nll_per_pixel']) and math.isfinite(row['gradient_norm']), 'Nonfinite retained update')
        original.require(raw['training_curve'] == [r for r in ledger if r['update'] == 1 or r['update'] % 100 == 0], 'Training curve differs')
        audited = []
        final_output = None
        final_correction = None
        for stage in ['initial', 'final']:
            guard()
            report = raw[stage]
            path = ROOT / report['raw_file']
            original.require(original.sha(path) == report['raw_sha256'], 'Actual generated arrays changed')
            inputs[str(path.relative_to(ROOT))] = original.sha(path)
            with np.load(path) as arrays:
                ink, correction = arrays['ink'], arrays['correction']
                actions, depths, rows, columns = (arrays[n] for n in ['actions', 'depths', 'rows', 'columns'])
                lengths, logp, remaining = arrays['lengths'], arrays['executed_prefix_log_probability'], arrays['remaining_error_events']
            for a in [ink, correction]:
                original.require(a.shape == (68948, 1, 32, 32) and a.dtype == np.uint8 and np.isin(a, [0, 1]).all(), 'Complete actual binary images required')
            for a in [actions, depths, rows, columns]:
                original.require(a.shape == (68948, 342) and a.dtype == np.uint8, 'Every actual physical prefix required')
            original.require(lengths.shape == (68948,) and ((lengths >= 1) & (lengths <= 342)).all() and logp.shape == (68948, 342) and logp.dtype == np.float32, 'Every actual event length/probability required')
            recounted_remaining = []
            for start in range(0, 68948, 256):
                guard()
                end = min(start + 256, 68948)
                valid = torch.arange(342)[None] < torch.from_numpy(lengths[start:end].astype(np.int64))[:, None]
                fields = [torch.from_numpy(a[start:end].astype(np.int64)) for a in [actions, depths, rows, columns]]
                sequence = OccupancySequence(*fields, valid, fields[1] == 0)
                decoded = decode_occupancy(sequence, mode='tree')
                original.require(np.array_equal(decoded.numpy(), correction[start:end]), 'Generated tree does not recover actual correction')
                original.require(torch.equal(base[start:end] ^ decoded, torch.from_numpy(ink[start:end])), 'Actual image is not base XOR generated correction')
                probability = logp[start:end]
                original.require(np.isfinite(probability).all() and (probability[valid.numpy()] <= 1e-6).all() and (probability[~valid.numpy()] == 0).all(), 'Finite actual prefix probabilities required')
                for field in fields:
                    original.require((field[~valid] == 0).all(), 'Unexecuted trace slots must remain zero')
                errors = xor_correction(torch.from_numpy(ink[start:end]), source[start:end])
                recounted_remaining.append(encode_occupancy(errors, 'tree').valid.sum(1))
            recounted_remaining = torch.cat(recounted_remaining)
            original.require(np.array_equal(recounted_remaining.numpy(), remaining), 'Every remaining-error event count differs')
            metrics = residual_motor_metrics(torch.from_numpy(ink), source, partition, recounted_remaining, torch.from_numpy(lengths.astype(np.int64)))
            original.require(metrics == report['metrics'], 'All actual physical metrics differ')
            original.require(report['maximum_event_probability_error'] <= 5e-5 and report['all_68948_images_and_actual_traces'] and report['actual_generation_not_true_residual_replay'], 'Bound neural execution replay was not complete')
            if stage == 'initial':
                original.require(np.array_equal(ink, base.numpy()) and not correction.any() and (lengths == 1).all(), 'Actual initial motor differs from unchanged base')
            else:
                final_output, final_correction = ink, correction
            audited.append(dict(stage=stage, images=68948, all_pixels_actions_and_event_costs_recounted=True, metrics=metrics))
            del actions, depths, rows, columns, lengths, logp, remaining
        b, q = raw['final']['metrics'], reg['quality']
        checks = dict(train_f1=b['train']['pixels']['f1'] >= .98, held_f1=b['held']['pixels']['f1'] >= .98,
                      held_macro_f1=b['held']['macro_form_f1'] >= .98, held_macro_exact=b['held']['macro_form_exact_rate'] >= .80,
                      retention=b['held']['pixels']['f1'] >= q['continuous_held_f1'] - .005,
                      gain=b['held']['pixels']['f1'] - q['base_held_f1'] >= .02,
                      false_blank=b['all']['pixels']['false_blank'] == 0, false_ink=b['all']['pixels']['false_ink'] == 0,
                      generated_event_work=b['all']['mean_plan_and_generated_events'] <= .60 * q['old_direct_tree_mean_events'])
        original.require(checks == raw['support_checks'] and all(checks.values()) == raw['support'], 'Original support decision differs')
        def first_distinct(held, count, nonempty=False):
            chosen, seen = [], set()
            for i in (partition['held'] == held).nonzero().flatten().tolist():
                group = int(partition['group'][i])
                if group in seen or (nonempty and not bool(source[i].any())):
                    continue
                seen.add(group)
                chosen.append(i)
                if len(chosen) == count:
                    return chosen
            raise ValueError('Not enough physical examples')
        held_indices = first_distinct(True, 64)
        train_indices = sorted([int((~source.bool().flatten(1).any(1)).nonzero()[0]), *first_distinct(False, 15, True)])
        OUT.mkdir()
        exports = []
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        for name, indices in [('held', held_indices), ('train', train_indices)]:
            canvas = Image.new('L', (4 * 364, ((len(indices) + 3) // 4) * 102 + 28), 255)
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 3), 'Supplied-plan motor: source | frozen base | motor | generated correction | remaining error (2x exact pixels)', font=font, fill=0)
            tiles = []
            for n, i in enumerate(indices):
                x, y = (n % 4) * 364, (n // 4) * 102 + 28
                draw.text((x + 3, y), f'{name} source-cell {i}', font=font, fill=0)
                target = source[i, 0].numpy()
                values = [target, base[i, 0].numpy(), final_output[i, 0], final_correction[i, 0], target ^ final_output[i, 0]]
                for j, value in enumerate(values):
                    raster = 255 * (1 - value)
                    px, py = x + 3 + 72 * j, y + 18
                    canvas.paste(Image.fromarray(raster).resize((64, 64), Image.Resampling.NEAREST), (px, py))
                    tiles.append((px, py, raster))
            path = OUT / f'{name}_fixed_images.png'
            canvas.save(path)
            back = np.asarray(Image.open(path))
            for x, y, raster in tiles:
                original.require(np.array_equal(back[y:y + 64:2, x:x + 64:2], raster), 'Image plate changed actual pixels')
            exports.append(dict(file=str(path.relative_to(ROOT)), sha256=original.sha(path), source_indices=indices, scale=2,
                                columns='source, frozen base, actual motor, generated correction, remaining error'))
        result = dict(raw)
        result['resources'] = dict(raw['resources'])
        result['resources'].pop('admission', None)
        result.update(input_sha256=inputs, audits=audited, exports=exports,
                      all_137896_actual_images_and_causal_traces_recounted=True, all_6000_retained_updates_checked=True,
                      audit_scope='Saved image/action/event/metric/checkpoint/ledger recount only; neural probability comparisons are the bound GPU execution, not repeated CPU inference',
                      auditor_sha256=original.sha(Path(__file__)),
                      audit_resources=dict(elapsed_seconds=time.perf_counter() - began, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                           minimum_available_host_gib=min(v['available_gib'] for v in samples), cpu_threads=2),
                      head_resource_amendment=reg['head_resource_amendment'],
                      supervisor_elapsed_seconds=terminal['finished_unix']-terminal['began_unix'])
        original.write_new(PUBLIC, result)
        print(json.dumps(dict(public=str(PUBLIC.relative_to(ROOT)), sha256=original.sha(PUBLIC), support=raw['support'], all_actual_images=137896)), flush=True)


if __name__ == '__main__':
    main()
