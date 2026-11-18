#!/usr/bin/env python3
"""Independent schedule audit and fixed autonomous image-only generation."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as common
from scripts.train_continuous_image_language import REG, RUN, ROOT, EVAL, STEPS, source_arrays, validate


def audit():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Audit saved tensors on CPU only')
    common.require(not (RUN / 'audit.json').exists(), 'Preserve original completion audit')
    import numpy as np
    import torch
    torch.set_num_threads(2)
    reg = common.load(REG)
    report = common.load(RUN / 'report.json')
    common.require(report['registration_sha256'] == common.sha(REG), 'Original full registration differs')
    arrays, offsets = source_arrays()
    from bisect import bisect_right
    expected = torch.randperm(reg['records'], generator=torch.Generator().manual_seed(reg['order_seed']))
    common.require(torch.equal(expected, torch.load(RUN / 'source_order.pt', weights_only=True)), 'Complete original order differs')
    actual_updates, actual_targets, actual_stops, snapshots = 0, 0, 0, 0
    for arm in report['arms']:
        directory = RUN / f'k{arm["components"]}'
        common.require(common.sha(directory / 'ledger.jsonl') == arm['ledger_sha256'], 'Original ledger changed')
        rows = [json.loads(line) for line in (directory / 'ledger.jsonl').read_text().splitlines()]
        common.require(len(rows) == reg['records'], 'Incomplete optimizer schedule')
        targets = stops = 0
        for i, (row, index) in enumerate(zip(rows, expected.tolist()), 1):
            shard = bisect_right(offsets, index) - 1
            local = index - offsets[shard]; a = arrays[shard]
            images = int(a['record_glyph_offsets'][local + 1] - a['record_glyph_offsets'][local])
            prompt = int(a['prompt_glyphs'][local]); count = images - max(1, prompt)
            common.require((row['update'], row['corpus_index'], row['images'], row['prompt_images'], row['targets'], row['stops'])
                == (i, index, images, prompt, count, count + 1), 'Whole original source schedule differs')
            common.require(all(np.isfinite(row[k]) for k in ('lr', 'code_nll', 'stop_bce', 'gradient_norm')), 'Nonfinite actual record ledger')
            targets += count; stops += count + 1
        common.require(targets == arm['target_exposures'] == reg['target_positions_per_arm']
            and stops == arm['stop_exposures'] == reg['stop_positions_per_arm'], 'Complete exposures differ')
        common.require([s['update'] for s in arm['snapshots']] == STEPS, 'Preserve all five actual snapshots')
        for snapshot in arm['snapshots']:
            path = ROOT / snapshot['file']
            common.require(common.sha(path) == snapshot['sha256'], 'Snapshot file changed')
            stored = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
            common.require(stored['update'] == snapshot['update'] and stored['registration_sha256'] == common.sha(REG)
                and stored['components'] == arm['components'], 'Snapshot identity differs')
            common.require(all(bool(torch.isfinite(p).all()) for p in stored['model'].values()), 'Nonfinite retained snapshot')
            snapshots += 1
        latest = directory / 'checkpoint_latest.pt'
        common.require(common.sha(latest) == arm['latest_sha256'], 'Latest optimizer file changed')
        stored = torch.load(latest, weights_only=True, map_location='cpu', mmap=True)
        common.require(stored['update'] == reg['records'] and stored['optimizer']['state'], 'Missing final actual optimizer')
        common.require(all(int(v['step']) == reg['records'] for v in stored['optimizer']['state'].values()), 'Incomplete actual optimizer counters')
        common.require(arm['codec_unchanged'], 'Fixed decoder failed original tensor check')
        actual_updates += len(rows); actual_targets += targets; actual_stops += stops
    common.require(actual_updates == 27570 and actual_targets == 5142954 and snapshots == 10, 'Both complete arms required')
    common.write_new(RUN / 'audit.json', dict(registration_sha256=common.sha(REG), report_sha256=common.sha(RUN / 'report.json'),
        complete=True, all_actual_updates_recounted=actual_updates, all_target_exposures_recounted=actual_targets,
        all_stop_exposures_recounted=actual_stops, actual_snapshots_checked=snapshots,
        neural_replay=False, language_quality_measured=False, goal_achieved=False))
    print(json.dumps(dict(updates=actual_updates, targets=actual_targets, snapshots=snapshots)), flush=True)


def plan(reg):
    jobs = []
    rows = reg['evaluation_inputs']
    for step in STEPS:
        selected = rows if step in (0, STEPS[-1]) else [r for r in rows if r['condition'] == 'normal']
        jobs.append(dict(step=step, cap=128, inputs=selected))
    jobs.append(dict(step=STEPS[-1], cap=640, inputs=[r for r in rows if r['id'].startswith('book_')]))
    common.require(sum(len(j['inputs']) for j in jobs) == 396, 'Complete autonomous plan changed')
    return jobs


def evaluate():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Physical GPU0 deterministic execution required')
    # Completion admission precedes neural import and generation-directory creation.
    audit_result = common.load(RUN / 'audit.json')
    common.require(audit_result['complete'] and audit_result['registration_sha256'] == common.sha(REG)
        and audit_result['report_sha256'] == common.sha(RUN / 'report.json'), 'Original complete fit audit required')
    reg, head = validate()
    common.require(not (RUN / 'evaluation').exists(), 'Preserve every generation execution')
    import numpy as np
    import torch
    from PIL import Image
    from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
    from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.train_continuous_image_language import CODEC
    destination = RUN / 'evaluation'
    destination.mkdir()
    monitor = GlyphReaderMonitor(destination, 7200)
    monitor.pressure_wait_allowance_seconds = 300
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('Autonomous generation deadline')))
    outputs = []
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            codec = load_qualified_writing_codec(CODEC, device)
            for components in reg['components']:
                for job in plan(reg):
                    monitor.guard()
                    path = RUN / f'k{components}' / f'model_{job["step"]:07d}.pt'
                    snapshot_sha = common.sha(path)
                    state = torch.load(path, map_location='cpu', weights_only=True)
                    model = ContinuousImageLanguage(codec, components=components).to(device).eval()
                    model.load_state_dict(state['model'], strict=True)
                    folder = destination / f'k{components}_step{job["step"]:07d}_cap{job["cap"]}'
                    folder.mkdir()
                    for row in job['inputs']:
                        monitor.position = dict(components=components, step=job['step'], cap=job['cap'], case=row['id'])
                        monitor.status('generating_actual_autonomous_images')
                        monitor.guard()
                        common.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3, 'Generation RSS ceiling')
                        source = EVAL / row['image']
                        common.require(common.sha(source) == row['image_sha256'], 'Input pixels changed')
                        with Image.open(source) as image:
                            physical = np.asarray(image.convert('L')).copy()
                        h, w = physical.shape
                        common.require(h % 32 == 0 and w % 32 == 0 and np.isin(physical, [0, 255]).all(), 'Native aligned binary page required')
                        tiles = (physical < 128).reshape(h // 32, 32, w // 32, 32).transpose(0, 2, 1, 3).reshape(-1, 1, 32, 32)
                        ink = torch.from_numpy(tiles[:row['glyphs']].astype(np.uint8).copy()).to(device)
                        generated, receipt = model.generate(ink, maximum_new=job['cap'])
                        pixels = generated.cpu().numpy()
                        raw = folder / (row['id'] + '.npz')
                        np.savez_compressed(raw, ink=pixels)
                        count = len(pixels); cols = min(32, max(1, count)); lines = max(1, (count + cols - 1) // cols)
                        page = np.full((lines * 32, cols * 32), 255, dtype=np.uint8)
                        for i, tile in enumerate(pixels):
                            y, x = divmod(i, cols)
                            page[y * 32:(y + 1) * 32, x * 32:(x + 1) * 32] = (1 - tile[0]) * 255
                        png = folder / (row['id'] + '.png')
                        Image.fromarray(page).save(png)
                        record = dict(components=components, step=job['step'], cap=job['cap'], input=row,
                            snapshot_sha256=snapshot_sha, output_file=str(png.relative_to(ROOT)), output_sha256=common.sha(png),
                            raw_file=str(raw.relative_to(ROOT)), raw_sha256=common.sha(raw), **receipt)
                        common.write_new(folder / (row['id'] + '.json'), record)
                        outputs.append(record)
                    del model, state
                    torch.cuda.empty_cache()
            common.require(len(outputs) == 792, 'Every prescribed stream required')
            common.write_new(destination / 'report.json', dict(registration_sha256=common.sha(REG), source_commit=head,
                streams=outputs, resources=monitor.receipt(), review_frozen=False,
                language_quality_measured=False, goal_achieved=False))
            monitor.status('complete_generation_awaiting_frozen_review')
    except BaseException as error:
        common.write_new(destination / 'failure.json', dict(error=repr(error), completed_streams=len(outputs),
            resources=monitor.receipt(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['audit', 'evaluate'], required=True)
    (audit if parser.parse_args().stage == 'audit' else evaluate)()
