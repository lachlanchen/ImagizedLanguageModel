#!/usr/bin/env python3
"""Complete native image-language fit; original pixel corpus, no text body."""
import argparse
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess

from scripts import diagnose_visual_context_gradients as common

ROOT = common.ROOT
REG = ROOT / 'references/evidence/continuous_autoregressive_full_registration_20261118.json'
RUN = ROOT / 'artifacts/continuous_autoregressive_full_v1'
DATA = ROOT / 'data/visual_sequence_corpus_v1'
EVAL = ROOT / 'artifacts/compact_visual_span_evaluation_v1/inputs'
CODEC = ROOT / 'artifacts/continuous_glyph_codec_v34_20260814/checkpoint_latest.pt'
STEPS = [0, 3447, 6893, 10339, 13785]


def source_arrays():
    import numpy as np
    report = common.load(DATA / 'report.json')
    arrays, offsets = [], [0]
    for shard in report['shards']:
        if shard['split'] != 'train':
            continue
        name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
        path = DATA / name
        common.require(common.sha(path) == shard['files'][name], 'Original source changed')
        with np.load(io.BytesIO(path.read_bytes()), allow_pickle=False) as data:
            arrays.append({k: data[k] for k in ('ink', 'record_glyph_offsets', 'prompt_glyphs')})
        offsets.append(offsets[-1] + shard['records'])
    return arrays, offsets


def register():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Register CPU only')
    import numpy as np
    preflight = common.load(ROOT / 'references/evidence/continuous_autoregressive_preflight_result_20261118.json')
    audit = common.load(ROOT / 'references/evidence/continuous_autoregressive_preflight_audit_20261118.json')
    common.require(preflight['execution_support'] and preflight['all_trial_updates_reset']
        and audit['all_actual_images_recounted'] == 2208 and audit['all_pixel_scores_agree'], 'Actual complete-source preflight/audit required')
    paths = {ROOT / n for n in preflight['source_hashes'] if not n.startswith('data/')}
    paths |= {Path(__file__), ROOT / 'scripts/evaluate_continuous_image_language.py',
        ROOT / 'tests/test_continuous_language_execution.py',
        ROOT / 'references/continuous_autoregressive_full_comparison_protocol_20261118.md',
        ROOT / 'references/evidence/continuous_autoregressive_preflight_result_20261118.json',
        ROOT / 'references/evidence/continuous_autoregressive_preflight_audit_20261118.json',
        DATA / 'report.json', DATA / 'verification.json', EVAL / 'manifest.json'}
    report = common.load(DATA / 'report.json')
    for shard in report['shards']:
        if shard['split'] == 'train':
            paths |= {DATA / n for n in shard['files'] if n.endswith('_pixels.npz')}
    manifest = common.load(EVAL / 'manifest.json')['inputs']
    common.require(len(manifest) == 120 and len({r['id'] for r in manifest}) == 120, 'All original inputs required')
    for row in manifest:
        path = EVAL / row['image']
        common.require(common.sha(path) == row['image_sha256'], 'Original evaluation image changed')
        paths.add(path)
    arrays, offsets = source_arrays()
    targets = sum(int((np.diff(a['record_glyph_offsets']) - np.maximum(1, a['prompt_glyphs'])).sum()) for a in arrays)
    maximum = max(int(np.diff(a['record_glyph_offsets']).max()) for a in arrays)
    common.require(offsets[-1] == 13785 and targets == 2571477 and maximum == 1101, 'Complete source accounting differs')
    common.write_new(REG, dict(schema='ilm-native-continuous-full-comparison-v1',
        input_sha256={str(p.relative_to(ROOT)): common.sha(p) for p in sorted(paths)},
        packages={n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'Pillow')},
        model_seed=202611198, order_seed=202611199, components=[1, 8], steps=STEPS,
        records=13785, target_positions_per_arm=targets, stop_positions_per_arm=targets + 13785,
        maximum_record_images=maximum, learning_rate=.0002, minimum_learning_rate=.00002,
        warmup=250, weight_decay=.01, gradient_clip=1., epochs=1,
        evaluation_inputs=manifest, evaluation_streams_per_arm=396,
        resources=dict(maximum_fit_seconds=7200, maximum_pressure_wait_seconds=300,
            maximum_process_rss_gib=8, maximum_allocated_gpu_gib=22, cpu_threads=2),
        previous_failed_actors_not_resumed=True, reserved_confirmation_untouched=True,
        language_quality_measured=False, goal_achieved=False))
    print(json.dumps(dict(registration=str(REG.relative_to(ROOT)), sha256=common.sha(REG), targets=targets)), flush=True)


def validate():
    reg = common.load(REG)
    common.require(reg['packages'] == {n: importlib.metadata.version(n) for n in reg['packages']}, 'Packages changed')
    for name, digest in reg['input_sha256'].items():
        common.require(common.sha(ROOT / name) == digest, 'Bound file changed: ' + name)
        if name.startswith(('ilm/', 'scripts/', 'tests/', 'references/')):
            common.require(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT) == (ROOT / name).read_bytes(), 'Commit bound file')
    common.require(subprocess.check_output(['git', 'show', 'HEAD:' + str(REG.relative_to(ROOT))], cwd=ROOT) == REG.read_bytes(), 'Commit registration')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    common.require(subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True, timeout=30).split()[0] == head, 'Push before execution')
    return reg, head


def atomic_save(value, path):
    import torch
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def run():
    common.require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID'
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Physical GPU0 and deterministic workspace required')
    common.require(not RUN.exists(), 'Preserve every first execution')
    reg, head = validate()
    import numpy as np
    import torch
    from bisect import bisect_right
    from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
    from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
    from ilm.visual_lm.full_rank_visual_span import tensor_state_sha
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    RUN.mkdir()
    monitor = GlyphReaderMonitor(RUN, 7200)
    monitor.pressure_wait_allowance_seconds = 300
    signal.signal(signal.SIGTERM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError('Full native fit external deadline')))
    model, optimizer, update, components = None, None, 0, None
    reports = []
    try:
        with full_reader_gpu() as (device, admission):
            monitor.admission = admission
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            def guard():
                monitor.guard()
                common.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 8 * 1024**3, 'Native fit RSS cap')
            arrays, offsets = source_arrays()
            common.require(offsets[-1] == reg['records'], 'Full original source required')
            codec = load_qualified_writing_codec(CODEC, device)
            frozen_sha = tensor_state_sha(codec.state_dict())
            order = torch.randperm(reg['records'], generator=torch.Generator().manual_seed(reg['order_seed']))
            atomic_save(order, RUN / 'source_order.pt')
            initial_common = None
            for components in reg['components']:
                torch.manual_seed(reg['model_seed'])
                model = ContinuousImageLanguage(codec, components=components).to(device).train()
                common_sha = tensor_state_sha({n: p for n, p in model.state_dict().items()
                    if not n.startswith(('codec.', 'writer.output.'))})
                if initial_common is None:
                    initial_common = common_sha
                common.require(common_sha == initial_common, 'Common initial model differs')
                directory = RUN / f'k{components}'
                directory.mkdir()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=reg['learning_rate'], betas=(.9, .95), weight_decay=reg['weight_decay'])
                snapshots, targets_seen, stops_seen = [], 0, 0
                def save_snapshot(step):
                    guard()
                    path = directory / f'model_{step:07d}.pt'
                    atomic_save(dict(registration_sha256=common.sha(REG), components=components,
                        model=model.state_dict(), update=step, whole_parameters=sum(p.numel() for p in model.parameters())), path)
                    snapshots.append(dict(update=step, file=str(path.relative_to(ROOT)), sha256=common.sha(path)))
                save_snapshot(0)
                with (directory / 'ledger.jsonl').open('x', buffering=1) as ledger:
                    for update, index in enumerate(order.tolist(), 1):
                        guard()
                        shard = bisect_right(offsets, index) - 1
                        local = index - offsets[shard]
                        a = arrays[shard]
                        first, last = a['record_glyph_offsets'][local:local + 2]
                        ink = torch.from_numpy(np.unpackbits(a['ink'][first:last], bitorder='big').reshape(-1, 1, 32, 32).copy()).to(device)
                        prompt = int(a['prompt_glyphs'][local])
                        common.require(len(ink) <= model.maximum_cells, 'No record truncation')
                        phase = min(1., update / reg['warmup'])
                        if update <= reg['warmup']:
                            lr = reg['learning_rate'] * phase
                        else:
                            fraction = (update - reg['warmup']) / (reg['records'] - reg['warmup'])
                            lr = reg['minimum_learning_rate'] + .5 * (reg['learning_rate'] - reg['minimum_learning_rate']) * (1 + math.cos(math.pi * fraction))
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        optimizer.zero_grad(set_to_none=True)
                        terms = model.training_terms(ink, prompt)
                        loss = terms['code_nll'] + terms['stop_bce']
                        common.require(bool(torch.isfinite(loss)), 'Finite complete-source likelihood required')
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                        optimizer.step()
                        targets_seen += terms['target_positions']; stops_seen += terms['stop_positions']
                        row = dict(update=update, corpus_index=index, images=len(ink), prompt_images=prompt,
                            targets=terms['target_positions'], stops=terms['stop_positions'], lr=lr,
                            code_nll=float(terms['code_nll']), stop_bce=float(terms['stop_bce']), gradient_norm=float(norm))
                        ledger.write(json.dumps(row) + '\n')
                        if update in STEPS:
                            save_snapshot(update)
                        if update % 250 == 0 or update == reg['records']:
                            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                update=update, components=components, registration_sha256=common.sha(REG)), directory / 'checkpoint_latest.pt')
                            monitor.position = dict(components=components, update=update, target_exposures=targets_seen)
                            monitor.status('fitting_complete_native_image_corpus')
                            print(json.dumps(monitor.position), flush=True)
                        del ink, terms, loss
                common.require(targets_seen == reg['target_positions_per_arm'] and stops_seen == reg['stop_positions_per_arm'], 'Complete exposure schedule differs')
                common.require(tensor_state_sha(codec.state_dict()) == frozen_sha, 'Fixed neural drawing codec changed')
                report = dict(components=components, updates=update, target_exposures=targets_seen,
                    stop_exposures=stops_seen, snapshots=snapshots, common_initial_sha256=common_sha,
                    ledger_sha256=common.sha(directory / 'ledger.jsonl'), latest_sha256=common.sha(directory / 'checkpoint_latest.pt'),
                    codec_unchanged=True, language_quality_measured=False)
                common.write_new(directory / 'report.json', report)
                reports.append(report)
                del model, optimizer
                model, optimizer = None, None
                torch.cuda.empty_cache()
            common.write_new(RUN / 'report.json', dict(registration_sha256=common.sha(REG), source_commit=head,
                arms=reports, resources=monitor.receipt(), language_quality_measured=False, goal_achieved=False))
            monitor.status('complete_fit_awaiting_independent_audit_and_language')
    except BaseException as error:
        if model is not None:
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                update=update, components=components, unqualified_failure_state=True), RUN / 'failure_state.pt')
        common.write_new(RUN / 'failure.json', dict(error=repr(error), components=components, completed_updates=update,
            completed_arms=reports, resources=monitor.receipt(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['register', 'run'], required=True)
    (register if parser.parse_args().stage == 'register' else run)()
