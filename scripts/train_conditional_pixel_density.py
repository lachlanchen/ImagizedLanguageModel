#!/usr/bin/env python3
"""Source-bound first-block learning probe; no backbone or teacher is loaded."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ilm.image_edit.baseline import ROOT, save_json, sha

CACHE = ROOT / 'artifacts/conditional_readout_v69_20261105/attempt01'
FILES = ('ilm/visual_lm/conditional_pixel_density.py',
    'scripts/train_conditional_pixel_density.py',
    'scripts/audit_conditional_pixel_density.py',
    'references/conditional_pixel_density_protocol.md')
SEED, UPDATES, BATCH = 202611091, 1200, 32


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def atomic_tensor(path, value):
    import torch
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)


def identity():
    for name in FILES:
        committed = subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)
        require(hashlib.sha256(committed).hexdigest() == sha(ROOT/name), 'commit implementation first: ' + name)
    return {name: sha(ROOT/name) for name in FILES}


def cache_identity():
    expected = {'capture_report.json': '605ffe0677816eacb9300560ca2dd46667b72e02fa570278b5ccdd109da0f9f7',
        'verification.json': '30bf620248f57b04aa08a5189adb17f1c3a50d031e695e46a74b14176f08c52d'}
    bound = {}
    def bind(path, digest=None):
        actual = sha(path)
        require(digest is None or actual == digest, 'source binding differs: ' + str(path))
        bound[str(path.relative_to(ROOT))] = actual
    for name, digest in expected.items():
        bind(CACHE/name, digest)
    capture, prep = load(CACHE/'capture_report.json'), load(CACHE/'preparation.json')
    bind(CACHE/'preparation.json', capture['preparation_sha256'])
    bind(CACHE/'prepared.pt', prep['prepared_sha256'])
    require(capture['sources'] == prep['sources'] and len(capture['features_sha256']) == 704,
        'complete original state capture required')
    for name, digest in prep['sources'].items():
        bind(ROOT/name, digest)
    for name, digest in capture['features_sha256'].items():
        require(Path(name).name == name, 'unsafe feature path')
        bind(CACHE/name, digest)
    return bound, prep


def prepare(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for CPU preparation')
    code = identity()
    bound, oldprep = cache_identity()
    import torch
    from ilm.visual_lm.conditional_pixel_density import pack_pixels, unpack_pixels
    torch.set_num_threads(4)
    rows = torch.load(CACHE/'prepared.pt', map_location='cpu', weights_only=True)
    require(len(rows) == 704, 'original 704-case cohort required')
    normal, blank, targets, cases = [], [], [], []
    for i, row in enumerate(rows):
        feature = torch.load(CACHE/f'features_{i:04}.pt', map_location='cpu', weights_only=True)
        require(feature['index'] == i and feature['sources'] == oldprep['sources']
            and feature['pixels_sha256'] == row['pixels_sha256'], 'feature/source case mismatch')
        for key in row['pixels_sha256']:
            value = row[key]
            require(hashlib.sha256(value.numpy().tobytes()).hexdigest() == row['pixels_sha256'][key],
                'owning physical source changed')
        h = feature['inputs']['normal']['states'].float()
        require(h.shape == (2, 4096) and bool(torch.isfinite(h).all()), 'actual input-only states required')
        normal.append(h)
        target = row['reference']
        require(target.shape == (1, 1, 32, 64) and bool(((target == -1) | (target == 1)).all()),
            'complete signed binary reference required')
        targets.append(target[0, 0] < 0)
        require(row['split'] == ('train' if i < 512 else 'development'), 'fixed train/development order differs')
        if i >= 512:
            blank.append(feature['inputs']['blank']['states'].float())
        cases.append({k: row[k] for k in ('split', 'source', 'id', 'article', 'source_file', 'pixels_sha256')})
    h, ink = torch.stack(normal), torch.stack(targets)
    patterns = pack_pixels(ink)
    require(torch.equal(unpack_pixels(patterns), ink), 'packing must preserve every source pixel')
    mean = h[:512].reshape(512, -1).mean(0)
    scale = h[:512].reshape(512, -1).std(0, unbiased=False).clamp_min(1e-4)
    rng = torch.Generator().manual_seed(SEED)
    schedule = torch.stack([torch.randperm(512, generator=rng).reshape(16, BATCH) for _ in range(75)]).reshape(UPDATES, BATCH)
    require(all(torch.equal(epoch.sort().values, torch.arange(512)) for epoch in schedule.reshape(75, 512)),
        '75 complete training epochs required')
    output.mkdir(parents=True, exist_ok=False)
    atomic_tensor(output/'data.pt', dict(normal=h, blank=torch.stack(blank), ink=ink, patterns=patterns,
        mean=mean, scale=scale, schedule=schedule))
    save_json(output/'cases.json', cases)
    save_json(output/'preparation.json', dict(code_sha256=code, bound_sha256=bound,
        data_sha256=sha(output/'data.pt'), cases_sha256=sha(output/'cases.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        seed=SEED, updates=UPDATES, batch=BATCH, training_cases=512, development_cases=192,
        erased_development_cases=192, training_normalization_only=True, pixels_per_target=2048,
        training_packing_bit_exact=True, source_state_interface='V68 custom image actor, not native source QA',
        maximum_execution_seconds=1800, maximum_allocated_vram_gib=4.,
        general_language_claim=False, complete_answer_claim=False))
    print(json.dumps(dict(prepared=str(output), training=512, development=192, erased=192)), flush=True)


def unchanged(output):
    meta = load(output/'preparation.json')
    require(identity() == meta['code_sha256'], 'registered implementation changed')
    for name, digest in meta['bound_sha256'].items():
        require(sha(ROOT/name) == digest, 'bound source changed: ' + name)
    require(sha(output/'data.pt') == meta['data_sha256'] and sha(output/'cases.json') == meta['cases_sha256'],
        'prepared data changed')
    return meta


def resources():
    values = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())
    swap = (values['SwapTotal'] - values['SwapFree']) / values['SwapTotal'] if values['SwapTotal'] else 0.
    return dict(available_ram_gib=values['MemAvailable']/1024**2, swap_fraction=swap, gpu0_free_mib=free,
        ready=values['MemAvailable'] >= 24*1024**2 and swap <= .75 and free >= 4096)


@contextmanager
def small_head_gpu():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
        and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'explicit physical GPU0 required')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = resources()
        require(observed['ready'], 'small-head memory admission failed: ' + json.dumps(observed))
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        torch.set_num_threads(4)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        yield torch.device('cuda:0'), observed


def numerical_checks(model, states, targets):
    import torch
    from ilm.visual_lm.conditional_pixel_density import pack_pixels, unpack_pixels
    model.eval()
    with torch.no_grad():
        logits = model(states, targets)
        changed = targets.clone()
        changed[:, 128:] = changed[:, 128:] ^ 255
        altered = model(states, changed)
        causal_error = float((logits[:, :129] - altered[:, :129]).abs().max())
        future_effect = float((logits[:, 129:] - altered[:, 129:]).abs().max())
        require(causal_error == 0. and future_effect > 0., 'target suffix causality check failed')
        condition, caches, parts = model.condition_features(states), None, []
        for j in range(256):
            part, caches = model.step(condition, None if j == 0 else targets[:, j-1], j, caches)
            parts.append(part)
        cache_error = float((torch.stack(parts, 1) - logits).abs().max())
        require(cache_error <= 1e-4, 'cached/whole-prefix logits differ beyond fixed tolerance')
        generated = model.generate(states)
        # Independently follow the full-prefix path on the same model weights.
        prefix = targets.new_zeros((len(states), 1))
        for j in range(256):
            prefix[:, j] = model(states, prefix)[:, -1].argmax(-1)
            if j < 255:
                prefix = torch.cat((prefix, prefix.new_zeros((len(states), 1))), 1)
        require(torch.equal(prefix, generated), 'cached and uncached greedy output differ')
        require(torch.equal(pack_pixels(unpack_pixels(targets)), targets), 'pixel packing round trip failed')
        dependence = float((model(states + .01, targets) - logits).abs().max())
        require(dependence > 0., 'condition is disconnected')
    return dict(suffix_causality_max_error=causal_error, altered_future_logit_effect=future_effect,
        cached_teacher_logits_max_error=cache_error, cached_uncached_greedy_bit_exact=True,
        all_256_positions_checked=True, numeric_condition_effect=dependence,
        language_understanding_inferred=False)


def equal_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def run(output):
    meta = unchanged(output)
    execution = output/'execution'
    execution.mkdir(exist_ok=False)
    started = time.perf_counter()
    completed = 0
    def status(phase, **extra):
        save_json(execution/'status.json', dict(phase=phase, pid=os.getpid(), completed_updates=completed,
            elapsed_seconds=time.perf_counter()-started, **extra))
    try:
        with small_head_gpu() as (device, admitted):
            import copy
            import numpy as np
            import torch
            from torch.nn import functional as F
            from PIL import Image
            from ilm.visual_lm.conditional_pixel_density import ConditionalPixelDensity, unpack_pixels
            data = torch.load(output/'data.pt', map_location='cpu', weights_only=True)
            model = ConditionalPixelDensity(data['mean'], data['scale']).to(device)
            parameters = sum(p.numel() for p in model.parameters())
            require(parameters < 4_000_000, 'small head parameter budget exceeded')
            h, blank, target = (data[k].to(device) for k in ('normal', 'blank', 'patterns'))
            status('numerical_preflight', parameters=parameters, admission=admitted)
            checks = numerical_checks(model, h[:2], target[:2])
            save_json(execution/'numerical_preflight.json', checks)
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_tensor(execution/'initial.pt', dict(model=initial, config=model.configuration()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), weight_decay=.01)
            history = []
            def measure_nll():
                model.eval()
                values = []
                with torch.no_grad():
                    for features, truth in ((h, target), (blank, target[512:])):
                        losses = []
                        for i in range(0, len(features), 64):
                            logits = model(features[i:i+64], truth[i:i+64])
                            # Explicit normalized likelihood avoids the old runtime's
                            # nondeterministic fused 3-D NLL CUDA kernel.
                            losses.append(-F.log_softmax(logits, dim=-1).gather(-1,
                                truth[i:i+64, :, None]).squeeze(-1).mean(1).cpu())
                        values.append(torch.cat(losses))
                return torch.cat(values)
            initial_nll = measure_nll()
            training_start = time.perf_counter()
            checkpoint = None
            replay = None
            for i, indices in enumerate(data['schedule']):
                model.train()
                step = i + 1
                rate = 3e-4 * min(step/32, 1.) * (.1 + .9*.5*(1+math.cos(math.pi*i/(UPDATES-1))))
                def update():
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group['lr'] = rate
                    logits = model(h[indices], target[indices])
                    loss = -F.log_softmax(logits, dim=-1).gather(-1,
                        target[indices, :, None]).mean()
                    require(bool(torch.isfinite(loss)), 'nonfinite training loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    require(bool(torch.isfinite(norm)), 'nonfinite training gradient')
                    optimizer.step()
                    return float(loss), float(norm)
                loss, norm = update()
                if step == 2:
                    atomic_tensor(execution/'update0002.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=2))
                    # AdamW's non-capturable step counters belong on CPU. Its
                    # loader places moment tensors on their parameter devices.
                    checkpoint = torch.load(execution/'update0002.pt', map_location='cpu', weights_only=True)
                if step == 3:
                    expected_model, expected_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    repeated_loss, repeated_norm = update()
                    require(loss == repeated_loss and norm == repeated_norm
                        and equal_tree(expected_model, model.state_dict())
                        and equal_tree(expected_optimizer, optimizer.state_dict()), 'actual update replay differs')
                    replay = dict(checkpoint_step=2, replayed_step=3, loss_bit_exact=True,
                        gradient_norm_bit_exact=True, all_parameters_bit_exact=True, optimizer_state_bit_exact=True)
                    save_json(execution/'update_replay.json', replay)
                    del checkpoint, expected_model, expected_optimizer
                completed = step
                history.append(dict(step=step, loss=loss, gradient_norm=norm, learning_rate=rate))
                if step % 50 == 0:
                    status('training', loss=loss, allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3)
                    print(json.dumps(history[-1]), flush=True)
                require(time.perf_counter()-started < meta['maximum_execution_seconds'], 'registered execution time budget reached')
                require(torch.cuda.max_memory_allocated()/1024**3 <= meta['maximum_allocated_vram_gib'], 'head VRAM budget exceeded')
            torch.cuda.synchronize()
            training_seconds = time.perf_counter()-training_start
            model.eval()
            final_nll = measure_nll()
            atomic_tensor(execution/'final.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                config=model.configuration(), logical_updates=UPDATES))
            atomic_tensor(execution/'nll.pt', dict(initial=initial_nll, final=final_nll))
            save_json(execution/'history.json', history)
            status('autonomous_first_blocks')
            generations = []
            generation_started = time.perf_counter()
            for features in (h, blank):
                for i in range(0, len(features), 64):
                    generations.append(unpack_pixels(model.generate(features[i:i+64])).cpu())
                    status('autonomous_first_blocks', completed_blocks=sum(len(x) for x in generations))
            bits = torch.cat(generations).numpy()
            require(bits.shape == (896, 32, 64), 'complete registered output cohort required')
            np.save(execution/'ink.npy', bits, allow_pickle=False)
            images = execution/'images'
            images.mkdir()
            hashes = {}
            for i, ink in enumerate(bits):
                path = images/f'{i:04}.png'
                Image.fromarray((~ink).astype('uint8')*255).save(path)
                hashes[str(path.relative_to(execution))] = sha(path)
            changed_parameters = sum(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items())
            require(changed_parameters > 0 and replay is not None, 'actual learning/replay required')
            save_json(execution/'report.json', dict(parameters=parameters, trained_parameters=parameters,
                large_backbone_loaded=False, condition_source='frozen V68 custom image-actor cache',
                logical_updates=UPDATES, optimizer_step_calls=UPDATES+1, example_draws=UPDATES*BATCH,
                exact_replay_extra_draws=BATCH, initial_nll=float(initial_nll[:512].mean()),
                final_nll=float(final_nll[:512].mean()), training_seconds=training_seconds,
                generation_seconds=time.perf_counter()-generation_started,
                elapsed_seconds=time.perf_counter()-started, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                output_sha256=sha(execution/'ink.npy'), final_checkpoint_sha256=sha(execution/'final.pt'),
                nll_sha256=sha(execution/'nll.pt'), png_sha256=hashes, numerical_checks=checks,
                changed_state_entries=changed_parameters, actual_update_replay=replay,
                general_ilm_goal_complete=False, complete_answers_generated=False, stopping_evaluated=False))
            status('completed')
    except BaseException as error:
        status('failed', reason=repr(error))
        raise
    finally:
        state = load(execution/'status.json') if (execution/'status.json').exists() else {}
        save_json(execution/'status.json', dict(state, pid=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else run)(args.output.resolve())


if __name__ == '__main__':
    main()
