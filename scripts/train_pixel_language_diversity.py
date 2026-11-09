#!/usr/bin/env python3
"""Complete pixel-language learning with bounded, freshly read prompt windows.

Preparation is CPU-only. Qualification executes the actual learner and replay.
The main run starts again from the declared initial state, never fitted controls.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha
from ilm.visual_lm.pixel_language_diversity import PixelLanguageCorpus, WindowSampler, native_batch, state_digest
from ilm.visual_lm.complete_pixel_runtime import RUNTIME_CORE
from scripts.preflight_native_pixel_actor import CONTROLS, CODEC, prefix_identity, save_output
from scripts.train_conditional_pixel_density import atomic_tensor, equal_tree, require
from scripts.train_complete_pixel_language import physical_score

SEED = 202611094
UPDATES = 16000
INITIAL = ROOT / 'artifacts/native_pixel_numerics_v1/execution/initial_actor.pt'
INITIAL_SHA = 'e3949fc68937affb19e0ec22386b577f702dfb9e45b07f10c6d6cca735a160d6'
FILES = tuple(dict.fromkeys((*RUNTIME_CORE,
    'ilm/visual_lm/complete_pixel_runtime.py', 'ilm/visual_lm/pixel_language_diversity.py',
    'ilm/visual_lm/complete_pixel_training.py', 'ilm/image_edit/baseline.py',
    'scripts/train_pixel_language_diversity.py', 'scripts/audit_pixel_language_diversity.py',
    'scripts/train_complete_pixel_language.py', 'scripts/train_conditional_pixel_density.py',
    'scripts/preflight_native_pixel_actor.py', 'references/pixel_language_diversity_training_protocol.md')))


def load(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity():
    code = {name: sha(ROOT / name) for name in FILES}
    for name, digest in code.items():
        require(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT)).hexdigest() == digest,
                'commit the registered learning implementation first: ' + name)
    return code


def prepare(output, corpus_root):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA during preparation')
    code = identity()
    corpus = PixelLanguageCorpus(corpus_root)
    require(len(corpus.train) == 512 and sha(INITIAL) == INITIAL_SHA, 'registered cohort or initial state differs')
    audit = load(corpus.root / 'audit.json')
    require(audit['all512_complete_targets_match_independently_unpacked_source_bits'], 'independent complete-target audit required')
    # Source-defined stress batches, fixed before any new model output.
    by_prompt = sorted(range(512), key=lambda i: (-math.prod(corpus.train[i]['prompt_shape']), i))
    by_length = sorted(range(512), key=lambda i: (-corpus.train[i]['answer_blocks'], i))
    varied = [next(i for i in by_prompt if corpus.train[i]['answer_blocks'] == n) for n in (1, 3, 4, 7)]
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'preparation.json', dict(
        code_sha256=code, corpus=str(corpus.root.relative_to(ROOT)), corpus_sha256=corpus.bindings(),
        bound_sha256={str(p.relative_to(ROOT)): sha(p) for p in (INITIAL, CONTROLS, CODEC)},
        initial_state_sha256=INITIAL_SHA, seed=SEED, maximum_updates=UPDATES,
        batch_size=4, window=32, repeats=4, maximum_seconds=43200,
        maximum_allocated_vram_gib=22., maximum_generated_blocks=128,
        variable_length_indices=varied, largest_prompt_indices=by_prompt[:4], longest_answer_indices=by_length[:4],
        assessment_updates=[0, 4000, 8000, 16000], interim_training_indices=list(range(64)),
        final_training_indices=list(range(512)), development_sealed=False,
        train_only_early_stop=False, development_checkpoint_selection=False,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        independent_foundation=False, language_quality_claim=False, general_ilm_goal_complete=False))
    print(json.dumps(load(output / 'preparation.json') | {'corpus_sha256': 'all PNG hashes retained'}, indent=2))


def unchanged(output):
    meta = load(output / 'preparation.json')
    require(identity() == meta['code_sha256'], 'registered learning source changed')
    for name, value in meta['bound_sha256'].items():
        require(sha(ROOT / name) == value, 'bound source changed: ' + name)
    corpus = PixelLanguageCorpus(ROOT / meta['corpus'])
    require(corpus.bindings() == meta['corpus_sha256'], 'prepared image corpus changed')
    return meta, corpus


def process_identity():
    return dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])


def alive(status):
    if not status.get('pid') or not Path('/proc', str(status['pid']), 'stat').exists():
        return False
    return (status.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and status.get('start_ticks') == Path('/proc', str(status['pid']), 'stat').read_text().rsplit(')', 1)[1].split()[19])


class Learner:
    def __init__(self, output, folder, meta, corpus, device, status):
        import torch
        from ilm.visual_lm.complete_pixel_runtime import load_pixel_actor
        self.torch, self.output, self.folder, self.meta, self.corpus, self.device = torch, output, folder, meta, corpus, device
        self.status = status
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        status('loading_initial_pixel_actor')
        state = torch.load(INITIAL, map_location='cpu', weights_only=True)
        raster = {k[len('density.'):]: v for k, v in state.items() if k.startswith('density.')}
        local = ROOT.parent / 'LocalLLM/.local'
        self.actor, self.provenance = load_pixel_actor(state, raster, controls=CONTROLS, codec=CODEC,
            ollama_root=local / 'models/ollama', gguf_source=local / 'tools/llama.cpp/gguf-py', device=device)
        self.writer = self.actor.writing
        for name, parameter in self.writer.named_parameters():
            if (name.startswith(('density.', 'image_input.', 'stop.')) or name == 'start_query'
                    or (name.startswith('continuation.context.') and name.endswith(('.down', '.up')))):
                parameter.requires_grad_(True)
        require(equal_tree(state, self.writer.trainable_state_dict()), 'cold initialization differs')
        self.parameters = [p for p in self.writer.parameters() if p.requires_grad]
        require(sum(p.numel() for p in self.parameters) == 13855681, 'trainable architecture differs')
        self.optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in self.writer.named_parameters() if p.requires_grad and n.startswith('density.')], peak=3e-4),
            dict(params=[p for n, p in self.writer.named_parameters() if p.requires_grad and not n.startswith('density.')], peak=1e-4)
        ], lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
        self.sampler = WindowSampler(len(corpus.train), seed=SEED + 1)
        self.pool, self.codes = {}, {}
        self.prefix_hashes = load(folder / 'prefix_identities.json') if (folder / 'prefix_identities.json').exists() else {}
        self.frozen = {n: (id(p), p._version) for n, p in self.actor.named_parameters() if not p.requires_grad}
        self.calls = len((folder / 'optimizer_calls.jsonl').read_text().splitlines()) if (folder / 'optimizer_calls.jsonl').exists() else 0
        self.collector = None
        encode = self.writer.writer.encode_targets
        def traced(images):
            values = encode(images)
            if self.collector is not None:
                self.collector.append(dict(images=images.detach().cpu().clone(), codes=values.detach().cpu().clone()))
            return values
        self.writer.writer.encode_targets = traced
        del state, raster
        gc.collect()
        torch.cuda.empty_cache()

    def guard(self):
        require(self.torch.cuda.max_memory_allocated() / 1024**3 <= self.meta['maximum_allocated_vram_gib'], 'registered 22 GiB allocation exceeded')
        require(all((id(p), p._version) == self.frozen[n] for n, p in self.actor.named_parameters() if n in self.frozen), 'frozen reader changed')

    def ensure_pool(self, indices):
        if set(self.pool) == set(indices):
            return
        require(1 <= len(indices) <= 32, 'bounded native pool required')
        self.pool.clear()
        gc.collect()
        info = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        require(info['MemAvailable'] >= 24 * 1024**2 and (info['SwapTotal'] == 0 or
            (info['SwapTotal'] - info['SwapFree']) / info['SwapTotal'] <= .75), 'RAM/swap threshold reached before pool refill')
        self.actor.eval()
        for i in indices:
            prefix, geometry = self.actor.prefill(self.corpus.prompt(i))
            require(geometry['all_prompt_keys_retained'] and prefix.blocked_before == 0, 'full owning image prefix required')
            if str(i) not in self.prefix_hashes:
                self.prefix_hashes[str(i)] = prefix_identity(prefix)
                save_json(self.folder / 'prefix_identities.json', self.prefix_hashes)
            if i not in self.codes:
                self.codes[i] = self.writer.image_codes(self.corpus.targets(i)).cpu()
            self.pool[i] = (prefix.to('cpu'), self.codes[i])
            append(self.folder / 'prefill_calls.jsonl', dict(index=i, prefix_length=prefix.length,
                prompt_sha256=self.corpus.train[i]['prompt_sha256'], reader_memory_sha256=geometry['reader_memory_sha256']))
            del prefix
            self.guard()

    def loss(self, indices, *, checkpoint=True):
        from ilm.visual_lm.complete_pixel_training import complete_pixel_loss
        prefix, codes, lengths = native_batch(self.pool, indices, self.device)
        return complete_pixel_loss(self.writer, prefix, [self.corpus.targets(i) for i in indices], codes, lengths,
                                   checkpoint_layers=checkpoint)

    def gradients(self, *, nonzero):
        norms = {k: 0. for k in ('density', 'image_input', 'start_query', 'suffix', 'stop')}
        for name, p in self.writer.named_parameters():
            if p.grad is None:
                continue
            require(bool(self.torch.isfinite(p.grad).all()), 'nonfinite gradient: ' + name)
            key = 'suffix' if name.startswith('continuation.context.') else name.split('.')[0]
            if key in norms:
                norms[key] += float(p.grad.double().square().sum())
        norms = {k: v**.5 for k, v in norms.items()}
        if nonzero:
            require(all(v > 0 for v in norms.values()), 'required learning gradient is zero')
        return norms

    def update(self, step):
        indices = self.sampler.take()
        self.ensure_pool(self.sampler.current)
        self.actor.train()
        self.optimizer.zero_grad(set_to_none=True)
        factor = min(step / 100, 1.) * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1) / (UPDATES - 1))))
        for group in self.optimizer.param_groups:
            group['lr'] = group['peak'] * factor
        started = time.perf_counter()
        value, parts = self.loss(indices)
        require(bool(self.torch.isfinite(value)), 'nonfinite complete-image objective')
        value.backward()
        gradients = self.gradients(nonzero=step <= 2)
        norm = self.torch.nn.utils.clip_grad_norm_(self.parameters, 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.torch.cuda.synchronize()
        self.calls += 1
        append(self.folder / 'optimizer_calls.jsonl', dict(call=self.calls, update=step, indices=indices, seconds=time.perf_counter()-started))
        self.guard()
        return dict(update=step, indices=indices, loss=float(value.detach()), image_nll=float(parts['image_nll']),
            stop=float(parts['stop']), complete_blocks=parts['complete_blocks'], gradient_norm=float(norm),
            gradient_components=gradients, density_learning_rate=self.optimizer.param_groups[0]['lr'],
            other_learning_rate=self.optimizer.param_groups[1]['lr'])

    def snapshot(self, step, *, complete=False):
        from ilm.visual_lm.complete_pixel_training import cpu_tree
        return dict(model=self.writer.trainable_state_dict(), optimizer=cpu_tree(self.optimizer.state_dict()),
            sampler=self.sampler.state_dict(), cpu_rng=self.torch.get_rng_state(), cuda_rng=self.torch.cuda.get_rng_state(),
            update=step, preparation_sha256=sha(self.output/'preparation.json'), training_complete=complete,
            frozen_weights_saved=False)

    def restore(self, value):
        require(value['preparation_sha256'] == sha(self.output / 'preparation.json'), 'checkpoint preparation differs')
        self.writer.load_trainable_state_dict(value['model'])
        self.optimizer.load_state_dict(value['optimizer'])
        self.sampler.load_state_dict(value['sampler'])
        self.torch.set_rng_state(value['cpu_rng'])
        self.torch.cuda.set_rng_state(value['cuda_rng'])

    def save(self, step, *, complete=False):
        latest = self.folder / 'checkpoint_latest.pt'
        if latest.exists():
            latest.replace(self.folder / 'checkpoint_previous.pt')
        atomic_tensor(latest, self.snapshot(step, complete=complete))

    def model_digest(self):
        return state_digest(self.writer.trainable_state_dict())

    def assess(self, step, *, resume):
        self.actor.eval()
        self.optimizer.zero_grad(set_to_none=True)
        stage = self.folder / f'assessment_{step:05}'
        stage.mkdir(exist_ok=True)
        state_hash = self.model_digest()
        monitor = stage / 'teacher_history.json'
        if monitor.exists():
            require(load(monitor)['model_sha256'] == state_hash, 'existing assessment state differs')
        else:
            values = []
            with self.torch.no_grad():
                for start in range(0, 512, 32):
                    self.ensure_pool(list(range(start, start+32)))
                    for offset in range(start, start+32, 4):
                        value, parts = self.loss(list(range(offset, offset+4)), checkpoint=False)
                        values.append(dict(loss=float(value), image_nll=float(parts['image_nll']), stop=float(parts['stop'])))
                    self.status('whole_training_correct_history_assessment', completed=start+32, total=512)
            save_json(monitor, dict(update=step, model_sha256=state_hash, questions=512,
                **{k: sum(r[k] for r in values)/len(values) for k in values[0]}))
        self.pool.clear()
        gc.collect()
        indices = list(range(512 if step == UPDATES else 64))
        jobs = [(False, i, 'normal') for i in indices] + [(True, i, route) for i in range(24) for route in ('normal', 'blank')]
        scores = []
        for done, (dev, index, route) in enumerate(jobs):
            key = ('development' if dev else 'train') + f'_{index:04}_{route}'
            folder = stage / key
            expected = dict(index=index, development=dev, route=route, update=step,
                model_sha256=state_hash, preparation_sha256=sha(self.output/'preparation.json'), actual_PNG_prefill=True)
            receipt = folder / 'receipt.json'
            if receipt.exists():
                saved = load(receipt)
                require(saved['identity'] == expected, 'saved generation belongs to a different state')
                require(all(sha(folder/n) == v for n, v in saved['hashes'].items()), 'saved generation changed')
                result = self.torch.load(folder/'result.pt', map_location='cpu', weights_only=True)
            else:
                if folder.exists():
                    require(resume, 'preserve an unexpected partial generation')
                    folder.rename(folder.with_name(folder.name + '_uncommitted_' + str(time.time_ns())))
                self.collector = []
                result = self.actor.generate(self.corpus.prompt(index, development=dev, route=route), maximum_blocks=128)
                output = save_output(folder, result)
                atomic_tensor(folder/'feedback.pt', self.collector)
                self.collector = None
                save_json(receipt, dict(identity=expected, output=output,
                    hashes={n: sha(folder/n) for n in ('result.pt', 'answer.png', 'feedback.pt')}))
            # Read desired images only after autonomous generation is durably saved.
            score = physical_score(result['images'], self.corpus.references(index, development=dev), result['stopped'], result['capped'])
            scores.append(dict(key=key, index=index, development=dev, route=route, **score,
                stopped=result['stopped'], capped=result['capped'], blocks=len(result['images'])))
            self.guard()
            self.status('autonomous_complete_image_assessment', completed=done+1, total=len(jobs))
        save_json(stage/'report.json', dict(update=step, model_sha256=state_hash, cases=scores, semantic_accuracy_assessed=False))


def qualify(learner):
    torch = learner.torch
    learner.ensure_pool(learner.meta['variable_length_indices'])
    selected = learner.meta['variable_length_indices']
    learner.actor.eval()
    with torch.no_grad():
        prefix, codes, lengths = native_batch(learner.pool, selected, learner.device)
        full = learner.writer.conditions(prefix, codes, checkpoint_layers=False)
        independent, relative, absolute = [], 0., 0.
        for b, i in enumerate(selected):
            own = learner.writer.conditions(learner.pool[i][0].to(learner.device), learner.pool[i][1][None].to(learner.device), checkpoint_layers=False)[0]
            delta = full[b, :len(own)] - own
            relative = max(relative, float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(own).clamp_min(1e-12)))
            absolute = max(absolute, float(delta.abs().max()))
            independent.append(float(learner.writer.loss_from_codes(learner.pool[i][0].to(learner.device),
                learner.corpus.targets(i).to(learner.device), learner.codes[i].to(learner.device), writer_blocks=128)[0]))
        combined = float(learner.loss(selected, checkpoint=False)[0])
        mean = sum(independent)/4
        require(relative <= 1e-4 and abs(combined-mean) <= 1e-4*max(1., abs(mean)), 'variable batch differs from owning calls')
        changed = codes.clone()
        for b, i in enumerate(selected):
            if len(learner.corpus.targets(i)) > 1:
                changed[b, 1] = learner.writer.image_codes(-learner.corpus.targets(i)[1:2])[0]
        future = learner.writer.conditions(prefix, changed, checkpoint_layers=False)
        require(torch.equal(full[:, :2], future[:, :2]), 'future image leaked into current query')
        for b, n in enumerate(lengths.tolist()):
            changed[b, n:] = 19.
        padded = learner.writer.conditions(prefix, changed, checkpoint_layers=False)
        require(all(torch.equal(future[b, :n+1], padded[b, :n+1]) for b, n in enumerate(lengths.tolist())), 'padded history contaminated valid queries')
        batch_report = dict(indices=selected, lengths=lengths.tolist(), normwise_error=relative, absolute_error=absolute,
            batch_loss=combined, mean_independent_loss=mean, future_image_exclusion_exact=True, future_padding_exclusion_exact=True)
    del prefix, codes, lengths, full, future, padded, changed, own
    stress = []
    for name in ('largest_prompt_indices', 'longest_answer_indices'):
        indices = learner.meta[name]
        learner.ensure_pool(indices)
        learner.actor.train()
        learner.optimizer.zero_grad(set_to_none=True)
        loss, parts = learner.loss(indices)
        require(bool(torch.isfinite(loss)), 'stress objective is nonfinite')
        loss.backward()
        stress.append(dict(name=name, indices=indices, loss=float(loss.detach()), gradients=learner.gradients(nonzero=True)))
        learner.guard()
        learner.optimizer.zero_grad(set_to_none=True)
        del loss, parts
    first = learner.update(1)
    atomic_tensor(learner.folder/'step_one.pt', learner.snapshot(1))
    snapshot = torch.load(learner.folder/'step_one.pt', map_location='cpu', weights_only=True)
    second = learner.update(2)
    expected = learner.snapshot(2)
    atomic_tensor(learner.folder/'step_two.pt', expected)
    learner.restore(snapshot)
    repeated = learner.update(2)
    require(second == repeated and equal_tree(expected, learner.snapshot(2)), 'actual model/Adam/sampler/RNG/loss/gradient replay differs')
    report = dict(preparation_sha256=sha(learner.output/'preparation.json'), batch=batch_report, stress=stress,
        first=first, second=second, repeated=repeated, actual_optimizer_calls=learner.calls,
        step_two_sha256=sha(learner.folder/'step_two.pt'), checks=dict(variable_batch_matches=True,
        future_and_padding_excluded=True, stress_backward_finite_nonzero=True, actual_replay_bit_exact=True),
        peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        source=learner.provenance, qualification_updates_retained_for_main_run=False)
    save_json(learner.folder/'report.json', report)


def run_main(learner, *, resume):
    torch = learner.torch
    qualification = learner.output/'qualification'
    qualified = load(qualification/'report.json')
    require(load(qualification/'status.json')['phase'] == 'completed' and all(qualified['checks'].values())
        and qualified['preparation_sha256'] == sha(learner.output/'preparation.json')
        and sha(qualification/'step_two.pt') == qualified['step_two_sha256'], 'actual qualification required')
    step = 0
    if resume:
        checkpoint = learner.folder/'checkpoint_latest.pt'
        if not checkpoint.exists():
            checkpoint = learner.folder/'checkpoint_previous.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(not state['training_complete'], 'finished training may not be resumed')
        learner.restore(state)
        step = state['update']
        del state
        history = learner.folder/'history.jsonl'
        lines = history.read_text().splitlines(keepends=True)
        require(len(lines) >= step and all(json.loads(s)['update'] == i+1 for i, s in enumerate(lines[:step])), 'committed history incomplete')
        if len(lines) > step:
            history.with_name('uncommitted_history_' + str(time.time_ns()) + '.jsonl').write_text(''.join(lines[step:]))
            history.write_text(''.join(lines[:step]))
    if step == 0:
        learner.assess(0, resume=resume)
    elif step in learner.meta['assessment_updates']:
        learner.assess(step, resume=resume)
    for next_step in range(step+1, UPDATES+1):
        record = learner.update(next_step)
        step = next_step
        append(learner.folder/'history.jsonl', record)
        if step <= 2:
            require(record == qualified['first' if step == 1 else 'second'], 'fresh main update differs from qualification')
            if step == 2:
                expected = torch.load(qualification/'step_two.pt', map_location='cpu', weights_only=True)
                require(equal_tree(expected, learner.snapshot(2)), 'fresh main model/Adam/sampler/RNG differs from qualification')
                del expected
                save_json(learner.folder/'cold_start_replay.json', dict(first_two_updates_bit_exact=True, qualification_report_sha256=sha(qualification/'report.json')))
                learner.save(step)
        if step % 25 == 0:
            learner.status('complete_image_training', update=step, retained_draws=learner.sampler.draws, last=record)
        if step % 250 == 0:
            learner.save(step)
        if step in learner.meta['assessment_updates']:
            learner.assess(step, resume=resume)
    learner.pool.clear()
    gc.collect()
    learner.actor.eval()
    for i in range(32):
        prefix, _ = learner.actor.prefill(learner.corpus.prompt(i))
        require(prefix_identity(prefix) == learner.prefix_hashes[str(i)], 'frozen prompt prefix changed after learning')
        del prefix
    require(len(learner.prefix_hashes) == 512, 'some training prompt was never read')
    learner.save(step, complete=True)
    save_json(learner.folder/'report.json', dict(updates=step, retained_draws=learner.sampler.draws,
        actual_optimizer_calls=learner.calls, checkpoint_sha256=sha(learner.folder/'checkpoint_latest.pt'),
        all32_control_full_prefixes_bit_exact_after_learning=True, all512_training_inputs_read=True,
        complete_parameters=sum(p.numel() for p in learner.actor.parameters()), trainable_parameters=sum(p.numel() for p in learner.parameters),
        source=learner.provenance, peak_allocated_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
        language_quality_not_yet_audited=True, general_ilm_goal_complete=False))


def execute(output, *, stage, resume=False):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID', 'physical GPU0 required')
    meta, corpus = unchanged(output)
    folder = output/('qualification' if stage == 'qualify' else 'execution')
    prior = {}
    if resume:
        require(stage == 'run' and folder.exists(), 'only the main execution supports explicit resume')
        prior = load(folder/'status.json')
        require(not alive(prior) and prior['phase'] != 'completed', 'a live or completed process may not be resumed')
        require(any((folder/n).exists() for n in ('checkpoint_latest.pt', 'checkpoint_previous.pt')), 'actual saved optimizer state required')
    else:
        folder.mkdir(exist_ok=False)
    started = time.perf_counter()
    previous_seconds = prior.get('active_seconds', 0.)
    pid = process_identity()
    def status(phase, **extra):
        value = dict(phase=phase, **pid, active_seconds=previous_seconds+time.perf_counter()-started, **extra)
        save_json(folder/'status.json', value)
        append(folder/'events.jsonl', value)
    def deadline(signum, frame):
        raise TimeoutError('registered 12-hour active-process budget exhausted')
    remaining = meta['maximum_seconds'] - previous_seconds
    require(remaining > 0, 'registered active-process budget exhausted')
    old_handler = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(max(1, math.ceil(remaining)))
    try:
        with gpu_zero() as device:
            learner = Learner(output, folder, meta, corpus, device, status)
            if stage == 'qualify':
                qualify(learner)
            else:
                run_main(learner, resume=resume)
            status('completed')
    except BaseException as error:
        status('failed', reason=repr(error))
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        value = load(folder/'status.json') if (folder/'status.json').exists() else {}
        save_json(folder/'status.json', dict(value, pid=None, active_seconds=previous_seconds+time.perf_counter()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'qualify', 'run'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, default=ROOT/'artifacts/pixel_language_diversity_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output.resolve(), args.corpus.resolve())
    else:
        execute(args.output.resolve(), stage=args.stage, resume=args.resume)
