#!/usr/bin/env python3
"""Post-training diagnosis of first-image learning versus greedy decoding.

Preparation authenticates actual assessed states and original saved images on
CPU. Neural scoring requires the original training owner to have exited and
the existing exclusive GPU0 admission. No answer image is generated or repaired.
"""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from ilm.image_edit.baseline import ROOT, gpu_zero, save_json, sha

EPSILON = 1e-8


def host_resources():
    info = {line.split(':')[0]: int(line.split()[1])
            for line in Path('/proc/meminfo').read_text().splitlines()}
    pressure = {}
    for line in Path('/proc/pressure/memory').read_text().splitlines():
        fields = line.split()
        pressure[fields[0]] = dict(field.split('=') for field in fields[1:])
    return dict(available_gib=info['MemAvailable']/1024**2,
                swap_fraction=(info['SwapTotal']-info['SwapFree'])/info['SwapTotal'] if info['SwapTotal'] else 0.,
                some_pressure_avg10=float(pressure['some']['avg10']),
                full_pressure_avg10=float(pressure['full']['avg10']))


def reserve_check(observed, *, entry=False):
    require(observed['available_gib'] >= (64 if entry else 24),
            'read-only diagnostic RAM reserve unavailable')
    require(observed['some_pressure_avg10'] <= .5 and observed['full_pressure_avg10'] <= .1,
            'read-only diagnostic memory pressure limit reached')


@contextmanager
def diagnostic_gpu():
    """Keep training admission intact; admit a bounded post-terminal reader.

    After checking that this project has no obsolete jobs, high global swap
    occupancy can coexist with ample available RAM. The alternate diagnostic
    admission requires a 64 GiB entry reserve and low actual memory stalls.
    It never changes swap, kernel settings, or another project's processes.
    """
    entry = host_resources()
    if entry['swap_fraction'] <= .75:
        with gpu_zero() as device:
            yield device, dict(mode='original_RAM_swap_admission', entry=entry)
        return
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'physical GPU0 required')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        entry = host_resources()
        reserve_check(entry, entry=True)
        free = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip()
        require(int(free) >= 20*1024, 'GPU0 needs20 GiB free before diagnostic loading')
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
            os.environ[name] = '1'
        import torch
        require(torch.cuda.is_available(), 'selected runtime cannot useGPU0')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        torch.set_num_threads(4)
        yield torch.device('cuda:0'), dict(mode='post_terminal_RAM_reserve_and_PSI', entry=entry,
            minimum_entry_available_gib=64, minimum_runtime_available_gib=24,
            maximum_some_pressure_avg10=.5, maximum_full_pressure_avg10=.1,
            global_swap_fraction_gate=False, training_admission_unchanged=True)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pixels(path):
    with Image.open(path) as image:
        result = np.asarray(image.convert('L')).copy()
    require(bool(((result == 0) | (result == 255)).all()), 'original binary pixels required')
    return result


def image_hash(array):
    require(array.shape == (32, 64), 'a complete first image block is required')
    return hashlib.sha256(array.tobytes()).hexdigest()


def owner_live(status):
    if not status.get('pid'):
        return False
    try:
        fields = Path('/proc', str(status['pid']), 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return (status.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and status.get('start_ticks') == fields[19])


def owner_gone(run):
    status = load(run/'execution/status.json')
    owners = [status]
    # The trainer clears status.pid in finally, slightly before its process
    # exits. Its event ledger retains the actual last owner identity.
    if not status.get('pid'):
        for line in reversed((run/'execution/events.jsonl').read_text().splitlines()):
            event = json.loads(line)
            if event.get('pid'):
                owners.append(event)
                break
    for owner in owners:
        require(not owner_live(owner),
                f"Original training owner {owner.get('pid')} is still live; refusing a second model stage; "
                f"torch_imported={'torch' in sys.modules}")
    require(status['phase'] in ('completed', 'failed'), 'the original execution must be terminal')


def state_spec(run, step, meta):
    report = run/'execution'/f'assessment_{step:05}'/'report.json'
    source = load(report)
    require(source['update'] == step, 'assessment update differs')
    if step == 0:
        path = next(ROOT/name for name, value in meta['bound_sha256'].items()
                    if value == meta['initial_state_sha256'])
    elif step in (4000, 8000):
        path = run/'milestone_models'/f'actor_update{step:05}.pt'
        receipt = load(path.with_suffix('.json'))
        require(receipt['assessment_sha256'] == sha(report)
                and receipt['model_sha256'] == source['model_sha256']
                and receipt['archive_sha256'] == sha(path), 'owning assessed archive required')
    else:
        require(step == meta['maximum_updates'], 'unsupported state')
        path = run/'execution/checkpoint_latest.pt'
    return dict(update=step, checkpoint=relative(path), checkpoint_sha256=sha(path),
                assessment=relative(report), assessment_sha256=sha(report), model_sha256=source['model_sha256'])


def read_state(spec, run):
    import torch
    from ilm.visual_lm.complete_pixel_runtime import read_pixel_checkpoint
    from ilm.visual_lm.pixel_language_diversity import state_digest
    path = ROOT/spec['checkpoint']
    require(sha(path) == spec['checkpoint_sha256'], 'owning state file changed')
    if spec['update'] == 0:
        state = torch.load(path, map_location='cpu', weights_only=True)
    else:
        state, _, identity = read_pixel_checkpoint(path, run, allow_incomplete=True)
        require(identity['checkpoint_update'] == spec['update'], 'state update differs')
        if spec['update'] == 16000:
            require(identity['checkpoint_training_finished'], 'the final checkpoint must be complete')
    require(len(state) == 286 and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                                    for v in state.values()), 'finite owning FP32 tensors required')
    require(state_digest(state) == spec['model_sha256'], 'assessed model fingerprint differs')
    return state


def prepare(output, run, steps):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for CPU preparation')
    require(not output.exists(), 'preserve existing and partial diagnostic preparations')
    import torch
    torch.set_num_threads(2)
    meta = load(run/'preparation.json')
    require(len(set(steps)) == len(steps) and all(s in meta['assessment_updates'] for s in steps),
            'distinct registered assessment states required')
    bindings = {relative(run/'preparation.json'): sha(run/'preparation.json'), **meta['bound_sha256']}
    code = {**meta['code_sha256'], relative(__file__): sha(__file__)}
    for name, digest in {**bindings, **code}.items():
        require(sha(ROOT/name) == digest, 'bound source changed: '+name)
    corpus = ROOT/meta['corpus']
    manifest = load(corpus/'manifest.json')
    bindings[relative(corpus/'manifest.json')] = meta['corpus_sha256']['manifest.json']
    require(sha(corpus/'manifest.json') == bindings[relative(corpus/'manifest.json')], 'corpus changed')
    specs = [state_spec(run, step, meta) for step in sorted(steps)]
    for spec in specs:
        state = read_state(spec, run)
        del state
        bindings[spec['checkpoint']] = spec['checkpoint_sha256']
        bindings[spec['assessment']] = spec['assessment_sha256']
        if spec['update'] in (4000, 8000):
            receipt = (ROOT/spec['checkpoint']).with_suffix('.json')
            bindings[relative(receipt)] = sha(receipt)
    jobs, references, by_pair = [], {}, {}
    indices = [(False, i, 'normal') for i in range(64)]
    indices += [(True, i, route) for i in range(24) for route in ('normal', 'blank')]
    for dev, index, route in indices:
        row = manifest['development' if dev else 'train'][index]
        key = ('development' if dev else 'train')+f'_{index:04}_{route}'
        prompt = row[route] if dev else row['prompt']
        expected = row[route+'_sha256'] if dev else row['prompt_sha256']
        bindings[relative(corpus/prompt)] = expected
        ids = []
        refs = row['references'] if dev else [dict(path=row['answer'], sha256=row['answer_sha256'])]
        for ref in refs:
            path = corpus/ref['path']
            require(sha(path) == ref['sha256'], 'owning reference changed')
            first = pixels(path)[:, :64]
            digest = image_hash(first)
            references.setdefault(digest, dict(path=relative(path), file_sha256=ref['sha256']))
            bindings[relative(path)] = ref['sha256']
            if digest not in ids:
                ids.append(digest)
        require(ids, 'a reference first image is required')
        answer = ('development' if dev else 'train')+f'_{index:04}'
        group = row['source']+'_'+route if dev else ('train_old' if index < 32 else 'train_new')
        pair = None if dev and row['source'] != 'cmrc' else ('development' if dev else 'train')+f"_{row['pair']}"
        job = dict(key=key, answer=answer, group=group, development=dev, index=index, route=route,
                   prompt=relative(corpus/prompt), reference_ids=ids, pair=pair, saved={})
        for spec in specs:
            folder = run/'execution'/f"assessment_{spec['update']:05}"/key
            receipt = load(folder/'receipt.json')
            require(receipt['identity'] == dict(index=index, development=dev, route=route, update=spec['update'],
                model_sha256=spec['model_sha256'], preparation_sha256=sha(run/'preparation.json'), actual_PNG_prefill=True),
                'owning original generation receipt differs')
            bindings[relative(folder/'receipt.json')] = sha(folder/'receipt.json')
            for name, digest in receipt['hashes'].items():
                require(sha(folder/name) == digest, 'saved actual generation changed')
                bindings[relative(folder/name)] = digest
            result = torch.load(folder/'result.pt', map_location='cpu', weights_only=True)
            images = result['images']
            n = len(images)
            require(images.shape == (n, 1, 32, 64) and bool(((images == -1) | (images == 1)).all()),
                    'actual binary image blocks required')
            first = pixels(folder/'answer.png')[:, :64]
            if n:
                require(np.array_equal(first, ((images[0, 0].numpy()+1)*127.5).astype('uint8')),
                        'original first PNG differs from emitted tensor')
            job['saved'][str(spec['update'])] = dict(path=relative(folder/'answer.png'), blocks=n,
                first_sha256=image_hash(first) if n else None,
                initial_stop_probability=result['stop_probabilities'][0], reader_memory_sha256=result['reader_memory_sha256'])
        jobs.append(job)
        if pair and route == 'normal':
            by_pair.setdefault(pair, []).append(key)
    require(all(len(v) == 2 for v in by_pair.values()), 'complete same-passage question pairs required')
    for name, digest in bindings.items():
        require(sha(ROOT/name) == digest, 'prepared input changed: '+name)
    output.mkdir(parents=True)
    save_json(output/'preparation.json', dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256=code, bindings_sha256=bindings, training_run=relative(run), states=specs, jobs=jobs,
        references=references, pairs=by_pair, decoder_gap_tolerance=EPSILON, head_batch=4,
        maximum_allocated_vram_gib=22., post_hoc_diagnostic=True, development_sealed=False,
        neural_forward_calls=0, optimizer_updates=0, CUDA_initialized=torch.cuda.is_initialized(),
        neural_scoring_executed=False, language_quality_claim=False,
        diagnostic_admission='Original GPU0 admission, or post-terminal64GiB entry reserve plus low memory PSI;24GiB runtime reserve; same GPU0 lock,20GiB cold free GPU and22GiB allocated cap. Training guard unchanged.'))
    print(json.dumps(dict(prepared=True, jobs=len(jobs), states=[s['update'] for s in specs],
        unique_reference_first_images=len(references), pairs=len(by_pair), neural_forward_calls=0), indent=2))


def logsumexp(values):
    highest = max(values)
    return highest+math.log(sum(math.exp(v-highest) for v in values))


def summarize(rows, jobs, tolerance):
    groups = {}
    for label in sorted({j['group'] for j in jobs}):
        selected = [rows[j['key']] for j in jobs if j['group'] == label]
        decoding = [r['maximum_reference_minus_greedy'] for r in selected
                    if r['maximum_reference_minus_greedy'] is not None]
        margins = [r['owning_minus_other_question'] for r in selected if 'owning_minus_other_question' in r]
        groups[label] = dict(cases=len(selected), nonempty_saved_outputs=len(decoding),
            reference_witness_of_nonMAP_head_image=sum(v > tolerance for v in decoding),
            witness_among_saved_first_image_mismatches=sum(
                r['maximum_reference_minus_greedy'] is not None and not r['first_image_matches_reference']
                and r['maximum_reference_minus_greedy'] > tolerance for r in selected),
            mean_maximum_reference_minus_greedy=sum(decoding)/len(decoding) if decoding else None,
            paired_question_comparisons=len(margins), owning_question_preference_positive=sum(v > tolerance for v in margins),
            mean_owning_question_margin=sum(margins)/len(margins) if margins else None)
    return groups


def run(output):
    prepared = load(output/'preparation.json')
    training = ROOT/prepared['training_run']
    owner_gone(training)  # Intentionally before importing any neural framework.
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID',
            'physical GPU0 required')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'use the original deterministic CUBLAS configuration')
    execution = output/'execution'
    require(not execution.exists(), 'preserve earlier or interrupted scoring; prepare a new directory')
    for name, digest in {**prepared['code_sha256'], **prepared['bindings_sha256']}.items():
        require(sha(ROOT/name) == digest, 'prepared source changed: '+name)
    for name, digest in prepared['code_sha256'].items():
        require(hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)).hexdigest() == digest,
                'commit the scoring implementation before neural execution')
    import torch
    from torch.nn import functional as F
    from ilm.visual_lm.complete_pixel_runtime import load_pixel_actor
    from ilm.visual_lm.conditional_pixel_density import pack_pixels
    execution.mkdir()
    began = time.perf_counter()
    pid = dict(pid=os.getpid(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
               start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    def status(phase, **fields):
        save_json(execution/'status.json', dict(phase=phase, **pid, seconds=time.perf_counter()-began, **fields))
    queries, scores = {}, {}
    calls = dict(native_prefill=0, writing_query=0, head_forward=0, autonomous_generation=0, optimizer_update=0)
    resource_samples = []
    def memory_check():
        require(torch.cuda.max_memory_allocated()/1024**3 <= prepared['maximum_allocated_vram_gib'],
                'diagnostic exceeds its allocation cap')
        sample = host_resources()
        if admission['mode'] == 'post_terminal_RAM_reserve_and_PSI':
            reserve_check(sample)
        resource_samples.append(sample)
    try:
        with diagnostic_gpu() as (device, admission), torch.no_grad():
            save_json(execution/'admission.json', admission)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            states = {s['update']: read_state(s, training) for s in prepared['states']}
            original = load(training/'preparation.json')
            control = next(ROOT/n for n in original['bound_sha256'] if Path(n).name == 'structural_vectors.pt')
            initial = next(ROOT/n for n, h in original['bound_sha256'].items() if h == original['initial_state_sha256'])
            codec = next(ROOT/n for n in original['bound_sha256'] if ROOT/n not in (control, initial))
            first = states[prepared['states'][0]['update']]
            local = ROOT.parent/'LocalLLM/.local'
            status('loading_single_actor')
            actor, provenance = load_pixel_actor(first, {k[8:]: v for k, v in first.items() if k.startswith('density.')},
                controls=control, codec=codec, ollama_root=local/'models/ollama',
                gguf_source=local/'tools/llama.cpp/gguf-py', device=device)
            writer = actor.writing
            memory_check()
            for done, job in enumerate(prepared['jobs'], 1):
                prompt = torch.from_numpy(pixels(ROOT/job['prompt'])).float()[None, None]/127.5-1
                prefix, geometry = actor.prefill(prompt)
                calls['native_prefill'] += 1
                for step, state in states.items():
                    saved = job['saved'][str(step)]
                    require(geometry['reader_memory_sha256'] == saved['reader_memory_sha256'], 'cold original-reader fingerprint differs')
                    writer.load_trainable_state_dict(state)
                    started = writer.start_prefix(prefix)
                    h = started.initial_state[:, 0].float()
                    stop_logit = writer.stop(h).float()[0, 0]
                    require(float(stop_logit.sigmoid()) == saved['initial_stop_probability'], 'cold initial stop probability differs')
                    queries.setdefault(str(step), {})[job['key']] = dict(raw=h.cpu().clone(),
                        normalized=writer.normalized_conditions(h).cpu().clone(),
                        continue_logp=float(F.logsigmoid(-stop_logit.double())))
                    calls['writing_query'] += 1
                    del started, h
                    memory_check()
                del prefix, prompt
                status('capturing_prompt_only_queries', completed=done, total=len(prepared['jobs']))
            torch.save(queries, execution/'queries.pt')
            # Keep only the small density after prompt capture; target images
            # have not entered the context computation.
            head = writer.density
            del actor, writer
            gc.collect()
            torch.cuda.empty_cache()
            targets = {key: pixels(ROOT/value['path'])[:, :64] for key, value in prepared['references'].items()}
            lookup = {j['key']: j for j in prepared['jobs']}
            for step, state in states.items():
                head.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith('density.')}, strict=True)
                result_rows = {}
                for done, job in enumerate(prepared['jobs'], 1):
                    key = job['key']
                    query = queries[str(step)][key]
                    condition = query['normalized'].to(device)
                    saved = job['saved'][str(step)]
                    candidate_scores, greedy_logp = {}, None
                    if saved['blocks']:
                        image = pixels(ROOT/saved['path'])[:, :64]
                        patterns = pack_pixels(torch.from_numpy(image == 0).to(device)[None])
                        logits = head(condition, patterns)
                        calls['head_forward'] += 1
                        require(bool(torch.isfinite(logits).all()) and torch.equal(logits.argmax(-1), patterns),
                                'cold full-forward saved-prefix argmax replay differs')
                        greedy_logp = float(F.log_softmax(logits, -1).gather(-1, patterns[..., None]).sum())
                        candidate_scores[saved['first_sha256']] = greedy_logp
                    needed = set(job['reference_ids'])
                    if job['pair']:
                        other = next(k for k in prepared['pairs'][job['pair']] if k != key.replace('_blank', '_normal'))
                        needed.update(lookup[other]['reference_ids'])
                    todo = sorted(needed-set(candidate_scores))
                    for start in range(0, len(todo), prepared['head_batch']):
                        selected = todo[start:start+prepared['head_batch']]
                        ink = np.stack([targets[k] == 0 for k in selected])
                        patterns = pack_pixels(torch.from_numpy(ink).to(device))
                        logits = head(condition.expand(len(selected), -1), patterns)
                        calls['head_forward'] += 1
                        require(bool(torch.isfinite(logits).all()), 'nonfinite reference likelihood')
                        values = F.log_softmax(logits, -1).gather(-1, patterns[..., None]).sum((1, 2)).cpu().tolist()
                        candidate_scores.update(zip(selected, values))
                    own = [candidate_scores[k] for k in job['reference_ids']]
                    result_rows[key] = dict(reference_logp={k: candidate_scores[k] for k in sorted(needed)},
                        owning_reference_event_logp=logsumexp(own), continue_logp=query['continue_logp'],
                        saved_first_image_logp=greedy_logp,
                        first_image_matches_reference=saved['first_sha256'] in job['reference_ids'] if saved['blocks'] else None,
                        maximum_reference_minus_greedy=max(own)-greedy_logp if greedy_logp is not None else None,
                        saved_prefix_argmax_replay_passed=True if saved['blocks'] else None)
                    memory_check()
                    status('scoring_first_images', update=step, completed=done, total=len(prepared['jobs']))
                for job in prepared['jobs']:
                    row = result_rows[job['key']]
                    if job['pair'] and job['route'] == 'normal':
                        other = next(k for k in prepared['pairs'][job['pair']] if k != job['key'])
                        alternative = result_rows[other]
                        other_event = logsumexp([alternative['reference_logp'][k] for k in job['reference_ids']])
                        row['owning_minus_other_question'] = row['owning_reference_event_logp']-other_event
                        row['owning_minus_other_with_continue'] = row['owning_minus_other_question']+row['continue_logp']-alternative['continue_logp']
                    if job['development'] and job['route'] == 'normal':
                        blank = result_rows[job['key'].replace('_normal', '_blank')]
                        row['normal_minus_erased_question'] = row['owning_reference_event_logp']-blank['owning_reference_event_logp']
                scores[str(step)] = result_rows
                save_json(execution/f'scores_{step:05}.json', result_rows)
            peak = torch.cuda.max_memory_allocated()/1024**3
            pairwise = {}
            for step, rows in scores.items():
                items = []
                for name, keys in prepared['pairs'].items():
                    a, b = (rows[k] for k in keys)
                    margin = a['owning_minus_other_question']+b['owning_minus_other_question']
                    with_continue = a['owning_minus_other_with_continue']+b['owning_minus_other_with_continue']
                    require(abs(margin-with_continue) <= prepared['decoder_gap_tolerance'], 'paired continuation factors failed to cancel')
                    items.append(dict(pair=name, keys=keys,
                        reference_first_events_disjoint=not bool(set(lookup[keys[0]]['reference_ids']) & set(lookup[keys[1]]['reference_ids'])),
                        summed_owning_question_margin=margin,
                        both_owning_margins_positive=all(v['owning_minus_other_question'] > prepared['decoder_gap_tolerance'] for v in (a, b))))
                pairwise[step] = items
            save_json(execution/'report.json', dict(preparation_sha256=sha(output/'preparation.json'),
                provenance=provenance, states=prepared['states'], queries_sha256=sha(execution/'queries.pt'),
                scores_sha256={str(s): sha(execution/f'scores_{s:05}.json') for s in states},
                calls=calls, peak_allocated_vram_gib=peak, elapsed_seconds=time.perf_counter()-began,
                resource_admission=admission,
                resource_observations=dict(samples=len(resource_samples),
                    minimum_available_gib=min(v['available_gib'] for v in resource_samples),
                    maximum_swap_fraction=max(v['swap_fraction'] for v in resource_samples),
                    maximum_some_pressure_avg10=max(v['some_pressure_avg10'] for v in resource_samples),
                    maximum_full_pressure_avg10=max(v['full_pressure_avg10'] for v in resource_samples)),
                summaries={step: summarize(rows, prepared['jobs'], prepared['decoder_gap_tolerance']) for step, rows in scores.items()},
                paired_question_margins=pairwise, decoder_gap_tolerance=prepared['decoder_gap_tolerance'],
                all_reader_fingerprints_and_initial_stop_probabilities_exact=True,
                all_nonempty_saved_first_images_pass_full_forward_argmax_replay=True,
                reference_images_used_only_as_likelihood_targets=True, new_answer_images_generated=0,
                semantic_accuracy_measured=False, language_quality_claim=False))
            status('completed')
    except BaseException as error:
        status('failed', reason=repr(error))
        raise
    finally:
        if (execution/'status.json').exists():
            save_json(execution/'status.json', dict(load(execution/'status.json'), pid=None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prepare', 'run'), required=True)
    parser.add_argument('--training-run', type=Path)
    parser.add_argument('--updates', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.stage == 'prepare':
            training = args.training_run or ROOT/'artifacts/pixel_language_diversity_training_v2'
            prepare(args.output.resolve(), training.resolve(), args.updates or [0, 4000])
        else:
            if args.training_run is not None or args.updates is not None:
                parser.error('--training-run and --updates apply to preparation; use a new preparation to change states')
            run(args.output.resolve())
    except RuntimeError as error:
        parser.exit(2, str(error)+'\n')
