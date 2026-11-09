#!/usr/bin/env python3
"""CPU audit of complete image outputs, actual feedback and training exposure."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'hide CUDA for independent CPU audit')
    import torch
    from ilm.visual_lm.codec_block_writer import CodecBlockWriter, load_qualified_writing_codec
    from scripts.preflight_native_pixel_actor import CODEC
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    meta = load(output/'preparation.json')
    execution = output/'execution'
    status, report = load(execution/'status.json'), load(execution/'report.json')
    require(status['phase'] == 'completed' and status['pid'] is None, 'actual model process must finish first')
    for name, digest in {**meta['code_sha256'], **meta['bound_sha256']}.items():
        require(sha(root/name) == digest, 'bound source changed: ' + name)
    corpus = root/meta['corpus']
    for name, digest in meta['corpus_sha256'].items():
        require(sha(corpus/name) == digest, 'prepared physical image changed: ' + name)
    manifest = load(corpus/'manifest.json')
    require(sha(execution/'checkpoint_latest.pt') == report['checkpoint_sha256'], 'final checkpoint changed')
    checkpoint = torch.load(execution/'checkpoint_latest.pt', map_location='cpu', weights_only=True)
    require(checkpoint['training_complete'] and checkpoint['update'] == report['updates'] == meta['maximum_updates'], 'fixed update budget not completed')
    require(checkpoint['preparation_sha256'] == sha(output/'preparation.json'), 'checkpoint source identity differs')
    history = rows(execution/'history.jsonl')
    require(len(history) == report['updates'] and all(r['update'] == i+1 for i, r in enumerate(history)), 'retained learning history differs')
    # Independent regeneration of the declared schedule, without the learner sampler.
    generator = torch.Generator().manual_seed(meta['seed']+1)
    expected = []
    while len(expected) < len(history):
        order = torch.randperm(512, generator=generator)
        for start in range(0, 512, 32):
            window = order[start:start+32]
            for repeat in range(4):
                local = torch.randperm(32, generator=generator)
                for offset in range(0, 32, 4):
                    expected.append(window[local[offset:offset+4]].tolist())
        # The final outer traversal may be incomplete.
    expected = expected[:len(history)]
    require(all(r['indices'] == indices for r, indices in zip(history, expected)), 'actual owning image draws differ from registered sampling')
    counts = np.bincount([i for indices in expected for i in indices], minlength=512)
    require(int(counts.sum()) == report['retained_draws'] == 64000, 'actual retained exposure differs')
    calls = rows(execution/'optimizer_calls.jsonl')
    require(len(calls) == report['actual_optimizer_calls'] and all(r['call'] == i+1 for i, r in enumerate(calls)), 'executed optimizer ledger differs')
    require(all(1 <= r['update'] <= len(history) and r['indices'] == expected[r['update']-1] for r in calls), 'an executed call used another owning batch')
    qualified = load(output/'qualification/report.json')
    require(all(qualified['checks'].values()) and qualified['actual_optimizer_calls'] == 3
        and qualified['second'] == qualified['repeated'], 'actual qualification replay differs')
    require(load(execution/'cold_start_replay.json')['first_two_updates_bit_exact'], 'fresh main initialization was not replayed')
    codec = CodecBlockWriter(load_qualified_writing_codec(CODEC, torch.device('cpu')))
    feedback_count, max_error, scores = 0, 0., []
    for step in meta['assessment_updates']:
        parent = execution/f'assessment_{step:05}'
        producer = load(parent/'report.json')
        monitor = load(parent/'teacher_history.json')
        require(monitor['questions'] == 512 and monitor['model_sha256'] == producer['model_sha256'], 'correct-history assessment state differs')
        jobs = [(False, i, 'normal') for i in range(512 if step == meta['maximum_updates'] else 64)]
        jobs += [(True, i, route) for i in range(24) for route in ('normal', 'blank')]
        require(len(producer['cases']) == len(jobs), 'some autonomous outputs were omitted')
        for position, (dev, index, route) in enumerate(jobs):
            key = ('development' if dev else 'train') + f'_{index:04}_{route}'
            folder = parent/key
            receipt = load(folder/'receipt.json')
            identity = receipt['identity']
            require(identity == dict(index=index, development=dev, route=route, update=step,
                model_sha256=producer['model_sha256'], preparation_sha256=sha(output/'preparation.json'), actual_PNG_prefill=True), 'generation source identity differs')
            require(all(sha(folder/n) == digest for n, digest in receipt['hashes'].items()), 'actual output changed')
            result = torch.load(folder/'result.pt', map_location='cpu', weights_only=True)
            images = result['images']
            n = len(images)
            require(images.dtype == torch.float32 and images.shape == (n, 1, 32, 64)
                and bool(((images == -1) | (images == 1)).all()), 'unmodified complete signed binary image stream required')
            probabilities = result['stop_probabilities']
            stopped, capped = result['stopped'], result['capped']
            require(type(stopped) is bool and type(capped) is bool and stopped != capped
                and n <= 128 and len(probabilities) == n+1 and all(np.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                and all(p < .5 for p in probabilities[:n]) and (probabilities[-1] >= .5) == stopped
                and (not capped or n == 128), 'learned structural stop trace differs')
            ink = np.concatenate([v[0].numpy() < 0 for v in images], axis=1) if n else np.zeros((32, 0), bool)
            display = (~ink).astype('uint8')*255 if n else np.full((32, 64), 255, 'uint8')
            with Image.open(folder/'answer.png') as image:
                require(np.array_equal(np.asarray(image.convert('L')), display), 'PNG differs from actual emitted pixels')
            row = manifest['development' if dev else 'train'][index]
            refs = row['references'] if dev else [dict(path=row['answer'], sha256=row['answer_sha256'])]
            exact, f1 = False, 0.
            for ref in refs:
                with Image.open(corpus/ref['path']) as image:
                    target = np.asarray(image.convert('L')) == 0
                width = max(target.shape[1], ink.shape[1])
                actual = np.pad(ink, ((0, 0), (0, width-ink.shape[1])))
                desired = np.pad(target, ((0, 0), (0, width-target.shape[1])))
                denominator = int(actual.sum()+desired.sum())
                f1 = max(f1, 2*int((actual & desired).sum())/denominator if denominator else 1.)
                exact |= stopped and not capped and ink.shape == target.shape and np.array_equal(ink, target)
            records = torch.load(folder/'feedback.pt', map_location='cpu', weights_only=True)
            require(len(records) == n == len(result['feedback_sha256']), 'actual emitted feedback count differs')
            for j, record in enumerate(records):
                require(torch.equal(record['images'], images[j:j+1]), 'feedback did not read its actual image')
                require(hashlib.sha256(record['codes'].numpy().tobytes()).hexdigest() == result['feedback_sha256'][j], 'actual image feedback hash differs')
                error = float((codec.encode_targets(record['images'])-record['codes']).abs().max())
                require(error <= 1e-4, 'CPU actual-image re-encoding exceeds declared tolerance')
                feedback_count += 1
                max_error = max(max_error, error)
            score = dict(key=key, index=index, development=dev, route=route, exact=bool(exact), foreground_f1=f1,
                         stopped=stopped, capped=capped, blocks=n)
            require(score == producer['cases'][position], 'independent complete pixel score differs')
            scores.append(dict(update=step, source=row['source'], pair=row['pair'], **score))
    groups, pairs, effects = {}, {}, {}
    for step in meta['assessment_updates']:
        stage = [r for r in scores if r['update'] == step]
        for label, chosen in [('old_controls', [r for r in stage if not r['development'] and r['index'] < 32]),
                              ('new_train', [r for r in stage if not r['development'] and r['index'] >= 32])]:
            groups[f'{step}_{label}'] = summarize(chosen)
        for source in ('cmrc', 'dolly'):
            for route in ('normal', 'blank'):
                chosen = [r for r in stage if r['development'] and r['source'] == source and r['route'] == route]
                groups[f'{step}_{source}_{route}'] = summarize(chosen)
            by = {(r['index'], r['route']): r for r in stage if r['development'] and r['source'] == source}
            ids = sorted({i for i, route in by})
            effects[f'{step}_{source}'] = float(np.mean([by[i, 'normal']['foreground_f1']-by[i, 'blank']['foreground_f1'] for i in ids]))
        for dev in (False, True):
            chosen = [r for r in stage if r['development'] == dev and r['route'] == 'normal' and r['pair'] is not None]
            ids = sorted({r['pair'] for r in chosen})
            pairs[f'{step}_' + ('development' if dev else 'train')] = dict(pairs=len(ids), both_exact=sum(
                all(r['exact'] for r in chosen if r['pair'] == i) for i in ids))
    result = dict(report_sha256=sha(execution/'report.json'), groups=groups, paired_questions=pairs,
        question_erasure_F1_effect=effects, retained_updates=len(history), actual_optimizer_calls=len(calls),
        minimum_exposure=int(counts.min()), maximum_exposure=int(counts.max()), exposure_per_question=counts.tolist(),
        optimizer_seconds_including_discarded_calls=sum(r['seconds'] for r in calls),
        all_output_PNGs_match_emitted_tensors=True, actual_feedback_records_verified=feedback_count,
        feedback_CPU_GPU_max_absolute_error=max_error, complete_cases=scores,
        semantic_accuracy_measured=False, development_sealed=False, general_ilm_goal_complete=False)
    require(not (output/'audit.json').exists(), 'completed audits are immutable')
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('complete_cases', 'exposure_per_question')}, indent=2))


def summarize(records):
    return dict(cases=len(records), exact=sum(r['exact'] for r in records),
                foreground_f1=float(np.mean([r['foreground_f1'] for r in records])),
                stopped=sum(r['stopped'] for r in records), capped=sum(r['capped'] for r in records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args().output.resolve())
