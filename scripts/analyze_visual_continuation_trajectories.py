#!/usr/bin/env python3
"""CPU-only measurements of the completed visual continuation trajectories."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
OUT = ROOT / 'artifacts/visual_continuation_trajectories_v1'
PUBLIC = ROOT / 'references/evidence/visual_continuation_trajectory_registration_20261113.json'
SOURCES = ('scripts/analyze_visual_continuation_trajectories.py',
           'tests/test_continuation_trajectory_statistics.py',
           'references/visual_continuation_trajectory_analysis_plan.md')


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def constant_optimum(sequences):
    """Exact context-independent optimum of sequence-balanced ink/white BCE."""
    a = b = None
    count = 0
    for ink in sequences:
        flat = np.asarray(ink, dtype=np.float64).reshape(len(ink), -1)
        require(len(flat) > 0 and np.isin(flat, [0, 1]).all(), 'Binary nonempty targets required')
        c = flat.sum(0)
        black = c.sum()
        white = flat.size - black
        if a is None:
            a, b = np.zeros_like(c), np.zeros_like(c)
        require(c.shape == a.shape, 'Pixel coordinates differ')
        a += c / max(black, 1)
        b += (len(flat) - c) / max(white, 1)
        count += 1
    require(count > 0 and np.all(a + b > 0), 'No valid target population')
    return a / (a + b), a / count, b / count


def quantile(histogram, q):
    require(0 < q <= 1 and histogram.sum() > 0, 'Invalid empirical quantile')
    return int(np.searchsorted(np.cumsum(histogram), math.ceil(q * int(histogram.sum()))))


def describe(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x))) if len(x) else None


def relative_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max((np.linalg.norm(a) + np.linalg.norm(b)) / 2, 1e-12))


def windows(x, length):
    return dict(all=describe(x), first8=describe(x[:8]), last8=describe(x[-8:]),
                long_first8=describe(x[:8]) if length >= 32 else None,
                long_last8=describe(x[-8:]) if length >= 32 else None)


class Guard:
    def __init__(self):
        self.began = time.perf_counter()
        self.minimum = float('inf')

    def check(self):
        mem = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
        available = mem['MemAvailable'] / 1024**2
        self.minimum = min(self.minimum, available)
        require(available >= 24, 'Host memory reserve unavailable')
        require(time.perf_counter() - self.began <= 600, 'CPU analysis wall ceiling exceeded')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2 <= 8, 'CPU RSS ceiling exceeded')


def register():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA')
    require(not OUT.exists() and not PUBLIC.exists(), 'Preserve prior analysis')
    review = RUN / 'review'
    require(load(review / 'protocol_pilot_audit.json')['registered_protocol_pilot_passed'] is False,
            'This analysis follows the failed complete trial')
    files = ['preparation.json', 'training_manifest.json', 'fit/report.json',
             'evaluation/report.json', 'parent_long_books/report.json',
             'review/mapping.json', 'review/frozen_readings.json', 'review/summary.json',
             'review/protocol_pilot_audit.json', 'review/reviewer_pre_unmask_freeze.json']
    bindings = {str((RUN / f).relative_to(ROOT)): sha(RUN / f) for f in files}
    corpus = ROOT / 'data/visual_sequence_corpus_v1/report.json'
    bindings[str(corpus.relative_to(ROOT))] = sha(corpus)
    require(sha(corpus) == load(RUN / 'preparation.json')['corpus_report_sha256'], 'Corpus changed')
    import torch
    value = dict(schema='visual-continuation-trajectory-registration-v1',
        runtime=dict(python=sys.version.split()[0], numpy=np.__version__, torch=torch.__version__),
        source_sha256={n: sha(ROOT / n) for n in SOURCES}, bindings_sha256=bindings,
        retrospective_after_frozen_language_scoring=True, new_statistics_computed=False,
        all_saved_streams=732, complete_training_targets=13766,
        windows=[8, 32], clean_density_quantile=.99, binary_threshold=.5,
        cpu_threads=2, maximum_seconds=600, maximum_peak_rss_gib=8,
        minimum_available_host_gib=24, new_model_calls=0, optimizer_updates=0,
        original_scores_or_readings_modified=False, goal_achieved=False)
    OUT.mkdir(parents=True)
    save_new(OUT / 'registration.json', value)
    save_new(PUBLIC, value)


def training_population(guard):
    preparation = load(RUN / 'preparation.json')
    rows = load(RUN / 'training_manifest.json')['rows']
    report = load(ROOT / 'data/visual_sequence_corpus_v1/report.json')
    selected = {r['corpus_index']: r for r in rows}
    histories = {k: np.zeros(1025, np.int64) for k in ['qa', 'book']}
    population = dict(records=0, target_glyphs=0)

    def sequences():
        offset = 0
        for shard in [s for s in report['shards'] if s['split'] == 'train']:
            name, = [n for n in shard['files'] if n.endswith('_pixels.npz')]
            path = ROOT / 'data/visual_sequence_corpus_v1' / name
            require(sha(path) == shard['files'][name], 'Training pixels changed')
            with np.load(path, allow_pickle=False) as z:
                bits, bounds = z['ink'], z['record_glyph_offsets']
                for local in range(shard['records']):
                    row = selected.get(offset + local)
                    if row is None:
                        continue
                    guard.check()
                    first, last = map(int, bounds[local:local + 2])
                    ink = np.unpackbits(bits[first + row['cut']:last], axis=1, bitorder='big')
                    require(len(ink) == row['target_glyphs'], 'Complete target coverage differs')
                    histories[row['kind']] += np.bincount(ink.sum(1).astype(np.int64), minlength=1025)
                    population['records'] += 1
                    population['target_glyphs'] += len(ink)
                    yield ink
            offset += shard['records']
    q, a, b = constant_optimum(sequences())
    require(population['records'] == 13766 and population['target_glyphs']
            == preparation['target_positions_per_epoch'] == 2036247, 'Target population incomplete')
    return q.reshape(32, 32), a, b, histories, population


def trace_rows(guard, q, histograms):
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'No CUDA state may be loaded')
    mapping = load(RUN / 'review/mapping.json')['rows']
    readings = load(RUN / 'review/frozen_readings.json')['readings']
    opaque = {(x['arm'], x['step'], x['case'], x['layout']): k for k, x in mapping.items()}
    result, contrasts = [], []
    firsts = {}
    template = q.reshape(-1) >= .5
    for folder in ['evaluation', 'parent_long_books']:
        report = load(RUN / folder / 'report.json')
        for arm in report['arms']:
            for rec in arm['records']:
                guard.check()
                row = rec['input']
                cap = rec['maximum_glyphs']
                directory = RUN / folder / ('long640' if folder == 'parent_long_books'
                    else f"step_{arm['step']:06}/" + ('core128' if cap == 128 else 'long640')) / row['id']
                require(sha(directory / 'output.pt') == rec['tensor_sha256']
                        and sha(directory / 'ink.npz') == rec['raw_sha256'], 'Saved trajectory changed')
                state = torch.load(directory / 'output.pt', map_location='cpu', weights_only=True)
                with np.load(directory / 'ink.npz', allow_pickle=False) as z:
                    ink = z['ink'].reshape(-1, 1024).copy()
                n = len(ink)
                h = state['states'].numpy().astype(np.float64)
                require(n == rec['emitted_glyphs'] and len(h) == n + 1
                        and np.isfinite(h).all(), 'Trace coverage/nonfinite failure')
                require(torch.equal(state['images'], 1 - 2 * torch.from_numpy(ink.reshape(-1, 1, 32, 32)).float()),
                        'Raster differs from actual emitted pixels')
                kind = 'book' if '__book_' in row['id'] else 'qa'
                identity = (arm['arm'], arm['step'], row['original_case'], row['layout'])
                key = opaque[identity]
                require(mapping[key]['image_sha256'] == rec['image_sha256'], 'Opaque trace identity differs')
                density = ink.mean(1)
                adjacent = (ink[1:] != ink[:-1]).mean(1) if n > 1 else np.empty(0)
                repeats = np.all(ink[1:] == ink[:-1], axis=1) if n > 1 else np.empty(0)
                intersection = (ink * template).sum(1)
                denominator = ink.sum(1) + template.sum()
                f1 = np.divide(2 * intersection, denominator, out=np.ones(n), where=denominator > 0)
                dh = h[1:] - h[:-1]
                norms = np.linalg.norm(h, axis=1)
                relative = np.linalg.norm(dh, axis=1) / np.maximum((norms[1:] + norms[:-1]) / 2, 1e-12)
                cosine = 1 - (h[1:] * h[:-1]).sum(1) / np.maximum(norms[1:] * norms[:-1], 1e-12)
                value = dict(opaque=key, arm=arm['arm'], step=arm['step'], layout=row['layout'],
                    case=row['original_case'], kind=kind, condition=row['condition'],
                    glyphs=n, stopped=rec['stopped'], uncertain=readings[key]['uncertain'],
                    text_present=bool(readings[key]['text']),
                    ink_fraction=windows(density, n), adjacent_hamming=windows(adjacent, n),
                    adjacent_exact_repeat=windows(repeats, n), null_template_f1=windows(f1, n),
                    state_relative_l2=windows(relative, n), state_cosine_distance=windows(cosine, n),
                    above_clean99_count=int((ink.sum(1) > quantile(histograms[kind], .99)).sum()),
                    tensor_sha256=rec['tensor_sha256'], raw_sha256=rec['raw_sha256'])
                result.append(value)
                if arm['arm'] == 'candidate_core128' and kind == 'qa':
                    firsts[arm['step'], row['layout'], row['original_case']] = (
                        h[0].copy(), ink[0].copy() if n else None, state['stop_probabilities'][0])
                del state, h
    for (step, layout, case), value in sorted(firsts.items()):
        if not case.endswith('_normal'):
            continue
        targets = [('question_blank', case[:-6] + 'question_blank'),
                   ('passage_blank', case[:-6] + 'passage_blank')]
        if '_a_normal' in case:
            targets.append(('paired_question', case.replace('_a_normal', '_b_normal')))
        for contrast, target in targets:
            other = firsts.get((step, layout, target))
            if other is None:
                continue
            contrasts.append(dict(step=step, layout=layout, case=case, contrast=contrast,
                state_relative_l2=relative_l2(value[0], other[0]),
                states_exact=bool(np.array_equal(value[0], other[0])),
                first_image_hamming=float((value[1] != other[1]).mean())
                    if value[1] is not None and other[1] is not None else None,
                first_stop_absolute_difference=abs(float(value[2]) - float(other[2]))))
    require(len(result) == len({x['opaque'] for x in result}) == 732, 'Keep all saved streams')
    require(len(contrasts) == 416, 'Keep all160 question pairs and256 erasure contrasts')
    return result, contrasts


def execute():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Hide CUDA')
    contract = load(OUT / 'registration.json')
    import torch
    require(contract['runtime'] == dict(python=sys.version.split()[0], numpy=np.__version__,
            torch=torch.__version__), 'Registered CPU runtime changed')
    require(PUBLIC.read_bytes() == (OUT / 'registration.json').read_bytes(), 'Registration differs')
    for name, digest in {**contract['source_sha256'], **contract['bindings_sha256']}.items():
        require(sha(ROOT / name) == digest, 'Registered source/data changed: ' + name)
    for name in [*SOURCES, str(PUBLIC.relative_to(ROOT))]:
        require((ROOT / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=ROOT),
                'Commit analysis and registration before execution')
    require(not any((OUT / n).exists() for n in ['report.json', 'records.json', 'null_template.npz']),
            'Preserve completed or partial analysis outputs')
    guard = Guard()
    guard.check()
    q, a, b, hist, population = training_population(guard)
    records, contrasts = trace_rows(guard, q, hist)
    guard.check()
    np.savez_compressed(OUT / 'null_template.npz', probability=q, black_weight=a, white_weight=b,
                        qa_ink_histogram=hist['qa'], book_ink_histogram=hist['book'])
    save_new(OUT / 'records.json', dict(records=records, first_step_contrasts=contrasts))
    groups = defaultdict(list)
    for row in records:
        groups[row['arm'], row['step'], row['layout'], row['kind'], row['condition']].append(row)
    aggregate = []
    for key, rows in sorted(groups.items()):
        metrics = {}
        for metric in ['ink_fraction', 'adjacent_hamming', 'adjacent_exact_repeat',
                       'null_template_f1', 'state_relative_l2', 'state_cosine_distance']:
            metrics[metric] = {w: describe([r[metric][w]['mean'] for r in rows if r[metric][w] is not None])
                               for w in ['all', 'first8', 'last8', 'long_first8', 'long_last8']}
        aggregate.append(dict(zip(['arm', 'step', 'layout', 'kind', 'condition'], key),
            streams=len(rows), glyphs=sum(r['glyphs'] for r in rows),
            streams_at_least32=sum(r['glyphs'] >= 32 for r in rows),
            above_clean99_count=sum(r['above_clean99_count'] for r in rows), metrics=metrics))
    save_new(OUT / 'report.json', dict(schema='visual-continuation-trajectory-analysis-v1',
        registration_sha256=sha(PUBLIC), records_sha256=sha(OUT / 'records.json'),
        template_sha256=sha(OUT / 'null_template.npz'), population=population,
        clean_ink_count_quantiles={k: {str(q): quantile(v, q) for q in [.5, .95, .99]}
                                  for k, v in hist.items()},
        null_template_ink_fraction=float((q >= .5).mean()), aggregate=aggregate,
        first_step_contrast_count=len(contrasts),
        elapsed_seconds=time.perf_counter() - guard.began,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        minimum_available_host_gib=guard.minimum, new_model_calls=0, optimizer_updates=0,
        original_language_scores_unchanged=True, semantic_cause_identified=False,
        stable_attractor_proven=False, goal_achieved=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['register', 'run'])
    a = p.parse_args()
    if a.action == 'register':
        register()
    else:
        began = time.perf_counter()
        attempt = OUT / ('attempt_' + str(time.time_ns()))
        save_new(attempt.with_suffix('.start.json'), dict(pid=os.getpid(),
            began_unix=time.time(), registration_sha256=sha(PUBLIC)))
        try:
            execute()
        except BaseException as error:
            save_new(attempt.with_suffix('.failed.json'), dict(error=repr(error),
                elapsed_seconds=time.perf_counter() - began, new_model_calls=0))
            raise
        else:
            save_new(attempt.with_suffix('.complete.json'), dict(
                elapsed_seconds=time.perf_counter() - began, report_sha256=sha(OUT / 'report.json'),
                new_model_calls=0))
