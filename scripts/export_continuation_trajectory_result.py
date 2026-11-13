#!/usr/bin/env python3
"""Export authenticated saved-trajectory measurements; never invoke a model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/visual_continuation_trajectories_v1'
PUBLIC = ROOT / 'references/evidence/visual_continuation_trajectories_20261113.json'


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    registration = load(RUN / 'registration.json')
    report = load(RUN / 'report.json')
    for name, digest in {**registration['source_sha256'], **registration['bindings_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    assert sha(RUN / 'records.json') == report['records_sha256']
    assert sha(RUN / 'null_template.npz') == report['template_sha256']
    assert sha(RUN / 'registration.json') == report['registration_sha256']
    assert (RUN / 'registration.json').read_bytes() == (ROOT /
        'references/evidence/visual_continuation_trajectory_registration_20261113.json').read_bytes()
    data = load(RUN / 'records.json')
    assert len(data['records']) == len({r['opaque'] for r in data['records']}) == 732
    assert len(data['first_step_contrasts']) == 416
    # Independent aggregation of stored per-stream statistics, including all erasures.
    groups = defaultdict(list)
    for row in data['records']:
        groups[tuple(row[k] for k in ['arm', 'step', 'layout', 'kind', 'condition'])].append(row)
    for group in report['aggregate']:
        rows = groups[tuple(group[k] for k in ['arm', 'step', 'layout', 'kind', 'condition'])]
        assert len(rows) == group['streams']
        assert sum(r['glyphs'] for r in rows) == group['glyphs']
        assert sum(r['above_clean99_count'] for r in rows) == group['above_clean99_count']
        assert sum(r['glyphs'] >= 32 for r in rows) == group['streams_at_least32']
        for metric, windows in group['metrics'].items():
            for window, result in windows.items():
                values = [r[metric][window]['mean'] for r in rows if r[metric][window] is not None]
                if not values:
                    assert result is None
                else:
                    assert result['count'] == len(values)
                    assert abs(result['mean'] - statistics.mean(values)) < 1e-12
                    assert result['median'] == statistics.median(values)
    contrasts = defaultdict(list)
    for row in data['first_step_contrasts']:
        contrasts[row['step'], row['layout'], row['contrast']].append(row)
    summaries = []
    for (step, layout, contrast), rows in sorted(contrasts.items()):
        images = [r['first_image_hamming'] for r in rows if r['first_image_hamming'] is not None]
        summaries.append(dict(step=step, layout=layout, contrast=contrast, count=len(rows),
            exact_first_states=sum(r['states_exact'] for r in rows), emitted_image_pairs=len(images),
            identical_first_images=sum(v == 0 for v in images),
            state_relative_l2_mean=statistics.mean(r['state_relative_l2'] for r in rows),
            first_image_hamming_mean=statistics.mean(images) if images else None,
            first_stop_absolute_difference_mean=statistics.mean(r['first_stop_absolute_difference'] for r in rows)))
    with np.load(RUN / 'null_template.npz', allow_pickle=False) as z:
        q, a, b = z['probability'].reshape(-1), z['black_weight'], z['white_weight']
        constant_loss = float(-.5 * (a * np.log(np.maximum(q, 1e-300)) +
                                    b * np.log(np.maximum(1 - q, 1e-300))).sum())
        template = dict(probability=q.reshape(32, 32).tolist(),
            threshold_ink_count=int((q >= .5).sum()), balanced_bce=constant_loss)
    failure = load(RUN / 'external_termination_attempt001.json')
    completed = list(RUN.glob('*.complete.json'))
    assert len(completed) == 1 and load(completed[0])['report_sha256'] == sha(RUN / 'report.json')
    return dict(schema='visual-continuation-trajectory-public-result-v1',
        report_sha256=sha(RUN / 'report.json'), registration_sha256=sha(RUN / 'registration.json'),
        original_records_sha256=report['records_sha256'], original_template_sha256=report['template_sha256'],
        source_bindings=registration['source_sha256'], input_bindings=registration['bindings_sha256'],
        population=report['population'], saved_streams=732,
        clean_ink_count_quantiles=report['clean_ink_count_quantiles'],
        null_template=template, aggregate=report['aggregate'], first_step_contrasts=summaries,
        successful_attempt=load(completed[0]), kernel_analysis_seconds=report['elapsed_seconds'],
        peak_rss_gib=report['peak_rss_gib'], minimum_available_host_gib=report['minimum_available_host_gib'],
        failed_attempt=failure,
        cost_limit='Successful elapsed and failed-attempt wall upper bound are separate; exact cumulative CPU/wall cost is unknown.',
        observational_only=True, semantic_cause_identified=False, stable_attractor_proven=False,
        new_model_calls=0, optimizer_updates=0, original_language_scores_unchanged=True, goal_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + '\n'
    if args.verify:
        assert PUBLIC.read_text() == payload, 'Published result differs'
    else:
        with PUBLIC.open('x') as f:
            f.write(payload)
    print(json.dumps(dict(verified=True, public_sha256=sha(PUBLIC), model_calls=0)))
