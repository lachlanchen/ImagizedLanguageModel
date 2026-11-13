#!/usr/bin/env python3
"""Recount complete clean-history pixels without neural calls or regrading."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/continuation_clean_history_v1'
PUBLIC = ROOT / 'references/evidence/continuation_clean_history_result_20261113.json'


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    registration = load(RUN / 'registration.json')
    report = load(RUN / 'execution/report.json')
    assert report['complete'] and len(report['records']) == 248
    assert report['registration_sha256'] == sha(RUN / 'registration.json')
    assert sha(RUN / 'targets.pt') == registration['targets_sha256']
    assert all(sha(ROOT / n) == digest for n, digest in registration['bindings_sha256'].items())
    targets = torch.load(RUN / 'targets.pt', weights_only=True, map_location='cpu')
    limits = load(ROOT / 'artifacts/visual_continuation_trajectories_v1/report.json')['clean_ink_count_quantiles']
    records = []; groups = defaultdict(list)
    for rec in report['records']:
        job = rec['job']; path = ROOT / rec['predictions_file']
        assert sha(path) == rec['predictions_sha256']
        value = torch.load(path, weights_only=True, map_location='cpu')
        gold = targets[job['id']].numpy().reshape(-1, 1024)
        ink = value['predicted_ink'].numpy().reshape(-1, 1024)
        stops = value['stop_probabilities'].numpy().reshape(-1)
        assert len(ink) == job['target_glyphs'] == len(gold) and len(stops) == len(gold) + 1
        assert np.isin(ink, [0, 1]).all() and np.isfinite(stops).all()
        tp = (ink * gold).sum(1); denom = ink.sum(1) + gold.sum(1)
        f1 = np.divide(2*tp, denom, out=np.ones(len(ink)), where=denom > 0)
        exact = np.all(ink == gold, axis=1)
        record = dict(job=job, step=rec['step'], pixels_sha256=rec['predictions_sha256'],
            positions=len(ink), exact=int(exact.sum()), first_f1=float(f1[0]),
            later_positions=len(ink)-1, later_exact=int(exact[1:].sum()),
            later_at_least_095=int((f1[1:] >= .95).sum()), f1_mean=float(f1.mean()),
            f1_median=float(np.median(f1)), later_f1_sum=float(f1[1:].sum()),
            ink_fraction=float(ink.mean()), above_clean99=int((ink.sum(1) > limits[job['kind']]['0.99']).sum()),
            premature_stop_positions=int((stops[:-1] >= .5).sum()), terminal_stop=bool(stops[-1] >= .5),
            balanced_image_loss=rec['balanced_image_loss'],
            agreement=rec.get('agreement'), original_first_state_error=rec.get('original_first_state_maximum_error'))
        records.append(record)
        groups[rec['step'], job['split'], job['kind'], job['layout']].append(record)
    aggregate = []
    for key, rows in sorted(groups.items()):
        total = sum(x['positions'] for x in rows); later = sum(x['later_positions'] for x in rows)
        dense = sum(x['above_clean99'] for x in rows)/total
        fraction095 = sum(x['later_at_least_095'] for x in rows)/later if later else None
        aggregate.append(dict(zip(['step', 'split', 'kind', 'layout'], key), cases=len(rows), positions=total,
            first_f1_macro=statistics.mean(x['first_f1'] for x in rows),
            exact=sum(x['exact'] for x in rows), later_positions=later,
            later_exact=sum(x['later_exact'] for x in rows),
            later_f1_position_mean=sum(x['later_f1_sum'] for x in rows)/later if later else None,
            later_fraction_f1_095=fraction095, above_clean99_fraction=dense,
            dense_description=dense > registration['dense_fraction'],
            accurate_later_description=fraction095 >= registration['later_fraction_f1_095'] if later else None,
            ink_fraction_macro=statistics.mean(x['ink_fraction'] for x in rows),
            image_loss_macro=statistics.mean(x['balanced_image_loss'] for x in rows),
            premature_stop_positions=sum(x['premature_stop_positions'] for x in rows),
            terminal_stops=sum(x['terminal_stop'] for x in rows)))
    result = dict(schema='continuation-clean-history-result-v1', registration_sha256=sha(RUN/'registration.json'),
        execution_report_sha256=sha(RUN/'execution/report.json'), summarizer_sha256=sha(Path(__file__)),
        original_source_bindings=registration['source_sha256'], records=records, aggregate=aggregate,
        successful_execution_resources={k:report[k] for k in ['elapsed_seconds', 'resource_wait_seconds',
            'peak_allocated_vram_gib', 'minimum_available_gib']},
        failed_attempt=load(ROOT / 'references/evidence/continuation_clean_history_launch_recovery_20261113.json'),
        supplied_reference_images=True, supplied_target_length=True, stopping_bypassed=True,
        new_optimizer_updates=0, new_autonomous_generations=0, original_language_scores_unchanged=True,
        autonomous_language_evidence=False, goal_achieved=False)
    with PUBLIC.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(dict(public_sha256=sha(PUBLIC), cases=248, new_model_calls=0)))


if __name__ == '__main__':
    main()
