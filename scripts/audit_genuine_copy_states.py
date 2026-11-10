#!/usr/bin/env python3
"""Offline geometry of completed genuine-copy states; no model inference.

Gallery labels are evaluator-only. Retrieval is never an ILM output or a
runtime repair. Both fixed geometries are reported, without selecting one.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/audit_genuine_copy_states.py'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def read_cache(run):
    meta = load(run / 'preparation.json')
    require(load(run / 'cache/status.json')['phase'] == 'completed',
            'wait for the complete cache; do not inspect a partial pool')
    require(load(run / 'cache/status.json')['pid'] is None,
            'wait for cache cleanup')
    report = load(run / 'cache/report.json')
    require(report['batch_single_maximum_error'] < meta['maximum_batch_single_state_error'],
            'batched reader equivalence failed')
    rows, tensors = [], []
    for item in report['files']:
        path = run / 'cache' / item['file']
        require(sha(path) == item['sha256'], 'cache shard changed')
        value = torch.load(path, map_location='cpu', weights_only=True)
        indices = value['indices']
        require(indices == item['indices'], 'shard indices changed')
        require(value['states'].shape == (len(indices), 4096), 'state geometry differs')
        require(value['states'].dtype == torch.float32, 'expected original FP32 states')
        require(bool(torch.isfinite(value['states']).all()), 'nonfinite cached state')
        for i, index in enumerate(indices):
            row = meta['records'][index]
            require(row['index'] == index and row['label'] == int(value['labels'][i])
                    and row['split'] == value['split'][i], 'cache metadata mismatch')
            rows.append(row)
        tensors.append(value['states'])
    require([r['index'] for r in rows] == list(range(len(meta['records']))),
            'cache must cover every original record once and in order')
    return meta, report, rows, torch.cat(tensors)


def evaluate(query, labels, gallery, gallery_labels, center):
    """Compare each known form against all gallery forms, merging two views."""
    forms = torch.unique(gallery_labels, sorted=True)
    views = len(gallery_labels) // len(forms)
    require(views in (1, 2) and len(gallery_labels) == views * len(forms),
            'unexpected number of training templates')
    require(torch.equal(gallery_labels, forms.repeat(views)), 'gallery order differs')
    expected = torch.searchsorted(forms, labels)
    require(torch.equal(forms[expected], labels), 'query identity absent from gallery')
    # All transforms are fixed or use training-gallery statistics only.
    g = gallery - center
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-12)
    q = query - center
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    records = []
    for start in range(0, len(q), 32):
        score = (q[start:start + 32] @ g.T).reshape(-1, views, len(forms)).amax(1)
        for i, scores in enumerate(score):
            j = start + i
            target = int(expected[j])
            top = scores.topk(5).indices
            winner = int(top[0])
            true_score = float(scores[target])
            records.append(dict(
                index=j, target_form=int(labels[j]), nearest_form=int(forms[winner]),
                top5_forms=forms[top].tolist(), identity_top1=winner == target,
                identity_top5=bool((top == target).any()),
                rank_optimistic=1 + int((scores > scores[target]).sum()),
                rank_pessimistic=int((scores >= scores[target]).sum()),
                target_cosine=true_score, maximum_cosine=float(scores[winner]),
                target_minus_winner_cosine=true_score - float(scores[winner])))
    n = len(records)
    return dict(cases=n, gallery_forms=len(forms), gallery_views=views,
                identity_top1=sum(r['identity_top1'] for r in records),
                identity_top5=sum(r['identity_top5'] for r in records),
                mean_optimistic_identity_rank=sum(r['rank_optimistic'] for r in records) / n,
                mean_pessimistic_identity_rank=sum(r['rank_pessimistic'] for r in records) / n), records


def main(run, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only audit requires CUDA hidden')
    require(not output.exists(), 'preserve previous audits')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + OWN], cwd=ROOT)
            == (ROOT / OWN).read_bytes(), 'commit audit code before execution')
    torch.set_num_threads(4)
    began = time.monotonic()
    meta, report, rows, states = read_cache(run)
    by_template = []
    for template in range(3):
        ids = [i for i, r in enumerate(rows) if r['template'] == template]
        by_template.append((states[ids], torch.tensor([rows[i]['label'] for i in ids])))
    h0, y0 = by_template[0]
    h1, y1 = by_template[1]
    held, yheld = by_template[2]
    require(torch.equal(y0, y1) and len(y0) == meta['unique_observable_forms']
            and len(yheld) == meta['held_template_cases'],
            'declared copy curriculum changed')
    selected = torch.searchsorted(y1, yheld)
    require(torch.equal(y1[selected], yheld), 'held identities differ')
    routes = {
        'training_wording1_against_wording0': (h1[selected], yheld, h0, y0),
        'held_wording2_against_both_training_wordings':
            (held, yheld, torch.cat((h0, h1)), torch.cat((y0, y1))),
    }
    summaries, records = {}, []
    for route, (query, labels, gallery, gallery_labels) in routes.items():
        for geometry, center in [('raw_cosine', torch.zeros(4096)),
                                 ('training_gallery_centered_cosine', gallery.mean(0))]:
            require(time.monotonic() - began < 120, 'CPU audit exceeded 120-second budget')
            key = route + '/' + geometry
            summary, actual = evaluate(query, labels, gallery, gallery_labels, center)
            require(time.monotonic() - began < 120, 'CPU audit exceeded 120-second budget')
            summaries[key] = summary
            records.extend(dict(route=route, geometry=geometry, **r) for r in actual)
    output.mkdir(parents=True)
    (output / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    receipt = dict(
        source_sha256=sha(ROOT / OWN),
        preparation_sha256=sha(run / 'preparation.json'),
        cache_report_sha256=sha(run / 'cache/report.json'),
        batch_single_maximum_error=report['batch_single_maximum_error'],
        summaries=summaries, records_sha256=sha(output / 'records.json'),
        elapsed_seconds=time.monotonic() - began, CPU_threads=4,
        model_forward_calls=0, optimizer_updates=0, CUDA_calls=0,
        labels_used_only_for_offline_evaluation=True,
        geometry_selected_after_results=False, held_statistics_used_for_centering=False,
        runtime_modified=False, emitted_images=0, language_accuracy_assessed=False,
        held_forms_seen_in_training=True,
        interpretation=('Success shows identity recoverable by this fixed offline similarity rule; '
                        'failure does not prove the states lack nonlinear identity information. '
                        'Neither is generated-image or language accuracy.'))
    (output / 'report.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'artifacts/genuine_glyph_readout_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.run.resolve(), args.output.resolve())
