#!/usr/bin/env python3
"""Join original training-target coverage to saved first-form diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OWN = 'scripts/audit_glyph_target_coverage.py'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main(output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only metadata audit')
    require(not output.exists(), 'preserve earlier audits')
    require(subprocess.check_output(['git', 'show', 'HEAD:' + OWN], cwd=ROOT)
            == (ROOT / OWN).read_bytes(), 'commit coverage audit first')
    run = ROOT / 'artifacts/semantic_glyph_language_v1'
    preparation = load(run / 'preparation.json')
    inventory_path = ROOT / preparation['primitive'] / 'inventory.json'
    require(sha(inventory_path) == preparation['bindings_sha256'][str(inventory_path.relative_to(ROOT))],
            'bound inventory changed')
    inventory = load(inventory_path)['records']
    lookup = {tuple(row['chars']): i for i, row in enumerate(inventory)}
    all_counts = Counter(label for job in preparation['jobs'] for label in job['labels'])
    first_counts = Counter(job['labels'][0] for job in preparation['jobs'])
    geometry_path = ROOT / 'references/evidence/semantic_glyph_geometry/report.json'
    geometry = load(geometry_path)
    require(geometry['checkpoint_sha256'] == load(run / 'fit/report.json')['checkpoint_sha256'],
            'geometry and training checkpoint differ')
    records = []
    for row in geometry['records']:
        if row['split'] != 'development' or row['route'] != 'normal':
            continue
        forms = []
        for value in row['reference_first_forms']:
            label = lookup[tuple(value['form'])]
            forms.append(dict(form_index=label, forms=value['form'],
                training_positive_occurrences=all_counts[label], training_initial_occurrences=first_counts[label],
                original_readout_rank=value['rank'], original_pixel_f1=value['pixel_f1'],
                original_pixel_exact=value['pixel_exact']))
        require(forms, 'retain unsupported cases explicitly rather than silently discard')
        initial = any(r['training_initial_occurrences'] > 0 for r in forms)
        positive = any(r['training_positive_occurrences'] > 0 for r in forms)
        group = ('reference_form_seen_initially' if initial else 'reference_form_seen_only_later'
                 if positive else 'no_reference_form_seen_as_language_target')
        records.append(dict(index=row['index'], group=group, allowed_first_forms=forms,
            best_reference_rank=min(r['original_readout_rank'] for r in forms),
            best_first_pixel_f1=max(r['original_pixel_f1'] for r in forms),
            exact_first_image=any(r['original_pixel_exact'] for r in forms)))
    require(len(records) == 16 and len({r['index'] for r in records}) == 16,
            'all sixteen original normal cases are required')
    summary = {}
    for group in ['reference_form_seen_initially', 'reference_form_seen_only_later',
                  'no_reference_form_seen_as_language_target']:
        rows = [r for r in records if r['group'] == group]
        summary[group] = dict(cases=len(rows), exact_first_images=sum(r['exact_first_image'] for r in rows),
            first_pixel_f1_at_least095=sum(r['best_first_pixel_f1'] >= .95 for r in rows),
            reference_nearest_in_code_cosine=sum(r['best_reference_rank'] == 1 for r in rows))
    require(sum(r['exact_first_image'] for r in records) == geometry['summary']['normal_first_image_exact']
            and sum(r['best_reference_rank'] == 1 for r in records) == geometry['summary']['normal_top_reference_form'],
            'joined diagnostics changed the original counts')
    output.mkdir(parents=True)
    report = dict(source_sha256=sha(ROOT / OWN),
        language_preparation_sha256=sha(run / 'preparation.json'), inventory_sha256=sha(inventory_path),
        original_geometry_report_sha256=sha(geometry_path), training_questions=len(preparation['jobs']),
        positive_glyph_events=sum(all_counts.values()), positive_target_forms=len(all_counts),
        initial_target_forms=len(first_counts), primitive_forms=len(inventory),
        records=records, summary=summary, model_forward_calls=0, optimizer_updates=0,
        source_vocabulary_loaded=False, CUDA_calls=0, new_image_outputs=0,
        evaluator_only_reference_join=True, development_sealed=False,
        interpretation=('Groups use any allowed reference first form. Training occurrence is not proof '
                        'of learned context-to-image transfer. These small reused-case strata do not '
                        'establish a causal effect of coverage or complete-answer accuracy.'))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/glyph_target_coverage_v1')
    main(parser.parse_args().output.resolve())
