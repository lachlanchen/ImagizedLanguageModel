#!/usr/bin/env python3
"""Export completed loss/blank measurements and deterministic original pixels."""
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/joint_context_learning_v1'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    began = time.perf_counter()
    def guard():
        mem = {x.split(':')[0]: int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] >= 48 * 1024**2 and time.perf_counter() - began < 180
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**2
    guard()
    files = [ROOT / 'references/evidence' / name for name in [
        'joint_context_physical_result_20261114.json', 'joint_context_blank_modes_result_20261114.json',
        'global_glyph_ranking_preflight_result_20261114.json']]
    physical, probability, preflight = map(load, files)
    assert not physical['physical_gate_passed'] and physical['new_autonomous_streams'] == 0
    assert probability['certified_blank_global_modes'] == 108 and probability['false_blanks'] == 711
    assert not probability['decision']['search_comparison_eligible']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for name, digest in physical['input_sha256'].items():
        path = RUN / name
        assert sha(path) == digest, 'Preserve completed physical evidence: ' + name
        sources[str(path.relative_to(ROOT))] = digest
    report = load(RUN / 'blank_modes/report.json')
    sources[str((RUN / 'blank_modes/report.json').relative_to(ROOT))] = sha(RUN / 'blank_modes/report.json')
    qual = {r['job']['id']: r for r in load(RUN / 'qualify/report.json')['records']}
    jobs = {r['id']: r for r in load(RUN / 'registration.json')['qualification_jobs']}
    diagnostics = physical['complete_supplied_history_pixel_diagnostics']
    groups = []
    labels = ['Dev books\nfixed', 'Train books\nfixed', 'Train books\nproportional']
    keys = ['development/book/fixed_cells', 'train/book/fixed_cells', 'train/book/proportional']
    for key, label in zip(keys, labels):
        row = next(r for r in diagnostics if '/'.join(r[k] for k in ['split', 'kind', 'layout']) == key)
        decision = probability['decision']['groups'][key]
        groups.append(dict(key=key, label=label, positions=int(row['positions']), source_white=int(row['source_blank']),
            control_nll=row['control_source_pattern_nll_mean'], context_nll=row['candidate_source_pattern_nll_mean'],
            control_false_blank=int(row['control_false_blank']), context_false_blank=int(row['candidate_false_blank']),
            source_preferred=decision['source_preferred'], required=decision['required'],
            certified_blank_modes=probability['groups'][key]['certified_blank_global_mode_false_blank']))
    assert sum(g['control_false_blank'] for g in groups) == 324
    assert sum(g['context_false_blank'] for g in groups) == 689
    sample_positions = []
    for row in sorted(report['cases'], key=lambda x: x['case']):
        guard(); job = jobs[row['case']]
        if (job['split'], job['kind'], job['layout']) != ('development', 'book', 'fixed_cells'):
            continue
        path = ROOT / row['file']; assert sha(path) == row['sha256']
        sources[row['file']] = row['sha256']
        with np.load(path, allow_pickle=False) as a:
            selected = a['emitted_blank'] & ~a['source_blank'] & (a['nonempty_upper_bound'] < a['blank_logp'] - .001)
            sample_positions.extend((row['case'], int(i)) for i in np.flatnonzero(selected))
    assert len(sample_positions) == 21
    samples = []; destination = HERE / 'figures/context_failure_samples'
    destination.mkdir(exist_ok=True)
    for index, (case, position) in enumerate(sample_positions[:6]):
        guard(); path = RUN / 'qualify/cases' / case / 'pixels.npz'
        assert sha(path) == qual[case]['pixels_sha256']; sources[str(path.relative_to(ROOT))] = sha(path)
        images = {}
        with np.load(path, allow_pickle=False) as a:
            for role, field in [('source', 'gold_ink'), ('head', 'parent_ink'), ('context', 'candidate_ink')]:
                pixels = a[field][position, 0]
                assert pixels.shape == (32, 32) and np.isin(pixels, [0, 1]).all()
                image = Image.fromarray(255 - 255 * pixels).resize((96, 96), Image.Resampling.NEAREST)
                p = destination / f'{index+1:02}_{role}.png'; image.save(p)
                images[role] = dict(file=str(p.relative_to(ROOT)), sha256=sha(p),
                    original_ink_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), ink_pixels=int(pixels.sum()))
        assert images['source']['ink_pixels'] > 0 and images['context']['ink_pixels'] == 0
        samples.append(dict(case=case, position=position, images=images))
    value = dict(schema='context-failure-native-figure-data-v1', source_sha256=sources, book_groups=groups,
        original_physical_groups=physical['groups'], samples=samples,
        sample_rule='First six certified empty-mode development-book false blanks, sorted by original case and zero-based position; all21 eligible retained in probe.',
        false_blanks=711, source_preferred_false_blanks=57, certified_blank_global_modes=108,
        book_control_false_blanks=324, book_context_false_blanks=689,
        review_blank_flag_disagreements=28, candidate_credited_empty=4, control_credited_empty=2,
        physical_gate_passed=False, original_readings_and_scores_changed=False,
        ranking_preflight_evidence_sha256=sha(files[2]), ranking_quality='unmeasured', complete_language='unmeasured',
        new_model_calls=0, new_optimizer_updates=0, elapsed_seconds=time.perf_counter() - began)
    output = HERE / 'evidence/context_failure_figure_data.json'
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(book_groups=len(groups), samples=len(samples), image_count=18,
                         elapsed_seconds=value['elapsed_seconds'], no_new_neural_calls=True)))


if __name__ == '__main__':
    main()
