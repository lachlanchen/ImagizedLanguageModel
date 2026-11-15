#!/usr/bin/env python3
"""Export completed ranking measurements and fixed-order original glyphs."""
import fcntl
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/global_glyph_ranking_v1'
RUN = ROOT / 'artifacts/white_target_ranking_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())


def owner():
    h = load(ROOT / 'artifacts/glyph_tail_current_handoff.json')['runtime_authority']
    result = []
    for key, start in [('active_controller', 'active_controller_start_ticks'),
                       ('active_model_worker_pid', 'active_model_worker_start_ticks')]:
        pid = h[key]; parts = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        assert parts[19] == h[start] and parts[0] != 'Z'
        result.append((pid, parts[19]))
    return result


def main():
    began = time.perf_counter(); protected = owner()
    def guard():
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        assert int(mem['MemAvailable'].split()[0]) >= 48 * 1024**2
        assert time.perf_counter() - began < 180 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**2
    guard()
    names = ['global_glyph_ranking_physical_result_20261115.json',
             'ranked_writer_whitespace_result_20261115.json', 'white_target_preflight_result_20261115.json']
    paths = [ROOT / 'references/evidence' / n for n in names]
    expected = ['13475253122e912f61e6f31b93523bce5f3bfc29b34ee531a4c1c28224321443',
                'c9e6ba7cae33988b9d6a527075bd70a98e7b68a5aab1b8ab562c56f2d9adf508',
                '7c8d143910883f8befdd0052cd4c479331a211eb70e87519e150e68c64d7aa70']
    assert [sha(p) for p in paths] == expected
    physical, probability, preflight = map(load, paths)
    assert not physical['physical_gate_passed'] and physical['readable_groups_passed'] and physical['credited_candidate_ink_passed']
    assert physical['totals']['candidate']['false_blanks'] == 37 and physical['totals']['candidate']['false_ink'] == 25
    assert not probability['decision']['separately_registered_decoding_comparison_eligible']
    assert probability['decision']['totals']['fixed_false_ink'] == 3 and probability['decision']['totals']['added_false_blanks'] == 1
    assert preflight['complete'] and preflight['all_updates_reset'] and preflight['actual_optimizer_updates'] == 4
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    report_path = OLD / 'qualify/report.json'; assert sha(report_path) == physical['qualification_report_sha256']
    sources[str(report_path.relative_to(ROOT))] = sha(report_path)
    report = load(report_path); records = report['records']
    groups = []
    for row in physical['readable_groups']:
        if row['role'] != 'candidate': continue
        key = '/'.join(row[k] for k in ['split', 'kind', 'layout'])
        control = next(g for g in physical['readable_groups'] if g['role'] == 'parent'
            and '/'.join(g[k] for k in ['split', 'kind', 'layout']) == key)
        assert row['passed'] and row['count'] == control['count']
        label = ('Dev' if row['split'] == 'development' else 'Train') + ' ' + ('book' if row['kind'] == 'book' else 'QA')
        label += ' / ' + ('fixed' if row['layout'] == 'fixed_cells' else 'proportional')
        groups.append(dict(key=key, label=label, count=row['count'],
            control=control['readable_nonblank'], ranking=row['readable_nonblank']))
    assert len(groups) == 7 and sum(g['ranking'] for g in groups) == 1002 and sum(g['control'] for g in groups) == 981
    samples, failures = [], []
    destination = HERE / 'figures/ranking_completed_samples'; destination.mkdir(exist_ok=False)
    def emit(record, position, index, kind):
        path = OLD / 'qualify/cases' / record['job']['id'] / 'pixels.npz'
        assert sha(path) == record['pixels_sha256']; sources[str(path.relative_to(ROOT))] = sha(path)
        images = {}
        with np.load(path, allow_pickle=False) as arrays:
            for role, field in [('source', 'gold_ink'), ('control', 'parent_ink'), ('ranking', 'candidate_ink')]:
                pixels = arrays[field][position, 0]
                assert pixels.dtype == np.uint8 and pixels.shape == (32, 32) and np.isin(pixels, [0, 1]).all()
                target = destination / f'{kind}_{index+1:02}_{role}.png'
                Image.fromarray(255 - 255 * pixels).resize((96, 96), Image.Resampling.NEAREST).save(target)
                images[role] = dict(file=str(target.relative_to(ROOT)), sha256=sha(target),
                    original_ink_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), ink_pixels=int(pixels.sum()))
        return dict(case=record['job']['id'], position=int(position), images=images)
    for record in records:
        guard(); job = record['job']
        if job['split'] != 'development' or job['kind'] != 'book': continue
        path = OLD / 'qualify/cases' / job['id'] / 'pixels.npz'; assert sha(path) == record['pixels_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            for position in job['positions']:
                if arrays['gold_ink'][position].any() and len(samples) < 6:
                    samples.append(emit(record, position, len(samples), 'glyph'))
        if len(samples) == 6: break
    for record in records:
        guard(); path = OLD / 'qualify/cases' / record['job']['id'] / 'pixels.npz'
        assert sha(path) == record['pixels_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            white = ~arrays['gold_ink'].reshape(record['target_glyphs'], -1).any(1)
            ink = arrays['candidate_ink'].reshape(record['target_glyphs'], -1).any(1)
            for position in np.flatnonzero(white & ink):
                if len(failures) < 3: failures.append(emit(record, int(position), len(failures), 'white_error'))
        if len(failures) == 3: break
    assert len(samples) == 6 and len(failures) == 3
    assert all(s['images']['source']['ink_pixels'] > 0 for s in samples)
    assert all(s['images']['source']['ink_pixels'] == 0 and s['images']['ranking']['ink_pixels'] > 0 for s in failures)
    current = ROOT / 'references/evidence/white_target_ranking_registration_20261115.json'
    registration = load(current)
    assert len(registration['source_sha256']) == 115
    assert all(sha(ROOT / n) == digest for n, digest in registration['source_sha256'].items())
    sources[str(current.relative_to(ROOT))] = sha(current)
    assert owner() == protected; guard()
    value = dict(schema='ranking-completed-native-figure-data-v1', source_sha256=sources,
        exporter_sha256=sha(Path(__file__)), groups=groups, totals=physical['totals'],
        samples=samples, whitespace_failures=failures,
        sample_rule='First six source-ink selected positions in original development-book case/position order, without inspecting candidate success.',
        failure_sample_rule='First three false-ink source-white positions in original complete-case/position order; explicitly failure-selected.',
        physical_review=physical['physical_review'], original_control_frozen_credits=970,
        fresh_control_review_credits=981, control_pixels_unchanged=True,
        probability_totals=probability['decision']['totals'], probability_decoding_eligible=False,
        physical_gate_passed=False, complete_language='unmeasured',
        new_blank_target_fit_quality='unmeasured', new_model_calls=0, new_optimizer_updates=0,
        pixel_expansion='3x nearest-neighbour from original binary32x32 pixels',
        elapsed_seconds=time.perf_counter() - began)
    output = HERE / 'evidence/ranking_completed_figure_data.json'
    with output.open('x') as stream: json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(dict(groups=7, original_glyph_samples=6, whitespace_failure_samples=3,
        image_instances=27, original_model_owners_preserved=True, new_model_calls=0,
        elapsed_seconds=value['elapsed_seconds'])))


if __name__ == '__main__':
    with (ROOT / 'artifacts/publication_cpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
