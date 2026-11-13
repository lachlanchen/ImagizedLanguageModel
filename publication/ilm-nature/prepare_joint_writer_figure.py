#!/usr/bin/env python3
"""Export completed physical measurements and every selected blank book output."""
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'artifacts/continuation_joint_writer_v1'


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    began = time.perf_counter()
    def guard():
        memory = {l.split(':')[0]: int(l.split()[1]) for l in Path('/proc/meminfo').read_text().splitlines()}
        assert memory['MemAvailable'] >= 48 * 1024**2, 'Retain48GiB available RAM during CPU figure preparation'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2 * 1024**2
        assert time.perf_counter() - began < 180
    guard()
    physical_path = ROOT / 'references/evidence/joint_writer_physical_result_20261113.json'
    probability_path = ROOT / 'references/evidence/joint_writer_blank_density_result_20261113.json'
    physical, probability = load(physical_path), load(probability_path)
    assert physical['physical_gate_passed'] is False and physical['new_autonomous_streams'] == 0
    assert probability['search_only_hypothesis_merits_separate_registration'] is False
    sources = {str(p.relative_to(ROOT)): sha(p) for p in [physical_path, probability_path]}
    for name in ['qualify/report.json', 'qualify/frozen_readings.json', 'qualify/mapping.json']:
        assert sha(RUN / name) == physical['input_sha256'][name], 'Completed source changed'
        sources[str((RUN / name).relative_to(ROOT))] = sha(RUN / name)
    frozen_bundle = load(RUN / 'qualify/frozen_readings.json')
    frozen = frozen_bundle['readings']
    mapping = load(RUN / 'qualify/mapping.json')['rows']
    qualified = {r['job']['id']: r for r in load(RUN / 'qualify/report.json')['records']}
    candidate = [g for g in physical['groups'] if g['role'] == 'candidate']
    control = {(g['split'], g['kind'], g['layout']): g for g in physical['groups'] if g['role'] == 'parent'}
    labels = ['Dev books\nfixed', 'Dev QA\nfixed', 'Dev QA\nproportional', 'Train books\nfixed',
              'Train books\nproportional', 'Train QA\nfixed', 'Train QA\nproportional']
    groups = []
    for label, c in zip(labels, candidate):
        p = control[c['split'], c['kind'], c['layout']]
        assert c['count'] == p['count'] and abs(c['fraction'] - c['readable_nonblank'] / c['count']) < 1e-15
        groups.append(dict(label=label, key='/'.join([c['split'], c['kind'], c['layout']]),
            count=c['count'], candidate=c['readable_nonblank'], control=p['readable_nonblank'],
            candidate_fraction=c['fraction'], control_fraction=p['fraction'], passed=c['passed']))
    assert len(groups) == 7 and sum(g['candidate'] for g in groups) == 970 and sum(g['control'] for g in groups) == 599
    selected = [(key, row) for key, row in mapping.items()
                if (row['role'], row['split'], row['kind'], row['layout']) == ('candidate', 'development', 'book', 'fixed_cells')]
    assert len(selected) == 108
    arrays = {}; disagreements = []; audit_groups = {}
    for key, row in mapping.items():
        guard(); case = row['case']; path = RUN / 'qualify/cases' / case / 'pixels.npz'
        if case not in arrays:
            assert sha(path) == qualified[case]['pixels_sha256']
            sources[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path, allow_pickle=False) as data:
                arrays[case] = {name: data[name].copy() for name in ['candidate_ink', 'parent_ink', 'gold_ink']}
        ink = arrays[case][row['role'] + '_ink'][row['position'], 0]
        page = RUN / 'qualify/pages' / (key + '.png')
        assert sha(page) == row['image_sha256'] == frozen_bundle['image_sha256'][key]
        with Image.open(page) as image:
            assert np.array_equal(np.asarray(image), 255 - 255 * ink), 'Original reviewed page differs from physical pixels'
        actual_blank = not bool(ink.any())
        reading = frozen[key]; credited = bool(reading['text'].strip()) and not reading['uncertain'] and not reading['blank']
        if actual_blank != reading['blank']:
            group = '/'.join([row['role'], row['split'], row['kind'], row['layout']])
            item = audit_groups.setdefault(group, dict(blank_flag_mismatch=0, credited_empty=0, blank_flag_on_nonempty=0))
            item['blank_flag_mismatch'] += 1; item['credited_empty'] += int(credited and actual_blank)
            item['blank_flag_on_nonempty'] += int(reading['blank'] and not actual_blank)
            disagreements.append(dict(opaque_id=key, role=row['role'], case=case, position=row['position'], group=group,
                actual_blank=actual_blank, credited=credited, original_reading=reading, pixels_sha256=sha(path)))
    candidate_empty = sum(v['actual_blank'] and v['credited'] and v['role'] == 'candidate' for v in disagreements)
    parent_empty = sum(v['actual_blank'] and v['credited'] and v['role'] == 'parent' for v in disagreements)
    assert len(disagreements) == 17 and candidate_empty == 3 and parent_empty == 1
    audit = dict(schema='joint-writer-blank-flag-audit-v1', selected_images=len(mapping), source_sha256=sources,
        groups=audit_groups, disagreements=disagreements, mismatched_blank_flags=len(disagreements),
        credited_empty=candidate_empty + parent_empty, candidate_credited_empty=candidate_empty, parent_credited_empty=parent_empty,
        original_reviewed_pages_match_authenticated_arrays=True,
        original_frozen_readings_changed=False, original_scores_changed=False, original_failed_verdict_changed=False,
        new_model_calls=0, new_optimizer_updates=0, semantic_regrading=False, goal_achieved=False)
    audit_path = ROOT / 'references/evidence/joint_writer_blank_flag_audit_20261114.json'
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    counts = dict(source_blank=0, output_blank=0, both_blank=0, false_blank=0, blank_source_nonblank_output=0)
    samples = []; destination = HERE / 'figures/joint_writer_samples'; destination.mkdir(exist_ok=True)
    for key, row in sorted(selected, key=lambda pair: (pair[1]['case'], pair[1]['position'])):
        guard(); case, position = row['case'], row['position']; path = RUN / 'qualify/cases' / case / 'pixels.npz'
        if case not in arrays:
            assert sha(path) == qualified[case]['pixels_sha256']
            sources[str(path.relative_to(ROOT))] = sha(path)
            with np.load(path, allow_pickle=False) as data:
                arrays[case] = {name: data[name].copy() for name in ['candidate_ink', 'parent_ink', 'gold_ink']}
        data = arrays[case]; gb = not bool(data['gold_ink'][position].any()); pb = not bool(data['candidate_ink'][position].any())
        counts['source_blank'] += int(gb); counts['output_blank'] += int(pb); counts['both_blank'] += int(gb and pb)
        counts['false_blank'] += int(pb and not gb); counts['blank_source_nonblank_output'] += int(gb and not pb)
        reading = frozen[key]; clear = bool(reading['text'].strip()) and not reading['uncertain'] and not reading['blank']
        if clear:
            continue
        record = dict(case=case, position=position, opaque_id=key, source_blank=gb, candidate_blank=pb,
            original_reading=reading, pixels_sha256=sha(path), images={})
        for role, name in [('source', 'gold_ink'), ('point', 'parent_ink'), ('joint', 'candidate_ink')]:
            ink = data[name][position, 0]
            assert ink.shape == (32, 32) and np.isin(ink, [0, 1]).all()
            filename = f'{case.split("__")[-1]}_{position:03}_{role}.png'
            output = destination / filename
            image = Image.fromarray(255 - 255 * ink.astype(np.uint8)).resize((96, 96), Image.Resampling.NEAREST)
            image.save(output)
            record['images'][role] = dict(file=str(output.relative_to(ROOT)), sha256=sha(output),
                ink_pixels=int(ink.sum()), original_shape=[32, 32], nearest_expansion=3)
        samples.append(record)
    assert len(samples) == 17 and sum(s['candidate_blank'] for s in samples) == 15
    assert counts == dict(source_blank=4, output_blank=17, both_blank=3,
        false_blank=14, blank_source_nonblank_output=1), counts
    selected_probability = probability['selected_positions']
    book = selected_probability['development/book/fixed_cells']; qa = selected_probability['train/qa/fixed_cells']
    assert book['false_blank'] == counts['false_blank'] == 14
    totals = {key: sum(g[key] for g in probability['groups'].values()) for key in
              ['false_blank', 'source_preferred_false_blank', 'blank_upper_bound_holds_false_blank']}
    assert totals['false_blank'] == 349 and totals['source_preferred_false_blank'] == 16
    value = dict(schema='joint-writer-native-figure-data-v1', source_sha256=sources, groups=groups,
        physical_gate_passed=False, threshold=.9, counts=counts, blank_book_samples=samples,
        blank_flag_audit_file=str(audit_path.relative_to(ROOT)), blank_flag_audit_sha256=sha(audit_path),
        all17uncredited_empty=False, empty_uncredited=15, nonempty_uncredited=2, credited_empty_development_book=2,
        original_review_credits_have_detected_errors=True,
        selected_search_categories=['Dev books', 'Train QA fixed'],
        selected_source_preferred=[book['source_preferred_false_blank'], qa['source_preferred_false_blank']],
        selected_false_blanks=[book['false_blank'], qa['false_blank']], selected_search_required=[7, 1],
        search_only_criterion_passed=False, probability_totals=totals,
        dev_book_false_blank_mode_supported=probability['groups']['development/book/fixed_cells']['blank_upper_bound_holds_false_blank'],
        dev_book_all_false_blank=probability['groups']['development/book/fixed_cells']['false_blank'],
        descriptive_denominator_correction=dict(previous_prose=13, actual_array_count=14, source_JSON_already_correct=True,
            cause='Only3of4blanksourcepositionsoverlap17blankoutputs', threshold_or_frozen_reading_changed=False),
        complete_language='unmeasured', new_model_calls=0, new_optimizer_updates=0, new_generation=False,
        physical_review=physical['review'], trained_head_parameters=physical['training']['head_parameters'],
        inherited_visual_contextual_parameters=physical['inherited_visual_contextual_parameters'])
    out = HERE / 'evidence/joint_writer_figure_data.json'; out.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    guard(); print(json.dumps(dict(data_sha256=sha(out), samples=len(samples), counts=counts, new_model_calls=0)))


if __name__ == '__main__':
    main()
