#!/usr/bin/env python3
"""Independently recount the completed fitted-density probe without model calls."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from scripts.analyze_joint_writer_blank_density import ROOT, OLD, RUN, PUBLIC, load, sha, require, write_new

OUTPUT = ROOT / 'references/evidence/joint_writer_blank_density_result_20261113.json'


def summarize():
    import numpy as np
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only recount')
    reg, report = load(PUBLIC), load(RUN / 'report.json')
    require(report['complete'] and report['registration_sha256'] == sha(PUBLIC), 'Wrong completed probe')
    for name, digest in reg['source_sha256'].items():
        require(sha(ROOT / name) == digest, 'Bound probe source changed: ' + name)
    for name, digest in reg['input_sha256'].items():
        require(sha(ROOT / name) == digest, 'Probe input changed: ' + name)
    jobs = {x['job']['id']: x['job'] for x in load(OLD / 'qualify/report.json')['records']}
    require(len(jobs) == len(report['cases']) == 124, 'Complete cohort required')
    require(len({x['case'] for x in report['cases']}) == 124, 'Duplicate probe case')
    groups, selected = defaultdict(Counter), defaultdict(Counter)
    arrays = {}; total = 0
    for row in report['cases']:
        path = ROOT / row['file']; require(sha(path) == row['sha256'], 'Probability array changed')
        arrays[row['file']] = row['sha256']; job = jobs[row['case']]
        group = '/'.join([job['split'], job['kind'], job['layout']])
        with np.load(path, allow_pickle=False) as a:
            n = len(a['gold_blank']); require(n == row['positions'] == job['target_glyphs'], 'Missing target positions')
            require(all(np.isfinite(a[k]).all() for k in ['source_logp', 'blank_logp', 'emitted_logp', 'nonempty_upper_bound']), 'Nonfinite scores')
            expected = np.zeros(n, dtype=bool); expected[job['positions']] = True
            require(np.array_equal(expected, a['selected']), 'Selected positions changed')
            false_blank = a['emitted_blank'] & ~a['gold_blank']
            better = a['source_logp'] > a['blank_logp'] + reg['numerical_margin_nats']
            bounded = a['nonempty_upper_bound'] < a['blank_logp'] - reg['numerical_margin_nats']
            groups[group].update(positions=n, gold_blank=int(a['gold_blank'].sum()),
                emitted_blank=int(a['emitted_blank'].sum()), false_blank=int(false_blank.sum()),
                source_preferred_false_blank=int((false_blank & better).sum()),
                blank_upper_bound_holds_false_blank=int((false_blank & bounded).sum()),
                saved_blank_parallel_argmax_disagreements=int(a['blank_argmax_disagreements'][a['emitted_blank']].sum()))
            selected[group].update(false_blank=int((expected & false_blank).sum()),
                source_preferred_false_blank=int((expected & false_blank & better).sum()),
                blank_upper_bound_holds_false_blank=int((expected & false_blank & bounded).sum()))
            total += n
    groups = {k: dict(v) for k, v in groups.items()}
    selected = {k: dict(v) for k, v in selected.items()}
    require(groups == report['groups'] and total == 12597, 'Independent full-position recount differs')
    book = selected['development/book/fixed_cells']['source_preferred_false_blank']
    qa = selected['train/qa/fixed_cells']['source_preferred_false_blank']
    decision = book >= 7 and qa >= 1
    require(decision == report['search_only_hypothesis_merits_separate_registration'], 'Registered action differs')
    return dict(schema='joint-writer-blank-density-completed-result-v1',
        export_source_sha256=sha(Path(__file__)), registration_sha256=sha(PUBLIC),
        private_report_sha256=sha(RUN / 'report.json'), case_array_sha256=arrays,
        complete_cases=124, positions=total, groups=groups, selected_positions=selected,
        numerical_margin_nats=reg['numerical_margin_nats'],
        search_only_hypothesis_merits_separate_registration=decision,
        action='Search-only criterion fails. Preserve the failed physical verdict and test whether joint image likelihood can train the continuation states, with a matched unchanged control and new registration.',
        resources={k: report[k] for k in ['total_child_seconds', 'elapsed_seconds',
            'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']},
        cost_scope='The script reports total time including neural imports; monitor time starts after imports. The allocation is for the small frozen head, not the complete reader.',
        reader_calls=0, new_optimizer_updates=0, generated_images=0,
        frozen_physical_readings_changed=False, autonomous_language_quality='unmeasured',
        finite_precision_bound_is_not_formal_certificate=True, independent_full_array_recount=True,
        independent_human_adjudication=False, goal_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); value = summarize()
    if args.verify:
        require(load(OUTPUT) == value, 'Export does not reproduce')
    else:
        write_new(OUTPUT, value)
    print(json.dumps(dict(verified=args.verify, positions=value['positions'],
        search_only_criterion=value['search_only_hypothesis_merits_separate_registration'], new_model_calls=0)))
