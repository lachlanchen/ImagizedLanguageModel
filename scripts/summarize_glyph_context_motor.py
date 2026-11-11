#!/usr/bin/env python3
"""Recount frozen motor answers, matched prefills, retention and resource costs."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import torch

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.prepare_compact_visual_span_evaluation import normalized
from scripts.review_glyph_context_motor import authenticate_review, CONTROLS, EVALUATIONS
from scripts.summarize_compact_visual_span_learning import clustered_interval, metric_rows, qa_values
from scripts.train_glyph_context_motor import load
from scripts.score_pixel_language_first_blocks import require

OWN = 'scripts/summarize_glyph_context_motor.py'


def recount(rows, references):
    groups = defaultdict(list)
    for row in rows:
        reference = references[row['case'].removesuffix('_'+row['condition'])]
        reading = row['reading']
        clear = not reading['uncertain'] and bool(normalized(reading['text'])) and row['emitted_glyphs'] > 0
        exact = clear and row['stopped'] and normalized(reading['text']) == normalized(reference['answer'])
        coherent = clear and row['stopped'] and reading['coherent_continuation'] is True
        require(row['readable'] == clear and row['complete_exact'] == exact
                and row['coherent_complete'] == coherent, 'Frozen per-answer score differs')
        groups[row['kind'], row['condition']].append(row)
    result = []
    for (kind, condition), selected in sorted(groups.items()):
        pairs = defaultdict(list)
        if kind == 'qa':
            for row in selected:
                reference = references[row['case'].removesuffix('_'+condition)]
                pairs[reference['paired_group']].append(row)
            require(set(pairs) == set(range(1, 17)) and all(len(p) == 2 for p in pairs.values()),
                    'All sixteen complete passage pairs required')
        both = sum(all(r['complete_exact'] for r in pair)
                   and pair[0]['image_sha256'] != pair[1]['image_sha256'] for pair in pairs.values())
        result.append(dict(step=12000, kind=kind, condition=condition, cases=len(selected),
            readable=sum(r['readable'] for r in selected), stopped=sum(r['stopped'] for r in selected),
            complete_exact=sum(r['complete_exact'] for r in selected),
            paired_both_correct=both if kind == 'qa' else None,
            coherent_complete=sum(r['coherent_complete'] for r in selected)
                if kind == 'book' and condition == 'normal' else None))
    return result


def main(run, cold, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only completed-output analysis')
    require(subprocess.check_output(['git', 'show', 'HEAD:'+OWN], cwd=ROOT) == (ROOT/OWN).read_bytes(),
            'Commit analysis implementation first')
    torch.set_num_threads(2)
    readings = run/'review/readings.json'
    mapping, manual = authenticate_review(run, readings)
    comparison = load(run/'review/comparison.json')
    manifest = load(run/'evaluation_manifest.json')
    references = {r['id']:r for r in load(EVALUATIONS['fixed_cells']/'references.json')['reference_rows']}
    state_checks, summaries, intervals, differences = [], {}, {}, {}
    for layout in ['fixed_cells', 'proportional']:
        scores = load(run/'review'/layout/'scores.json')
        selected = [row for row in manifest['inputs'] if row['layout'] == layout]
        require(len(selected) == len(scores['records']), 'Missing completed cases')
        actual = recount(scores['records'], references)
        require(actual == scores['results'] == comparison['results'][layout], 'Independent aggregate recount differs')
        summaries[layout] = actual
        intervals[layout] = metric_rows(scores['records'], references, 12000)
        parent = load(CONTROLS[layout]/'review/scores.json')
        differences[layout] = []
        for paired in [False, True]:
            new = qa_values(scores['records'], references, 'normal', paired)
            old = qa_values(parent['records'], references, 'normal', paired)
            differences[layout].append(dict(metric='both_correct' if paired else 'complete_exact',
                comparison='candidate_minus_matched_parent', **clustered_interval(new-old)))
        prior = load(CONTROLS[layout]/'generation/report.json')
        old_records = {r['id']:r for r in prior['records']}
        for row in selected:
            new_dir = run/('generation_'+layout)/row['id']
            old_dir = CONTROLS[layout]/'generation'/row['original_case']
            old_record = old_records[row['original_case']]
            require(old_record['input_sha256'] == row['sha256']
                    and old_record['legacy_output_sha256'] == sha(old_dir/'legacy_output.pt'),
                    'Matched parent physical input or retained tensor changed')
            left = torch.load(old_dir/'legacy_output.pt', map_location='cpu', weights_only=True)
            right = torch.load(new_dir/'legacy_output.pt', map_location='cpu', weights_only=True)
            state_checks.append(dict(case=row['id'], old_tensor_sha256=sha(old_dir/'legacy_output.pt'),
                new_tensor_sha256=sha(new_dir/'legacy_output.pt'),
                initial_context_state_bit_exact=torch.equal(left['states'][0], right['states'][0]),
                initial_state_max_absolute_difference=float((left['states'][0]-right['states'][0]).abs().max())))
    cv = load(cold/'verification.json')
    require(cv['fixed_case'] == 'fixed_cells__book_01_normal'
            and cv['all_nested_pixels_states_stopping_feedback_noise_and_motor_values_bit_exact']
            and not cv['forbidden_paths'] and not cv['IP_connection_attempts'], 'Fixed successful cold replay required')
    output.mkdir(parents=True, exist_ok=False)
    generations = {layout:load(run/('generation_'+layout)/'report.json') for layout in summaries}
    fit = load(run/'fit/report.json')
    summary = dict(schema='glyph-context-motor-language-v1', source_sha256=sha(ROOT/OWN),
        results=summaries, pointwise_passage_group_intervals=intervals,
        candidate_minus_parent_intervals=differences,
        original_exact_retention=comparison['original_exact_retention'],
        prospective_development_pilot_passed=comparison['prospective_development_pilot_passed'],
        all216_initial_context_states_bit_exact_to_matched_parent=all(x['initial_context_state_bit_exact'] for x in state_checks),
        initial_context_state_comparison_count=len(state_checks),
        independent_per_answer_and_aggregate_recount_passed=True,
        actual_outputs=216, opaque_boards_viewed=54, book_prefixes_viewed=12,
        clear_whole_readings=sum(not r['uncertain'] for r in manual['readings'].values()),
        uncertain_whole_readings=sum(r['uncertain'] for r in manual['readings'].values()),
        readings_sha256=sha(readings), reading_audit_sha256=sha(run/'review/reading_audit.json'),
        comparison_sha256=sha(run/'review/comparison.json'),
        generation_report_sha256={k:sha(run/('generation_'+k)/'report.json') for k in generations},
        generation_resources={k:{n:v[n] for n in ['elapsed_seconds', 'actor_loading_seconds',
            'total_generation_seconds', 'resource_wait_seconds', 'peak_allocated_vram_gib', 'minimum_available_gib']}
            for k,v in generations.items()},
        head_fit_report_sha256=sha(run/'fit/report.json'),
        head_fit_parameters=24477696, head_fit_updates=fit['actual_optimizer_updates'],
        inherited_visual_contextual_parameters=7522463984,
        successful_cold_verification_sha256=sha(cold/'verification.json'),
        failed_access_replay=load(ROOT/'artifacts/glyph_context_motor_cold_v1/failed_access_attempt_pixel_check.json'),
        reused_development_cohort=True, untouched_confirmation_complete=False,
        independent_human_adjudication=False, model_reviewer_prior_cohort_familiarity=True,
        strict_canonical_mismatch_does_not_establish_semantic_falsity=True,
        uncertainty='Pointwise95% percentile bootstrap,10000 resamples of sixteen paired passage groups; small reused development cohort, no simultaneous coverage. Book continuations from only two works have no generalization interval.',
        generalized_language_learning_proven=False, task_efficiency_advantage_proven=False, goal_achieved=False)
    require(len(state_checks) == 216, 'Complete matched first-state comparison required')
    save_json(output/'initial_state_comparison.json', dict(records=state_checks))
    summary['initial_state_comparison_sha256'] = sha(output/'initial_state_comparison.json')
    save_json(output/'summary.json', summary)
    save_json(ROOT/'references/evidence/glyph_context_motor_learning_20261111.json', summary)
    require(not torch.cuda.is_initialized(), 'CPU analysis initialized CUDA')
    print(json.dumps(dict(results=summaries, pilot_passed=summary['prospective_development_pilot_passed'],
        all216_initial_states_exact=summary['all216_initial_context_states_bit_exact_to_matched_parent'],
        summary_sha256=sha(output/'summary.json'), goal_achieved=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'artifacts/glyph_context_motor_v1')
    parser.add_argument('--cold', type=Path, default=ROOT/'artifacts/glyph_context_motor_cold_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/glyph_context_motor_learning_v1')
    args = parser.parse_args(); main(args.run.resolve(), args.cold.resolve(), args.output.resolve())
