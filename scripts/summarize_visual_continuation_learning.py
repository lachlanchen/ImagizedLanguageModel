#!/usr/bin/env python3
"""Export the completed, frozen continuation comparison without model calls."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2'
OUTPUT = ROOT / 'references/evidence/visual_continuation_learning_20261113.json'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def recount(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['step'], row['kind'], row['condition']].append(row)
    result = []
    for (step, kind, condition), rows in sorted(groups.items()):
        pairs = defaultdict(list)
        if kind == 'qa':
            require(len(rows) == 32, 'Keep every QA case')
            for row in rows:
                pairs[row['case'].split('_')[1]].append(row)
            require(len(pairs) == 16 and all(len(x) == 2 for x in pairs.values()),
                    'Keep both questions in every passage group')
        else:
            require(len(rows) == 12, 'Keep every book case')
        result.append(dict(step=step, kind=kind, condition=condition,
            cases=len(rows), readable=sum(x['readable'] for x in rows),
            stopped=sum(x['stopped'] for x in rows),
            complete_exact=sum(x['complete_exact'] for x in rows),
            paired_both_correct=sum(all(x['complete_exact'] for x in pair)
                and len({x['image_sha256'] for x in pair}) == 2
                for pair in pairs.values()) if kind == 'qa' else None,
            coherent_complete=sum(x['coherent_complete'] for x in rows)
                if kind == 'book' and condition == 'normal' else None))
    return result


def summarize(run):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU export only')
    review = run / 'review'
    registration = load(run / 'preparation.json')
    fit = load(run / 'fit/report.json')
    generation = load(run / 'evaluation/report.json')
    parent = load(run / 'parent_long_books/report.json')
    recovery = load(run / 'evaluation_recovery_002/report.json')
    summary = load(review / 'summary.json')
    audit = load(review / 'protocol_pilot_audit.json')
    freeze = load(review / 'reviewer_pre_unmask_freeze.json')
    frozen = load(review / 'frozen_readings.json')
    mapping = load(review / 'mapping.json')['rows']
    receipts = load(review / 'reviewer_board_receipts.json')
    require(fit['complete'] and generation['complete'] and parent['complete'],
            'Finish training and all generation first')
    require(generation['total_streams'] == 708 and parent['total_streams'] == 24,
            'Keep the full prescribed comparison')
    require(summary['registration_sha256'] == fit['registration_sha256']
            == sha(run / 'preparation.json'),
            'Wrong training registration')
    require(audit['original_summary_sha256'] == sha(review / 'summary.json')
            and audit['frozen_readings_sha256'] == summary['frozen_readings_sha256']
            == freeze['ready_readings_sha256'] == sha(review / 'frozen_readings.json'),
            'Frozen score identity differs')
    require(sha(review / 'mapping.json') == freeze['mapping_sha256']
            == frozen['mapping_sha256'] and freeze['mapping_hashed_without_decoding']
            and not freeze['reference_answers_opened'], 'Review must precede unmasking')
    require(frozen['all_boards_actually_viewed'] and receipts['all_boards_actually_viewed']
            and len(receipts['reviewed_boards']) == 183
            and set(mapping) == set(frozen['readings']) and len(mapping) == 732,
            'Complete frozen review required')
    for key, row in mapping.items():
        require(sha(review / 'pages' / (key + '.png')) == row['image_sha256'],
                'Reviewed pixels changed')
    require(sum(len(x['visible_book_prefixes_viewed'])
                for x in receipts['reviewed_boards']) == 96, 'Missing prefix review')
    cores = {layout: load(review / (layout + '_core_scores.json'))
             for layout in ['fixed_cells', 'proportional']}
    require(sum(len(x['records']) for x in cores.values()) == 660,
            'Missing core scored records')
    for layout, core in cores.items():
        require(recount(core['records']) == core['results']
                == summary['core_results'][layout], 'Per-answer recount differs')
    launch_path = ROOT / 'references/evidence/visual_continuation_full_launch_20261112.json'
    account = load(launch_path)['model_account']
    sources = ['preparation.json', 'fit/report.json', 'evaluation/report.json',
        'parent_long_books/report.json', 'evaluation_recovery_002/report.json',
        'review/summary.json', 'review/protocol_pilot_audit.json',
        'review/frozen_readings.json', 'review/reviewer_pre_unmask_freeze.json',
        'review/reviewer_board_receipts.json', 'review/mapping.json', 'review/boards.json',
        'review/fixed_cells_core_scores.json', 'review/proportional_core_scores.json']
    return dict(schema='visual-continuation-complete-language-v1',
        export_source_sha256=sha(Path(__file__)),
        post_scoring_export=True, input_sha256={name: sha(run / name) for name in sources},
        model_account_source_sha256=sha(launch_path),
        model_parameters={key: account[key] for key in ['total_unique_parameters',
            'trainable_parameters', 'inherited_visual_contextual_parameters']},
        training=dict(sequences=registration['training_cases'], kinds=registration['counts'],
            epochs=registration['epochs'], updates=fit['actual_optimizer_steps'],
            scheduled_target_positions=fit['scheduled_target_exposure'],
            snapshots={k: v['sha256'] for k, v in fit['snapshots'].items()},
            cumulative_child_seconds=fit['cumulative_elapsed_seconds'],
            pressure_wait_seconds=fit['resource_wait_seconds'],
            peak_allocated_vram_gib=fit['peak_allocated_vram_gib']),
        generated_streams=732, reviewed_boards=183, reviewed_book_prefixes=96,
        core_results=summary['core_results'], long_book_results=summary['long_book_results'],
        retention=summary['retention'], pilot_checks=audit['checks'],
        all_four_QA_erasure_arms=audit['all_four_QA_erasure_arms'],
        original_pilot_passed=summary['pilot_passed'],
        registered_protocol_pilot_passed=audit['registered_protocol_pilot_passed'],
        evaluation_resources=dict(cumulative_child_seconds=recovery['stage_child_cost_seconds'],
            candidate_report_seconds=generation['elapsed_seconds'],
            parent_report_seconds=parent['elapsed_seconds'],
            candidate_pressure_wait_seconds=generation['resource_wait_seconds'],
            parent_pressure_wait_seconds=parent['resource_wait_seconds'],
            candidate_peak_allocated_vram_gib=generation['peak_allocated_vram_gib'],
            parent_peak_allocated_vram_gib=parent['peak_allocated_vram_gib']),
        cost_scope='Child costs include preserved failed attempts; resource queues, offline preparation, CPU/review work and inherited pretraining are separate. No matched-quality efficiency claim.',
        frozen_readings_sha256=sha(review / 'frozen_readings.json'),
        independent_per_answer_aggregate_recount=True,
        reused_development_cohort=True, prior_cohort_familiarity=True,
        independent_human_adjudication=False, untouched_confirmation_complete=False,
        strict_canonical_mismatch_is_not_semantic_falsity=True,
        record_boundary_is_not_whole_document_end=True,
        new_model_calls=0, new_optimizer_updates=0, goal_achieved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    value = summarize(args.run)
    if args.verify:
        require(load(args.output) == value, 'Export is not reproducible')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write('\n')
    print(json.dumps(dict(output=str(args.output.relative_to(ROOT)),
        verified=args.verify, streams=value['generated_streams'],
        pilot_passed=value['registered_protocol_pilot_passed'], new_model_calls=0)))
