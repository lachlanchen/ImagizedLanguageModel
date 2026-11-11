#!/usr/bin/env python3
"""Authenticate proportional-prompt outputs and their matched fixed-cell control."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.compare_glyph_tail_input_presentation import validate,verify_case,EVALUATION
from scripts.review_compact_visual_span import score
from scripts.prepare_compact_visual_span_evaluation import normalized
from scripts.summarize_compact_visual_span_learning import (
    authenticate_registration,metric_rows,qa_values,clustered_interval,SEED,REPLICATES)


def load(path):return json.loads(path.read_text())


def recheck_scores(evaluation,review,readings):
    saved=load(review/'scores.json')
    with tempfile.TemporaryDirectory(prefix='ilm-presentation-score-') as tmp:
        p=Path(tmp);(p/'mapping.json').symlink_to((review/'mapping.json').resolve())
        (p/'pages').symlink_to((review/'pages').resolve())
        with redirect_stdout(io.StringIO()):score(evaluation,p,readings)
        if saved!=load(p/'scores.json'):raise ValueError('Saved scores differ from frozen readings')
    return saved


def main(run,readings,audit,cold,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only analysis')
    import torch
    torch.set_num_threads(2)
    request=validate(run);evaluation=run/'evaluation';generation=run/'generation';review=run/'review'
    report=load(generation/'report.json');frozen=load(audit)
    if (frozen['readings_sha256']!=sha(readings) or frozen['images_reviewed']!=96
        or not frozen['frozen_before_reference_scoring'] or not frozen['all_display_and_original_hashes_verified']):
        raise ValueError('Complete and freeze every opaque reading before scoring')
    if len(report['records'])!=96 or not report['all_inputs_completed']:raise ValueError('Incomplete generation')
    if report['request_sha256']!=sha(generation/'request.json'):raise ValueError('Generation request changed')
    by_id={r['id']:r for r in request['inputs']}
    if set(by_id)!={r['id'] for r in report['records']}:raise ValueError('Registered input set changed')
    for r in report['records']:
        if verify_case(generation/r['id'],by_id[r['id']])!=r:raise ValueError('Actual generation case changed')
    mapping=load(review/'mapping.json')
    for row in mapping['rows'].values():
        actual=next(r for r in report['records'] if r['id']==row['case'])
        if (row['image_sha256']!=actual['image_sha256'] or row['stopped']!=actual['stopped']
            or row['emitted_glyphs']!=actual['emitted_glyphs']
            or row['checkpoint_sha256']!=request['checkpoint_sha256']
            or row['run_report_sha256']!=sha(generation/'report.json')):raise ValueError('Review relabels actual generation')
    scores=recheck_scores(evaluation,review,readings)
    parent=ROOT/'artifacts/glyph_tail_compact_cohort_v1'
    parent_readings=ROOT/'artifacts/glyph_tail_compact_blind_review_v1/readings.json'
    control=recheck_scores(EVALUATION,parent/'review',parent_readings)
    if sha(parent/'registration.json')!=request['original_control_registration_sha256']:
        raise ValueError('Fixed-cell control changed')
    if sha(evaluation/'references.json')!=request['references_sha256']:
        raise ValueError('References changed')
    _,references=authenticate_registration(EVALUATION)
    records=scores['records'];prior=[r for r in control['records'] if r['kind']=='qa']
    metrics=metric_rows(records,references,18000);differences=[]
    for paired in [False,True]:
        name='qa_both_correct' if paired else 'qa_complete_exact'
        normal=qa_values(records,references,'normal',paired)
        for condition in ['question_blank','passage_blank']:
            delta=normal-qa_values(records,references,condition,paired)
            differences.append(dict(comparison='normal_minus_'+condition,metric=name,**clustered_interval(delta)))
        for condition in ['normal','question_blank','passage_blank']:
            delta=qa_values(records,references,condition,paired)-qa_values(prior,references,condition,paired)
            differences.append(dict(comparison='proportional_minus_fixed_cells',condition=condition,metric=name,
                                    **clustered_interval(delta)))
    normal=[r for r in records if r['condition']=='normal'];by_case={r['case']:r for r in normal}
    old_normal=[r for r in prior if r['condition']=='normal']
    old_exact=[r for r in old_normal if r['complete_exact']];old_clear=[r for r in old_normal if r['readable']]
    if len(old_exact)!=2 or len(old_clear)!=3:raise ValueError('Expected fixed original control')
    retention=[]
    for old in old_clear:
        new=by_case[old['case']]
        retention.append(dict(case=old['case'],old_reading=old['reading'],new_reading=new['reading'],
            old_complete_exact=old['complete_exact'],new_complete_exact=new['complete_exact'],
            readable_case_retained=new['readable'] and new['stopped'],
            original_readable_core_retained=new['readable'] and new['stopped'] and
                (new['complete_exact'] or normalized(new['reading']['text']).startswith(normalized(old['reading']['text'])))))
    normal_metrics={r['metric']:r['count'] for r in metrics if r['condition']=='normal'}
    counts=dict(cases=32,stopped=sum(r['stopped'] for r in normal),capped=sum(not r['stopped'] for r in normal),
        empty=sum(r['emitted_glyphs']==0 for r in normal),readable=sum(r['readable'] for r in normal))
    conditions=dict(exact_QA_increases=normal_metrics['qa_complete_exact']>2,
        both_correct_pairs_increase=normal_metrics['qa_both_correct']>1,
        complete_readability_increases=counts['readable']>3,
        original_exact_answers_retained=all(by_case[r['case']]['complete_exact'] for r in old_exact),
        original_readable_cases_retained=all(r['readable_case_retained'] for r in retention),
        original_readable_core_retained=all(r['original_readable_core_retained'] for r in retention),
        normal_QA_caps_do_not_increase=counts['capped']<=24)
    cv=load(cold/'verification.json')
    if (cv['checkpoint_sha256']!=request['checkpoint_sha256'] or cv['fixed_case']!='qa_01_a_normal'
        or not cv['actual_pixels_states_stopping_and_feedback_bit_exact']
        or cv['forbidden_paths'] or cv['IP_connection_attempts']):raise ValueError('Fixed cold replay missing')
    output.mkdir(parents=True,exist_ok=False)
    selected={'qa_01_a_normal','qa_01_b_normal',*[r['case'] for r in old_clear]}
    selected|={r['case'] for r in normal if r['readable']}
    examples=[]
    for key in sorted(selected):
        folder=output/'raw_examples'/key;folder.mkdir(parents=True)
        shutil.copyfile(evaluation/'inputs'/by_id[key]['image'],folder/'input.png')
        shutil.copyfile(generation/key/'answer.png',folder/'answer.png')
        examples.append(dict(case=key,file=f'raw_examples/{key}/answer.png',
            input_sha256=by_id[key]['sha256'],image_sha256=sha(folder/'answer.png'),
            prospective_first_pair=key in {'qa_01_a_normal','qa_01_b_normal'},
            original_clear_case=key in {r['case'] for r in old_clear},
            retrospective_complete_readable_case=by_case[key]['readable']))
    result=dict(schema='frozen-glyph-input-presentation-analysis-v1',
        registration_sha256=sha(run/'registration.json'),generation_report_sha256=sha(generation/'report.json'),
        checkpoint_sha256=request['checkpoint_sha256'],rendering_audit_sha256=request['rendering_audit_sha256'],
        references_sha256=request['references_sha256'],input_manifest_sha256=request['input_manifest_sha256'],
        original_control_registration_sha256=request['original_control_registration_sha256'],
        scores_sha256=sha(review/'scores.json'),readings_sha256=sha(readings),review_audit_sha256=sha(audit),
        cold_verification_sha256=sha(cold/'verification.json'),metrics=metrics,paired_differences=differences,
        normal_output_counts=counts,retention=retention,decision_conditions=conditions,
        retain_adapter_development_rule_passed=all(conditions.values()),all_condition_counts=scores['results'],
        frozen_review=frozen,readings=load(readings),raw_examples=examples,
        source_sha256=sha(Path(__file__)),new_optimizer_updates=0,retained_native_parameters=7522463984,
        generation_resources={k:v for k,v in report.items() if k not in ['records','provenance','admission']},
        attempt_resources=[{k:v for k,v in load(p).items() if k!='terminal_status'}|dict(
            terminal_phase=load(p)['terminal_status']['phase'],original_attempt_sha256=sha(p))
            for p in sorted((generation/'attempts').glob('attempt_*.json'))],
        analysis_retention_implementation='Written during generation before any new output inspection: retain original exact answers and readable case coverage; original readable core-prefix retention reported separately and also required conservatively.',
        bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,
        interval_scope='Pointwise percentile bootstrap of16 paired passage groups, same paired draws; no independent seed replication or untouched confirmation.',
        comparison_scope='Presentation changes font face/layout/spacing together. Same full prompt content, weights, native observer and output rules. Not a pure font or length causal isolation.',
        interpretation='Development transfer diagnostic, no new training or broad language qualification',
        independent_human_adjudication=False,goal_achieved=False,
        cold_replay_passed=True,all_original_pixels_states_feedback_verified=True,
        scores_independently_recomputed=True)
    save_json(output/'summary.json',result)
    if torch.cuda.is_initialized():raise ValueError('CPU analysis initialized CUDA')
    print(json.dumps(dict(metrics=[r for r in metrics if r['condition']=='normal'],
        normal_output_counts=counts,retention=retention,decision_conditions=conditions),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_tail_input_presentation_v1')
    for name in ['readings','audit','cold','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.readings.resolve(),a.audit.resolve(),a.cold.resolve(),a.output.resolve())
