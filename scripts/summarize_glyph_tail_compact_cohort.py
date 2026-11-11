#!/usr/bin/env python3
"""Authenticate the frozen earlier actor's matched-cohort language evaluation."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import tempfile

from ilm.image_edit.baseline import ROOT, save_json, sha
from scripts.compare_glyph_tail_compact_cohort import EVALUATION, validate, verify_case
from scripts.review_compact_visual_span import score
from scripts.summarize_compact_visual_span_learning import (
    authenticate_registration, metric_rows, qa_values, clustered_interval, SEED, REPLICATES)


def load(path):
    return json.loads(path.read_text())


def main(run, readings, audit, cold, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU analysis only')
    import torch
    torch.set_num_threads(2)
    request=validate(run);review=run/'review';generation=run/'generation'
    report=load(generation/'report.json');mapping=load(review/'mapping.json')
    frozen=load(audit)
    if (frozen['readings_sha256']!=sha(readings) or not frozen['frozen_before_reference_scoring']
            or frozen['images_reviewed']!=120 or not frozen['all_display_and_original_hashes_verified']):
        raise ValueError('Require all actual opaque readings frozen before reference scoring')
    if len(report['records'])!=120 or not report['all_inputs_completed']:
        raise ValueError('Incomplete registered generation')
    by_id={r['id']:r for r in request['inputs']}
    if set(by_id)!={r['id'] for r in report['records']}:raise ValueError('Input set differs')
    for record in report['records']:
        actual=verify_case(generation/record['id'],by_id[record['id']])
        if actual!=record:raise ValueError('Generation report differs from actual saved case')
    if len(mapping['rows'])!=120:raise ValueError('Review mapping incomplete')
    for row in mapping['rows'].values():
        actual=next(r for r in report['records'] if r['id']==row['case'])
        if (row['image_sha256']!=actual['image_sha256'] or row['stopped']!=actual['stopped']
            or row['emitted_glyphs']!=actual['emitted_glyphs']
            or row['checkpoint_sha256']!=request['checkpoint_sha256']
            or row['run_report_sha256']!=sha(generation/'report.json')):
            raise ValueError('Review mapping changed actual generation')
    scores=load(review/'scores.json')
    with tempfile.TemporaryDirectory(prefix='ilm-matched-cohort-score-') as temporary:
        check=Path(temporary)
        (check/'mapping.json').symlink_to((review/'mapping.json').resolve())
        (check/'pages').symlink_to((review/'pages').resolve())
        with redirect_stdout(io.StringIO()):score(EVALUATION,check,readings)
        if load(check/'scores.json')!=scores:raise ValueError('Authenticated rescoring differs')
    _,references=authenticate_registration(EVALUATION)
    records=scores['records'];metrics=metric_rows(records,references,18000);differences=[]
    for paired in [False,True]:
        normal=qa_values(records,references,'normal',paired)
        for condition in ['question_blank','passage_blank']:
            value=normal-qa_values(records,references,condition,paired)
            differences.append(dict(comparison='normal_minus_'+condition,
                metric='qa_both_correct' if paired else 'qa_complete_exact',**clustered_interval(value)))
    verification=load(cold/'verification.json')
    if (verification['checkpoint_sha256']!=request['checkpoint_sha256']
            or verification['fixed_case']!='book_01_normal'
            or not verification['actual_pixels_states_stopping_and_feedback_bit_exact']
            or verification['forbidden_paths'] or verification['IP_connection_attempts']):
        raise ValueError('Missing fixed-case independent cold replay')
    normal=[r for r in records if r['condition']=='normal']
    output.mkdir(parents=True,exist_ok=False)
    examples=[]
    for key in ['qa_01_a_normal','qa_01_b_normal','book_01_normal']:
        row=by_id[key];actual=next(r for r in report['records'] if r['id']==key)
        folder=output/'raw_examples'/key;folder.mkdir(parents=True)
        for name,source in [('input.png',EVALUATION/'inputs'/row['image']),('answer.png',generation/key/'answer.png')]:
            shutil.copyfile(source,folder/name)
        examples.append(dict(case=key,input_sha256=row['sha256'],image_sha256=actual['image_sha256'],
            emitted_glyphs=actual['emitted_glyphs'],stopped=actual['stopped'],file=f'raw_examples/{key}/answer.png'))
    attempts=[dict(file=p.name,sha256=sha(p),**load(p)) for p in sorted((generation/'attempts').glob('attempt_*.json'))]
    bindings={n:sha(run/n) for n in ['registration.json','overlap_audit.json','serialization_check.json',
              'generation/report.json','review/mapping.json','review/scores.json']}
    result=dict(schema='earlier-glyph-actor-compact-cohort-analysis-v1',bindings=bindings,
        checkpoint_sha256=request['checkpoint_sha256'],retained_native_parameters=request['retained_native_parameters'],
        new_optimizer_updates=0,original_optimizer_updates=18000,
        review=load(readings),review_audit=frozen,readings_sha256=sha(readings),audit_sha256=sha(audit),
        scorer_recomputed_from_frozen_readings=True,all_original_signed_raw_PNG_feedback_verified=True,
        metrics=metrics,paired_erasure_differences=differences,
        book_counts=[r for r in scores['results'] if r['kind']=='book'],
        normal_output_counts=dict(cases=len(normal),stopped=sum(r['stopped'] for r in normal),
            capped=sum(not r['stopped'] for r in normal),empty=sum(r['emitted_glyphs']==0 for r in normal),
            readable=sum(r['readable'] for r in normal)),
        all_condition_counts=scores['results'],raw_examples=examples,
        cold_verification_sha256=sha(cold/'verification.json'),cold_verified=True,
        generation_resources={k:v for k,v in report.items() if k not in ['records','provenance']},
        attempt_resources=attempts,overlap_audit=load(run/'overlap_audit.json'),
        bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,
        uncertainty='Pointwise percentile bootstrap of 16 paired passage groups. Only two book works: no book interval. Constant observations give degenerate intervals, not zero population success.',
        comparison_limit='Same physical inputs and scoring, but different inherited foundations, observers, adaptation data and writers. No causal model-size or matched-quality efficiency claim.',
        earlier_16_of_32_clear_result_different_cohort_and_metric=True,
        inherited_pretraining_exposure_unknown=True,independent_human_adjudication=False,
        goal_achieved=False,source_sha256=sha(Path(__file__)))
    save_json(output/'summary.json',result)
    if torch.cuda.is_initialized():raise ValueError('CPU analysis initialized CUDA')
    print(json.dumps(dict(metrics=metrics,normal_output_counts=result['normal_output_counts'],book_counts=result['book_counts']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_tail_compact_cohort_v1')
    p.add_argument('--readings',type=Path,required=True);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--cold',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.readings.resolve(),a.audit.resolve(),a.cold.resolve(),a.output.resolve())
