#!/usr/bin/env python3
"""CPU-only authentication and paired scoring of the complete helper diagnosis."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ilm.image_edit.baseline import ROOT,save_json,sha
from scripts.diagnose_glyph_tail_native_reading import validate,verify_case,load,EVALUATION
from scripts.prepare_compact_visual_span_evaluation import normalized
from scripts.summarize_compact_visual_span_learning import (
    qa_values,clustered_interval,SEED,REPLICATES)


def analyze(run,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():
        raise ValueError('CPU analysis into a new directory required')
    torch.set_num_threads(2)
    r=validate(run);registration_sha=sha(run/'registration.json')
    report=load(run/'execution/report.json')
    if (report['registration_sha256']!=registration_sha or not report['all448_completed']
            or len(report['records'])!=448 or report['adapted_image_prefills_bit_exact']!=192):
        raise ValueError('Complete every diagnostic arm before interpretation')
    from scripts.check_native_visual_transfer_v71 import archive_and_tokenizer
    archive,tokenizer,_=archive_and_tokenizer()
    eog=set(load(ROOT/r['compiler'])['eog_ids']);vocabulary=len(archive.field('tokenizer.ggml.tokens'))
    references={x['id']:x for x in load(EVALUATION/'references.json')['reference_rows'] if x['kind']=='qa'}
    generated={x['job']['id']:x for x in report['records']}
    if len(generated)!=448 or set(generated)!={x['id'] for x in r['jobs']}:
        raise ValueError('Generation coverage differs from registered jobs')
    arms=defaultdict(list);records=[];states={}
    for job in r['jobs']:
        directory=run/'execution'/job['id'];actual=verify_case(directory,job,registration_sha)
        if actual!=generated[job['id']]:raise ValueError('Summary relabels an actual case')
        value=torch.load(directory/'native_output.pt',weights_only=True,map_location='cpu')
        ids=value['selected_ids'];stopped=value['stopped']
        if (not 1<=len(ids)<=128 or any(type(i) is not int or not 0<=i<vocabulary for i in ids)
                or any(i in eog for i in ids[:-1]) or (ids[-1] in eog)!=stopped
                or value['capped']!=(not stopped) or (not stopped and len(ids)!=128)):
            raise ValueError('Native IDs, cap or stop decision differs')
        if tokenizer.decode(ids[:-1] if stopped else ids,skip_special_tokens=False)!=value['text']:
            raise ValueError('Raw decoded text was repaired or changed')
        base=job['case'].removesuffix('_'+job['condition']);reference=references[base]
        exact=bool(stopped and normalized(value['text'])==normalized(reference['answer']))
        row=dict(case=job['case'],kind='qa',condition=job['condition'],tail=job['tail'],layout=job['layout'],
            paired_group=reference['paired_group'],text=value['text'],complete_exact=exact,
            stopped=stopped,capped=not stopped,source_units=len(ids),
            image_sha256=hashlib.sha256(normalized(value['text']).encode()).hexdigest(),
            output_sha256=actual['output_sha256'],prefill_state_sha256=actual['prefill_state_sha256'],
            actual_actor_prefill_bit_exact=actual['actual_actor_prefill_bit_exact'])
        records.append(row);arms[job['tail']+'/'+job['layout']].append(row)
        states[job['id']]=value['hidden_states'][0]
    metrics=[];counts=[];differences=[]
    for arm,rows in arms.items():
        for condition in sorted({x['condition'] for x in rows}):
            current=[x for x in rows if x['condition']==condition]
            if len(current)!=32:raise ValueError('Missing complete question pair')
            counts.append(dict(arm=arm,condition=condition,cases=32,stopped=sum(x['stopped'] for x in current),
                capped=sum(x['capped'] for x in current),empty=sum(not x['text'].strip() for x in current),
                total_source_units=sum(x['source_units'] for x in current)))
            for paired in [False,True]:
                values=qa_values(rows,references,condition,paired)
                metrics.append(dict(arm=arm,condition=condition,
                    metric='qa_both_correct' if paired else 'qa_complete_exact',
                    count=int(values.sum()),cases=values.size,**clustered_interval(values)))
                if arm.endswith('/text'):continue
                if condition!='normal':
                    delta=qa_values(rows,references,'normal',paired)-values
                    differences.append(dict(comparison='normal_minus_'+condition,arm=arm,
                        metric='qa_both_correct' if paired else 'qa_complete_exact',**clustered_interval(delta)))
    glyphs={}
    for layout,receipt,score in [
        ('fixed_cells','glyph_tail_compact_learning_20261111.json','glyph_tail_compact_cohort_v1'),
        ('proportional','glyph_tail_input_presentation_learning_20261111.json','glyph_tail_input_presentation_v1')]:
        public=ROOT/'references/evidence'/receipt;bound=load(public)
        path=ROOT/'artifacts'/score/'review/scores.json'
        if sha(path)!=bound['scores_sha256']:raise ValueError('Frozen glyph scores changed')
        glyphs[layout]=[x for x in load(path)['records'] if x['kind']=='qa']
    for layout in ['fixed_cells','proportional']:
        for paired in [False,True]:
            metric='qa_both_correct' if paired else 'qa_complete_exact'
            for condition in ['normal','question_blank','passage_blank']:
                adapted=qa_values(arms['adapted/'+layout],references,condition,paired)
                original=qa_values(arms['source_zero_output/'+layout],references,condition,paired)
                glyph=qa_values(glyphs[layout],references,condition,paired)
                for name,delta in [('adapted_minus_source',adapted-original),('native_adapted_minus_glyph',adapted-glyph)]:
                    differences.append(dict(comparison=name,layout=layout,condition=condition,metric=metric,
                        **clustered_interval(delta)))
            for tail in ['adapted','source_zero_output']:
                delta=qa_values(arms[tail+'/text'],references,'normal',paired)-qa_values(
                    arms[tail+'/'+layout],references,'normal',paired)
                differences.append(dict(comparison='text_minus_image',tail=tail,layout=layout,
                    condition='normal',metric=metric,**clustered_interval(delta)))
    for tail in ['adapted','source_zero_output']:
        for paired in [False,True]:
            delta=qa_values(arms[tail+'/proportional'],references,'normal',paired)-qa_values(
                arms[tail+'/fixed_cells'],references,'normal',paired)
            differences.append(dict(comparison='proportional_minus_fixed_cells',tail=tail,condition='normal',
                metric='qa_both_correct' if paired else 'qa_complete_exact',**clustered_interval(delta)))
    contrasts=[]
    for row in records:
        if row['layout']=='text' or row['condition']!='normal':continue
        glyph=next(x for x in glyphs[row['layout']] if x['case']==row['case'])
        contrasts.append(dict(tail=row['tail'],layout=row['layout'],case=row['case'],
            native_complete_exact=row['complete_exact'],glyph_complete_exact=glyph['complete_exact'],
            glyph_readable=glyph['readable'],native_text=row['text'],glyph_frozen_reading=glyph['reading']))
    prefill_effects=[]
    for job in r['jobs']:
        if job['tail']!='adapted' or job['layout']=='text':continue
        adapted=states[job['id']].double();original=states['source_zero_output__'+job['key']].double()
        prefill_effects.append(dict(case=job['case'],layout=job['layout'],
            relative_RMS=float((adapted-original).norm()/original.norm().clamp_min(1e-30)),
            bit_exact=torch.equal(adapted,original)))
    attempts=[]
    for path in sorted((run/'execution/attempts').glob('attempt_*.json')):
        value=load(path)
        attempts.append({k:v for k,v in value.items() if k!='terminal_status'}|dict(
            terminal_phase=value['terminal_status']['phase'],original_attempt_sha256=sha(path)))
    output.mkdir(parents=True)
    result=dict(schema='actual-glyph-native-reading-analysis-v1',registration_sha256=registration_sha,
        generation_report_sha256=sha(run/'execution/report.json'),analysis_source_sha256=sha(Path(__file__)),
        checkpoint_sha256=r['checkpoint_sha256'],expected_prefills_sha256=r['expected_prefills_sha256'],
        references_sha256=sha(EVALUATION/'references.json'),metrics=metrics,output_counts=counts,
        paired_differences=differences,records=records,casewise_native_glyph_contrasts=contrasts,
        adapted_source_prefill_effects=prefill_effects,all448_cases_authenticated=True,
        actual_adapted_image_prefills_bit_exact=192,raw_IDs_text_states_stops_verified=True,
        helper_vocabulary_not_registered_on_actor=report['helper_vocabulary_not_registered_on_actor'],
        attempts=attempts,measured_resources={k:v for k,v in report.items() if k not in ['records','source_provenance','admission']},
        bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,
        interval_scope='Pointwise passage-group bootstrap; repeated development cohort, no untouched confirmation or independent seed.',
        strict_mismatch_not_semantic_falsity=True,semantic_accuracy_adjudicated=False,
        helper_only=True,new_optimizer_updates=0,ILM_image_answers_generated=0,
        development_training_allowed=False,goal_achieved=False)
    save_json(output/'summary.json',result)
    if torch.cuda.is_initialized():raise ValueError('CPU analysis initialized CUDA')
    print(json.dumps([x for x in metrics if x['condition']=='normal'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'artifacts/glyph_tail_native_reading_v1')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/glyph_tail_native_reading_learning_v1')
    a=p.parse_args();analyze(a.run.resolve(),a.output.resolve())
