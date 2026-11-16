#!/usr/bin/env python3
"""Blind complete-stream review, immutable readings and separate language scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.register_visual_context_comparison import ROOT,REG,load,sha


def location():
    reg=load(REG);run=ROOT/reg['run'];return reg,run,run/'review'


def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')


def prepare():
    from scripts.review_compact_visual_span import verified_answer
    reg,run,review=location();manifest=load(run/'generation_manifest.json')
    if manifest['registration_sha256']!=sha(REG):raise ValueError('Comparison registration changed')
    evaluation=ROOT/reg['evaluation']
    inputs={x['id']:x for x in load(evaluation/'inputs/manifest.json')['inputs']}
    review.mkdir(exist_ok=False);pages=review/'pages';pages.mkdir();mapping={};readings={};cohorts=set()
    for directory in manifest['runs']:
        folder=ROOT/directory;arm=folder.parent.name;request=load(folder/'request.json');report=load(folder/'report.json')
        if report['request_sha256']!=sha(folder/'request.json') or report['reference_inputs'] or report['output_repaired']:
            raise ValueError('Preserve actual reference-free answers')
        profile=request['profile'];expected=set(inputs)
        step=report['optimizer_updates'];cohort=(arm,step,profile)
        if cohort in cohorts or arm not in ('trainable','frozen'):raise ValueError('Repeated or invalid matched cohort')
        cohorts.add(cohort)
        settings=reg['evaluation_configuration']
        cap=settings['long_book_cap'] if profile=='long_books' else settings['short_cap']
        if request['seed']!=settings['seed'] or request['flow_steps']!=settings['flow_steps'] or request['maximum_glyphs']!=cap:
            raise ValueError('Registered generation settings changed')
        if profile=='normal':expected={k for k,v in inputs.items() if v['condition']=='normal'}
        if profile=='long_books':expected={k for k in inputs if k.startswith('book_')}
        records=report['records']
        if len(records)!=len(expected) or {r['id'] for r in records}!=expected or {r['id'] for r in request['inputs']}!=expected:
            raise ValueError('Keep the complete registered cohort')
        for row in records:
            key=row['id'];original=inputs[key]
            if row['input_sha256']!=original['image_sha256'] or row['input_glyphs']!=original['glyphs']:
                raise ValueError('Evaluation prompt changed')
            image=verified_answer(folder/key,row)
            opaque=hashlib.sha256((sha(REG)+'/'+arm+'/'+request['checkpoint_sha256']+'/'+profile+'/'+key).encode()).hexdigest()[:24]
            if opaque in mapping:raise ValueError('Repeated blind image identity')
            shutil.copyfile(image,pages/(opaque+'.png'))
            if key.startswith('book_') and original['condition']=='normal':
                prefix=evaluation/'inputs'/original['image']
                if sha(prefix)!=original['image_sha256']:raise ValueError('Book prefix changed')
                shutil.copyfile(prefix,pages/(opaque+'_prefix.png'))
            mapping[opaque]=dict(case=key,arm=arm,step=report['optimizer_updates'],profile=profile,
                condition=original['condition'],image_sha256=row['image_sha256'],
                stopped=row['stopped'],capped=row['capped'],emitted_glyphs=row['emitted_glyphs'],
                report_sha256=sha(folder/'report.json'))
            readings[opaque]=dict(text=None,uncertain=None,coherent_continuation=None)
    required={(arm,step,'all' if step in (0,8172) else 'normal')
              for arm in ('trainable','frozen') for step in reg['snapshot_updates']}
    required|={(arm,8172,'long_books') for arm in ('trainable','frozen')}
    if len(mapping)!=792 or cohorts!=required:raise ValueError('Complete all 792 prescribed streams and fixed cohorts')
    write_new(review/'mapping.json',dict(registration_sha256=sha(REG),rows=mapping))
    write_new(review/'readings_template.json',dict(evaluator=None,readings=dict(sorted(readings.items()))))
    print(json.dumps(dict(images=792,references_opened=False,readings_frozen=False)))


def freeze(path):
    _,_,review=location();mapping=load(review/'mapping.json')['rows'];read=load(path)
    if not isinstance(read['evaluator'],str) or not read['evaluator'].strip() or read['readings'].keys()!=mapping.keys():
        raise ValueError('Identify reviewer and cover every actual stream')
    for key,row in mapping.items():
        item=read['readings'][key]
        if type(item['uncertain']) is not bool or not isinstance(item['text'],str):raise ValueError('Keep uncertainty explicit')
        if row['case'].startswith('book_') and row['condition']=='normal' and type(item['coherent_continuation']) is not bool:
            raise ValueError('Judge complete book continuation against the visible prefix')
        if sha(review/'pages'/(key+'.png'))!=row['image_sha256']:raise ValueError('Reviewed image changed')
    write_new(review/'readings_frozen.json',dict(readings_sha256=sha(path),mapping_sha256=sha(review/'mapping.json'),
        evaluator=read['evaluator'],images=792,frozen_before_reference_scoring=True))


def score(path):
    from scripts.prepare_compact_visual_span_evaluation import normalized
    reg,run,review=location();frozen=load(review/'readings_frozen.json')
    if sha(path)!=frozen['readings_sha256'] or sha(review/'mapping.json')!=frozen['mapping_sha256']:
        raise ValueError('Freeze every reading before scoring; no repair from reference answers')
    refs=ROOT/reg['evaluation']/'references.json';prep=load(ROOT/reg['evaluation']/'preparation.json')
    if sha(refs)!=prep['references_sha256']:raise ValueError('References changed')
    references={r['id']:r for r in load(refs)['reference_rows']};readings=load(path)['readings'];rows=[]
    for key,row in load(review/'mapping.json')['rows'].items():
        item=readings[key]
        if sha(review/'pages'/(key+'.png'))!=row['image_sha256']:raise ValueError('Frozen image changed')
        reference=references[row['case'].removesuffix('_'+row['condition'])]
        readable=not item['uncertain'] and bool(normalized(item['text'])) and row['emitted_glyphs']>0
        rows.append(dict(**row,reading=item,readable=readable,
            exact=readable and row['stopped'] and normalized(item['text'])==normalized(reference['answer']),
            coherent=readable and row['stopped'] and bool(item['coherent_continuation']),kind=reference['kind']))
    summaries=[]
    for arm in ('trainable','frozen'):
        for step in reg['snapshot_updates']:
            short=[r for r in rows if r['arm']==arm and r['step']==step and r['profile']!='long_books']
            normal=[r for r in short if r['kind']=='qa' and r['condition']=='normal']
            pairs={r['case'].split('_')[1] for r in normal}
            both=sum(all(r['exact'] for r in normal if r['case'].split('_')[1]==pair) for pair in pairs)
            erasures={c:sum(r['exact'] for r in short if r['kind']=='qa' and r['condition']==c)
                      for c in ('question_blank','passage_blank')}
            books=[r for r in rows if r['arm']==arm and r['step']==step and r['profile']=='long_books'
                   and r['condition']=='normal']
            summaries.append(dict(arm=arm,step=step,exact=sum(r['exact'] for r in normal),both_correct_pairs=both,
                readable=sum(r['readable'] for r in normal),erasure_exact=erasures,
                normal_minus_best_erased=sum(r['exact'] for r in normal)-max(erasures.values()),
                coherent_complete_long_books=sum(r['coherent'] for r in books),
                long_book_stops=sum(r['stopped'] for r in books),long_book_caps=sum(r['capped'] for r in books)))
    final=next(s for s in summaries if s['arm']=='trainable' and s['step']==8172)
    gate=(final['exact']>=12 and final['both_correct_pairs']>=4 and final['readable']>=24
          and final['coherent_complete_long_books']>=4 and max(final['erasure_exact'].values())<=2
          and final['normal_minus_best_erased']>=8)
    write_new(review/'scores.json',dict(registration_sha256=sha(REG),readings_sha256=sha(path),records=rows,
        summaries=summaries,development_stage_passed=gate,untouched_confirmation_measured=False,
        proportional_layout_measured=False,historical_forms_measured=False,goal_achieved=False))
    print(json.dumps(dict(summaries=summaries,development_stage_passed=gate,goal_achieved=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','freeze','score'))
    p.add_argument('--readings',type=Path);a=p.parse_args()
    if a.action!='prepare' and a.readings is None:p.error('Supply all actual image readings')
    if a.action=='prepare':prepare()
    elif a.action=='freeze':freeze(a.readings)
    else:score(a.readings)
