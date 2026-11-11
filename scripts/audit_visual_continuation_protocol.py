#!/usr/bin/env python3
"""CPU-only boundary measurements and complete registered pilot gate audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/'references/evidence/visual_continuation_protocol_audit_registration_20261112.json'
FILES=('scripts/audit_visual_continuation_protocol.py','tests/test_continuation_protocol.py',
       'scripts/review_visual_continuation.py','scripts/review_compact_visual_span.py',
       'references/visual_continuation_protocol_audit.md')


def load(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT))
def require(value,message):
    if not value:raise ValueError(message)
def save_new(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('x') as file:json.dump(value,file,indent=2,ensure_ascii=False);file.write('\n')


def boundary_audit(run):
    r=load(run/'preparation.json');rows=load(run/'training_manifest.json')['rows']
    source=ROOT/'data/visual_grammar/chinese_wikisource_public_domain.jsonl'
    require(sha(source)==r['bindings_sha256'][relative(source)],'Registered book source changed')
    require(sha(run/'training_manifest.json')==r['training_manifest_sha256'],'Registered training manifest changed')
    books={row['id']:row for row in (json.loads(line) for line in source.read_text().splitlines())}
    counts=Counter();endings=Counter();lengths=Counter();sentence=set('。！？!?');clause=sentence|set('；;')
    closers=set("」』”’）》】〕〉〗\"'")
    for row in rows:
        if row['kind']!='book':continue
        text=books[row['id']]['text'];require(hashlib.sha256(text.encode()).hexdigest()==row['source_text_sha256'],'Book text changed')
        require(row['cut']==min(128,max(1,len(text)//4)) and len(text)-row['cut']==row['target_glyphs'],
                'Registered complete target/prefix boundary changed')
        tail=text.rstrip()
        while tail and tail[-1] in closers:tail=tail[:-1].rstrip()
        counts['book_records']+=1;counts['direct_sentence_mark']+=text[-1] in sentence
        counts['sentence_mark_after_closing_quotes']+=bool(tail and tail[-1] in sentence)
        counts['clause_mark_after_closing_quotes']+=bool(tail and tail[-1] in clause)
        counts['length_at_nominal640']+=len(text)==640;counts['length_at641']+=len(text)==641
        endings[text[-1]]+=1;lengths[len(text)]+=1
    require(counts['book_records']==r['counts']['book']==4762,'Keep every registered book record')
    builder=ROOT/'scripts/build_visual_grammar_manifest.py'
    return dict(schema='visual-continuation-book-boundaries-v1',registration_sha256=sha(run/'preparation.json'),
        source_sha256={relative(source):sha(source),relative(builder):sha(builder),
            relative(run/'training_manifest.json'):sha(run/'training_manifest.json'),relative(Path(__file__)):sha(Path(__file__))},
        counts=dict(counts),final_character_counts=dict(endings),record_length_range=[min(lengths),max(lengths)],
        rule='Terminal sentence punctuation, directly or after stripping closing quotes/brackets; semicolon included only in clause count.',
        punctuation_is_not_semantic_completeness=True,record_boundary_is_not_whole_document_end=True,
        data_or_labels_modified=False,new_model_calls=0,language_quality_evaluated=False,goal_achieved=False)


def pilot_checks(registration,summary):
    final=registration['total_steps'];limits=registration['pilot'];core=summary['core_results']
    require(type(summary['pilot_passed']) is bool,'Original pilot result must be a boolean')
    def only(rows,**fields):
        selected=[x for x in rows if all(x.get(k)==v for k,v in fields.items())]
        require(len(selected)==1,'Exactly one aggregate required: '+str(fields));return selected[0]
    normal=only(core['proportional'],step=final,kind='qa',condition='normal')
    require(normal['cases']==32,'Complete proportional normal QA required')
    require(all(type(normal[k]) is int for k in ['complete_exact','paired_both_correct','readable'])
        and 0<=normal['complete_exact']<=normal['readable']<=32 and 0<=normal['paired_both_correct']<=16,
        'Invalid normal QA aggregates')
    erased=[]
    for layout in ['fixed_cells','proportional']:
        for condition in ['question_blank','passage_blank']:
            row=only(core[layout],step=final,kind='qa',condition=condition)
            require(row['cases']==32,'Complete QA-erasure arm required')
            require(type(row['complete_exact']) is int and 0<=row['complete_exact']<=32,'Invalid exact aggregate')
            erased.append(dict(layout=layout,condition=condition,complete_exact=row['complete_exact']))
    book=only(summary['long_book_results'],arm='candidate_long640',step=final,condition='normal')
    require(book['cases']==12 and type(book['coherent_complete']) is int and 0<=book['coherent_complete']<=12,
        'All twelve long-book cases and valid counts required')
    retention={}
    for layout,required in registration['exact_retention'].items():
        value=summary['retention'][layout]
        require(value['required']==required and len(set(value['retained']))==len(value['retained'])
            and set(value['retained'])<=set(required),'Retention identities changed')
        retention[layout]=value['retained']==required
    best=max(x['complete_exact'] for x in erased if x['layout']=='proportional')
    checks=dict(proportional_exact=normal['complete_exact']>=limits['proportional_exact_minimum'],
        proportional_pairs=normal['paired_both_correct']>=limits['proportional_pairs_minimum'],
        proportional_readable=normal['readable']>=limits['proportional_readable_minimum'],
        long_books=book['coherent_complete']>=limits['long_book_coherent_complete_minimum'],
        each_QA_erasure_arm=all(x['complete_exact']<=limits['each_QA_erasure_exact_maximum'] for x in erased),
        proportional_erasure_difference=normal['complete_exact']-best>=limits['normal_minus_best_erased_exact_minimum'],
        exact_retention=all(retention.values()))
    passed=all(checks.values())
    require(not passed or summary['pilot_passed'],'Protocol audit must not rescue a failed original pilot')
    return dict(checks=checks,all_four_QA_erasure_arms=erased,retention=retention,
        original_summary_pilot_passed=summary['pilot_passed'],registered_protocol_pilot_passed=passed,
        untouched_confirmation_required=True,goal_achieved=False)


def register(run):
    require(not PUBLIC.exists(),'Preserve the prospective audit registration')
    require(not (run/'evaluation').exists() and not (run/'parent_long_books').exists()
        and not (run/'review').exists(),'Register before new output generation or review')
    r=load(run/'preparation.json');sources={}
    for name in FILES:
        require((ROOT/name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT),
                'Commit audit implementation first: '+name);sources[name]=sha(ROOT/name)
    for name,digest in r['source_sha256'].items():require(sha(ROOT/name)==digest,'Do not modify live registered source: '+name)
    counts=Counter((x['layout'],'book' if '__book_' in x['id'] else 'qa',x['condition']) for x in r['evaluation_inputs'])
    require(all(counts[layout,'qa',condition]==32 for layout in ['fixed_cells','proportional']
        for condition in ['normal','question_blank','passage_blank']),'Complete four QA-erasure arms required')
    return dict(schema='visual-continuation-protocol-audit-registration-v1',recorded_unix=time.time(),
        training_registration_sha256=sha(run/'preparation.json'),source_sha256=sources,
        original_training_sources_unchanged=True,new_language_outputs_available=0,
        scope='Independently apply the registered each-QA-erasure rule to all four arms; preserve original scores and summary.',
        conditions=['question_blank','passage_blank'],layouts=['fixed_cells','proportional'],
        numeric_thresholds_unchanged=r['pilot'],training_data_schedule_weights_and_generation_unchanged=True,
        original_review_script_unmodified=True,corpus_or_scoring_outcomes_used_to_relax_thresholds=False,
        new_model_calls=0,goal_achieved=False)


def audit_pilot(run):
    require(PUBLIC.read_bytes()==subprocess.check_output(['git','show','HEAD:'+relative(PUBLIC)],cwd=ROOT),
            'Commit the prospective audit registration before using it')
    contract=load(PUBLIC);r=load(run/'preparation.json')
    require(contract['training_registration_sha256']==sha(run/'preparation.json'),'Wrong full trial')
    require(contract['numeric_thresholds_unchanged']==r['pilot'],'Registered gate thresholds changed')
    for name,digest in contract['source_sha256'].items():require(sha(ROOT/name)==digest,'Audit source changed: '+name)
    review=run/'review';summary=load(review/'summary.json');mapping=load(review/'mapping.json');frozen=load(review/'frozen_readings.json')
    require(summary['registration_sha256']==mapping['registration_sha256']==sha(run/'preparation.json'),'Wrong reviewed run')
    require(summary['frozen_readings_sha256']==sha(review/'frozen_readings.json') and frozen['all_boards_actually_viewed']
        and frozen['mapping_sha256']==sha(review/'mapping.json'),'Complete frozen review required')
    require(set(mapping['rows'])==set(frozen['readings']) and len(mapping['rows'])==732,'All732 frozen streams required')
    generation=load(run/'evaluation/report.json');parent=load(run/'parent_long_books/report.json')
    require(load(run/'fit/report.json')['complete'] and generation['complete'] and generation['total_streams']==708
        and parent['complete'] and parent['total_streams']==24,'Complete prescribed fit and generation required')
    counts=Counter(x['arm'] for x in mapping['rows'].values())
    require(counts==dict(candidate_core128=660,candidate_long640=48,balanced_parent_long640=24),'Preserve original arms')
    for key,row in mapping['rows'].items():require(sha(review/'pages'/(key+'.png'))==row['image_sha256'],'Reviewed pixels changed')
    for layout in ['fixed_cells','proportional']:
        core=load(review/(layout+'_core_scores.json'))
        require(core['results']==summary['core_results'][layout],'Original aggregates changed')
        selected={tuple(row[k] for k in ['step','case','condition']):(key,row) for key,row in mapping['rows'].items()
            if row['arm']=='candidate_core128' and row['layout']==layout}
        require(len(core['records'])==len(selected),'Original core records incomplete')
        seen=set()
        for row in core['records']:
            identity=tuple(row[k] for k in ['step','case','condition'])
            require(identity in selected and identity not in seen,'Repeated or relabelled core record');seen.add(identity)
            key,original=selected[identity]
            require(all(row[k]==v for k,v in original.items()) and row['reading']==frozen['readings'][key],
                'Original core metadata or frozen reading changed')
    result=pilot_checks(r,summary)
    return dict(result,schema='visual-continuation-protocol-pilot-audit-v1',
        audit_registration_sha256=sha(PUBLIC),original_summary_sha256=sha(review/'summary.json'),
        frozen_readings_sha256=sha(review/'frozen_readings.json'),source_sha256=sha(Path(__file__)),
        all732_reviewed_image_hashes_verified=True,original_scores_preserved=True,
        audit_scope='Aggregate protocol gates only; no new interpretation or transcription of answers.',new_model_calls=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['boundaries','register','pilot'])
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='boundaries':value=boundary_audit(a.run);output=a.output or a.run/'book_boundary_audit.json'
    elif a.action=='register':value=register(a.run);output=a.output or PUBLIC
    else:value=audit_pilot(a.run);output=a.output or a.run/'review/protocol_pilot_audit.json'
    save_new(output,value);print(json.dumps(dict(output=relative(output),new_model_calls=0,goal_achieved=False)))
