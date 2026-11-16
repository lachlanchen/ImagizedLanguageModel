#!/usr/bin/env python3
"""Reserve new human-labelled source groups for evaluation only, without images.

No student/teacher model, tokenizer, renderer or scoring function is imported.
Raw references stay in ignored evaluation-only storage and are not printed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
import unicodedata
import urllib.request

import opencc

ROOT=Path(__file__).resolve().parents[1]
REGISTRATION=ROOT/'references/evidence/drcd_confirmation_reservation_registration_20261116.json'
REVISION='b944790de5af02c5fbb7cd9cb1473d27d169eebf'
FILES={
    'DRCD_test.json':(2125600,'341eda001e27a375f14e6b40976b2492b925eae2'),
    'README.md':(6956,'59f5599b2bebca5d42d77542cbbcdda16937774c'),
}
NAMESPACE='ilm-drcd-confirmation-reservation-v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def host_available():
    line=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(line.split()[1])/1024**2


def budget(start):
    if time.monotonic()-start>180 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>1:
        raise RuntimeError('Reservation exceeded180seconds or1GiB RSS')
    if host_available()<24:raise RuntimeError('Host available RAM below24GiB')


def acquire(cache,fetch,start):
    cache.mkdir(parents=True,exist_ok=True)
    for name,(size,blob) in FILES.items():
        path=cache/name
        if not path.exists():
            if not fetch:raise FileNotFoundError('Use --fetch for the pinned public source')
            temp=cache/(name+'.part')
            url=f'https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/{REVISION}/{name}'
            total=0
            with temp.open('xb') as out,urllib.request.urlopen(url,timeout=20) as response:
                while block:=response.read(1024*1024):
                    total+=len(block)
                    if total>size:raise ValueError('Source exceeds pinned byte count')
                    out.write(block);budget(start)
            if total!=size:raise ValueError('Incomplete pinned download')
            value=temp.read_bytes()
            if hashlib.sha1(f'blob {size}\0'.encode()+value).hexdigest()!=blob:
                raise ValueError('Downloaded source differs from pinned Git blob')
            os.replace(temp,path)
        value=path.read_bytes()
        if len(value)!=size or hashlib.sha1(f'blob {size}\0'.encode()+value).hexdigest()!=blob:
            raise ValueError('Existing source differs from pinned Git blob')
    return json.loads((cache/'DRCD_test.json').read_text())


def reserve(cache,output,fetch):
    start=time.monotonic()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or output.exists():
        raise ValueError('Use CPU-only execution and a new reservation directory')
    if host_available()<48:raise RuntimeError('48GiB available host RAM required at entry')
    os.nice(10)
    reg=json.loads(REGISTRATION.read_text())
    for p,h in reg['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('Bound source changed: '+p)
    package=Path(opencc.__file__).parent
    for p,h in reg['normalizer_package_sha256'].items():
        if sha(package/p)!=h:raise ValueError('Normalizer resource changed: '+p)
    converter=opencc.OpenCC('t2s')
    def normalize(text):
        return ''.join(converter.convert(unicodedata.normalize('NFKC',text)).split())
    titles={normalize('基督新教')}
    documents=set();windows=set();questions=set();known_counts=Counter()
    def add_document(text):
        value=normalize(text)
        if not value:return
        documents.add(text_sha(value))
        windows.update(value[i:i+64] for i in range(0,max(0,len(value)-63),32))
    for name in reg['cmrc_files']:
        d=json.loads((ROOT/name).read_text())
        for article in d['data']:
            titles.add(normalize(article['title']))
            for paragraph in article['paragraphs']:
                add_document(paragraph['context']);known_counts['cmrc_paragraphs']+=1
                questions.update(normalize(q['question']) for q in paragraph['qas'])
                budget(start)
    with (ROOT/reg['wikisource_file']).open() as stream:
        for line in stream:
            row=json.loads(line);titles.add(normalize(row['source_title']));add_document(row['text'])
            known_counts['wikisource_records']+=1
            if known_counts['wikisource_records']%128==0:budget(start)
    with (ROOT/reg['preference_pairs_file']).open() as stream:
        for i,line in enumerate(stream):
            row=json.loads(line)['raw']
            for field in ['system','prompt','chosen','rejected']:add_document(row[field])
            questions.add(normalize(row['prompt']));known_counts['preference_rows']+=1
            if i%128==0:budget(start)
    source=acquire(cache,fetch,start)
    output.mkdir(parents=True)
    accepted=[];quarantine=[];totals=Counter();article_reasons=Counter();annotation_reasons=Counter()
    for article in source['data']:
        totals['articles']+=1;title=normalize(article['title']);reasons=set()
        if title in titles:reasons.add('known_or_viewed_title')
        candidates=[]
        for paragraph in article['paragraphs']:
            totals['paragraphs']+=1;context=paragraph['context'];norm=normalize(context)
            if text_sha(norm) in documents:reasons.add('exact_normalized_known_document')
            if any(norm[i:i+64] in windows for i in range(max(0,len(norm)-63))):
                reasons.add('known_exact64character_window')
            valid=[]
            for q in paragraph['qas']:
                totals['questions']+=1
                if normalize(q['question']) in questions:reasons.add('known_normalized_question')
                answers=q.get('answers',[])
                bad=not answers or any(not isinstance(a.get('answer_start'),int)
                    or a['answer_start']<0 or not isinstance(a.get('text'),str) or not a['text'].strip()
                    or context[a['answer_start']:a['answer_start']+len(a['text'])]!=a['text'] for a in answers)
                if bad:
                    annotation_reasons['invalid_complete_answer_span']+=1
                    quarantine.append({'kind':'question','id':q['id'],'reason':'invalid_complete_answer_span'})
                    continue
                valid.append({'id':q['id'],'question':q['question'],'answers':answers})
            ordered=sorted(valid,key=lambda q:text_sha(NAMESPACE+'\0'+q['id']))
            pair=next(((a,b) for a,b in itertools.combinations(ordered,2)
                if normalize(a['question'])!=normalize(b['question'])
                and {normalize(x['text']) for x in a['answers']}.isdisjoint({normalize(x['text']) for x in b['answers']})),None)
            if pair is None:
                annotation_reasons['no_distinct_answer_question_pair']+=1
                quarantine.append({'kind':'paragraph','id':paragraph['id'],'reason':'no_distinct_answer_question_pair'})
                continue
            candidates.append({'article_id':article['id'],'paragraph_id':paragraph['id'],
                'title':article['title'],'context':context,'normalized_context_sha256':text_sha(norm),
                'questions':valid,'paired_question_ids':[q['id'] for q in pair],
                'evaluation_only':True,'training_or_reward_use_allowed':False})
        if reasons:
            article_reasons.update(reasons)
            quarantine.append({'kind':'article','id':article['id'],'reasons':sorted(reasons),
                               'excluded_eligible_paragraphs':len(candidates)})
        else:accepted.extend(candidates)
        budget(start)
    accepted.sort(key=lambda row:text_sha(NAMESPACE+'\0'+row['paragraph_id']))
    # No limit, answer-length filter or favorable example selection: reserve all
    # structurally paired groups surviving the declared overlap policy.
    pool=output/'evaluation_only_pool.jsonl'
    with pool.open('x') as stream:
        for row in accepted:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    qfile=output/'quarantine.json';qfile.write_text(json.dumps(quarantine,ensure_ascii=False,indent=2)+'\n')
    if not accepted:raise RuntimeError('No confirmation groups survived; do not relax screening')
    for p,h in reg['input_sha256'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('Bound source changed during reservation: '+p)
    for p,h in reg['normalizer_package_sha256'].items():
        if sha(package/p)!=h:raise RuntimeError('Normalizer changed during reservation: '+p)
    budget(start)
    report={'schema':'drcd-confirmation-reservation-result-v1','registration_sha256':sha(REGISTRATION),
        'source_repository':'DRCKnowledgeTeam/DRCD','source_revision':REVISION,'source_split':'test',
        'source_files_sha256':{p:sha(cache/p) for p in FILES},'source_files_git_blob_sha1':{p:v[1] for p,v in FILES.items()},
        'license':'CC-BY-SA-3.0','python_version':sys.version.split()[0],
        'unicode_database_version':unicodedata.unidata_version,
        'totals':dict(totals),'known_source_counts':dict(known_counts),
        'normalized_known_titles':len(titles),'normalized_known_documents':len(documents),'known64character_windows':len(windows),
        'article_exclusion_counts_overlapping':dict(article_reasons),'annotation_exclusion_counts':dict(annotation_reasons),
        'reserved_articles':len({r['article_id'] for r in accepted}),'reserved_paired_paragraphs':len(accepted),
        'reserved_questions':sum(len(r['questions']) for r in accepted),'fixed_pair_questions':2*len(accepted),
        'all_eligible_surviving_groups_reserved':True,'question_or_reference_texts_printed':False,
        'pool_sha256':sha(pool),'quarantine_sha256':sha(qfile),'candidate_image_rendering_complete':False,
        'student_or_baseline_outputs_generated':0,'student_or_baseline_scores_computed':0,
        'new_model_calls':0,'optimizer_updates':0,'training_or_reward_use_allowed':False,
        'prior_project_DRCD_search_scope':'Current tracked model/docs/reference/scripts and known cache paths; not complete historic/external exposure proof',
        'inherited_pretraining_exposure_unknown':True,'paraphrase_or_semantic_decontamination_proved':False,
        'elapsed_seconds':time.monotonic()-start,'peak_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'ceilings':reg['ceilings'],'evaluation_execution_admitted':False,'goal_achieved':False}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/evaluation_only/drcd_confirmation_v1/source')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args();reserve(args.cache,args.output,args.fetch)
