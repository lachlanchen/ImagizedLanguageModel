"""Resolve only unparsed offline teacher observations, preserving every original decision."""
import json

import numpy as np
import requests
from scripts.probe_phrase_teacher import MODEL, OUT, SERVICE, prompts
from scripts.train_sentence_choice import load, require, write_new


def run():
    base=load(OUT/'result.json')
    cases=list(prompts())
    string_labels={'甲':'A','乙':'B','甲（A）':'A','乙（B）':'B','甲(A)':'A','乙(B)':'B'}
    second=OUT/'format'
    second.mkdir(exist_ok=True)
    opts=dict(base['options'],num_predict=12)
    session=requests.Session()
    tags=session.get(SERVICE+'/api/tags',timeout=10).json()['models']
    require(next(m['digest'] for m in tags if m['name']==MODEL)==base['digest'],'Teacher checkpoint changed')
    correct=[]
    labels=missing=0
    for n,case in enumerate(cases):
        old=load(OUT/f'{n:03}.json')
        require(old['expected']==case['expected'],'Choice key changed')
        answer=old['choice']
        if answer is None and old['answer'] in string_labels:
            answer=string_labels[old['answer']]
            labels+=1
        if answer is None:
            dest=second/f'{n:03}.json'
            if not dest.exists():
                response=session.post(SERVICE+'/api/chat',json=dict(model=MODEL,think=False,stream=False,
                    options=opts,format={'type':'string','enum':['A','B']},
                    messages=[dict(role='user',content=case['prompt'])],keep_alive='30s'),timeout=120)
                response.raise_for_status()
                data=response.json()
                text=data['message']['content'].strip()
                try:
                    value=json.loads(text)
                except ValueError:
                    value=None
                write_new(dest,dict(raw=text,choice=value if value in ('A','B') else None,old_binding=old['binding']))
            retry=load(dest)
            require(retry['old_binding']==old['binding'],'Response does not match original case')
            answer=retry['choice']
        missing+=int(answer is None)
        correct.append(answer==case['expected'])
    ratings=np.asarray(correct).reshape(32,2,2,2)
    both=ratings.all(3)
    delta=(both[:,:,0].astype(int)-both[:,:,1]).mean(1)
    reps=np.random.default_rng(2).integers(0,32,(10000,32))
    interval=np.quantile(delta[reps].mean(1),[.025,.975]).tolist()
    report=dict(original_two_order_scores=base['both_orders_correct'],additional_literal_labels=labels,
                constrained_calls=len(list(second.glob('[0-9]*.json'))),unresolved=missing,
                both_orders_correct=both.sum((0,1)).tolist(),both_recipients=both.all(1).sum(0).tolist(),
                delta_interval=interval,meets_exploratory_threshold=bool(both[:,:,0].sum()>=40 and interval[0]>0))
    write_new(second/'result.json',report)
    print(report)


if __name__=='__main__':
    run()
