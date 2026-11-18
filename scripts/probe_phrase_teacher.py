"""Check an offline Chinese teacher on ambiguous chapter-matched insertions before distillation."""
import hashlib
import json
import re
import time

import numpy as np
import requests

from scripts.train_sentence_choice import WORK, load, require, sha, write_new

MODEL = 'qwen3:4b-q4_K_M'
SERVICE = 'http://127.0.0.1:11434'
OUT = WORK.parent/'first_emission_v1/teacher'
SEED = 8012


def prompts():
    record = load(WORK/'preparation.json')
    require(sha(WORK/'rows.jsonl')==record['rows_sha256'] and sha(WORK/'sites.npz')==record['pairs_sha256'], 'Source inventory changed')
    with np.load(WORK/'sites.npz') as z:
        coords = z['locations']
        pairs = z['pairs'][z['calibration']]
    coords = coords[pairs[np.random.default_rng(SEED).permutation(len(pairs))]]
    eligible = [pair for pair in coords if all(j and k-j>=2 for i,c,j,k in pair)]
    needed = {i for pair in eligible for i,c,j,k in pair}
    with (WORK/'rows.jsonl').open() as reader:
        rows = {i:json.loads(line) for i,line in enumerate(reader) if i in needed}
    count = 0
    for pair in eligible:
        phrases = [rows[i]['target'][j:k] for i,c,j,k in pair]
        if phrases[0][0]==phrases[1][0]:
            continue
        require(rows[pair[0,0]]['g']!=rows[pair[1,0]]['g'], 'Unmatched chapters')
        for side,(i,_,j,_) in enumerate(pair):
            for swap in (False, True):
                s,c,_,_ = pair[1-side if swap else side]
                source = rows[s]['source']
                for reverse in (False, True):
                    choices = phrases[::-1] if reverse else phrases
                    question = (f"读文言原句并选出最合适的现代汉语译文续写。竖线标示当前原文位置。"
                                f"\n原文：{source[:c]}｜{source[c:]}\n已有译文：{rows[i]['target'][:j]}"
                                f"\n甲：{choices[0]}\n乙：{choices[1]}\n甲用A表示，乙用B表示。只回答A或B。")
                    yield dict(pair=count,side=side,swap=swap,reverse=reverse,coords=pair.tolist(),
                               prompt=question,expected='AB'[side ^ reverse])
        count += 1
        if count==32:
            return
    raise ValueError('Insufficient matching calibration pairs')


def run():
    OUT.mkdir(exist_ok=True)
    cases=list(prompts())
    require(len(cases)==256,'Incomplete counterbalancing')
    session=requests.Session()
    tags=session.get(SERVICE+'/api/tags',timeout=10).json()['models']
    digest=next(v['digest'] for v in tags if v['name']==MODEL)
    options=dict(num_gpu=0,num_thread=4,num_ctx=2048,num_predict=5,temperature=0,seed=SEED)
    started=time.monotonic()
    for n,case in enumerate(cases):
        dest=OUT/f'{n:03}.json'
        binding=hashlib.sha256(json.dumps([case,MODEL,digest,options],ensure_ascii=False).encode()).hexdigest()
        if dest.exists():
            require(load(dest)['binding']==binding,'Teacher observation changed')
            continue
        response=session.post(SERVICE+'/api/chat',json=dict(model=MODEL,think=False,stream=False,options=options,
            messages=[dict(role='user',content=case['prompt'])],keep_alive='30s'),timeout=120)
        response.raise_for_status()
        data=response.json()
        answer=data['message']['content'].strip()
        match=re.fullmatch(r'([AB])[。.]?',answer)
        write_new(dest,dict(binding=binding,answer=answer,choice=match[1] if match else None,
                            expected=case['expected'],prompt_sha256=hashlib.sha256(case['prompt'].encode()).hexdigest(),
                            eval_count=data.get('eval_count'),duration_ns=data.get('total_duration')))
        if n%32==31:
            print('Responses',n+1,flush=True)
    results=[load(OUT/f'{n:03}.json') for n in range(len(cases))]
    correct=np.array([r['choice']==r['expected'] for r in results]).reshape(32,2,2,2)
    both=correct.all(axis=3)
    deltas=both[:,:,0].astype(int)-both[:,:,1]
    sample=np.random.default_rng(2).integers(0,32,(10000,32))
    interval=np.quantile(deltas.mean(1)[sample].mean(1),[.025,.975]).tolist()
    report=dict(model=MODEL,digest=digest,options=options,seed=SEED,pairs=32,sites=64,observations=len(cases),
                parse_failures=sum(r['choice'] is None for r in results),both_orders_correct=both.sum((0,1)).tolist(),
                both_recipients=both.all(1).sum(0).tolist(),one_order_only=(correct.sum(3)==1).sum((0,1)).tolist(),
                authentic_minus_exchanged_interval=interval,qualified=bool(both[:,:,0].sum()>=40 and interval[0]>0),
                total_inference_seconds=sum(r['duration_ns'] for r in results)/1e9,seconds=time.monotonic()-started,
                roster_sha256=hashlib.sha256(json.dumps(cases,ensure_ascii=False).encode()).hexdigest())
    write_new(OUT/'result.json',report)
    print(report)


if __name__=='__main__':
    run()
