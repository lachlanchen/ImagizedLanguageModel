"""Blind complete authentic sentence pairs and measure simple physical matching shortcuts."""
import json
from collections import Counter

import numpy as np

from scripts.probe_phrase_teacher import SEED, WORK, prompts
from scripts.train_first_emission import WORK as CURRENT
from scripts.train_local_ink import load, require, sha, tiles, write_new

OUT=CURRENT/'sentence_reading'


def cosine(left,right):
    norm=np.sqrt(sum(n*n for n in left.values())*sum(n*n for n in right.values()))
    return sum(n*right[k] for k,n in left.items())/norm if norm else 0.


def run():
    plan=[c for c in prompts() if c['side']==0 and not c['reverse'] and not c['swap']]
    require(len(plan)==32,'Incomplete calibration roster')
    indices={i for case in plan for i,c,j,k in case['coords']}
    with (WORK/'rows.jsonl').open() as reader:
        rows={i:json.loads(line) for i,line in enumerate(reader) if i in indices}
    bank=tiles()
    rng=np.random.default_rng(SEED+1)
    questions,answers,scores=[],[],[]
    for case in plan:
        pair=[rows[v[0]] for v in case['coords']]
        source=[Counter(map(bytes,bank[r['x']])) for r in pair]
        target=[Counter(map(bytes,bank[r['y']])) for r in pair]
        matrix=np.array([[cosine(x,y) for y in target] for x in source])
        overlap=matrix[0,0]+matrix[1,1]-matrix[0,1]-matrix[1,0]
        x,y=[[len(r[k]) for r in pair] for k in ('x','y')]
        length=abs(x[0]-y[1])+abs(x[1]-y[0])-abs(x[0]-y[0])-abs(x[1]-y[1])
        order=rng.permutation(2)
        questions.append(dict(case=len(questions)+1,sources=[r['source'] for r in pair],
                              translations=[pair[i]['target'] for i in order]))
        answers.append(order.tolist())
        scores.append(dict(case=len(scores)+1,overlap_margin=float(overlap),length_margin=length))
    OUT.mkdir()
    write_new(OUT/'packet.json',questions)
    write_new(OUT/'mapping.json',answers)
    metrics={}
    for field in ('overlap_margin','length_margin'):
        values=np.array([row[field] for row in scores])
        metrics[field]=dict(correct=int((values>1e-8).sum()),ties=int((abs(values)<=1e-8).sum()),
                            wrong=int((values < -1e-8).sum()))
    write_new(OUT/'preparation.json',dict(pairs=len(questions),metrics=metrics,
        roster_sha256=sha(WORK/'sites.npz'),packet_sha256=sha(OUT/'packet.json'),mapping_sha256=sha(OUT/'mapping.json'),
        scores=scores))
    print(metrics)


if __name__=='__main__':
    run()
