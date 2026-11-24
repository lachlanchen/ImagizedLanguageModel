"""Measure passage-specific likelihood on reciprocal, same-volume ending choices."""
from collections import defaultdict, Counter
from itertools import product
import json
import numpy as np
import torch

from ilm.visual_lm.book_predict import Predictor
from scripts.prepare_book_prefix import corpus, MODEL, ROOT
from scripts.package_book_prefix import ARMS
from scripts.prepare_dialogue_ink import sha, dump
from scripts.train_span_choice import gpu_owner

PUB = ROOT / 'references/evidence/book_assignment_v1'
EPS = 1.e-5


def select(meta, stream):
    positions = defaultdict(list)
    considered = covered = 0
    for split, work, volume, _, a, b in meta['records']:
        if split != 'test':
            continue
        for pos in range(a+32,b-7):
            considered += 1
            if stream[pos:pos+8].max() <= meta['train_shapes']:
                covered += 1
                positions[(volume, tuple(stream[pos-2:pos]))].append(pos)
    pairs = []
    for (volume, _), group in positions.items():
        used = set()
        for i, a in enumerate(group):
            if a in used:
                continue
            for b in group[i+1:]:
                if b < a+40 or b in used:
                    continue
                if np.array_equal(stream[a:a+8],stream[b:b+8]) or np.array_equal(stream[a-32:a],stream[b-32:b]):
                    continue
                pairs.append((volume,a,b))
                used.update((a,b))
                break
    return sorted(pairs), considered, covered


def score(model, prefixes, endings):
    scores = []
    for i in range(0,len(prefixes),32):
        x = prefixes[i:i+32]
        y = endings[i:i+32]
        history = np.column_stack((x,y[:,:-1]))
        p = model.probabilities(history,x.shape[1]-1)
        target = torch.as_tensor(y.astype(np.int64),device=p.device)
        score = p.gather(-1,target[:,:,None]).squeeze(-1).double().log().sum(-1)
        scores.append(score.cpu().numpy())
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise ValueError('Positive support violated')
    return result


def margins(matrix):
    # Difference of row contrasts removes a source-only offset before summing.
    return (matrix[:,0,0]-matrix[:,0,1]) + (matrix[:,1,1]-matrix[:,1,0])


def evaluate(matrix, rows):
    m = matrix[rows]
    d = margins(m)
    return dict(n=len(m),win=int((d>EPS).sum()),loss=int((d < -EPS).sum()),
                tie=int((abs(d)<=EPS).sum()),both_rows=int(((m[:,0,0]>m[:,0,1]+EPS)&(m[:,1,1]>m[:,1,0]+EPS)).sum()),
                half_rate=float(((d>EPS) + .5*(abs(d)<=EPS)).mean()) if len(d) else None)


def main():
    meta, data = corpus()
    stream = data['stream']
    pairs, considered, covered = select(meta,stream)
    docs = np.asarray([doc for doc, _, _ in pairs])
    works = np.array([d.split('_')[0] for d in docs])
    starts = np.array([[a,b] for _,a,b in pairs])
    x = stream[starts[:,:,None]+np.arange(-32,0)]
    y = stream[starts[:,:,None]+np.arange(8)]
    lexical = (x[:,:,None,:,None]==y[:,None,:,None,:]).sum(axis=(3,4))
    lexical_tie = margins(lexical)==0
    matrices = dict(lexical=lexical)
    with gpu_owner():
        for arm in (*ARMS,'counts'):
            model = Predictor(MODEL,arm,data,'cuda')
            for window in (32,2):
                m = np.zeros((len(x),2,2))
                for i,j in product(range(2),repeat=2):
                    m[:,i,j] = score(model,x[:,i,-window:],y[:,j])
                if window==2 and not np.allclose(m[:,0],m[:,1],atol=EPS,rtol=0):
                    raise AssertionError('A suffix-only reader distinguished identical inputs')
                matrices[f'{arm}-{window}'] = m
    volumes,volume_index=np.unique(docs,return_inverse=True)
    members = np.array([v.split('_')[0] for v in volumes])
    rng=np.random.default_rng(92631)
    samples = np.column_stack([rng.choice(idx,(5000,len(idx))) for w in sorted(set(works))
                               if len(idx:=np.flatnonzero(members==w))])
    stats={}
    for name,m in matrices.items():
        stats[name] = {str(w):evaluate(m,works==w) for w in sorted(set(works))}
        stats[name]['all'] = evaluate(m,np.ones(len(m),bool))
        stats[name]['lexical_ties'] = evaluate(m,lexical_tie)
        d = margins(m)
        success = (d>EPS)+.5*(abs(d)<=EPS)
        num = np.bincount(volume_index,weights=success)[samples].sum(1)
        denom = np.bincount(volume_index)[samples].sum(1)
        stats[name]['ci'] = np.quantile(num/denom,[.025,.975]).tolist()
    PUB.mkdir(exist_ok=True)
    path=PUB/'likelihoods.npz'
    np.savez_compressed(path,starts=starts, volumes=docs, **matrices)
    result=dict(arms=stats,considered=considered,covered=covered,pairs=len(x),volumes=len(volumes),
                corpus=meta['corpus'],receipts=sha(path),checkpoints=sha(MODEL/'bundle.json'))
    dump(PUB/'results.json',result)
    print(len(x), {k:(v['all']['half_rate'],v['all']['both_rows']) for k,v in stats.items()},flush=True)


if __name__=='__main__':
    main()
