"""Measure achievable and modal next-glyph support among source-retrieved bilingual passages."""
from collections import Counter,defaultdict
from pathlib import Path
from time import monotonic
from zipfile import ZipFile
import json
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from ilm.visual_lm.optical_lexicon import StripInventory
from scripts.prepare_visual_sequence_preference import ROOT,WORK,PUB,base,prior
from scripts.evaluate_visual_sequence_preference import ARCHIVE

RULE=ROOT/'references/visual_passage_memory_probe.md'


def index(rows,vocab):
    counts=np.count_nonzero(rows,axis=1)
    terms=rows[rows!=0].astype(np.int32)
    indptr=np.r_[0,counts.cumsum()].astype(np.int32)
    m=csr_matrix((np.ones(len(terms),np.float32),terms,indptr),shape=(len(rows),vocab))
    m.sum_duplicates()
    idf=1+np.log((len(rows)+1)/(np.bincount(m.indices,minlength=vocab)+1))
    m.data=(1+np.log(m.data))*idf[m.indices].astype(np.float32)
    normalize(m,norm='l2',copy=False)
    return m,idf,counts


def neighbors(query,m,idf):
    terms,n=np.unique(query,return_counts=True)
    weights=(1+np.log(n))*idf[terms]
    q=csr_matrix((weights.astype(np.float32),terms,np.array([0,len(terms)])),shape=(1,m.shape[1]))
    sims=(m@q.T).toarray().ravel()
    floor=np.partition(sims,-32)[-32]
    top=np.flatnonzero(sims>=floor)
    return top[np.lexsort((top,-sims[top]))[:32]]


def support(selected,target_strips,table,content,gold,truth,errors):
    counts=defaultdict(Counter)
    union=set()
    exact=False
    for idx in selected:
        s=target_strips[idx]
        s=s[s!=0].tolist()
        union.update(s)
        seq=[g for strip in s for g in table[strip]]
        exact|=seq==truth
        hist=[0,0,0]
        for val in seq+[1]:
            counts[tuple(hist[-3:])][val]+=1
            hist.append(val)
    hist=[0,0,0]
    out=Counter(covered=len(union&gold&content),reference=len(gold&content),target_identical=int(exact))
    for k,val in enumerate(truth+[1]):
        if errors[k]:
            ctr=counts[tuple(hist[-3:])]
            out['errors']+=1
            out['available']+=int(ctr[val]>0)
            out['proposed']+=int(bool(ctr))
            if ctr:
                choice=min(ctr,key=lambda cell:(-ctr[cell],cell))
                out['correct_vote']+=int(choice==val)
        hist.append(val)
    return out


def main():
    start=monotonic()
    rng=np.random.default_rng(89927)
    corpus=base.read(prior.PUB/'corpus.json')
    tables=base.read(base.TABLE)['strip']
    inv=base.read(base.BASE/'inventories.json')
    names=base.read(base.BASE/'readback.json')
    strip=[StripInventory(a,b) for a,b in zip(tables,inv,strict=True)]
    matrix=[]
    for t in (0,1):
        path=prior.WORK/f'strips-{t}.npy'
        assert base.sha(path)==corpus['strips'][t]
        matrix.append(np.load(path,mmap_mode='r'))
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    detail=base.read(PUB/'paths.json')
    assert base.sha(ARCHIVE)==base.read(PUB/'generation.json')['archive']
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
    totals=defaultdict(Counter)
    records=[]
    for t in (0,1):
        s=1-t
        m,idf,lengths=index(matrix[s],len(tables[s]))
        bins={v:np.flatnonzero(lengths//8==v) for v in np.unique(lengths//8)}
        content={k for k,seq in enumerate(tables[t]) if any(c.isalnum() for g in seq for c in names.get(str(g),''))}
        for i,r in enumerate(rows):
            if r['t']!=t:continue
            source=face.encode(r['source'])
            truth=face.encode(r['ref'])
            query=strip[s].source_strips(source)[1:]
            gold=set(strip[t].source_strips(truth)[1:])
            top=neighbors(query,m,idf)
            assert all(matrix[s][j][matrix[s][j]!=0].tolist()!=query for j in top)
            random=[]
            for j in top:
                choices=bins[lengths[j]//8]
                while True:
                    candidate=int(rng.choice(choices))
                    if candidate not in random and candidate not in top:break
                random.append(candidate)
            errors=np.array(detail[i]['parent']['ref']['prefix_margins'])<0
            assert len(errors)==len(truth)+1
            rec=dict(i=i,t=t)
            for name,ids in (('related',top),('random',random)):
                measured=support(ids,matrix[t],tables[t],content,gold,truth,errors)
                totals[name].update(measured)
                rec[name]=dict(ids=list(map(int,ids)),counts=measured)
            records.append(rec)
    assert not face.bad
    base.dump(PUB/'passage-memory.json',dict(archive=base.sha(ARCHIVE),rule=base.sha(RULE),sources=corpus['strips'],
              implementation=base.sha(Path(__file__)),seconds=monotonic()-start,totals=totals,records=records))
    print('related versus random',dict(totals),flush=True)


if __name__=='__main__':
    main()
