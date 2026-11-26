"""Compare the learned optical prior with an interpolated order-five glyph count model."""
from time import monotonic
import numpy as np
from scripts.prepare_book_order_prior import PUB, WORK, base


def get(keys, counts, queries):
    i=np.searchsorted(keys,queries).clip(max=len(keys)-1)
    return np.where(keys[i]==queries,counts[i],0)


def main():
    files=base.read(PUB/'data.json')['files']
    assert all(base.sha(WORK/f)==files[f] for f in ('train.npz','validation.npz'))
    held=np.load(WORK/'validation.npz')
    train=np.load(WORK/'train.npz')
    inv=np.array(base.read(base.BASE/'inventories.json')[0])
    # Reserve zero for left padding, count the paragraph stop as a valid prediction.
    lookup=np.zeros(inv.max()+1,dtype=np.int64)
    lookup[inv]=np.arange(1,len(inv)+1)
    radix=len(inv)+1
    assert radix**5<2**63
    y=lookup[held['intact']]
    valid=y[:,32:]>1
    results={}
    for arm in ('intact','scrambled'):
        start=monotonic()
        x=lookup[train[arm]]
        history=np.zeros_like(x)
        eval_history=np.zeros_like(y)
        prob=np.full_like(y,1/len(inv),dtype='float64')
        history_shift=1
        models=[]
        for depth in range(5):
            keys,counts=np.unique((history*radix+x)[x!=0],return_counts=True)
            parents,first,types=np.unique(keys//radix,return_index=True,return_counts=True)
            totals=np.add.reduceat(counts,first)
            index=(eval_history*radix+y)
            both=get(keys,counts,index)
            n=get(parents,totals,eval_history)
            different=get(parents,types,eval_history)
            prob=np.where(n>0,(both+different*prob)/np.maximum(1,n+different),prob)
            bits=-np.log2(prob[:,32:][valid]).mean()
            models.append(dict(order=depth+1,entries=len(keys),bits=float(bits),
                               count_positions=int(counts.sum())))
            # Append one more, older observation to the left of the existing history.
            k=depth+1
            history[:,k:]+=x[:,:-k]*history_shift
            eval_history[:,k:]+=y[:,:-k]*history_shift
            history_shift*=radix
        results[arm]=dict(seconds=monotonic()-start,orders=models)
        del history,x
    assert results['intact']['orders'][0]['bits']==results['scrambled']['orders'][0]['bits']
    base.dump(PUB/'counts.json',dict(scores=results,positions=int(valid.sum()),files=files,
        prediction='next glyph from the four immediately preceding cells; Witten–Bell interpolation with uniform zeroth order'))
    print({k:[round(z['bits'],4) for z in v['orders']] for k,v in results.items()})


if __name__=='__main__':
    main()
