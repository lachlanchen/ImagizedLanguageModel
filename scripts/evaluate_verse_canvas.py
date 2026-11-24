"""Write from blank and damaged lines with fixed held authors, fonts and cell orders."""
from collections import defaultdict
from time import monotonic
import json
import numpy as np
import torch

from scripts.prepare_verse_canvas import corpus, MODEL, PUB
from scripts.train_verse_canvas import banks
from scripts.write_verse_canvas import load
from scripts.prepare_dialogue_ink import dump,sha
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot,consumed


def cases(data, subset='test', limit=4096, seed=17531):
    rng=np.random.default_rng(seed)
    rows=data[subset]
    idx=rng.permutation(len(rows))[:limit]
    test=rows[idx]
    donors=np.zeros(len(idx),int)
    for length in (5,7):
        pool=np.flatnonzero(test[:,1]==length)
        for i in pool:
            match=pool[test[pool,0]==test[i,0]]
            match=match[(test[match,2:2+length]!=test[i,2:2+length]).any(1)]
            if not len(match):
                match=pool[(test[pool,2:2+length]!=test[i,2:2+length]).any(1)]
            donors[i]=rng.choice(match)
    gaps=rng.integers(test[:,1]-1)
    pos=np.arange(7)
    return idx,test,donors,(pos>=gaps[:,None])&(pos<gaps[:,None]+2)


def generate(net,bank,rows,donor,mask,face,order):
    out=rows[:,9:16].copy()
    for length in (5,7):
        group=np.flatnonzero(rows[:,1]==length)
        for a in range(0,len(group),96):
            r=group[a:a+96]
            source=torch.tensor(rows[donor[r],2:2+length].astype(np.int64),device=bank.device)
            truth=torch.tensor(out[r,:length].astype(np.int64),device=bank.device)
            hidden=torch.tensor(mask[r,:length],device=bank.device)
            pred=net.fill(bank[face,source],bank[0,truth.masked_fill(hidden,0)],~hidden,bank[0],order)
            out[r,:length]=torch.where(hidden,pred,truth).cpu().numpy()
    return out


def aligned(train,rows):
    out=np.zeros_like(rows[:,9:16])
    for length in (5,7):
        fitting=train[train[:,1]==length]
        sel=rows[:,1]==length
        for i in range(length):
            x,y=fitting[:,2+i],fitting[:,9+i]
            pairs,counts=np.unique((x.astype(np.int64)<<16)+y,return_counts=True)
            # Orthographic-ID tie break shared with the neural argmax.
            order=np.lexsort((pairs&65535,-counts,pairs>>16))
            keys=pairs[order]
            distinct,first=np.unique(keys>>16,return_index=True)
            choice=np.full(max(int(rows[:,2+i].max()),int(x.max()))+1,np.bincount(y).argmax())
            choice[distinct]=keys[first]&65535
            out[sel,i]=choice[rows[sel,2+i]]
    return out


def main():
    meta,data=corpus()
    idx,rows,donors,gaps=cases(data)
    same=np.arange(len(rows))
    full=np.arange(7)<rows[:,1,None]
    allpred={}
    costs={}
    foreign_ok=(data['tiles'][2].any(axis=1)[rows[:,2:9]]|~full).all(1)
    with gpu_owner():
        torch.set_num_threads(2)
        for mode in ('ink','byte'):
            clock=monotonic()
            initial=snapshot()
            net,_=load(MODEL,mode,'cuda')
            bank=banks(data,mode,'cuda')
            for name,mask in [('blank',full),('gap',gaps)]:
                for cond,donor,face,order in [('original',same,0,'confidence'),('left',same,0,'left'),
                                              ('exchanged',donors,0,'confidence'),('newfont',same,2,'confidence')]:
                    if mode=='byte' and cond=='newfont':
                        continue
                    allpred[f'{mode}_{name}_{cond}']=generate(net,bank,rows,donor,mask,face,order)
            costs[mode]=dict(seconds=monotonic()-clock, energy=consumed(initial,snapshot()))
    align=aligned(data['train'],rows)
    for name,mask in [('blank',full),('gap',gaps)]:
        allpred['aligned_'+name]=np.where(mask,align,rows[:,9:16])
    known_lines={tuple(r[9:9+r[1]]) for r in data['train']}
    truth=rows[:,9:16]
    summary={}
    for key,pred in allpred.items():
        mask=gaps if '_gap' in key else full
        select=foreign_ok if key.endswith('newfont') else np.ones(len(rows),bool)
        hit=(pred==truth)|~mask
        seq=[tuple(p[:n]) for p,n in zip(pred[select],rows[select,1])]
        summary[key]=dict(lines=int(select.sum()),positions=int(mask[select].sum()),
                          hit=int(((pred==truth)&mask)[select].sum()),exact=int(hit[select].all(1).sum()),
                          unique=len(set(seq)),in_training=sum(s in known_lines for s in seq),
                          repeated=sum(len(s)!=len(set(s)) for s in seq))
    poets,ix=np.unique(rows[:,0],return_inverse=True)
    rng=np.random.default_rng(15097)
    sample=rng.integers(len(poets),size=(5000,len(poets)))
    tests={}
    for task,mask in [('blank',full),('gap',gaps)]:
        ours=(truth==allpred[f'ink_{task}_original'])&mask
        for k in [f'ink_{task}_left',f'ink_{task}_exchanged',f'byte_{task}_original',f'aligned_{task}']:
            other=(truth==allpred[k])&mask
            numerator=ours.sum(1)-other.sum(1)
            denom=mask.sum(1)
            boot=np.bincount(ix,weights=numerator)[sample].sum(1)/np.bincount(ix,weights=denom)[sample].sum(1)
            tests[k]=dict(delta=float(numerator.sum()/denom.sum()),ci=np.quantile(boot,[.025,.975]).tolist(),
                          original_only_exact=int(((ours==mask).all(1)&~(other==mask).all(1)).sum()),
                          control_only_exact=int(((other==mask).all(1)&~(ours==mask).all(1)).sum()))
    archive=PUB/'outputs.npz'
    np.savez_compressed(archive,idx=idx,donors=donors,gaps=gaps,foreign_ok=foreign_ok,**allpred)
    dump(PUB/'results.json',dict(summary=summary,comparisons=tests,authors=len(poets),
          donors_same_author=int((rows[donors,0]==rows[:,0]).sum()),archive=sha(archive),
          fitting=sha(MODEL/'writer.json'),corpus=meta['corpus'],costs=costs))
    print(summary,tests,flush=True)


if __name__=='__main__':
    main()
