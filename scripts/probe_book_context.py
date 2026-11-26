"""Does earlier prose distinguish two attested next words after identical recent ink?"""
from collections import defaultdict
from hashlib import sha256
import re

import numpy as np
import torch
from scripts.evaluate_book_order_prior import PUB,WORK,ARMS,base,restore,next_logits,continue_stream
from scripts.score_book_order_prior import interval


def select(held,names):
    by_book=defaultdict(lambda:defaultdict(list))
    for i,(seq,b) in enumerate(zip(held['intact'],held['book'],strict=True)):
        text=''.join(names[str(k)] for k in seq if k>1).replace('　',' ')
        for m in re.finditer(r'(?<= )[A-Za-z]{3,}(?= )',text):
            j=m.start()
            if j>=32:
                by_book[int(b)][text[j-8:j]].append((i,j,m.group()))
    cases=[]
    for b,groups in sorted(by_book.items()):
        chosen=set()
        pool=[]
        for items in groups.values():
            for k,u in enumerate(items):
                for v in items[k+1:]:
                    if u[0]!=v[0] and u[2].lower()!=v[2].lower():
                        digest=sha256(repr((b,u,v,'book-context-762035')).encode()).digest()
                        pool.append((digest,u,v))
        for _,u,v in sorted(pool):
            if u[0] in chosen or v[0] in chosen:
                continue
            cases.append((b,u,v));chosen.update((u[0],v[0]))
            if len(chosen)>=32:break
    return cases


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    held=np.load(WORK/'validation.npz')
    assert base.sha(WORK/'validation.npz')==base.read(PUB/'data.json')['files']['validation.npz']
    names=base.read(base.BASE/'readback.json')
    cases=select(held,names)
    res={}
    for arm in ARMS:
        w=restore(arm,'prior','cpu')
        bank=w.model.optical(w.features)
        records=[]
        for b,left,right in cases:
            sides=(left,right)
            contexts=[held['intact'][i,j-32:j] for i,j,_ in sides]
            assert np.array_equal(contexts[0][-8:],contexts[1][-8:])
            words=[held['intact'][i,j:j+len(word)+1] for i,j,word in sides]
            lls=[]
            for word in words:
                stream=torch.tensor(np.stack([np.concatenate((c,word)) for c in contexts]),dtype=torch.long)
                score=next_logits(w.model,bank,w.inv[0],stream[:,:-1])[:,32:]
                lls.append(score.gather(-1,w.local[0][stream[:,32:]].unsqueeze(-1)).double().sum((1,2)).tolist())
            margin=np.array(lls)[0]-np.array(lls)[1]
            records.append(dict(book=b,log_prob=lls,ordered=bool(margin[0]>margin[1]),
                                both=bool(margin[0]>0 and margin[1]<0)))
        pref=torch.tensor(np.stack([held['intact'][i,j-32:j] for _,l,r in cases for i,j,_ in (l,r)]),dtype=torch.long)
        free=[]
        for chunk in pref.split(32):
            ids,stops=continue_stream(w,chunk)
            for seq,stop in zip(ids,stops,strict=True):
                s=''.join(names[str(i)] for i in seq if i>1).replace('　',' ')
                match=re.match(r'^([A-Za-z]+)(?:[^A-Za-z]|$)',s)
                word=match[1] if match and (match.end(1)<len(s) or stop) else ''
                free.append(word)
        for row,(_,l,r),pa,pb in zip(records,cases,free[::2],free[1::2],strict=True):
            row['free_words']=[pa,pb]
            row['free_hits']=[pa==l[2],pb==r[2]]
        res[arm]=dict(rows=records,ordered=sum(r['ordered'] for r in records),both=sum(r['both'] for r in records),
                      free_words=sum(sum(r['free_hits']) for r in records),free_pairs=sum(all(r['free_hits']) for r in records))
    ci=interval([int(a['ordered'])-int(b['ordered']) for a,b in zip(res['intact']['rows'],res['scrambled']['rows'],strict=True)],
                [b for b,_,_ in cases])
    base.dump(PUB/'distant_context.json',dict(cases=cases,results=res,ordered_difference_ci=ci,books=len(set(r[0] for r in cases)),
        selection='Same held book, identical last eight of 32 pictured cells, unequal observed words of three or more letters followed by a space; hash-ordered disjoint windows, up to 16 pairs per book. These are attested continuations, not uniquely valid alternatives.'))
    print(len(cases),{k:{n:v for n,v in r.items() if n!='rows'} for k,r in res.items()},ci)


if __name__=='__main__':
    main()
