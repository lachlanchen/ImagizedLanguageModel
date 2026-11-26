"""Recount clipped-memory comparisons after dropping alphabetic fragments at crop boundaries."""
import json
from zipfile import ZipFile
import numpy as np
from scripts.evaluate_book_page_memory import PUB,ARCHIVE,base,PageWordMemory,dataset,gpu_owner,configure_cuda_cli,unpack


def complete(route,page,left,right):
    start=max(0,left)
    end=min(len(page),right)
    if start>0 and page[start-1] in route.alpha:
        while start<end and page[start] in route.alpha:start+=1
    if end<len(page) and page[end] in route.alpha:
        while end>start and page[end-1] in route.alpha:end-=1
    return route.spans(page[start:end])


def main():
    result=base.read(PUB/'result.json')
    assert base.sha(ARCHIVE)==result['archive']
    with gpu_owner(),ZipFile(ARCHIVE) as z:
        route=PageWordMemory('cuda')
        rows,_=dataset(route)
        original=json.loads(z.read('records.json'))
        by_id={r['i']:r for r in rows}
        totals={'short':0,'foreign':0,'page':0}
        changed=dict.fromkeys(totals,0)
        gains=losses=0
        first=None
        for j in range(0,len(rows),64):
            news=rows[j:j+64];old=original[j:j+64]
            queries=[];banks=[];free=[]
            for r,f in zip(news,old,strict=True):
                assert r['i']==f['index']
                p=r['source'];d=by_id[f['donor']]['source'];n=len(p)
                queries.append(p[-32:])
                banks.append(dict(short=complete(route,p,n-96,n),page=route.spans(p),
                    foreign=list(dict.fromkeys(complete(route,p,n-32,n)+complete(route,d,0,len(d)-32)))))
                blob=z.read(f"{r['i']}/free.png")
                free.append(route.encode(unpack(blob,f['answers']['free']['cells'])))
            chosen=route.rank_many(queries,banks,free)
            for r,f,vals in zip(news,old,chosen,strict=True):
                credit={}
                for m,ids in vals.items():
                    out=route.motor[list(ids)]
                    credit[m]=np.array_equal(out,r['target_image'])
                    totals[m]+=int(credit[m])
                    prior=unpack(z.read(f"{r['i']}/{m}.png"),f['answers'][m]['cells'])
                    changed[m]+=int(not np.array_equal(out,prior))
                gains+=int(credit['page'] and not credit['short'])
                losses+=int(credit['short'] and not credit['page'])
                if first is None and credit['page'] and not credit['short']:
                    first=r['i']
        assert changed['page']==0 and totals['page']==result['totals']['page']
    base.dump(PUB/'crop.json',dict(original=result['archive'],complete_word_counts=totals,changed_outputs=changed,
        paired_gains=gains,paired_losses=losses,first_gain=first))
    print(totals,changed,gains,losses)


if __name__=='__main__':
    configure_cuda_cli()
    if not (PUB/'crop.json').exists():main()
