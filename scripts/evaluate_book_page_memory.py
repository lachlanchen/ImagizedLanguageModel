"""Evaluate page-supported optical word writing on a separately frozen book partition."""
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import urlretrieve
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pyarrow.parquet as pq
import torch

from ilm.visual_lm.page_word_memory import PageWordMemory
from scripts.evaluate_book_order_prior import base, ROOT, emit as put, unpack, png, gpu_owner, configure_cuda_cli, restore
from scripts.measure_book_discourse import emit as free_write, DIGEST
from scripts.score_book_order_prior import interval

REV='5953bd97664b64b95754f299b2309ecfbfbe81b9'
URL=f'https://huggingface.co/datasets/cimec/lambada/resolve/{REV}/plain_text/validation-00000-of-00001.parquet'
HASH='dc06d2ead0fc40bd0845936d6b2375f00396d7baac92d8335707d6a68b3d25a9'
PUB=ROOT/'references/evidence/book_page_memory_v1'
WORK=ROOT/'artifacts/book_order_prior_v1'
RAW=ROOT/'data/raw/pg19-optical'
ARCHIVE=ROOT/'publication/ilm-nature/evidence/book_page_words.zip'
RULE=ROOT/'references/book_page_word_memory_protocol.md'
ARMS=('free','recent','short','foreign','page')


def normalize(s):
    return ' '.join(unicodedata.normalize('NFKC',s).translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-','…':'...'})).split())


def identity(s):
    return ''.join(c for c in normalize(s).lower() if c.isalnum())


def dataset(route):
    p=RAW/'lambada-validation.parquet'
    if not p.exists():urlretrieve(URL,p)
    assert base.sha(p)==HASH and base.sha(RAW/'lambada-test.jsonl')==DIGEST
    overlaps={identity(json.loads(l)['text']) for l in (RAW/'lambada-test.jsonl').read_text().splitlines()}
    text=pq.read_table(p,columns=['text'])['text'].to_pylist()
    assert len(text)==4869
    face=base.Ink(); tile={}
    rows=[];excluded={}
    for i,s in enumerate(text):
        s=normalize(s);prefix,target=s.rsplit(' ',1);prefix+=' '
        for c in set(prefix+target)-tile.keys():
            face.encode(c)
            idx=face.encode(c)[0]
            tile[c]=face.array()[idx].copy()
        img=np.stack([tile[c] for c in prefix])
        seq=route.encode(img)
        reason=None
        if identity(s) in overlaps:reason='overlap'
        elif len(seq)<32 or any(k<0 for k in seq[-32:]):reason='unsupported local query'
        if reason:
            excluded[reason]=excluded.get(reason,0)+1
            continue
        target_ink=np.stack([tile[c] for c in target])
        rows.append(dict(i=i,source=seq,source_text=prefix,target=target,target_image=target_ink))
    return rows,excluded


@torch.inference_mode()
def run():
    PUB.mkdir(exist_ok=True)
    with gpu_owner():
        route=PageWordMemory('cuda');w=restore('intact','prior','cuda')
        rows,excluded=dataset(route)
        sort=sorted(range(len(rows)),key=lambda i:(len(rows[i]['source']),i))
        donors=dict(zip(sort,sort[1:]+sort[:1]))
        metrics=[]; examples=[]
        tmp=ARCHIVE.with_suffix('.partial')
        with ZipFile(tmp,'w',ZIP_DEFLATED) as z:
            for j in range(0,len(rows),64):
                batch=rows[j:j+64]
                queries=[r['source'][-32:] for r in batch]
                inp=torch.tensor([[int(w.lookup[route.motor[v].tobytes()]) for v in q] for q in queries],device='cuda')
                streams,stops=free_write(w,inp)
                free=[route.first_word(route.encode(w.palette[x[x>1]]),stop) for x,stop in zip(streams,stops,strict=True)]
                banks=[]
                for k,r in enumerate(batch,j):
                    q=r['source'];donor=rows[donors[k]]['source']
                    banks.append(dict(page=route.spans(q),short=route.spans(q[-96:]),
                        foreign=list(dict.fromkeys(route.spans(q[-32:])+route.spans(donor[:-32])))))
                chosen=route.rank_many(queries,banks,free)
                for ix,(r,opt,guess,bank) in enumerate(zip(batch,chosen,free,banks,strict=True),j):
                    pred=dict(**opt,free=guess,recent=next(iter(bank['page']),()))
                    m=dict(index=r['i'],donor=rows[donors[ix]]['i'],within=route.encode(r['target_image']) in bank['page'],
                           short_within=route.encode(r['target_image']) in bank['short'],answers={})
                    for arm in ARMS:
                        ink=route.motor[list(pred[arm])]
                        entry=put(z,f"{r['i']}/{arm}",ink)
                        m['answers'][arm]=dict(**entry,exact=bool(np.array_equal(ink,r['target_image'])))
                    metrics.append(m)
                    if (len(examples)<8 and m['answers']['page']['exact'] and not m['answers']['short']['exact']):
                        examples.append(r['i'])
                if j%1024==0:print('written',len(metrics),flush=True)
            z.writestr('records.json',json.dumps(metrics))
        tmp.replace(ARCHIVE)
    totals={m:sum(r['answers'][m]['exact'] for r in metrics) for m in ARMS}
    ci={k:interval([int(r['answers']['page']['exact'])-int(r['answers'][k]['exact']) for r in metrics],np.arange(len(metrics))) for k in ('short','foreign','recent')}
    # Independently re-read the first complete-page gains through the public, single-image CPU path.
    cpu=PageWordMemory('cpu'); rep=[]
    lookup={r['i']:r for r in rows}; face=base.Ink()
    with ZipFile(ARCHIVE) as z:
        ix=examples+[r['i'] for r in rows[:4]]
        for i in dict.fromkeys(ix):
            row=lookup[i];ids=face.encode(row['source_text']);pixels=face.array()[ids]
            out=cpu.write(pixels)
            rep.append(dict(index=i,same=png(out)==z.read(f'{i}/page.png')))
    base.dump(PUB/'result.json',dict(protocol=base.sha(RULE),model=base.sha(ROOT/'models/book_order_prior/intact/prior/weights.safetensors'),
        source=URL,source_hash=HASH,archive=base.sha(ARCHIVE),requested=4869,eligible=len(rows),excluded=excluded,
        totals=totals,intervals=ci,coverage=sum(r['within'] for r in metrics),short_coverage=sum(r['short_within'] for r in metrics),
        absent_hits={arm:sum(r['answers'][arm]['exact'] and not r['within'] for r in metrics) for arm in ARMS},
        examples=examples,replay=rep,passed=bool(ci['short'][0]>0 and ci['foreign'][0]>0)))
    print(totals,ci,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    if not (PUB/'result.json').exists():run()
