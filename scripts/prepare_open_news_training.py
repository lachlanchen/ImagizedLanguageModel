"""Build disjoint optical news supervision from professionally reviewed translations."""
from collections import defaultdict
import csv
from hashlib import sha256
from pathlib import Path
import re
from urllib.request import urlopen
import shutil

import numpy as np
from scripts.calibrate_translation_graders import ROOT, SOURCE, SOURCE_HASH, REV, base

PUB=ROOT/'references/evidence/open_news_training_v1'
WORK=ROOT/'artifacts/open_news_training_v1'
RULE=ROOT/'references/open_news_training_protocol.md'
FILES={
    '2021':('newstest2021/zhen/mqm_newstest2021_zhen.tsv','78c79128752881f5ce2b6256a04e4412eafd843c4ea6a583423b00facc1f68bd'),
    '2022':('generalMT2022/zhen/mqm_generalMT2022_zhen.tsv','f7905f799c47a9a2510996dac62d1fa9ce0a5d74e772bfff4029febd58f5cad0'),
    '2023':('generalMT2023/zhen/mqm_generalMT2023_zhen.tsv','d368e6bd77e00c469ecf92a6f56bf07320ed2fd951661e61580bd8d49708603f'),
    'ted':('ted/zhen/mqm_ted_zhen.tsv','d91fc0dd21c1949f1e7570671d01ceb1798ebd8d7ea8998844b1a35983de4ed3'),
}


def digest(s):
    return sha256(s.encode()).hexdigest()


def annotations(path):
    with path.open(encoding='utf-8',newline='') as stream:
        yield from csv.DictReader(stream,delimiter='\t',quoting=csv.QUOTE_NONE)


def clean(s):
    return re.sub(r'</?v>','',s).strip()


def key(s):
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())


def main():
    PUB.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    assert base.sha(SOURCE)==SOURCE_HASH
    excluded=[set(),set()]
    for r in annotations(SOURCE):
        for idx,col in enumerate(('source','target')):
            excluded[idx].add(key(clean(r[col])))
    variants=defaultdict(lambda:defaultdict(lambda:defaultdict(set)))
    for name,(rel,expected) in FILES.items():
        p=SOURCE.parent/(name+'-zhen.tsv')
        if not p.exists():
            tmp=p.with_suffix('.partial')
            with urlopen(f'https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/{REV}/{rel}',timeout=90) as res,tmp.open('wb') as out:
                shutil.copyfileobj(res,out)
            assert base.sha(tmp)==expected
            tmp.replace(p)
        assert base.sha(p)==expected
        for r in annotations(p):
            a,b=clean(r['source']),clean(r['target'])
            if not (1<=len(a)<=96 and 1<=len(b)<=96):
                continue
            variants[(a,b)][(name,r['doc'])][r['rater']].add(r['category'].casefold())
    grouped=defaultdict(list)
    for (a,b),occurrences in variants.items():
        if not any('\u3400'<=c<='\u9fff' for c in a) or not any(c.isascii() and c.isalpha() for c in b):
            continue
        if key(a) in excluded[0] or key(b) in excluded[1] or not key(b):
            continue
        if any(tags!={'no-error'} for raters in occurrences.values() for tags in raters.values()):
            continue
        docs=sorted(occurrences)
        # A sentence reused between documents cannot bind those documents to different partitions.
        if len(docs)!=1:
            continue
        name,doc=docs[0]
        grouped[a].append((len(occurrences[docs[0]]),digest(b),b,name,doc))
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    inv=list(map(set,base.read(base.BASE/'inventories.json')))
    chosen=[];seen=[set(),set()]
    for a,alts in sorted(grouped.items(),key=lambda v:digest(v[0])):
        for raters,_,b,year,doc in sorted(alts,key=lambda x:(-x[0],x[1])):
            cells,targets=face.encode(a),face.encode(b)
            if face.bad.intersection(a+b) or not set(targets)<=inv[0] or key(a) in seen[0] or key(b) in seen[1]:
                continue
            seen[0].add(key(a));seen[1].add(key(b))
            held=int(digest('open-news-7149/'+doc)[:12],16)%5==0
            chosen.append(dict(source=a,target=b,year=year,doc=doc,held=held,raters=raters,open=bool(set(cells)-inv[1])))
            break
    train=[r for r in chosen if not r['held']];held=[r for r in chosen if r['held']]
    assert {r['doc'] for r in train}.isdisjoint({r['doc'] for r in held})
    assert all(len({key(r[k]) for r in chosen})==len(chosen) for k in ('source','target'))
    stats={k:dict(count=len(rows),unfamiliar=sum(r['open'] for r in rows),documents=len({r['doc'] for r in rows}),
                  multiple_raters=sum(r['raters']>=2 for r in rows)) for k,rows in [('train',train),('held',held)]}
    path=WORK/'sentences.json';base.dump(path,chosen)
    panel=[i for flag in (False,True) for i in [j for j,r in enumerate(chosen) if r['held'] and r['open']==flag][:96]]
    result=dict(sources=FILES,revision=REV,excluding=SOURCE_HASH,protocol=base.sha(RULE),rows=base.sha(path),
                counts=stats,panel=panel,passed=len(train)>=500 and stats['train']['unfamiliar']>=100 and stats['held']['unfamiliar']>=30)
    base.dump(PUB/'selection.json',result)
    print(stats,'admitted',result['passed'],flush=True)


if __name__=='__main__':
    main()
