"""Reserve the WMT17 external source and reference pairs before reviewed-news training."""
import tarfile
from urllib.request import urlopen
import shutil
from lxml import etree
import numpy as np
from scripts.prepare_open_news_training import (ROOT, SOURCE, SOURCE_HASH, annotations, clean, key, digest,
                                               base, PUB as PRIOR, WORK as ORIGINAL)
PUB=ROOT/'references/evidence/open_news_external_v1'
WORK=ROOT/'artifacts/open_news_external_v1'
RULE=ROOT/'references/open_news_external_training_protocol.md'
SHA='aa1a76f61c0ca68fbf62233ee90ea9ebe2b01b43323092728510037e05e73d8d'


def main():
    PUB.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True)
    p=SOURCE.parent/'wmt17-test.tgz'
    if not p.exists():
        tmp=p.with_suffix('.partial')
        with urlopen('https://data.statmt.org/wmt17/translation-task/test.tgz',timeout=90) as res,tmp.open('wb') as f:
            shutil.copyfileobj(res,f)
        assert base.sha(tmp)==SHA
        tmp.replace(p)
    assert base.sha(p)==SHA and base.sha(SOURCE)==SOURCE_HASH
    prev=base.read(PRIOR/'selection.json');assert prev['rows']==base.sha(ORIGINAL/'sentences.json')
    forbid=[{key(r[k]) for r in base.read(ORIGINAL/'sentences.json')} for k in ('source','target')]
    for r in annotations(SOURCE):
        for j,k in enumerate(('source','target')):forbid[j].add(key(clean(r[k])))
    aligned=[]
    with tarfile.open(p) as z:
        for suffix in ('src.zh','ref.en'):
            stream=z.extractfile('test/newstest2017-zhen-'+suffix+'.sgm')
            root=etree.fromstring(stream.read(),etree.XMLParser(recover=True,resolve_entities=False,no_network=True))
            rows={(d.get('docid'),s.get('id')):''.join(s.itertext()).strip() for d in root.iter('doc') for s in d.iter('seg')}
            assert rows and None not in [k[0] for k in rows]
            aligned.append(rows)
    assert aligned[0].keys()==aligned[1].keys()
    face=base.Ink(np.load(base.BASE/'palette.npy')); inv=list(map(set,base.read(base.BASE/'inventories.json')))
    pool=[]
    for d,i in sorted(aligned[0],key=lambda k:digest('/'.join(k))):
        a,b=(x[d,i] for x in aligned)
        if not (1<=len(a)<=96 and 1<=len(b)<=96) or key(a) in forbid[0] or key(b) in forbid[1]:continue
        ax,bx=face.encode(a),face.encode(b)
        if face.bad.intersection(a+b) or not set(bx)<=inv[0] or not any('\u3400'<=c<='\u9fff' for c in a):continue
        pool.append(dict(doc=d,segment=i,source=a,target=b,open=bool(set(ax)-inv[1])))
        forbid[0].add(key(a));forbid[1].add(key(b))
    panel=[r for v in (False,True) for r in [item for item in pool if item['open']==v][:96]]
    base.dump(WORK/'test.json',panel)
    counts=[sum(r['open']==v for r in panel) for v in (False,True)]
    result=dict(protocol=base.sha(RULE),parent=base.sha(PRIOR/'selection.json'),source=SHA,segments=len(aligned[0]),
                train=prev['counts']['train'],eligible=len(pool),counts=counts,documents=len({r['doc'] for r in panel}),
                panel=base.sha(WORK/'test.json'),passed=min(counts)>=30 and prev['counts']['train']['count']>=500 and prev['counts']['train']['unfamiliar']>=100)
    base.dump(PUB/'selection.json',result)
    print(result,flush=True)


if __name__=='__main__':
    main()
