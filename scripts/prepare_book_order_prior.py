"""Reserve pictured prose, matched scrambled controls and external translation panels."""
import csv
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import re
import tarfile
import unicodedata

from lxml import etree
import numpy as np
import requests
from scripts.reserve_open_news_external import ROOT, ORIGINAL, WORK as OLD, SOURCE, annotations, clean, base, key
from scripts.prepare_bilingual_ink import source as opus_rows
from scripts.prepare_opus_paired_supervision import PUB as CERT, WORK as CERT_WORK

PUB=ROOT/'references/evidence/book_order_prior_v1'
WORK=ROOT/'artifacts/book_order_prior_v1'
RAW=ROOT/'data/raw/pg19-optical'
RULE=ROOT/'references/book_order_prior_protocol.md'
GCS='https://storage.googleapis.com/deepmind-gutenberg/'
LEX=re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
SEED=762035

def digest(s):
    return sha256(s.encode()).hexdigest()

def download(url,path,expected=None):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        resp=requests.get(url,timeout=75);resp.raise_for_status()
        tmp=path.with_suffix('.part');tmp.write_bytes(resp.content);tmp.replace(path)
    if expected and base.sha(path)!=expected:raise ValueError(f'Changed source file {path}')

def reserve_books():
    entries={}
    for split,n in [('train',96),('validation',16)]:
        r=requests.get('https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o',
                       params={'prefix':split+'/','maxResults':1000,'fields':'items(name,size,md5Hash,generation)'},timeout=30)
        r.raise_for_status()
        valid=[x for x in r.json()['items'] if x['name'].endswith('.txt') and 80_000<=int(x['size'])<=950_000]
        entries[split]=sorted(valid,key=lambda x:digest('ilm-books-762035/'+x['name']))[:n]
        if len(entries[split])!=n:raise RuntimeError(f'Not enough {split} books')
    def fetch(x):
        import base64,hashlib
        dest=RAW/x['name'];download(GCS+x['name']+'?generation='+x['generation'],dest)
        if base64.b64encode(hashlib.md5(dest.read_bytes()).digest()).decode()!=x['md5Hash']:raise ValueError(dest)
        return dict(name=x['name'],bytes=dest.stat().st_size,sha256=base.sha(dest),generation=x['generation'])
    with ThreadPoolExecutor(6) as pool:
        return {k:list(pool.map(fetch,items)) for k,items in entries.items()}

def external():
    url='https://data.statmt.org/wmt18/translation-task/test.tgz';p=RAW/'wmt18.tgz';download(url,p)
    with tarfile.open(p) as z:
        cols=[]
        for suffix in ('src.zh','ref.en'):
            name=next(n for n in z.getnames() if n.endswith('/newstest2018-zhen-'+suffix+'.sgm'))
            doc=etree.fromstring(z.extractfile(name).read(),etree.XMLParser(recover=True,resolve_entities=False,no_network=True))
            cols.append({(d.get('docid'),s.get('id')):''.join(s.itertext()).strip() for d in doc.iter('doc') for s in d.iter('seg')})
    assert cols[0].keys()==cols[1].keys() and len(cols[0])>500
    ban=[set(),set()]
    for item in base.read(ORIGINAL/'sentences.json')+base.read(OLD/'test.json'):
        for j,k in enumerate(('source','target')):ban[j].add(key(item[k]))
    for r in annotations(SOURCE):
        for j,k in enumerate(('source','target')):ban[j].add(key(clean(r[k])))
    # Track only matches against candidate surfaces while streaming the million previous training pairs.
    cand=[{key(x) for x in col.values()} for col in cols]
    for en,zh in opus_rows('train'):
        for j,s in enumerate((zh,en)):
            k=key(s)
            if k in cand[j]:ban[j].add(k)
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    inv=list(map(set,base.read(base.BASE/'inventories.json')))
    panel=[];counts=[0,0]
    for d,i in sorted(cols[0],key=lambda k:digest('wmt18-762035'+'/'.join(k))):
        a,b=[c[d,i] for c in cols]
        if not (1<=len(a)<=96 and 1<=len(b)<=96) or key(a) in ban[0] or key(b) in ban[1]:continue
        src,tgt=face.encode(a),face.encode(b)
        if not any('\u3400'<=c<='\u9fff' for c in a) or face.bad.intersection(a+b) or not set(tgt)<=inv[0]:continue
        novel=bool(set(src)-inv[1])
        if counts[novel]>=64:continue
        panel.append(dict(doc=d,segment=i,source=a,target=b,open=novel));counts[novel]+=1
        ban[0].add(key(a));ban[1].add(key(b))
    assert counts==[64,64]
    votes=np.load(CERT/'confirmation.npz')
    groups=base.read(CERT_WORK/'families.json')
    verified=(votes['votes']==np.array([1,2])[None,None,:,None]).all(axis=(1,2,3))
    assert verified.sum()==88 and base.sha(CERT/'confirmation.npz')==base.read(CERT/'confirmation.json')['votes']
    edits=[dict(family=int(i),source=g[1],target=g[0],side=s) for i in votes['ids'][verified] for s,g in enumerate(groups[i])]
    return panel,edits,base.sha(p)

def windows(text):
    text=unicodedata.normalize('NFKC',text).translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"','—':'-','–':'-','…':'...','_':''}))
    for para in re.split(r'\n\s*\n',text):
        words=para.split()
        if len(words)<8:continue
        chunks=[];size=0
        for w in words:
            if size+len(w)+1>120 and chunks:
                yield ' '.join(chunks),False
                chunks=[];size=0
            chunks.append(w);size+=len(w)+int(len(chunks)>1)
        if chunks:yield ' '.join(chunks),True

def main():
    PUB.mkdir(exist_ok=True);WORK.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    if (PUB/'data.json').exists():raise SystemExit('Dataset already reserved')
    books=reserve_books()
    news,edits,wmt=external()
    base.dump(WORK/'panel.json',news);base.dump(WORK/'edits.json',edits)
    face=base.Ink(np.load(base.BASE/'palette.npy'));inv=set(base.read(base.BASE/'inventories.json')[0])
    glyph={chr(i):face.encode(chr(i))[0] for i in range(32,127)}
    glyph={k:v for k,v in glyph.items() if v in inv and k not in face.bad}
    banned={key(r['target']) for r in news+edits}
    seen=set();hist={};files={}
    for split in ('validation','train'):
        rows=[];conditions=[];book_ids=[];ending=[]
        for ix,book in enumerate(books[split]):
            rng=np.random.default_rng(int(book['sha256'][:12],16))
            pool=[]
            for text,stop in windows((RAW/book['name']).read_text(encoding='utf-8-sig')):
                if 64<=len(text)<=120 and set(text)<=glyph.keys() and sum(c.isalpha() for c in text)>.63*len(text) and len(LEX.findall(text))>=7:
                    pool.append((text,stop))
            count=0
            for j in rng.permutation(len(pool)):
                text,stop=pool[j];matches=LEX.findall(text)
                perm=iter(rng.permutation(matches));other=LEX.sub(lambda _:next(perm),text)
                sigs={key(text),key(other)}
                if len(sigs)!=2 or seen.intersection(sigs) or banned.intersection(sigs):continue
                seen.update(sigs);assert sorted(text)==sorted(other)
                for out,s in ((rows,text),(conditions,other)):
                    cells=[glyph[c] for c in s]+([1] if stop else [])
                    out.append(cells+[0]*(121-len(cells)))
                book_ids.append(ix);ending.append(stop);count+=1
                if count==(512 if split=='train' else 96):break
            book['windows']=count
        p=WORK/(split+'.npz')
        a=np.array(rows,dtype=np.int16);b=np.array(conditions,dtype=np.int16)
        assert np.array_equal(np.sort(a,axis=1),np.sort(b,axis=1))
        np.savez_compressed(p,intact=a,scrambled=b,book=np.array(book_ids),stop=np.array(ending))
        hist[split]=dict(windows=len(a),supervised=int(np.count_nonzero(a)),books=len(set(book_ids)))
        files[p.name]=base.sha(p)
    for name in ('panel.json','edits.json'):files[name]=base.sha(WORK/name)
    base.dump(PUB/'data.json',dict(protocol=base.sha(RULE),books=books,counts=hist,files=files,wmt2018=wmt,
              wmt_documents=len({r['doc'] for r in news}),edits=len(edits)//2))
    print(hist,'news',len(news),'edits',len(edits),flush=True)

if __name__=='__main__':main()
