"""Freeze paired whole-book splits, pictured word caches and natural prediction targets."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import base64
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
import requests

from scripts.evaluate_book_page_memory import normalize, RAW, REV, ARCHIVE
from scripts.prepare_book_order_prior import base, ROOT, GCS, download

PUB=ROOT/'references/evidence/passage_selector_v1'
WORK=ROOT/'artifacts/passage_pointer_v1'
MODEL=ROOT/'models/optical_passage_selector'
RULE=ROOT/'references/optical_passage_selector_protocol.md'
SHARD=ROOT/'data/raw/passage-pointer/lambada-train.parquet'
SHARD_HASH='903493579c4f687f8d01b4270eac6e3dd50593ea4c58561137b13e772311ff0c'
WORD=re.compile('[A-Za-z]+')
SEED=862173
WIDTH=96


def gap(s):
    for j,marks in enumerate(('.!?','"\'',';',',',':','-'),1):
        if any(c in marks for c in s):return j
    return 7 if s.strip() else 0


def lex(s):
    found=list(WORD.finditer(s))
    words=[m[0] for m in found]
    gaps=np.array([gap(s[m.end():found[i+1].start() if i+1<len(found) else len(s)]) for i,m in enumerate(found)],dtype=np.uint8)
    return words,gaps


def sig(words):
    return sha256(' '.join(words[-17:]).lower().encode()).digest()[:16]


def fresh_pg():
    url='https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o'
    r=requests.get(url,params={'prefix':'test/','maxResults':1000,'fields':'items(name,size,generation,md5Hash)'},timeout=45)
    r.raise_for_status()
    opts=[b for b in r.json()['items'] if b['name'].endswith('.txt') and 80000<=int(b['size'])<=950000]
    pool=sorted(opts,key=lambda b:sha256(('862173/'+b['name']).encode()).digest())[:24]
    assert len(pool)==24
    def get(b):
        file=RAW/b['name'];download(GCS+b['name']+'?generation='+b['generation'],file)
        assert base64.b64encode(hashlib.md5(file.read_bytes()).digest()).decode()==b['md5Hash']
        return dict(name=b['name'],sha=base.sha(file),generation=b['generation'])
    with ThreadPoolExecutor(5) as workers:return list(workers.map(get,pool))


def fiction():
    source=pq.ParquetFile(SHARD)
    i=0
    for batch in source.iter_batches(batch_size=12,columns=['text','domain']):
        for v in batch.to_pylist():
            yield i,v['text'],v['domain']
            i+=1


def main():
    if (PUB/'data.json').exists():raise SystemExit('Frozen data already present')
    PUB.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if not SHARD.exists():
        download(f'https://huggingface.co/datasets/cimec/lambada/resolve/{REV}/plain_text/train-00000-of-00002.parquet',SHARD,SHARD_HASH)
    assert base.sha(SHARD)==SHARD_HASH
    pgtrain=base.read(ROOT/'references/evidence/book_order_prior_v1/data.json')['books']['train']
    pgtest=fresh_pg()
    papers={};ids={}
    for i,s,domain in fiction():
        h=sha256(s.encode()).hexdigest()
        if h not in papers:
            papers[h]=dict(index=i,sha=h,domain=domain);ids[i]=h
    order=sorted(papers,key=lambda h:sha256(('862173/'+h).encode()).digest())
    assignment={papers[h]['index']:('fiction' if j<96 else 'train') for j,h in enumerate(order[:864])}
    assert len(assignment)==864
    seen=set();vocab={'':0};forms=['']
    def handles(words):
        result=[]
        for w in words:
            if w not in vocab:vocab[w]=len(forms);forms.append(w)
            result.append(vocab[w])
        return result
    def reserve(inputs,target,future,task,end,group,seps,stops):
        source=np.zeros(WIDTH,np.int32);marks=np.zeros(WIDTH,np.uint8)
        source[-len(inputs):]=handles(inputs);marks[-len(inputs):]=seps
        ys=(handles([target,*future])+[0,0,0])[:3]
        return source,marks,ys,task,end,group,stops
    panels={k:[] for k in ('train','fiction','gutenberg','lambada')}
    books={k:[] for k in ('train','fiction','gutenberg')}
    for line in (RAW/'lambada-test.jsonl').read_text().splitlines():
        seen.add(sig(WORD.findall(normalize(json.loads(line)['text']))))
    text=pq.read_table(RAW/'lambada-validation.parquet',columns=['text'])['text'].to_pylist()
    with ZipFile(ARCHIVE) as z:indices=[r['index'] for r in json.loads(z.read('records.json'))]
    for i,s in enumerate(text):
        seen.add(sig(WORD.findall(normalize(s))))
    for i in indices:
        s,y=normalize(text[i]).rsplit(' ',1);words,gaps=lex(s+' ')
        assert words
        panels['lambada'].append(reserve(words[-WIDTH:],y,[],1,1,i,gaps[-WIDTH:],[1,0,0]))
    df=Counter()
    def book(s,meta,split,limit):
        h=meta['sha']
        words,gaps=lex(normalize(s))
        if split=='train' and meta.get('pg'):
            df.update(set(w.lower() for w in re.findall('[A-Za-z]+',s)))
        ends=gaps==1
        rng=np.random.default_rng(int(h[:14],16)^SEED)
        group=len(books[split]);taken=[0,0]
        for task in (1,0):
            candidates=np.arange(96,len(words)-2)
            if task:candidates=candidates[ends[candidates]]
            for j in rng.permutation(candidates):
                if len(words[j])>23:continue
                signature=sig(words[j-16:j+1])
                if signature in seen:continue
                seen.add(signature)
                n=int(rng.integers(26,WIDTH+1))
                tail=words[j+1:j+3]
                panels[split].append(reserve(words[j-n:j],words[j],tail,task,int(ends[j]),group,gaps[j-n:j],ends[j:j+3].tolist()))
                taken[task]+=1
                if taken[task]>=limit//2:break
        books[split].append(dict(**meta,ordinary=taken[0],final=taken[1]))
    for b in pgtest:book((RAW/b['name']).read_text(encoding='utf-8-sig'),dict(**b,pg=True),'gutenberg',96)
    # Commit all challenge data before selecting a single training surface.
    for split in ('fiction','train'):
        for i,s,_ in fiction():
            if assignment.get(i)==split:book(s,papers[ids[i]],split,96 if split=='fiction' else 256)
        if split=='fiction':print('reserved held fiction',len(panels[split]),flush=True)
    for b in pgtrain:
        assert base.sha(RAW/b['name'])==b['sha256']
        book((RAW/b['name']).read_text(encoding='utf-8-sig'),dict(name=b['name'],pg=True,sha=b['sha256']),'train',512)
    files={};counts={}
    for split,rows in panels.items():
        source,marks,ys,task,end,group,ends=zip(*rows)
        path=WORK/(split+'.npz')
        np.savez_compressed(path,source=np.stack(source),marks=np.stack(marks),target=np.array(ys,np.int32),
            task=np.array(task,np.uint8),end=np.array(end,np.uint8),group=np.array(group,np.int32),ends=np.array(ends,np.uint8))
        files[path.name]=base.sha(path);counts[split]=[len(rows),sum(task)]
    base.dump(WORK/'forms.json',forms);files['forms.json']=base.sha(WORK/'forms.json')
    MODEL.mkdir(parents=True,exist_ok=True);base.dump(MODEL/'document_frequency.json',df)
    base.dump(PUB/'data.json',dict(books=books,counts=counts,files=files,forms=len(forms),prior_df=base.sha(MODEL/'document_frequency.json'),
        protocol=base.sha(RULE),source=SHARD_HASH,seed=SEED,script=base.sha(Path(__file__))))
    print(counts,'vocabulary cache',len(forms),flush=True)


if __name__=='__main__':main()
