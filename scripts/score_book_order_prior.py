"""Score archived visual answers, checking original ink before semantic assessment."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import sqlite3
from zipfile import ZipFile

import numpy as np
from scripts.evaluate_book_order_prior import ANSWERS, ARMS, PUB, WORK, base, unpack, gpu_owner, configure_cuda_cli
from scripts.calibrate_translation_graders import QWEN, STORE, MODELS, ONE, BINARY, chat, server
from scripts.certify_opus_paired_supervision import INSTRUCTION, SCHEMA, CHOICES


def interval(difference, clusters):
    _, ix = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(ix)
    draws = np.random.default_rng(762035).integers(len(sizes),size=(10000,len(sizes)))
    means = np.bincount(ix,weights=difference)[draws].sum(1)/sizes[draws].sum(1)
    return np.quantile(means,[.025,.975]).tolist()


def score():
    generation = base.read(PUB/'generation.json')
    assert base.sha(ANSWERS)==generation['archive'] and generation['data']==base.sha(PUB/'data.json')
    names = base.read(base.BASE/'readback.json')
    letters = {c.tobytes():names[str(i)] for i,c in enumerate(np.load(base.BASE/'palette.npy')) if str(i) in names}
    with ZipFile(ANSWERS) as archive:
        rows=json.loads(archive.read('records.json'))
        for i,row in enumerate(rows):
            for m,a in row['answers'].items():
                blob=archive.read(f'{i}/{m}.png')
                assert sha256(blob).hexdigest()==a['sha256']
                assert ''.join(letters[c.tobytes()] for c in unpack(blob,a['cells']))==a['text']
    assert base.sha(STORE/'blobs'/('sha256-'+MODELS[QWEN]))==MODELS[QWEN]
    queries={}
    jobs={}
    def bind(key, method, packet):
        payload=json.dumps(packet,ensure_ascii=False)
        h=sha256((method+payload).encode()).hexdigest()
        queries[h]=(method,payload)
        jobs[key]=h
    news=[r for r in rows if r['group']=='news']
    edits=[r for r in rows if r['group']=='edit']
    assert len(news)==128 and len(edits)==176
    for i,r in enumerate(news):
        candidates={m:r['answers'][m]['text'] for m in ARMS}
        candidates.update(reference=r['target'],foil=news[(i+65)%128]['target'])
        for name,t in candidates.items():
            bind(('news',i,name),'single',dict(source=r['source'],target=t))
    for i in range(0,len(edits),2):
        pair=edits[i:i+2]
        assert pair[0]['family']==pair[1]['family'] and [r['side'] for r in pair]==[0,1]
        for side,r in enumerate(pair):
            for arm in ARMS:
                for rev in (0,1):
                    bind(('edit',i+side,arm,rev),'pair',dict(source=r['source'],target='English',
                        A=pair[rev]['answers'][arm]['text'],B=pair[1-rev]['answers'][arm]['text']))
    binding=sha256((generation['archive']+ONE+INSTRUCTION+json.dumps([BINARY,SCHEMA])+MODELS[QWEN]).encode()).hexdigest()
    with sqlite3.connect(WORK/'semantic.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS grades(binding TEXT, hash TEXT, value TEXT, PRIMARY KEY(binding,hash))')
        votes=dict(db.execute('SELECT hash,value FROM grades WHERE binding=?',(binding,)))
        missing=sorted(set(queries)-set(votes))
        with gpu_owner(),server() as url,ThreadPoolExecutor(4) as pool:
            def vote(h):
                method,payload=queries[h]
                single=method=='single'
                response=chat(url,QWEN,(ONE if single else INSTRUCTION)+payload,BINARY if single else SCHEMA)
                assert response['stop']
                result=json.loads(response['text'])
                if single:
                    assert type(result['faithful']) is bool
                    return 'yes' if result['faithful'] else 'no'
                assert result['choice'] in CHOICES
                return result['choice']
            for j in range(0,len(missing),32):
                for h,v in zip(missing[j:j+32],pool.map(vote,missing[j:j+32]),strict=True):
                    votes[h]=v
                    db.execute('INSERT INTO grades VALUES (?,?,?)',(binding,h,v))
                db.commit()
                if j%256==0:print('assessed',len(votes),'of',len(queries),flush=True)
    grades=[{m: votes[jobs['news',i,m]]=='yes' and (m not in ARMS or r['answers'][m]['stop'])
             for m in (*ARMS,'reference','foil')} for i,r in enumerate(news)]
    boundaries=[]
    for i,r in enumerate(edits):
        side=r['side']
        boundaries.append({m:[votes[jobs['edit',i,m,v]] for v in (0,1)] for m in ARMS})
    families=[{m: all(edits[i+s]['answers'][m]['stop'] and
        boundaries[i+s][m]==(['A','B'] if s==0 else ['B','A']) for s in (0,1)) for m in ARMS}
        for i in range(0,len(edits),2)]
    totals={}
    ci={}
    for name,sel in [('news',list(range(128))),('familiar',[i for i,r in enumerate(news) if not r['open']]),
                     ('unfamiliar',[i for i,r in enumerate(news) if r['open']])]:
        totals[name]={m:sum(grades[i][m] for i in sel) for m in grades[0]}
        ci[name]=interval([int(grades[i]['intact'])-int(grades[i]['scrambled']) for i in sel],[news[i]['doc'] for i in sel])
    totals['edits']={m:sum(f[m] for f in families) for m in ARMS}
    ci['edits']=interval([int(f['intact'])-int(f['scrambled']) for f in families],np.arange(len(families)))
    retention={}
    for name in ('roles','reversed_roles'):
        retention[name]={}
        for m in ARMS:
            hit=np.array([r['answers'][m]['exact'] for r in rows if r['group']==name])
            retention[name][m]=dict(exact=int(hit.sum()),polarity=int(hit.reshape(-1,2).all(1).sum()),roles=int(hit.reshape(-1,4).all(1).sum()))
    passed=all(ci[k][0]>0 and totals[k]['intact']>totals[k]['baseline'] for k in ('news','edits')) and all(
        r['intact'][k]>=r['baseline'][k] for r in retention.values() for k in ('roles','polarity'))
    base.dump(PUB/'meaning.json',dict(archive=generation['archive'],binding=binding,judge=MODELS[QWEN],news=grades,
        edit_votes=boundaries,edit_families=families,totals=totals,intervals=ci,retention=retention,
        stopped={k:{m:sum(r['answers'][m]['stop'] for r in rows if r['group']==k) for m in ARMS} for k in ('news','edit','roles','reversed_roles')},
        passed=bool(passed),distinct_judgements=len(queries)))
    print(totals,retention,'passed',passed,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    if not (PUB/'meaning.json').exists():
        score()
