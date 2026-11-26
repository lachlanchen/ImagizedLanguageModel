"""Archive and assess matched open-ink writers on the reserved WMT 2017 news cohort."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import sqlite3
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from scripts.train_open_news_writer import PUB,WORK,ROOT,MODEL,base,restore,admitted,gpu_owner,configure_cuda_cli
from scripts.measure_paired_visual_writer import png,unpack
from scripts.calibrate_translation_graders import QWEN,STORE,MODELS,ONE,BINARY,chat,server
from ilm.visual_lm.open_source_cells import OpenSourceCells

ARCHIVE=ROOT/'publication/ilm-nature/evidence/reviewed_news_writers.zip'
MODES=('parent','hidden','ink','ink_erased')


def generate():
    panel=base.read(WORK/'test.json')
    nets={k:restore(k,'cuda') for k in MODES[:3]}
    routes={k:OpenSourceCells(v,0) for k,v in nets.items()}
    face=base.Ink();codes=[[face.encode(r[k]) for k in ('source','target')] for r in panel]
    assert not face.bad
    tiles=face.array();names=base.read(base.BASE/'readback.json')
    rows=[]
    tmp=ARCHIVE.with_suffix('.partial')
    with ZipFile(tmp,'w',ZIP_DEFLATED) as z:
        for i,(r,(source,target)) in enumerate(zip(panel,codes,strict=True)):
            image=tiles[source];ref=tiles[target]
            seen=all(c.tobytes() in routes['ink'].known for c in image)
            assert seen != r['open']
            row=dict(index=i,doc=r['doc'],open=r['open'],answers={},cells=len(image))
            for k,material in [('source',image),('reference',ref)]:
                blob=png(material);z.writestr(f'{i}/{k}.png',blob)
                row[k]=sha256(blob).hexdigest()
            for mode in MODES:
                arm='ink' if mode=='ink_erased' else mode
                route=routes[arm]
                out,stop=route.translate(image,hide_unfamiliar=mode in ('hidden','ink_erased'))
                ids=nets[arm].addresses(out)
                blob=png(out);z.writestr(f'{i}/{mode}.png',blob)
                row['answers'][mode]=dict(text=''.join(names[str(k)] for k in ids),stop=bool(stop),
                    sha256=sha256(blob).hexdigest(),cells=len(out),exact=bool(stop and np.array_equal(out,ref)))
            if seen:assert row['answers']['ink']==row['answers']['ink_erased']
            rows.append(row)
        z.writestr('records.json',json.dumps(rows,ensure_ascii=False))
    tmp.replace(ARCHIVE)
    # The independent reproduction uses one item and a fresh source-image decode.
    replay={}
    with ZipFile(ARCHIVE) as z:
        for mode in MODES:
            arm='ink' if mode=='ink_erased' else mode
            route=OpenSourceCells(restore(arm,'cpu'),0)
            replay[mode]=[]
            for i in [j for flag in (False,True) for j in [k for k,r in enumerate(rows) if r['open']==flag][:4]]:
                row=rows[i]; material=unpack(z.read(f'{i}/source.png'),row['cells'])
                out,stop=route.translate(material,hide_unfamiliar=mode in ('hidden','ink_erased'))
                replay[mode].append(dict(index=i,same=png(out)==z.read(f'{i}/{mode}.png') and bool(stop)==row['answers'][mode]['stop']))
    base.dump(PUB/'generation.json',dict(panel=base.sha(WORK/'test.json'),archive=base.sha(ARCHIVE),
              fits={s:base.sha(MODEL/s/'fit.json') for s in ('hidden','ink')},replay=replay))


def assess():
    panel=base.read(WORK/'test.json')
    gen=base.read(PUB/'generation.json')
    assert base.sha(ARCHIVE)==gen['archive'] and gen['panel']==base.sha(WORK/'test.json')
    assert base.sha(STORE/'blobs'/('sha256-'+MODELS[QWEN]))==MODELS[QWEN]
    palette=np.load(base.BASE/'palette.npy');names=base.read(base.BASE/'readback.json')
    letters={c.tobytes():names[str(i)] for i,c in enumerate(palette) if str(i) in names}
    labels=(*MODES,'reference','foil')
    surfaces={}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,(row,r) in enumerate(zip(rows,panel,strict=True)):
            for k in MODES:
                blob=z.read(f'{i}/{k}.png');answer=row['answers'][k]
                assert sha256(blob).hexdigest()==answer['sha256']
                txt=''.join(letters[c.tobytes()] for c in unpack(blob,answer['cells']))
                assert txt==answer['text']
                surfaces[i,k]=txt
            surfaces[i,'reference']=r['target'];surfaces[i,'foil']=panel[(i+len(panel)//2+1)%len(panel)]['target']
    jobs=sorted(set((i,v) for (i,_),v in surfaces.items()))
    binding=sha256((gen['archive']+gen['panel']+ONE+json.dumps(BINARY)+MODELS[QWEN]).encode()).hexdigest()
    with sqlite3.connect(WORK/'votes.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS votes(binding TEXT,i INT,text TEXT,v INT,PRIMARY KEY(binding,i,text))')
        votes={(i,t):bool(v) for i,t,v in db.execute('SELECT i,text,v FROM votes WHERE binding=?',(binding,))}
        needed=[k for k in jobs if k not in votes]
        with server() as url,ThreadPoolExecutor(4) as workers:
            def ask(job):
                i,s=job
                res=chat(url,QWEN,ONE+json.dumps(dict(source=panel[i]['source'],target=s),ensure_ascii=False),BINARY)
                assert res['stop']
                vote=json.loads(res['text'])['faithful']
                assert type(vote) is bool
                return vote
            for j in range(0,len(needed),32):
                part=needed[j:j+32]
                for job,v in zip(part,workers.map(ask,part),strict=True):
                    db.execute('INSERT INTO votes VALUES(?,?,?,?)',(binding,*job,int(v)));votes[job]=v
                db.commit()
    grades=[{k:bool(votes[i,surfaces[i,k]] and (k not in MODES or row['answers'][k]['stop'])) for k in labels} for i,row in enumerate(rows)]
    totals={}; intervals={}; exact={};stopped={}
    for name,flag in [('unfamiliar',True),('familiar',False)]:
        ix=[i for i,r in enumerate(rows) if r['open']==flag]
        totals[name]={k:sum(grades[i][k] for i in ix) for k in labels}
        exact[name]={k:sum(rows[i]['answers'][k]['exact'] for i in ix) for k in MODES}
        stopped[name]={k:sum(rows[i]['answers'][k]['stop'] for i in ix) for k in MODES}
        docs,group=np.unique([rows[i]['doc'] for i in ix],return_inverse=True)
        draws=np.random.default_rng(189211).integers(len(docs),size=(10000,len(docs)))
        sizes=np.bincount(group)[draws].sum(1);intervals[name]={}
        for a,b in [('ink','hidden'),('ink','ink_erased'),('ink','parent'),('hidden','parent')]:
            diff=[int(grades[i][a])-int(grades[i][b]) for i in ix]
            means=np.bincount(group,weights=diff)[draws].sum(1)/sizes
            intervals[name][a+'-'+b]=np.quantile(means,[.025,.975]).tolist()
    passed=all(intervals['unfamiliar'][k][0]>0 for k in ('ink-hidden','ink-ink_erased')) and totals['familiar']['ink']>=totals['familiar']['parent']
    base.dump(PUB/'meaning.json',dict(archive=gen['archive'],binding=binding,grades=grades,totals=totals,
              exact=exact,stopped=stopped,intervals=intervals,passed=bool(passed),unique_votes=len(jobs)))
    print(totals,'passed',passed,flush=True)


if __name__=='__main__':
    admitted();configure_cuda_cli()
    with gpu_owner():
        if not (PUB/'generation.json').exists():generate()
        if not (PUB/'meaning.json').exists():assess()
