"""Compare autonomous related and random passage memory on previously unused bilingual editions."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
from time import monotonic
from zipfile import ZipFile,ZIP_DEFLATED
import json
import numpy as np
import torch
from scripts.prepare_retrieved_glyph_writer import ROOT,PUB,WORK,base,previous
from scripts.write_retrieved_glyphs import PassageWriter,PictureEditor
from scripts.evaluate_glyph_draft_editor import ARCHIVE as CONTROL,png,blind_votes,server,chat,QWEN,JUDGE,STORE,MODELS
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/'publication/ilm-nature/evidence/retrieved_glyph_writer.zip'
ARMS=('parent','related','random','exchanged')


def records():
    assert base.sha(WORK/'panel.json')==base.read(PUB/'selection.json')['panel']
    books=base.read(WORK/'panel.json')
    rows=[]
    for i,r in enumerate(books):
        donor=(i+12)%len(books)
        for t in (0,1):
            rows.append(dict(doc=r['doc'],source=r['text'][1-t],ref=r['text'][t],t=t,group='book',
                             foil=books[donor]['text'][t],donor=2*donor+t,answers={}))
    assert base.sha(CONTROL)==base.read(previous.PUB/'generation.json')['archive']
    with ZipFile(CONTROL) as z:old=json.loads(z.read('records.json'))[96:]
    for j,r in enumerate(old):rows.append(dict(r,answers={},donor=96+(j//4)*4+(j%4^2)))
    return rows,old


def generate():
    if (PUB/'generation.json').exists():
        assert base.sha(ARCHIVE)==base.read(PUB/'generation.json')['archive']
        return
    rows,old=records()
    net=PictureEditor('editor','cuda');face=base.Ink(net.palette)
    src=[face.encode(r['source']) for r in rows];ref=[face.encode(r['ref']) for r in rows]
    assert not face.bad
    strengths=base.read(PUB/'calibration.json')['chosen']
    names=base.read(base.BASE/'readback.json')
    partial=ARCHIVE.with_suffix('.partial')
    traces=[{} for _ in rows]
    times=defaultdict(float)
    with ZipFile(partial,'w',ZIP_DEFLATED) as z,ZipFile(CONTROL) as ctrl:
        def store(i,m,seq,stop,copy=False):
            blob=png(net.palette[seq]);z.writestr(f'{i:04d}/{m}.png',blob)
            rows[i]['answers'][m]=dict(png=sha256(blob).hexdigest(),text=''.join(names[str(g)] for g in seq),
                            cells=len(seq),stop=bool(stop),exact=bool(stop and seq==ref[i]),complete_neighbor=copy)
        for i,x in enumerate(src):
            blob=png(net.palette[x])
            if i>=96:assert blob==ctrl.read(f'{i:04d}/source.png')
            z.writestr(f'{i:04d}/source.png',blob)
        for t in (0,1):
            w=PassageWriter(t,'cuda',net)
            times['index']+=w.index_seconds
            ids=[i for i,r in enumerate(rows) if r['t']==t]
            begin=monotonic()
            ds={i:net.addresses(w.phrase.translate(net.palette[src[i]])[0]) for i in ids}
            times['phrase']+=monotonic()-begin
            for j in range(0,len(ids),24):
                batch=ids[j:j+24]
                for mode in ARMS:
                    before=monotonic()
                    source_ids=[rows[i]['donor'] if mode=='exchanged' else i for i in batch]
                    sources=[src[i] for i in source_ids]
                    drafts=[ds[i] for i in source_ids]
                    if mode=='parent':
                        memory=[None]*len(batch)
                    else:
                        memory=[w.support(s,'random' if mode=='random' else 'related') for s in sources]
                    times[mode+'_support']+=monotonic()-before
                    step=monotonic()
                    outputs=w.write(sources,drafts,memory,0. if mode=='parent' else strengths[t])
                    times[mode+'_decode']+=monotonic()-step
                    for i,support,(seq,stop) in zip(batch,memory,outputs,strict=True):
                        copy=support is not None and seq in support.targets
                        store(i,mode,seq,stop,copy)
                        if support is not None:traces[i][mode]=support.ids
                        if i>=96 and mode=='parent':
                            assert rows[i]['answers'][mode]['png']==old[i-96]['answers']['editor']['png']
        z.writestr('records.json',json.dumps(rows,ensure_ascii=False))
        z.writestr('supports.json',json.dumps(traces))
    partial.replace(ARCHIVE)
    base.dump(PUB/'generation.json',dict(archive=base.sha(ARCHIVE),requests=len(rows),outputs=len(rows)*len(ARMS),
              calibration=base.sha(PUB/'calibration.json'),times=times))


def assess():
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==r['answers'][m]['png']
    books=rows[:96]
    for m in (QWEN,JUDGE):assert base.sha(STORE/'blobs'/('sha256-'+MODELS[m]))==MODELS[m]
    cache=WORK/'qwen.json'
    if not cache.exists():
        with server() as url,ThreadPoolExecutor(max_workers=4) as pool:
            def convert(r):
                return chat(url,QWEN,'Translate into '+('English','Mandarin Chinese')[r['t']]+'. Return only the translation.\n'+r['source'])
            q=list(pool.map(convert,books))
        base.dump(cache,dict(archive=base.sha(ARCHIVE),rows=q))
    saved=base.read(cache)
    assert saved['archive']==base.sha(ARCHIVE)
    resp=[dict(r['answers'],qwen=v,reference=dict(text=r['ref'],stop=True),foil=dict(text=r['foil'],stop=True))
          for r,v in zip(books,saved['rows'],strict=True)]
    votes=blind_votes(books,resp)
    counts=defaultdict(Counter);grades=[]
    for i,r in enumerate(books):
        g={m:bool(a['stop'] and votes[2*i][m] and votes[2*i+1][m]) for m,a in resp[i].items()}
        for group in (r['doc'],('en','zh')[r['t']],'all'):
            counts[group].update(g);counts[group]['n']+=1
        grades.append(g)
    delta=[(counts[w]['related']-counts[w]['parent'])/counts[w]['n'] for w in dict.fromkeys(r['doc'] for r in books)]
    ci=np.quantile([np.mean(p) for p in product(delta,repeat=4)],[.025,.975]).tolist()
    roles={}
    for m in ARMS:
        a=[r['answers'][m]['exact'] for r in rows[96:]]
        roles[m]=dict(exact=sum(a),polarity=sum(all(a[i:i+2]) for i in range(0,472,2)),
                      crossed=sum(all(a[i:i+4]) for i in range(0,472,4)))
    base.dump(PUB/'assessment.json',dict(counts=counts,grades=grades,votes=votes,qwen=saved['rows'],roles=roles,work_interval=ci,
              archive=base.sha(ARCHIVE),stops={m:sum(r['answers'][m]['stop'] for r in rows) for m in ARMS},
              same_as_parent={m:sum(r['answers'][m]['png']==r['answers']['parent']['png'] for r in books) for m in ARMS},
              exact_neighbor={m:sum(r['answers'][m]['complete_neighbor'] for r in books) for m in ARMS}))
    print(counts['all'],roles,ci,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        assess()
