"""Free generation and two-order grading for the paired corpus-diversity comparison."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from scripts.prepare_visual_training_breadth import ROOT,WORK,PUB,MODEL,base
from scripts.train_visual_training_breadth import restore,gpu_owner
from scripts.write_covered_phrases import CoveredWriter
from scripts.evaluate_retrieved_glyphs import ARCHIVE as PRIOR, PUB as PRIOR_PUB
from scripts.evaluate_glyph_draft_editor import png,blind_votes,server,chat,JUDGE,QWEN,STORE,MODELS
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE = ROOT/'publication/ilm-nature/evidence/visual_training_breadth.zip'
ARMS = ('parent','narrow','broad','exchanged')


def cases():
    assert base.sha(WORK/'panel.json')==base.read(PUB/'selection.json')['files']['panel.json']
    panel=base.read(WORK/'panel.json')
    result=[]
    for i,r in enumerate(panel):
        donor=(i+12)%len(panel)
        assert panel[donor]['doc']!=r['doc']
        for t in (0,1):
            result.append(dict(t=t, doc=r['doc'], group='book', source=r['text'][1-t], ref=r['text'][t],
                               foil=panel[donor]['text'][t],donor=2*donor+t))
    assert base.sha(PRIOR)==base.read(PRIOR_PUB/'generation.json')['archive']
    with ZipFile(PRIOR) as z:
        old=json.loads(z.read('records.json'))[96:]
        result.extend(dict(r,answers={}) for r in old)
    return result,old


def generate():
    if (PUB/'generation.json').exists():
        assert base.sha(ARCHIVE)==base.read(PUB/'generation.json')['archive']
        return
    rows,old=cases()
    readers={m:restore(m,'cuda') for m in ARMS[:3]}
    net=readers['parent']
    face=base.Ink(net.palette)
    src=[face.encode(r['source']) for r in rows]
    refs=[face.encode(r['ref']) for r in rows]
    assert not face.bad
    names=base.read(base.BASE/'readback.json')
    partial=ARCHIVE.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as z:
        for i,(s,r) in enumerate(zip(src,rows,strict=True)):
            r['answers']={}
            z.writestr(f'{i:04d}/source.png',png(net.palette[s]))
        for t in (0,1):
            phrase=CoveredWriter(t)
            ids=[i for i,r in enumerate(rows) if r['t']==t]
            drafts={i:phrase.translate(net.palette[src[i]])[0] for i in ids}
            for mode in ARMS:
                reader=readers['broad' if mode=='exchanged' else mode]
                for j in range(0,len(ids),24):
                    group=ids[j:j+24]
                    inputs=[rows[i]['donor'] if mode=='exchanged' else i for i in group]
                    ans=reader.translate([net.palette[src[i]] for i in inputs],t,[drafts[i] for i in inputs])
                    for i,(pixels,stop) in zip(group,ans,strict=True):
                        cells=net.addresses(pixels)
                        blob=png(pixels)
                        digest=sha256(blob).hexdigest()
                        z.writestr(f'{i:04d}/{mode}.png',blob)
                        rows[i]['answers'][mode]=dict(text=''.join(names[str(g)] for g in cells),stop=bool(stop),
                                                     cells=len(cells),exact=bool(stop and cells==refs[i]),png=digest)
                        if mode=='parent' and i>=96:
                            assert digest==old[i-96]['answers']['parent']['png']
        z.writestr('records.json',json.dumps(rows,ensure_ascii=False))
    partial.replace(ARCHIVE)
    base.dump(PUB/'generation.json',dict(archive=base.sha(ARCHIVE),requests=len(rows),images=len(rows)*len(ARMS),
              fits={m:base.sha(MODEL/m/'fit.json') for m in ('narrow','broad')}))


def assess():
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==r['answers'][m]['png']
    book=rows[:96]
    cache=WORK/'comparator.json'
    for m in (JUDGE,QWEN):
        assert base.sha(STORE/'blobs'/('sha256-'+MODELS[m]))==MODELS[m]
    if not cache.exists():
        with server() as url,ThreadPoolExecutor(4) as workers:
            def translate(r):
                return chat(url,QWEN,'Translate into '+('English','Mandarin Chinese')[r['t']]+'. Return only the translation.\n'+r['source'])
            responses=list(workers.map(translate,book))
        base.dump(cache,dict(archive=base.sha(ARCHIVE),values=responses))
    saved=base.read(cache)
    assert saved['archive']==base.sha(ARCHIVE)
    answers=[dict(r['answers'],qwen=q,reference=dict(text=r['ref'],stop=True),foil=dict(text=r['foil'],stop=True))
             for r,q in zip(book,saved['values'],strict=True)]
    votes=blind_votes(book,answers)
    counts=defaultdict(Counter)
    grades=[]
    for i,(r,cands) in enumerate(zip(book,answers,strict=True)):
        flags={m:bool(a['stop'] and votes[2*i][m] and votes[2*i+1][m]) for m,a in cands.items()}
        grades.append(flags)
        for group in (r['doc'],('en','zh')[r['t']],'all'):
            counts[group].update(flags)
            counts[group]['n']+=1
    intervals={}
    for other in ('parent','narrow'):
        diff=[(counts[w]['broad']-counts[w][other])/24 for w in dict.fromkeys(r['doc'] for r in book)]
        intervals[other]=np.quantile([np.mean(v) for v in product(diff,repeat=4)],[.025,.975]).tolist()
    roles={}
    for m in ARMS:
        exact=[r['answers'][m]['exact'] for r in rows[96:]]
        roles[m]=dict(exact=sum(exact),polarity=sum(all(exact[i:i+2]) for i in range(0,472,2)),
                      crossed=sum(all(exact[i:i+4]) for i in range(0,472,4)))
    base.dump(PUB/'assessment.json',dict(archive=base.sha(ARCHIVE),counts=counts,intervals=intervals,roles=roles,
              stops={m:sum(r['answers'][m]['stop'] for r in rows) for m in ARMS},grades=grades,votes=votes,
              comparator=saved['values'],models={k:MODELS[k] for k in (JUDGE,QWEN)}))
    print(dict(counts['all']),roles,intervals,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/'assessment.json').exists():
            assess()
        assert base.read(PUB/'assessment.json')['archive']==base.sha(ARCHIVE)
