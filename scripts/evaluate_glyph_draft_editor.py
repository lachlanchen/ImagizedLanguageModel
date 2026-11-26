"""Evaluate correct and exchanged visual drafts on held-out works and controlled events."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
from zipfile import ZipFile,ZIP_DEFLATED
import json
import numpy as np
from scripts.prepare_glyph_draft_editor import ROOT,PUB,WORK,MODEL,base,parent
from scripts.evaluate_covered_phrases import ARCHIVE as PREVIOUS,png
from scripts.evaluate_proposition_translation import chat,server,QWEN,JUDGE,STORE,MODELS
from scripts.evaluate_visual_library import blind_votes
from scripts.write_full_image_roles import PicturedTranslator
from scripts.write_glyph_draft_editor import PictureEditor
from scripts.write_covered_phrases import CoveredWriter
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/'publication/ilm-nature/evidence/glyph_draft_editor.zip'
ARMS=('neural','phrase','matched','editor','exchange')


def generate():
    if (PUB/'generation.json').exists():
        assert base.read(PUB/'generation.json')['archive']==base.sha(ARCHIVE)
        return
    assert base.sha(WORK/'panel.json')==base.read(PUB/'selection.json')['hashes']['panel.json']
    books=base.read(WORK/'panel.json')
    panel=[]
    for i,r in enumerate(books):
        other=books[(i+12)%len(books)]
        assert other['doc']!=r['doc']
        for t in (0,1):
            panel.append(dict(group='book',id=i,doc=r['doc'],source=r['text'][1-t],ref=r['text'][t],
                              t=t,foil=other['text'][t],donor=2*((i+12)%len(books))+t,answers={}))
    assert base.sha(PREVIOUS)==base.read(parent.PUB/'generation.json')['archive']
    with ZipFile(PREVIOUS) as z:
        frozen=json.loads(z.read('records.json'))[96:]
    for j,r in enumerate(frozen):
        panel.append(dict(r,answers={},donor=96+(j+4)%len(frozen)))
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    inputs=[face.encode(r['source']) for r in panel]
    refs=[face.encode(r['ref']) for r in panel]
    assert not face.bad
    palette=face.array()
    sources=[palette[x] for x in inputs]
    names=base.read(base.BASE/'readback.json')
    reverse={p.tobytes():names[str(i)] for i,p in enumerate(palette) if i>=2 and str(i) in names}
    old=PicturedTranslator('marked','cuda')
    editor=PictureEditor('editor','cuda')
    matched=PictureEditor('matched','cuda')
    partial=ARCHIVE.with_suffix('.partial')
    with ZipFile(PREVIOUS) as previous,ZipFile(partial,'w',ZIP_DEFLATED) as z:
        def record(i,m,output,stop):
            data=png(output)
            z.writestr(f'{i:04d}/{m}.png',data)
            panel[i]['answers'][m]=dict(text=''.join(reverse[p.tobytes()] for p in output),cells=len(output),stop=bool(stop),
                        exact=bool(stop and np.array_equal(output,palette[refs[i]])),png=sha256(data).hexdigest())
        for i,img in enumerate(sources):
            raw=png(img)
            if i>=96:
                assert raw==previous.read(f'{i:04d}/source.png')
            z.writestr(f'{i:04d}/source.png',raw)
        for t in (0,1):
            phrase=CoveredWriter(t)
            order=[i for i,r in enumerate(panel) if r['t']==t]
            drafts={i:phrase.translate(sources[i])[0] for i in order}
            for i in order:
                assert panel[i]['donor'] in drafts and panel[i]['source']!=panel[panel[i]['donor']]['source']
                record(i,'phrase',drafts[i],True)
                if i>=96:
                    assert panel[i]['answers']['phrase']['png']==frozen[i-96]['answers']['phrase']['png']
                    panel[i]['answers']['neural']=frozen[i-96]['answers']['neural']
                    z.writestr(f'{i:04d}/neural.png',previous.read(f'{i:04d}/neural.png'))
            for m,net in (('neural',old),('editor',editor),('exchange',editor),('matched',matched)):
                ids=[i for i in order if m!='neural' or i<96]
                for j in range(0,len(ids),24):
                    batch=ids[j:j+24]
                    imgs=[sources[i] for i in batch]
                    if m in ('editor','exchange'):
                        ds=[drafts[i if m=='editor' else panel[i]['donor']] for i in batch]
                        outputs=net.translate(imgs,t,ds)
                    else:
                        outputs=net.translate(imgs,t)
                    for i,(p,stop) in zip(batch,outputs,strict=True):
                        record(i,m,p,stop)
        z.writestr('records.json',json.dumps(panel,ensure_ascii=False))
    partial.replace(ARCHIVE)
    base.dump(PUB/'generation.json',dict(archive=base.sha(ARCHIVE),count=len(panel),outputs=len(panel)*len(ARMS),
                selection=base.sha(PUB/'selection.json'),fits={m:base.sha(MODEL/m/'fit.json') for m in ('editor','matched')}))


def assess():
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            for m in ARMS:
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==r['answers'][m]['png']
    book=rows[:96]
    for m in (QWEN,JUDGE):
        assert base.sha(STORE/'blobs'/('sha256-'+MODELS[m]))==MODELS[m]
    cache=WORK/'qwen.json'
    if not cache.exists():
        with server() as url,ThreadPoolExecutor(max_workers=4) as pool:
            def translation(r):
                return chat(url,QWEN,'Translate into '+('English','Mandarin Chinese')[r['t']]+'. Return only the translation.\n'+r['source'])
            q=list(pool.map(translation,book))
        base.dump(cache,dict(archive=base.sha(ARCHIVE),responses=q))
    q=base.read(cache)
    assert q['archive']==base.sha(ARCHIVE)
    responses=[dict(r['answers'],qwen=v,reference=dict(text=r['ref'],stop=True),foil=dict(text=r['foil'],stop=True))
               for r,v in zip(book,q['responses'],strict=True)]
    votes=blind_votes(book,responses)
    grades=[]
    counts=defaultdict(Counter)
    for i,r in enumerate(book):
        grade={m:bool(v['stop'] and votes[i*2][m] and votes[i*2+1][m]) for m,v in responses[i].items()}
        grades.append(grade)
        for group in (r['doc'],('en','zh')[r['t']],'all'):
            counts[group].update(grade)
            counts[group]['n']+=1
    intervals={}
    for m in ('neural','matched','exchange'):
        delta=[(counts[w]['editor']-counts[w][m])/counts[w]['n'] for w in dict.fromkeys(r['doc'] for r in book)]
        intervals[m]=np.quantile([np.mean(p) for p in product(delta,repeat=4)],[.025,.975]).tolist()
    roles={}
    for m in ARMS:
        exact=[r['answers'][m]['exact'] for r in rows[96:]]
        roles[m]=dict(exact=sum(exact),polarity=sum(all(exact[i:i+2]) for i in range(0,len(exact),2)),
                      crossed=sum(all(exact[i:i+4]) for i in range(0,len(exact),4)))
    same={m:sum(r['answers']['editor']['png']==r['answers'][m]['png'] for r in book) for m in ARMS}
    base.dump(PUB/'assessment.json',dict(counts=counts,grades=grades,votes=votes,roles=roles,intervals=intervals,
             identical_book_images=same,stopped={m:sum(r['answers'][m]['stop'] for r in rows) for m in ARMS},
             archive=base.sha(ARCHIVE),teacher=q['responses'],teacher_model=MODELS[QWEN],judge_model=MODELS[JUDGE]))
    print(dict(counts['all']),roles,same,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        assess()
