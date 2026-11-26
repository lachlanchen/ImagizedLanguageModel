"""Evaluate held free answer images, complete path probabilities, prefix errors and swapped-source controls."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from scripts.prepare_visual_sequence_preference import ROOT,PUB,WORK,MODEL,base,previous,QWEN,JUDGE,MODELS,STORE,server,chat
from scripts.evaluate_glyph_draft_editor import ARCHIVE as PRIOR,png,blind_votes
from scripts.write_visual_sequence_preference import SequenceWriter
from scripts.write_glyph_draft_editor import PictureEditor,pack,pictured_input
from scripts.write_covered_phrases import CoveredWriter
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

ARCHIVE=ROOT/'publication/ilm-nature/evidence/visual_sequence_preference.zip'
ARMS=('parent','likelihood','preference','exchanged','phrase')


@torch.inference_mode()
def path_detail(net,src,draft,target,t,ended=True):
    x=pack([src],net.device)
    path=target+([1] if ended else [])
    if not path:
        raise ValueError('An empty path requires an observed stop')
    y=pack([path],net.device)
    d=pack([draft or [0]],net.device)
    field=(255*net.marker(net.features,x).sigmoid()*x.ne(0).unsqueeze(-1)).round().byte()
    scores=net.model(net.features,pictured_input(x,field),y,net.inv[t],t,d,net.local[t])[0].float()
    local=net.local[t][y[0]]
    assert (local>=0).all()
    ref=scores.gather(1,local[:,None]).flatten()
    mode=scores.argmax(-1)
    incorrect=mode.ne(local)
    first=int(incorrect.int().argmax()) if incorrect.any() else len(local)
    rivals=scores.scatter(1,local[:,None],float('-inf')).max(-1).values
    margin=(ref-rivals).cpu().tolist()
    return dict(total=ref.double().sum().item(),per_cell=ref.double().mean().item(),length=len(local),ended=ended,
                first_error=first,prefix_margins=margin,top1=int((~incorrect).sum()))


def panel_rows():
    selection=base.read(PUB/'selection.json')
    assert base.sha(WORK/'panel.json')==selection['panel']
    books=base.read(WORK/'panel.json')
    rows=[]
    for i,row in enumerate(books):
        other=(i+12)%len(books)
        assert row['doc']!=books[other]['doc']
        for t in (0,1):
            rows.append(dict(doc=row['doc'],group='book',t=t,source=row['text'][1-t],ref=row['text'][t],
                             foil=books[other]['text'][t],donor=2*other+t,answers={}))
    assert base.sha(PRIOR)==base.read(previous.PUB/'generation.json')['archive']
    with ZipFile(PRIOR) as z:
        old=json.loads(z.read('records.json'))[96:]
    for j,r in enumerate(old):
        # Reverse the participants, preserving polarity in the four-item group.
        rows.append(dict(r,donor=96+(j//4)*4+(j%4^2),answers={}))
    return rows,old


def generate():
    if (PUB/'generation.json').is_file():
        binding=base.read(PUB/'generation.json')
        assert binding['archive']==base.sha(ARCHIVE) and binding['paths']==base.sha(PUB/'paths.json')
        return
    rows,old=panel_rows()
    parent=PictureEditor('editor','cuda')
    fitted={m:SequenceWriter(m,'cuda') for m in ('likelihood','preference')}
    face=base.Ink(parent.palette)
    src=[face.encode(r['source']) for r in rows]
    targets=[face.encode(r['ref']) for r in rows]
    assert not face.bad
    tiles=parent.palette
    names=base.read(base.BASE/'readback.json')
    paths=[{} for _ in range(96)]
    partial=ARCHIVE.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as z,ZipFile(PRIOR) as previous_raw:
        def record(i,m,pixels,stop):
            ids=parent.addresses(pixels)
            raw=png(pixels)
            z.writestr(f'{i:04d}/{m}.png',raw)
            rows[i]['answers'][m]=dict(png=sha256(raw).hexdigest(),stop=bool(stop),cells=len(ids),
                 text=''.join(names[str(v)] for v in ids),exact=bool(stop and ids==targets[i]))
            return ids
        for i,ids in enumerate(src):
            data=png(tiles[ids])
            if i>=96:
                assert data==previous_raw.read(f'{i:04d}/source.png')
            z.writestr(f'{i:04d}/source.png',data)
        for t in (0,1):
            order=[i for i,r in enumerate(rows) if r['t']==t]
            phrase=CoveredWriter(t)
            drafts={i:phrase.translate(tiles[src[i]])[0] for i in order}
            for i in order:
                record(i,'phrase',drafts[i],True)
                if i>=96:
                    assert rows[i]['answers']['phrase']['png']==old[i-96]['answers']['phrase']['png']
                    rows[i]['answers']['parent']=old[i-96]['answers']['editor']
                    z.writestr(f'{i:04d}/parent.png',previous_raw.read(f'{i:04d}/editor.png'))
            for m,net in (('parent',parent),*fitted.items(),('exchanged',fitted['preference'])):
                selected=[i for i in order if i<96 or m!='parent']
                for j in range(0,len(selected),24):
                    batch=selected[j:j+24]
                    donors=[rows[i]['donor'] if m=='exchanged' else i for i in batch]
                    result=net.translate([tiles[src[k]] for k in donors],t,[drafts[k] for k in donors])
                    for i,(pixels,stop) in zip(batch,result,strict=True):
                        emitted=record(i,m,pixels,stop)
                        if i<96 and m!='exchanged':
                            ds=net.addresses(drafts[i])
                            paths[i][m]=dict(free=path_detail(net,src[i],ds,emitted,t,ended=stop),
                                             ref=path_detail(net,src[i],ds,targets[i],t))
        z.writestr('records.json',json.dumps(rows,ensure_ascii=False))
    partial.replace(ARCHIVE)
    base.dump(PUB/'paths.json',paths)
    base.dump(PUB/'generation.json',dict(archive=base.sha(ARCHIVE),paths=base.sha(PUB/'paths.json'),requests=len(rows),
              outputs=len(rows)*len(ARMS),selection=base.sha(PUB/'selection.json'),
              fits={m:base.sha(MODEL/m/'fit.json') for m in fitted}))


def evaluate():
    if (PUB/'assessment.json').is_file():
        assert base.read(PUB/'assessment.json')['archive']==base.sha(ARCHIVE)
        return
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            for arm in ARMS:
                assert sha256(z.read(f'{i:04d}/{arm}.png')).hexdigest()==r['answers'][arm]['png']
    books=rows[:96]
    for m in (QWEN,JUDGE):
        assert base.sha(STORE/'blobs'/('sha256-'+MODELS[m]))==MODELS[m]
    cache=WORK/'qwen.json'
    if not cache.exists():
        def translate(r):
            return chat(url,QWEN,'Translate into '+('English','Mandarin Chinese')[r['t']]+'. Return only the translation.\n'+r['source'])
        with server() as url,ThreadPoolExecutor(max_workers=4) as threads:
            q=list(threads.map(translate,books))
        base.dump(cache,dict(rows=q,archive=base.sha(ARCHIVE)))
    q=base.read(cache)
    assert q['archive']==base.sha(ARCHIVE)
    cand=[dict(r['answers'],qwen=ans,reference=dict(text=r['ref'],stop=True),foil=dict(text=r['foil'],stop=True))
          for r,ans in zip(books,q['rows'],strict=True)]
    votes=blind_votes(books,cand)
    grades=[]
    counts=defaultdict(Counter)
    for i,(r,g) in enumerate(zip(books,cand,strict=True)):
        ok={m:bool(x['stop'] and votes[2*i][m] and votes[2*i+1][m]) for m,x in g.items()}
        for group in (r['doc'],('en','zh')[r['t']],'all'):
            counts[group].update(ok)
            counts[group]['n']+=1
        grades.append(ok)
    delta=[(counts[w]['preference']-counts[w]['likelihood'])/counts[w]['n'] for w in dict.fromkeys(r['doc'] for r in books)]
    ci=np.quantile([np.mean(p) for p in product(delta,repeat=4)],[.025,.975]).tolist()
    roles={}
    for m in ARMS:
        exact=[r['answers'][m]['exact'] for r in rows[96:]]
        roles[m]=dict(exact=sum(exact),polarity=sum(all(exact[i:i+2]) for i in range(0,472,2)),
                      crossed=sum(all(exact[i:i+4]) for i in range(0,472,4)))
    detail=base.read(PUB/'paths.json')
    paths={}
    for m in ('parent','likelihood','preference'):
        ds=[r[m]['ref']['per_cell']-r[m]['free']['per_cell'] for r in detail]
        paths[m]=dict(mean_gap=float(np.mean(ds)),median_gap=float(np.median(ds)),reference_wins=sum(x>1e-5 for x in ds),
                      first_error_at_zero=sum(r[m]['ref']['first_error']==0 for r in detail),
                      correct_prefix_positions=sum(r[m]['ref']['top1'] for r in detail),total=sum(r[m]['ref']['length'] for r in detail))
    base.dump(PUB/'assessment.json',dict(archive=base.sha(ARCHIVE),counts=counts,votes=votes,grades=grades,teacher=q['rows'],
            roles=roles,work_interval=ci,paths=paths,stops={m:sum(r['answers'][m]['stop'] for r in rows) for m in ARMS}))
    print(dict(counts['all']),roles,paths,ci,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        evaluate()
