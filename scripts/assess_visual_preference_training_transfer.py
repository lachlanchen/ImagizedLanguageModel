"""Measure free meaning on repeated preferences versus ordinary rehearsed bilingual passages."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from scripts.evaluate_visual_sequence_preference import ROOT,WORK,PUB,base,png,blind_votes,PictureEditor,SequenceWriter,configure_cuda_cli,gpu_owner
from scripts.prepare_visual_sequence_preference import previous,SEED
from scripts.train_glyph_draft_editor import corpus

ARCHIVE=WORK/'training-transfer.zip'


def make():
    if ARCHIVE.exists():return
    rng=np.random.default_rng(SEED+127)
    mined=base.read(WORK/'emissions.json')
    verified=base.read(PUB/'confirmed.json')['admitted']
    mining=set(base.read(PUB/'selection.json')['indices'])
    remaining=sorted(set(range(base.read(previous.PUB/'selection.json')['real']))-mining)
    ordinary=rng.choice(remaining,48,replace=False).tolist()
    plan=[]
    for t in (0,1):
        accepted=[j for j in verified if mined[j]['t']==t]
        sampled=rng.choice(accepted,min(len(accepted),48),replace=False)
        plan.extend(('contrast',mined[j]['i'],t) for j in sampled)
        plan.extend(('ordinary',i,t) for i in ordinary)
    pairs,lens,drafts,widths=corpus()
    names=base.read(base.BASE/'readback.json')
    models={'parent':PictureEditor('editor','cuda'),**{m:SequenceWriter(m,'cuda') for m in ('likelihood','preference')}}
    palette=models['parent'].palette
    spell=lambda ids: ''.join(names[str(int(x))] for x in ids if x>1).replace('\u3000',' ')
    panel=[]
    partial=ARCHIVE.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as z:
        for group in ('contrast','ordinary'):
            for t in (0,1):
                selection=[i for g,i,d in plan if g==group and d==t]
                for j in range(0,len(selection),24):
                    arr=selection[j:j+24]
                    images=[palette[pairs[i,1-t,:lens[i,1-t]].cpu().long().numpy()] for i in arr]
                    ds=[palette[drafts[t][i,:widths[i,t]].cpu().long().numpy()] for i in arr]
                    results={m:n.translate(images,t,ds) for m,n in models.items()}
                    for k,i in enumerate(arr):
                        idx=len(panel)
                        raw=png(images[k])
                        z.writestr(f'{idx:04d}/source.png',raw)
                        ref=spell(pairs[i,t].cpu().tolist())
                        r=dict(i=i,t=t,group=group,source=spell(pairs[i,1-t].cpu().tolist()),ref=ref,answers={})
                        for m,values in results.items():
                            p,stop=values[k]
                            data=png(p)
                            z.writestr(f'{idx:04d}/{m}.png',data)
                            r['answers'][m]=dict(text=spell(models[m].addresses(p)),stop=bool(stop),png=sha256(data).hexdigest())
                        panel.append(r)
        for t in (0,1):
            ids=[i for i,r in enumerate(panel) if r['t']==t]
            for j,i in enumerate(ids):
                donor=next(k for k in ids[j+1:]+ids[:j] if panel[k]['source']!=panel[i]['source'] and panel[k]['ref']!=panel[i]['ref'])
                panel[i]['foil']=panel[donor]['ref']
        z.writestr('records.json',json.dumps(panel,ensure_ascii=False))
    partial.replace(ARCHIVE)


def assess():
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            for m,a in r['answers'].items():
                assert sha256(z.read(f'{i:04d}/{m}.png')).hexdigest()==a['png']
    candidates=[dict(r['answers'],reference=dict(stop=True,text=r['ref']),foil=dict(stop=True,text=r['foil'])) for r in rows]
    votes=blind_votes(rows,candidates)
    counts=defaultdict(Counter)
    docs=[]
    for i,r in enumerate(rows):
        grade={m:bool(a['stop'] and votes[2*i][m] and votes[2*i+1][m]) for m,a in candidates[i].items()}
        counts[r['group']].update(grade)
        counts[r['group']]['n']+=1
        docs.append(dict(group=r['group'],i=r['i'],t=r['t'],grade=grade,png={m:a['png'] for m,a in r['answers'].items()}))
    base.dump(PUB/'training-transfer.json',dict(archive=base.sha(ARCHIVE),votes=votes,counts=counts,rows=docs,
               protocol=base.sha(ROOT/'references/visual_preference_training_transfer.md')))
    print(counts,flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        make()
        assess()
