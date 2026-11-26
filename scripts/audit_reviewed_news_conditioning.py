"""Compare actual and exchanged sources while holding each complete news reference history fixed."""
import json
from zipfile import ZipFile
import numpy as np
import torch
from scripts.train_open_news_writer import PUB,WORK,ROOT,base,restore,pack,pictured_input
from scripts.measure_paired_visual_writer import unpack
from ilm.visual_lm.open_source_cells import OpenSourceCells
from ilm.visual_lm.book_prefix import observations


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    arch=ROOT/'publication/ilm-nature/evidence/reviewed_news_writers.zip'
    assert base.sha(arch)==base.read(PUB/'meaning.json')['archive']
    panel=base.read(WORK/'test.json')
    nets={k:restore(k,'cpu') for k in ('parent','hidden','ink')}
    parent=nets['parent'];route=OpenSourceCells(parent,0)
    extras={};src=[];drafts=[];targets=[];emitted={k:[] for k in nets}
    with ZipFile(arch) as z:
        rows=json.loads(z.read('records.json'))
        for i,r in enumerate(rows):
            s=unpack(z.read(f'{i}/source.png'),r['cells'])
            ids=[]
            for cell in s:
                k=cell.tobytes();a=route.known.get(k)
                if a is None:a=extras.setdefault(k,len(parent.features)+len(extras))
                ids.append(a)
            src.append(ids);drafts.append(route.phrase_cells(s))
            targets.append(parent.addresses(unpack(z.read(f'{i}/reference.png'),len(panel[i]['target'])))+[1])
            for k in nets:
                a=r['answers'][k]
                emitted[k].append(parent.addresses(unpack(z.read(f'{i}/{k}.png'),a['cells']))+([1] if a['stop'] else []))
    extra=torch.from_numpy(observations({'tiles':np.frombuffer(b''.join(extras),np.uint8).reshape(-1,128)},'ink'))
    details={};totals={}
    for arm,net in nets.items():
        features=torch.cat((net.features,route.mean.expand_as(extra) if arm=='hidden' else extra))
        totals[arm]={}
        details[arm]=[]
        for flag in (False,True):
            ix=[i for i,r in enumerate(rows) if r['open']==flag]
            group=[]
            for offset in range(0,len(ix),8):
                ii=ix[offset:offset+8]
                other=[ix[(j+1)%len(ix)] for j in range(offset,offset+len(ii))]
                refs=[]
                for contexts,target in [(ii,targets),(other,targets),(ii,emitted[arm])]:
                    s=pack([src[i] for i in contexts],'cpu')
                    d=pack([drafts[i] for i in contexts],'cpu')
                    y=pack([target[i] for i in ii],'cpu')
                    f=(255*net.marker(features,s).sigmoid()*s.ne(0).unsqueeze(-1)).round().byte()
                    model=net.model(features,pictured_input(s,f),y,net.inv[0],0,d,net.local[0])
                    valid=y.ne(0); gold=net.local[0][y]
                    ll=(model.gather(-1,gold.clamp_min(0).unsqueeze(-1)).squeeze(-1)*valid).sum(1)/valid.sum(1)
                    hit=((model.argmax(-1)==gold)&valid).sum(1)
                    refs.append(list(zip(ll.tolist(),hit.tolist(),valid.sum(1).tolist(),strict=True)))
                for i,own,swapped,free in zip(ii,*refs,strict=True):
                    group.append(dict(i=i,own=own,swapped=swapped,free=free,stop=rows[i]['answers'][arm]['stop']))
            details[arm]+=group
            totals[arm]['unfamiliar' if flag else 'familiar']=dict(
                requests=len(group),positions=sum(r['own'][2] for r in group),correct=sum(r['own'][1] for r in group),
                exchanged_correct=sum(r['swapped'][1] for r in group),true_source_preferred=sum(r['own'][0]>r['swapped'][0] for r in group),
                mean_source_lift=float(np.mean([r['own'][0]-r['swapped'][0] for r in group])),
                stopped=sum(r['stop'] for r in group),
                own_answer_preferred=sum(r['stop'] and r['free'][0]>r['own'][0] for r in group))
    base.dump(PUB/'conditioning.json',dict(archive=base.sha(arch),details=details,totals=totals))
    print(totals,flush=True)

if __name__=='__main__':
    main()
