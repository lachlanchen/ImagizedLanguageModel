"""Compare the first two divergent reference glyphs under controlled source exchange."""
import numpy as np
import torch

from ilm.visual_lm.open_source_cells import OpenSourceCells
from scripts.evaluate_book_order_prior import PUB, WORK, base, ARMS, restore
from scripts.train_open_news_writer import pack, pictured_input


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    rows=base.read(WORK/'edits.json')
    results={}
    for arm in ARMS:
        w=restore(arm,'adapted','cpu')
        route=OpenSourceCells(w,0)
        face=base.Ink(w.palette)
        cases=[]
        for left,right in zip(rows[::2],rows[1::2],strict=True):
            assert left['family']==right['family']
            pair=[left,right]
            src=[face.encode(r['source']) for r in pair]
            ref=[face.encode(r['target'])+[1] for r in pair]
            assert not face.bad and all(all(i in route.known.values() for i in s) for s in src)
            split=next(i for i,(a,b) in enumerate(zip(*ref)) if a!=b)
            branches=[r[split] for r in ref]
            source=pack(src,'cpu')
            fields=(255*w.marker(w.features,source).sigmoid()*source.ne(0).unsqueeze(-1)).round().byte()
            picture=pictured_input(source,fields)
            history=pack([ref[0][:split+1]]*2,'cpu')
            margins=[]
            for side in (0,1):
                hints=route.phrase_cells(w.palette[src[side]])
                common_draft=pack([hints]*2,'cpu')
                pred=w.model(w.features,picture,history,w.inv[0],0,common_draft,w.local[0])[:,-1]
                odds=pred[:,w.local[0][branches[0]]]-pred[:,w.local[0][branches[1]]]
                margins.append(odds.tolist())
            cases.append(dict(family=left['family'],prefix=split,log_odds=margins))
        a=np.array([q['log_odds'] for q in cases])
        correctly_ordered=(a[:,:,0]>a[:,:,1]).all(1)
        both=((a[:,:,0]>0)&(a[:,:,1]<0)).all(1)
        results[arm]=dict(correct_source_order=int(correctly_ordered.sum()),both_next_glyphs=int(both.sum()),
            median_interaction_nats=float(np.median((a[:,:,0]-a[:,:,1]).mean(1))),cases=cases)
    base.dump(PUB/'source_interaction.json',dict(
        files=base.read(PUB/'data.json')['files'],unit='log odds of reference branch A versus B, natural logarithm',
        dimensions=['family','common phrase draft','source'],results=results))
    print({k:(r['correct_source_order'],r['both_next_glyphs']) for k,r in results.items()})


if __name__=='__main__':
    main()
