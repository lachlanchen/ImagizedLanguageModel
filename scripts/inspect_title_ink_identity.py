"""Identify which absent shapes the forced ink branch draws, using Unicode only after prediction."""
from collections import Counter,defaultdict
import numpy as np
import torch

from ilm.visual_lm.title_transducer import TitleTransducer,article_memory,visual_features
from scripts.diagnose_visual_context_gradients import load,sha,write_new
from scripts.prepare_visual_sequence_corpus import CMRC,Rasterizer
from scripts.train_visual_title_transducer import OUT,Examples


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    weights=OUT/'weights.pt'; model=TitleTransducer().eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    raster=Rasterizer(); labels={}
    for article in load(CMRC/'cmrc2018_dev.json')['data']:
        t=article['title']
        try:
            raw=b''.join(raster.tile(c) for c in t)
        except ValueError:
            continue
        labels[raw]=t
    data=Examples('dev'); counts=defaultdict(Counter)
    for i,r in enumerate(data.rows):
        if r['full_prompt_class']!='missing_glyph':continue
        p,y=data.pair(i); label=labels[y.tobytes()]
        _,x,gids,local=article_memory(p,data.question)
        x=torch.tensor(x)[None]; v=torch.ones(x.shape[:2],dtype=torch.bool); mem,h=model.encode(x,v)
        focus=v.float()*0;feedback=h*0
        for glyph,truth in zip(label,y,strict=True):
            h,positions,route,ink=model.step(mem,v,h,focus,feedback)
            group=local.get(truth.tobytes(),-1)
            if group<0:
                exact=np.array_equal(np.packbits((ink[0]>0).numpy()),truth)
                c=counts[glyph];c['positions']+=1;c['forced_exact']+=int(exact);c['draw_and_exact']+=int(exact and route.argmax().item()==1)
            m=positions*torch.from_numpy(gids==group)[None]
            focus=m/m.sum().clamp_min(1e-12)
            feedback=torch.tensor(visual_features(truth[None]))
    assert sum(c['forced_exact'] for c in counts.values())==load(OUT/'routing.json')['counts']['forced_draw_exact']
    write_new(OUT/'identity.json',dict(by_reference_character=dict(sorted(counts.items())),weights_sha256=sha(weights),
              scope='Unicode is joined only for evaluation of physical emission, after the trained model selected its ink.'))


if __name__=='__main__':
    main()
