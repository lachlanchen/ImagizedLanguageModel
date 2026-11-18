"""Measure the trained draw decision under the genuine prefix at source-absent title positions."""
import numpy as np
import torch
from collections import Counter

from ilm.visual_lm.title_transducer import TitleTransducer,article_memory,visual_features
from scripts.diagnose_visual_context_gradients import sha,write_new
from scripts.train_visual_title_transducer import OUT,Examples


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    model=TitleTransducer().eval()
    model.load_state_dict(torch.load(OUT/'weights.pt',map_location='cpu',weights_only=True))
    train=Examples('train'); counts=Counter()
    for i in range(len(train.rows)):
        p,y=train.pair(i); visible=set(x.tobytes() for x in np.concatenate((p,train.question)))
        counts['training_absent']+=sum(x.tobytes() not in visible for x in y)
        counts['training_characters']+=len(y)
    dev=Examples('dev')
    for i,r in enumerate(dev.rows):
        if r['full_prompt_class']!='missing_glyph':continue
        p,y=dev.pair(i)
        tiles,features,ids,lookup=article_memory(p,dev.question)
        x=torch.tensor(features)[None]
        valid=torch.ones(x.shape[:2],dtype=torch.bool)
        mem,h=model.encode(x,valid)
        focus=valid.float()*0;prev=h*0
        for expected in y:
            h,positions,gate,ink=model.step(mem,valid,h,focus,prev)
            group=lookup.get(expected.tobytes(),-1)
            if group<0:
                counts['development_absent']+=1
                counts[['select','draw','stop'][gate.argmax().item()]]+=1
                output=np.packbits((ink[0]>0).numpy())
                counts['forced_draw_exact']+=int(np.array_equal(output,expected))
                counts['forced_draw_blank']+=int(not output.any())
            m=positions*torch.from_numpy(ids==group)[None]
            focus=m/m.sum().clamp_min(1e-12)
            prev=torch.tensor(visual_features(expected[None]))
    write_new(OUT/'routing.json',dict(counts=dict(counts),weights_sha256=sha(OUT/'weights.pt'),
              scope='Post hoc, CPU, supplied true visual history. Draw branch is scored even when the route would select or stop.'))


if __name__=='__main__':
    main()
