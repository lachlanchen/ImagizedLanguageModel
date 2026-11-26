"""Recount target incidence and the bounded action margin on supplied book histories."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.reserve_source_supported_writing import BASE,PUB,ROOT,RUN,read,dump,sha
from scripts.score_optical_action_sources import encoded_targets
from scripts.write_source_supported import SupportedWriter


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    rows=read(RUN/"panel.json")
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    bits=torch.from_numpy(np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32))
    records=[]
    for t in (0,1):
        paths=encoded_targets(rows,t)["strip"]
        for arm in ("aligned","shuffled"):
            writer=SupportedWriter(arm,"cpu")
            body,head=writer.model,writer.head
            prior=getattr(head,f"prior_{t}")
            hist,keys=body.actions(writer.tiles,writer.tables[t])
            for i,(r,path) in enumerate(zip(rows,paths,strict=True)):
                src=bits[r["cells"][1-t]][None]
                mem,valid=body.encode(pool_tiles(src),torch.tensor([src.size(1)]))
                logits=head.logits(mem,valid,t)[0]
                odds=head.bonus(mem,valid,t)[0]
                unique=set(path)
                count=len(unique)
                hits=lambda x:len(unique&set((x[1:].topk(count).indices+1).tolist()))
                y=torch.tensor(path)
                embeddings=torch.cat((body.begin[t,None],hist[y]),0)[None]
                latent,_=body.decode(embeddings,mem,valid)
                scores=(latent[0]@keys.T)*keys.size(-1)**-.5
                labels=F.pad(y,(0,1))
                base=scores.argmax(-1)
                gaps=scores.amax(1)-scores.gather(1,labels[:,None])[:,0]
                boosted=scores.clone()
                for step,label in enumerate(labels):
                    boosted[step]+=odds
                    odds[label]=0
                picked=boosted.argmax(-1)
                changed=base.ne(picked)
                repaired=base.ne(labels)&picked.eq(labels)
                broken=base.eq(labels)&picked.ne(labels)
                beyond=gaps.gt(2.40001)
                assert not (beyond&picked.eq(labels)).any()
                records.append(dict(i=i,t=t,arm=arm,n=len(labels),unique=count,prior_hits=hits(prior),
                     hits=hits(logits),base=int(base.eq(labels).sum()),after=int(picked.eq(labels).sum()),
                     changed=int(changed.sum()),repaired=int(repaired.sum()),lost=int(broken.sum()),
                     outside=int(beyond.sum())))
    summary={arm:{k:sum(r[k] for r in records if r["arm"]==arm) for k in
                  ("n","unique","prior_hits","hits","base","after","changed","repaired","lost","outside")}
             for arm in ("aligned","shuffled")}
    dump(PUB/"prefixes.json",dict(summary=summary,rows=records,panel=sha(RUN/"panel.json"),
          program=sha(Path(__file__)),proof=sha(ROOT/"formal/span_list/SupportedWritingMargin.lean")))
    print(summary)


if __name__=="__main__":
    main()
