"""Measure what a frozen pictured writer can assemble with a correct unordered inventory."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile,ZIP_DEFLATED

import numpy as np
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.score_optical_action_sources import encoded_targets
from scripts.reserve_independent_optical_actions import PUB as PRIOR
from scripts.reserve_source_supported_writing import ROOT,BASE,PUB,RUN,read,dump,sha
from scripts.write_independent_optical_actions import ImageWriter
from scripts.write_neural_glyphs import save_ink


@torch.inference_mode()
def write_bag(writer,grids,bag,t):
    body=writer.model
    src=torch.nn.utils.rnn.pad_sequence(grids,batch_first=True)
    mem,mask=body.encode(pool_tiles(src),torch.tensor(list(map(len,grids))))
    feedback,keys=body.actions(writer.tiles,writer.tables[t])
    remaining=bag.clone()
    state=body.begin[t].expand(len(grids),1,-1)
    cache=None
    chosen=[]
    for step in range(int(remaining.sum(1).max())+1):
        latent,cache=body.decode(state,mem,mask,step,cache)
        legal=remaining.gt(0)
        legal[:,0]=~legal.any(1)
        action=(latent[:,0]@keys.T).masked_fill(~legal,-torch.inf).argmax(-1)
        chosen.append(action)
        remaining.scatter_add_(1,action[:,None],-action.ne(0).long()[:,None])
        if action.eq(0).all():
            break
        state=feedback[action,None]
    assert remaining.eq(0).all()
    return torch.stack(chosen,dim=1)


def main():
    torch.set_num_threads(2)
    groups={"book":read(RUN/"panel.json"),"roles":read(PRIOR/"role-reversal.json")["cases"]}
    rows=groups["book"]+groups["roles"]
    palette=np.load(BASE/"palette.npy")
    bits=torch.from_numpy(np.unpackbits(palette,axis=1).reshape(-1,32,32))
    w=ImageWriter("strip","cpu")
    archive=ROOT/"publication/ilm-nature/evidence/visual_content_inventory.zip"
    tmp=archive.with_suffix(".tmp")
    records=[]
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        for t in (0,1):
            paths=encoded_targets(rows,t)["strip"]
            for start in range(0,len(rows),16):
                end=min(start+16,len(rows))
                sources=[bits[r["cells"][1-t]] for r in rows[start:end]]
                bag=torch.zeros((end-start,len(w.tables[t])),dtype=torch.long)
                for j,path in enumerate(paths[start:end]):
                    labels,counts=np.unique(path,return_counts=True)
                    bag[j,labels]=torch.from_numpy(counts)
                outputs=write_bag(w,sources,bag,t)
                for j,line in enumerate(outputs):
                    i=start+j
                    proposal=line[line!=0].tolist()
                    gold=paths[i]
                    assert Counter(proposal)==Counter(gold) and 0 in line.tolist()
                    tiles=w.tables[t][proposal].flatten()
                    ink=w.ink[tiles[tiles!=0]].numpy()
                    reference=bits[rows[i]["cells"][t]].numpy()
                    assert ink.shape==reference.shape
                    isbook=i<len(groups["book"])
                    key=i if isbook else i-len(groups["book"])
                    group="book" if isbook else "roles"
                    blob=BytesIO()
                    save_ink(ink,blob,image_format="PNG")
                    raw=blob.getvalue()
                    z.writestr(f"{group}/{key:03d}/{t}.png",raw)
                    records.append(dict(group=group,i=key,t=t,exact=np.array_equal(ink,reference),
                        same_bag_pair=None if isbook else Counter(gold)==Counter(paths[len(groups["book"])+(key^1)]),
                        actions=len(gold),cells=len(ink),sha256=sha256(raw).hexdigest()))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    counts={}
    for group in groups:
        sample=[r for r in records if r["group"]==group]
        counts[group]=dict(answers=len(sample),exact=sum(r["exact"] for r in sample))
    same=[r for r in records if r["group"]=="roles" and r["same_bag_pair"]]
    paired={(r["i"],r["t"]):r["exact"] for r in same}
    counts["same_role_bag"]=dict(pairs=len(paired)//2,both=sum(correct and paired[i^1,t] for (i,t),correct in paired.items() if i%2==0))
    dump(PUB/"oracle-inventory.json",dict(archive=sha(archive),counts=counts,rows=records,
                protocol=sha(ROOT/"references/visual_content_inventory_diagnostic.md")))
    print(counts,flush=True)


if __name__=="__main__":
    main()
