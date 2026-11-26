"""Test a fixed missing interval with its correct and exchanged pictured source."""
from hashlib import sha256
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.evaluate_strip_restoration import panel, png, replacement
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, dump, read, sha
from scripts.select_visual_translation import VisualSelector
from scripts.write_strip_restoration import StripWriter


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    selector = VisualSelector()
    glyph = np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)
    archive = ROOT/"publication/ilm-nature/evidence/strip_restoration_gaps.zip"
    records=[]
    tmp=archive.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for domain in ("book","opus"):
            rows=panel(domain)
            pairs=(read(ROOT/"artifacts/visual_strip_restoration_v1/panel.json") if domain=="book" else
                   read(ROOT/"artifacts/source_supported_writing_v1/opus.json"))
            swap=replacement(rows)
            for t in (0,1):
                paths=[selector.actions(glyph[r["cells"][t]],t) for r in pairs]
                for arm in ("mask","restore"):
                    writer=StripWriter(arm,"cpu")
                    body=writer.model
                    table=writer.tables[t]
                    actions=body.actions(writer.tiles,table)
                    for base in range(0,len(pairs),16):
                        group=paths[base:base+16]
                        gold=torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in group],batch_first=True)
                        hidden=torch.zeros_like(gold,dtype=torch.bool)
                        ids=[2*(base+j)+t for j in range(len(group))]
                        for j,(i,target) in enumerate(zip(ids,group,strict=True)):
                            width=(len(target)+3)//4
                            seed=sha256(f"visual-gap-19373/{domain}/{i}/{t}".encode()).digest()
                            start=int.from_bytes(seed[:8],"little")%(len(target)-width+1)
                            hidden[j,start:start+width]=True
                        canvas=gold.masked_fill(hidden,0)
                        for condition in ("source","exchange"):
                            sources=[torch.from_numpy(glyph[rows[i if condition=="source" else swap[i]]["input"]]) for i in ids]
                            pictured=pool_tiles(torch.nn.utils.rnn.pad_sequence(sources,batch_first=True))
                            mem,valid=body.encode(pictured,torch.tensor(list(map(len,sources))))
                            logits=body.position_scores(canvas,gold.ne(0),mem,valid,actions,t)
                            restored=torch.where(hidden,logits.argmax(-1)+1,gold)
                            lp=logits.log_softmax(-1).gather(-1,(gold-1).clamp_min(0).unsqueeze(-1)).squeeze(-1)
                            for j,i in enumerate(ids):
                                ink_ids=table[restored[j]].flatten()
                                ink=writer.ink[ink_ids[ink_ids.ne(0)]].numpy()
                                blob=png(ink)
                                name=f"{domain}/{arm}/{condition}/{i:04d}.png"
                                z.writestr(name,blob)
                                mask=hidden[j]
                                records.append(dict(domain=domain,i=i,t=t,arm=arm,condition=condition,
                                    selected=mask.nonzero().flatten().tolist(),correct=int((gold[j,mask]==restored[j,mask]).sum()),
                                    n=int(mask.sum()),bits=-float(lp[j,mask].double().sum())/math.log(2),
                                    exact=np.array_equal(ink,glyph[pairs[i//2]["cells"][t]]),
                                    member=name,sha256=sha256(blob).hexdigest()))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    summary=[]
    for domain in ("book","opus"):
        for arm in ("mask","restore"):
            for condition in ("source","exchange"):
                subset=[r for r in records if (r["domain"],r["arm"],r["condition"])==(domain,arm,condition)]
                n=sum(r["n"] for r in subset)
                summary.append(dict(domain=domain,arm=arm,condition=condition,n=len(subset),gaps=n,
                    exact=sum(r["exact"] for r in subset),correct_slots=sum(r["correct"] for r in subset),
                    bits_per_strip=sum(r["bits"] for r in subset)/n))
    dump(PUB/"gaps.json",dict(archive=sha(archive),summary=summary))
    print(summary,flush=True)


if __name__=="__main__":
    main()
