"""Evaluate the deployed two-choice reader on active/passive role reversals."""
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.evaluate_strip_restoration import png
from scripts.read_pictured_order import PicturedOrderReader, candidate_bits, grid_bits
from scripts.reserve_print_translation import Ink, key
from scripts.select_visual_translation import VisualSelector
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, dump, read, sha

PARTICIPLE = dict(help="helped",call="called",follow="followed",see="seen",trust="trusted",visit="visited")


def main():
    torch.set_num_threads(2)
    selector=PicturedOrderReader()
    reference=ROOT/"references/evidence/independent_optical_actions_v1/role-reversal.json"
    cases=read(reference)["cases"]
    palette=np.load(BASE/"palette.npy")
    ink=Ink(palette)
    bits=np.unpackbits(palette,axis=1).reshape(-1,32,32)
    train=np.load(ROOT/"artifacts/independent_optical_actions_v1/cells.npy",mmap_mode="r")[:,0]
    names={int(k):v for k,v in read(BASE/"readback.json").items()}
    names[0]=""
    eligible=[]
    for i in range(0,len(cases),2):
        if cases[i]["negative"]:
            continue
        candidates=[bits[c["cells"][1]] for c in cases[i:i+2]]
        a,b= [c["text"][0].split()[0] for c in cases[i:i+2]]
        orders=[VisualSelector.actions(selector,img,1) for img in candidates]
        if sorted(orders[0])!=sorted(orders[1]):
            continue
        passive=[f"{patient} is {PARTICIPLE[cases[i]['verb']]} by {agent}." for agent,patient in ((a,b),(b,a))]
        eligible.append((i,passive,[ink.encode(x) for x in passive]))
    all_text={key(s) for _,pair,_ in eligible for s in pair}
    all_ink={tuple(g) for _,_,pair in eligible for g in pair}
    denied_text=set()
    denied_ink=set()
    for row in train:
        ids=tuple(row[:np.count_nonzero(row)].tolist())
        if ids in all_ink:
            denied_ink.add(ids)
        text=key("".join(names[v] for v in ids))
        if text in all_text:
            denied_text.add(text)
    archive=ROOT/"publication/ilm-nature/evidence/strip_restoration_voice.zip"
    tmp=archive.with_suffix(".tmp")
    records=[]
    rng=np.random.default_rng(43159)
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for i,passive,inputs in eligible:
            if any(key(s) in denied_text for s in passive) or any(tuple(v) in denied_ink for v in inputs):
                continue
            choices=[]
            for j,c in enumerate(cases[i:i+2]):
                physical=png(bits[c["cells"][1]])
                z.writestr(f"{i:03d}/candidate-{j}.png",physical)
                choices.append(candidate_bits(BytesIO(physical),len(c["cells"][1])))
            for j,c in enumerate(cases[i:i+2]):
                for voice,words,ids in (("active",c["text"][0],c["cells"][0]),("passive",passive[j],inputs[j])):
                    blob=png(bits[ids])
                    member=f"{i:03d}/{j}-{voice}.png"
                    z.writestr(member,blob)
                    src=grid_bits(BytesIO(blob),len(ids))
                    order=list(rng.permutation(2))
                    k,score=selector.choose(src,[choices[v] for v in order],1)
                    rev,_=selector.choose(src,[choices[v] for v in order[::-1]],1)
                    assert rev==1-k
                    records.append(dict(i=i,j=j,voice=voice,source=words,member=member,digest=sha256(blob).hexdigest(),
                        correct=bool(order[k]==j),selected=int(order[k]),margin=float(score[order.index(j)]-score[order.index(1-j)])))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    counts={v:sum(r["correct"] for r in records if r["voice"]==v) for v in ("active","passive")}
    by={(r["i"],r["j"],r["voice"]):r["correct"] for r in records}
    groups=sorted({i for i,_,_ in by})
    count_pairs={v:sum(by[i,0,v] and by[i,1,v] for i in groups) for v in counts}
    dump(PUB/"voice.json",dict(archive=sha(archive),pairs=len(groups),eligible=len(eligible),denied_text=len(denied_text),
        denied_ink=len(denied_ink),correct=counts,both=count_pairs,
        all_four=sum(all(by[i,j,v] for j in (0,1) for v in counts) for i in groups),
        protocol=sha(ROOT/"references/pictured_order_voice_protocol.md"),cases=sha(reference)))
    print(len(groups),counts,count_pairs,flush=True)


if __name__=="__main__":
    main()
