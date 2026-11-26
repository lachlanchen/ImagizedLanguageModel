"""Replay every pictured equal-inventory choice independently of the training/evaluation batch."""
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import torch

from scripts.evaluate_strip_restoration import png
from scripts.read_pictured_order import PicturedOrderReader,candidate_bits,grid_bits
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, dump, read, sha


def main():
    torch.set_num_threads(2)
    reader=PicturedOrderReader()
    archive=ROOT/"publication/ilm-nature/evidence/strip_restoration_supplied.zip"
    assert sha(archive)==read(PUB/"supplied.json")["archive"]
    cases=read(ROOT/"references/evidence/independent_optical_actions_v1/role-reversal.json")["cases"]
    glyph=np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)
    with ZipFile(archive) as z:
        records=json.loads(z.read("records.json"))
        matches=[]
        rng=np.random.default_rng(72503)
        for r in records:
            if (r["arm"],r["domain"],r["condition"],r["same_bag"])!=("mask","roles","blank",True):
                continue
            i,t=r["i"],r["t"]
            sides=list(rng.permutation([i,i^1]))
            imgs=[]
            for j in sides:
                ids=cases[j]["cells"][t]
                imgs.append(candidate_bits(BytesIO(png(glyph[ids])),len(ids)))
            raw=z.read(f"source/roles/{i:03d}-{t}.png")
            src=grid_bits(BytesIO(raw),len(cases[i]["cells"][1-t]))
            k,_=reader.choose(src,imgs,t)
            kr,_=reader.choose(src,imgs[::-1],t)
            assert k==1-kr and (sides[k]==i)==r["picked"]
            matches.append(dict(i=i,t=t,correct=bool(sides[k]==i)))
    assert len(matches)==168
    dump(PUB/"order-reader.json",dict(archive=sha(archive),calls=2*len(matches),correct=sum(r["correct"] for r in matches),
         identical_under_reversal=len(matches),program=sha(ROOT/"scripts/read_pictured_order.py"),rows=matches))


if __name__=="__main__":
    main()
