"""Evaluate single-query selection through physical inputs and autonomous candidate PNGs."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import torch

from scripts.assess_supported_opus import ARCHIVE as OPUS
from scripts.evaluate_source_supported_writing import ARCHIVE as BOOK
from scripts.reserve_independent_optical_actions import PUB as PRIOR
from scripts.reserve_source_supported_writing import BASE,PUB,ROOT,RUN,read,sha,dump
from scripts.select_visual_translation import VisualSelector,candidate_bits,grid_bits

METRICS=("raw","average","corrected")
ARMS=("parent","aligned","shuffled")


def main():
    torch.set_num_threads(2)
    selector=VisualSelector()
    bits=np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32).astype(bool)
    books=read(RUN/"panel.json")
    roles=read(PRIOR/"role-reversal.json")["cases"]
    rows=books+roles
    bias=read(PUB/"assignment-bias.json")
    choices=[]
    for case in bias["rows"]:
        a,b,t=case["a"],case["b"],case["t"]
        opts=[bits[rows[j]["cells"][t]] for j in (a,b)]
        decisions=[selector.score(bits[rows[j]["cells"][1-t]],opts,t) for j in (a,b)]
        # Certify that numeric merges and single-source scoring recover the
        # independently tokenized probabilities on both possible answers.
        baseline=np.array(case["scores"])[:2]
        drift=float(np.max(np.abs(baseline-np.array([[s["raw"] for s in d] for d in decisions]))))
        if drift>0.007:
            raise ValueError(f"Canonical path score disagreement: {drift}")
        choices.append(dict(a=a,b=b,t=t,group=case["group"],same_bag=case["same_bag"],drift=drift,
             scores=decisions,**{m:decisions[0][0][m]>decisions[0][1][m] and decisions[1][1][m]>decisions[1][0][m] for m in METRICS}))
    counts={}
    for group in ("book","roles","same_role_bag"):
        items=[r for r in choices if r["group"]==group or (group=="same_role_bag" and r["group"]=="roles" and r["same_bag"])]
        counts[group]=dict(n=len(items),**{m:sum(item[m] for item in items) for m in METRICS})
    dump(PUB/"independent-choices.json",dict(rows=choices,counts=counts,calibration="only current source length",codec_max_score_drift=max(r["drift"] for r in choices)))
    for domain,path,assessed in (("book",BOOK,"assessment.json"),("opus",OPUS,"opus-assessment.json")):
        results=[]
        evaluation=read(PUB/assessed)
        assert sha(path)==evaluation["archive"]
        with ZipFile(path) as z:
            records=json.loads(z.read("records.json"))
            for i,(r,ok) in enumerate(zip(records,evaluation["primary"]["records"],strict=True)):
                t=r.get("t",0 if r.get("direction")=="en" else 1)
                pairs=books if domain=="book" else read(RUN/"opus.json")
                n=len(pairs[i//2]["cells"][1-t])
                src=grid_bits(BytesIO(z.read(f"{i:04d}/source.png")),n)
                images=[]
                for arm in ARMS:
                    png=z.read(f"{i:04d}/{arm}.png")
                    meta=r["answers"][arm]
                    assert sha256(png).hexdigest()==meta["sha256"]
                    images.append(candidate_bits(BytesIO(png),meta["cells"]))
                scores=dict(zip(ARMS,selector.score(src,images,t),strict=True))
                eligible=[arm for arm in ARMS if r["answers"][arm]["stop"]] or list(ARMS)
                picked={m:max(eligible,key=lambda a:scores[a][m]) for m in METRICS}
                results.append(dict(i=i,t=t,scores=scores,selected=picked,ceiling=any(ok[a] for a in ARMS),
                                    correct={m:ok[a] for m,a in picked.items()}))
        total=dict(n=len(results),ceiling=sum(r["ceiling"] for r in results),
                   **{m:sum(r["correct"][m] for r in results) for m in METRICS})
        dump(PUB/f"{domain}-candidate-selection.json",dict(rows=results,counts=total,archive=sha(path)))
        print(domain,total,flush=True)
    print(counts,flush=True)


if __name__=="__main__":
    main()
