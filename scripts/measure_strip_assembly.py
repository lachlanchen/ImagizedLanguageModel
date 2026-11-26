"""Measure order recovery when the exact unordered target strips are supplied."""
from collections import Counter
from hashlib import sha256
import json
import math
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.evaluate_strip_restoration import png
from scripts.reserve_independent_optical_actions import PUB as ORIGINAL
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, RUN, dump, read, sha
from scripts.select_visual_translation import VisualSelector
from scripts.write_strip_restoration import StripWriter


def permute(gold):
    # Same multiset always yields the same permutation, even across sources and cases.
    ordered = np.array(sorted(gold),dtype="<i4")
    seed = int.from_bytes(sha256(ordered.tobytes()).digest()[:8],"little")
    return np.random.default_rng(seed).permutation(ordered).tolist()


@torch.inference_mode()
def assemble(writer, sources, bags, direction):
    body = writer.model
    memory,valid = body.encode(pool_tiles(torch.nn.utils.rnn.pad_sequence(sources,batch_first=True)),
                               torch.tensor(list(map(len,sources))))
    table = writer.tables[direction]
    observed = torch.nn.utils.rnn.pad_sequence([torch.tensor(b) for b in bags],batch_first=True)
    occupied = observed.ne(0)
    actions = body.actions(writer.tiles,table)
    predicted = body.length_scores(memory,valid,torch.tensor(list(map(len,sources))),direction).argmax(1)+1
    for condition,canvas in (("scrambled",observed),("blank",torch.zeros_like(observed))):
        scores = body.position_scores(canvas,occupied,memory,valid,actions,direction)
        for j,bag in enumerate(bags):
            n = len(bag)
            positions,slots = linear_sum_assignment(scores[j,:n,torch.tensor(bag)-1].numpy(),maximize=True)
            assert list(positions)==list(range(n))
            prediction = [bag[k] for k in slots]
            assert Counter(prediction)==Counter(bag)
            yield j,condition,prediction,int(predicted[j]),scores[j,:n]


def main():
    torch.set_num_threads(2)
    groups = {"book":read(RUN/"panel.json"),"roles":read(ORIGINAL/"role-reversal.json")["cases"]}
    pal = np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)
    selector = VisualSelector()
    records = []
    archive = ROOT/"publication/ilm-nature/evidence/strip_restoration_supplied.zip"
    tmp = archive.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        for arm in ("mask","restore"):
            writer = StripWriter(arm,"cpu")
            for domain,rows in groups.items():
                for t in (0,1):
                    paths = [selector.actions(pal[r["cells"][t]],t) for r in rows]
                    bags = list(map(permute,paths))
                    if arm=="restore":
                        for i,r in enumerate(rows):
                            z.writestr(f"source/{domain}/{i:03d}-{t}.png",png(pal[r["cells"][1-t]]))
                    for start in range(0,len(rows),16):
                        block = rows[start:start+16]
                        images = [torch.from_numpy(pal[r["cells"][1-t]]) for r in block]
                        for j,condition,result,length,logits in assemble(writer,images,bags[start:start+16],t):
                            i = start+j
                            ids = writer.tables[t][result].flatten()
                            ink = writer.ink[ids[ids.ne(0)]].numpy()
                            reference = pal[rows[i]["cells"][t]]
                            assert ink.shape==reference.shape
                            member=f"{arm}/{domain}/{i:03d}-{t}-{condition}.png"
                            blob=png(ink)
                            z.writestr(member,blob)
                            same = domain=="roles" and bags[i]==bags[i^1]
                            margin = None
                            picked = None
                            if same:
                                target, alternative = paths[i],paths[i^1]
                                assert target!=alternative
                                idx = torch.arange(len(target))
                                delta = logits[idx,torch.tensor(target)-1]-logits[idx,torch.tensor(alternative)-1]
                                margin = delta.double().sum().item()
                                picked = margin>0 or (margin==0 and target<alternative)
                            records.append(dict(arm=arm,domain=domain,i=i,t=t,condition=condition,length=length,
                                true_length=len(result),initial_exact=bags[i]==paths[i],member=member,
                                exact=np.array_equal(reference,ink),strip_exact=result==paths[i],
                                correct_slots=sum(a==b for a,b in zip(result,paths[i],strict=True)),
                                same_bag=same,margin=margin,picked=picked,sha256=sha256(blob).hexdigest()))
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    summary=[]
    for arm in ("mask","restore"):
        for domain in groups:
            for cond in ("scrambled","blank"):
                v=[r for r in records if (r["arm"],r["domain"],r["condition"])==(arm,domain,cond)]
                pairs={(r["i"],r["t"]):r["exact"] for r in v if r["same_bag"]}
                choices={(r["i"],r["t"]):r["picked"] for r in v if r["same_bag"]}
                error=1-sum(choices.values())/len(choices) if choices else None
                information=(1 + sum(x*math.log2(x) for x in (error,1-error) if x)) if choices else None
                summary.append(dict(choice_correct=sum(choices.values()),choice_n=len(choices),orientation_bits_lower=information,
                     choice_both=sum(a and choices[i^1,t] for (i,t),a in choices.items() if i%2==0),arm=arm,domain=domain,condition=cond,n=len(v),exact=sum(r["exact"] for r in v),
                     length_exact=sum(r["length"]==r["true_length"] for r in v),
                     length_mae=sum(abs(r["length"]-r["true_length"]) for r in v)/len(v),
                     slot_accuracy=sum(r["correct_slots"] for r in v)/sum(r["true_length"] for r in v),
                     random_exact=sum(r["initial_exact"] for r in v),same_bag_pairs=len(pairs)//2,
                     both=sum(a and pairs[i^1,t] for (i,t),a in pairs.items() if i%2==0)))
    dump(PUB/"supplied.json",dict(archive=sha(archive),summary=summary,images=len(records)))
    print(summary,flush=True)


if __name__=="__main__":
    main()
