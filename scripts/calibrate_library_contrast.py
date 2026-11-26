"""Fit the gain on development book images, using four fixed training backgrounds."""

import math
from pathlib import Path
import numpy as np
import torch
from ilm.visual_lm.background_contrast import backgrounds, evidence
from scripts.sample_visual_library import RUN, ROOT, EVIDENCE, MODELS, library, read, sha
from scripts.prepare_teacher_translation import dump
from scripts.train_visual_prefix import gpu_owner, pack, material
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli

GAINS = (0.,.25,.5,1.,2.)


def background_table(rows):
    rng=np.random.default_rng(381793)
    order=rng.permutation(len(rows))
    table=np.zeros((2,97,4,96),np.int32)
    for t in (0,1):
        lengths=np.asarray([len(rows[i]["cells"][1-t]) for i in order])
        for length in range(1,97):
            used=set()
            candidates=order[np.argsort(abs(lengths-length),kind="stable")]
            for j in candidates:
                r=rows[j]
                if r["work"] in used:
                    continue
                cells=r["cells"][1-t]
                table[t,length,len(used),:len(cells)]=cells
                used.add(r["work"])
                if len(used)==4:
                    break
            if len(used)!=4:
                raise ValueError("Insufficient background diversity")
    return table


@torch.inference_mode()
def calibrate(dev, table):
    f,inv,maps=material("cuda")
    net=restore("library-broad","cuda")
    totals=np.zeros(len(GAINS))
    derivative=0.
    count=0
    entropy=0.
    for t in (0,1):
        for j in range(0,len(dev),8):
            rows=dev[j:j+8]
            x=pack([r["cells"][1-t] for r in rows],"cuda")
            y=pack([r["cells"][t]+[1] for r in rows],"cuda")
            z=backgrounds(x,table[t])
            batch,k=z.shape[:2]
            logits=net(f,z.flatten(0,1),y.repeat_interleave(k,0),inv[t],t)
            full,ratio=evidence(logits.reshape(batch,k,y.size(1),-1))
            mask=y!=0
            label=maps[t][y].clamp_min(0).unsqueeze(-1)
            true=ratio.gather(-1,label).squeeze(-1)
            derivative+=float(((full.exp()*ratio).sum(-1)-true)[mask].double().sum())
            entropy-=float((full.exp()*full).sum(-1)[mask].double().sum())
            for g,a in enumerate(GAINS):
                lp=(full+a*ratio).log_softmax(-1).gather(-1,label).squeeze(-1)
                totals[g]-=float(lp[mask].double().sum())
            count+=int(mask.sum())
    scores=totals/count
    return dict(positions=count,requests=2*len(dev),gains=GAINS,nll=scores.tolist(),
                selected=GAINS[int(np.argmin(scores))],derivative_at_zero=derivative/count,
                entropy_at_zero_nats=entropy/count)


def main():
    configure_cuda_cli()
    p=MODELS/"broad/backgrounds.npy"
    if not p.exists():
        np.save(p,background_table(library("train")))
    bind=dict(weights=sha(MODELS/"broad/weights.safetensors"),backgrounds=sha(p),
              protocol=sha(ROOT/"references/library_ink_contrast_protocol.md"),
              algorithm=sha(ROOT/"ilm/visual_lm/background_contrast.py"),program=sha(Path(__file__)))
    out=MODELS/"broad/contrast.json"
    if out.exists():
        if read(out)["binding"]!=bind:
            raise ValueError("Calibration inputs changed")
        return
    with gpu_owner():
        result=calibrate(library("dev"),torch.as_tensor(np.load(p),device="cuda",dtype=torch.long))
    dump(out,dict(binding=bind,**result))
    print(result,flush=True)


if __name__=="__main__":
    main()
