"""Teach a visual writer to incorporate a supplied physical evidence span."""
from dataclasses import replace
import fcntl
import os
import numpy as np
import torch

from scripts.grounded_proposition_corpus import materialize,Face,FONTS,LANG,label
from scripts.pointing_glyph_corpus import load
from scripts.train_grounded_propositions import ROOT,SEED,fit,boards
from scripts.vary_grounded_questions import PHRASES


def condition(page,evidence):
    return evidence+bytes(128)+page


def main():
    rows,meta=materialize(boards()["train"],"train")
    face=Face((FONTS[0],2))
    earlier,_=load()
    rng=np.random.default_rng(SEED+991)
    keep=[earlier[i] for i in rng.permutation(len(earlier))[:1024]]
    questions=[]
    for row,m in zip(rows,meta):
        s,r,_=m["facts"][0]
        if rng.random()<.6:
            subject=label(s,r)+"，" if row.task=="subject" or rng.random()<.5 else ""
            questions.append(face(subject+str(rng.choice(PHRASES[r]))+"？"))
        else:questions.append(row.q)
    for arm,control in (("evidence",False),("wrongcue",True)):
        batch=[]
        for row,m,q in zip(rows,meta,questions):
            index=1-m["request"] if control else m["request"]
            evidence=face(m["all_facts"][index][2])
            batch.append(replace(row,p=condition(row.p,evidence),q=q))
        torch.cuda.reset_peak_memory_stats()
        fit(batch+keep,arm,"varied",epochs=2)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(2);main()
