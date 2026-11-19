"""Compare learning ordered claims to identical targets with randomized order instructions."""
from dataclasses import replace
import fcntl
import os
import numpy as np
import torch

from scripts.grounded_proposition_corpus import Face, FONTS, LANG, materialize, label
from scripts.pointing_glyph_corpus import load
from scripts.train_grounded_propositions import ROOT, SEED, boards, fit

WITHHELD={tuple(sorted(p)) for p in (("作词","作曲"),("导演","编剧"))}


def pair_key(board):
    return tuple(sorted(f[1] for f in board["facts"]))


def request(facts,style):
    s,r,_=facts[0]
    _,u,_=facts[1]
    title=label(s,r);a,b=LANG[r][0],LANG[u][0]
    styles=(f"请分两句介绍{title}的{a}和{b}，顺序照此。",
            f"针对{title}写两句介绍。先说明{a}是谁，然后说明{b}是谁。",
            f"关于{title}，把{a}写在第一句，把{b}写在第二句。",
            f"依次说明{title}的{a}、{b}，各写一句。")
    return styles[style]


def main():
    boards_all=boards()["train"]
    data=[b for b in boards_all if b["kind"]=="relation" and pair_key(b) not in WITHHELD]
    compound,meta=materialize(data,"train",True)
    single,_=materialize(boards_all,"train")
    older,_=load()
    face=Face((FONTS[0],2))
    g=np.random.default_rng(SEED+19)
    keep=[single[i] for i in g.permutation(len(single))[:1536]]+[older[i] for i in g.permutation(len(older))[:768]]
    swapped=g.integers(2,size=len(data))
    for arm in ("ordered","unordered"):
        rows=[]
        for row,m in zip(compound,meta,strict=True):
            facts=m["facts"]
            if arm=="unordered" and swapped[m["board"]]:facts=facts[::-1]
            rows.append(replace(row,q=face(request(facts,m["board"]%4))))
        torch.cuda.reset_peak_memory_stats()
        fit(rows+keep,arm,"matched",epochs=2)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
