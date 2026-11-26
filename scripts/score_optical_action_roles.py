"""Test frozen writers on role reversals with exactly matched glyph multisets."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.reserve_independent_optical_actions import BASE, PUB, ROOT, RUN, Ink, dump, read, sha
from scripts.score_optical_action_sources import encoded_targets
from scripts.write_independent_optical_actions import ImageWriter

NAMES = (("Anna","安娜","David","大卫"),("Mary","玛丽","John","约翰"),("Alice","爱丽丝","Tom","汤姆"),
         ("Peter","彼得","Rose","露丝"),("Paul","保罗","Jane","简"),("Lucy","露西","Jack","杰克"),
         ("Mark","马克","Helen","海伦"),("James","詹姆斯","Emma","艾玛"))
VERBS = (("helps","help","帮助"),("calls","call","呼叫"),("follows","follow","跟随"),
         ("sees","see","看见"),("trusts","trust","信任"),("visits","visit","探望"))


def material():
    raw = np.load(RUN/"cells.npy",mmap_mode="r")
    ink = Ink(np.load(BASE/"palette.npy"))
    allowed = list(map(set,read(BASE/"inventories.json")))
    known = [{v[:np.count_nonzero(v)].tobytes() for v in raw[:,t]} for t in (0,1)]
    pairs, omissions = [], Counter()
    for a,aa,b,bb in NAMES:
        for finite,stem,zh in VERBS:
            for negative in (False,True):
                enpred = "does not "+stem if negative else finite
                zhpred = "不"+zh if negative else zh
                forms = [[f"{s} {enpred} {o}.",x+zhpred+y+"。"] for s,x,o,y in ((a,aa,b,bb),(b,bb,a,aa))]
                cells = [[ink.encode(s) for s in f] for f in forms]
                if any(not set(c[t])<=allowed[t] or ink.bad.intersection(f[t]) for c,f in zip(cells,forms) for t in (0,1)):
                    omissions["outside_repertoire"] += 1
                    continue
                if any(np.asarray(c[t],dtype=raw.dtype).tobytes() in known[t] for c in cells for t in (0,1)):
                    omissions["train_overlap"] += 1
                    continue
                assert all(Counter(cells[0][t])==Counter(cells[1][t]) and cells[0][t]!=cells[1][t] for t in (0,1))
                for text,tiles in zip(forms,cells,strict=True):
                    pairs.append(dict(text=text,cells=tiles,verb=stem,negative=negative))
    if not pairs:
        raise ValueError("No in-repertoire controlled cases")
    return pairs,omissions


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    rows,skips = material()
    ink = np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)
    report = []
    for t in (0,1):
        encoded = encoded_targets(rows,t)
        for arm in ("cell","strip"):
            net = ImageWriter(arm,"cpu")
            for i in range(0,len(rows),2):
                src = torch.from_numpy(np.stack([ink[rows[j]["cells"][1-t]] for j in (i,i+1,i+1,i)]))
                ys = [torch.tensor(encoded[arm][j]) for j in (i,i+1,i,i+1)]
                y = torch.nn.utils.rnn.pad_sequence(ys,batch_first=True)
                logits = net.model(pool_tiles(src),torch.full((4,),src.size(1)),y,net.tiles,net.tables[t],t)
                labels = torch.nn.functional.pad(y,(0,1))
                logp = logits.log_softmax(-1).gather(-1,labels[...,None]).squeeze(-1)
                logp = logp.masked_fill(torch.arange(logp.size(1))[None]>torch.tensor(list(map(len,ys)))[:,None],0).double().sum(1)
                report.append(dict(pair=i//2,arm=arm,t=t,verb=rows[i]["verb"],negative=rows[i]["negative"],
                                   margin=float(logp[0]+logp[1]-logp[2]-logp[3])))
    counts = defaultdict(Counter)
    for r in report:
        c = counts[r["arm"]]
        c["n"]+=1
        c["wins"]+=int(r["margin"]>0)
        c["ties"]+=int(r["margin"]==0)
        c[("en_wins","zh_wins")[r["t"]]]+=int(r["margin"]>0)
    dump(PUB/"role-reversal.json",dict(cases=rows,omissions=skips,counts=counts,rows=report,
        program=sha(Path(__file__)),protocol=sha(ROOT/"references/optical_action_role_reversal.md")))
    print(dict(counts),skips,flush=True)


if __name__=="__main__":
    main()
