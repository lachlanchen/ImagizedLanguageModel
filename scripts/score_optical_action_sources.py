"""Do two complete held translations have higher likelihood with their true images?"""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from ilm.visual_lm.optical_actions import pool_tiles
from scripts.prepare_independent_optical_actions import OFFSET, MAX_CELLS
from scripts.reserve_independent_optical_actions import BASE, PUB, ROOT, RUN, dump, read, sha
from scripts.write_independent_optical_actions import ImageWriter


def encoded_targets(pairs,t):
    tables = read(RUN/"tables.json")
    ids = {tuple(v):i for i,v in enumerate(tables["strip"][t])}
    raw = np.load(RUN/"cells.npy",mmap_mode="r")[:,t]
    words = ["".join(chr(OFFSET+int(c)) for c in row[:np.count_nonzero(row)]) for row in raw]
    atoms = read(BASE/"inventories.json")[t][1:]
    tok = Tokenizer(BPE())
    tok.train_from_iterator(words,trainer=BpeTrainer(vocab_size=len(atoms)+2048,min_frequency=2,
            max_token_length=MAX_CELLS,initial_alphabet=[chr(OFFSET+c) for c in atoms],show_progress=False))
    translate = np.array([ids[tuple(ord(c)-OFFSET for c in tok.id_to_token(i))]
                          for i in range(tok.get_vocab_size())])
    targets = np.load(RUN/f"strip_{t}.npy",mmap_mode="r")
    for i in range(0,len(raw),1024):
        for j,obj in enumerate(tok.encode_batch(words[i:i+1024]),i):
            if not np.array_equal(translate[obj.ids], targets[j,:np.count_nonzero(targets[j])]):
                raise ValueError("Rebuilt strip segmentation disagrees with a training image")
    uni = {seq[0]:i for i,seq in enumerate(tables["cell"][t]) if seq}
    cell = [[uni[c] for c in row["cells"][t]] for row in pairs]
    strip = [translate[tok.encode("".join(chr(OFFSET+c) for c in row["cells"][t])).ids].tolist() for row in pairs]
    return dict(cell=cell,strip=strip)


@torch.inference_mode()
def score():
    torch.set_num_threads(2)
    assert sha(RUN/"panel.json")==read(PUB/"panel.json")["sha256"]
    rows = read(RUN/"panel.json")
    ink = np.unpackbits(np.load(BASE/"palette.npy"),axis=1).reshape(-1,32,32)
    dev = torch.device("cpu")
    stats = []
    for t in (0,1):
        encoded = encoded_targets(rows,t)
        works = defaultdict(list)
        for i,row in enumerate(rows):
            works[row["doc"]].append(i)
        paired = []
        for work,group in works.items():
            group.sort(key=lambda i:(len(rows[i]["cells"][1-t]),i))
            paired += list(zip(group[::2],group[1::2],strict=True))
        for arm in ("cell","strip"):
            w = ImageWriter(arm,dev)
            for a,b in paired:
                x = [rows[i]["cells"][1-t] for i in (a,b,b,a)]
                y = [encoded[arm][i] for i in (a,b,a,b)]
                xlen = torch.tensor(list(map(len,x)))
                ylen = torch.tensor(list(map(len,y)))
                src = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(ink[s]) for s in x],batch_first=True)
                tgt = torch.nn.utils.rnn.pad_sequence(list(map(torch.tensor,y)),batch_first=True)
                logits = w.model(pool_tiles(src),xlen,tgt,w.tiles,w.tables[t],t)
                labels = torch.nn.functional.pad(tgt,(0,1))
                logp = logits.log_softmax(-1).gather(-1,labels[:,:,None]).squeeze(-1)
                logp = logp.masked_fill(torch.arange(logp.size(1))[None]>ylen[:,None],0).double().sum(1)
                margin = float(logp[0]+logp[1]-logp[2]-logp[3])
                l0,l1 = (len(rows[i]["cells"][t]) for i in (a,b))
                delta = len(x[0])-len(x[1])
                length = delta*(l0-l1)
                stats.append(dict(arm=arm,t=t,work=rows[a]["doc"],a=a,b=b,margin=margin,
                                  source_delta=delta,length_vote=int(np.sign(length))))
    summary = {}
    for arm in ("cell","strip"):
        data = [r for r in stats if r["arm"]==arm]
        summary[arm] = dict(n=len(data),wins=sum(r["margin"]>0 for r in data),
                           ties=sum(r["margin"]==0 for r in data),equal_length=sum(r["source_delta"]==0 for r in data),
                           equal_wins=sum(r["source_delta"]==0 and r["margin"]>0 for r in data),
                           length_correct=sum((r["length_vote"]+1)/2 for r in data))
    dump(PUB/"source-assignment.json",dict(summary=summary,rows=stats,panel=sha(RUN/"panel.json"),
                protocol=sha(ROOT/"references/independent_optical_actions_source_test.md"),program=sha(Path(__file__))))
    print(summary,flush=True)


if __name__=="__main__":
    score()
