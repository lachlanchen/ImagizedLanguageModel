"""Export physical BPE merges and background image addresses for the independent reader."""
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from scripts.prepare_independent_optical_actions import OFFSET,MAX_CELLS
from scripts.reserve_independent_optical_actions import RUN as CORPUS,PUB as PREP
from scripts.reserve_source_supported_writing import ROOT,BASE,PUB,read,sha,dump

PACKAGE=ROOT/"models/source_supported_writing/selection"


def main():
    PACKAGE.mkdir(exist_ok=True,parents=True)
    raw=np.load(CORPUS/"cells.npy",mmap_mode="r")
    lens=np.count_nonzero(raw,axis=2)
    tables=read(CORPUS/"tables.json")["strip"]
    verified=read(PREP/"preparation.json")["files"]
    assert all(sha(CORPUS/k)==verified[k] for k in ("cells.npy","tables.json","strip_0.npy","strip_1.npy"))
    bank=np.zeros((2,96,16,96),dtype=np.uint16)
    indices=np.zeros((2,96,16),dtype=np.int32)
    counts=[]
    mergers=[]
    for t in (0,1):
        order=np.random.default_rng(13941+t).permutation(len(raw))
        for n in range(1,97):
            # Preserve original shuffled order across lengths tied for distance.
            eligible=order[np.abs(lens[order,1-t]-n)==0]
            if len(eligible)<16:
                eligible=order[np.argsort(np.abs(lens[order,1-t]-n),kind="stable")]
            selected=eligible[:16]
            bank[t,n-1]=raw[selected,1-t]
            indices[t,n-1]=selected
        words=["".join(chr(OFFSET+int(k)) for k in x[:size]) for x,size in zip(raw[:,t],lens[:,t],strict=True)]
        ids={tuple(seq):i for i,seq in enumerate(tables[t])}
        atoms=read(BASE/"inventories.json")[t][1:]
        tok=Tokenizer(BPE())
        tok.train_from_iterator(words,trainer=BpeTrainer(vocab_size=len(atoms)+2048,min_frequency=2,
                max_token_length=MAX_CELLS,initial_alphabet=[chr(OFFSET+k) for k in atoms],show_progress=False))
        def action(word):
            return ids[tuple(ord(ch)-OFFSET for ch in word)]
        pairs=json.loads(tok.to_str())["model"]["merges"]
        merges=[]
        for pair in pairs:
            a,b=pair.split() if isinstance(pair,str) else pair
            merges.append([action(a),action(b),action(a+b)])
        mergers.append(merges)
        lookup=np.array([action(tok.id_to_token(i)) for i in range(tok.get_vocab_size())])
        targets=np.load(CORPUS/f"strip_{t}.npy",mmap_mode="r")
        for i in range(0,len(raw),1024):
            for j,enc in enumerate(tok.encode_batch(words[i:i+1024]),i):
                assert np.array_equal(lookup[enc.ids],targets[j,:np.count_nonzero(targets[j])])
        counts.append(len(merges))
    np.savez_compressed(PACKAGE/"background.npz",cells=bank,indices=indices)
    dump(PACKAGE/"merges.json",mergers)
    dump(PACKAGE/"assets.json",dict(files={k:sha(PACKAGE/k) for k in ("background.npz","merges.json")},
        tables=verified["tables.json"],palette=sha(BASE/"palette.npy"),source=verified["cells.npy"],
        verified_training_sides=len(raw)*2,merges=counts,program=sha(Path(__file__))))
    print(counts,flush=True)


if __name__=="__main__":
    main()
