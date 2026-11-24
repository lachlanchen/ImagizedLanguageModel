"""Matched wider question framing and a separate family-held-out phrasing test."""
import argparse

import numpy as np

from scripts.atlas_questions import TRAIN,corpus_chars
from scripts.prepare_glyph_atlas import WORK,PUB,MODEL,sha,dump,read,FACES,draw,questions
from scripts.train_glyph_atlas import datafile,train,gpu_owner

EXTRA = (
    ("曾查{d}，現在請列出{h}的{f}。","{d}只是例子，問題是{h}的{f}。","我要的是{h}的{f}；{d}等一下再查。","若已讀{d}，那麼{h}的{f}是什麼？","關於{h}，請給我{f}。","想知道{h}的{f}。","請說明{h}的{f}，不必說明{d}。","關於{d}的問題已結束。下一個字是{h}，請看{f}。","{h}的{f}是什麼？","先列{h}的{f}，再談{d}。"),
    ("I have read about {d}. Please give me the {f} of {h}.","Could you tell me the {f} of {h}?","What is the {f} of {h}?","I need the {f} of {h}, not {d}.","I already know {d}; what about the {f} of {h}?","Please look at {h} and show its {f}.","Tell me about {h}: I want its {f}, not the entry for {d}.","The {f} of {h} is my question. Leave {d} for another time.","We have finished {d}. For {h}, could you show the {f}?","Show me how the dictionary describes the {f} of {h}."),
)
FRESH = (
    ("從{d}那頁翻過來，我要找{h}的{f}。","這回想讀{h}的{f}，不是前一頁的{d}。","讀完{d}之後，能說說{h}的{f}嗎？","我在{h}這一頁，想看{f}。書籤還停在{d}。"),
    ("Yesterday I looked at {d}. Today I need the {f} of {h}.","For my notes on {h}, find the {f}. Save {d} for tomorrow.","Having seen {d}, could I now see the {f} of {h}?","My question concerns {h}: what is its {f}? I can look up {d} later."),
)


def prepare():
    entries=read(WORK/"sources.json")
    base_chars=[""]+sorted(corpus_chars()|{e["char"] for e in entries})
    extra=sorted(set("".join(s for frames in (*EXTRA,*FRESH) for s in frames))-set(base_chars))
    chars=base_chars+extra
    ix={c:i for i,c in enumerate(chars)}
    obs=datafile("observations.npz")
    faces=[draw(extra,font) for font in FACES]
    new=np.array([[np.frombuffer(face.get(c,bytes(128)),np.uint8) for c in extra] for face in faces])
    codes=np.zeros((len(extra),4),np.uint8)
    for i,c in enumerate(extra):
        codes[i,:len(c.encode())]=list(c.encode())
    np.savez_compressed(WORK/"wording-obs.npz",tiles=np.concatenate((obs["tiles"],new),axis=1),
                        bytes=np.concatenate((obs["bytes"],codes)),keys=obs["keys"])
    frames=tuple(a+b for a,b in zip(TRAIN,EXTRA))
    tr=questions(entries,"train",ix,training=frames)
    fresh=questions(entries,"development",ix,held=FRESH)
    pick=np.char.startswith(fresh["kind"],"novel")
    np.savez_compressed(WORK/"wording-train.npz",**tr)
    np.savez_compressed(WORK/"wording-fresh.npz",**{k:v[pick] for k,v in fresh.items()})
    files={k:sha(WORK/("wording-"+k+".npz")) for k in ("obs","train","fresh")}
    dump(PUB/"wording-corpus.json",dict(files=files,parent=sha(PUB/"corpus.json"),fresh=int(pick.sum()),
                                        train=len(tr["head"]),extra_frames=20,fresh_frames=8))


def load(key):
    path=WORK/("wording-"+key+".npz")
    if sha(path)!=read(PUB/"wording-corpus.json")["files"][key]:
        raise ValueError(key)
    with np.load(path) as a:
        return dict(a)


def fit():
    o,t=load("obs"),load("train")
    with gpu_owner():
        r={m+"-wording":train(m,o,t,sha(PUB/"wording-corpus.json"),"-wording") for m in ("ink","byte")}
    if len({v["schedule"] for v in r.values()})!=1:
        raise ValueError("Mismatched observation arms")
    dump(MODEL/"wording-weights.json",dict(weights={k:v["weights"] for k,v in r.items()}))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("prepare","fit"))
    if ap.parse_args().action=="prepare":
        prepare()
    else:
        from scripts.verse_gpu_runtime import configure_cuda_cli
        configure_cuda_cli()
        fit()
