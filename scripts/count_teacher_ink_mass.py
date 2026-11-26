"""Account for every observed teacher-probability unit by the ink it supervises."""
from pathlib import Path
import unicodedata
import numpy as np
from scripts.reserve_target_glyph_book_panel import BASE, PUB, RUN, read, dump, sha


def kind(text):
    if text.isspace(): return "whitespace"
    category = unicodedata.category(text[0])
    return {"L":"letter_or_ideograph","N":"number","P":"punctuation"}.get(category[0],"other")


def main():
    inventories=read(BASE/"inventories.json")
    readback=read(BASE/"readback.json")
    results={}
    for t,lang in enumerate(("en","zh")):
        path=RUN/f"{lang}-optical.npz"
        if read(PUB/"targets.json")[lang]["sha256"]!=sha(path): raise ValueError("Cached probabilities changed")
        with np.load(path) as data:
            codes,mass=data["choices"],data["weights"]
        seen=np.bincount(codes.ravel(),weights=mass.ravel(),minlength=len(inventories[t]))
        by_kind={}
        for glyph,value in zip(inventories[t],seen,strict=True):
            group="stop" if glyph==1 else kind(readback[str(glyph)])
            by_kind[group]=by_kind.get(group,0.)+float(value)/len(mass)
        by_kind["unresolved"]=1.-sum(by_kind.values())
        totals=mass.sum(1)
        outcomes=(mass>0).sum(1)
        kblank=[i for i,g in enumerate(inventories[t]) if g!=1 and readback[str(g)].isspace()]
        blank_mass=np.where(np.isin(codes,kblank),mass,0.).sum(1)
        results[lang]=dict(events=len(mass),mean_probability=by_kind,distinct_glyphs_mean=float(outcomes.mean()),
                           distinct_glyphs_median=float(np.median(outcomes)),
                           observed_whitespace_majority=int((blank_mass>.5).sum()),
                           observed_whitespace_above_half_resolved=int((blank_mass>totals*.5).sum()))
    dump(PUB/"mass_kinds.json",dict(directions=results,targets=sha(PUB/"targets.json"),program=sha(Path(__file__))))
    print(results,flush=True)


if __name__=="__main__":main()
