"""Check exact glyph-prefix marginalization and verify the archived inference outputs."""
from itertools import product
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from scripts.audit_full_image_roles import image_cells
from scripts.fit_optical_lexicon import BASE,MODEL,PUB,ROOT,WORK,TABLE,Ink,read,dump,sha
from scripts.write_optical_lexicon import LexicalWriter
from ilm.visual_lm.optical_lexicon import StripInventory,StripPrefix


def exact_check():
    tokens=[(),(2,),(3,),(2,2),(2,3),(3,2,2)]
    law=np.array([0.,.15,.16,.25,.24,.20])
    vocab=StripInventory(tokens,[1,2,3])
    @lru_cache(maxsize=None)
    def mass(prefix):
        if not prefix:
            return 1.
        result=0.
        for piece,p in zip(tokens[1:],law[1:],strict=True):
            n=min(len(prefix),len(piece))
            if prefix[:n]==piece[:n]:
                result+=p*(1. if len(prefix)<=len(piece) else mass(prefix[len(piece):]))
        return result
    error=0.
    cases=0
    for n in range(8):
        for prefix in product((2,3),repeat=n):
            state=StripPrefix(vocab,law)
            for c in prefix:
                state.probabilities()
                state.advance(vocab.global_to_local[c],c)
            nxt=state.probabilities()
            expected=[mass(prefix+(c,))/mass(prefix) for c in (2,3)]
            error=max(error,*np.abs(nxt[1:]-expected))
            cases+=1
    assert error<1e-12
    return dict(prefixes=cases,max_error=float(error))


def main():
    torch.set_num_threads(2)
    toy=exact_check()
    writer=LexicalWriter()
    params=read(PUB/"calibration.json")
    strength=params["strength"] or .5
    archive=ROOT/"publication/ilm-nature/evidence/optical_lexicon.zip"
    assert read(PUB/"generation.json")["archive"]==sha(archive)
    sums=[np.abs(law.sum(1)-1).max().item() for law,prior in writer.models]
    assert max(sums)<1e-5
    checks={}
    with ZipFile(archive) as z:
        rows=json.loads(z.read("records.json"))
        for i in (0,1,24,49,72,95):
            row=rows[i]
            inp=image_cells(z.read(f"{i:04d}/source.png"),len(row["source"]))
            for mode,w in (("neural",0.),("lexical",strength)):
                (raw,stop),=writer.translate([inp],row["t"],w)
                recorded=row["answers"][mode]
                im=image_cells(z.read(f"{i:04d}/{mode}.png"),recorded["cells"])
                checks[f"{i}/{mode}"]=bool(stop==recorded["stop"] and np.array_equal(im,raw))
        hashes=0
        from hashlib import sha256
        for i,r in enumerate(rows):
            for mode,o in r["answers"].items():
                assert o["png"]==sha256(z.read(f"{i:04d}/{mode}.png")).hexdigest()
                hashes+=1
    logits=torch.randn(9,224,generator=torch.Generator().manual_seed(4821))
    ratios=torch.randn(9,223,generator=torch.Generator().manual_seed(119))
    modified=writer.calibrated(logits,ratios,1.)
    diff=float((logits.softmax(1)[:,0]-modified.softmax(1)[:,0]).abs().max())
    assert diff<1e-6
    dump(PUB/"audit.json",dict(marginal=toy,stop_probability_error=diff,normalization_errors=sums,cpu_raw_replay=checks,
         verified_png_hashes=hashes,archive=sha(archive),checker=sha(Path(__file__))))
    print(checks,toy,flush=True)


if __name__=="__main__":
    main()
