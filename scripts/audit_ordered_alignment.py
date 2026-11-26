"""Check the glyph marginal against explicit segmentations and replay the actual PNGs on CPU."""
from functools import lru_cache
from hashlib import sha256
from itertools import product
import json
from zipfile import ZipFile
import numpy as np
import torch
from ilm.visual_lm.ordered_alignment import AlignedPrefix,movement
from ilm.visual_lm.optical_lexicon import StripInventory,StripPrefix
from scripts.audit_full_image_roles import image_cells
from scripts.evaluate_ordered_alignment import ARCHIVE,ARMS
from scripts.prepare_ordered_alignment import PUB,ROOT,base
from scripts.write_ordered_alignment import OrderedWriter


def enumerate_prefixes():
    strips=[(),(2,),(3,),(2,2),(2,3),(3,2,2)]
    emission=np.array([[0.,.3,.12,.1,.38,.1],[0.,.13,.3,.4,.05,.12],[0.,.04,.29,.14,.13,.4]])
    v=StripInventory(strips,[1,2,3])
    tests,error,uniform_error=0,0.,0.
    for params in ([.2,.1,.1,.45,.15],[1.,0.,0.,0.,0.]):
        kernel=movement(3,params)
        @lru_cache(None)
        def continuation(position,prefix):
            if not prefix:
                return 1.
            mass=0.
            for i,strip in enumerate(strips[1:],1):
                n=min(len(prefix),len(strip))
                if prefix[:n]!=strip[:n]:
                    continue
                suffix=prefix[len(strip):]
                after=sum(kernel[position,k]*continuation(k,suffix) for k in range(3)) if suffix else 1.
                mass+=emission[position,i]*after
            return mass
        def p(prefix):
            return sum(continuation(k,prefix) for k in range(3))/3
        for n in range(7):
            for prefix in product((2,3),repeat=n):
                state=AlignedPrefix(v,emission,kernel)
                unordered=StripPrefix(v,emission.mean(0))
                for c in prefix:
                    for x in (state,unordered):
                        x.probabilities()
                        x.advance(v.global_to_local[c],c)
                prediction=state.probabilities()[1:]
                error=max(error,*abs(prediction-np.array([p(prefix+(c,))/p(prefix) for c in (2,3)])))
                if params[0]==1.:
                    uniform_error=max(uniform_error,*abs(prediction-unordered.probabilities()[1:]))
                tests+=1
    assert max(error,uniform_error)<1e-12
    return dict(prefixes=tests,max_enumeration_error=error,max_nested_model_error=uniform_error)


def main():
    torch.set_num_threads(2)
    exact=enumerate_prefixes()
    writer=OrderedWriter()
    sha=base.sha(ARCHIVE)
    assert sha==base.read(PUB/"generation.json")["archive"]
    checks=[]
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read("records.json"))
        for i,r in enumerate(rows):
            for arm in ARMS:
                assert sha256(z.read(f"{i:04d}/{arm}.png")).hexdigest()==r["answers"][arm]["png"]
        for i in (0,1,24,25,48,49,72,73,96,97,566,567):
            r=rows[i]
            img=image_cells(z.read(f"{i:04d}/source.png"),len(r["source"]))
            (out,stop),=writer.translate([img],r["t"],.5)
            a=r["answers"]["ordered"]
            saved=image_cells(z.read(f"{i:04d}/ordered.png"),a["cells"])
            assert stop==a["stop"] and np.array_equal(out,saved)
            checks.append(i)
    base.dump(PUB/"audit.json",dict(archive=sha,prefix=exact,replayed=checks,all_png_hashes=len(rows)*3,
                      lean=base.sha(ROOT/"formal/span_list/OrderedStripLikelihood.lean")))
    print(exact,len(checks))


if __name__=="__main__":
    main()
