"""Free-write the same role reversals that the trained bodies scored conditionally."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.reserve_independent_optical_actions import BASE, PUB, ROOT, dump, read, sha
from scripts.write_independent_optical_actions import ImageWriter
from scripts.write_neural_glyphs import save_ink


def main():
    torch.set_num_threads(2)
    pairs = read(PUB/"role-reversal.json")["cases"]
    palette = np.load(BASE/"palette.npy")
    bits = np.unpackbits(palette,axis=1).reshape(-1,32,32).astype(bool)
    archive = ROOT/"publication/ilm-nature/evidence/optical_action_role_outputs.zip"
    tmp = archive.with_suffix(".tmp")
    scores = {}
    records = []
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        for arm in ("cell","strip"):
            w = ImageWriter(arm,"cpu")
            single = Counter()
            both = Counter()
            for t in (0,1):
                exact = []
                for k in range(0,len(pairs),16):
                    group = pairs[k:k+16]
                    source = [bits[r["cells"][1-t]] for r in group]
                    for j,(ink,stop) in enumerate(w.write(source,t)):
                        i = k+j
                        prefix = f"{i:03d}/{t}/"
                        if arm=="cell":
                            buf = BytesIO()
                            save_ink(source[j],buf,image_format="PNG")
                            z.writestr(prefix+"source.png",buf.getvalue())
                        buf = BytesIO()
                        save_ink(ink,buf,image_format="PNG")
                        blob = buf.getvalue()
                        z.writestr(prefix+arm+".png",blob)
                        canonical = stop and np.array_equal(ink,bits[pairs[i]["cells"][t]])
                        opposite = stop and np.array_equal(ink,bits[pairs[i^1]["cells"][t]])
                        exact.append(canonical)
                        single["exact"]+=int(canonical)
                        single["opposite"]+=int(opposite)
                        single["stopped"]+=int(stop)
                        single["n"]+=1
                        records.append(dict(arm=arm,t=t,case=i,exact=canonical,opposite=opposite,stop=stop,
                                            cells=len(ink),sha256=sha256(blob).hexdigest()))
                both["exact"]+=sum(a and b for a,b in zip(exact[::2],exact[1::2],strict=True))
                both["n"]+=len(exact)//2
            scores[arm] = dict(answers=single,pairs=both)
        z.writestr("records.json",json.dumps(records))
    tmp.replace(archive)
    dump(PUB/"free-roles.json",dict(archive=sha(archive),scores=scores,
                                    cases=sha(PUB/"role-reversal.json"),readout="exact stopped bitmap"))
    print(scores,flush=True)


if __name__=="__main__":
    main()
