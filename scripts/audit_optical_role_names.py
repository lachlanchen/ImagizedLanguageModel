"""Measure whether either given participant name survives free pixel generation."""
import json
import re
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile

import numpy as np

from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.reserve_independent_optical_actions import BASE, PUB, ROOT, dump, read, sha
from scripts.score_optical_action_roles import NAMES


def main():
    archive = ROOT/"publication/ilm-nature/evidence/optical_action_role_outputs.zip"
    assert sha(archive)==read(PUB/"free-roles.json")["archive"]
    cases = read(PUB/"role-reversal.json")["cases"]
    known = [{p[t] for p in NAMES} | {p[2+t] for p in NAMES} for t in (0,1)]
    palette = np.load(BASE/"palette.npy")
    table = {bytes(palette[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    def mentions(line,name,t):
        return name in line if t else bool(re.search(r"(?<![a-z])"+re.escape(name)+r"(?![a-z])",line,re.I))
    stats = {a:Counter() for a in ("cell","strip")}
    rows = []
    with ZipFile(archive) as z:
        for record in json.loads(z.read("records.json")):
            i,t,arm = record["case"],record["t"],record["arm"]
            target = cases[i]["text"][t]
            names = [s for s in sorted(known[t]) if mentions(target,s,t)]
            assert len(names)==2
            blob = z.read(f"{i:03d}/{t}/{arm}.png")
            assert sha256(blob).hexdigest()==record["sha256"]
            text = "".join(table[bytes(c)] for c in grid_bits_out(blob,record["cells"])).replace("\u3000"," ")
            found = [name for name in names if mentions(text,name,t)]
            stats[arm][("neither","one","both")[len(found)]]+=1
            rows.append(dict(case=i,t=t,arm=arm,names=names,found=found))
    dump(PUB/"free-role-names.json",dict(archive=sha(archive),counts=stats,rows=rows))
    print(stats)


if __name__=="__main__":
    main()
