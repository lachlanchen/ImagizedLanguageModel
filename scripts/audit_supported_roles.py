"""Measure literal participants and predicates in freely emitted physical answers."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import re
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import torch

from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.reserve_independent_optical_actions import PUB as PRIOR
from scripts.reserve_source_supported_writing import BASE,PUB,ROOT,dump,read,sha
from scripts.score_optical_action_roles import NAMES,VERBS
from scripts.write_source_supported import SupportedWriter
from scripts.write_neural_glyphs import grid_bits,save_ink


def mentions(line,word,t):
    return word in line if t else bool(re.search(r"(?<![a-z])"+re.escape(word)+r"(?![a-z])",line,re.I))


def main():
    torch.set_num_threads(2)
    cases=read(PRIOR/"role-reversal.json")["cases"]
    old=ROOT/"publication/ilm-nature/evidence/optical_action_role_outputs.zip"
    assert sha(old)==read(PRIOR/"free-roles.json")["archive"]
    target=ROOT/"publication/ilm-nature/evidence/supported_role_outputs.zip"
    tmp=target.with_suffix(".tmp")
    palette=np.load(BASE/"palette.npy")
    alphabet={bytes(palette[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    names=[{v[t] for v in NAMES}|{v[2+t] for v in NAMES} for t in (0,1)]
    verbs={base:(finite,base,zh) for finite,base,zh in VERBS}
    rows=[]
    with ZipFile(old) as prev,ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        old_meta={(r["case"],r["t"]):r for r in json.loads(prev.read("records.json")) if r["arm"]=="strip"}
        for arm in ("parent","aligned","shuffled"):
            writer=SupportedWriter(arm,"cpu") if arm!="parent" else None
            for t in (0,1):
                for k in range(0,len(cases),16):
                    group=cases[k:k+16]
                    source=[prev.read(f"{i:03d}/{t}/source.png") for i in range(k,k+len(group))]
                    outputs=[]
                    if writer is not None:
                        grids=[grid_bits(BytesIO(blob),len(r["cells"][1-t])) for blob,r in zip(source,group,strict=True)]
                        for bitmap,stop in writer.write(grids,t):
                            buf=BytesIO()
                            save_ink(bitmap,buf,image_format="PNG")
                            outputs.append((buf.getvalue(),len(bitmap),stop))
                    for j,r in enumerate(group):
                        i=k+j
                        if arm=="parent":
                            record=old_meta[i,t]
                            blob=prev.read(f"{i:03d}/{t}/strip.png")
                            assert sha256(blob).hexdigest()==record["sha256"]
                            n,stop=record["cells"],record["stop"]
                            z.writestr(f"{i:03d}/{t}/source.png",source[j])
                        else:
                            blob,n,stop=outputs[j]
                        z.writestr(f"{i:03d}/{t}/{arm}.png",blob)
                        pixels=grid_bits_out(blob,n)
                        text="".join(alphabet[bytes(c)] for c in pixels).replace("\u3000"," ")
                        wanted={s for s in names[t] if mentions(r["text"][t],s,t)}
                        assert len(wanted)==2
                        found={s for s in names[t] if mentions(text,s,t)}
                        finite,stem,zh=verbs[r["verb"]]
                        predicate=mentions(text,zh,t) if t else any(mentions(text,s,t) for s in (finite,stem))
                        rows.append(dict(case=i,t=t,arm=arm,names=len(wanted&found),extra=sorted(found-wanted),
                               predicate=predicate,exact=bool(stop and np.array_equal(pixels,palette[r["cells"][t]])),
                               stopped=stop,text=text,cells=n,sha256=sha256(blob).hexdigest()))
        z.writestr("records.json",json.dumps(rows,ensure_ascii=False))
    tmp.replace(target)
    counts={}
    for arm in ("parent","aligned","shuffled"):
        data=[r for r in rows if r["arm"]==arm]
        counts[arm]=dict(n=len(data),names_both=sum(r["names"]==2 for r in data),neither=sum(r["names"]==0 for r in data),
                         predicate=sum(r["predicate"] for r in data),unsupported_name=sum(bool(r["extra"]) for r in data),
                         names_and_predicate=sum(r["names"]==2 and r["predicate"] for r in data),exact=sum(r["exact"] for r in data))
    dump(PUB/"roles.json",dict(archive=sha(target),counts=counts,parent_archive=sha(old)))
    print(counts,flush=True)


if __name__=="__main__":
    main()
