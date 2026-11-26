"""Recount literal content, repetition and stopping directly from frozen output archives."""
from collections import Counter
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np

from scripts.audit_supported_roles import mentions
from scripts.evaluate_pictured_path_translation import grid_bits_out
from scripts.evaluate_strip_restoration import ARMS, archive, panel
from scripts.reserve_visual_strip_restoration import BASE, PUB, ROOT, dump, read, sha
from scripts.score_optical_action_roles import NAMES, VERBS


def main():
    names=[{v[t] for v in NAMES}|{v[2+t] for v in NAMES} for t in (0,1)]
    verbs={base:(finite,base,zh) for finite,base,zh in VERBS}
    cases=read(ROOT/"references/evidence/independent_optical_actions_v1/role-reversal.json")["cases"]
    palette=np.load(BASE/"palette.npy")
    lookup={bytes(palette[int(k)]):v for k,v in read(BASE/"readback.json").items() if int(k)>=2}
    totals={}
    for domain in ("book","opus","roles"):
        rows=panel(domain)
        assert read(PUB/f"{domain}-generation.json")["archive"]==sha(archive(domain))
        with ZipFile(archive(domain)) as z:
            records=json.loads(z.read("records.json"))
            counts={arm:Counter() for arm in ARMS}
            for i,(r,record) in enumerate(zip(rows,records,strict=True)):
                t=r["t"]
                for arm in ARMS:
                    a=record["answers"][arm]
                    blob=z.read(f"{i:04d}/{arm}.png")
                    pixels=grid_bits_out(blob,a["cells"])
                    text="".join(lookup[bytes(c)] for c in pixels)
                    assert text==a["text"]
                    text=text.replace("\u3000"," ")
                    c=counts[arm]
                    c["n"]+=1
                    c["stop"]+=a["stop"]
                    c["cells"]+=len(pixels)
                    words=[text[j:j+4] for j in range(max(0,len(text)-3))]
                    c["repeated_fourgram"]+=len(words)!=len(set(words))
                    c["exact_text"]+=text==r["ref"].replace("\u3000"," ")
                    if domain=="roles":
                        target={s for s in names[t] if mentions(r["ref"],s,t)}
                        assert len(target)==2
                        found={s for s in names[t] if mentions(text,s,t)}
                        finite,stem,zh=verbs[cases[i//2]["verb"]]
                        pred=mentions(text,zh,t) if t else any(mentions(text,s,t) for s in (finite,stem))
                        c["both_names"]+=target<=found
                        c["wrong_name"]+=bool(found-target)
                        c["predicate"]+=pred
            totals[domain]=counts
    dump(PUB/"free-diagnosis.json",totals)


if __name__=="__main__":
    main()
