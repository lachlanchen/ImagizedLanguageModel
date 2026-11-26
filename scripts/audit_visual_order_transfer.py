"""Recount order-only ceilings from archived pixels and bootstrap paired differences by family."""
from collections import defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scripts.evaluate_visual_order import ARCHIVE, PUB, ROOT, dump, read, sha
from scripts.train_origin_value_reader import rows


def main():
    assert sha(ARCHIVE)==read(PUB/"evaluation.json")["archive"]
    ds=rows("sealed")
    dest=PUB/"paired-audit.json"
    relatives=defaultdict(lambda:np.zeros((2,4),dtype=int))
    nochange=0
    with ZipFile(ARCHIVE) as z:
        records=json.loads(z.read("records.json"))
        pairs=defaultdict(dict)
        for r in records:
            a,b=r["answers"]["varied"],r["answers"]["ordered"]
            assert sorted(a["rows"])==sorted(b["rows"]) and a["stop"]==b["stop"] and a["cells"]==b["cells"]
            nochange+=a["image"]==b["image"]
            if r["kind"] in ("new","new_swap"): pairs[r["i"]][r["kind"]]=r
        for i,p in pairs.items():
            a,b=p["new"],p["new_swap"]
            assert a["page"]==b["page"] and a["entry"]==ds[i]["supervision"]["entry"]
            targets={a["target"],b["target"]}
            assert len(targets)==2
            candidate,control=[all(r["answers"][m]["image"]==r["target"] and r["answers"][m]["stop"]
                for r in (a,b)) for m in ("ordered","varied")]
            possible=all(r["answers"]["varied"]["image"] in targets and r["answers"]["varied"]["stop"] for r in (a,b))
            assert not candidate or possible
            relatives[ds[i]["supervision"]["family"]][("en","zh").index(a["language"])]+=(1,candidate,control,possible)
    a=np.stack(list(relatives.values()))
    rng=np.random.default_rng(7384)
    boot=np.concatenate([a[rng.integers(len(a),size=(200,len(a)))].sum(1) for _ in range(100)])
    groups={}
    for k,all_,draws in (("both",a.sum((0,1)),boot.sum(1)),("en",a[:,0].sum(0),boot[:,0]),("zh",a[:,1].sum(0),boot[:,1])):
        n,ordered,varied,available=map(int,all_)
        interval=np.quantile((draws[:,1]-draws[:,2])/draws[:,0],[.025,.975]).tolist()
        groups[k]=dict(n=n,ordered=ordered,varied=varied,ceiling=available,delta=(ordered-varied)/n,interval=interval)
    dump(dest,dict(groups=groups,families=len(a),repetitions=len(boot),identical=nochange,total_records=len(records),
         proof=sha(ROOT/"formal/span_list/VisualFieldOrder.lean"),archive=sha(ARCHIVE),program=sha(Path(__file__))))
    print(groups,flush=True)


if __name__=="__main__":
    main()
