"""Independently verify stored answers and locate divergence from the pictured reference."""
from collections import Counter

import numpy as np

from scripts.compare_bitmap_history_writing import WORK
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_composed_answer_writer import DATA, OUT
from scripts.train_paired_passage import unpack_sequences


def audit():
    result=load(WORK/"result.json")
    for f,h in result["archives"].items():assert sha(WORK/f)==h
    with np.load(WORK/"conditional.npz") as z:
        counts={arm:int(np.count_nonzero(z[arm]==z["truth"])) for arm in ("ordered","shuffled","neural")}
        assert counts==result["conditional"]["exact"] and len(z["truth"])==1525
    path=DATA/"development"
    tiles=np.load(path/"pixels.npy")
    ps,qs,ys=[unpack_sequences(path/(name+".npz")) for name in ("passages","questions","answers")]
    mem=set(map(bytes,np.load(OUT/"memory.npy")))
    src=load(path/"sources.json")
    stats={}
    for arm in ("neural","ordered","shuffled","rotated"):
        groups=Counter();stops=0
        audit=Counter()
        with np.load(WORK/(arm+".npz")) as z:
            off=z["offsets"];pix=z["pixels"];orig=z["origin"]
            assert len(off)==len(ys)+1 and off[0]==0 and off[-1]==len(pix)==len(orig)
            for i,(p,q,y,is_stopped) in enumerate(zip(ps,qs,ys,z["stopped"],strict=True)):
                ref=list(map(bytes,tiles[y]))
                visible=set(map(bytes,tiles[np.r_[p,q]]))
                actual_q=qs[(i+1)%len(qs)] if arm=="rotated" else q
                presented=set(map(bytes,tiles[np.r_[p,actual_q]]))
                out=list(map(bytes,pix[off[i]:off[i+1]]))
                actions=orig[off[i]:off[i+1]]
                assert all((a==0 and g in presented) or (a==1 and g in mem) for a,g in zip(actions,out,strict=True))
                stops+=int(is_stopped)
                groups[src[i]["kind"]]+=int(is_stopped and ref==out)
                if src[i]["kind"]=="absent_content_ink":
                    first=next(k for k,g in enumerate(ref) if g not in visible)
                    divergence=next((k for k,(g,h) in enumerate(zip(out,ref)) if g!=h),min(len(out),len(ref)))
                    label="before" if divergence<first else "at" if divergence==first else "after"
                    audit[label]+=1
                    audit["starts_with_missing"]+=int(first==0)
                    audit["first_cell_correct"]+=int(bool(out) and out[0]==ref[0])
        assert all(count==result["scores"][arm][group]["exact"] for group,count in groups.items())
        assert stops==result["scores"][arm]["all"]["stopped"]
        stats[arm]=dict(audit)
    return dict(reference_missing_cell_divergence=stats,conditional_recount=counts,
                autonomous_streams=len(ys)*4,conditional_positions=1525)


if __name__=="__main__":
    values=audit()
    write_json(WORK/"independent_recount.json",values)
    print(values)
