"""Factorially exchange two visual source paths on all equal-length development pairs."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from scripts.train_field_to_glyph import ARMS, PUB, ROOT, RUN, library, material, scores, data_for, restore, dump, sha


def partners(rows,t):
    pools=defaultdict(list)
    for i,r in enumerate(rows):
        pools[r["work"],len(r["cells"][1-t])].append(i)
    return [(a,b) for subset in pools.values() for x,y in zip(subset[::2],subset[1::2])
            for a,b in ((x,y),(y,x))]


@torch.inference_mode()
def examine(net,rows,t,optical):
    features,inv,maps=optical
    x,y,lens=data_for(rows,t,"cpu")
    bank=net.writer.optical(features)
    keys=net.writer.prototypes(bank,inv[t])
    pairs=partners(rows,t)
    counts=Counter()
    diffs=Counter()
    records=[]
    for j in range(0,len(pairs),12):
        ix,ix2=np.asarray(pairs[j:j+12]).T
        w,v=lens[ix].max(0)
        src=x[ix,:w]
        alt=x[ix2,:w]
        assert torch.equal(src.ne(0),alt.ne(0))
        target=y[ix,:v]
        mask=target.ne(0)
        labels=maps[t][target[mask]].unsqueeze(1)
        field=net.readers[t](features,src)[0][:,1:w+1]
        other=net.readers[t](features,alt)[0][:,1:w+1]
        ll={}
        choices={}
        for name,source,z in (("both",src,field),("base",src,other),("field",alt,field),
                              ("neither",alt,other),("zero",src,torch.zeros_like(field))):
            logits=scores(net,bank,source,target,z,t,keys)[mask].float()
            ll[name]=logits.log_softmax(-1).gather(1,labels).squeeze(1).double()
            choices[name]=logits.argmax(-1)
            counts["nll_"+name]-=float(ll[name].sum())
        counts["positions"]+=len(labels)
        for k in choices: diffs[k]+=int((choices[k]!=choices["both"]).sum())
        base=(ll["both"]-ll["field"]+ll["base"]-ll["neither"])/2
        marginal=(ll["both"]-ll["base"]+ll["field"]-ll["neither"])/2
        joint=ll["both"]-ll["neither"]
        if not torch.allclose(joint,base+marginal,rtol=0,atol=1e-12): raise ValueError("Decomposition")
        cycle=ll["both"]-ll["base"]-ll["field"]+ll["neither"]
        counts["base_gain"]+=float(base.sum())
        counts["field_gain"]+=float(marginal.sum())
        counts["joint_gain"]+=float(joint.sum())
        counts["mean_abs_cycle"]+=float(cycle.abs().sum())
        for part,i in enumerate(ix):
            offsets=np.r_[0,np.cumsum(lens[ix,1])]
            sl=slice(offsets[part],offsets[part+1])
            records.append(dict(row=int(i),partner=int(ix2[part]),positions=sl.stop.item()-sl.start.item(),
                                net_gain=float(joint[sl].sum()),field_gain=float(marginal[sl].sum())))
    n=counts.pop("positions")
    return dict(pairs=len(pairs)//2,positions=n,per_position={k:z/n for k,z in counts.items()},next_choices_changed=diffs),records


def main():
    torch.set_num_threads(2)
    rows=library("dev")
    optical=material("cpu")
    result={}
    records={}
    for arm in ARMS:
        net,_=restore(arm,"cpu")
        result[arm],records[arm]={},{}
        for t,lang in enumerate(("en","zh")):
            result[arm][lang],records[arm][lang]=examine(net,rows,t,optical)
    dump(RUN/"path-records.json",records)
    dump(PUB/"paths.json",dict(conditions=result,program=sha(Path(__file__)),proof=sha(ROOT/"formal/span_list/VisualSourcePaths.lean"),
                              protocol=sha(ROOT/"references/field_transfer_source_intervention.md"),records=sha(RUN/"path-records.json")))
    print(result,flush=True)


if __name__=="__main__":
    main()
