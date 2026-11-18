"""Independently regrade complete stored rasters, quantify paired differences and export examples."""
import json
from pathlib import Path

import numpy as np
import torch
from scripts.infer_pictured_entry import page
from scripts.infer_paired_passage import restore
from scripts.prepare_pictured_entry_answers import Rasterizer, load, write_json, sha
from scripts.read_paired_passage_questions import PairedPassages
from scripts.train_paired_passage import ROOT, WORK, DEST


def saved(name):
    with np.load(WORK/(name+".npz")) as z:
        pixels, offsets, stop = (z[k] for k in ("pixels","offsets","stopped"))
    assert offsets[0]==0 and offsets[-1]==len(pixels) and np.all(np.diff(offsets)>=0)
    return [pixels[l:r].tobytes() for l,r in zip(offsets[:-1],offsets[1:],strict=True)],stop


def main():
    evidence = ROOT/"references/evidence/paired_passage_20261119"
    evidence.mkdir(exist_ok=False)
    data = PairedPassages(DEST,"development")
    n = len(data.pairs)
    expected = [data.target(i) for i in range(n)] + list(data.answers)
    evaluation = load(WORK/"evaluation.json")
    fit = load(WORK/"fit.json")
    write_json(evidence/"fit.json",fit)
    write_json(evidence/"evaluation.json",evaluation)
    outcomes = {}
    for arm,count in evaluation["counts"].items():
        pred,stops = saved(arm)
        assert sha(WORK/(arm+".npz"))==count["archive_sha256"]
        ok = np.array([s and o==t for s,o,t in zip(stops,pred,expected[:len(pred)],strict=True)])
        assert (int(ok[:n].sum()),int(ok[:n].reshape(-1,2).all(1).sum()))==(count["pair_exact"],count["both_orders"])
        if arm!="erased":
            assert int(ok[n:].sum())==count["single_exact"]
        else:
            assert all(pred[i]==pred[i+1] and stops[i]==stops[i+1] for i in range(0,n,2))
        outcomes[arm]=ok
    candidate,stops=saved("candidate")
    original_archive=ROOT/"artifacts/mixed_ink_reader_v1/normal.npz"
    assert sha(original_archive)==load(DEST/"report.json")["archive_sha"]
    with np.load(original_archive) as z:
        pix,off,end=z["pixels"],z["offsets"],z["stopped"]
        baseline=[pix[a:b].tobytes() for a,b in zip(off[:-1],off[1:],strict=True)]
    control=np.array([end[a] and end[b] and baseline[a]+data.separator+baseline[b]==expected[i]
                      for i,(_,a,b) in enumerate(data.pairs)])
    assert int(control.sum())==440
    groups=data.pairs[:,0]
    diff=outcomes["candidate"][:n].astype(int)-control.astype(int)
    gcount=len(data.passages)
    size=np.bincount(groups,minlength=gcount)
    weighted=np.bincount(groups,weights=diff,minlength=gcount)
    sampled=np.random.default_rng(19303).integers(gcount,size=(10000,gcount))
    intervals=np.quantile(100*weighted[sampled].sum(1)/size[sampled].sum(1),[.025,.975])
    raw = load(ROOT/"data/raw/cmrc2018/cmrc2018_train.json")
    records={qa["id"]:(p["context"],qa) for a in raw["data"] for p in a["paragraphs"] for qa in p["qas"]}
    render = Rasterizer()
    picked = int(np.flatnonzero(outcomes["candidate"][:n].reshape(-1,2).all(1))[0])*2
    missed = int(np.flatnonzero(~outcomes["candidate"][:n])[0])
    torch.set_num_threads(2)
    net=restore(WORK/"passage_composer.safetensors")
    examples={}
    for label,i in (("forward",picked),("reverse",picked+1),("missed",missed)):
        passage,prompt=data.inputs(i)
        as_rows=lambda s: np.frombuffer(s,np.uint8).reshape(-1,128)
        result,stop = net.answer(as_rows(passage),as_rows(prompt))
        assert result==candidate[i] and stop==stops[i]
        _,a,b=data.pairs[i]
        text,qa=records[data.source_ids[a]]
        _,qb=records[data.source_ids[b]]
        chars=set(text+qa["question"]+qb["question"]+"；")
        ink={}
        for ch in chars:
            ink.setdefault(render.tile(ch),set()).add(ch)
        decoded=lambda s:"".join(next(iter(ink[v])) if len(ink.get(v,set()))==1 else "□" for v in
                                 [s[j:j+128] for j in range(0,len(s),128)])
        examples[label]=dict(row=i,ids=[data.source_ids[a],data.source_ids[b]],questions=[qa["question"],qb["question"]],
                             output=decoded(result),reference=decoded(expected[i]),stopped=bool(stop))
        images={"prompt":prompt,"answer":result}
        if label=="missed": images["reference"]=expected[i]
        for name,image in images.items():
            page(image,width=44).save(evidence/(label+"_"+name+".png"))
    gains=int((diff>0).sum())
    losses=int((diff<0).sum())
    details=dict(first_examples=examples,gains=gains,losses=losses,interval_pp=intervals.tolist(),
                 complete_candidate_recount=int(outcomes["candidate"][:n].sum()),cpu_replayed_rows=[picked,picked+1,missed])
    write_json(evidence/"audit.json",details)
    counts=evaluation["counts"]
    rows=[["Reader and input","Ordered requests","Both orders","Single questions"],
          ["Supplied split, original reader","440 / 6,086","220 / 3,043","434 / 1,788"]]
    for arm,name in (("original_reader","Learned split, original reader"),("candidate","Learned split, refined reader"),
                     ("erased","Refined reader, order erased")):
        c=counts[arm]
        rows.append([name,f'{c["pair_exact"]:,} / 6,086',f'{c["both_orders"]:,} / 3,043',
                     f'{c["single_exact"]:,} / 1,788' if "single_exact" in c else "—"])
    series=[[round(100*float(a)/n,4) for a in (440, counts["original_reader"]["pair_exact"],counts["candidate"]["pair_exact"],counts["erased"]["pair_exact"])],
            [round(100*float(a)/3043,4) for a in (220,counts["original_reader"]["both_orders"],counts["candidate"]["both_orders"],counts["erased"]["both_orders"])]]
    write_json(evidence/"figure.json",dict(table=rows,series=series))
    print(json.dumps(details,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
