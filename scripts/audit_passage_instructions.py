"""Recount crossed question orders from the saved ink for every official wrapper."""
from pathlib import Path
import shutil
import numpy as np
import torch
from scripts import test_official_passage_composition as ext
from scripts.compare_passage_instructions import OUT,TEST
from scripts.infer_paired_passage import restore
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import load,write_json,sha
from scripts.train_paired_passage import ROOT,WORK,pad
from collections import Counter

EVIDENCE=ROOT/"references/evidence/paired_passage_20261119"


def main():
    stats=load(OUT/"comparison.json")
    ext.WRAPPERS=TEST
    _,questions,aliases,pairs,_,prefixes,join,tiles,_=ext.prepare()
    n=len(pairs)
    q=len(questions)
    separator=tiles[join].tobytes()
    series=[]
    records={}
    structural={}
    torch.set_num_threads(2)
    for arm in ("fixed","varied"):
        structural[arm]={}
        with torch.inference_mode():
            net=restore(OUT/(arm+".safetensors"))
            features=net.features(tiles)
            for prefix,text in zip(prefixes,TEST,strict=True):
                shapes=Counter()
                right=Counter()
                for offset in range(0,n,128):
                    ids=pairs[offset:offset+128]
                    prompts=[np.concatenate((prefix,questions[a],join,questions[b])) for _,a,b in ids]
                    qh,ln=pad(prompts,features.device)
                    tags=net.plan(features,qh,ln).argmax(-1).numpy()
                    for row,actions,(_,a,b) in zip(prompts,tags,ids,strict=True):
                        parts,stop=net.regions(actions[:len(row)])
                        shapes[" ".join(k for k,_,_ in parts)]+=1
                        chunks=[row[l:r] for k,l,r in parts if k=="span"]
                        emit=[row[l:r] for k,l,r in parts if k=="prompt"]
                        for role,value in (("first",questions[a]),("second",questions[b])):
                            right[role]+=int(any(np.array_equal(c,value) for c in chunks))
                        right["separator_only"]+=int(len(emit)==1 and np.array_equal(emit[0],join))
                structural[arm][text]=dict(common_programs=shapes.most_common(4), **right)
        result=stats[arm]["official"]
        path=OUT/(arm+".npz")
        assert sha(path)==result["archive_sha256"]
        with np.load(path) as z:
            pixels,off,stops=z["pixels"],z["offsets"],z["stopped"]
        assert off[-1]==len(pixels) and len(stops)==q+n*len(TEST) and len(off)==1+len(stops)
        def image(i):
            return pixels[off[i]:off[i+1]].tobytes()
        single=sum(bool(stops[i]) and image(i) in refs for i,refs in enumerate(aliases))
        assert single==result["single_exact"]
        series.append([])
        scores=[]
        for j,text in enumerate(TEST):
            start=q+n*j
            ok=np.array([stops[start+i] and any(image(start+i)==x+separator+y for x in aliases[a] for y in aliases[b])
                         for i,(_,a,b) in enumerate(pairs)])
            both=ok.reshape(-1,2).all(axis=1)
            assert (int(ok.sum()),int(both.sum()))==(result[text]["exact"],result[text]["both_orders"])
            series[-1].append(round(100*int(both.sum())/(n//2),4))
            scores.append(both)
        records[arm]=np.stack(scores)
    delta=records["varied"].astype(int)-records["fixed"]
    group=np.array([g for g,a,b in pairs[::2]])
    rng=np.random.default_rng(19304)
    draw=rng.integers(stats["official_passages"],size=(5000,stats["official_passages"]))
    size=np.bincount(group,minlength=stats["official_passages"])
    intervals=[]
    for row in delta:
        per_g=np.bincount(group,weights=row,minlength=len(size))
        intervals.append(np.quantile(100*per_g[draw].sum(1)/size[draw].sum(1),[.025,.975]).tolist())
    audit=dict(planner_diagnosis=structural,regraded_outputs=2*(q+n*len(TEST)),rescued_pairs=(delta>0).sum(1).tolist(),
               lost_pairs=(delta<0).sum(1).tolist(),bootstrap_95_pp=intervals)
    write_json(EVIDENCE/"instruction_audit.json",audit)
    shutil.copyfile(OUT/"comparison.json",EVIDENCE/"instruction_comparison.json")
    shutil.copyfile(WORK/"official_v2/result.json",EVIDENCE/"official.json")
    f=load(EVIDENCE/"figure.json")
    f["series"]=series
    v=stats["varied"]
    f["table"].append(["Varied instructions, original reader",f'{v["pair_exact"]:,} / 6,086',
                       f'{v["both_orders"]:,} / 3,043',f'{v["single_exact"]:,} / 1,788'])
    f["externalNote"]="Each instruction introduces the same 4,800 question pairs. Both orders must reproduce an accepted reference image and stop. The passage reader stayed frozen during instruction training."
    write_json(EVIDENCE/"figure.json",f)
    print(audit)


if __name__=="__main__":
    main()
