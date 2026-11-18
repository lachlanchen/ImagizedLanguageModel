"""Regrade archived answer ink, paired uncertainty, and standalone CPU deployment."""
from pathlib import Path
import json

import numpy as np
import torch
from safetensors.torch import load_model

from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts import test_official_passage_composition as official
from scripts.compare_traditional_passage_training import (
    CMRC_PREFIXES, DRCD_PREFIXES, DATA, INITIAL, OUT, ROOT, TraditionalPassages,
)
from scripts.infer_pictured_entry import bitmap, page
from scripts.prepare_pictured_entry_answers import Rasterizer, load, write_json, sha
from scripts.read_paired_passage_questions import PairedPassages

ARM_NAMES = ("released","replay","mixed")
EVIDENCE = ROOT/"references/evidence/traditional_passage_20261119"


def archived(path, digest):
    assert sha(path) == digest
    with np.load(path) as z:
        pixels,off,stop = z["pixels"],z["offsets"],z["stopped"]
    assert off[0]==0 and off[-1]==len(pixels) and len(off)==len(stop)+1 and np.all(np.diff(off)>=0)
    return [pixels[l:r].tobytes() for l,r in zip(off[:-1],off[1:],strict=True)],stop


def clustered_delta(candidate,control,article_ids):
    _,which=np.unique(article_ids,return_inverse=True)
    sizes=np.bincount(which)
    diff=candidate.astype(int)-control.astype(int)
    totals=np.bincount(which,weights=diff)
    sample=np.random.default_rng(19307).integers(len(sizes),size=(10000,len(sizes)))
    delta=100*totals[sample].sum(1)/sizes[sample].sum(1)
    return dict(articles=len(sizes),gained=int((diff>0).sum()),lost=int((diff<0).sum()),
                difference_pp=float(100*diff.mean()),ci95_pp=np.quantile(delta,[.025,.975]).tolist())


def assess(name, arms, data, prefixes):
    _,qs,aliases,pairs,_,_,join,bank,_=data
    singles=len(qs)
    sep=bank[join].tobytes()
    ok_by_arm={}
    for arm in ARM_NAMES:
        report=arms[arm]["scores"][name]
        images,ends=archived(OUT/(arm+"_"+name+".npz"),report["archive_sha256"])
        ok=np.array([ends[i] and images[i] in alias for i,alias in enumerate(aliases)])
        assert int(ok.sum())==report["single_exact"]
        offset=singles
        for wrapper in prefixes:
            paired=np.array([ends[offset+i] and any(images[offset+i]==x+sep+y for x in aliases[a] for y in aliases[b])
                             for i,(_,a,b) in enumerate(pairs)])
            assert (int(paired.sum()),int(paired.reshape(-1,2).all(1).sum()))==(report[wrapper]["exact"],report[wrapper]["both_orders"])
            assert int(ends[offset:offset+len(pairs)].sum())==report[wrapper]["stopped"]
            offset+=len(pairs)
        assert offset==len(images)
        ok_by_arm[arm]=ok
    return ok_by_arm,offset*len(ARM_NAMES)


def examples(drcd,indices):
    paths = {"mixed":OUT/"mixed.safetensors", "replay":OUT/"replay.safetensors","released":INITIAL}
    models = {}
    for arm,path in paths.items():
        models[arm]=OrderedPassageComposer().eval()
        load_model(models[arm],str(path))
    source=load(ROOT/"data/raw/drcd/DRCD_dev.json")
    rows={qa["id"]:(para["context"],qa) for article in source["data"] for para in article["paragraphs"] for qa in para["qas"]}
    render=Rasterizer()
    arms=load(OUT/"result.json")
    archives={a:archived(OUT/(a+"_drcd.npz"),arms[a]["scores"]["drcd"]["archive_sha256"]) for a in ARM_NAMES}
    result={}
    for label,i in indices.items():
        text,qa=rows[drcd.ids[i]]
        p=drcd.bank[drcd.passages[drcd.groups[i]]]
        q=drcd.bank[drcd.questions[i]]
        allowed=sorted({(b-a,int(a),int(b)) for a,b in drcd.intervals[i]})
        _,a,b=allowed[0]
        ref=p[a:b].tobytes()
        ink=dict(passage=p.tobytes(),question=q.tobytes(),reference=ref)
        entry=dict(row=i,id=drcd.ids[i],question=qa["question"],reference=text[a:b],answers={})
        path=OUT/"examples"/label
        path.mkdir(parents=True,exist_ok=False)
        for arm,net in models.items():
            pixels,stop=net.answer(p,q)
            saved,stops=archives[arm]
            assert (pixels,stop)==(saved[i],stops[i])
            ink[arm]=pixels
            starts=[j for j in range(len(p)) if len(pixels) and p[j:j+len(pixels)//128].tobytes()==pixels]
            if starts:
                answer=text[starts[0]:starts[0]+len(pixels)//128]
                assert b"".join(map(render.tile,answer))==pixels
            else:
                answer=None
            entry["answers"][arm]=dict(stopped=bool(stop),cells=len(pixels)//128,source_transcription=answer)
        for key,pixels in ink.items():
            page(pixels).save(path/(key+".png"))
            if key in ("question","mixed","replay","reference"):
                page(pixels,width=42).save(EVIDENCE/(label+"_"+key+".png"))
        for key,original in (("passage",p),("question",q)):
            assert bitmap(path/(key+".png"),len(original))==original.tobytes()
        entry["passage_cells"],entry["question_cells"]=len(p),len(q)
        write_json(path/"example.json",entry)
        result[label]=entry
    return result


def main():
    EVIDENCE.mkdir(exist_ok=False)
    report=load(OUT/"result.json")
    torch.set_num_threads(2)
    drcd=TraditionalPassages("development")
    data=drcd.for_scoring()
    ok,n1=assess("drcd",report,data,DRCD_PREFIXES)
    official.WRAPPERS=CMRC_PREFIXES
    cmrc=official.prepare()
    _,n2=assess("cmrc",report,cmrc,CMRC_PREFIXES)
    internal=PairedPassages(ROOT/"data/paired_passage_questions_v1","development")
    int_targets=[internal.target(i) for i in range(len(internal.pairs))]+list(internal.answers)
    for arm in ARM_NAMES:
        rec=report[arm]["scores"]["internal"]
        pred,stops=archived(OUT/(arm+"_internal.npz"),rec["archive_sha256"])
        grades=np.array([s and im==t for s,im,t in zip(stops,pred,int_targets,strict=True)])
        n=len(internal.pairs)
        assert (int(grades[:n].sum()),int(grades[:n].reshape(-1,2).all(1).sum()),int(grades[n:].sum()))==(
            rec["ordered"],rec["both"],rec["singles"])
    old=load(ROOT/"references/evidence/paired_passage_20261119/ordered_decoder.json")
    assert report["released"]["scores"]["cmrc"]==old["official"]
    source=load(ROOT/"data/raw/drcd/DRCD_dev.json")
    ids={q["id"]:a["id"] for a in source["data"] for p in a["paragraphs"] for q in p["qas"]}
    groups=[ids[qid] for qid in drcd.ids]
    intervals={a:clustered_delta(ok["mixed"],ok[a],groups) for a in ("released","replay")}
    gained=np.flatnonzero(ok["mixed"] & ~ok["replay"])
    missed=np.flatnonzero(~ok["mixed"])
    indices={"missed":int(missed[0])}
    if len(gained):
        indices["gained"]=int(gained[0])
    cases=examples(drcd,indices)
    baseline=report["released"]["scores"]
    scores=report["mixed"]["scores"]
    gate=dict(drcd_singles=all(intervals[a]["difference_pp"]>=5 for a in intervals),
              drcd_both=all(scores["drcd"][DRCD_PREFIXES[1]]["both_orders"]>report[a]["scores"]["drcd"][DRCD_PREFIXES[1]]["both_orders"] for a in intervals),
              cmrc_single=scores["cmrc"]["single_exact"]>=baseline["cmrc"]["single_exact"]-.01*report["cmrc_questions"],
              cmrc_both=scores["cmrc"][CMRC_PREFIXES[1]]["both_orders"]>=baseline["cmrc"][CMRC_PREFIXES[1]]["both_orders"]-.01*report["cmrc_pairs"]/2,
              internal_pairs=scores["internal"]["ordered"]>=430,internal_singles=scores["internal"]["singles"]>=420)
    paired=data[3]
    specificity={}
    for arm in ARM_NAMES:
        pred,stops=archived(OUT/(arm+"_drcd.npz"),report[arm]["scores"]["drcd"]["archive_sha256"])
        specificity[arm]=dict(correct_own=sum(int(stops[a] and pred[a] in data[2][a]) for _,a,b in paired),
                              correct_sibling=sum(int(stops[a] and pred[a] in data[2][b]) for _,a,b in paired),
                              directed_questions=len(paired))
    audit=dict(comparison=intervals,admission=gate,passed=all(gate.values()),examples=cases,
               question_exchange=specificity,archives_regraded=n1+n2+len(int_targets)*3)
    write_json(EVIDENCE/"audit.json",audit)
    write_json(EVIDENCE/"results.json",report)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
