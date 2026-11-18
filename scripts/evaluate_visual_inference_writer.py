"""Score stopped visual decisions and real missing text with both inputs independently removable."""
import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_same_passage_writer import generate, grade, image_archive, reference_sets
from scripts.train_general_entry_writer import Entries, PairCorpus, small
from scripts.train_visual_inference_writer import (ARMS,DEVICE,PRIVATE,PUBLIC,START,WORK,Corpus,Rasterizer,
        load,load_visual_writer,memory_bank,physical_tables,pictured,sha,write_json)


def paired_interval(a,b,groups):
    _,inv=np.unique(groups,return_inverse=True)
    size=np.bincount(inv)
    diff=np.asarray(a,int)-np.asarray(b,int)
    totals=np.bincount(inv,weights=diff)
    rng=np.random.default_rng(32719)
    out=[]
    for _ in range(100):
        idx=rng.integers(len(size),size=(100,len(size)))
        out.extend(100*totals[idx].sum(1)/size[idx].sum(1))
    return dict(difference_pp=float(100*diff.mean()),interval95_pp=np.quantile(out,[.025,.975]).tolist(),
                groups=len(size),gained=int((diff>0).sum()),lost=int((diff<0).sum()))


@torch.inference_mode()
def evaluate():
    bank=memory_bank()
    train,dev=[Corpus(bank,name) for name in ("train","development")]
    entries=Entries(bank,"development")
    panels=prompts(bank,entries)
    bounds=pictured(bank,Rasterizer(),"【】")
    dr=PairCorpus(bank,"development")
    drids=np.unique(dr.rows[:,1:3]);drpairs=np.searchsorted(drids,dr.rows[:,1:3])
    drrefs=reference_sets(dr,drids,bank)
    du,groups=small(bank,"development")
    to_pixels=lambda s:b"".join(bank.tiles[t] for t in s)
    durefs=[{to_pixels(y)} for y in du.y]
    blank=np.array([bank.lookup[bytes(128)]])
    parts={}
    with np.load(PRIVATE) as split:
        for j,(name,ds) in enumerate((("validation",train),("official",dev))):
            rows=split[name]
            triples=np.searchsorted(rows,split[name+"_triples"])
            assert np.array_equal(rows[triples],split[name+"_triples"])
            good,holes,answers=ds.holes(rows,np.random.default_rng(73121+j))
            targets=[to_pixels(s) for s in answers]
            missing=np.array([bool(set(y)-set(ds.p[i])-set(q)) for y,q,i in zip(answers,holes,good,strict=True)])
            parts[name]=dict(ds=ds,rows=rows,triples=triples,good=good,holes=holes,span_targets=targets,
                absent=missing,labels=[to_pixels(ds.y[i]) for i in rows],shapes=np.array(list(map(len,answers))),
                clusters=np.array([ds.meta[i]["premise"] for i in rows]),mask_clusters=np.array([ds.meta[i]["premise"] for i in good]))
            image_archive(WORK/(name+"_cloze_targets.npz"),targets,[True]*len(good))
            image_archive(WORK/(name+"_cloze_requests.npz"),[to_pixels(q) for q in holes],[True]*len(good))
            np.save(WORK/(name+"_cloze_rows.npy"),good)
    features,mem,_=physical_tables(bank,DEVICE)
    records={};flags={}
    for arm in ("parent",*ARMS):
        folder=WORK/arm
        if arm=="parent":folder.mkdir()
        net=load_visual_writer(START if arm=="parent" else folder,DEVICE)
        ink=net.ink(features)
        keys=net.write_key(ink[mem]);rec={};marked={}
        for name,d in parts.items():
            ds,rows,valid=d["ds"],d["rows"],d["good"]
            tasks=[("decision",rows,[ds.q[i] for i in rows],d["labels"]),
                   ("restoration",valid,d["holes"],d["span_targets"])]
            for task,ids,questions,targets in tasks:
                modes=["intact","passage_erased"]
                if task=="decision":modes.append("claim_erased")
                for mode in modes:
                    tag=f"{name}_{task}_{mode}"
                    ps=[ds.p[i] for i in ids] if mode!="passage_erased" else [blank]*len(ids)
                    qs=questions if mode!="claim_erased" else [ds.decide]*len(ids)
                    imgs,stop=generate(net,ink,mem,keys,ps,qs,bank)
                    image_archive(folder/(tag+".npz"),imgs,stop)
                    ok=np.array([e and x==y for x,e,y in zip(imgs,stop,targets,strict=True)])
                    counts=dict(records=len(ok),exact=int(ok.sum()),stopped=sum(stop))
                    if task=="decision":
                        triple=ok[d["triples"]].all(1)
                        counts.update(three=int(triple.sum()),three_total=len(triple))
                        if mode=="claim_erased":assert not triple.any()
                        choices=set(targets)
                        counts["valid_sentence"]=sum(e and x in choices for x,e in zip(imgs,stop,strict=True))
                    else:
                        counts["lengths"]={str(n):[int(ok[d["shapes"]==n].sum()),int((d["shapes"]==n).sum())] for n in (1,2,3)}
                        counts.update(absent=int(ok[d["absent"]].sum()),absent_total=int(d["absent"].sum()))
                    rec[tag]=counts;marked[tag]=ok
        for tag in ("english_free","chinese_free"):
            qs,rows=panels[tag]
            ps=[select_visible_record(entries.cards[entries.card_of[i]],q,*bounds)[0] for q,i in zip(qs,rows,strict=True)]
            imgs,ends=generate(net,ink,mem,keys,ps,qs,bank)
            image_archive(folder/(tag+".npz"),imgs,ends)
            rec[tag]=grade(imgs,ends,[{to_pixels(entries.y[i])} for i in rows])[0]
        for tag,ps,qs,refs,g in (("dureader",du.p,du.q,durefs,groups),
                ("drcd",[dr.passages[dr.groups[i]] for i in drids],[dr.questions[i] for i in drids],drrefs,None)):
            imgs,ends=generate(net,ink,mem,keys,ps,qs,bank)
            image_archive(folder/(tag+".npz"),imgs,ends)
            rec[tag],ok=grade(imgs,ends,refs,g)
            if tag=="drcd":rec[tag]["both"]=int(ok[drpairs].all(1).sum())
        records[arm]=rec;flags[arm]=marked
        print(arm,{k:v["exact"] for k,v in rec.items() if k.startswith("official")},flush=True)
    effects={}
    for name,d in parts.items():
        for task,g in (("decision",d["clusters"]),("restoration",d["mask_clusters"])):
            prefix=f"{name}_{task}_"
            matched=flags["paired"][prefix+"intact"]
            for control,comp in (("mispaired",flags["mispaired"][prefix+"intact"]),
                                 ("passage_erased",flags["paired"][prefix+"passage_erased"])):
                effects[prefix+control]=paired_interval(matched,comp,g)
                if task=="restoration":
                    ids=d["absent"]
                    effects[prefix+control+"_absent"]=paired_interval(matched[ids],comp[ids],g[ids])
    write_json(PUBLIC/"result.json",dict(arms=records,contrasts=effects,fits={a:load(WORK/a/"fit.json") for a in ARMS},
        archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.np*")}))
