"""Measure optical translation decisions and separately archive the independent writer's pixels."""
from collections import Counter, defaultdict
from contextlib import nullcontext
import argparse

from scripts.readout_work_interval import work_interval
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers.modeling_outputs import BaseModelOutput

from ilm.visual_lm.glyph_measure import weighted_cross_attention
from ilm.visual_lm.portable_writer import read_image, save_grid
from scripts.evaluate_optical_state_reader import as_answer, replacement
from scripts.evaluate_proposition_translation import semantic
from scripts.evaluate_teacher_translation import paired
from scripts.evaluate_visual_library import BASE, blind_votes
from scripts.train_readout_objective import (
    ARMS, MODELS, OUTPUT, PUB, ROOT, RUN, OpticalStateReader, configure_cuda_cli,
    decoder_hidden, dump, gpu_owner, library, material, pack, project, read, sha, teacher_model,
)
from scripts.write_ordered_translation import restore, stopped

OPTICAL=("start",*ARMS)
MODES=(*OPTICAL,"native","ilm")
GEN=dict(do_sample=False,num_beams=1,max_new_tokens=128,forced_eos_token_id=None)


def restore_reader(t,arm):
    lang=("en","zh")[t]
    base=MODELS/lang if arm=="start" else OUTPUT/lang/arm
    weights=base/"weights.safetensors"
    if sha(weights)!=read(base/"fit.json")["weights"]: raise ValueError("Reader fingerprint")
    net=OpticalStateReader().cuda().eval()
    net.load_state_dict(load_file(str(weights)))
    return net


@torch.inference_mode()
def evaluate_direction(t,rows,sources,answers):
    teacher,tok=teacher_model(t)
    readers={a:restore_reader(t,a) for a in OPTICAL}
    f=material("cuda")[0]
    dev=library("dev")
    sums=Counter()
    dev_free=[]
    for j in range(0,len(dev),12):
        batch=dev[j:j+12]
        x=tok([r["text"][1-t] for r in batch],padding=True,return_tensors="pt").to("cuda")
        y=tok(text_target=[r["text"][t] for r in batch],padding=True,return_tensors="pt").input_ids.cuda()
        valid=y.ne(tok.pad_token_id)
        base=teacher.model.encoder(**x).last_hidden_state
        h=decoder_hidden(teacher,base,x.attention_mask,y)[valid]
        logp=project(teacher,h).float().log_softmax(-1)
        sums["positions"]+=len(logp)
        sums["top32_mass"]+=float(logp.topk(32).values.exp().sum())
        src=pack([r["cells"][1-t] for r in batch],"cuda")
        seq={"native":teacher.generate(**x,**GEN).cpu()}
        for a,net in readers.items():
            z,mask,mass=net.states(f,src)
            with weighted_cross_attention(teacher,mass):
                dec=decoder_hidden(teacher,z,mask,y)
                logq=project(teacher,dec[valid]).float().log_softmax(-1)
                sums[a]+=float((logp.exp()*(logp-logq)).double().sum())
                seq[a]=teacher.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z),attention_mask=mask,**GEN).cpu()
        dev_free.extend([{a:as_answer(v[i],tok) for a,v in seq.items()} for i in range(len(batch))])
    swap=replacement(rows)
    ordered=[i for i,r in enumerate(rows) if r["t"]==t]
    for j in range(0,len(ordered),12):
        ids=ordered[j:j+12]
        x=tok([rows[i]["source"] for i in ids],padding=True,return_tensors="pt").to("cuda")
        result={"native":teacher.generate(**x,**GEN).cpu()}
        for a in (*OPTICAL,"exchange"):
            ix=[swap[i] for i in ids] if a=="exchange" else ids
            net=readers["prediction" if a=="exchange" else a]
            z,mask,mass=net.states(f,pack([sources[i] for i in ix],"cuda"))
            with weighted_cross_attention(teacher,mass):
                result[a]=teacher.generate(encoder_outputs=BaseModelOutput(last_hidden_state=z),attention_mask=mask,**GEN).cpu()
        for n,i in enumerate(ids):
            answers[i].update({a:as_answer(v[n],tok) for a,v in result.items()})
    return dict(positions=sums["positions"],mean_retained_mass=sums["top32_mass"]/sums["positions"],
                full_kl={a:sums[a]/sums["positions"] for a in OPTICAL}),dev_free


@torch.inference_mode()
def write(rows):
    f,inv,_=material("cuda")
    bank=np.load(BASE/"palette.npy")
    labels=read(BASE/"readback.json")
    index={bytes(v):i for i,v in enumerate(bank) if i>=2}
    answers=[{} for r in rows]
    sources=[]
    net=restore("role-grounded","cuda")
    for i,r in enumerate(rows):
        folder=RUN/"pixels"/f"{i:04d}"
        folder.mkdir(parents=True,exist_ok=True)
        p=folder/"input.png"
        save_grid(bank[r["input"]].tobytes(),p)
        cells=[index[bytes(v)] for v in read_image(p,len(r["input"]))]
        if cells!=r["input"]: raise ValueError("Source pixel round trip")
        sources.append(cells)
    for t in (0,1):
        subset=[i for i,r in enumerate(rows) if r["t"]==t]
        for j in range(0,len(subset),24):
            ix=subset[j:j+24]
            stream=net.generate(f,pack([sources[i] for i in ix],"cuda"),inv[t],t).cpu()
            for i,s in zip(ix,stream,strict=True):
                cells,done=stopped(s)
                path=RUN/"pixels"/f"{i:04d}"/"ilm.png"
                save_grid(bank[np.asarray(cells,dtype=int)].tobytes(),path)
                answers[i]["ilm"]=dict(cells=cells,stop=done,text="".join(labels[str(c)] for c in cells),png=sha(path))
    dev={}
    for t,lang in enumerate(("en","zh")):
        dev[lang],free=evaluate_direction(t,rows,sources,answers)
        dump(RUN/(lang+"-dev-free.json"),free)
    return answers,dev


def summarize(rows,answers,votes):
    works=defaultdict(Counter)
    directions=defaultdict(Counter)
    records=[]
    for i,(r,a) in enumerate(zip(rows,answers,strict=True)):
        ok={m:bool(a[m]["stop"] and votes[2*i][m] and votes[2*i+1][m]) for m in votes[2*i]}
        for group in (works[r["doc"]],directions[("en","zh")[r["t"]]]):
            group.update(ok)
            group["n"]+=1
        records.append(ok)
    result=dict(works=works,directions=directions,total=sum(works.values(),Counter()),records=records)
    if "regression" in records[0]:
        result["paired_95"]=work_interval(works)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-saved",action="store_true",help="Recount the sealed outputs and votes on CPU")
    saved=parser.parse_args().score_saved
    if not saved:
        configure_cuda_cli()
    if sha(RUN/"fresh.json")!=read(PUB/"fresh.json")["sha256"]: raise ValueError("Reserved requests")
    rows=paired(read(RUN/"fresh.json"))
    assert len(rows)==192
    training=read(PUB/"training.json")
    for lang,arms in training.items():
        for a,digest in arms.items():
            if digest!=sha(OUTPUT/lang/a/"fit.json"): raise ValueError("Fit registration")
        a,b=[read(OUTPUT/lang/arm/"fit.json")["history"] for arm in ARMS]
        assert all(x["order_sha256"]==y["order_sha256"] and x["positions"]==y["positions"] for x,y in zip(a,b,strict=True))
    bind=dict(training=sha(PUB/"training.json"),panel=sha(RUN/"fresh.json"),program=sha(Path(__file__)))
    raw=RUN/"generated.json"
    if saved:
        previous=read(RUN/"before-exact-percentile.json")
        if previous["raw"]!=sha(raw) or any(sha(RUN/name)!=value for name,value in previous["votes"].items()):
            raise ValueError("Original generation or grades changed")
        # The previous evaluator calculated percentile endpoints with binary floats.
        bind["program"]=previous["binding"]["program"]
    with nullcontext() if saved else gpu_owner():
        torch.set_num_threads(2)
        if not raw.exists():
            answers,development=write(rows)
            dump(raw,dict(binding=bind,answers=answers,dev=development))
        gen=read(raw)
        if gen["binding"]!=bind: raise ValueError("Changed output provenance")
        if not saved:
            torch.cuda.empty_cache()
        guided=RUN/"reference.json"
        if not guided.exists():
            a=[{m:r[m] for m in MODES} for r in gen["answers"]]
            grades=semantic(rows,a,MODES)
            dump(guided,dict(raw=sha(raw),answers=a,votes=grades["votes"],judge=grades["judge"],qwen=grades["qwen"]))
        marked=read(guided)
        if marked["raw"]!=sha(raw): raise ValueError("Grading mismatch")
        scopes={"source":marked["answers"],"exchange":[dict(prediction=r["prediction"],exchange=z["exchange"],
                reference=r["reference"],foil=r["foil"]) for r,z in zip(marked["answers"],gen["answers"],strict=True)]}
        results={}
        for name,group in scopes.items():
            path=RUN/(name+"-votes.json")
            if not path.exists():
                dump(path,dict(reference=sha(guided),votes=blind_votes(rows,group)))
            votes=read(path)
            assert votes["reference"]==sha(guided)
            results[name]=summarize(rows,group,votes["votes"])
        results["guided"]=summarize(rows,marked["answers"],marked["votes"])
    source=results["source"]
    totals=source["total"]
    checks=dict(more=totals["prediction"]>max(totals["start"],totals["regression"]),
                positive_interval=source["paired_95"][0]>0,
                lower_full_kl=all(d["full_kl"]["prediction"]<d["full_kl"]["regression"] for d in gen["dev"].values()))
    result=dict(binding=bind,scorer=sha(Path(__file__)),exact_interval=sha(ROOT/"scripts/readout_work_interval.py"),raw=sha(raw),votes={p.name:sha(p) for p in (guided,RUN/"source-votes.json",RUN/"exchange-votes.json")},
                dev=gen["dev"],stops={m:sum(a[m]["stop"] for a in marked["answers"]) for m in marked["answers"][0]},
                exchange_stops=sum(a["exchange"]["stop"] for a in gen["answers"]),
                qualification=checks,passed=all(checks.values()),**results)
    dump(PUB/"evaluation.json",result)
    print(totals,gen["dev"],checks,flush=True)


if __name__=="__main__":
    main()
