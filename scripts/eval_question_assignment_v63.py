#!/usr/bin/env python3
"""Locked free-image answer audit and separate reference-assignment diagnostic."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.correlated_native_runtime import generate_correlated,load_correlated
from ilm.visual_lm.native_pixel_instruction import image_cells
from ilm.visual_lm.question_assignment import image_assignment_energies
from ilm.visual_lm.question_first_reading import blank_question_images
from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
from ilm.visual_lm.visual_qa_evaluation import answer_scores,image_keys,save_answer_image
from scripts.eval_native_pixel_instruction_v61 import foreground_f1
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json,_atomic_save
from scripts.train_correlated_native_density_v62 import configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT,DEFAULT_DATA,ImageData
from scripts.train_question_assignment_v63 import SOURCES,INITIAL_SHA
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES


def bootstrap(values,articles):
    values,articles=np.asarray(values,float),np.asarray(articles)
    groups=[np.flatnonzero(articles==a) for a in sorted(set(articles))]
    sums=np.array([values[g].sum() for g in groups]);counts=np.array([len(g) for g in groups])
    draws=np.random.default_rng(20266329).integers(0,len(groups),(2000,len(groups)))
    means=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        examples=len(values),articles=len(groups),replicates=2000,seed=20266329)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/question_assignment_v63_20261105")
    run=Path(parser.parse_args().run);out=run/"development";out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():raise FileExistsError("completed audit is immutable")
    preflight=json.loads((run/"preflight/report.json").read_text())
    if not all(preflight["checks"].values()):raise ValueError("preflight must pass")
    expected={p:sha256(p) for p in SOURCES};trainings={};weights={};replays={}
    for arm in ("control","aligned"):
        row=json.loads((run/arm/"training_report.json").read_text())
        if row["smoke_only"] or row["update"]!=2000 or row["arm"]!=arm or row["assignment_weight"]!=float(arm=="aligned"):
            raise ValueError("both complete matched arms required")
        if row["provenance"]!=expected or row["provenance"]!=preflight["provenance"] or row["initialization_sha256"]!=INITIAL_SHA:
            raise ValueError("training identity changed")
        if row["data_sha256"]!=preflight["data_sha256"]:raise ValueError("training pairs changed")
        replay=json.loads((run/"standalone"/arm/"verification.json").read_text())
        weights[arm]=sha256(run/arm/"model_final.pt")
        if replay["checkpoint_sha256"]!=weights[arm] or not all(replay["checks"].values()):raise ValueError("offline replay failed")
        trainings[arm]=row;replays[arm]=replay
    if (run/"control/sample_order.jsonl").read_bytes()!=(run/"aligned/sample_order.jsonl").read_bytes():raise ValueError("matched data order differs")
    data=Path("data/visual_qa/cmrc2018_short")
    if any(sha256(data/p)!=h for p,h in DATA_HASHES.items()):raise ValueError("question source changed")
    identity=dict(checkpoint_sha256=weights,data_sha256=DATA_HASHES,preflight_sha256=sha256(run/"preflight/report.json"),
        source_sha256={p:sha256(p) for p in [__file__,"ilm/visual_lm/question_assignment.py",
            "ilm/visual_lm/correlated_native_runtime.py","ilm/visual_lm/visual_qa_evaluation.py",
            "ilm/visual_lm/question_first_reading.py","scripts/eval_native_pixel_instruction_v61.py"]})
    if (out/"identity.json").exists() and json.loads((out/"identity.json").read_text())!=identity:raise ValueError("partial audit identity changed")
    _atomic_json(identity,out/"identity.json")
    device=configure_gpu();cache=ImageData(DEFAULT_DATA);renderer=VisualQARenderer()
    questions=sorted(read_questions(data/"development.jsonl"),key=lambda r:r["id"])
    train=read_questions(data/"train.jsonl");by_id={r["id"]:r for r in questions}
    overlap={k:len({r[k] for r in train}&{r[k] for r in questions}) for k in ("context_hash","article_title")}
    if any(overlap.values()):raise ValueError("article/source split overlaps")
    fixed=json.loads(Path("artifacts/correlated_native_density_v62_20261105/development/report.json").read_text())["primary_ids"]
    primary=[questions[i] for i in np.random.default_rng(20265720).choice(len(questions),256,replace=False)]
    pairs=[json.loads(s) for s in (data/"counterfactual_pairs.jsonl").read_text().splitlines()]
    if [r["id"] for r in primary]!=fixed or len(pairs)!=393:raise ValueError("locked audit cases differ")
    begun=time.perf_counter();answers={};likelihood={};loading={};pair_results={}
    def predict(model,record,route):
        folder=out/"predictions"/route;folder.mkdir(parents=True,exist_ok=True);path=folder/(record["id"]+".pt")
        if path.exists():return torch.load(path,map_location="cpu",weights_only=False)
        pixels,boundary=cache.get("development/"+record["id"]+".pt")
        prompt=image_cells(pixels[...,:boundary*64])[:len(record["prompt"])]
        if not torch.equal(prompt,renderer(record["prompt"])):raise ValueError("prompt cache pixels differ")
        if route.endswith("_blank"):prompt=blank_question_images(prompt,record,"context")
        result=generate_correlated(model,prompt.to(device));images=result.pop("images").cpu()
        refs=[renderer(r["text"]) for r in record["references"]];keys=image_keys(images)
        row=dict(id=record["id"],article=record["article_title"],images=images,
            blank_answer=float(not len(images)),pixel_foreground_f1=foreground_f1(images,refs),
            adjacent_repeat=float(np.mean([a==b for a,b in zip(keys,keys[1:])])) if len(keys)>1 else 0.)
        row.update(result);row.update(answer_scores(images,refs,stopped=result["stopped"]))
        _atomic_save(row,path);save_answer_image(images,path.with_suffix(".png"));return row
    for arm in ("control","aligned"):
        t=time.perf_counter();model=load_correlated(run/arm/"model_final.pt",DEFAULT_SNAPSHOT,device)
        loading[arm]=time.perf_counter()-t;ll=[]
        for record in questions:
            pixels,boundary=cache.get("development/"+record["id"]+".pt")
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):loss=model.answer_nll(pixels.to(device),boundary)
            ll.append(dict(id=record["id"],answer_pixel_nll=float(loss)))
        likelihood[arm]=ll;_atomic_json(ll,out/(arm+"_likelihood.json"))
        for blank in (False,True):
            route=arm+("_blank" if blank else "");rows=[]
            for i,record in enumerate(primary):
                rows.append(predict(model,record,route))
                if i%32==0:print(json.dumps(dict(route=route,completed=i+1)),flush=True)
            answers[route]=rows;_atomic_json([{k:v for k,v in r.items() if k!="images"} for r in rows],out/(route+"_answers.json"))
        ranked=[]
        for i,pair in enumerate(pairs):
            records=[by_id[key] for key in pair["record_ids"]]
            generated=[predict(model,r,arm) for r in records]
            loaded=[cache.get("development/"+r["id"]+".pt") for r in records]
            prefixes=[p[...,:b*64] for p,b in loaded];suffixes=[p[...,b*64:] for p,b in loaded]
            streams=[(torch.cat((p,y),-1),p.shape[-1]//64) for p in prefixes for y in suffixes]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):e=image_assignment_energies(model,streams,device).float().cpu()
            if not bool(torch.isfinite(e).all()):raise FloatingPointError("nonfinite reference-assignment energy")
            margin=float(e[0,1]+e[1,0]-e[0,0]-e[1,1])
            ranked.append(dict(record_ids=pair["record_ids"],article=records[0]["article_title"],energies=e.tolist(),
                assignment_margin=margin,diagonal_assignment=float(margin>0),near_zero_margin=float(abs(margin)<=1e-6),
                both_energy_rank_correct=float(e[0,0]<e[0,1] and e[1,1]<e[1,0]),
                both_generated_exact=float(all(r["exact_answer"] for r in generated))))
            if i%64==0:print(json.dumps(dict(arm=arm,pairs_completed=i+1)),flush=True)
        pair_results[arm]=ranked;_atomic_json(ranked,out/(arm+"_pair_results.json"))
        del model;torch.cuda.empty_cache()
    metrics=("exact_answer","raster_f1","pixel_foreground_f1","output_cells","stopped","capped","blank_answer","adjacent_repeat")
    measured={route:{m:float(np.mean([r[m] for r in rows])) for m in metrics} for route,rows in answers.items()}
    comparisons={}
    for left,right in (("aligned","control"),("aligned","aligned_blank"),("control","control_blank")):
        for m in ("exact_answer","raster_f1","pixel_foreground_f1"):
            comparisons[f"{left}_minus_{right}_{m}"]=bootstrap([a[m]-b[m] for a,b in zip(answers[left],answers[right])],
                [r["article"] for r in answers[left]])
    paired={arm:float(np.mean([r["both_generated_exact"] for r in rows])) for arm,rows in pair_results.items()}
    ranking={arm:{k:float(np.mean([r[k] for r in rows])) for k in ("diagonal_assignment","near_zero_margin","both_energy_rank_correct","assignment_margin")} for arm,rows in pair_results.items()}
    elapsed=time.perf_counter()-begun-sum(loading.values());peak=torch.cuda.max_memory_allocated(device)/1024**3
    implementation=dict(image_only=all(not any(t["boundary"].values()) for t in trainings.values()),
        split_resource_finite=not any(overlap.values()) and peak<20 and elapsed<8*3600 and all(np.isfinite(r["answer_pixel_nll"]) for rows in likelihood.values() for r in rows),
        backward_resume_causality=all(preflight["checks"].values()),offline_replay=all(all(r["checks"].values()) for r in replays.values()),matched_sample_order=True)
    gates={arm:dict(**implementation,exact_answers=measured[arm]["exact_answer"]>=.1,raster_f1=measured[arm]["raster_f1"]>=.25,
        question_dependence=comparisons[f"{arm}_minus_{arm}_blank_raster_f1"]["ci95"][0]>0,
        paired_answers=paired[arm]>=.02,capped_outputs=measured[arm]["capped"]<.2) for arm in ("control","aligned")}
    passes={a:all(g.values()) for a,g in gates.items()};intervention=passes["aligned"] and comparisons["aligned_minus_control_raster_f1"]["ci95"][0]>0
    report=dict(**identity,measurements=measured,comparisons=comparisons,paired_both_exact=paired,reference_assignment=ranking,
        free_reading_gates=gates,free_reading_pass=passes,matched_raster_gain=comparisons["aligned_minus_control_raster_f1"]["ci95"][0]>0,
        decision="assignment_intervention_pass" if intervention else "reject_assignment_intervention",
        likelihood={a:float(np.mean([r["answer_pixel_nll"] for r in rows])) for a,rows in likelihood.items()},
        primary_ids=fixed,pairs=393,split_overlap=overlap,development_reused=True,loading_seconds=loading,
        elapsed_seconds=elapsed,peak_allocated_vram_gib=peak,external_pretraining_is_not_local_training=True,
        qualification="general high-quality image language remains unqualified")
    _atomic_json(report,out/"report.json");print(json.dumps({k:report[k] for k in ("measurements","reference_assignment","decision","free_reading_pass")},indent=2),flush=True)


if __name__=="__main__":main()
