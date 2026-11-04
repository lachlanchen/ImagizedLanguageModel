#!/usr/bin/env python3
"""Locked matched free-image language audit; no OCR or output repair."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.correlated_native_runtime import generate_correlated, load_correlated
from ilm.visual_lm.native_pixel_instruction import image_cells
from ilm.visual_lm.question_first_reading import blank_question_images
from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from ilm.visual_lm.visual_qa_evaluation import answer_scores, image_keys, save_answer_image
from scripts.eval_native_pixel_instruction_v61 import foreground_f1
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save
from scripts.train_correlated_native_density_v62 import configure_gpu
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT, DEFAULT_DATA, ImageData
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES


def bootstrap(values,articles):
    values,articles=np.asarray(values,float),np.asarray(articles)
    rows=[np.flatnonzero(articles==a) for a in sorted(set(articles))]
    totals=np.asarray([values[x].sum() for x in rows]);counts=np.asarray([len(x) for x in rows])
    draws=np.random.default_rng(20266229).integers(0,len(rows),(2000,len(rows)))
    means=totals[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
                articles=len(rows),examples=len(values),replicates=2000,seed=20266229)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default="artifacts/correlated_native_density_v62_20261105")
    parser.add_argument("--snapshot",default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--data",default="data/visual_qa/cmrc2018_short")
    parser.add_argument("--cache",default=DEFAULT_DATA)
    args=parser.parse_args();run,data=Path(args.run),Path(args.data)
    out=run/"development";out.mkdir(parents=True,exist_ok=True)
    if (out/"report.json").exists():raise FileExistsError("completed matched audit is immutable")
    preflight=json.loads((run/"preflight/report.json").read_text())
    if not all(preflight["checks"].values()):raise ValueError("real preflight must pass")
    trainings,replays,weights={},{},{}
    for k in (1,8):
        row=json.loads((run/f"k{k}/training_report.json").read_text())
        if row["smoke_only"] or row["update"]!=1000 or row["components"]!=k:
            raise ValueError("both matched production arms must finish")
        if row["provenance"]!=preflight["provenance"]:raise ValueError("preflight differs")
        if any(sha256(name)!=digest for name,digest in row["provenance"].items()):
            raise ValueError("trained source changed")
        weights[str(k)]=sha256(run/f"k{k}/model_final.pt")
        replay=json.loads((run/f"standalone/k{k}/verification.json").read_text())
        if replay["checkpoint_sha256"]!=weights[str(k)] or not all(replay["checks"].values()):
            raise ValueError("offline replay must pass")
        trainings[str(k)],replays[str(k)]=row,replay
    if (run/"k1/sample_order.jsonl").read_bytes()!=(run/"k8/sample_order.jsonl").read_bytes():
        raise ValueError("matched training sample order differs")
    for name,digest in DATA_HASHES.items():
        if sha256(data/name)!=digest:raise ValueError("question data changed")
    identity=dict(weights_sha256=weights,data_sha256=DATA_HASHES,
        preflight_sha256=sha256(run/"preflight/report.json"),
        source_sha256={name:sha256(name) for name in [__file__,"ilm/visual_lm/correlated_native_runtime.py",
            "ilm/visual_lm/visual_qa_evaluation.py","ilm/visual_lm/question_first_reading.py",
            "scripts/eval_native_pixel_instruction_v61.py"]})
    if (out/"identity.json").exists() and json.loads((out/"identity.json").read_text())!=identity:
        raise ValueError("partial audit identity changed")
    _atomic_json(identity,out/"identity.json")
    device=configure_gpu();cache=ImageData(args.cache);renderer=VisualQARenderer()
    questions=sorted(read_questions(data/"development.jsonl"),key=lambda r:r["id"])
    train=read_questions(data/"train.jsonl")
    overlap={key:len({r[key] for r in train}&{r[key] for r in questions}) for key in ("article_title","context_hash")}
    if any(overlap.values()):raise ValueError("source articles overlap")
    primary=[questions[i] for i in np.random.default_rng(20265720).choice(len(questions),256,replace=False)]
    fixed=json.loads(Path("artifacts/question_first_reading_v60_20261105/development/report.json").read_text())["primary_ids"]
    pairs=[json.loads(line) for line in (data/"counterfactual_pairs.jsonl").read_text().splitlines()]
    if [r["id"] for r in primary]!=fixed or len(pairs)!=393:raise ValueError("fixed audit cases differ")
    by_id={r["id"]:r for r in questions}
    answers,likelihood,loading,pair_results={},{},{},{}
    begun=time.perf_counter()

    def predict(model,record,route):
        folder=out/"predictions"/route;folder.mkdir(parents=True,exist_ok=True)
        path=folder/(record["id"]+".pt")
        if path.exists():return torch.load(path,map_location="cpu",weights_only=False)
        pixels,boundary=cache.get("development/"+record["id"]+".pt")
        prompt=image_cells(pixels[...,:boundary*64])[:len(record["prompt"])]
        if not torch.equal(prompt,renderer(record["prompt"])):raise ValueError("cached prompt pixels differ")
        if route.endswith("_blank"):prompt=blank_question_images(prompt,record,"context")
        result=generate_correlated(model,prompt.to(device));images=result.pop("images").cpu()
        refs=[renderer(ref["text"]) for ref in record["references"]];keys=image_keys(images)
        row=dict(id=record["id"],article=record["article_title"],images=images,
            pixel_foreground_f1=foreground_f1(images,refs),blank_answer=float(not len(images)),
            adjacent_repeat=float(np.mean([x==y for x,y in zip(keys,keys[1:])])) if len(keys)>1 else 0.)
        row.update(result);row.update(answer_scores(images,refs,stopped=result["stopped"]))
        _atomic_save(row,path);save_answer_image(images,path.with_suffix(".png"));return row

    for k in (1,8):
        arm=f"k{k}";t=time.perf_counter()
        model=load_correlated(run/arm/"model_final.pt",args.snapshot,device)
        loading[arm]=time.perf_counter()-t
        ll=[]
        for record in questions:
            pixels,boundary=cache.get("development/"+record["id"]+".pt")
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                loss=model.answer_nll(pixels.to(device),boundary)
            ll.append(dict(id=record["id"],answer_pixel_nll=float(loss)))
        likelihood[arm]=ll;_atomic_json(ll,out/(arm+"_likelihood.json"))
        for blank in (False,True):
            route=arm+("_blank" if blank else "");rows=[]
            for i,record in enumerate(primary):
                rows.append(predict(model,record,route))
                if i%32==0:print(json.dumps(dict(route=route,completed=i+1)),flush=True)
            answers[route]=rows
            _atomic_json([{key:v for key,v in row.items() if key!="images"} for row in rows],out/(route+"_answers.json"))
        pair_rows=[]
        for i,pair in enumerate(pairs):
            rows=[predict(model,by_id[key],arm) for key in pair["record_ids"]]
            pair_rows.append(dict(record_ids=pair["record_ids"],context_hash=pair["context_hash"],
                                  both_exact=float(all(r["exact_answer"] for r in rows))))
            if i%64==0:print(json.dumps(dict(arm=arm,pairs_completed=i+1)),flush=True)
        pair_results[arm]=pair_rows;_atomic_json(pair_rows,out/(arm+"_paired_answers.json"))
        del model;torch.cuda.empty_cache()
    metrics=("exact_answer","raster_f1","pixel_foreground_f1","output_cells","stopped","capped","blank_answer","adjacent_repeat")
    measured={route:{metric:float(np.mean([r[metric] for r in rows])) for metric in metrics}
              for route,rows in answers.items()}
    comparisons={}
    for left,right in (("k8","k1"),("k8","k8_blank"),("k1","k1_blank")):
        for metric in ("exact_answer","raster_f1","pixel_foreground_f1"):
            comparisons[f"{left}_minus_{right}_{metric}"]=bootstrap(
                [x[metric]-y[metric] for x,y in zip(answers[left],answers[right])],[r["article"] for r in answers[left]])
    usage={route:dict(selected_experts=dict(Counter(str(e["expert"]) for row in rows for e in row["events"])),
        mean_gate_entropy=float(np.mean([e["gate_entropy"] for row in rows for e in row["events"]])))
        for route,rows in answers.items()}
    pair_accuracy={arm:float(np.mean([row["both_exact"] for row in rows])) for arm,rows in pair_results.items()}
    elapsed=time.perf_counter()-begun-sum(loading.values());peak=torch.cuda.max_memory_allocated(device)/1024**3
    implementation=dict(image_only=all(not any(row["boundary"].values()) for row in trainings.values()),
        split_resource_finite=not any(overlap.values()) and peak<20 and elapsed<4*3600 and
            all(np.isfinite(row["answer_pixel_nll"]) for rows in likelihood.values() for row in rows),
        backward_resume_causality=all(preflight["checks"].values()),offline_replay=all(all(r["checks"].values()) for r in replays.values()),
        matched_sample_order=True)
    intervention=dict(**implementation,matched_raster_gain=comparisons["k8_minus_k1_raster_f1"]["ci95"][0]>0,
        question_dependence=comparisons["k8_minus_k8_blank_raster_f1"]["ci95"][0]>0,capped_outputs=measured["k8"]["capped"]<.2)
    reading={}
    for arm in ("k1","k8"):
        reading[arm]=dict(**implementation,exact_answers=measured[arm]["exact_answer"]>=.1,
            raster_f1=measured[arm]["raster_f1"]>=.25,
            question_dependence=comparisons[f"{arm}_minus_{arm}_blank_raster_f1"]["ci95"][0]>0,
            paired_answers=pair_accuracy[arm]>=.02,capped_outputs=measured[arm]["capped"]<.2)
    report=dict(**identity,measurements=measured,comparisons=comparisons,expert_usage=usage,
        paired_both_exact=pair_accuracy,pairs=len(pairs),density_intervention_gates=intervention,free_reading_gates=reading,
        decision="density_intervention_pass" if all(intervention.values()) else "reject_density_intervention",
        free_reading_pass={arm:all(gates.values()) for arm,gates in reading.items()},
        likelihood={arm:float(np.mean([row["answer_pixel_nll"] for row in rows])) for arm,rows in likelihood.items()},
        primary_ids=fixed,split_overlap=overlap,development_reused=True,elapsed_seconds=elapsed,
        loading_seconds=loading,peak_allocated_vram_gib=peak,external_pretraining_is_not_local_training=True,
        qualification="general high-quality image language remains unqualified")
    _atomic_save({route:[r["images"] for r in rows[:8]] for route,rows in answers.items()},out/"first_primary_image_evidence.pt")
    _atomic_json(report,out/"report.json")
    print(json.dumps({key:report[key] for key in ("measurements","comparisons","expert_usage","decision","free_reading_pass")},indent=2),flush=True)


if __name__=="__main__":main()
