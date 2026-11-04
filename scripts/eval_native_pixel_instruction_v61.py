#!/usr/bin/env python3
"""Locked free-pixel answer audit after complete native instruction training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ilm.visual_lm.native_pixel_instruction import answer_only_bce, generate, image_cells
from ilm.visual_lm.native_pixel_runtime import load_trained
from ilm.visual_lm.question_first_reading import blank_question_images
from ilm.visual_lm.visual_qa_data import VisualQARenderer, read_questions
from ilm.visual_lm.visual_qa_evaluation import answer_scores, image_keys, save_answer_image
from scripts.prepare_native_pixel_qa_v61 import sha256
from scripts.train_canonical_glyph_binding_v43 import _atomic_json, _atomic_save
from scripts.train_native_pixel_instruction_v61 import DEFAULT_SNAPSHOT, DEFAULT_DATA, ImageData, load_base
from scripts.train_visual_answer_blocks_v58 import DATA_HASHES


def bootstrap(values, articles):
    values, articles = np.asarray(values, float), np.asarray(articles)
    rows = [np.flatnonzero(articles == a) for a in sorted(set(articles))]
    totals = np.asarray([values[x].sum() for x in rows])
    counts = np.asarray([len(x) for x in rows])
    draws = np.random.default_rng(20266219).integers(0, len(rows), (2000, len(rows)))
    means = totals[draws].sum(1)/counts[draws].sum(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025,.975]).tolist(),
                articles=len(rows), examples=len(values), replicates=2000, seed=20266219)


def foreground_f1(output, references):
    values = []
    for reference in references:
        n = max(len(output), len(reference))
        x = torch.zeros(n, 1, 32, 32); y = x.clone()
        x[:len(output)] = output; y[:len(reference)] = reference
        values.append(float(2*(x*y).sum()/(x.sum()+y.sum()).clamp(min=1)))
    return max(values)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="artifacts/native_pixel_instruction_v61_20261105")
    p.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    p.add_argument("--data", default="data/visual_qa/cmrc2018_short")
    p.add_argument("--cache", default=DEFAULT_DATA)
    a = p.parse_args()
    run, data = Path(a.run), Path(a.data)
    out = run/"development"; out.mkdir(parents=True, exist_ok=True)
    if (out/"report.json").exists():
        raise FileExistsError("completed native audit is immutable")
    trained = run/"training/model_final.pt"
    training = json.loads((run/"training/training_report.json").read_text())
    if training["smoke_only"] or training["update"] != 1000:
        raise ValueError("native instruction training must finish before audit")
    for name, digest in training["provenance"].items():
        if sha256(name) != digest:
            raise ValueError("trained native source changed")
    for name, digest in DATA_HASHES.items():
        if sha256(data/name) != digest:
            raise ValueError("prepared question dataset changed")
    preflight = json.loads(Path("artifacts/native_pixel_instruction_v61_preflight.json").read_text())
    if preflight["provenance"] != training["provenance"] or not all(preflight["checks"].values()):
        raise ValueError("native backward/resume/causality preflight differs")
    replay = json.loads((run/"standalone/verification.json").read_text())
    trained_sha = sha256(trained)
    if replay["checkpoint_sha256"] != trained_sha or not all(replay["checks"].values()):
        raise ValueError("native exported-model image replay must pass before audit")
    identity = dict(checkpoint_sha256=trained_sha, data_sha256=DATA_HASHES,
        preflight_sha256=sha256("artifacts/native_pixel_instruction_v61_preflight.json"),
        replay_sha256=sha256(run/"standalone/verification.json"),
        source_sha256={name:sha256(name) for name in [__file__,
            "ilm/visual_lm/native_pixel_runtime.py", "ilm/visual_lm/native_pixel_instruction.py",
            "ilm/visual_lm/visual_qa_evaluation.py"]})
    path = out/"audit_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("partial native audit identity changed")
    _atomic_json(identity, path)
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0"); torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    cache = ImageData(a.cache); renderer = VisualQARenderer()
    questions = sorted(read_questions(data/"development.jsonl"), key=lambda r:r["id"])
    train = read_questions(data/"train.jsonl")
    overlap = {k:len({r[k] for r in train}&{r[k] for r in questions})
               for k in ("article_title", "context_hash")}
    if any(overlap.values()):
        raise ValueError("source articles overlap")
    primary = [questions[i] for i in np.random.default_rng(20265720).choice(len(questions),256,replace=False)]
    by_id = {r["id"]:r for r in questions}
    pairs = [json.loads(s) for s in (data/"counterfactual_pairs.jsonl").read_text().splitlines()]
    old = json.loads(Path("artifacts/question_first_reading_v60_20261105/development/report.json").read_text())
    if [r["id"] for r in primary] != old["primary_ids"] or len(pairs) != 393:
        raise ValueError("fixed primary IDs or pairs differ")
    answers, likelihood, loading = {}, {}, {}
    begun = time.perf_counter()

    def predict(model, record, route):
        folder = out/"predictions"/route; folder.mkdir(parents=True,exist_ok=True)
        path = folder/(record["id"]+".pt")
        if path.exists():
            return torch.load(path,map_location="cpu",weights_only=False)
        pixels, boundary = cache.get("development/"+record["id"]+".pt")
        prompt = image_cells(pixels[...,:boundary*64])[:len(record["prompt"])]
        if not torch.equal(prompt, renderer(record["prompt"])):
            raise ValueError("scoring renderer differs from cached prompt images")
        if route.endswith("_blank"):
            prompt = blank_question_images(prompt,record,"context")
        result = generate(model,prompt.to(device))
        images = result.pop("images").cpu()
        references = [renderer(ref["text"]) for ref in record["references"]]
        keys = image_keys(images)
        row = dict(id=record["id"],article=record["article_title"],images=images,
                   pixel_foreground_f1=foreground_f1(images,references),
                   blank_answer=float(not len(images)),
                   adjacent_repeat=float(np.mean([x==y for x,y in zip(keys,keys[1:])])) if len(keys)>1 else 0.)
        row.update(result);row.update(answer_scores(images,references,stopped=result["stopped"]))
        _atomic_save(row,path);save_answer_image(images,path.with_suffix(".png"))
        return row

    for arm in ("trained","base"):
        t = time.perf_counter()
        model = (load_trained(trained,a.snapshot,device) if arm=="trained"
                 else load_base(a.snapshot,device).requires_grad_(False).eval())
        loading[arm] = time.perf_counter()-t
        ll=[]
        for record in questions:
            pixels,boundary=cache.get("development/"+record["id"]+".pt");pixels=pixels.to(device)
            mask=torch.ones(1,pixels.shape[-1]//64,device=device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                logits=model(inputs_embeds=pixels,attention_mask=mask,use_cache=False).logits
                loss=answer_only_bce(logits,pixels,boundary)
            ll.append(dict(id=record["id"],answer_pixel_nll=float(loss)))
        likelihood[arm]=ll;_atomic_json(ll,out/(arm+"_likelihood.json"))
        for blank in (False,True):
            route=arm+("_blank" if blank else "")
            rows=[]
            for i,record in enumerate(primary):
                rows.append(predict(model,record,route))
                if i%32==0:print(json.dumps(dict(route=route,completed=i+1)),flush=True)
            answers[route]=rows
            _atomic_json([{k:v for k,v in r.items() if k!="images"} for r in rows],out/(route+"_answers.json"))
        if arm=="trained":
            pair_rows=[]
            for i,pair in enumerate(pairs):
                rows=[predict(model,by_id[key],"trained") for key in pair["record_ids"]]
                pair_rows.append(dict(record_ids=pair["record_ids"],context_hash=pair["context_hash"],
                                      both_exact=float(all(r["exact_answer"] for r in rows))))
                if i%32==0:print(json.dumps(dict(pairs_completed=i+1)),flush=True)
            _atomic_json(pair_rows,out/"paired_answers.json")
        del model
        torch.cuda.empty_cache()
    measured={route:{k:float(np.mean([r[k] for r in rows])) for k in
        ("exact_answer","raster_f1","pixel_foreground_f1","output_cells","stopped","capped","blank_answer","adjacent_repeat")}
        for route,rows in answers.items()}
    comparisons={}
    for right in ("base","trained_blank"):
        for metric in ("exact_answer","raster_f1","pixel_foreground_f1"):
            comparisons[f"trained_minus_{right}_{metric}"]=bootstrap(
                [x[metric]-y[metric] for x,y in zip(answers["trained"],answers[right])],
                [r["article"] for r in answers["trained"]])
    pair_accuracy=float(np.mean([r["both_exact"] for r in pair_rows]))
    elapsed=time.perf_counter()-begun-sum(loading.values())
    peak=torch.cuda.max_memory_allocated(device)/1024**3
    gates=dict(image_only_free_runtime=all(training["boundary"][k] is False for k in
                   ("tokenizer","unicode_ids","ocr","glyph_inventory","external_model_call")),
        split_finite_resource=not any(overlap.values()) and peak<20 and elapsed<4*3600
            and all(np.isfinite(r["answer_pixel_nll"]) for ll in likelihood.values() for r in ll),
        full_backward_resume_causality=all(preflight["checks"].values()),native_export_replay=all(replay["checks"].values()),
        exact_answers=measured["trained"]["exact_answer"]>=.10,
        raster_f1=measured["trained"]["raster_f1"]>=.25,
        adaptation_gain=comparisons["trained_minus_base_raster_f1"]["ci95"][0]>0,
        question_dependence=comparisons["trained_minus_trained_blank_raster_f1"]["ci95"][0]>0,
        both_correct_pairs=pair_accuracy>=.02,capped_outputs=measured["trained"]["capped"]<.20)
    report=dict(**identity,measurements=measured,comparisons=comparisons,gates=gates,
        gates_passed=sum(gates.values()),decision="bounded_native_free_reading_pass" if all(gates.values()) else "reject_native_free_reading",
        paired_questions=dict(pairs=len(pairs),both_exact_fraction=pair_accuracy),
        likelihood={arm:float(np.mean([r["answer_pixel_nll"] for r in rows])) for arm,rows in likelihood.items()},
        primary_ids=[r["id"] for r in primary],elapsed_seconds=elapsed,loading_seconds=loading,
        peak_allocated_vram_gib=peak,split_overlap=overlap,development_reused=True,
        qualification="unrestricted pixel answers; general high-quality language remains unqualified",
        external_pretraining_is_not_local_training=True,
        likelihood_units="mean nats per answer/termination pixel per example")
    _atomic_save({route:[r["images"] for r in rows[:8]] for route,rows in answers.items()},out/"first_primary_image_evidence.pt")
    _atomic_json(report,out/"report.json")
    print(json.dumps({k:report[k] for k in ("measurements","comparisons","paired_questions","gates","decision")},indent=2),flush=True)


if __name__=="__main__":
    main()
