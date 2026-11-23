"""Recompute exact scores from returned images, with uncertainty by original excerpt."""

from collections import defaultdict

import numpy as np
import torch

from scripts.evaluate_role_question_binding import populations, scores
from scripts.prepare_role_question_binding import PUBLIC,WORK,WEIGHTS,read
from scripts.read_book_participant import answer,restore
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_typeface_writer import FACES,Face


def run():
    refs=list(populations());face=Face(FACES[0]);vectors={}
    images=0;replays=0;identical=0
    trained=[read(PUBLIC/(a+"_fit.json")) for a in ("aligned","human")]
    for k in ("initial","learn_sha","updates","questions","parameters"):assert trained[0][k]==trained[1][k]
    assert trained[0]["initial"]==sha(WEIGHTS)
    model=restore(WORK/"aligned/reader.safetensors","cpu")
    for arm in ("aligned","original","human","without_question","text"):
        path=WORK/(arm+"_responses.json");receipt=read(PUBLIC/(arm+"_scores.json"))
        assert sha(path)==receipt["response_sha"]
        results=read(path);cases=refs if arm!="text" else [c for c in refs if c["group"]=="classical"]
        assert scores(results)==receipt["counts"]
        grouped=defaultdict(lambda:defaultdict(list))
        for r,e in zip(results,cases,strict=True):
            assert all(r[k]==e[k] for k in ("group","key","world","lang","h"))
            is_image=arm!="text"
            pred=bytes.fromhex(r["reply"]) if is_image else r["reply"]
            if is_image:
                p=face(e["page"])
                assert 0<len(pred)<=1024 and len(pred)%128==0 and any(p.startswith(pred,k) for k in range(0,len(p),128))
                images+=1
                if arm=="aligned" and e["group"]=="classical":
                    replay=answer(model,p,face(e["question"]))
                    replays+=1;identical+=replay==pred
            gold=face(e["answer"]) if is_image else e["answer"]
            rival=face(e["other"]) if is_image else e["other"]
            assert r["correct"]==(r["ended"] and gold==pred)
            assert r["rival"]==(r["ended"] and rival==pred)
            grouped[e["group"],e["lang"]][e["key"]].append(r["correct"])
        for (group,lang),scenes in grouped.items():
            values=np.array([scenes[k] for k in sorted(scenes)])
            vectors[arm,group,lang,"individual"]=values.mean(1)
            vectors[arm,group,lang,"complete"]=values.all(1).astype(float)
    rng=np.random.default_rng(241134);contrasts={}
    for lang in ("zh","en"):
        sample=rng.integers(28,size=(10000,28))
        for metric in ("individual","complete"):
            for other in ("human","original"):
                delta=vectors["aligned","classical",lang,metric]-vectors[other,"classical",lang,metric]
                contrasts["/".join((lang,metric,other))]=dict(delta=float(delta.mean()),ci95=np.quantile(delta[sample].mean(1),[.025,.975]).tolist())
    record(PUBLIC/"audit.json",dict(images_verified=images,cpu_replayed=replays,cpu_identical=identical,bootstrap_unit="classical source excerpt",resamples=10000,contrasts=contrasts))


if __name__=="__main__":
    torch.set_num_threads(2);run()
