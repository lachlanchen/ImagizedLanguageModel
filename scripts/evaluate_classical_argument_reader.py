"""Score original and edited classical source passages after argument training."""

import argparse
import torch

from scripts.prepare_classical_argument_reader import PUBLIC,WORK,WEIGHTS,read,sha,record
from scripts.evaluate_role_question_binding import populations as first_populations,decode,scores
from scripts.prepare_role_question_binding import PUBLIC as PREVIOUS, WORK as LAST
from scripts.train_span_choice import gpu_owner


def populations():
    path=WORK/"reserved.json";assert sha(path)==read(PUBLIC/"corpus.json")["test_sha"]
    rows=read(path)
    train=WORK/"learn.json";assert sha(train)==read(PUBLIC/"corpus.json")["learn_sha"]
    for kind in ("single","joined"):
        rows.extend(dict(**r,group="train_"+kind,key=f"train-{kind}-{i}") for i,r in enumerate([r for r in read(train) if r["kind"]==kind][:24]))
    for r in rows:
        for w,p in enumerate(r["pages"]):
            for lang in ("zh","en"):
                for h in (0,1):
                    yield dict(group=r["group"],key=r["key"],world=w,lang=lang,h=h,page=p,
                               question=r["qs"][h][lang],answer=r["qs"][h^w]["answer"],other=r["qs"][1^h^w]["answer"])
    for r in first_populations():
        if r["group"]=="novels":yield r


def evaluate(arm):
    cases=list(populations())
    trained="aligned" if arm=="without_question" else arm
    f=WEIGHTS if arm=="original" else WORK/trained/"reader.safetensors"
    expected=None if arm=="text" else (read(PUBLIC/"corpus.json")["initial"] if arm=="original" else read(PUBLIC/(trained+"_fit.json"))["weights"])
    if arm=="text":cases=[r for r in cases if r["group"]=="zuo"]
    details,cost=decode(arm,cases,WORK,f,expected,measure_margin=True)
    if arm=="text":
        path=LAST/"text_responses.json";assert sha(path)==read(PREVIOUS/"text_scores.json")["response_sha"]
        details.extend(dict(r,group="shishuo") for r in read(path))
        cost["prior_shishuo_responses"]=sha(path)
    path=WORK/(arm+"_responses.json");record(path,details)
    record(PUBLIC/(arm+"_scores.json"),dict(counts=scores(details),response_sha=sha(path),**cost))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("arm",choices=("original","independent","aligned","without_question","text"))
    args=ap.parse_args()
    with gpu_owner():
        torch.set_num_threads(2);evaluate(args.arm)
