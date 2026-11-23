"""Time the released image route and the same trial questions as text, one request at a time."""

import json
import re
from time import monotonic
import torch

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.benchmark_interval_teacher import MODEL, SYSTEM, DIGEST, summarize
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import load_streams,store_streams
from scripts.local_translation_teacher import local_teacher
from scripts.typeface_writer_cases import extra_sources, pictures
from scripts.fit_font_invariant_reader import Face
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_typeface_writer import WORK,PUBLIC,FACES,HELD
from scripts.train_span_choice import gpu_owner


def run():
    directory=WORK/"serving"
    directory.mkdir(exist_ok=True)
    train_face,held=Face(FACES[0]),Face(HELD)
    rows,_=extra_sources()
    start=monotonic()
    cases=pictures(rows,held,train_face)
    pairs=[(tensor(c.p),tensor(c.q)) for c in cases]
    preparation=monotonic()-start
    ids={c.question_id for c in cases}
    rows=[v for v in rows if v["id"] in ids]
    assert [v["id"] for v in rows]==[c.question_id for c in cases]
    reference=json.loads((PUBLIC/"result.json").read_text())
    report={}
    for arm in ("replay","mapped"):
        model=PreparedGlyphResponder(WORK/arm/"page","cuda")
        warm=(tensor(held("小河流入大海。")),tensor(held("小河流入哪裡？")))
        model.answer_many([warm])
        torch.cuda.synchronize()
        start=monotonic()
        outputs=[]
        for item in pairs:
            ((im,stop),)=model.answer_many([item])
            outputs.append((im.numpy().tobytes(),bool(stop)))
        torch.cuda.synchronize()
        elapsed=monotonic()-start
        old=WORK/"evaluation"/arm/"trial_hana.npz"
        assert sha(old)==reference["scores"][arm]["trial_hana"]["sha256"]
        old_pixels,old_ends=load_streams(old)
        drift=sum((y,s)!=(old_y,old_s) for (y,s),old_y,old_s in zip(outputs,old_pixels,old_ends,strict=True))
        images,ends=zip(*outputs,strict=True)
        path=directory/(arm+".npz")
        store_streams(path,images,ends)
        exact=[c.grade(y,s)[2] for c,(y,s) in zip(cases,outputs,strict=True)]
        report[arm]=dict(**summarize(rows,exact),seconds=elapsed,batch_differences=drift,stream_hash=sha(path),
                         model_bytes=sum(p.stat().st_size for p in (WORK/arm/"page").iterdir()))
        del model
    torch.cuda.empty_cache()
    ledger=directory/"qwen.jsonl"
    answers=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    assert all(a["id"]==r["id"] for a,r in zip(answers,rows))
    with local_teacher(MODEL,directory) as (ask,digest,size), ledger.open("a") as f:
        assert digest==DIGEST
        def answer(p,q):
            return ask(SYSTEM,dict(role="user",content=f"材料：{p}\n问题：{q}"),think=False,
                       options=dict(seed=241179,temperature=0,num_predict=80,num_ctx=2048,num_thread=2))
        answer("小河流入大海。","小河流入哪裡？")
        for v in rows[len(answers):]:
            start=monotonic()
            a=answer(v["p"],v["q"])
            a.update(seconds=monotonic()-start,id=v["id"])
            f.write(json.dumps(a,ensure_ascii=False)+"\n");f.flush();answers.append(a)
    ok=[a["stop"]=="stop" and a["text"] in v["refs"] for a,v in zip(answers,rows,strict=True)]
    report["qwen"]=dict(**summarize(rows,ok),seconds=sum(a["seconds"] for a in answers),model=MODEL,model_bytes=size,
                        gpu_layers=re.findall(r"offloaded (\d+/\d+) layers to GPU",(directory/"server.log").read_text()),
                        ledger=sha(ledger),manifest=digest)
    record(PUBLIC/"serving.json",dict(methods=report,preparation_seconds=preparation))
    print(report,flush=True)


if __name__=="__main__":
    with gpu_owner():
        run()
