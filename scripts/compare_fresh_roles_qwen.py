"""Evaluate the pinned local text reference on the 64 fresh Chinese role questions."""

import fcntl
import json
import os
from time import monotonic

from scripts.benchmark_visual_inference_qwen import DIGEST, MODEL, local_teacher
from scripts.compare_explicit_participants import SYSTEM, counts
from scripts.fit_font_invariant_reader import PUBLIC, ROOT, WORK, sha, write_json
from scripts.pictured_participant_scenes import fresh_cases


def main():
    rows=list(fresh_cases())
    path=WORK/"fresh_qwen.jsonl"
    stored=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    assert len(stored)<=len(rows)
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with local_teacher(MODEL,WORK) as (ask,digest,size),path.open("a") as f:
            assert digest==DIGEST
            start=monotonic()
            for i,(p,q,y) in enumerate(rows):
                if i<len(stored):
                    assert stored[i]["index"]==i
                    continue
                answer=ask(SYSTEM,dict(role="user",content=f"材料：{p}\n问题：{q}"),think=False,
                           options=dict(temperature=0,seed=91991,num_ctx=1024,num_predict=32,num_thread=2))
                item=dict(index=i,**answer)
                f.write(json.dumps(item,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
                stored.append(item)
                if monotonic()-start>600:raise TimeoutError("Text reference exceeded its allocation")
    flags=[s["text"]==y and s["stop"]=="stop" for s,(_,_,y) in zip(stored,rows,strict=True)]
    rec=dict(**counts(flags),archive=sha(path),model=MODEL,digest=digest,bytes=size,
             duration_s=sum(s["duration_ns"] for s in stored)*1e-9)
    write_json(PUBLIC/"qwen.json",rec)
    print(rec)


if __name__=="__main__":
    main()
