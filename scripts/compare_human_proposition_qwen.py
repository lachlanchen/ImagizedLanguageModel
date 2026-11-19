"""Run the pinned 8B text reference on the same human question and bounded passage."""
import fcntl
import json
import os
from opencc import OpenCC
from scripts.compare_grounded_qwen import DIGEST,MODEL,SYSTEM,local_teacher
from scripts.evaluate_human_proposition_questions import cases,forms
from scripts.grounded_proposition_corpus import PUBLIC,WORK,ROOT,record,sha


if __name__=="__main__":
    _,meta,_,_=cases()
    path=WORK/"qwen_human.jsonl"
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    norm=OpenCC("t2s")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with local_teacher(MODEL,WORK) as (ask,digest,size),path.open("a") as f:
            assert digest==DIGEST
            for i,m in enumerate(meta):
                if i<len(rows):
                    assert rows[i]["key"]==m["key"];continue
                answer=ask(SYSTEM,dict(role="user",content=f"材料：{m['p']}\n问题：{m['q']}"),think=False,
                           options=dict(temperature=0,seed=34712,num_predict=80,num_ctx=1024,num_thread=2))
                row=dict(key=m["key"],**answer)
                f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
                rows.append(row)
    stats={}
    for m,a in zip(meta,rows,strict=True):
        g=stats.setdefault(m["group"],dict(n=0,reference=0,accepted=0))
        recorded=norm.convert(a["text"]).rstrip("。.!！")
        g["n"]+=1;g["reference"]+=norm.convert(m["o"]) in recorded
        g["accepted"]+=a["stop"]=="stop" and recorded in {norm.convert(v) for v in forms((m["s"],m["r"],m["o"]))}
    record(PUBLIC/"qwen_human.json",dict(model=MODEL,manifest=digest,scores=stats,ledger=sha(path),
                                        elapsed_requests_s=sum(r["duration_ns"] for r in rows)/1e9))
