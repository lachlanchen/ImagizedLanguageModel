"""Answer the independently phrased paired notes with a pinned local text model."""
import fcntl
import json
import os
from scripts.benchmark_visual_inference_qwen import MODEL,DIGEST,local_teacher
from scripts.evaluate_grounded_propositions import release_notes, assess, Face, FONTS
from scripts.grounded_proposition_corpus import WORK, PUBLIC, ROOT, record, sha

SYSTEM="请仅根据给出的材料，用一句完整、简明的中文回答问题。"
if __name__=="__main__":
    rows,meta=release_notes()
    path=WORK/"qwen_notes.jsonl"
    answers=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with local_teacher(MODEL,WORK) as (ask,digest,size),path.open("a") as f:
            assert digest==DIGEST
            for i,m in enumerate(meta):
                if i<len(answers):
                    assert answers[i]["index"]==i and answers[i]["request"]==m["q"] and answers[i]["page"]==m["p"]
                    continue
                response=ask(SYSTEM,dict(role="user",content=f"材料：{m['p']}\n问题：{m['q']}"),think=False,
                             options=dict(seed=34712,temperature=0,num_predict=80,num_ctx=1024,num_thread=2))
                row=dict(index=i,request=m["q"],page=m["p"],**response)
                f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
                answers.append(row)
    face=Face((FONTS[0],2))
    stats,_=assess(rows,meta,[face(a["text"]) for a in answers],[a["stop"]=="stop" for a in answers],face)
    record(PUBLIC/"qwen.json",dict(model=MODEL,manifest=digest,installed_bytes=size,ledger=sha(path),scores=stats,
                                  answers=[a["text"] for a in answers],seconds=sum(a["duration_ns"] for a in answers)/1e9))
