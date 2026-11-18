"""Compare the independent visual writer to local Qwen on the same 35 official premise triples."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.author_entry_requests import MODEL
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_visual_inference_pairs import RAW, DEST, RESPONSES
from scripts.train_visual_inference_writer import ROOT, PUBLIC, WORK, PRIVATE, Corpus, DECIDE, RESTORE, Rasterizer, load, sha, write_json
from scripts.evaluate_visual_inference_writer import paired_interval

REG=PUBLIC/"qwen_binding.json"
LOCATION=WORK/"qwen_reference"
DECISION_SYSTEM="只写以下三个完整句子中的一个："+ "".join(RESPONSES)
RESTORATION_SYSTEM="说法中有一个□，代替一到三个原文字符。根据材料和上下文只写缺失的字，不加引号、标点或解释。"
DIGEST="e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130"


def registration():
    with np.load(PRIVATE) as split:
        rows=split["official_triples"].ravel().tolist()
    return dict(program_sha256=sha(Path(__file__)),cohort_sha256=sha(PRIVATE),
        checkpoint_sha256=sha(WORK/"paired/final.safetensors"),model=MODEL,digest=DIGEST,
        official_rows=rows,decision_system=DECISION_SYSTEM,restoration_system=RESTORATION_SYSTEM,
        requests=len(rows)*2,notes="All three labels from all 35 complete official premise groups. UTF-8 text teacher; free decoding; no teacher data supplied to the visual student.")


def cases():
    original=[json.loads(line) for line in (RAW/"dev.json").read_text().splitlines()]
    meta=load(DEST/"development/rows.json")
    result=load(PUBLIC/"result.json")
    def read(path):
        return archived(WORK/path,result["archives"][path])
    requests,_=read("official_cloze_requests.npz")
    targets,_=read("official_cloze_targets.npz")
    mask_rows=np.load(WORK/"official_cloze_rows.npy")
    positions={v:k for k,v in enumerate(mask_rows.tolist())}
    dr,de=read("paired/official_decision_intact.npz")
    mr,me=read("paired/official_restoration_intact.npz")
    r=Rasterizer()
    bits=lambda t:b"".join(r.tile(c) for c in t)
    records=[]
    for i in load(REG)["official_rows"]:
        row=original[meta[i]["line"]]
        p,h=row["sentence1"],row["sentence2"]
        for task in ("decision","restoration"):
            if task=="decision":
                wanted=RESPONSES[("entailment","contradiction","neutral").index(row["label"])]
                system=DECISION_SYSTEM;inp=h;instruction=DECIDE
                student=bool(de[i] and dr[i]==bits(wanted))
            else:
                j=positions[i]
                q=requests[j]
                cells=[q[k:k+128] for k in range(0,len(q),128)]
                start=cells.index(Corpus.mask_tile())-len(RESTORE)
                length=len(targets[j])//128
                wanted=h[start:start+length]
                inp=h[:start]+"□"+h[start+length:]
                assert bits(wanted)==targets[j] and bits(RESTORE+inp[:start])+Corpus.mask_tile()+bits(inp[start+1:])==q
                system=RESTORATION_SYSTEM;instruction=RESTORE
                student=bool(me[j] and mr[j]==targets[j])
            records.append((int(i),task,system,f"{instruction}\n材料：{p}\n说法：{inp}",wanted,student))
    return records


def run():
    if registration()!=load(REG):raise ValueError("Comparison inputs changed")
    LOCATION.mkdir(exist_ok=True)
    ledger=LOCATION/"answers.jsonl"
    dataset=cases()
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        if len(previous)>len(dataset):raise ValueError("Too many stored teacher calls")
        with local_teacher(MODEL,LOCATION) as (ask,digest,size),ledger.open("a") as f:
            assert digest==DIGEST
            start=monotonic()
            for idx,(i,task,system,prompt,wanted,student) in enumerate(dataset):
                if idx<len(previous):
                    assert [i,task]==[previous[idx]["row"],previous[idx]["task"]]
                    continue
                response=ask(system,dict(role="user",content=prompt),think=False,
                        options=dict(temperature=0,seed=32719,num_ctx=1024,num_predict=48,num_thread=2))
                record=dict(row=i,task=task,**response)
                f.write(json.dumps(record,ensure_ascii=False)+"\n");f.flush();os.fsync(f.fileno())
                if (idx+1)%45==0:print(idx+1,flush=True)
                if monotonic()-start>900:raise TimeoutError("Teacher comparison time ceiling")
    stored=[json.loads(line) for line in ledger.read_text().splitlines()]
    score={}
    for task in ("decision","restoration"):
        teacher=[];student=[];normalized=[];seconds=tokens=stopped=0
        for (i,t,_,_,wanted,correct),response in zip(dataset,stored,strict=True):
            if task!=t:continue
            assert i==response["row"] and task==response["task"]
            stopped+=response["stop"]=="stop"
            teacher.append(response["stop"]=="stop" and response["text"]==wanted)
            normalized.append(response["stop"]=="stop" and response["text"].rstrip("。.")==wanted.rstrip("。."))
            student.append(correct)
            seconds+=response["duration_ns"]/1e9;tokens+=response["tokens"]
        score[task]=dict(cases=len(teacher),qwen=sum(teacher),qwen_punctuation_tolerant=sum(normalized),
                visual=sum(student),qwen_three=int(np.array(teacher).reshape(-1,3).all(1).sum()),
                visual_three=int(np.array(student).reshape(-1,3).all(1).sum()),qwen_seconds=seconds,
                qwen_output_tokens=tokens,qwen_stopped=stopped,
                visual_minus_qwen=paired_interval(student,teacher,np.arange(len(teacher))//3))
    write_json(PUBLIC/"qwen_comparison.json",dict(scores=score,model=MODEL,digest=digest,installed_bytes=size,
        ledger_sha256=sha(ledger),binding_sha256=sha(REG)))
    print(score)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--register",action="store_true")
    if p.parse_args().register:
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,registration())
    else:run()
