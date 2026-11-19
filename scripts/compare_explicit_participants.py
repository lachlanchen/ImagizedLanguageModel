"""Compare complete visual outputs and a local text teacher on explicit role exchanges."""
import fcntl
import hashlib
import json
import os
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, load_model

from ilm.visual_lm.interval_writer import IntervalWriter
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.benchmark_visual_inference_qwen import DIGEST, MODEL, local_teacher
from scripts.evaluate_same_passage_writer import generate, image_archive
from scripts.explicit_participant_cases import cases
from scripts.measure_natural_role_questions import BASE
from scripts.train_interval_writer import (DEVICE, PUBLIC, ROOT, START, WORK, load, load_visual_writer,
    memory_bank, physical_tables, sha, to_batch, write_json)
from scripts.train_visual_role_writer import Tiles

SYSTEM="请阅读所给材料，直接写出问题所问之人的姓名，不加标点或解释。"


def counts(flags):
    cube=np.array(flags).reshape(-1,2,2)
    return dict(exact=int(cube.sum()),both_questions=int(cube.all(2).sum()),
                both_orders=int(cube.all(1).sum()),all_four=int(cube.all((1,2)).sum()))


@torch.inference_mode()
def visual(dest,records):
    bank=memory_bank();tile=Tiles(bank)
    ps,qs,ys=([tile(s) for s in field] for field in zip(*records,strict=True))
    px=lambda arr:b"".join(bank.tiles[int(x)] for x in arr)
    refs=[px(y) for y in ys]
    feat,mem,_=physical_tables(bank,DEVICE)
    writer=load_visual_writer(START,DEVICE);ink=writer.ink(feat);keys=writer.write_key(ink[mem])
    result={}
    for arm in ("matched","randomized","parent","passage_reader"):
        if arm=="parent":
            imgs,ends=generate(writer,ink,mem,keys,ps,qs,bank)
        elif arm=="passage_reader":
            reader=OrderedPassageComposer().eval()
            path=BASE/"mixed.safetensors"
            assert sha(path)==load(BASE/"mixed_result.json")["weights_sha256"]
            load_model(reader,str(path));grid=lambda s:np.frombuffer(px(s),np.uint8).reshape(-1,128)
            imgs,ends=zip(*(reader.answer(grid(p),grid(q)) for p,q in zip(ps,qs,strict=True)),strict=True)
        else:
            locator=IntervalWriter().to(DEVICE).eval()
            path=WORK/arm/"interval.safetensors"
            assert sha(path)==load(WORK/arm/"fit.json")["weights"]
            locator.load_state_dict(load_file(str(path)))
            imgs=[];ends=[]
            for i in range(0,len(ps),24):
                p,q=to_batch(ps[i:i+24],qs[i:i+24])
                out,stop=locator.generate(writer,ink,p,q,keys,mem)
                imgs.extend(px(row[row!=0]) for row in out.cpu().numpy());ends.extend(stop.tolist())
        ok=[s and x==r for s,x,r in zip(ends,imgs,refs,strict=True)]
        archive=dest/(arm+".npz");image_archive(archive,imgs,ends)
        result[arm]=dict(**counts(ok),archive=sha(archive),stopped=int(sum(ends)))
    write_json(dest/"visual.json",result)


def run():
    torch.set_num_threads(2)
    dest=WORK/"explicit_participants";dest.mkdir(exist_ok=True)
    records=list(cases())
    binding=load(PUBLIC/"explicit_binding.json")
    assert binding["cases"]==sha(ROOT/"scripts/explicit_participant_cases.py") and binding["system"]==SYSTEM
    assert binding["program"]==sha(__file__)
    assert binding["records"]==hashlib.sha256(json.dumps(records,ensure_ascii=False).encode()).hexdigest()
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (dest/"visual.json").exists():visual(dest,records)
        path=dest/"qwen.jsonl"
        stored=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        with local_teacher(MODEL,dest) as (ask,digest,size),path.open("a") as out:
            assert digest==DIGEST
            start=monotonic()
            for i,(p,q,y) in enumerate(records):
                if i<len(stored):
                    assert stored[i]["index"]==i
                    continue
                answer=ask(SYSTEM,dict(role="user",content=f"材料：{p}\n问题：{q}"),think=False,
                           options=dict(temperature=0,seed=91991,num_ctx=1024,num_predict=32,num_thread=2))
                row=dict(index=i,**answer)
                out.write(json.dumps(row,ensure_ascii=False)+"\n");out.flush();os.fsync(out.fileno())
                stored.append(row)
                if monotonic()-start>600:raise TimeoutError("Teacher exceeded ten minutes")
    result=load(dest/"visual.json")
    result["qwen_text"]=dict(**counts([s["text"]==y and s["stop"]=="stop" for s,(_,_,y) in zip(stored,records,strict=True)]),
                             records=sha(path),model=MODEL,digest=digest,bytes=size,
                             seconds=sum(s["duration_ns"] for s in stored)*1e-9)
    write_json(PUBLIC/"explicit_result.json",result)


if __name__=="__main__":
    run()
