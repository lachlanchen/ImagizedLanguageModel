"""Score complete physically generated answers on the registered source groups."""
from collections import Counter, defaultdict
import fcntl
import json
import os

import numpy as np
import torch
from torch.nn import functional as F

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from scripts.fit_font_invariant_reader import WORK as READER
from scripts.train_causal_glyph_writer import in_cells
from scripts.train_pointing_causal_writer import (
    HELD, PUBLIC, PRIOR, WORK, ROOT, CausalGlyphWriter, PointingCausalWriter,
    digest, inputs, load, record, sha,
)


def store_streams(path,streams,stops):
    packed=np.frombuffer(b"".join(streams),np.uint8).reshape(-1,128)
    np.savez_compressed(path,pixels=packed,offsets=np.cumsum([0,*[len(s)//128 for s in streams]]),
                        stopped=np.array(stops,bool))


def load_streams(path):
    with np.load(path) as f:
        data,off,ends=f["pixels"],f["offsets"],f["stopped"]
    return [data[i:j].tobytes() for i,j in zip(off[:-1],off[1:])],ends.tolist()


def metrics(rows,streams,stops):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r.task].append(i)
    scores={}
    for name,idx in groups.items():
        ok=[];f1=[];new_ink=0
        for i in idx:
            r=rows[i]
            y,s=map(in_cells,(r.y,streams[i]))
            overlap=sum((Counter(y)&Counter(s)).values())
            f1.append(2*overlap/(len(y)+len(s)))
            ok.append(bool(stops[i] and r.y==streams[i]))
            new_ink+=bool(set(s)-set(in_cells(r.p)+in_cells(r.q)))
        info=dict(n=len(idx),exact=sum(ok),stopped=sum(stops[i] for i in idx),
                  answer_with_new_ink=new_ink,mean_overlap=float(np.mean(f1)))
        if name in ("cmrc","drcd","role","historical","historical_rephrased","yan"):
            for a,b in zip(idx[::2],idx[1::2],strict=True):
                assert rows[a].p==rows[b].p and rows[a].y!=rows[b].y
            info["both"]=sum(a and b for a,b in zip(ok[::2],ok[1::2]))
            info["pairs"]=len(ok)//2
            info["same_response"]=sum(streams[i]==streams[j] and stops[i]==stops[j] for i,j in zip(idx[::2],idx[1::2]))
        if name=="role":
            info["complete_exchange_scenes"]=int(np.asarray(ok).reshape(-1,4).all(1).sum())
        scores[name]=info
    return scores


@torch.inference_mode()
def generate(net,rows,uniform=False,parent=False):
    read,out,pairs,_=inputs(net.writer,rows)
    fixed=[bytes(t) for t in net.writer.packed.cpu().numpy()]
    known=set(fixed)
    extra=sorted({c for r in rows for s in (r.p,r.q,r.y) for c in in_cells(s) if c not in known})
    bitmaps=[b"",*fixed,*extra]
    assert len(bitmaps)==len(read)-1
    streams=[None]*len(rows);stops=[False]*len(rows)
    ordered=sorted(range(len(rows)),key=lambda i:len(rows[i].p)+2*len(rows[i].q))
    for begin in range(0,len(ordered),20):
        ids=ordered[begin:begin+20]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            if parent:
                prompts=[read[[len(read)-1,*pairs[i][0],0,*pairs[i][1],0]] for i in ids]
                returned,ended=net.writer.generate(prompts,out,limit=48)
            else:
                returned,ended=net.generate(read,out,[pairs[i] for i in ids],uniform=uniform)
        for i,y,stop in zip(ids,returned,ended,strict=True):
            assert all(j<len(out) or j in pairs[i][0] for j in y)
            streams[i]=b"".join(bitmaps[j] for j in y)
            stops[i]=stop
    return streams,stops


@torch.inference_mode()
def reader_control(rows):
    reader=FontInvariantReader.restore(READER,"cuda")
    results=[]
    for start in range(0,len(rows),80):
        chunk=rows[start:start+80]
        outputs=contextualize(reader,[(r.p,r.q) for r in chunk])
        for r,(_,v) in zip(chunk,outputs,strict=True):
            prior=v[:,768:770].float()
            z=prior[:,0,None]+F.pad(prior[:,1],(0,47),value=-torch.inf).unfold(0,48,1)
            j,k=divmod(int(z.argmax()),48)
            results.append(r.p[j*128:(j+k+1)*128])
    return results,[True]*len(rows)


def strips(rows,streams):
    for r,s in zip(rows,streams,strict=True):
        if r.task=="yan" or (r.task=="role" and int(r.key)<4):
            stem=f"{r.task}_{r.key.replace('/','_')}"
            for key,data in (("passage",r.p),("question",r.q),("target",r.y),("answer",s)):
                write_strip(np.frombuffer(data,np.uint8).reshape(-1,128),WORK/(stem+"_"+key+".png"))


def main():
    _,rows=load()
    assert digest(rows)==HELD
    results={}
    for name in ("pointing","uniform","foil","parent","extractive"):
        cohort=rows if name not in ("parent","extractive") else [r for r in rows if r.task in ("cmrc","drcd","historical","historical_rephrased","yan")]
        if name=="extractive":
            cohort=[r for r in cohort if r.task in ("cmrc","drcd")]
            streams,stops=reader_control(cohort)
        else:
            if name=="parent":
                net=PointingCausalWriter(CausalGlyphWriter.restore(PRIOR/"matched","cuda")).cuda().eval()
            else:
                fit=json.loads((PUBLIC/(name+"_fit.json")).read_text())["weights"]
                assert all(sha(WORK/name/k)==v for k,v in fit.items())
                net=PointingCausalWriter.restore(WORK/name,"cuda")
            streams,stops=generate(net,cohort,uniform=name=="uniform",parent=name=="parent")
            del net
        file=WORK/(name+"_answers.npz")
        if file.exists():raise FileExistsError(file)
        store_streams(file,streams,stops)
        restored,ended=load_streams(file)
        assert restored==streams and ended==stops
        results[name]={"scores":metrics(cohort,streams,stops),"stream_sha256":sha(file),"cohort_digest":digest(cohort)}
        if name=="pointing":strips(cohort,streams)
        print(name,{k:(v["exact"],v.get("both")) for k,v in results[name]["scores"].items()},flush=True)
        record(WORK/"partial_scores.json",results)
    record(PUBLIC/"scores.json",results)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
