"""Balance short natural contexts using pictured sentences from the training source."""
import fcntl
import json
import os
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file

from scripts.train_pictured_answer_selection import features, fit, metrics, record, store_streams, load_streams, ROOT, WORK, PUBLIC, READER, FontInvariantReader
from scripts.pointing_glyph_corpus import Row, digest, load
from scripts.train_pointing_causal_writer import TRAIN, HELD, sha
from scripts.prepare_paired_passage_questions import contains
from scripts.fit_font_invariant_reader import Face, FONTS
from scripts.train_causal_glyph_writer import in_cells


def excerpts(rows):
    face = Face((FONTS[0],2))
    marks = {face(c) for c in "。！？!?\n"}
    rng = np.random.default_rng(2201)
    positive, random_windows = [], []
    for r in rows:
        if r.task != "cmrc":continue
        cells, target = in_cells(r.p), in_cells(r.y)
        j = next(i for i in range(len(cells)) if cells[i:i+len(target)]==target)
        left = next((i+1 for i in range(j-1,-1,-1) if cells[i] in marks),0)
        end = j+len(target)
        right = next((i+1 for i in range(end,len(cells)) if cells[i] in marks),len(cells))
        n = right-left
        if not 8<=n<=96 or n==len(cells):continue
        start=int(rng.integers(len(cells)-n+1))
        for index,collection in ((left,positive),(start,random_windows)):
            p=r.p[index*128:(index+n)*128]
            collection.append(Row(r.task,r.key,p,r.q,r.y,r.foil))
    return positive, random_windows


def main():
    t=monotonic()
    train,dev=load()
    assert digest(train)==TRAIN and digest(dev)==HELD
    reader=FontInvariantReader.restore(READER,"cuda")
    x,_=features(reader,train)
    dx,quotes=features(reader,dev)
    written,stopped=load_streams(WORK/"pointing_answers.npz")
    result={}
    positive, random_windows=excerpts(train)
    assert len(positive)==len(random_windows)>400
    labels=torch.tensor([contains(r.p,r.y) for r in train]).float()
    for arm,extra in (("sentences",positive),("random_windows",random_windows)):
        ex,_=features(reader,extra)
        ey=torch.tensor([contains(r.p,r.y) for r in extra]).float()
        net=fit(torch.cat((x,ex)),torch.cat((labels,ey)),False)
        model=WORK/(arm+"_selection.safetensors")
        save_file(net.state_dict(),str(model))
        with torch.no_grad():choice=net(dx).flatten().ge(0).tolist()
        streams=[q if u else s for q,u,s in zip(quotes,choice,written,strict=True)]
        ends=[u or s for u,s in zip(choice,stopped,strict=True)]
        out=metrics(dev,streams,ends)
        for k,v in out.items():v["quoted"]=sum(u for r,u in zip(dev,choice) if r.task==k)
        file=WORK/("development_"+arm+"_selected.npz")
        store_streams(file,streams,ends)
        result[arm]=dict(scores=out,additional=len(extra),additional_positive=int(ey.sum()),cropped_sha256=digest(extra),
                         weights=sha(model),stream_sha256=sha(file))
        print(arm,{k:(v["exact"],v["quoted"]) for k,v in out.items()},flush=True)
    from scripts.prepare_composed_chinese_answers import Rasterizer
    from ilm.visual_lm.pictured_answer_selection import AnswerSelection
    raster=Rasterizer()
    def render(s):return b"".join(raster.tile(c) for c in s)
    corpus=json.loads((WORK/"windows.json").read_text())["fresh"]
    fresh=[Row("fresh",a["key"],render(a["passages"][0]),render(a["question"]),render(a["target"]),b"") for a in corpus]
    fx,fq=features(reader,fresh)
    w,ends=load_streams(WORK/"fresh_write_selected.npz")
    for arm in result:
        m=AnswerSelection.restore(WORK/(arm+"_selection.safetensors"))
        with torch.no_grad():flags=m(fx).flatten().ge(0).tolist()
        streams=[a if u else b for a,u,b in zip(fq,flags,w)]
        stops=[u or s for u,s in zip(flags,ends)]
        mixed=sum(s and a in [render(v) for v in r["refs"]] for s,a,r in zip(stops,streams,corpus))
        result[arm]["fresh"]=dict(multi_exact=mixed,quoted=sum(flags),n=len(fresh))
    result.update(seconds=monotonic()-t,peak_bytes=torch.cuda.max_memory_allocated())
    record(PUBLIC/"short_selection.json",result)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        main()
