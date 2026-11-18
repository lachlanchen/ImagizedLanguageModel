"""Evaluate an image-history spelling prior against the identical shuffled Chinese corpus."""
from collections import Counter
from pathlib import Path
from time import monotonic
import argparse
import fcntl
import os
import resource

import numpy as np
import torch
from safetensors.torch import load_file, load_model

from ilm.visual_lm.bitmap_history import BitmapHistory
from ilm.visual_lm.composed_answer_writer import WRITE, MAX_SPAN, ComposedAnswerWriter, descriptors
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.train_composed_answer_writer import (
    BATCH, DATA, OUT, PRIOR, ROOT, InputImages, Split, archive, batch_inputs, generate,
    segments, source_fields, teacher_batch, tensor,
)
from scripts.prepare_visual_answer_language import DEST
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.audit_composed_writer_outputs import distance

WORK=ROOT/"artifacts/bitmap_history_writing_v1"
REG=ROOT/"references/evidence/bitmap_history_writing_registration.json"


def bindings():
    paths=[Path(__file__), ROOT/"ilm/visual_lm/bitmap_history.py",
           ROOT/"scripts/prepare_visual_answer_language.py",ROOT/"scripts/train_composed_answer_writer.py",
           ROOT/"ilm/visual_lm/composed_answer_writer.py",ROOT/"scripts/audit_composed_writer_outputs.py",
           ROOT/"references/bitmap_history_writing_protocol.md",DEST/"receipt.json",OUT/"fit.json",
           PRIOR,OUT/"memory.npy",OUT/"final.safetensors",OUT/"finite.npz",DATA/"preparation.json"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


@torch.inference_mode()
def conditional(net,prior,ds,tiles,grammars,device):
    fields=source_fields(prior,ds,ds.q,device)
    lookup={bytes(t):k for k,t in enumerate(tiles)}
    mapping={i:lookup.get(bytes(t),-1) for i,t in enumerate(ds.tiles)}
    align=[segments(p,q,y,mapping) for p,q,y in zip(ds.p,ds.q,ds.y,strict=True)]
    ink=net.ink(descriptors(tensor(np.unpackbits(ds.tiles,axis=1),device)))
    literals=net.ink(descriptors(tensor(np.unpackbits(tiles,axis=1),device)))
    truth=[]; picks={name:[] for name in ("neural",*grammars)}
    for i in range(0,len(ds.y),BATCH):
        ix=list(range(i,min(i+BATCH,len(ds.y))))
        ids,f,valid=batch_inputs(ds,ds.q,fields,ix,device)
        act,_,_,cur,_,first,last,means,counts,_=[tensor(v,device) for v in teacher_batch(ds,align,ix,ids.shape[1])]
        context,h=net.encode(ink[ids],f,valid)
        mask=torch.arange(MAX_SPAN,device=device)<counts[...,None]
        avg=(ink[means]*mask.unsqueeze(-1)).sum(-2)/counts[...,None]
        fb=net.feedback_ink(ink[first],avg,ink[last]);fb[:,0]=0
        state,_=net.recurrence(fb,h)
        _,_,focused=net.plan(context,f,valid,state,cur)
        logits=net.bitmap_scores(focused[act==WRITE],literals)
        prefixes=[]
        for n in ix:
            for kind,_,_,_,_,offset,_ in align[n]:
                if kind==WRITE:
                    y=ds.y[n]
                    prefixes.append(ds.tiles[y[:offset]].tobytes())
                    truth.append(mapping[y[offset]])
        picks["neural"].extend(logits.argmax(-1).tolist())
        for name,model in grammars.items():
            picks[name].extend(model.choose(logits,prefixes).tolist())
    y=np.asarray(truth)
    correct={k:np.asarray(v)==y for k,v in picks.items()}
    table={k:int(v.sum()) for k,v in correct.items()}
    paired={k:dict(gains=int((correct["ordered"]&~correct[k]).sum()),
                   losses=int((~correct["ordered"]&correct[k]).sum())) for k in ("neural","shuffled")}
    assert len(y)==1525 and table["neural"]==162
    np.savez(WORK/"conditional.npz",truth=y,**picks)
    return dict(cells=len(y),exact=table,ordered_vs_control=paired)


def evaluate(ds,streams,stops,qs):
    groups={}
    for i,(stream,stop) in enumerate(zip(streams,stops,strict=True)):
        ref=list(map(bytes,ds.tiles[ds.y[i]]))
        visible=set(map(bytes,ds.tiles[ds.p[i]]))|set(map(bytes,ds.tiles[ds.q[i]]))
        out=[stream[j:j+128] for j in range(0,len(stream),128)]
        want,gave=Counter(g for g in ref if g not in visible),Counter(g for g in out if g not in visible)
        stats=dict(records=1,exact=int(stop and out==ref),stopped=int(stop),length=len(out),
                   edit_distance=distance(ref,out),reference_cells=len(ref),input_absent_overlap=sum((want&gave).values()),
                   input_absent_exact_position=sum(g==t and t not in visible for g,t in zip(out,ref)))
        for k in ("all",ds.groups[i]):groups.setdefault(k,Counter()).update(stats)
    return {k:dict(v) for k,v in groups.items()}


def run():
    if load(REG)!=bindings() or os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Registered sources or GPU selection changed")
    receipt=load(DEST/"receipt.json")
    for key,expected in receipt["files"].items():assert sha(DEST/key)==expected
    WORK.mkdir()
    torch.set_num_threads(2);torch.set_float32_matmul_precision("high")
    device="cuda:0"
    net=ComposedAnswerWriter().to(device).eval()
    net.load_state_dict(load_file(str(OUT/"final.safetensors")))
    prior=OrderedPassageComposer().to(device).eval()
    load_model(prior,str(PRIOR))
    ds=Split("development")
    ps=[ds.tiles[s] for s in ds.p];qs=[ds.tiles[s] for s in ds.q]
    tiles=np.load(OUT/"memory.npy")
    inputs=InputImages(ps,qs)
    models={arm:BitmapHistory(DEST,tiles,arm) for arm in ("ordered","shuffled")}
    began=monotonic()
    report={}
    for arm in ("neural","ordered","shuffled","rotated"):
        inp=InputImages(ps,qs[1:]+qs[:1]) if arm=="rotated" else inputs
        spelling=models.get("ordered" if arm=="rotated" else arm)
        streams,kinds,stops=generate(net,prior,tiles,inp,"finite",device,spelling)
        path=WORK/(arm+".npz")
        archive(path,streams,kinds,stops)
        if arm=="neural":
            with np.load(path) as a,np.load(OUT/"finite.npz") as b:
                for k in a.files:assert np.array_equal(a[k],b[k]),k
        report[arm]=evaluate(ds,streams,stops,qs)
        print(arm,report[arm]["all"],flush=True)
    check=conditional(net,prior,ds,tiles,models,device)
    result=dict(scores=report,conditional=check,corpus=receipt,seconds=monotonic()-began,
                archives={p.name:sha(p) for p in WORK.glob("*.npz")},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(WORK/"result.json",result)
    print(check,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("register","run"))
    if p.parse_args().action=="register":
        if REG.exists():raise FileExistsError(REG)
        write_json(REG,bindings())
    else:
        with (ROOT/"artifacts/composed_answer_gpu0.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run()
