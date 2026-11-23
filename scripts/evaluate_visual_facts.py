"""Evaluate actual bitmap statements and the individual fields used to generate them."""
from collections import Counter,defaultdict
import fcntl
import hashlib
import json
import os
import torch
import numpy as np
from safetensors.torch import load_file

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.bound_fact_reader import FactHead,features_batch,best_bounds,render_fact
from scripts.train_visual_facts import rows,visual_cache,FontInvariantReader,READER
from scripts.evaluate_grounded_propositions import forms,store_streams
from scripts.prepare_visual_facts import Face,FONTS,WORK,PUBLIC,ROOT,RELATIONS,statement,simplify,write_json


@torch.inference_mode()
def evaluate():
    held=rows("held")
    face=Face((FONTS[0],2))
    reader=FontInvariantReader.restore(READER,"cuda")
    values=visual_cache("held",held,face,reader)
    stats={}
    for arm in ("genuine","shuffled"):
        folder=WORK/arm
        fit=json.loads((PUBLIC/(arm+"_fit.json")).read_text())
        if hashlib.sha256((folder/"head.safetensors").read_bytes()).hexdigest()!=fit["weights"]:raise ValueError(arm)
        frames=[[bytes.fromhex(k) for k in f] for f in json.loads((folder/"frames.json").read_text())]
        head=FactHead(len(frames)).cuda().eval()
        head.load_state_dict(load_file(str(folder/"head.safetensors")))
        totals=defaultdict(Counter);predictions=[];images=[]
        for start in range(0,len(held),32):
            chunk=held[start:start+32]
            batch=features_batch(values[start:start+len(chunk)],"cuda")
            role,ends=head(*batch)
            sub=best_bounds(ends[:,:,:2]).tolist()
            obj=best_bounds(ends[:,:,2:]).tolist()
            prior=best_bounds(batch[1][:,:,768:770].masked_fill(~batch[2][:,:,None],-1e4)).tolist()
            for x,k,(a,b),(c,d),(u,v) in zip(chunk,role.argmax(1).tolist(),sub,obj,prior,strict=True):
                p,s,r,o=x["p"],x["s"],x["r"],x["o"]
                ps,pr,po=p[a:b],RELATIONS[k],p[c:d]
                raw=face(p)
                y=render_fact(frames[k],raw[128*a:128*b],raw[128*c:128*d])
                if y!=face(statement(ps,pr,po)):raise ValueError("Emitted strip and field diagnostics differ")
                target=face(statement(s,r,o))
                seq=statement(ps,pr,po)
                eq=lambda l,t:simplify(l)==simplify(t)
                accepted={simplify(f) for f in forms((s,r,o))}
                scores=dict(subject=eq(ps,s),relation=pr==r,object=eq(po,o),
                    prior_object=eq(p[u:v],o),exact_pixels=y==target,
                    statement=simplify(seq).rstrip("。") in accepted,
                    oracle_object=simplify(statement(ps,pr,o)).rstrip("。") in accepted,
                    prior_statement=simplify(statement(ps,pr,p[u:v])).rstrip("。") in accepted,
                    representable_subject=simplify(s) in simplify(p),all_three=eq(ps,s) and pr==r and eq(po,o),
                    object_exact_pixels=face(po)==face(o))
                counter=totals[x["group"]];counter["n"]+=1;counter.update(scores)
                predictions.append(dict(key=x["key"],group=x["group"],subject=ps,relation=pr,object=po,output=seq,metrics=scores))
                images.append(y)
        for group,counter in totals.items():
            n=counter["n"]
            if n-counter["statement"]>sum(n-counter[k] for k in ("subject","relation","object")):
                raise ValueError("Corpus component error bound failed")
        notes=[p for p in predictions if p["group"]=="notes"]
        all_four=sum(all(n["metrics"]["statement"] for n in notes[i:i+4]) for i in range(0,32,4))
        stats[arm]=dict(groups=dict(totals),all_four_notes=all_four)
        store_streams(WORK/(arm+"_outputs.npz"),images,[True]*len(images))
        write_json(WORK/(arm+"_readings.json"),predictions)
        if arm=="genuine":
            for i,(x,y,pred) in enumerate(zip(held,images,predictions,strict=True)):
                if x["group"]=="notes" and int(x["key"])<4:
                    for kind,raw in (("answer",y),("question",face(x["q"])),("passage",face(x["p"]))):
                        write_strip(np.frombuffer(raw,np.uint8).reshape(-1,128),PUBLIC/f"film_{x['key']}_{kind}.png")
        print(arm,stats[arm],flush=True)
    write_json(PUBLIC/"results.json",stats)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use assigned GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(3);evaluate()
