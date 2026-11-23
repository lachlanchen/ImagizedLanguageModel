"""Test autonomous glyph emission from supplied and visually predicted factual fields."""
from collections import Counter,defaultdict
import fcntl
import json
import os
import numpy as np
import torch

from scripts.train_fact_realizer import START,COMMANDS,visual_fields,target
from scripts.train_visual_facts import rows
from scripts.pointing_glyph_corpus import Row
from scripts.evaluate_grounded_propositions import generate,store_streams
from scripts.train_grounded_propositions import PointingCausalWriter,sha
from scripts.prepare_visual_facts import ROOT,WORK,PUBLIC,Face,FONTS,LANG,RELATIONS,write_json
from ilm.visual_lm.causal_glyph_writer import write_strip


@torch.inference_mode()
def evaluate():
    held=rows("held")
    predicted=json.loads((WORK/"genuine_readings.json").read_text())
    assert len(held)==len(predicted) and all(x["key"]==y["key"] and x["group"]==y["group"] for x,y in zip(held,predicted))
    face=Face((FONTS[0],2))
    folder=WORK/"writer"
    expected=json.loads((PUBLIC/"writer_fit.json").read_text())["weights"]
    assert all(sha(folder/n)==v for n,v in expected.items())
    write_json(folder/"cues.json",dict(roles=[face(LANG[r][0]).hex() for r in RELATIONS],
        commands={k:face(v).hex() for k,v in zip(("subject","object","natural"),COMMANDS)}))
    results={};samples=[]
    for arm,weights in (("adapted",folder),("parent",START)):
        net=PointingCausalWriter.restore(weights,"cuda")
        from scripts.train_causal_glyph_writer import WORK as GPT
        legend=dict(zip((r.tobytes() for r in net.writer.packed.cpu().numpy()),json.loads((GPT/"audit_legend.json").read_text()),strict=True))
        scores={}
        for condition in ("predicted","oracle"):
            records=[];texts=[];matched=[];indices=[]
            for i,(x,z) in enumerate(zip(held,predicted)):
                s,r,o=(x["s"],x["r"],x["o"]) if condition=="oracle" else (z["subject"],z["relation"],z["object"])
                source=visual_fields(face(s),face(LANG[r][0]),face(o))
                for char in s+o:legend[face(char)]=char
                modes=(0,1,2) if i<144 and arm=="adapted" and condition=="predicted" else (0,1)
                for mode in modes:
                    text=target(x,mode) if mode<2 else ""
                    records.append(Row("write",str(i),source,face(COMMANDS[mode]),source[:128],b""))
                    texts.append(text);indices.append((i,mode))
            ys,stops=generate(net,records)
            name=arm+"_"+condition
            store_streams(WORK/(name+"_free.npz"),ys,stops)
            stats=defaultdict(Counter);both=defaultdict(list);natural=defaultdict(Counter)
            for ((i,k),y,stop,want) in zip(indices,ys,stops,texts,strict=True):
                x=held[i];g=x["group"]
                if k==2:
                    cells=[y[j:j+128] for j in range(0,len(y),128)]
                    reading="".join(legend.get(b,"□") for b in cells)
                    c=natural[g];c["n"]+=1;c["stopped"]+=stop;c["both_strings"]+=x["s"] in reading and x["o"] in reading
                    if i<8 or 48<=i<52:samples.append(dict(key=x["key"],group=g,output=reading,subject=x["s"],relation=x["r"],object=x["o"],stopped=stop))
                    continue
                correct=bool(stop and y==face(want));both[i].append(correct)
                c=stats[g];c["count_orders"]+=1;c[("subject_first","object_first")[k]]+=correct;c["stops"]+=stop
                if arm=="adapted" and condition=="predicted" and g=="notes" and int(x["key"]) in (0,1) and k==1:
                    write_strip(np.frombuffer(y,np.uint8).reshape(-1,128),PUBLIC/f"free_film_{x['key']}.png")
            for i,pair in both.items():
                assert len(pair)==2
                c=stats[held[i]["group"]];c["inputs"]+=1;c["both_orders"]+=all(pair)
            scores[condition]=dict(groups=dict(stats),natural=dict(natural),raw_sha256=sha(WORK/(name+"_free.npz")))
            print(arm,condition,dict(stats),flush=True)
        results[arm]=scores
    write_json(PUBLIC/"writer_results.json",results)
    write_json(PUBLIC/"natural_readings.json",samples)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":raise RuntimeError("Use GPU 0")
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(3);evaluate()
