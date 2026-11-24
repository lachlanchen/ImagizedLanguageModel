"""Locate held composition errors given correct previous words and legal copy targets."""

from collections import Counter
import argparse
import gzip
import json
from pathlib import Path
import torch

from scripts.answer_dialogue_ink import load
from scripts.dialogue_ink_data import Corpus, WordImages, read_data, pack
from scripts.evaluate_optical_passage_qa import normalized
from scripts.prepare_dialogue_ink import RELEASE,dump
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face,FACES


@torch.inference_mode()
def measure(model, decisions):
    net,opt=load(model,"cuda")
    data=read_data()
    recs=json.loads(gzip.decompress(decisions.read_bytes()))
    face=Face(FACES[0])
    out={}
    for group in ("reserved","dev"):
        ds=Corpus(data,group,net.reader,opt)
        cases=[r for r in recs if r["group"]==group and r["strict"]]
        # Only this post-fit diagnostic can read the gold previous word.
        gold={face(w) for r in cases for w in r["answer"].split()}
        gb=WordImages(gold,net.reader,opt)
        totals=Counter();semantic=Counter()
        for start in range(0,len(cases),32):
            rows=cases[start:start+32]
            t=[[gb.ids[face(w)] for w in r["answer"].split()] for r in rows]
            prev,sizes=pack([[0]+v[:16] for v in t],"cuda")
            x,variants=ds.inputs([r["index"] for r in rows])
            mem,_=net.memory(*x)
            ch,scores,latent,_=net.step(mem,gb.features[prev.long(),:50],mem[-1][None])
            choice=scores.masked_fill(variants[:,None]==0,-1e4).flatten(2).argmax(2)
            picked=variants.flatten(1).gather(1,choice).tolist()
            routes=ch.argmax(-1).tolist()
            latent_indices=[];expected=[];spell_records=[]
            page_correct=[True]*len(rows)
            for i,r in enumerate(rows):
                fields=[data["pages"][r["story"]],r["question"],r["history"]]
                nref=normalized(r["answer"])
                deeper=len(nref.split())>=2 and all((" "+nref+" ") not in (" "+normalized(s)+" ") for s in fields)
                if deeper:
                    semantic["cases"]+=1
                    for arm,e in r["out"].items():
                        semantic[arm+"_exact"]+=e["exact"]
                        semantic[arm+"_pixels"]+=e["pixels"]
                originals={v for s in fields for v in s.split()}
                accessible={ds.bank.raw[int(k)] for k in variants[i].unique().tolist() if k}
                for j,word in enumerate(r["answer"].split()[:16]):
                    raw=face(word)
                    cp=raw in accessible
                    totals["words"]+=1;totals["copyable" if cp else "must_spell"]+=1
                    route=routes[i][j]
                    totals["channel_correct"]+=route==(1 if cp else 2)
                    if cp:
                        ok=ds.bank.raw[picked[i][j]]==raw
                        totals["copy_correct_given_channel"]+=ok
                        page_correct[i]&=ok and route==1
                    else:
                        totals["must_spell_with_normalized_input_word"]+=any(normalized(w)==normalized(word) for w in originals)
                        latent_indices.append((i,j));expected.append(raw);spell_records.append((i,route))
                l=len(t[i]);st=l<=16 and routes[i][l]==0
                totals["correct_stop_given_prefix"]+=st
                page_correct[i]&=st
            if expected:
                b,j=zip(*latent_indices)
                glyph,end=net.write_words(latent[list(b),list(j)])
                palette=net.palette.cpu().numpy()
                for gs,done,ref,(i,route) in zip(glyph.tolist(),end.tolist(),expected,spell_records):
                    img=palette[[v for v in gs if v]].tobytes()
                    ok=done and img==ref
                    totals["spelled_given_channel"]+=ok
                    page_correct[i]&=ok and route==2
            totals["fully_correct_canonical_routes_given_prefixes"]+=sum(page_correct)
        out[group]=dict(strict=len(cases),conditional=totals,normalized_absent_multiword=semantic)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("decisions",type=Path);p.add_argument("output",type=Path)
    args=p.parse_args()
    with gpu_owner():
        torch.set_num_threads(2);result=measure(RELEASE,args.decisions)
    dump(args.output,result);print(result)
