"""Read real held dialogues, exchange their inputs, and recount independent image streams."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch

from ilm.visual_lm.causal_glyph_writer import write_strip
from ilm.visual_lm.portable_writer import read_image
from scripts.answer_dialogue_ink import generate,answer,load
from scripts.answer_optical_passage_qa import load as load_span
from scripts.dialogue_ink_data import Corpus,read_data
from scripts.evaluate_optical_passage_qa import normalized,metrics,save_page
from scripts.prepare_dialogue_ink import RELEASE,WORK,dump,sha
from scripts.train_span_choice import gpu_owner
from scripts.train_typeface_writer import Face,FACES

ARMS=("writer","exchanged_question","exchanged_passage","span")


def exchanges(rows):
    groups=defaultdict(list)
    for i,r in enumerate(rows): groups[r[1]].append(i)
    keys=list(groups)
    question=list(range(len(rows))); passage=list(range(len(rows)))
    for k,key in enumerate(keys):
        group=groups[key];other=groups[keys[(k+1)%len(keys)]]
        for j,i in enumerate(group):
            question[i]=next((group[(j+d)%len(group)] for d in range(1,len(group))
                              if rows[group[(j+d)%len(group)]][3]!=rows[i][3]),i)
            passage[i]=other[j%len(other)]
    return np.array(question),np.array(passage)


def summary(records):
    result={}
    for group in ("reserved","dev"):
        subset=[r for r in records if r["group"]==group]
        result[group]={}
        for name in ("all","first","binary","absent_multiword","strict","strict_exchangeable"):
            rows=[r for r in subset if name=="all" or r[name]]
            cols={}
            for arm in ARMS:
                ex=[r["out"][arm] for r in rows]
                cols[arm]=dict(exact=sum(e["exact"] for e in ex),pixels=sum(e["pixels"] for e in ex),
                               stopped=sum(e["stopped"] for e in ex),
                               mean_f1=sum(e["f1"] for e in ex)/len(ex) if ex else None)
            paired={}
            for ctrl in ARMS[1:]:
                paired[ctrl]=[sum(r["out"][a]["exact"] and not r["out"][b]["exact"] for r in rows)
                              for a,b in (("writer",ctrl),(ctrl,"writer"))]
            result[group][name]=dict(n=len(rows),arms=cols,paired=paired,
                    exclusively_correct=sum(r["out"]["writer"]["exact"] and all(not r["out"][arm]["exact"] for arm in ARMS[1:]) for r in rows))
    return result


@torch.inference_mode()
def run(folder):
    data=read_data()
    training={normalized(r[5]) for r in data["rows"] if r[0]=="train"}
    net,opt=load(folder,"cuda");span,_=load_span("cuda")
    face=Face(FACES[0]); alphabet={face.cell(chr(i)):chr(i) for i in range(32,127)}
    records=[]
    for group in ("reserved","dev"):
        ds=Corpus(data,group,net.reader,opt)
        widthbank=torch.tensor([len(w)//128 for w in ds.bank.raw],device=ds.device)
        questions,passages=exchanges(ds.rows)
        for start in range(0,len(ds.rows),48):
            ids=np.arange(start,min(start+48,len(ds.rows)))
            batch=[]
            for i in ids:
                _,key,turn,q,h,ref,_,domain=ds.rows[i]
                sources=[data["pages"][key],q,h]
                absent=len(ref.split())>=2 and all(ref.casefold() not in s.casefold() for s in sources)
                strict=absent and normalized(ref) not in training
                batch.append(dict(group=group,index=int(i),story=key,turn=turn,domain=domain,
                   question=q,history=h,answer=ref,first=not turn,binary=normalized(ref) in ("yes","no","unknown"),
                   absent_multiword=absent,strict=strict,strict_exchangeable=bool(strict and questions[i]!=i),
                   question_control=int(questions[i]),passage_control=int(passages[i]),out={}))
            for arm in ARMS:
                qidx=questions[ids] if arm=="exchanged_question" else ids
                pidx=passages[ids] if arm=="exchanged_passage" else ids
                inputs,variants=ds.inputs(ids,qidx,pidx)
                if arm=="span":
                    scores,trim=span(*inputs[:4])
                    widths=ds.page[ds.pindex[ids]].long()
                    # True physical cell counts, never decoded word identities.
                    selected=span.select(scores,trim,inputs[1],widthbank[widths]).tolist()
                    pixels=[]
                    for r,(l,u,dl,dr) in zip(batch,selected):
                        words=data["pages"][r["story"]].split()[l:u+1]
                        raw=face(" ".join(words));pixels.append(raw[128*dl:len(raw)-128*dr])
                    stopped=[True]*len(batch);paths=[["interval"]]*len(batch)
                else:
                    pixels,stopped,paths=generate(net,opt,ds.bank,inputs,variants)
                for r,raw,end,path in zip(batch,pixels,stopped,paths,strict=True):
                    pred="".join(alphabet[raw[j:j+128]] for j in range(0,len(raw),128))
                    exact,f1=metrics(pred,[r["answer"]])
                    r["out"][arm]=dict(text=pred,stopped=end,exact=bool(end and exact),f1=f1 if end else 0.,
                                       pixels=end and raw==face(r["answer"]),actions=path)
            records+=batch
    return records,data


def replay(folder,out,records,data):
    face=Face(FACES[0]);net,opt=load(folder,"cpu")
    dest=out/"examples";dest.mkdir()
    examples=[]
    for group in ("reserved","dev"):
        cases=[r for r in records if r["group"]==group]
        def earns(r):
            return r["out"]["writer"]["pixels"] and all(not r["out"][a]["exact"] for a in ARMS[1:])
        for label,predicate in [("strict-source",lambda r:r["strict_exchangeable"] and earns(r)),
                                ("strict-exact",lambda r:r["strict"] and r["out"]["writer"]["exact"]),
                                ("strict-miss",lambda r:r["strict"] and not r["out"]["writer"]["exact"]),
                                ("answer",lambda r:r["out"]["writer"]["pixels"] and not r["binary"])]:
            r=next((r for r in cases if predicate(r)),None)
            if not r:continue
            prefix=group+"-"+label
            paths=[]
            for n,s in (("passage",data["pages"][r["story"]]),("question",r["question"]),("history",r["history"])):
                f=dest/(prefix+"-"+n+".png")
                if s:save_page(s,f,face);paths.append(read_image(f,None))
                else:paths.append(np.empty((0,128),np.uint8))
            ink,decision=answer(net,opt,*paths)
            e=r["out"]["writer"]
            if ink.tobytes()!=face(e["text"]) or decision!=dict(stopped=e["stopped"],actions=e["actions"]):
                raise ValueError(("CPU PNG replay differs",group,r["index"],decision,e))
            write_strip(ink,dest/(prefix+"-answer.png"))
            examples.append(dict(prefix=prefix,index=r["index"],question=r["question"],reference=r["answer"],
                                 generated=e["text"],stopped=e["stopped"],actions=e["actions"]))
    return examples


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,default=RELEASE);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    with gpu_owner():
        torch.set_num_threads(2);records,data=run(args.model)
    f=args.output/"decisions.json.gz"
    f.write_bytes(gzip.compress(json.dumps(records,separators=(",",":")).encode(),mtime=0))
    result=summary(records)
    dump(args.output/"scores.json",result)
    samples=replay(args.model,args.output,records,data)
    dump(args.output/"replay.json",dict(decisions=sha(f),examples=samples))
    for g,s in result.items():
        print(g,{n:(r["n"],*[r["arms"][a]["exact"] for a in ARMS]) for n,r in s.items()})
