"""Select original response and reference pixels from the two experiment archives."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_traditional_passage_training import archived
from scripts.infer_pictured_entry import page
from scripts.prepare_long_scope_questions import HELD, requested_words, long_question
from scripts.train_english_request_writer import ENTRIES, PUBLIC, WORK, load, write_json
from scripts.train_paired_passage import unpack_sequences
from scripts import train_scoped_request_writer as scope


def figure(scoped):
    root,public=(scope.WORK,scope.PUBLIC) if scoped else (WORK,PUBLIC)
    rec=load(root/"result.json")
    folder=ENTRIES/"development"
    meta=load(folder/"index.json")
    qrows=[json.loads(l) for l in (PUBLIC/"english.jsonl").read_text().splitlines()]
    pix=np.load(folder/"pixels.npy")
    y=[pix[s].tobytes() for s in unpack_sequences(folder/"answers.npz")]
    dimensions={}
    examples={}
    def put(name,stream):
        im=page(stream,width=max(1,len(stream)//128))
        im.save(public/(name+".png"))
        dimensions[name]=im.size
    def streams(arm,cond):
        k=arm+"/"+cond+".npz"
        return archived(root/k,rec["archives"][k])
    if scoped:
        table=[["Writer","Familiar cues¹","New cues¹","English / 96","Chinese / 96","DRCD pairs"]]
        for arm,label in (("parent","Local"),("unordered","Global, no order"),("ordered","Global, ordered")):
            a=rec["arms"][arm]
            table.append([label,*(str(a[k]["four"]) for k in ("long","new_cues")),
                          *(str(a[k]["exact"]) for k in ("english_free","chinese_free")),str(a["drcd"]["both"])])
        names=["parent","unordered","ordered"]
        series=[[rec["arms"][a][key]["four"] for a in names] for key in ("long","new_cues")]
        # Fixed before reading this run's output: first family, first and fourth intents.
        for i in (0,3):
            arm_images,ends=streams("ordered","long")
            local,local_stops=streams("parent","long")
            examples[str(i)]=dict(row=i,question=long_question(requested_words(meta,i),HELD,False,False),
                field=meta[i]["field"],value=meta[i]["value"],stage=meta[i]["stage"],label=meta[i]["glyph"],
                ordered_exact=bool(ends[i] and arm_images[i]==y[i]),local_exact=bool(local_stops[i] and local[i]==y[i]))
            for name,im in (("local",local[i]),("ordered",arm_images[i]),("reference",y[i])):
                put(name+str(i),im)
    else:
        table=[["Writer","Standard / 1,136","Crossed / 1,136","English / 96","Chinese / 96","DuReader / 571"]]
        for arm,label in (("parent","Parent"),("scrambled","Shuffled"),("matched","Matched")):
            a=rec["arms"][arm]
            table.append([label,*(str(a[k]["exact"]) for k in ("english_located","crossed_located","english_free_located","chinese_free_located","dureader"))])
        names=["parent","scrambled","matched"]
        series=[[rec["arms"][a]["english_free_located"]["exact"] for a in names]]
        imgs,ends=streams("matched","english_free_located")
        baseline,oldend=streams("parent","english_free_located")
        excluded=set(map(int,load(PUBLIC/"intent_review.json")["excluded_from_secondary_only"]))
        for category in ("gained","missed"):
            i=next(j for j,(x,e,q) in enumerate(zip(imgs,ends,qrows)) if j not in excluded and
                   (bool(e and x==y[q["row"]]) if category=="gained" else not(e and x==y[q["row"]]))
                   and (category!="gained" or not(oldend[j] and baseline[j]==y[q["row"]])))
            r=qrows[i]["row"]
            examples[category]=dict(index=i,row=r,question=qrows[i]["text"],value=meta[r]["value"],stage=meta[r]["stage"])
            put(category+"_output",imgs[i]);put(category+"_reference",y[r])
    write_json(public/"figure.json",dict(table=table,series=series,dimensions=dimensions,examples=examples))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scoped",action="store_true")
    figure(p.parse_args().scoped)
