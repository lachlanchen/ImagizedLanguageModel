"""Compare replay and request variation from the same frozen span-and-glyph writer."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from safetensors.torch import save_file,load_file

from ilm.visual_lm.font_invariant_reader import FontInvariantReader
from ilm.visual_lm.ordered_page_features import contextualize
from scripts.cache_ordered_page_features import PUBLIC, WORK, ROOT, READER, fingerprint, sha
from scripts.evaluate_ordered_page_writer import recount
from scripts.evaluate_same_passage_writer import image_archive
from scripts.ordered_image_corpus import PLATES, PAIRS, STAGES, Face, FONTS, PairedPassages, plate_examples
from scripts.train_ordered_page_writer import align,examples,interpret,restore,inputs,dump,owner

OUT = WORK/"varied_requests"
BATCH, EPOCHS, SEED = 48, 6, 40992


def different_requests(rows):
    rng = np.random.default_rng(SEED)
    font = Face((FONTS[0],2))
    cards = json.loads((PLATES/"train_cards.json").read_text())
    cmrc = PairedPassages(PAIRS,"train")
    natural_pairs = sorted({int(r.case)//2 for r in rows if r.task=="paired"})
    singles = set(rng.choice(natural_pairs,size=len(natural_pairs)//2,replace=False))
    alternative = []
    for r in rows:
        if r.task=="paired" and int(r.case)//2 in singles:
            _,i,_ = cmrc.pairs[int(r.case)]
            r = replace(r,question=cmrc.questions[i],target=cmrc.answers[i])
        elif r.task=="plate":
            card,_,name,order=r.case.split("/")
            stages=sorted({x["stage"] for x in cards[int(card)]},key=list(STAGES).index)
            a,b=[STAGES[s] for s in stages]
            names=([a],[b],[a,b],[b,a])[int(order)]
            first="、".join(names)
            prompts=(None, f"{name}：{first}。請逐項照錄。", f"把{first}所用的{name}抄下來。",
                     f"書裡怎麼寫「{name}」？我要看{'和'.join(names)}。", f"按以下次序給出{name}的形體：{first}。",
                     f"題字是{name}，請謄出{'再'.join(names)}。", f"從資料中取出{name}在{first}時的寫法。")
            q=prompts[rng.integers(len(prompts))]
            if q is not None:
                r=replace(r,question=font(q))
        alternative.append(r)
    return alternative


def fresh_requests():
    rng=np.random.default_rng(6750)
    cards=json.loads((PLATES/"development_cards.json").read_text())
    selected=rng.permutation(len(cards))[24:48]
    tiles=np.load(PLATES/"pixels.npy")
    font=Face((FONTS[0],2))
    rows=[]
    for n in selected:
        a,b=sorted({x["stage"] for x in cards[n]},key=list(STAGES).index)
        for variant in (0,3):
            for r in plate_examples(cards[n],tiles,font,int(n),variant,False,"fresh_wording"):
                _,_,c,j=r.case.split("/")
                stages=([a],[b],[a,b],[b,a])[int(j)]
                labels=[STAGES[s] for s in stages]
                q=f"翻到字頭「{c}」，先列{labels[0]}"
                q+=(f"，接下來列{labels[1]}" if len(labels)==2 else "")
                rows.append(replace(r,question=font(q+"，原樣即可。")))
    return rows


def compare():
    reg=json.loads((PUBLIC/"variation_registration.json").read_text())
    assert all(sha(ROOT/k)==v for k,v in reg["sources"].items())
    OUT.mkdir()
    train,old_features=examples("train")
    dev,dev_features=examples("development")
    varied=different_requests(train)
    fresh=fresh_requests()
    # Reuse original contextual features byte-for-byte where both streams are unchanged.
    changed=[i for i,(a,b) in enumerate(zip(train,varied,strict=True)) if a.question!=b.question]
    encoder=FontInvariantReader.restore(READER,"cuda:0")
    started=monotonic()
    new_features=list(old_features)
    for i,v in zip(changed,contextualize(encoder,[(varied[i].passage,varied[i].question) for i in changed]),strict=True):
        new_features[i]=v
    fresh_features=contextualize(encoder,[(r.passage,r.question) for r in fresh])
    preparation=monotonic()-started
    del encoder
    memory=list(map(bytes,np.load(WORK/"writing_memory.npy")))
    indices={v:i for i,v in enumerate(memory)}
    assert sha(WORK/"writer.safetensors")==json.loads((PUBLIC/"fit.json").read_text())["weights"]
    evaluation=dev+fresh
    joined_features=dev_features+fresh_features
    arms={}
    for arm,rows,features in (("parent",None,None),("replay",train,old_features),("varied",varied,new_features)):
        model,_=restore(WORK,"cuda:0")
        trace=[]
        steps=0
        torch.cuda.reset_peak_memory_stats()
        start=monotonic()
        if rows is not None:
            model.train()
            targets=[align(r.passage,r.target,indices) for r in rows]
            assert len(indices)==len(memory)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0004)
            rng=np.random.default_rng(SEED)
            # Use exactly the same batch membership and order in both fits.
            widths=np.array([len(r.passage) for r in train])
            for epoch in range(EPOCHS):
                batches=[]
                for chunk in np.array_split(rng.permutation(len(rows)),12):
                    bucket=chunk[np.argsort(widths[chunk],kind="stable")]
                    batches.extend(bucket[i:i+BATCH] for i in range(0,len(bucket),BATCH))
                total=0.
                for j in rng.permutation(len(batches)):
                    ids=batches[j]
                    gs,v,mask=inputs(features,ids,"cuda:0")
                    t=pad_sequence([targets[i] for i in ids],batch_first=True)
                    for n,i in enumerate(ids):
                        t[n,len(targets[i]):,0]=-1
                    optimizer.zero_grad(set_to_none=True)
                    loss=model.loss(gs,v,mask,t.cuda())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
                    optimizer.step()
                    total+=float(loss.detach())*len(ids)
                    steps+=1
                trace.append(total/len(rows))
                print(arm,epoch+1,trace[-1],flush=True)
                if monotonic()-start>600 or torch.cuda.max_memory_allocated()>12*2**30:
                    raise RuntimeError("Fit budget exceeded")
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(OUT/(arm+".safetensors")))
        duration=monotonic()-start
        peak=torch.cuda.max_memory_allocated()
        model.eval()
        images,stops=[],[]
        for first in range(0,len(evaluation),16):
            batch=evaluation[first:first+16]
            trajectories=model.generate(*inputs(joined_features,range(first,first+len(batch)),"cuda:0"))
            for r,a in zip(batch,trajectories,strict=True):
                image,stopped=interpret(r.passage,a,memory)
                images.append(image)
                stops.append(stopped)
        path=OUT/(arm+".npz")
        image_archive(path,images,stops)
        ok=np.asarray([s and im==r.target for r,im,s in zip(evaluation,images,stops,strict=True)])
        arms[arm]=dict(scores=recount(evaluation,ok),seconds=duration,peak_bytes=peak,updates=steps,
                       loss=trace,output_sha256=sha(path),stopped=int(sum(stops)))
        if rows is not None:
            arms[arm]["weights"]=sha(OUT/(arm+".safetensors"))
        print(arm,arms[arm]["scores"],flush=True)
    assert arms["parent"]["scores"]["yan"]==json.loads((PUBLIC/"result.json").read_text())["arms"]["ordered"]["scores"]["yan"]
    dump(PUBLIC/"variation.json",dict(arms=arms,records=len(train),changed_questions=len(changed),
         new_train_sha256=fingerprint(varied),fresh_sha256=fingerprint(fresh),feature_seconds=preparation,
         devices=torch.cuda.get_device_name(),register=sha(PUBLIC/"variation_registration.json")))


if __name__=="__main__":
    with owner():
        compare()
