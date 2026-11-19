"""Count the actual selected action and complete returned images from the calibrated writer."""
import numpy as np
import torch
from safetensors.torch import load_file

from ilm.visual_lm.interval_writer import IntervalWriter
from scripts.audit_traditional_passage_training import archived
from scripts.calibrate_interval_writer import (DEST, DEVICE, PUBLIC, START, WHO, WORK, SentenceCorpus,
    load,load_visual_writer,memory_bank,physical_tables,sha,to_batch,write_json)
from scripts.evaluate_inverse_role_writer import available_sets
from scripts.evaluate_same_passage_writer import image_archive
from scripts.explicit_participant_cases import cases
from scripts.train_visual_role_writer import Tiles


@torch.inference_mode()
def evaluate():
    bank=memory_bank();tests,ds,_=available_sets(bank)
    corpus=SentenceCorpus(ds,bank,"development")
    pack=lambda s:b"".join(bank.tiles[x] for x in s)
    caseset=list(cases());tile=Tiles(bank)
    tests["explicit"]=([tile(p) for p,_,_ in caseset],[tile(q) for _,q,_ in caseset],[{pack(tile(y))} for _,_,y in caseset])
    ps=[];qs=[];refs=[]
    for a,b,l,r in corpus.pairs:
        for i in (a,b):
            p=ds.passages[ds.groups[i]]
            ps.append(p[l:r]);qs.append(ds.questions[i])
            refs.append({pack(p[u:v]) for u,v in ds.spans[i] if l<=u and v<=r})
    tests["same_sentence"]=(ps,qs,refs)
    people=set(corpus.people);mask=np.array([row in people for row in corpus.pairs])
    with np.load(WHO/"cohort.npz") as z:who=z["development"];pairs=z["development_pairs"]
    marks=set(tile("。！？!?"));sub=[]
    for i in who:
        g=ds.groups[i];p=ds.passages[g];q=set(ds.questions[i])-marks
        bits=[set(p[a:b])-marks for a,b in corpus.bounds[g]]
        j=max(range(len(bits)),key=lambda k:len(q&bits[k])/max(1,len(q|bits[k])))
        l,r=corpus.bounds[g][j];sub.append(p[l:r])
    tests["overlap_who"]=(sub,[ds.questions[i] for i in who],[tests["drcd"][2][i] for i in who])
    feat,mem,_=physical_tables(bank,DEVICE)
    writer=load_visual_writer(START,DEVICE);ink=writer.ink(feat);keys=writer.write_key(ink[mem])
    net=IntervalWriter().to(DEVICE).eval();net.load_state_dict(load_file(str(DEST/"interval.safetensors")))
    orig=load(WORK/"result.json")["arms"]["matched"];result={}
    for tag,(ps,qs,refs) in tests.items():
        images=[];ends=[];actions=[]
        for i in range(0,len(qs),24):
            p,q=to_batch(ps[i:i+24],qs[i:i+24])
            a,_=net(writer.encode(ink,p,q),p)
            actions.extend((a>=0).tolist())
            predicted,stop=net.generate(writer,ink,p,q,keys,mem)
            images.extend(pack(row[row!=0]) for row in predicted.cpu().numpy());ends.extend(stop.tolist())
        path=DEST/(tag+".npz");image_archive(path,images,ends)
        ok=np.array([s and img in ref for s,img,ref in zip(ends,images,refs,strict=True)])
        rec=dict(exact=int(ok.sum()),rows=len(ok),read=sum(actions),stopped=sum(ends),archive=sha(path))
        if tag in orig:
            old,done=archived(WORK/"matched"/(tag+".npz"),orig[tag]["archive"])
            was=np.array([s and pix in ref for s,pix,ref in zip(done,old,refs,strict=True)])
            rec.update(gained=int((ok&~was).sum()),lost=int((was&~ok).sum()))
        if tag=="same_sentence":
            both=ok.reshape(-1,2).all(1);rec.update(both=int(both.sum()),human_both=int(both[mask].sum()))
        elif tag in ("overlap_who","drcd"):
            who_ok=ok if tag=="overlap_who" else ok[who]
            rec.update(who=int(who_ok.sum()),who_pairs=int(who_ok[pairs].all(1).sum()))
        elif tag in ("explicit","historic_identify","print_identify"):
            sq=ok.reshape(-1,2,2)
            rec.update(both_orders=int(sq.all(1).sum()),both_questions=int(sq.all(2).sum()),all_four=int(sq.all((1,2)).sum()))
        result[tag]=rec
    write_json(PUBLIC/"calibrated.json",dict(scores=result,fit=load(DEST/"fit.json"),
               binding=sha(PUBLIC/"calibration_binding.json")))
