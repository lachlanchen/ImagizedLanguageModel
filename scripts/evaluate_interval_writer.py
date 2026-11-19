"""Read all complete outputs on shared sentences, ordinary passages and free-writing tests."""
import numpy as np
import torch
from safetensors.torch import load_file, load_model

from ilm.visual_lm.interval_writer import IntervalWriter
from ilm.visual_lm.ordered_passage_planner import OrderedPassageComposer
from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_inverse_role_writer import available_sets
from scripts.evaluate_same_passage_writer import generate, image_archive
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.infer_pictured_entry import page
from scripts.measure_natural_role_questions import BASE
from scripts.shared_sentence_corpus import SentenceCorpus
from scripts.train_interval_writer import (DEVICE, PREVIOUS, PUBLIC, ROOT, START, WORK, WHO, load, load_visual_writer,
    memory_bank, physical_tables, sha, to_batch, write_json)


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(2)
    bank=memory_bank();tests,dr,_=available_sets(bank)
    corpus=SentenceCorpus(dr,bank,"development")
    group=[];ps=[];qs=[];refs=[];people=[]
    human=set(corpus.people)
    pack=lambda seq:b"".join(bank.tiles[k] for k in seq)
    for record in corpus.pairs:
        a,b,l,r=record
        group.append(dr.groups[a])
        people.append(record in human)
        for p,q,spans in corpus.pair(record):
            ps.append(p);qs.append(q)
            # Include every annotated length, also when it exceeds the copy limit.
            i=a if len(ps)%2 else b
            refs.append({pack(dr.passages[dr.groups[i]][u:v]) for u,v in dr.spans[i] if l<=u and v<=r})
    tests["same_sentence"]=(ps,qs,refs)
    people=np.asarray(people)
    with np.load(WHO/"cohort.npz") as z:who=z["development"];pairs=z["development_pairs"]
    p,q,y=tests["drcd"]
    from scripts.train_visual_role_writer import Tiles
    marks=set(Tiles(bank)("。！？!?"))
    scoped=[]
    for i in who:
        regions=corpus.bounds[dr.groups[i]]
        query=set(q[i])-marks
        def similarity(region):
            a,b=region;bits=set(p[i][a:b])-marks
            return len(bits&query)/max(1,len(bits|query))
        a,b=max(regions,key=similarity);scoped.append(p[i][a:b])
    tests["overlap_who"]=(scoped,[q[i] for i in who],[y[i] for i in who])
    feat,mem,_=physical_tables(bank,DEVICE)
    writer=load_visual_writer(START,DEVICE);ink=writer.ink(feat);keys=writer.write_key(ink[mem])
    outcomes={};report={}
    for arm in ("matched","randomized","parent","passage_reader"):
        dest=WORK/arm;dest.mkdir(exist_ok=True)
        if arm=="passage_reader":
            reader=OrderedPassageComposer().eval()
            path=BASE/"mixed.safetensors"
            assert sha(path)==load(BASE/"mixed_result.json")["weights_sha256"]
            load_model(reader,str(path))
        elif arm!="parent":
            net=IntervalWriter().to(DEVICE).eval();weights=dest/"interval.safetensors"
            assert sha(weights)==load(dest/"fit.json")["weights"]
            net.load_state_dict(load_file(str(weights)))
        outcomes[arm]={};stats={}
        for tag,(ps,qs,refs) in tests.items():
            if arm in ("parent","passage_reader") and tag not in ("same_sentence","overlap_who"):continue
            if arm=="passage_reader" and tag=="overlap_who":
                rec=load(ROOT/"references/evidence/inverse_role_writer_20261119/sentence_controls.json")["arms"][arm]
                imgs,ends=archived(PREVIOUS/"sentence_controls/passage_reader.npz",rec["archive"])
            elif arm=="passage_reader":
                grid=lambda s:np.frombuffer(pack(s),np.uint8).reshape(-1,128)
                imgs,ends=zip(*(reader.answer(grid(p),grid(q)) for p,q in zip(ps,qs,strict=True)),strict=True)
            elif arm=="parent":
                imgs,ends=generate(writer,ink,mem,keys,ps,qs,bank)
            else:
                imgs=[];ends=[]
                for start in range(0,len(ps),24):
                    p,q=to_batch(ps[start:start+24],qs[start:start+24])
                    out,stop=net.generate(writer,ink,p,q,keys,mem,limit=64)
                    imgs.extend(b"".join(bank.tiles[k] for k in row if k) for row in out.tolist())
                    ends.extend(stop.tolist())
            path=dest/(tag+".npz");image_archive(path,imgs,ends)
            ok=np.array([bool(s and pix in ref) for s,pix,ref in zip(ends,imgs,refs,strict=True)])
            rec=dict(exact=int(ok.sum()),total=len(ok),stopped=int(sum(ends)),archive=sha(path))
            if tag=="same_sentence":
                both=ok.reshape(-1,2).all(1)
                rec.update(pairs=len(both),both=int(both.sum()),human_pairs=int(people.sum()),
                           human_both=int(both[people].sum()),human_exact=int(ok.reshape(-1,2)[people].sum()))
                outcomes[arm].update(shared=both,shared_who=both[people])
                if arm=="matched":
                    first=int(np.flatnonzero(people)[0])*2
                    for j in (first,first+1):
                        for name,raw in (("passage",pack(ps[j])),("question",pack(qs[j])),("answer",imgs[j])):
                            page(raw).save(dest/f"pair{j-first}_{name}.png")
            elif tag in ("drcd","overlap_who"):
                single=ok[who] if tag=="drcd" else ok
                both=single[pairs].all(1)
                rec.update(who=int(single.sum()),who_pairs=int(both.sum()))
                if tag=="overlap_who":outcomes[arm].update(overlap=single,overlap_pairs=both)
            elif tag in ("historic_four","print_four"):
                rec["all_rotations"]=int(ok.reshape(-1,4,4).all(1).sum())
            elif tag in ("print_identify","historic_identify","print_events"):
                rec["both_orders"]=int(ok.reshape(-1,2,2).all(1).sum())
            stats[tag]=rec
        report[arm]=stats
        print(arm,{k:(v["exact"],v.get("human_both")) for k,v in stats.items()},flush=True)
    effects={}
    clusters=dict(shared=group,shared_who=np.asarray(group)[people],overlap=np.asarray(dr.groups)[who],
                  overlap_pairs=np.asarray(dr.groups)[who[pairs[:,0]]])
    for tag,g in clusters.items():
        effects[tag]={arm:paired_interval(outcomes["matched"][tag],outcomes[arm][tag],g) for arm in
                      ("randomized","parent","passage_reader")}
    write_json(WORK/"result.json",dict(arms=report,effects=effects,registration=sha(PUBLIC/"registration.json"),
               fits={a:load(WORK/a/"fit.json") for a in ("matched","randomized")}))
