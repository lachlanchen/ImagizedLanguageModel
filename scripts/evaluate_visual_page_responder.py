"""Evaluate a fixed image-only copy-or-compose rule on semantic and writing tasks."""

import fcntl
import json
import os
import tempfile
from zipfile import ZipFile

import numpy as np
import torch

from ilm.visual_lm.visual_page_responder import VisualPageResponder
from scripts.evaluate_inverse_role_writer import available_sets, history
from scripts.evaluate_same_passage_writer import image_archive
from scripts.explicit_participant_cases import cases
from scripts.fit_font_invariant_reader import Face,FONTS,KAI,PUBLIC,WORK,ROOT,PARENT_PUBLIC,sha,write_json
from scripts.infer_pictured_entry import page
from scripts.pictured_participant_scenes import fresh_cases
from scripts.train_interval_writer import WHO, WORK as OLD, PUBLIC as OLD_PUBLIC, SentenceCorpus, memory_bank
from scripts.train_visual_role_writer import Tiles

DEST=WORK/"responder"


def evaluate():
    DEST.mkdir()
    bank=memory_bank()
    sets,dr,_=available_sets(bank)
    pack=lambda seq:b"".join(bank.tiles[i] for i in seq)
    tests={}
    for name in ("historic_identify","historic_describe","historic_four","familiar_write","new_layout_write","dureader","drcd","english_free","chinese_free"):
        p,q,y=sets[name]
        tests[name]=[(pack(a),pack(b),r) for a,b,r in zip(p,q,y,strict=True)]
    with np.load(WHO/"cohort.npz") as z:who=z["development"];duo=z["development_pairs"]
    corpus=SentenceCorpus(dr,bank,"development")
    marks=set(Tiles(bank)("。！？!?"))
    tests["overlap_who"]=[]
    for i in who:
        p=dr.passages[dr.groups[i]];q=set(dr.questions[i])-marks
        def score(region):
            x,y=region;cells=set(p[x:y])-marks
            return len(cells&q)/max(1,len(cells|q))
        l,r=max(corpus.bounds[dr.groups[i]],key=score)
        _,question,target=tests["drcd"][i]
        tests["overlap_who"].append((pack(p[l:r]),question,target))
    for tag,rows in (("fresh",list(fresh_cases())),("explicit",list(cases()))):
        for mode,desc in (("sans",(FONTS[0],2)),("bold",(FONTS[2],2)),("kai",KAI)):
            render=Face(desc);records=[]
            for p,q,y in rows:
                try:records.append((render(p),render(q),{render(y)}))
                except ValueError:records.append(None)
            tests[tag+"_"+mode]=records
    archive=OLD/"shared-sentence-writer.zip"
    assert sha(archive)==json.loads((OLD_PUBLIC/"portable.json").read_text())["bundle_sha256"]
    fit=json.loads((WORK/"fit.json").read_text())
    assert all(sha(WORK/n)==v for n,v in fit["weights"].items())
    scores={}
    with tempfile.TemporaryDirectory() as temporary:
        with ZipFile(archive) as z:
            for n in ("interval.safetensors","final.safetensors","memory.npy"):z.extract(n,temporary)
        responder=VisualPageResponder(WORK,temporary,"cuda:0")
        for tag,rows in tests.items():
            imgs=[];ends=[];correct=[];actions=[]
            for i in range(0,len(rows),16):
                batch=rows[i:i+16]
                predicted=iter(responder.answer_batch([(x[0],x[1]) for x in batch if x is not None]))
                for x in batch:
                    out,s,act=next(predicted) if x is not None else (b"",False,"unrenderable")
                    imgs.append(out);ends.append(s);actions.append(act);correct.append(x is not None and s and out in x[2])
            path=DEST/(tag+".npz");image_archive(path,imgs,ends)
            rec=dict(exact=sum(correct),rows=len(rows),read=actions.count("read"),compose=actions.count("compose"),
                     stopped=sum(ends),unrenderable=actions.count("unrenderable"),archive=sha(path))
            if tag.startswith(("fresh_","explicit_")):
                rec["all_four"]=int(np.array(correct).reshape(-1,4).all(1).sum())
            if tag=="overlap_who":rec["pairs"]=int(np.array(correct)[duo].all(1).sum())
            if tag=="historic_four":rec["all_rotations"]=int(np.array(correct).reshape(-1,4,4).all((1,2)).sum())
            if tag in ("fresh_sans","fresh_bold","fresh_kai","historic_describe","familiar_write","overlap_who","english_free"):
                for i in range(min(2,len(rows))):
                    if rows[i] is None:continue
                    for label,raw in (("passage",rows[i][0]),("question",rows[i][1]),("answer",imgs[i])):
                        page(raw).save(DEST/f"{tag}_{i}_{label}.png")
                    rec.setdefault("examples",[]).append(dict(index=i,action=actions[i],exact=bool(correct[i]),
                        source_cells=len(rows[i][0])//128,question_cells=len(rows[i][1])//128))
            scores[tag]=rec
            print(tag,rec['exact'],rec['rows'],rec['read'],flush=True)
    write_json(PUBLIC/"responder.json",dict(arms=scores,runner=sha(__file__),reader=sha(WORK/"fit.json"),
        writer_archive=sha(archive),training_updates=0))


if __name__=="__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="0"
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        evaluate()
