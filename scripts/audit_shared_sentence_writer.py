"""Reconstruct shared sources from DRCD text and regrade complete saved answer images."""
from collections import defaultdict
from itertools import combinations
from functools import lru_cache
import json
import re
import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.explicit_participant_cases import cases
from scripts.infer_pictured_entry import page
from scripts.prepare_pictured_entry_answers import Rasterizer
from scripts.train_interval_writer import DRCD, ROOT, WORK, PUBLIC, WHO, load, sha, write_json


def audit():
    result=load(WORK/"result.json")
    ids=load(DRCD/"development/question_ids.json")
    raw=ROOT/"data/raw/drcd/DRCD_dev.json"
    rows={qa["id"]:(p["context"],qa) for a in load(raw)["data"] for p in a["paragraphs"] for qa in p["qas"]}
    raster=Rasterizer()
    @lru_cache(maxsize=12000)
    def ink(s):
        return b"".join(raster.tile(c) for c in s)
    referenced=[]
    locations=defaultdict(list)
    sections={}
    for i,key in enumerate(ids):
        text,item=rows[key]
        accepted={a["text"] for a in item["answers"] if a["text"] and a["text"] in text}
        referenced.append({ink(s) for s in accepted})
        if text not in sections:
            sections[text]=[(m.start(),m.end(),m.group()) for m in re.finditer(r"[^。！？!?]*[。！？!?]|[^。！？!?]+$",text)]
        for left,right,sentence in sections[text]:
            if any(y in sentence for y in accepted):
                locations[text,left,right].append(i)
    common=defaultdict(list)
    for (text,l,r),members in locations.items():
        for i,j in combinations(members,2):
            if referenced[i]&referenced[j] or ink(rows[ids[i]][1]["question"])==ink(rows[ids[j]][1]["question"]):continue
            common[i,j].append((l,r))
    with np.load(WHO/"cohort.npz") as z:human_pairs=set(map(tuple,z["development"][z["development_pairs"]]))
    group=[];metadata=[];targets=[];mask=[]
    for (i,j),spans in common.items():
        l,r=min(spans,key=lambda s:(s[1]-s[0],s[0]))
        text=rows[ids[i]][0];sentence=text[l:r]
        group.append(text)
        mask.append((i,j) in human_pairs)
        metadata.append((i,j,sentence))
        for index in (i,j):
            q=rows[ids[index]][1]
            targets.append({ink(y["text"]) for y in q["answers"] if y["text"] and y["text"] in sentence})
    assert (len(common),sum(mask))==(864,27)
    mask=np.array(mask);support=load(PUBLIC/"support_reading.json")
    for idx,s in zip(np.flatnonzero(mask),support["pairs"],strict=True):
        i,j,_=metadata[idx];assert [ids[i],ids[j]]==s["ids"]
    audit={};example_indices=[int(np.flatnonzero(mask)[0]),int(np.flatnonzero(mask)[1])]
    explicit=list(cases());exp=load(PUBLIC/"explicit_result.json");cal=load(PUBLIC/"calibrated.json")["scores"]
    for arm in ("parent","passage_reader","randomized","matched","calibrated"):
        rec=cal["same_sentence"] if arm=="calibrated" else result["arms"][arm]["same_sentence"]
        images,ends=archived(WORK/arm/"same_sentence.npz",rec["archive"])
        ok=np.array([end and pix in refs for end,pix,refs in zip(ends,images,targets,strict=True)])
        both=ok.reshape(-1,2).all(1)
        assert (int(ok.sum()),int(both.sum()),int(both[mask].sum()))==(rec["exact"],rec["both"],rec["human_both"])
        er=cal["explicit"] if arm=="calibrated" else exp[arm]
        path=WORK/"calibrated/explicit.npz" if arm=="calibrated" else WORK/"explicit_participants"/(arm+".npz")
        eraw,estop=archived(path,er["archive"])
        eok=np.array([stop and pix==ink(y) for stop,pix,(_,_,y) in zip(estop,eraw,explicit,strict=True)])
        assert int(eok.sum())==er["exact"] and int(eok.reshape(-1,2,2).all((1,2)).sum())==er["all_four"]
        record=dict(shared_pairs=int(both.sum()),direct_pair=bool(both[example_indices[1]]),explicit=int(eok.sum()))
        if arm in ("matched","calibrated","randomized"):
            src=cal["drcd"] if arm=="calibrated" else result["arms"][arm]["drcd"]
            allout,stops=archived(WORK/arm/"drcd.npz",src["archive"])
            assert sum(e and pix in refs for e,pix,refs in zip(stops,allout,referenced,strict=True))==src["exact"]
        if arm=="calibrated":
            examples=[]
            for n,k in enumerate(example_indices):
                i,j,text=metadata[k]
                path=PUBLIC/f"pair{n}_passage.png";page(ink(text)).save(path)
                seq=[(s,len(s)) for s in (text,rows[ids[i]][1]["question"],rows[ids[j]][1]["question"])]
                answers=[]
                for m,t in enumerate((i,j)):
                    q=rows[ids[t]][1]["question"];pix=images[k*2+m]
                    page(ink(q)).save(PUBLIC/f"pair{n}_q{m}.png");page(pix).save(PUBLIC/f"pair{n}_a{m}.png")
                    dictionary={raster.tile(c):c for c in text+q}
                    transcription="".join(dictionary.get(pix[h:h+128],"□") for h in range(0,len(pix),128))
                    answers.append(dict(id=ids[t],question=q,emitted=transcription,
                        correct=bool(ok[k*2+m]),reference=[a["text"] for a in rows[ids[t]][1]["answers"]]))
                examples.append(dict(passage=text,cells=[b for a,b in seq],answers=answers))
            record["examples"]=examples
        audit[arm]=record
    actual=[json.loads(l) for l in (WORK/"explicit_participants/qwen.jsonl").read_text().splitlines()]
    assert [r["index"] for r in actual]==list(range(len(explicit)))
    qok=np.array([r["stop"]=="stop" and r["text"]==y for r,(_,_,y) in zip(actual,explicit,strict=True)])
    assert int(qok.sum())==exp["qwen_text"]["exact"]
    out=PUBLIC/"audit.json"
    if out.exists():raise FileExistsError(out)
    write_json(out,dict(arms=audit,raw_sha256=sha(raw),primary_result=sha(WORK/"result.json"),
                       calibrated_result=sha(PUBLIC/"calibrated.json"),qwen_exact=int(qok.sum()),program=sha(__file__)))


if __name__=="__main__":
    audit()
