"""Decode source-matched ink for inspecting short answers and copied evidence sentences."""
from collections import defaultdict, Counter
import json
import numpy as np
from scripts.audit_ordered_page_answers import read_outputs
from scripts.cache_ordered_page_features import PUBLIC
from scripts.compare_ordered_requests import OUT
from scripts.ordered_image_corpus import corpus,Face,FONTS
from scripts.pictured_participant_scenes import fresh_cases
from scripts.train_ordered_page_writer import dump

def run():
    rows=corpus("development")
    scenes=list(fresh_cases())
    by_id={int(r.case):(i,r) for i,r in enumerate(rows) if r.task=="single"}
    render=Face((FONTS[0],2))
    chars={c for p,q,y in scenes for c in p+q+y}
    lut={render(c):c for c in chars}
    assert len(lut)==len(chars)
    v=json.loads((PUBLIC/"variation.json").read_text())["arms"]["varied"]
    outs,stops=read_outputs(OUT,"varied",v["output_sha256"])
    readings=[]
    counts=Counter()
    by_passage=defaultdict(list)
    for j,(p,q,y) in enumerate(scenes):
        idx,r=by_id[j]
        assert r.passage==render(p) and r.question==render(q)
        raw=outs[idx]
        txt="".join(lut.get(raw[i:i+128],"□") for i in range(0,len(raw),128))
        readings.append(dict(index=j,passage=p,question=q,returned=txt,stopped=bool(stops[idx]),
                             unknown_cells=txt.count("□")))
        by_passage[p].append(j)
        if raw==r.passage: counts['whole_passage']+=1
    for p,ids in by_passage.items():
        assert len(ids)==2
        if readings[ids[0]]["returned"]==readings[ids[1]]["returned"]: counts["same_output_both_questions"]+=1
        for n,j in enumerate(ids):
            y=scenes[j][2]
            other=scenes[ids[1-n]][2]
            r=readings[j]["returned"]
            counts[("wanted" if y in r else "no_wanted")+("_other" if other in r else "_no_other")]+=1
    dump(PUBLIC/"role_readings.json",dict(counts=dict(counts),rows=readings,
        transcription="Exact lookup over the glyphs in these 64 printed passages and prompts; square marks unrecognized pixels."))
    for r in readings:
        print(r["index"],r["question"],r["returned"])
    print(counts)

if __name__=="__main__":
    run()
