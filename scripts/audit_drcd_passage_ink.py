"""Verify prepared passage and question images and every registered answer interval against DRCD."""
from pathlib import Path
import numpy as np
from scripts.prepare_drcd_passage_ink import DEST,RAW
from scripts.prepare_pictured_entry_answers import Rasterizer,load,sha,write_json
from scripts.train_paired_passage import unpack_sequences,ROOT

def audit():
    report=load(DEST/"preparation.json")
    render=Rasterizer()
    result={}
    for s,raw_name in (("train","DRCD_training.json"),("development","DRCD_dev.json")):
        src=RAW/raw_name
        assert sha(src)==report["upstream"]["files"][raw_name]
        raw={q["id"]:(p["context"],q) for a in load(src)["data"] for p in a["paragraphs"] for q in p["qas"]}
        folder=DEST/s
        for f,h in report[s]["files"].items():assert sha(folder/f)==h
        tiles=np.load(folder/"pixels.npy")
        ps=unpack_sequences(folder/"passages.npz")
        qs=unpack_sequences(folder/"questions.npz")
        seen={}
        with np.load(folder/"evidence.npz") as z:
            g,off,spans=z["group"],z["offsets"],z["spans"]
        assert len(qs)==len(g) and off[0]==0 and off[-1]==len(spans)
        for i,qid in enumerate(load(folder/"question_ids.json")):
            text,qa=raw[qid]
            group=int(g[i])
            if group not in seen:
                assert tiles[ps[group]].tobytes()==b"".join(map(render.tile,text))
                seen[group]=text
            assert seen[group]==text and tiles[qs[i]].tobytes()==b"".join(map(render.tile,qa["question"]))
            actual={tuple(pair) for pair in spans[off[i]:off[i+1]]}
            expected=set()
            for a in {a["text"] for a in qa["answers"] if a["text"]}:
                for start in range(len(text)-len(a)+1):
                    if text.startswith(a,start):expected.add((start,start+len(a)))
            assert actual and actual==expected
        assert len(seen)==len(ps)
        result[s]=dict(questions=len(qs),passages=len(ps),all_exact_intervals=len(spans))
    target=ROOT/"references/evidence/drcd_passage_preparation_20261119.json"
    receipt=dict(preparation=report,independent_recount=result)
    if target.exists():
        assert load(target)==receipt, "Existing receipt differs from replay"
    else:
        write_json(target,receipt)
    print(result)

if __name__=="__main__":audit()
