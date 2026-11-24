"""Audit physical overlap and independently recount complete responses from archived ink."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np

from scripts.prepare_visual_clause_roles import ROOT, PUBLIC, WORK, read, dump, sha
from scripts.prepare_pronoun_writer import window
from scripts.train_asked_gloss import rows_and_split
from scripts.train_typeface_writer import Face,FACES


def main():
    face=Face(FACES[0])
    @lru_cache(None)
    def glyph(c):return face.cell(c)
    # Hash all original target windows, including training pages excluded from
    # the new relation corpus. Exact byte strings cover visually identical forms.
    prev,_=rows_and_split()
    old={tuple(map(glyph,window(r["page"],i))) for r in prev if r["split"]=="train" for i in r["focus"]}
    rows=read(PUBLIC/"readings.json")
    overlap=[i for i,r in enumerate(rows) if r.get("primary") and any(tuple(map(glyph,window(r["page"],j))) in old for j in r["focus"])]
    q=read(PUBLIC/"replay.json")
    assert sha(PUBLIC/"emitted.npz")==q["pixels_sha256"]
    table={glyph(c):c for c in "abcdefghijklmnopqrstuvwxyz-"}
    checked={}
    with np.load(PUBLIC/"emitted.npz") as raw:
        offsets=raw["offsets"]; ink=raw["ink"]
        assert len(offsets)==1+len(q["replies"]) and offsets[0]==0 and offsets[-1]==len(ink)
        for a,(i,request,pos,ended,_) in enumerate(q["replies"]):
            strip=ink[offsets[a]:offsets[a+1]]
            word="".join(table.get(t.tobytes(),"�") for t in strip) if ended else "<unfinished>"
            k=request%2;r=rows[i]
            checked[i,request]=pos==r["focus"][k] and word in r["gloss"][k]
    d=read(PUBLIC/"transfer_result.json")
    pairs=[(i,i+1) for i,r in enumerate(rows) if r.get("primary") and r["side"]=="front"]
    strict=sum(all(checked[i,q] for i in pair for q in range(4)) for pair in pairs)
    swapped=sum(checked[i,0] and checked[i,2] and checked[j,1] and checked[j,3] for i,j in d["polysemy"]["indices"])
    assert strict==d["opposite_order_written"]["classroom"][2] and swapped==d["polysemy"]["classroom_written"][2]
    logs=[read(p) for p in sorted((WORK/"teacher").glob("raw*.json"))]
    dump(PUBLIC/"audit.json",dict(previous_focus_window_overlap=overlap,actual_ink_replies=len(checked),primary_complete=strict,
        distinct_readings_complete=swapped,teacher_calls=len(logs),teacher_generation_seconds=sum(r["generation_ns"] for r in logs)/1e9,
        teacher_reported_seconds=sum(r["duration_ns"] for r in logs)/1e9,
        teacher_generated_tokens=sum(r["tokens"] for r in logs)))
    print("additional overlap",overlap,"scores",strict,swapped)


if __name__=="__main__":main()
