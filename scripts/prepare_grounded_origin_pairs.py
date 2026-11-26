"""Prepare image-only questions with an English source gloss and an attested Chinese specimen."""
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from tempfile import TemporaryDirectory

import numpy as np
from scripts.prepare_region_writer import Ink
from scripts.reserve_pictured_path_translation import ROOT, dump, read, sha

DB = ROOT/"data/chinese_origin_library_v1/origin.sqlite3"
OUT = ROOT/"artifacts/grounded_origin_pairs_v1"
RECEIPT = ROOT/"references/evidence/grounded_origin_pairs_v1.json"
STAGES = {"oracle":"甲骨文", "bronze":"金文", "seal":"篆書"}
SPLITS = ("train","development","sealed")


def order(key):
    return hashlib.sha256(key.encode()).hexdigest()


def collect(db, ink):
    data = defaultdict(dict)
    # Even different SVGs can rasterize identically. Screen against every recorded
    # specimen before choosing a target, including those in previously used splits.
    pixels_in = defaultdict(set)
    query = """SELECT e.split,e.headword,e.entry_id,e.family,e.fields,f.stage,f.label,i.pixels32
               FROM entry e JOIN form f USING(headword,entry_id) JOIN image i USING(svg_sha)
               WHERE f.has_ink=1 ORDER BY e.split,e.headword,e.entry_id,f.stage,f.label"""
    for split,head,eid,family,fields,stage,label,bits in db.execute(query):
        pixels = bytes(b ^ 255 for b in bits)
        pixels_in[pixels].add(split)
        if stage not in STAGES: continue
        key = (head,eid)
        if key not in data[split]:
            gloss = " ".join((json.loads(fields).get("original_meaning") or "").split())
            data[split][key]=dict(head=head, entry=eid, family=family, gloss=gloss, forms=defaultdict(list))
        data[split][key]["forms"][stage].append((label,pixels))
    groups={}
    for split in SPLITS:
        group=[]
        for row in data[split].values():
            gloss=row["gloss"]
            if not (7<=len(gloss)<=108 and gloss.startswith("Meaning ") and "not applicable" not in gloss.lower()):
                continue
            ink.encode(row["head"]+gloss)
            if ink.bad.intersection(row["head"]+gloss): continue
            choices={}
            used=set()
            for stage in STAGES:
                for label,glyph in row["forms"].get(stage,[]):
                    if (pixels_in[glyph]=={split} and glyph not in used and
                            glyph not in (bytes(128),bytes([255])*128)):
                        choices[stage]=(label,glyph)
                        used.add(glyph)
                        break
            if len(choices)>=2:
                group.append(dict(row,forms=choices))
        groups[split]=sorted(group,key=lambda r:order(r["family"]+r["entry"]+r["head"]))
    for i,s in enumerate(SPLITS):
        for other in SPLITS[i+1:]:
            for key in ("head","family"):
                assert set(r[key] for r in groups[s]).isdisjoint(r[key] for r in groups[other])
    return groups


def prepare(stage):
    manifest=read(DB.with_name("manifest.json"))
    assert sha(DB)==manifest["database_sha256"]
    ink=Ink()
    with closing(sqlite3.connect(f"file:{DB}?mode=ro",uri=True)) as db:
        groups=collect(db,ink)
    def shape(raw):
        if raw not in ink.byink:
            ink.byink[raw]=len(ink.tiles)
            ink.tiles.append(raw)
        return ink.byink[raw]
    scores={}
    for split,group in groups.items():
        n=0
        total=Counter()
        with (stage/f"{split}.jsonl").open("w") as out:
            for i,a in enumerate(group):
                b=next(group[(i+k)%len(group)] for k in range(1,len(group))
                       if group[(i+k)%len(group)]["family"]!=a["family"] and group[(i+k)%len(group)]["gloss"]!=a["gloss"])
                page=[]
                for entry in (a,b):
                    h=entry["head"]
                    page.append((h,"meaning",ink.encode(h+" · 本義 / meaning · "),ink.encode(entry["gloss"])))
                    for form,(_,raw) in entry["forms"].items():
                        page.append((h,form,ink.encode(h+" · "+STAGES[form]+" / "+form+" · "),[shape(raw)]))
                rng=random.Random(order(split+a["entry"]+a["head"]))
                rng.shuffle(page)
                stages=list(a["forms"])
                rng.shuffle(stages)
                for j,form in enumerate(stages[:2]):
                    for first in (False,True):
                        english=(i+j+int(first))%2==0
                        h=a["head"]
                        name=STAGES[form]
                        if english:
                            q=(f"For {h}, give the {form} form and then the original meaning." if first else
                               f"For {h}, give the original meaning and then the {form} form.")
                        else:
                            q=(f"請先列「{h}」的{name}，再寫本義。" if first else f"請先寫「{h}」的本義，再列{name}。")
                        kinds=(form,"meaning") if first else ("meaning",form)
                        positions=[next(k for k,row in enumerate(page) if row[:2]==(h,kind)) for kind in kinds]
                        answer=[v for j,k in enumerate(positions) for v in ((ink.encode("  ") if j else [])+page[k][3])]
                        record=dict(question=ink.encode(q),page=[dict(label=row[2],value=row[3]) for row in page],answer=answer,
                                    supervision=dict(value_rows=positions,entry=a["entry"],head=h,form=form,
                                                     specimen=a["forms"][form][0],family=a["family"],english=english))
                        out.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
                        n+=1
                        total["gloss_first" if not first else "form_first"]+=1
        scores[split]=dict(entries=len(group),families=len({x["family"] for x in group}),examples=n,order=total)
    # Missing glyphs in discarded dictionary rows can remain in the renderer's
    # cache. Check the shared prompt and label alphabet independently.
    required="For give the form and then original meaning.請先列「」的，再寫本義。 · /"+"".join(STAGES)+"".join(STAGES.values())
    assert not ink.bad.intersection(required)
    np.save(stage/"tiles.npy",ink.array(),allow_pickle=False)
    return dict(source=manifest["database_sha256"],splits=scores,tile_count=len(ink.tiles),
                files={p.name:sha(p) for p in stage.iterdir()},program=sha(Path(__file__)),
                support="Two real headwords per page, labelled field boundaries supplied. Questions and values represented by packed 32-square images. English database glosses in both prompt languages.")


def main():
    if RECEIPT.exists():
        receipt=read(RECEIPT)
        assert receipt["program"]==sha(Path(__file__)) and all(sha(OUT/k)==v for k,v in receipt["files"].items())
        return
    OUT.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        stage=Path(tmp)
        receipt=prepare(stage)
        OUT.mkdir(exist_ok=True)
        for f in stage.iterdir():
            dest=OUT/f.name
            if dest.exists() and sha(dest)!=sha(f): raise ValueError("Changed local corpus: "+dest.name)
            f.replace(dest)
    dump(RECEIPT,receipt)
    print(receipt["splits"],flush=True)


if __name__=="__main__":
    main()
