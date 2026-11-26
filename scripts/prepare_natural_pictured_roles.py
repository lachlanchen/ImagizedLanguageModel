"""Align UD transitive-clause heads to original English text and fixed visual cells."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests
from scripts.reserve_visual_strip_restoration import ROOT, dump, read, sha

RUN=ROOT/"artifacts/natural_pictured_roles_v1"
PUB=ROOT/"references/evidence/natural_pictured_roles_v1"
MODELS=ROOT/"models/natural_pictured_roles"
RAW=ROOT/"data/raw/ud_english_roles_215"
FONT=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
REVS={"EWT":"4dc8e10cf32352e11ab2c46e024b19853b91546e","GUM":"34d01cb603867d0c085896e8286a0fe01229fa37"}
WINDOW=224


def fetch(corpus,name):
    file=RAW/corpus/name
    if not file.exists():
        file.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-{corpus}/{REVS[corpus]}/{name}"
        res=requests.get(url,timeout=90);res.raise_for_status()
        tmp=file.with_suffix(".part");tmp.write_bytes(res.content);os.replace(tmp,file)
    return file


def clauses(block,counts,corpus):
    meta=dict(line[2:].split(" = ",1) for line in block.splitlines() if line.startswith("# ") and " = " in line)
    text=meta.get("text","")
    if not text:
        return []
    tokens=[];children=defaultdict(list);offset=0
    for line in block.splitlines():
        fields=line.split("\t")
        if len(fields)!=10 or not fields[0].isdigit():
            continue
        start=text.find(fields[1],offset)
        if start<0 or text[offset:start].strip():
            counts["unaligned_sentences"]+=1
            return []
        offset=start+len(fields[1])
        token=dict(i=int(fields[0]),w=fields[1],tag=fields[3],head=int(fields[6]),rel=fields[7],span=(start,offset))
        tokens.append(token);children[token["head"]].append(token)
    rows=[]
    for v in tokens:
        if v["tag"]!="VERB":continue
        deps=children[v["i"]]
        roles=defaultdict(list)
        for dep in deps:roles[dep["rel"]].append(dep)
        if len(roles["nsubj:pass"])==len(roles["obl:agent"])==1:
            a,b=roles["obl:agent"][0],roles["nsubj:pass"][0]
            if not any(x["rel"]=="case" and x["w"].lower()=="by" for x in children[a["i"]]):continue
            voice="passive"
        elif len(roles["nsubj"])==len(roles["obj"])==1:
            a,b=roles["nsubj"][0],roles["obj"][0]
            voice="active"
        else:continue
        if not all(arg["tag"] in ("NOUN","PROPN","PRON") for arg in (a,b)):continue
        spans=[v["span"],a["span"],b["span"]]
        lo=min(x[0] for x in spans);hi=max(x[1] for x in spans)
        if hi-lo>WINDOW:continue
        left=max(0,min(len(text)-WINDOW,(lo+hi-WINDOW)//2))
        shifted=[(x-left,y-left) for x,y in spans]
        vspan,ag,pat=shifted
        rows.append(dict(corpus=corpus,sentence=meta.get("sent_id"),text=text[left:left+WINDOW],
            full=sha256(" ".join(text.casefold().split()).encode()).hexdigest(),voice=voice,verb=vspan,
            candidates=sorted([ag,pat]),label=int(ag>pat)))
    return rows


def bitmap(char,font):
    image=Image.new("L",(32,32),0)
    draw=ImageDraw.Draw(image)
    box=draw.textbbox((0,0),char,font=font)
    # Common baseline and origin preserve differences between letters and punctuation.
    draw.text((max(0,(32-(box[2]-box[0]))//2-box[0]),0),char,font=font,fill=255)
    return np.array(image)


def main():
    RUN.mkdir(exist_ok=True);PUB.mkdir(exist_ok=True,parents=True);MODELS.mkdir(exist_ok=True,parents=True)
    seen=set();out={};hashes={};filters=Counter()
    for split in ("test","dev","train"):
        rows=[]
        seen_split=set()
        for corpus in REVS:
            file=fetch(corpus,f"en_{corpus.lower()}-ud-{split}.conllu")
            hashes[f"{corpus}/{file.name}"]=sha(file)
            for block in file.read_text().split("\n\n"):
                group=clauses(block,filters,corpus)
                if not group:continue
                key=group[0]["full"]
                if key in seen or key in seen_split:
                    filters["duplicate_sentences"]+=1
                    continue
                seen_split.add(key)
                rows.extend(group)
        seen.update(seen_split)
        dump(RUN/(split+".json"),rows);out[split]=rows
    # Every represented character uses a physical raster. Resolve identical ink to one address.
    font=ImageFont.truetype(str(FONT),24)
    marks=sorted({c for rows in out.values() for r in rows for c in r["text"]})
    from_bits={};palette=[np.zeros((32,32),dtype=np.uint8)];codes={}
    for c in marks:
        ink=bitmap(c,font);key=ink.tobytes()
        if key not in from_bits:from_bits[key]=len(palette);palette.append(ink)
        codes[c]=from_bits[key]
    np.save(MODELS/"palette.npy",np.stack(palette));dump(RUN/"codes.json",codes)
    summary={s:dict(Counter(r["corpus"]+"/"+r["voice"] for r in rows)) for s,rows in out.items()}
    dump(PUB/"corpus.json",dict(revisions=REVS,files=hashes,filters=filters,counts=summary,
        split_hashes={s:sha(RUN/(s+".json")) for s in out},palette=sha(MODELS/"palette.npy"),
        font=sha(FONT),protocol=sha(ROOT/"references/natural_pictured_roles_protocol.md")))
    print(summary,flush=True)


if __name__=="__main__": main()
