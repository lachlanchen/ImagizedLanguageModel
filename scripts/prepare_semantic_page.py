"""Render the cleaned human E2E NLG descriptions and bilingual input cards."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import requests

from scripts.prepare_dialogue_ink import ROOT,sha,dump
from scripts.prepare_visual_sequence_corpus import Rasterizer

REV="164acdf986146d76f18893b3de9488725deda4c2"
RAW=ROOT/"data/raw/e2e_cleaned"
WORK=ROOT/"artifacts/semantic_page_v1"
PUB=ROOT/"references/evidence/semantic_page_v1"
MODEL=ROOT/"models/semantic_page"
FILES={"train":"train-fixed.no-ol.csv","dev":"devel-fixed.no-ol.csv","test":"test-fixed.csv"}
KEYS=("name","eatType","food","priceRange","customer rating","area","familyFriendly","near")
ZH=("名称","类别","菜系","价格","顾客评分","地点","适合家庭","邻近")
VALUES={"coffee shop":"咖啡馆","restaurant":"餐厅","pub":"酒馆","Chinese":"中餐","English":"英式","Fast food":"快餐","French":"法式","Indian":"印度菜","Italian":"意式","Japanese":"日式","cheap":"便宜","high":"高","less than £20":"低于20英镑","moderate":"中等","more than £30":"高于30英镑","£20-25":"20至25英镑","low":"低","average":"平均","1 out of 5":"一星","3 out of 5":"三星","5 out of 5":"五星","city centre":"市中心","riverside":"河边","no":"否","yes":"是"}
WIDTH=40


def fetch():
    base=f"https://raw.githubusercontent.com/tuetschek/e2e-cleaning/{REV}/"
    raw=requests.get(f"https://api.github.com/repos/tuetschek/e2e-cleaning/git/trees/{REV}",params={"recursive":1},timeout=30)
    raw.raise_for_status()
    blobs={r["path"]:r["sha"] for r in raw.json()["tree"]}
    RAW.mkdir(parents=True,exist_ok=True)
    files={}
    for name in [*FILES.values(),"slot_error.py","README.md"]:
        key="cleaned-data/"+name if name.endswith(".csv") else name
        p=RAW/name
        if p.exists():
            bits=p.read_bytes()
        else:
            response=requests.get(base+key,timeout=40);response.raise_for_status()
            bits=response.content
        if hashlib.sha1(f"blob {len(bits)}\0".encode()+bits).hexdigest()!=blobs[key]:
            raise ValueError("Source digest differs: "+name)
        if not p.exists():p.write_bytes(bits)
        files[name]=hashlib.sha256(bits).hexdigest()
    return files


def prepare():
    inventory=fetch()
    for p in (MODEL,PUB,WORK):p.mkdir(parents=True,exist_ok=True)
    rows={}
    pairs=set()
    for split,file in FILES.items():
        with (RAW/file).open(encoding="utf8",newline="") as f:
            result=[]
            for row in csv.DictReader(f):
                facts=tuple(sorted((k.strip(),v) for k,v in re.findall(r'([^,\[]+)\[([^\]]*)\]',row["mr"])))
                if not facts or any(k not in KEYS for k,v in facts):
                    raise ValueError(row["mr"])
                pairs.update(facts)
                result.append((facts," ".join(row["ref"].split())))
            rows[split]=result
    concepts=[None]+sorted(pairs)
    ci={v:i for i,v in enumerate(concepts)}
    surfaces=[["",""]]+[[k+": "+v,ZH[KEYS.index(k)]+"："+VALUES.get(v,v)] for k,v in concepts[1:]]
    raster=Rasterizer()
    chars={c for sides in surfaces for s in sides for c in s}
    chars.update(c for records in rows.values() for _,r in records for c in r)
    alphabet=[""]+sorted(chars)
    glyphs={}
    for c in alphabet[1:]:
        try:glyphs[c]=raster.tile(c)
        except ValueError:pass
    input_chars=[""]+[c for c in alphabet[1:] if c in glyphs]
    ix={c:i for i,c in enumerate(input_chars)}
    tiles=np.array([np.frombuffer(glyphs.get(c,bytes(128)),np.uint8) for c in input_chars])
    codes=np.zeros((len(ix),4),np.uint8)
    for c,i in ix.items():codes[i,:len(c.encode())]=list(c.encode())
    surface=np.zeros((2,len(concepts),WIDTH),np.int32)
    for i,both in enumerate(surfaces):
        for lang,s in enumerate(both):
            surface[lang,i,:len(s)]=[ix[c] for c in s]
    out_chars=sorted({c for _,r in rows["train"] for c in r if c in glyphs})
    # Row zero denotes stop. Visible whitespace is a distinct printable output.
    physical={};out_ids={};reps=[""]
    for c in out_chars:
        pixel=glyphs[c]
        if pixel not in physical:
            physical[pixel]=len(reps);reps.append(c)
        out_ids[c]=physical[pixel]
    palette=np.array([np.zeros(128,np.uint8)]+[np.frombuffer(glyphs[c],np.uint8) for c in reps[1:]])
    np.save(MODEL/"palette.npy",palette)
    dump(MODEL/"readback.json",reps)
    info={}
    archives={}
    prior=set()
    seen_train={fact for fact,_ in rows["train"]}
    for split,records in rows.items():
        accepted=[(facts,r) for facts,r in records if r and (split!="train" or all(c in out_ids for c in r))]
        slots=np.zeros((len(accepted),max(len(facts) for facts,_ in accepted)),np.int32)
        width=max(len(r) for _,r in accepted)+1
        y=np.full((len(accepted),width),-1,np.int16)
        for i,(facts,r) in enumerate(accepted):
            slots[i,:len(facts)]=[ci[p] for p in facts]
            y[i,:len(r)]=[out_ids.get(c,-2) for c in r];y[i,len(r)]=0
        np.savez_compressed(WORK/(split+".npz"),slots=slots,y=y)
        dump(WORK/(split+"-refs.json"),[r for _,r in accepted])
        unique={f for f,r in accepted}
        info[split]=dict(references=len(accepted),records=len(unique),skipped=len(records)-len(accepted),
                         training_overlap=len(unique&seen_train) if split!="train" else None)
        archives[split]=sha(WORK/(split+".npz"))
    np.savez_compressed(WORK/"bank.npz",tiles=tiles,bytes=codes,surface=surface)
    dump(WORK/"schema.json",dict(concepts=concepts,surfaces=surfaces,input_chars=input_chars))
    dump(PUB/"corpus.json",dict(revision=REV,files=inventory,splits=info,archives=archives,
         bank=sha(WORK/"bank.npz"),schema=sha(WORK/"schema.json"),vocab=len(reps),palette=sha(MODEL/"palette.npy")))
    print(info,len(reps),flush=True)


if __name__=="__main__":
    prepare()
