"""Construct blinded local reading material and join frozen judgments to output arms."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import argparse
import json

import numpy as np
from PIL import Image,ImageDraw
from scripts.prepare_composed_chinese_answers import ARCHIVE
from scripts.train_composed_answer_writer import ARMS, DATA, OUT
from scripts.train_paired_passage import unpack_sequences
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json

ARMS_ALL = ("released",*ARMS)
FOLDER = OUT/"blinded_reading"


def panel(seq, columns=32):
    height=max(1,(len(seq)+columns-1)//columns)
    bits=np.zeros((height*columns,32,32),np.uint8)
    if seq:
        bits[:len(seq)]=np.unpackbits(np.frombuffer(b"".join(seq),np.uint8)).reshape(-1,32,32)
    return Image.fromarray(255-255*bits.reshape(height,columns,32,32).transpose(0,2,1,3).reshape(height*32,columns*32))


def prepare():
    rows=load(DATA/"development/sources.json")
    chosen=[]
    for group in sorted({m["kind"] for m in rows}):
        ids=[i for i,m in enumerate(rows) if m["kind"]==group]
        def rank(i):
            meta=rows[i]
            return sha256(("composed-reader-19401/"+meta["member"]+meta["question_id"]).encode()).digest()
        chosen+=sorted(ids,key=rank)[:4]
    tiles=list(map(bytes,np.load(DATA/"development/pixels.npy")))
    def unpack(name):
        return [[tiles[int(i)] for i in s] for s in unpack_sequences(DATA/("development/"+name+".npz"))]
    ps,qs=unpack("passages"),unpack("questions")
    raster=Rasterizer()
    characters=set()
    with ZipFile(ARCHIVE) as archive:
        for name in {r["member"] for r in rows}:
            keys={m["line"]:i for i,m in enumerate(rows) if m["member"]==name}
            with archive.open(name) as f:
                for l,line in enumerate(f):
                    if l not in keys:continue
                    raw=json.loads(line)
                    i=keys[l]
                    m=rows[i]
                    text=raw["documents"][m["doc"]]["paragraphs"][m["paragraph"]][m["start"]:][:len(ps[i])]
                    characters.update(text+raw["question"]+"".join(raw["answers"]))
    charmap=defaultdict(list)
    for c in sorted(characters):
        try: charmap[raster.tile(c)].append(c)
        except ValueError: pass
    def transcription(seq):
        return "".join("/".join(charmap[g]) if len(charmap[g])<3 and charmap[g] else "□" for g in seq)
    outputs={}
    for arm in ARMS_ALL:
        with np.load(OUT/(arm+".npz")) as d:
            outputs[arm]=[list(map(bytes,x)) for x in np.split(d["pixels"],d["offsets"][1:-1])]
    FOLDER.mkdir()
    keys={}
    readings={}
    for i in sorted(chosen):
        imgs={tuple(outputs[a][i]) for a in ARMS_ALL}
        imgs=sorted(imgs,key=lambda s:sha256(str(i).encode()+b"".join(s)).digest())
        code={s:chr(65+j) for j,s in enumerate(imgs)}
        keys[str(i)]={a:code[tuple(outputs[a][i])] for a in ARMS_ALL}
        readings[str(i)]={"question":transcription(qs[i]),"source":transcription(ps[i]),
                          "outputs":{code[s]:transcription(s) for s in imgs}}
        parts=[("question",qs[i]),("source",ps[i])]+[(code[s],s) for s in imgs]
        strips=[(name,panel(seq)) for name,seq in parts]
        sheet=Image.new("L",(1050,sum(im.height+24 for _,im in strips)),255)
        draw=ImageDraw.Draw(sheet);y=0
        for name,im in strips:
            draw.text((8,y+3),name,fill=30);sheet.paste(im,(8,y+19));y+=im.height+24
        sheet.save(FOLDER/(str(i)+".png"))
    write_json(FOLDER/"blind.json",readings)
    write_json(FOLDER/"decode_after_grading.json",keys)


def score():
    judgments=load(FOLDER/"judgments.json")
    mapping=load(FOLDER/"decode_after_grading.json")
    assert set(judgments)==set(mapping)
    rows=load(DATA/"development/sources.json")
    counts=defaultdict(lambda:defaultdict(Counter))
    for key,arms in mapping.items():
        assert set(judgments[key])==set(arms.values())
        for arm,letter in arms.items():
            label=judgments[key][letter]
            assert label in ("supported","partial","incorrect","unreadable")
            counts[arm][rows[int(key)]["kind"]][label]+=1
    write_json(OUT/"semantic_scores.json",dict(counts=counts,judgments_sha256=sha(FOLDER/"judgments.json"),
               blind_sha256=sha(FOLDER/"blind.json"),mapping_sha256=sha(FOLDER/"decode_after_grading.json"),
               reviewer="One model-based assessment, arm labels hidden, question and retrieved passage visible."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","score"))
    if p.parse_args().action=="prepare":prepare()
    else:score()
