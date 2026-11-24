"""Build an optically addressed reference collection from verified local sources."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.atlas_questions import FIELDS, TRAIN, HELD, LIMIT, corpus_chars, render_question
from scripts.prepare_verse_canvas import FACES, draw
from scripts.prepare_source_plate_questions import SPLITS,CACHE
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.prepare_dialogue_ink import ROOT,sha,dump

WORK = ROOT/"artifacts/glyph_atlas_v1"
MODEL = ROOT/"models/glyph_atlas"
PUB = ROOT/"references/evidence/glyph_atlas_v1"


def read(path):
    return json.loads(path.read_text())


def cleaned(value):
    text = " ".join(value.split())
    if text.lower().strip(". /") in ("","not exist","not exists","not applicable","none","unknown"):
        return ""
    return text


def questions(entries,section,ids):
    train = section=="train"
    rng = np.random.default_rng(20930 if train else 20931)
    pool = np.array([i for i,e in enumerate(entries) if e["split"]==section])
    q,marks,head,book,kind,languages,groups = [],[],[],[],[],[],[]
    def add(index,foil,fields,lang,frame,v,group,typ):
        text,pos,labels = render_question(frame,entries[index]["char"],entries[foil]["char"],fields,lang,v)
        encoded = np.zeros(LIMIT,np.uint16)
        encoded[:len(text)] = [ids[c] for c in text]
        tag = np.zeros(LIMIT,np.uint8)
        tag[:len(text)] = labels
        q.append(encoded); marks.append(tag); head.append(pos); book.append(index)
        kind.append(typ); languages.append(lang); groups.append(group)
    if train:
        for i in range(64000):
            a,b = rng.choice(pool,2,replace=False)
            lang = int(rng.integers(2))
            frame = rng.choice(TRAIN[lang])
            fields = [6] if rng.random()<.2 else rng.choice(6,int(rng.integers(1,4)),replace=False)
            add(a,b,fields,lang,frame,int(rng.integers(2)),i,"train")
    else:
        for i,a in enumerate(pool):
            b = pool[(i+1)%len(pool)]
            fields = rng.choice(6,2,replace=False).tolist()
            for lang in (0,1):
                for k,templates in enumerate((TRAIN,HELD)):
                    frame = templates[lang][(i%len(templates[lang])) if k else 2+i%4]
                    group = (i*2+lang)*2+k
                    for x,y in ((a,b),(b,a)):
                        for order in (fields,fields[::-1]):
                            add(x,y,order,lang,frame,i%2,group,("familiar","novel")[k])
                    add(a,b,[6],lang,frame,i%2,group,("familiar-origin","novel-origin")[k])
    return dict(query=np.stack(q),tag=np.stack(marks),head=np.asarray(head,np.uint8),
                entry=np.asarray(book,np.int32),kind=np.asarray(kind),lang=np.asarray(languages,np.uint8),
                group=np.asarray(groups,np.int32))


def prepare():
    for p in (WORK,MODEL,PUB):
        p.mkdir(exist_ok=True,parents=True)
    report = read(SPLITS/"report.json")
    if sha(CACHE)!=report["raster_cache"]["cache_sha256"]:
        raise ValueError("Historical cache digest differs")
    raster = Rasterizer()
    image = torch.load(CACHE,weights_only=True,map_location="cpu")["pixels"]
    packed = np.packbits(image.numpy().reshape(len(image),1024),axis=1)
    del image
    pixel_ids = {bytes(128):0}
    def store(pixel):
        return pixel_ids.setdefault(pixel,len(pixel_ids))
    entries,heads,spans,stream = [],[],[],[]
    rejected = Counter()
    used = set()
    for split in ("train","development"):
        path = SPLITS/(split+".json")
        if sha(path)!=report["partitions"][split]["file_sha256"]:
            raise ValueError(split)
        candidates = sorted(read(path),key=lambda e:(e["char"],-len(set(g["stage"] for g in e["glyphs"])),
                                               -len(e["glyphs"]),e["entry_id"]))
        for e in candidates:
            char=e["char"]
            try:
                key=raster.tile(char)
            except ValueError:
                rejected["head-unrenderable"]+=1
                continue
            if key in used:
                rejected["duplicate head image"]+=1
                continue
            slices=[]
            sources=[]
            for f in FIELDS:
                bits=[]
                source=None
                if f in FIELDS[:4]:
                    options=sorted((g for g in e["glyphs"] if g["stage"]==f),key=lambda g:g["label"])
                    if options:
                        g=options[0]
                        tile=packed[g["raster_index"]].tobytes()
                        if hashlib.sha256(tile).hexdigest()!=g["raster_sha256"] or not any(tile):
                            raise ValueError(g["label"])
                        bits=[store(tile)]
                        source=g["label"]
                else:
                    source=cleaned(e["fields"].get(f,""))
                    try:
                        bits=[store(raster.tile(c)) for c in source]
                    except ValueError:
                        rejected["field-unrenderable"]+=1
                        source=None
                        bits=[]
                slices.append((len(stream),len(bits)))
                stream.extend(bits);sources.append(source)
            if not any(n for _,n in slices):
                rejected["empty record"]+=1
                continue
            used.add(key); heads.append(store(key)); spans.append(slices)
            entries.append(dict(char=char,entry=e["entry_id"],family=e["family_sha256"],
                                split=split,sources=sources))
    book=WORK/"atlas.npz"
    tiles=np.frombuffer(b"".join(pixel_ids),np.uint8).reshape(-1,128)
    np.savez_compressed(book,tiles=tiles,heads=np.array(heads,np.int32),spans=np.array(spans,np.int32),
                        stream=np.array(stream,np.int32))
    dump(WORK/"sources.json",entries)
    chars=[""]+sorted(corpus_chars()|{e["char"] for e in entries})
    codes=np.zeros((len(chars),4),np.uint8)
    for i,c in enumerate(chars):
        codes[i,:len(c.encode())]=list(c.encode())
    faces=[draw(chars[1:],font) for font in FACES]
    # Spaces and padding share white ink. Array indices are used only to cache observations.
    qt=np.array([[np.frombuffer(face.get(c,bytes(128)),np.uint8) for c in chars] for face in faces])
    ids={c:i for i,c in enumerate(chars)}
    keys=[ids[e["char"]] for e in entries]
    if not np.array_equal(qt[0,keys],tiles[heads]):
        raise ValueError("The two canonical renderers disagree")
    families=[{e["family"] for e in entries if e["split"]==split} for split in ("train","development")]
    if families[0]&families[1]:
        raise ValueError("Shared headword families")
    np.savez_compressed(WORK/"observations.npz",tiles=qt,bytes=codes,keys=keys)
    counts={}
    for split in ("train","development"):
        records=questions(entries,split,ids)
        np.savez_compressed(WORK/(split+".npz"),**records)
        counts[split]=dict(questions=len(records["head"]),entries=sum(e["split"]==split for e in entries))
    files={p.name:sha(p) for p in [book,WORK/"sources.json",WORK/"observations.npz",WORK/"train.npz",WORK/"development.npz"]}
    dump(PUB/"corpus.json",dict(files=files,counts=counts,family_overlap=0,rejected=dict(rejected),
                               glyph_cache=sha(CACHE),fields=list(FIELDS),source_split=sha(SPLITS/"report.json"),
                               source_pixels=len(tiles),book_bytes=book.stat().st_size))
    print(counts,len(tiles),rejected,flush=True)


if __name__=="__main__":
    prepare()
