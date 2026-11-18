"""Prepare traditional-Chinese passage evidence, reserving official articles and shared text."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unicodedata import normalize
import numpy as np
from opencc import OpenCC
from scripts.prepare_pictured_entry_answers import Rasterizer, save_sequences, sha, write_json, load
from scripts.train_paired_passage import ROOT

RAW=ROOT/"data/raw/drcd"
DEST=ROOT/"data/traditional_passage_evidence_v1"
COMMIT="b944790de5af02c5fbb7cd9cb1473d27d169eebf"
convert=OpenCC("t2s")


def canonical(s):
    return "".join(ch for ch in convert.convert(normalize("NFKC",s)).casefold() if ch.isalnum())


class ReservedText:
    def __init__(self):
        self.titles=set()
        self.passages=set()
        self.segments=set()

    def add(self,articles):
        for a in articles:
            self.titles.add(canonical(a["title"]))
            for p in a["paragraphs"]:
                s=canonical(p["context"])
                self.passages.add(s)
                self.segments.update(s[i:i+32] for i in range(0,len(s)-31,32))

    def overlaps(self,s):
        return s in self.passages or any(s[i:i+32] in self.segments for i in range(len(s)-31))


def prepare_split(articles, reserved, folder):
    folder.mkdir()
    raster=Rasterizer()
    bank={}
    def encode(s):
        return np.asarray([bank.setdefault(raster.tile(c),len(bank)) for c in s],np.int32)
    ps,qs,gs,spans,offsets,ids=[],[],[],[],[0],[]
    seen=set()
    counts=Counter()
    for a in articles:
        title_blocked=canonical(a["title"]) in reserved.titles
        for p in a["paragraphs"]:
            counts["source_questions"]+=len(p["qas"])
            text=p["context"]
            norm=canonical(text)
            if title_blocked or reserved.overlaps(norm) or norm in seen:
                counts["questions_excluded_overlap"]+=len(p["qas"])
                continue
            try: passage=encode(text)
            except ValueError:
                counts["questions_excluded_missing_passage_glyph"]+=len(p["qas"])
                continue
            g=len(ps)
            for qa in p["qas"]:
                try: question=encode(qa["question"])
                except ValueError:
                    counts["questions_excluded_missing_question_glyph"]+=1
                    continue
                candidates=set()
                for answer in qa["answers"]:
                    value=answer["text"]
                    if not value: continue
                    start=text.find(value)
                    while start>=0:
                        candidates.add((start,start+len(value)))
                        start=text.find(value,start+1)
                if not candidates or not len(question):
                    counts["questions_excluded_unaligned"]+=1
                    continue
                assert all(0<=l<r<=len(passage) for l,r in candidates)
                qs.append(question)
                gs.append(g)
                spans.extend(sorted(candidates))
                offsets.append(len(spans))
                ids.append(qa["id"])
            if gs and gs[-1]==g:
                ps.append(passage)
                seen.add(norm)
    save_sequences(folder/"passages.npz",ps)
    save_sequences(folder/"questions.npz",qs)
    np.savez_compressed(folder/"evidence.npz",group=np.asarray(gs,np.int32),
                        offsets=np.asarray(offsets,np.int64),spans=np.asarray(spans,np.int32))
    np.save(folder/"pixels.npy",np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128))
    write_json(folder/"question_ids.json",ids)
    assert len(set(ids))==len(qs) and counts["source_questions"]==len(qs)+sum(v for k,v in counts.items() if k.startswith("questions_excluded"))
    return dict(passages=len(ps),questions=len(qs),aligned_intervals=len(spans),cells_in_unique_passages=sum(map(len,ps)),
                exclusions=dict(counts),files={p.name:sha(p) for p in folder.iterdir()})


if __name__=="__main__":
    if DEST.exists():raise FileExistsError(DEST)
    origin=load(RAW/"origin.json")
    assert origin["commit"]==COMMIT
    for f,h in origin["files"].items():assert sha(RAW/f)==h
    reserved=ReservedText()
    cmrc=ROOT/"data/raw/cmrc2018"
    for f in ("cmrc2018_train.json","cmrc2018_dev.json"):
        reserved.add(load(cmrc/f)["data"])
    dev=load(RAW/"DRCD_dev.json")["data"]
    test=load(RAW/"DRCD_test.json")["data"]
    train=load(RAW/"DRCD_training.json")["data"]
    with TemporaryDirectory(dir=DEST.parent) as tmp:
        folder=Path(tmp)/"corpus";folder.mkdir()
        result=dict(upstream=origin,program_sha256=sha(Path(__file__)),cmrc_files={p.name:sha(p) for p in cmrc.glob("*.json")})
        result["development"]=prepare_split(dev,reserved,folder/"development")
        reserved.add(dev)
        reserved.add(test)
        result["train"]=prepare_split(train,reserved,folder/"train")
        write_json(folder/"preparation.json",result)
        folder.rename(DEST)
    for s in ("train","development"):print(s,result[s]["questions"],result[s]["passages"])
