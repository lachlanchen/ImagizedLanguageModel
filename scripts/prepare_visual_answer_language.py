"""Build a local ordered and order-shuffled physical glyph-history index from real Chinese answers."""
from collections import Counter,defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZipFile
import json

import numpy as np

from scripts.prepare_composed_chinese_answers import ARCHIVE, normalized
from scripts.prepare_pictured_entry_answers import Rasterizer, sha, load, write_json
from scripts.train_composed_answer_writer import DATA, ROOT

DEST=ROOT/"data/visual_answer_language_v1"
SEED=19413
DEPTH=4


def reserved(archive):
    """Remove original fitted and held-out records; block literal held-out content leakage."""
    train=load(DATA/"train/sources.json")
    dev=load(DATA/"development/sources.json")
    excluded={r["question_id"] for r in train+dev}
    pages=defaultdict(dict)
    for m in dev:
        pages[m["member"]][m["line"]]=m
    fragments,whole,questions=set(),set(),set()
    for member,keys in pages.items():
        with archive.open(member) as source:
            for idx,line in enumerate(source):
                if idx not in keys:continue
                m=keys[idx];r=json.loads(line)
                q=normalized(r["question"])
                questions.add(q)
                # Screen the whole selected paragraph, covering the displayed window.
                passage=r["documents"][m["doc"]]["paragraphs"][m["paragraph"]]
                for s in (passage,r["question"],*r["answers"]):
                    key=normalized(s)
                    whole.add(key)
                    fragments.update(key[i:i+16] for i in range(len(key)-15))
    return excluded,fragments,whole,questions


def build_index(path,seq,lengths,base,shuffled):
    seq=seq.copy()
    boundaries=np.r_[0,np.cumsum(lengths)]
    if shuffled:
        rng=np.random.default_rng(SEED)
        for l,r in zip(boundaries[:-1],boundaries[1:]):
            rng.shuffle(seq[l:r])
    keys=np.zeros(len(seq),np.int64)
    # No n-gram crosses an answer boundary.
    age=np.arange(len(seq))-np.repeat(boundaries[:-1],lengths)
    for offset in range(1,DEPTH+1):
        prev=np.roll(seq,offset)
        keys*=base
        keys+=np.where(age>=offset,prev,0)
    order=np.argsort(keys,kind="stable")
    np.save(path/"keys.npy",keys[order])
    np.save(path/"next.npy",seq[order])


def prepare():
    if DEST.exists():raise FileExistsError(DEST)
    t=monotonic()
    expected=load(DATA/"preparation.json")["archive_sha256"]
    assert sha(ARCHIVE)==expected
    with ZipFile(ARCHIVE) as archive, TemporaryDirectory(dir=DEST.parent) as tmp:
        out=Path(tmp)/"corpus";out.mkdir()
        raster=Rasterizer()
        excluded,fragments,whole,questions=reserved(archive)
        ids={}
        seen=set()
        sequences=bytearray()
        counts=Counter()
        lengths=[]
        with (out/"locators.jsonl").open("w") as locators:
            for subset in ("search","zhidao"):
                member=f"raw/trainset/{subset}.train.json"
                with archive.open(member) as f:
                    for idx,line in enumerate(f):
                        raw=json.loads(line)
                        counts["screens"]+=1
                        if str(raw["question_id"]) in excluded or normalized(raw["question"]) in questions:
                            continue
                        for j,text in enumerate(raw["answers"]):
                            text=text.strip()
                            if not 6<=len(text)<=96:continue
                            key=normalized(text)
                            if len(key)<5 or key in seen or key in whole:continue
                            if any(key[i:i+16] in fragments for i in range(len(key)-15)):continue
                            try:tiles=[raster.tile(c) for c in text]
                            except ValueError:continue
                            indexes=[ids.setdefault(b,len(ids)+1) for b in tiles]
                            sequences.extend(np.asarray(indexes,dtype=np.uint32).tobytes())
                            lengths.append(len(tiles));seen.add(key);counts[subset]+=1
                            locators.write(json.dumps([member,idx,j,str(raw["question_id"])])+"\n")
                            break
        seq=np.frombuffer(sequences,np.uint32)
        size=len(ids)+1
        assert size**DEPTH < 2**63 and 0<seq.min()<=seq.max()<size
        np.save(out/"tiles.npy",np.frombuffer(bytes(128)+b"".join(ids),np.uint8).reshape(-1,128))
        np.save(out/"lengths.npy",np.array(lengths,np.int32))
        np.save(out/"sequence.npy",seq)
        for arm in ("ordered","shuffled"):
            path=out/arm;path.mkdir()
            build_index(path,seq,lengths,size,arm=="shuffled")
        (out/"License.docx").write_bytes(archive.read("raw/License.docx"))
        receipt=dict(records=len(lengths),positions=len(seq),distinct_bitmaps=len(ids),counts=dict(counts),
                     archive_sha256=expected,preparation_seconds=monotonic()-t,
                     files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()})
        write_json(out/"receipt.json",receipt)
        out.rename(DEST)
    print(receipt,flush=True)


if __name__=="__main__":prepare()
