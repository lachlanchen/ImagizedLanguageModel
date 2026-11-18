"""Retrieve a passage for the verified pictured question/answer pairs using question text alone."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from zipfile import ZipFile
import json
import numpy as np
from scripts.prepare_composed_chinese_answers import ARCHIVE, normalized
from scripts.prepare_visual_answer_language import DEST as ANSWERS, reserved
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json

QUESTIONS = ANSWERS.parent / "visual_answer_questions_v1"
DEST = ANSWERS.parent / "grounded_visual_answers_v1"


def retrieve(q, documents):
    text = normalized(q)
    pairs = {text[i:i+2] for i in range(len(text)-1)}
    chars = set(text)
    best, best_score = None, 0
    for d, doc in enumerate(documents[:4]):
        for k, paragraph in enumerate(doc["paragraphs"][:5]):
            for start in (0, 128):
                part = paragraph[start:start+192]
                if len(part)<8:
                    break
                s = normalized(part)
                matched = len(chars.intersection(s))
                score = (3*sum(pair in s for pair in pairs)+matched)/(12+len(s))**.26
                if score>best_score:
                    best, best_score = (part,d,k,start), score
                if start+192>=len(paragraph):
                    break
    return best


def prepare():
    if DEST.exists():
        raise FileExistsError(DEST)
    begin = monotonic()
    expected = load(ANSWERS/"receipt.json")
    assert sha(ARCHIVE)==expected["archive_sha256"]
    assert sha(ANSWERS/"locators.jsonl")==expected["files"]["locators.jsonl"]
    meta = load(QUESTIONS/"receipt.json")
    for name in ("tiles.npy","answer_rows.npy"):
        assert sha(QUESTIONS/name)==meta["files"][name]
    base = np.load(QUESTIONS/"tiles.npy")
    tiles = list(map(bytes,base))
    lookup = {g:i for i,g in enumerate(tiles) if i}
    rows = np.load(QUESTIONS/"answer_rows.npy")
    assert np.array_equal(rows,np.arange(len(rows)))
    raster = Rasterizer()
    counts = Counter()
    kept, extents = [], [0]
    with ZipFile(ARCHIVE) as z, TemporaryDirectory(dir=DEST.parent) as tmp:
        out = Path(tmp)/"corpus"
        out.mkdir()
        _, fragments, whole, _ = reserved(z)
        with (ANSWERS/"locators.jsonl").open() as loc, (out/"source_handles.bin").open("wb") as dest, (out/"windows.jsonl").open("w") as log:
            following = iter(loc)
            target = json.loads(next(following))
            n=0
            for subset in ("search","zhidao"):
                member=f"raw/trainset/{subset}.train.json"
                with z.open(member) as src:
                    for i,line in enumerate(src):
                        if target is None or target[:2]!=[member,i]:
                            continue
                        record=json.loads(line)
                        assert target[3]==str(record["question_id"])
                        row=n;n+=1
                        item=next(following,None)
                        target=json.loads(item) if item is not None else None
                        best=retrieve(record["question"],record["documents"])
                        if best is None:
                            counts["no_window"]+=1;continue
                        p,d,k,offset=best
                        key=normalized(p)
                        if key in whole or any(key[j:j+16] in fragments for j in range(len(key)-15)):
                            counts["held_out_overlap"]+=1;continue
                        try:
                            images=[raster.tile(c) for c in p]
                        except ValueError:
                            counts["unrenderable"]+=1;continue
                        ids=[]
                        for g in images:
                            if g not in lookup:
                                lookup[g]=len(tiles);tiles.append(g)
                            ids.append(lookup[g])
                        dest.write(np.asarray(ids,np.uint32).tobytes())
                        kept.append(row);extents.append(extents[-1]+len(ids))
                        log.write(json.dumps([row,d,k,offset,len(p)])+'\n')
                        counts[subset]+=1
                print(subset,n,len(kept),flush=True)
        assert n==len(rows) and target is None
        np.save(out/"tiles.npy",np.frombuffer(b"".join(tiles),np.uint8).reshape(-1,128))
        np.save(out/"offsets.npy",np.asarray(extents,np.int64))
        np.save(out/"rows.npy",np.asarray(kept,np.int32))
        assert (out/"source_handles.bin").stat().st_size==extents[-1]*4
        record=dict(records=len(kept),screened=n,extent=extents[-1],counts=dict(counts),
                    seconds=monotonic()-begin,source_sha256=sha(Path(__file__)),
                    questions_sha256=sha(QUESTIONS/"receipt.json"),
                    answers_sha256=sha(ANSWERS/"receipt.json"),
                    files={p.name:sha(p) for p in out.iterdir()})
        write_json(out/"receipt.json",record)
        out.rename(DEST)
    print(record,flush=True)


if __name__=="__main__":
    prepare()
