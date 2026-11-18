"""Verify pictured DuReader questions, retrieved excerpts and nonextractive answer classes."""
from collections import Counter,defaultdict
import json
from zipfile import ZipFile

import numpy as np
from scripts.prepare_composed_chinese_answers import ARCHIVE,DEST
from scripts.prepare_pictured_entry_answers import Rasterizer,load,sha,write_json
from scripts.train_paired_passage import ROOT,unpack_sequences


def contained(stream,needle):
    return any(stream[i:i+len(needle)]==needle for i in range(0,len(stream)-len(needle)+1,128))


def main():
    report=load(DEST/"preparation.json")
    assert sha(ARCHIVE)==report["archive_sha256"]
    raster=Rasterizer()
    def encoded(text):
        return b"".join(raster.tile(c) for c in text)
    audit={}
    retained={}
    with ZipFile(ARCHIVE) as archive:
        for split in ("development","train"):
            folder=DEST/split
            for name,expected in report[split]["files"].items():
                assert sha(folder/name)==expected
            tiles=np.load(folder/"pixels.npy")
            ps,qs,ys=[unpack_sequences(folder/(name+"s.npz")) for name in ("passage","question","answer")]
            rows=load(folder/"sources.json")
            pending=defaultdict(dict)
            for i,row in enumerate(rows):
                assert row["line"] not in pending[row["member"]]
                pending[row["member"]][row["line"]]=(i,row)
            counts=Counter()
            source_set=set()
            question_set=set()
            answer_set=set()
            for member,by_line in pending.items():
                with archive.open(member) as f:
                    for l,line in enumerate(f):
                        if l not in by_line:
                            continue
                        i,meta=by_line.pop(l)
                        raw=json.loads(line)
                        assert str(raw["question_id"])==meta["question_id"]
                        p=tiles[ps[i]].tobytes()
                        q=tiles[qs[i]].tobytes()
                        y=tiles[ys[i]].tobytes()
                        part=raw["documents"][meta["doc"]]["paragraphs"][meta["paragraph"]]
                        assert encoded(part[meta["start"]:meta["start"]+len(ps[i])])==p and encoded(raw["question"].strip())==q
                        references=[]
                        for candidate in raw["answers"]:
                            a=candidate.strip()
                            if len(a)!=len(ys[i]):
                                continue
                            try:
                                same=encoded(a)==y
                            except ValueError:
                                continue
                            if same:references.append(a)
                        assert references
                        visible=set(map(bytes,tiles[np.concatenate((ps[i],qs[i]))]))
                        if contained(p,y) or contained(q,y):
                            kind="single_span"
                        elif all(bytes(t) in visible for t in tiles[ys[i]]):
                            kind="distributed_ink"
                        else:
                            assert any(c.isalnum() and raster.tile(c) not in visible for c in references[0])
                            kind="absent_content_ink"
                        assert kind==meta["kind"]
                        counts[kind]+=1
                        source_set.add(p)
                        question_set.add(q)
                        answer_set.add(y)
                        if not by_line:break
                assert not by_line
            assert len(rows)==len(ps)==len(qs)==len(ys)==sum(counts.values())
            retained[split]=(source_set,question_set,answer_set)
            audit[split]=dict(records=len(rows),recount=dict(counts),exact_one_input_span_ceiling=counts["single_span"],
                              arbitrary_input_cell_copy_ceiling=len(rows)-counts["absent_content_ink"])
    overlap=[len(a&b) for a,b in zip(retained["development"],retained["train"],strict=True)]
    assert overlap[0]==overlap[1]==0
    audit["shared_distinct_images"]=dict(passage=overlap[0],question=overlap[1],answer=overlap[2])
    receipt=dict(preparation=report,audit=audit)
    path=ROOT/"references/evidence/composed_chinese_answers_preparation_20261119.json"
    if path.exists():assert load(path)==receipt
    else:write_json(path,receipt)
    print(audit)


if __name__=="__main__":
    main()
