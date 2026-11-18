"""Attach verified human question images to the independent pictured-answer corpus."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import numpy as np

from scripts.prepare_composed_chinese_answers import ARCHIVE
from scripts.prepare_pictured_entry_answers import Rasterizer,load,sha,write_json
from scripts.prepare_visual_answer_language import DEST


def prepare():
    output=DEST.parent/"visual_answer_questions_v1"
    if output.exists():raise FileExistsError(output)
    report=load(DEST/"receipt.json")
    assert sha(ARCHIVE)==report["archive_sha256"]
    for name in ("tiles.npy","lengths.npy","sequence.npy","locators.jsonl"):
        assert sha(DEST/name)==report["files"][name]
    original=np.load(DEST/"tiles.npy")
    shape_ids={bytes(t):i for i,t in enumerate(original) if i}
    shapes=list(map(bytes,original))
    lengths=np.load(DEST/"lengths.npy")
    ends=np.r_[0,lengths.cumsum()]
    seq=np.load(DEST/"sequence.npy",mmap_mode="r")
    renderer=Rasterizer()
    counters=Counter()
    handles=bytearray()
    rows=[]; extent=[0]
    with (DEST/"locators.jsonl").open() as loc, ZipFile(ARCHIVE) as z:
        def advance():
            line=loc.readline()
            return json.loads(line) if line else None
        item=advance()
        index=0
        for subset in ("search","zhidao"):
            member=f"raw/trainset/{subset}.train.json"
            with z.open(member) as src:
                for i,line in enumerate(src):
                    if item is None or item[:2]!=[member,i]:continue
                    r=json.loads(line)
                    assert item[3]==str(r["question_id"])
                    answer=r["answers"][item[2]].strip()
                    truth=original[seq[ends[index]:ends[index+1]]].tobytes()
                    assert truth==b"".join(renderer.tile(c) for c in answer)
                    q=r["question"].strip()
                    if not 2<=len(q)<=128:
                        counters["outside_extent"]+=1
                    else:
                        try:tiles=[renderer.tile(c) for c in q]
                        except ValueError:
                            counters["missing_shape"]+=1
                        else:
                            ids=[]
                            for tile in tiles:
                                if tile not in shape_ids:
                                    shape_ids[tile]=len(shapes)
                                    shapes.append(tile)
                                ids.append(shape_ids[tile])
                            encoded=np.array(ids,np.uint32)
                            handles.extend(encoded.tobytes())
                            extent.append(extent[-1]+len(ids))
                            rows.append(index)
                            counters[subset]+=1
                    item=advance();index+=1
        assert item is None and index==report["records"]
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage=Path(tmp)/"questions";stage.mkdir()
        np.save(stage/"tiles.npy",np.frombuffer(b"".join(shapes),np.uint8).reshape(-1,128))
        np.save(stage/"questions.npy",np.frombuffer(handles,np.uint32))
        np.save(stage/"offsets.npy",np.asarray(extent,np.int64))
        np.save(stage/"answer_rows.npy",np.asarray(rows,np.int32))
        # Check the complete join and its prefix-stable physical bitmap numbering.
        bank=np.load(stage/"tiles.npy",mmap_mode="r");q=np.load(stage/"questions.npy",mmap_mode="r")
        a=np.load(stage/"answer_rows.npy");off=np.load(stage/"offsets.npy")
        assert np.array_equal(bank[:len(original)],original)
        assert len(off)==len(a)+1 and off[-1]==len(q) and np.all(np.diff(off)>=2)
        assert np.all(np.diff(a)>0) and np.all(a<len(lengths))
        assert q.min()>0 and q.max()<len(bank)
        metadata=dict(parent_corpus_sha256=sha(DEST/"receipt.json"),records=len(a),
             rendered_question_positions=len(q),corresponding_answer_positions=int(lengths[a].sum()),
             raw_answer_streams_verified=index,distinct_nonpadding_tiles=len(bank)-1,counts=dict(counters),
             files={p.name:sha(p) for p in stage.iterdir()})
        write_json(stage/"receipt.json",metadata);stage.rename(output)
    print(metadata,flush=True)


if __name__=="__main__":prepare()
