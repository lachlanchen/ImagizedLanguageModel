"""Reconstruct each opposite-stance pair from the independent raw member and score blind baselines."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json

import numpy as np

from scripts.prepare_paired_stance_images import ARCHIVE, DEST, PUBLIC, Rasterizer, normalized, sha, write_json
from scripts.train_paired_passage import unpack_sequences

# These markers and the instruction are specified before measuring the official development split.
INSTRUCTION="这段材料对下面的问题持什么立场？写明立场，再引一句材料原话。"
NEGATIVE=("不能","不是","不可以","不可能","没有","不会","不行","不对","不建议","不需要","不一定")
POSITIVE=("可以","能够","当然","没问题","肯定","没错","确实","是的","能的")


def audit():
    prep=json.loads((DEST/"preparation.json").read_text())
    raster=Rasterizer()
    def bits(text):
        return b"".join(raster.tile(c) for c in text)
    neg=[bits(v) for v in NEGATIVE];pos=[bits(v) for v in POSITIVE]
    def present(im,needle):
        return any(im[k:k+len(needle)]==needle for k in range(0,len(im)-len(needle)+1,128))
    all_stats={}
    reserved=set()
    for split in ("development","train"):
        folder=DEST/split
        for f,digest in prep["splits"][split]["files"].items():
            if sha(folder/f)!=digest:raise ValueError("Changed stance input")
        rows=json.loads((folder/"rows.json").read_text())
        bank=np.load(folder/"pixels.npy")
        q,p,y=[[bank[s].tobytes() for s in unpack_sequences(folder/(k+".npz"))] for k in ("questions","passages","answers")]
        count=Counter()
        correct=[]
        with ZipFile(ARCHIVE) as z:
            for member in sorted({r["member"] for r in rows}):
                ids={r["line"]:i for i,r in enumerate(rows) if r["member"]==member and i%2==0}
                with z.open(member) as raw:
                    for line,b in enumerate(raw):
                        if line not in ids:continue
                        start=ids[line];r=json.loads(b)
                        assert q[start]==q[start+1] and p[start]!=p[start+1] and y[start]!=y[start+1]
                        if split=="train":assert normalized(r["question"]) not in reserved
                        else:reserved.add(normalized(r["question"]))
                        pair=[]
                        for i in (start,start+1):
                            m=rows[i];label="Yes" if i%2==0 else "No"
                            body=r["answers"][m["annotation"]].strip()
                            target=("肯定。" if i%2==0 else "否定。")+body
                            assert r["yesno_answers"][m["annotation"]]==m["side"]==label
                            doc=r["documents"][m["doc"]];paragraph=doc["paragraphs"][m["paragraph"]]
                            src=paragraph[m["start"]:m["start"]+192]
                            assert doc["is_selected"] and body in src
                            assert bits(r["question"].strip())==q[i] and bits(src)==p[i] and bits(target)==y[i]
                            assert not present(p[start+1-(i%2)],bits(body))
                            for k in range(0,len(q[i]),128):assert len(q[i][k:k+128])==128
                            signed=sum(present(p[i],n) for n in pos)-sum(present(p[i],n) for n in neg)
                            pair.append((signed>=0)==(label=="Yes"))
                            count["absent_stance_content_with_instruction"]+=int(any(
                                not present(bits(INSTRUCTION)+q[i]+p[i],bits(c)) for c in target[:2]))
                            count["responses"]+=1
                        count["physical_marker_correct"]+=sum(pair)
                        count["physical_marker_both"]+=all(pair)
                        count["pairs"]+=1
        assert count["responses"]==len(rows) and count["pairs"]*2==len(rows)
        all_stats[split]=dict(count)
    write_json(PUBLIC/"natural_stance_audit.json",dict(splits=all_stats, instruction=INSTRUCTION,
               positive=POSITIVE,negative=NEGATIVE,exact_dual_ceiling_question_only=0,program_sha256=sha(__file__)))
    print(all_stats)


if __name__=="__main__":audit()
