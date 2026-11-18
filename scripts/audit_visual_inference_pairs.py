"""Verify OCNLI rasterization and freeze disjoint premise families for a paired writing comparison."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.prepare_visual_inference_pairs import DEST, RAW, LABELS, RESPONSES, PUBLIC as SOURCE
from scripts.prepare_paired_stance_images import Rasterizer, normalized, sha, write_json
from scripts.train_paired_passage import ROOT, unpack_sequences

PUBLIC=ROOT/"references/evidence/visual_inference_writer_20261119"
AUDIT=PUBLIC/"cohort.json"
PRIVATE=ROOT/"data/ocnli_images_v1/writer_split.npz"


def select(rows):
    groups=defaultdict(dict)
    for i,r in enumerate(rows):
        groups[r["premise"]].setdefault(r["label"],i)
    return {k:[v[l] for l in LABELS] for k,v in groups.items() if set(v)==set(LABELS)}


def run():
    if AUDIT.exists() or PRIVATE.exists():
        raise FileExistsError("Existing inference cohort")
    r=Rasterizer()
    cache={}
    def physical(seq):
        return b"".join(r.tile(c) for c in seq)
    arrays={}
    rawdata={}
    public={}
    for name,member in (("development","dev.json"),("train","train.50k.json")):
        folder=DEST/name
        meta=json.loads((folder/"rows.json").read_text())
        original=[json.loads(line) for line in (RAW/member).read_text().splitlines()]
        pixels=np.load(folder/"pixels.npy")
        p,h,y=[unpack_sequences(folder/(f+".npz")) for f in ("premises","claims","responses")]
        assert len(meta)==len(p)==len(h)==len(y)
        content=[]
        for i,m in enumerate(meta):
            sample=original[m["line"]]
            assert sample["label"]==m["label"] and m["genre"]==sample["genre"]
            pr,hy=sample["sentence1"],sample["sentence2"]
            pk,hk=normalized(pr),normalized(hy)
            assert hashlib.sha256(pk.encode()).hexdigest()==m["premise"]
            for seq,s in ((p[i],pr),(h[i],hy),(y[i],RESPONSES[LABELS.index(m["label"])])):
                assert pixels[seq].tobytes()==physical(s)
            content.append((pk,hk))
        rawdata[name]=content
        groups=select(meta)
        if name=="development":
            official_groups=sorted(groups)
            arrays["official_triples"]=np.array([groups[k] for k in official_groups])
            arrays["official"]=np.arange(len(meta))
            public["official_triples"]=len(groups)
            continue
        cohort=sorted(groups,key=lambda k:hashlib.sha256(("ocnli-32719"+k).encode()).digest())[:512]
        valset=set(cohort)
        val=[i for i,m in enumerate(meta) if m["premise"] in valset]
        refs={content[i][1] for i in val}
        train=[i for i,m in enumerate(meta) if m["premise"] not in valset and content[i][1] not in refs]
        assert len(set(valset))==512
        arrays.update(fit=np.asarray(train),validation=np.asarray(val),
                      validation_triples=np.array([groups[k] for k in cohort]))
        public.update(fit=len(train),validation=len(val),removed_claim_overlap=len(meta)-len(train)-len(val),
                      validation_triples=len(cohort))
    for side in (0,1):
        d={s[side] for s in rawdata["development"]}
        fit={rawdata["train"][i][side] for i in arrays["fit"]}
        val={rawdata["train"][i][side] for i in arrays["validation"]}
        assert not (d & fit or d & val or fit & val)
    np.savez_compressed(PRIVATE,**arrays)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    write_json(AUDIT,dict(**public,official=len(arrays["official"]),files_sha256=sha(PRIVATE),
               preparation_sha256=sha(SOURCE),source_sha256=sha(Path(__file__)),
               outcome="Every retained premise, claim and target image matched the upstream record. Normalized premises and claims have zero same-field cross-partition overlaps."))
    print(public)


if __name__=="__main__":
    run()
