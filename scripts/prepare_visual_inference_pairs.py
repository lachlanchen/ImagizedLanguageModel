"""Prepare original Chinese premise–hypothesis pairs as two losslessly packed glyph streams."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import requests

from scripts.prepare_composed_chinese_answers import normalized
from scripts.prepare_pictured_entry_answers import Rasterizer, save_sequences, sha, write_json
from scripts.train_paired_passage import ROOT

REVISION="b53efdee17257a5c33993cf6fcf8ffff0497ea0e"
BLOBS={"README.md":"83edac210ab8f8e092136130e8f526dbcfd88105",
       "data/ocnli/dev.json":"562b7c18c7ec7cdca0650df37bcd88e07bb2ee2b",
       "data/ocnli/train.50k.json":"13e1456ee3bf6c78a3a3f6edb887c8b919d324a4"}
LABELS=("entailment","contradiction","neutral")
RESPONSES=("材料支持这个说法。","这个说法与材料矛盾。","仅凭材料无法确定。")
RAW=ROOT/"data/raw/ocnli"
DEST=ROOT/"data/ocnli_images_v1"
PUBLIC=ROOT/"references/evidence/paired_stance_writer_20261119/ocnli_preparation.json"


def acquire(name,expected):
    path=RAW/Path(name).name
    if path.exists():
        raw=path.read_bytes()
    else:
        response=requests.get(f"https://raw.githubusercontent.com/CLUEbenchmark/OCNLI/{REVISION}/{name}",timeout=90)
        response.raise_for_status()
        raw=response.content
    if hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()!=expected:
        raise ValueError(f"Unexpected upstream blob {name}")
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(raw)
    return path


def prepare():
    if DEST.exists() or PUBLIC.exists():
        raise FileExistsError("Prepared visual NLI corpus already exists")
    RAW.mkdir(exist_ok=True,parents=True)
    sources={n:acquire(n,h) for n,h in BLOBS.items()}
    dev_p,dev_h=set(),set()
    renderer=Rasterizer()
    splits={}
    with TemporaryDirectory(dir=DEST.parent) as tmp:
        root=Path(tmp)/"dataset"
        root.mkdir()
        (root/"README.upstream.md").write_bytes(sources["README.md"].read_bytes())
        for split,name in (("development","dev.json"),("train","train.50k.json")):
            folder=root/split
            folder.mkdir()
            rows=[json.loads(line) for line in (RAW/name).read_text().splitlines()]
            by_pair=defaultdict(set)
            for row in rows:
                p,h=normalized(row["sentence1"]),normalized(row["sentence2"])
                if split=="development":
                    dev_p.add(p);dev_h.add(h)
                if row["label"] in LABELS:
                    by_pair[p,h].add(row["label"])
            seen=set();bank={};premises=[];hypotheses=[];answers=[];meta=[];skips=Counter()
            intent=defaultdict(set)
            def encode(text):
                return [bank.setdefault(renderer.tile(c),len(bank)) for c in text]
            for i,row in enumerate(rows):
                p,h,label=row["sentence1"],row["sentence2"],row["label"]
                pk,hk=normalized(p),normalized(h)
                pair=(pk,hk)
                if label not in LABELS:
                    skips["no_consensus"]+=1
                elif not pk or not hk or len(p)>192 or len(h)>96:
                    skips["empty_or_long"]+=1
                elif split=="train" and (pk in dev_p or hk in dev_h):
                    skips["development_overlap"]+=1
                elif len(by_pair[pair])!=1:
                    skips["conflicting_pair"]+=1
                elif pair in seen:
                    skips["duplicate_pair"]+=1
                else:
                    try:
                        ps,hs,ys=encode(p),encode(h),encode(RESPONSES[LABELS.index(label)])
                    except ValueError:
                        skips["font_coverage"]+=1
                        continue
                    seen.add(pair);intent[pk].add(label)
                    premises.append(ps);hypotheses.append(hs);answers.append(ys)
                    meta.append(dict(line=i,premise=hashlib.sha256(pk.encode()).hexdigest(),
                                     label=label,genre=row["genre"]))
            for key,seq in (("premises",premises),("claims",hypotheses),("responses",answers)):
                save_sequences(folder/(key+".npz"),seq)
            np.save(folder/"pixels.npy",np.frombuffer(b"".join(bank),np.uint8).reshape(-1,128))
            write_json(folder/"rows.json",meta)
            assert len(meta)+sum(skips.values())==len(rows)
            splits[split]=dict(raw=len(rows),kept=len(meta),excluded=dict(skips),
                labels=dict(Counter(m["label"] for m in meta)),premises=len(intent),
                premises_with_three_labels=sum(len(v)==3 for v in intent.values()),
                premise_cells=sum(map(len,premises)),claim_cells=sum(map(len,hypotheses)),
                files={p.name:sha(p) for p in folder.iterdir()})
        doc=dict(upstream="CLUEbenchmark/OCNLI",revision=REVISION,git_blobs=BLOBS,
                 source_sha256={n:sha(p) for n,p in sources.items()},splits=splits,
                 responses=list(RESPONSES),program_sha256=sha(Path(__file__)))
        write_json(root/"preparation.json",doc)
        root.rename(DEST)
    write_json(PUBLIC,doc)
    print({k:(v["kept"],v["premises_with_three_labels"]) for k,v in splits.items()})


if __name__=="__main__":
    prepare()
