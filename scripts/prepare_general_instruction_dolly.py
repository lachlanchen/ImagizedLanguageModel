#!/usr/bin/env python3
"""Pin broad public instructions and group exact shared sources before splitting."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import requests

REVISION="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
REPOSITORY="databricks/databricks-dolly-15k"
EXPECTED_SHA256={
    "databricks-dolly-15k.jsonl":"2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
    "README.md":"db06c60090a11b20e79d1933b2f316f781605f07761ff2ae2795d6523e3e960b",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(text):
    return " ".join(text.split())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw",default="data/raw/databricks_dolly_15k")
    p.add_argument("--out",default="data/general_instructions/dolly15k")
    a=p.parse_args();raw,out=Path(a.raw),Path(a.out)
    if (out/"preparation_report.json").exists():
        raise FileExistsError("completed instruction partition exists")
    raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    files={}
    for name in ("databricks-dolly-15k.jsonl","README.md"):
        path=raw/name
        url=f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{name}"
        if not path.exists():
            response=requests.get(url,timeout=45,headers={"User-Agent":"ILM-public-data-research/1.0"})
            response.raise_for_status();temporary=path.with_suffix(path.suffix+".partial")
            temporary.write_bytes(response.content);temporary.replace(path)
        if digest(path)!=EXPECTED_SHA256[name]:
            raise ValueError("raw instruction source differs from pinned author bytes")
        files[name]=dict(url=url,sha256=digest(path),bytes=path.stat().st_size)
    source=[json.loads(s) for s in (raw/"databricks-dolly-15k.jsonl").read_text().splitlines()]
    parents=list(range(len(source)))

    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i

    def union(i,j):
        i,j=root(i),root(j)
        if i!=j:parents[max(i,j)]=min(i,j)

    seen={};records=[]
    for i,row in enumerate(source):
        fields={name:row[name].replace("\r\n","\n").replace("\r","\n").strip()
                for name in ("instruction","context","response")}
        if not fields["instruction"] or not fields["response"]:
            raise ValueError("empty instruction or response; inspect source rather than silently dropping")
        for field in ("instruction","context"):
            value=flat(fields[field]).casefold()
            if not value:continue
            key=(field,value)
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        prefix=("Context: "+fields["context"]+"\n") if fields["context"] else ""
        prompt=prefix+"Question: "+fields["instruction"]+"\nAnswer:"
        records.append(dict(id=f"dolly_{i:05d}",source_row=i,category=row["category"],
            instruction=fields["instruction"],context=fields["context"],answer=fields["response"],
            prompt=prompt,prompt_flat=flat(prompt),answer_flat=flat(fields["response"])))
    components={}
    for i in range(len(source)):components.setdefault(root(i),[]).append(i)
    splits={name:[] for name in ("train","development","sealed_test")}
    for ids in components.values():
        key="\n".join(records[i]["id"] for i in ids)
        value=int(hashlib.sha256(("ilm-dolly-20266120\n"+key).encode()).hexdigest(),16)%100
        split="train" if value<80 else "development" if value<90 else "sealed_test"
        group=hashlib.sha256(key.encode()).hexdigest()
        for i in ids:splits[split].append(dict(records[i],group=group))
    overlaps={}
    names=list(splits)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            for field in ("instruction","context"):
                keys=lambda split:{flat(r[field]).casefold() for r in splits[split] if flat(r[field])}
                overlaps[f"{left}__{right}__{field}"]=len(keys(left)&keys(right))
    if any(overlaps.values()):raise AssertionError("shared instruction/context crossed partitions")
    bounds={}
    for prompt_max,answer_max in ((480,128),(480,256),(960,512),(2048,1024)):
        bounds[f"prompt{prompt_max}_answer{answer_max}"]={
            split:dict(examples=sum(len(r["prompt_flat"])<=prompt_max and len(r["answer_flat"])<=answer_max for r in rows),
                       categories=dict(Counter(r["category"] for r in rows if len(r["prompt_flat"])<=prompt_max and len(r["answer_flat"])<=answer_max)))
            for split,rows in splits.items()}
    output_hashes={}
    for split,rows in splits.items():
        path=out/(split+".jsonl")
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sorted(rows,key=lambda r:r["id"])))
        output_hashes[path.name]=digest(path)
    attribution=dict(source_repository=REPOSITORY,revision=REVISION,license="CC-BY-SA-3.0",
        source_card="https://huggingface.co/datasets/"+REPOSITORY,
        transformations="line-ending/outer-space normalization; derived flat views; exact-source grouped partitions",
        responses_independently_verified=False,original_pretraining_overlap_unknown=True)
    (out/"attribution.json").write_text(json.dumps(attribution,indent=2)+"\n")
    report=dict(**attribution,raw_files=files,examples=len(source),groups=len(components),
        split_counts={k:len(v) for k,v in splits.items()},exact_source_overlap=overlaps,
        categories={k:dict(Counter(r["category"] for r in rows)) for k,rows in splits.items()},
        flat_lengths={key:dict(median=float(np.median([len(r[key]) for r in records])),
                              maximum=max(len(r[key]) for r in records)) for key in ("prompt_flat","answer_flat")},
        full_source_retention_by_bounds=bounds,output_sha256=output_hashes,
        images_rendered=False,model_calls=0,model_quality_claim=False,V61_data_changed=False,
        grouping_limitation="exact normalized instructions/contexts only; no article or semantic-near-duplicate guarantee")
    (out/"preparation_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("examples","groups","split_counts","flat_lengths","full_source_retention_by_bounds")},indent=2),flush=True)


if __name__=="__main__":main()
