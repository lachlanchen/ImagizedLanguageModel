"""Separate questions and evidence before fitting the pictured stance writer."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts.prepare_paired_stance_images import ARCHIVE, DEST, normalized, sha, write_json
from scripts.train_paired_passage import ROOT

WORK = ROOT / "artifacts/paired_stance_writer_v1"
PUBLIC = ROOT / "references/evidence/paired_stance_writer_20261119"
SPLIT = PUBLIC / "split.json"
LOCATORS = ROOT / "data/visual_answer_language_v1/locators.jsonl"


def evidence_keys(text):
    clean = normalized(text)
    return {clean} | {clean[k:k+32] for k in range(len(clean)-31)}


def partition():
    prepared = json.loads((DEST / "preparation.json").read_text())
    assert sha(ARCHIVE) == prepared["archive_sha256"]
    rows = {}
    for name in ("development", "train"):
        for filename, digest in prepared["splits"][name]["files"].items():
            assert sha(DEST/name/filename) == digest
        rows[name] = json.loads((DEST/name/"rows.json").read_text())
    ancestors = set()
    with LOCATORS.open() as locators:
        for line in locators:
            member, index, *_ = json.loads(line)
            ancestors.add((member,index))

    groups = defaultdict(set)
    prior_rows = set()
    with ZipFile(ARCHIVE) as archive:
        for split, examples in rows.items():
            for member in sorted({r["member"] for r in examples}):
                by_line = {r["line"]:i for i,r in enumerate(examples) if r["member"]==member and i%2==0}
                with archive.open(member) as raw:
                    for line, content in enumerate(raw):
                        if line not in by_line:
                            continue
                        i = by_line[line]
                        record = json.loads(content)
                        key = (split,i//2)
                        groups["q",normalized(record["question"])].add(key)
                        for item in examples[i:i+2]:
                            assert item["member"] == member and item["line"] == line
                            paragraph = record["documents"][item["doc"]]["paragraphs"][item["paragraph"]]
                            for window in evidence_keys(paragraph[item["start"]:item["start"]+192]):
                                groups["p",window].add(key)
                        if (member,line) in ancestors:
                            prior_rows.add(key)
    count = len(rows["train"])//2
    parents = list(range(count))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    exclude = set()
    for keys in groups.values():
        train = sorted(i for split,i in keys if split=="train")
        if train:
            for i in train[1:]:
                parents[root(i)] = root(train[0])
            if any(split=="development" for split,_ in keys):
                exclude.update(train)
    components = defaultdict(list)
    for i in range(count):
        components[root(i)].append(i)
    def priority(group):
        inherited = any(("train",i) in prior_rows for i in group)
        return inherited, hashlib.sha256(f"stance-31613-{min(group)}".encode()).digest()
    fit, valid, removed = [], [], []
    for component in sorted(components.values(), key=priority):
        if exclude.intersection(component):
            removed.extend(component)
        elif len(valid)<144:
            valid.extend(component)
        else:
            fit.extend(component)
    assert len(set(fit+valid+removed)) == count and fit and valid
    membership = {i:label for label,ids in (("fit",fit),("validation",valid)) for i in ids}
    for keys in groups.values():
        labels = {membership[i] if split=="train" else "official" for split,i in keys if split!="train" or i in membership}
        assert len(labels)<=1
    return dict(fit=sorted(fit), validation=sorted(valid), removed=sorted(removed),
                official=list(range(len(rows["development"])//2)), train_components=len(components),
                max_component=max(map(len,components.values())),
                parent_large_corpus_records={label:sum((sp,i) in prior_rows for i in ids)
                    for label,sp,ids in (("fit","train",fit),("validation","train",valid),
                                          ("official","development",range(len(rows["development"])//2)))},
                data_sha256=sha(DEST/"preparation.json"), ancestor_sha256=sha(LOCATORS),
                source_sha256=sha(Path(__file__)))


if __name__ == "__main__":
    PUBLIC.mkdir(parents=True, exist_ok=True)
    if SPLIT.exists():
        raise FileExistsError(SPLIT)
    split = partition()
    write_json(SPLIT,split)
    print({k:len(v) if isinstance(v,list) else v for k,v in split.items()})
