"""Measure how often opposed human answers to the same real question can be grounded in different printed passages."""
from collections import Counter
from hashlib import sha256
from zipfile import ZipFile
import json

from scripts.prepare_composed_chinese_answers import ARCHIVE, normalized
from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_english_request_writer import ROOT

WORK=ROOT/"artifacts/natural_stance_evidence_v1"
PUBLIC=ROOT/"references/evidence/english_request_writer_20261119"


def survey():
    WORK.mkdir()
    expected=json.loads((ROOT/"data/composed_chinese_answers_v1/preparation.json").read_text())["archive_sha256"]
    if sha(ARCHIVE)!=expected:
        raise ValueError("DuReader source archive changed")
    report={}
    heldout=set()
    with ZipFile(ARCHIVE) as z, (WORK/"candidates.jsonl").open("x") as private:
        for split,suffix in (("devset","dev"),("trainset","train")):
            counts=Counter()
            seen=set()
            for subset in ("search","zhidao"):
                member=f"raw/{split}/{subset}.{suffix}.json"
                with z.open(member) as stream:
                    for line,b in enumerate(stream):
                        r=json.loads(b)
                        counts["examined"]+=1
                        if r["question_type"]!="YES_NO":continue
                        counts["yes_no"]+=1
                        key=normalized(r["question"])
                        if not key or key in seen or (split=="trainset" and key in heldout):continue
                        seen.add(key)
                        ys,labels=r["answers"],r["yesno_answers"]
                        if len(ys)!=len(labels):
                            counts["unaligned_labels"]+=1
                            continue
                        if not {"Yes","No"}<=set(labels):continue
                        counts["opposed_annotations"]+=1
                        # A matched document indicator is coarse. Also require literal answer support.
                        sides={"Yes":[],"No":[]}
                        evidence=[(d,k,normalized(p)) for d,doc in enumerate(r["documents"]) if doc["is_selected"]
                                  for k,p in enumerate(doc["paragraphs"]) if p.strip()]
                        for y,(answer,label) in enumerate(zip(ys,labels,strict=True)):
                            if label not in sides:continue
                            ink=normalized(answer)
                            if not 8<=len(ink)<=100:continue
                            for d,k,p in evidence:
                                if ink in p:sides[label].append((d,k,y))
                        if any(sides.values()):counts["at_least_one_grounded_side"]+=1
                        pair=next(((a,b) for a in sides["Yes"] for b in sides["No"] if a[:2]!=b[:2]),None)
                        if pair:
                            counts["both_grounded_in_distinct_paragraphs"]+=1
                            counts["both_grounded_in_distinct_documents"]+=int(any(a[0]!=c[0] for a in sides["Yes"] for c in sides["No"]))
                            private.write(json.dumps(dict(member=member,line=line,question_hash=sha256(key.encode()).hexdigest(),
                                                        yes=pair[0],no=pair[1]))+"\n")
            if split=="devset":heldout.update(seen)
            report[split]=dict(counts)
    write_json(PUBLIC/"natural_stance_feasibility.json",dict(source_sha256=expected,split_counts=report,program_sha256=sha(__file__),
               candidate_sha256=sha(WORK/"candidates.jsonl")))
    print(report)


if __name__=="__main__":
    survey()
