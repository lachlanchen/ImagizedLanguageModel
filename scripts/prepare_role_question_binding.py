"""Prepare bilingual question pairs and verify the prospective classical source."""

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from scripts.prepare_book_participant_questions import HELD, matches
from scripts.prepare_book_span_distillation import LAST, PUBLIC as PARENT_PUBLIC, WORK as PARENT
from scripts.prepare_bilingual_novel_questions import WORK as NOVELS, PUBLIC as NOVEL_PUBLIC
from scripts.train_causal_glyph_writer import ROOT, record, sha
from scripts.train_typeface_writer import Face, FACES

PUBLIC = ROOT / "references/evidence/role_question_binding_20261124"
WORK = ROOT / "artifacts/role_question_binding_v1"
SOURCE = ROOT.parent / "ZhJpBook/books/shishuo-xinyu/markdown/wenyan.md"
SOURCE_SHA = "4dc9b756ddd83663260a2a4a785bb7b7ef4a108e59efa126d958afb886ab4c17"
WEIGHTS = ROOT / "models/book_span_distillation/reader.safetensors"


def read(path):
    return json.loads(path.read_text())


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def swap(page, a, b):
    assert a not in b and b not in a
    return re.sub(f"{re.escape(a)}|{re.escape(b)}", lambda m: b if m[0]==a else a, page)


def reserve():
    source=SOURCE.read_text()
    assert sha(SOURCE)==SOURCE_SHA
    rows=[]
    with (PUBLIC/"classical.tsv").open() as f:
        for i,r in enumerate(csv.DictReader(f,delimiter="\t")):
            p,a,b=r["page"],r["a"],r["b"]
            assert source.count(p)==1 and p.count(a) and p.count(b)
            q=[dict(zh=r["zh_"+k],en=r["en_"+k],answer=r[k]) for k in ("a","b")]
            assert all(n not in v["zh"] for v in q for n in (a,b))
            edited=swap(p,a,b)
            assert swap(edited,a,b)==p
            rows.append(dict(key=f"classical-{i+1:02d}",pages=[p,edited],qs=q,
                             source_line=source[:source.index(p)].count("\n")+1))
    assert len(rows)==28 and len({p for r in rows for p in r["pages"]})==56
    return rows


def local_support(page, answer):
    """Keep the complete source sentence and its immediate neighbors."""
    if page.count(answer)!=1:
        return None
    sentences=re.findall(r"[^。！？!?\n]+(?:[。！？!?]+[」』”’]*|$)|[。！？!?]+",page)
    if "".join(sentences)!=page:
        return None
    for i,s in enumerate(sentences):
        if answer in s:
            return "".join(sentences[max(0,i-1):i+2])
    return None


def prepare():
    WORK.mkdir(exist_ok=True)
    rows=reserve()
    face=Face(FACES[0])
    def supported(value):
        return all(c.isspace() or ord(c) in face.coverage for c in value)
    assert all(supported(v) for r in rows for v in [*r["pages"],*(q[lang] for q in r["qs"] for lang in ("zh","en"))])
    f=PARENT/"pairs.json"
    assert sha(f)==read(PARENT_PUBLIC/"corpus.json")["sha256"]
    held={r["page"] for r in read(LAST/"reserved.json")}
    classical=[p.rstrip("。，") for r in rows for p in r["pages"]]
    def check(p):
        assert p not in held and not any(s in p for s in classical)
    paired=[]
    for r in read(f):
        check(r["page"])
        if r["source"]=="novels":
            paired.append(dict(kind="natural",parts=[r["page"]],qs=r["qs"]))
    f=NOVELS/"train.jsonl"
    assert sha(f)==read(NOVEL_PUBLIC/"corpus.json")["splits"]["train"]["sha256"]
    singles=[]
    for line in f.read_text().splitlines():
        r=json.loads(line);p=r["page"]
        if p in held or any(w in q["id"] for w in HELD for q in r["questions"]):continue
        check(p)
        for q in r["questions"]:
            if not matches(q,p): continue
            a=q["refs"][0]
            s=local_support(p,a)
            if s and len(s)<=230 and a not in q["zh"]:
                singles.append((digest(p),s,dict(id=q["id"],zh=q["zh"],en=q["en"],answer=a)))
    singles.sort(key=lambda x:digest(x[2]["id"]))
    used=set()
    for i,(p,s,a) in enumerate(singles):
        if i in used:continue
        for j in range(i+1,len(singles)):
            if j in used:continue
            other,t,b=singles[j]
            if (p==other or a["answer"] in t+b["zh"] or b["answer"] in s+a["zh"]
               or any(a[k]==b[k] for k in ("en","zh"))):continue
            paired.append(dict(kind="combined",parts=[s,t],qs=[a,b],source_pages=[p,other]))
            used.update((i,j))
            break
    assert len(paired)>2300
    for r in paired:
        assert all(q["answer"] in "".join(r["parts"]) for q in r["qs"])
    record(WORK/"learn.json",paired)
    record(WORK/"classical.json",rows)
    record(PUBLIC/"corpus.json",dict(classical_sha=sha(WORK/"classical.json"),classical_source=SOURCE_SHA,
        classical_tsv=sha(PUBLIC/"classical.tsv"),classical_excerpts=len(rows),learn_sha=sha(WORK/"learn.json"),
        training_pairs=dict(Counter(r["kind"] for r in paired)),single_questions=len(singles),parent_weights=sha(WEIGHTS)))


if __name__=="__main__":
    prepare()
