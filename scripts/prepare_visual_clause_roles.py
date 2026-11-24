"""Prepare visual arc supervision and reserve independent opposite-order clauses."""

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import csv
import os
import re

from scripts.prepare_asked_gloss import ROOT, PRIOR, HELD, SOURCES, read, dump, sha
from scripts.prepare_pronoun_writer import WORK as PERSON, window
from scripts.train_asked_gloss import rows_and_split
from scripts.train_typeface_writer import Face, FACES

WORK = ROOT / "artifacts/visual_clause_roles_v1"
PUBLIC = ROOT / "references/evidence/visual_clause_roles_v1"


def extract(path, part):
    for block in path.read_text().split("\n\n"):
        sid = re.search(r"(?m)^# sent_id = (.+)$",block)
        if not sid:
            continue
        book = sid[1].split("_")[0]
        words = [line.split("\t") for line in block.splitlines() if line.split("\t")[0].isdigit()]
        page = "".join(w[1] for w in words)
        if not 3<=len(page)<=32 or not all("\u3400" <= c <= "\u9fff" for c in page):
            continue
        verbs = [w for w in words if w[3]=="VERB" and w[7]=="root"]
        if len(verbs)!=1:
            continue
        verb=verbs[0]
        objs=[w for w in words if w[6]==verb[0] and w[7]=="obj"]
        if len(objs)!=1 or any(len(w[1])!=1 for w in (verb,objs[0])):
            continue
        pos={};i=0
        for w in words:
            pos[w[0]]=i
            i+=len(w[1])
        split="reserved_work" if book in HELD else ("train" if part=="train" else "reserved_section")
        yield dict(book=book,id=sid[1],split=split,page=page,focus=[pos[verb[0]],pos[objs[0][0]]])


def prepare():
    WORK.mkdir(exist_ok=True)
    face=Face(FACES[0]); bitmap={}
    @lru_cache(None)
    def cell(c):
        return bitmap.setdefault(face.cell(c),len(bitmap))
    def key(text):
        return tuple(map(cell,text))
    previous,_=rows_and_split()
    old_pages={key(r["page"]) for r in previous if r["split"]=="train"}
    old_tests=read(ROOT/"references/evidence/everyday_gloss_v1/positions.json")
    previous_test={key(r["page"]) for r in old_tests}
    writer_reg=read(ROOT/"references/evidence/visual_word_senses_20261124/split.json")
    assert sha(PRIOR/"train.json")==writer_reg["data"]["train"]["meta"]
    assert sha(PERSON/"train.json")==read(ROOT/"references/evidence/pronoun_writer_v1/split.json")["train_sha256"]
    trained_windows={key(left.rjust(8)+c+right.ljust(8)) for _,_,_,left,c,right,_ in read(PRIOR/"train.json")}
    trained_windows.update(key(r[0]) for r in read(PERSON/"train.json"))
    groups=defaultdict(dict)
    for part,h in SOURCES.items():
        path=ROOT/f"data/raw/ud_classical_kyoto_r216/lzh_kyoto-ud-{part}.conllu"
        assert sha(path)==h
        for r in extract(path,part):
            if all(ord(c) in face.coverage for c in r["page"]):
                groups[key(r["page"])].setdefault((r["split"],tuple(r["focus"])),r)
    train=[];test=[];removed=Counter()
    for ink, g in groups.items():
        if len(g)!=1:
            removed["conflict_or_split_overlap"]+=1
            continue
        r=next(iter(g.values()))
        if r["split"]=="train":
            if ink not in previous_test:
                train.append(r)
        elif ink not in old_pages and ink not in previous_test and not any(key(window(r["page"],i)) in trained_windows for i in r["focus"]):
            test.append(r)
    fit_pages=old_pages|{key(r["page"]) for r in train}
    # Also exclude target contexts used by the new location fit on longer pages.
    focus_windows={key(window(r["page"],i)) for r in train for i in r["focus"]}
    test=[r for r in test if not any(key(window(r["page"],i)) in focus_windows for i in r["focus"])]
    corpus=Path(os.environ.get("ILM_CLASSICAL_BOOKS",ROOT.parent/"ZhJpBook/books"))
    pairs=[]; sources={}
    for idx,row in enumerate(csv.DictReader((PUBLIC/"pairs.tsv").open(),delimiter="\t")):
        book=row['book'];path=corpus/book/"markdown/wenyan.md";text=path.read_text()
        sources.setdefault(book,sha(path))
        both=[]
        for side in ("front","post"):
            page=row["fronted" if side=="front" else "postposed"]
            start=text.index(page)
            assert all(page.count(row[k])==1 for k in ("verb","object"))
            indices=[page.index(row[k]) for k in ("verb","object")]
            assert (indices[0]>indices[1])==(side=="front")
            overlap=key(page) in fit_pages
            window_overlap=any(key(window(page,i)) in trained_windows or key(window(page,i)) in focus_windows for i in indices)
            both.append(dict(book=book,pair=idx,side=side,page=page,focus=indices,
                chapter=text[:start].rsplit("\n## ",1)[-1].split("\n",1)[0],
                gloss=[row["verb_glosses"].split("|"),row[side+"_object"].split("|")],
                fit_page_overlap=overlap,fit_window_overlap=window_overlap,prior_test=key(page) in previous_test))
        primary=not any(r["fit_page_overlap"] or r["fit_window_overlap"] or r["prior_test"] for r in both)
        pairs.extend(dict(r,primary=primary) for r in both)
    for file,rows in ((WORK/"train.json",train),(PUBLIC/"treebank.json",test),(PUBLIC/"external.json",pairs)):
        dump(file,rows)
    dump(PUBLIC/"split.json",dict(train=sha(WORK/"train.json"),treebank=sha(PUBLIC/"treebank.json"),external=sha(PUBLIC/"external.json"),
        editions=sources,udp=SOURCES,training_pages=len(train),fronted=sum(r["focus"][0]>r["focus"][1] for r in train),
        held=Counter(r["split"] for r in test),primary_pairs=sum(r["primary"] for r in pairs)//2,removed=removed))
    print(len(train), len(test),sum(r["primary"] for r in pairs)//2)


if __name__=="__main__":
    prepare()
