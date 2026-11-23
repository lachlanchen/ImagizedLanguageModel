"""Build source-disjoint pictured factual QA with annotated subject, relation and object."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from opencc import OpenCC
from scripts.evaluate_grounded_propositions import NOTES
from scripts.evaluate_human_proposition_questions import cases
from scripts.grounded_proposition_corpus import Face, FONTS, LANG, ROOT, corpora, exchange, label, query, statement
from scripts.vary_grounded_questions import PHRASES

WORK = ROOT / "artifacts/role_binding_v1"
PUBLIC = ROOT / "references/evidence/role_binding_20261123"
RELATIONS = list(LANG)
SEED = 23093
simplify = OpenCC("t2s").convert
traditional = OpenCC("s2t").convert


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fact(group, key, p, q, s, r, o):
    return dict(group=group, key=str(key), p=p, q=q, s=s, r=r, o=o)


def official(split, limit, blocked, rng):
    candidates = corpora(split)
    rng.shuffle(candidates)
    by_role = defaultdict(list)
    for idx, page, tuples in candidates:
        for s, r, o in tuples:
            if simplify(s) not in blocked[0] and simplify(page) not in blocked[1]:
                by_role[r].append((idx, page, s, r, o))
    rows, pages = [], set()
    while len(rows) < limit:
        prev = len(rows)
        for r in RELATIONS:
            while by_role[r]:
                idx, p, s, r, o = by_role[r].pop()
                if simplify(p) in pages:
                    continue
                if split == "train":
                    phrasing = rng.randrange(6)
                    q = query(s,r,phrasing) if phrasing < 2 else (
                        (label(s,r) + "，" if rng.random() < .8 else "") + PHRASES[r][phrasing-2] + "？")
                    if rng.random() < .30:
                        p, q, s, o = map(traditional, (p,q,s,o))
                    if p.count(s) != 1 or p.count(o) != 1:
                        continue
                else:
                    q = query(s,r,2)
                rows.append(fact("duie",idx,p,q,s,r,o))
                pages.add(simplify(p))
                break
            if len(rows) >= limit:break
        if len(rows) == prev:break
    return rows


def natural_training(blocked):
    expressions = {"导演":"导演|执导","编剧":"编剧|编写剧本","主演":"主演|领衔主演","歌手":"演唱|主唱",
                   "作词":"作词|填词","作曲":"作曲|谱曲","父亲":"父亲|爸爸","母亲":"母亲|妈妈",
                   "作者":"作者|著者|撰写者","毕业院校":"毕业于|哪所.*毕业|母校","祖籍":"祖籍","制片人":"制片人"}
    rows = []
    for split, path in (("cmrc","data/raw/cmrc2018/cmrc2018_train.json"),("drcd","data/raw/drcd/DRCD_training.json")):
        for article in json.loads((ROOT/path).read_text())["data"]:
            title = article["title"]
            for para in article["paragraphs"]:
                p = para["context"][:416]
                if simplify(p) in blocked[1]:continue
                for ex in para["qas"]:
                    q = ex["question"]
                    roles = [r for r, pat in expressions.items() if re.search(pat,simplify(q))]
                    quoted = re.findall("《([^》]{1,40})》",q)
                    subjects = quoted or [t for t in dict.fromkeys((title,traditional(title),simplify(title))) if t in q]
                    if len(roles)!=1 or len(subjects)!=1 or not ex["answers"] or len(q)+len(p)>509:continue
                    s,o = subjects[0],ex["answers"][0]["text"]
                    if simplify(s) in blocked[0] or not o or p.count(s)!=1 or p.count(o)!=1 or len(o)>48 or s==o:continue
                    rows.append(fact(split,ex["id"],p,q,s,roles[0],o))
    return rows


def prepare():
    rng = random.Random(SEED)
    _, human, _, _ = cases()
    held = [fact(m["group"],m["key"],m["p"],m["q"],m["s"],m["r"],m["o"]) for m in human]
    for i,(s,r,a,t,b,p,q1,q2) in enumerate(NOTES):
        for swapped in (False,True):
            page = exchange(p,a,b) if swapped else p
            for k,(role,obj,question) in enumerate(((r,b if swapped else a,q1),(t,a if swapped else b,q2))):
                held.append(fact("notes",4*i+2*swapped+k,page,question,s,role,obj))
    old = json.loads((ROOT/"artifacts/grounded_proposition_v1/boards.json").read_text())
    oldheads = {simplify(s) for v in old.values() for board in v for s,_,_ in board["facts"]}
    oldpages = {simplify(part) for v in old.values() for b in v for part in b["p"].split("　")}
    protected = ({simplify(x["s"]) for x in held},{simplify(x["p"]) for x in held})
    held += official("dev",1200,(oldheads|protected[0],oldpages|protected[1]),rng)
    protected = ({simplify(x["s"]) for x in held},{simplify(x["p"]) for x in held})
    train = official("train",20000,protected,rng) + natural_training(protected)
    # Contrast examples with identical glyph inventory and different factual assignments.
    for i,b in enumerate(old["train"]):
        (s,r,a),(t,u,obj)=b["facts"]
        for flipped,p in enumerate((b["p"],exchange(b["p"],a,obj))):
            for k,(h,rel,value) in enumerate(((s,r,obj if flipped else a),(t,u,a if flipped else obj))):
                if simplify(h) in protected[0] or simplify(p) in protected[1]:continue
                train.append(fact("exchange",f"{i}-{flipped}-{k}",p,query(h,rel,rng.randrange(2)),h,rel,value))
    face=Face((FONTS[0],2))
    def supported(row):return all(c.isspace() or ord(c) in face.coverage for f in ("p","q","s","o") for c in row[f])
    train=[r for r in train if supported(r)]
    assert all(supported(r) for r in held)
    assert all(simplify(x["s"]) not in protected[0] and simplify(x["p"]) not in protected[1] for x in train)
    assert len(held)==1280 and len(train)>20000
    rng.shuffle(train)
    WORK.mkdir(exist_ok=True,parents=True)
    for name,rows in (("training",train),("held",held)):
        (WORK/(name+".jsonl")).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows))
    from hashlib import sha256
    write_json(PUBLIC/"corpus.json",dict(train=dict(Counter(x["group"] for x in train)), held=dict(Counter(x["group"] for x in held)),
        distinct_training_passages=len({x["p"] for x in train}),training_by_relation=dict(Counter(x["r"] for x in train)),
        normalized_subject_and_full_passage_overlap=0,
        sha256={n:sha256((WORK/(n+".jsonl")).read_bytes()).hexdigest() for n in ("training","held")}))
    print("training",len(train),"held",len(held),flush=True)


if __name__=="__main__":
    prepare()
