"""Build paired pictured propositions from the official DuIE research corpus."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

from scripts.fit_font_invariant_reader import Face, FONTS
from scripts.pointing_glyph_corpus import Row
from scripts.train_pointing_causal_writer import ROOT, record, sha

ARCHIVE = ROOT/"data/raw/duie2/DuIE_2_0.zip"
ARCHIVE_HASH = "df223b18f3bd0fe19a61d18d7d59a22cd3f145c342c58470e93851bb4d1b40ed"
WORK = ROOT/"artifacts/grounded_proposition_v1"
PUBLIC = ROOT/"references/evidence/grounded_proposition_20261119"
# Neutral statements permit both factual text and explicitly counterfactual pages.
# Each relation has two seen questions and one held paraphrase.
LANG = {
    "作者": ("作者","是谁写的","作者叫什么","这部作品出自谁的笔下"),
    "作词": ("作词人","是谁作词的","歌词是谁写的","谁为它填了词"),
    "作曲": ("作曲人","是谁作曲的","曲子由谁创作","这段旋律出自谁之手"),
    "歌手": ("演唱者","由谁演唱","歌手是谁","谁唱了这首歌"),
    "导演": ("导演","由谁执导","是谁导演的","镜头背后执导的人是哪位"),
    "编剧": ("编剧","编剧是谁","剧本是谁写的","谁负责把故事写成剧本"),
    "主演": ("主演","由谁主演","主要演员是哪位","谁在其中担纲演出"),
    "制片人": ("制片人","制片人是谁","由谁担任制片人","谁负责制片工作"),
    "父亲": ("父亲","的父亲是谁","的爸爸叫什么","谁是这个人的父亲"),
    "母亲": ("母亲","的母亲是谁","的妈妈叫什么","谁是这个人的母亲"),
    "毕业院校": ("毕业院校","毕业于哪所学校","的毕业院校叫什么","哪所学校是其母校"),
    "祖籍": ("祖籍","的祖籍是哪里","祖籍在哪儿","其家族原籍何处"),
}
PERSON = {"父亲","母亲","毕业院校","祖籍"}


def label(s, r):
    return s if r in PERSON else f"《{s}》"


def noun(s,r):
    return label(s,r)+"的"+LANG[r][0]


def statement(s,r,o):
    return noun(s,r)+"是"+o+"。"


def query(s,r,variant):
    who = label(s,r)
    if variant == 2:
        return f"关于{who}，{LANG[r][3]}？请用一句话回答。"
    return who+LANG[r][variant+1]+"？请用完整的句子作答。"


def compound_question(facts, reverse):
    first,second = facts[::-1] if reverse else facts
    s,r,_=first
    _,t,_=second
    return f"介绍{label(s,r)}，先写{LANG[r][0]}，再写{LANG[t][0]}。分别用一句话说明。"


def corpora(split):
    rows=[]
    with ZipFile(ARCHIVE) as z, z.open(f"DuIE_2_0/{split}.json") as stream:
        for idx,line in enumerate(stream):
            item=json.loads(line)
            p=item["text"]
            if not 14<=len(p)<=145 or any(ord(c)<32 for c in p):continue
            groups=defaultdict(set)
            for rel in item["spo_list"]:
                s,r,o=rel["subject"],rel["predicate"],rel["object"]["@value"]
                if r in LANG:groups[s,r].add(o)
            facts=[]
            for (s,r),objects in groups.items():
                if len(objects)!=1:continue
                o=next(iter(objects))
                if not (2<=len(s)<=13 and 2<=len(o)<=10 and s in p and p.count(o)==1):continue
                if not all(re.fullmatch(r"[\u3400-\u9fffA-Za-z0-9·]+",v) for v in (s,o)):continue
                if s in o or o in s:continue
                facts.append((s,r,o))
            if facts:rows.append((idx,p,sorted(facts)))
    return rows


def exchange(p,a,b):
    if p.count(a)!=1 or p.count(b)!=1:raise ValueError("Ambiguous mention")
    return p.replace(a,"\0").replace(b,a).replace("\0",b)


def admissible(p,facts):
    (s,r,a),(t,u,b)=facts
    if a==b or a in b or b in a or any(x in v or v in x for x in (a,b) for v in (s,t)):return False
    if p.count(a)!=1 or p.count(b)!=1:return False
    if r in ("父亲","母亲") and u in ("父亲","母亲") and r!=u:return False
    pp=exchange(p,a,b)
    if Counter(pp)!=Counter(p) or pp==p:return False
    for page,vs in ((p,(a,b)),(pp,(b,a))):
        for (head,rel,_),val in zip(facts,vs):
            if statement(head,rel,val) in page:return False
    return True


def select(split, blocked, limit):
    rng=random.Random(29189 if split=="train" else 29190)
    items=corpora(split)
    rng.shuffle(items)
    dual=[];paired=[];seen=set();heads=set();waiting={};used=set()
    oldheads,oldpages=blocked
    face=Face((FONTS[0],2))
    for idx,p,fs in items:
        if p in oldpages or p in seen:continue
        try:face(p)
        except ValueError:continue
        fs=[f for f in fs if f[0] not in oldheads and f[0] not in heads]
        eligible=[(a,b) for a,b in itertools.combinations(fs,2)
                  if a[0]==b[0] and a[1]!=b[1] and admissible(p,(a,b))]
        if len(dual)<limit and eligible:
            a,b=rng.choice(eligible)
            dual.append(dict(kind="relation",ids=[idx],p=p,facts=[a,b]))
            heads.add(a[0]);seen.add(p);used.add(idx)
        if len(dual)>=limit:break
    for idx,p,fs in items:
        if p in oldpages or p in seen or idx in used:continue
        fs=[f for f in fs if f[0] not in oldheads and f[0] not in heads]
        if not fs:continue
        try:face(p)
        except ValueError:continue
        a=rng.choice(fs);s,r,o=a
        if r not in waiting:
            waiting[r]=(idx,p,a);continue
        j,q,b=waiting[r]
        if b[0] in heads or b[0]==s or b[0] in p or s in q or q==p or j in used:
            waiting[r]=(idx,p,a);continue
        paragraph=q+"　"+p
        if len(paragraph)>218 or not admissible(paragraph,(b,a)):continue
        paired.append(dict(kind="subject",ids=[j,idx],p=paragraph,facts=[b,a]))
        heads.update((b[0],s));seen.update((q,p));used.update((idx,j));del waiting[r]
        if len(paired)>=limit:break
    if min(len(dual),len(paired))<min(limit,80):raise ValueError((split,len(dual),len(paired)))
    return dual+paired, (heads,seen)


def views(board):
    p=board["p"]
    (s,r,a),(t,u,b)=board["facts"]
    return [(p,[(s,r,a),(t,u,b)]),(exchange(p,a,b),[(s,r,b),(t,u,a)])]


def materialize(boards,split,compound=False):
    face=Face((FONTS[0],2));rows=[];meta=[]
    for i,board in enumerate(boards):
        for edit,(p,facts) in enumerate(views(board)):
            if compound and board["kind"]!="relation":continue
            for variant in ([0] if split=="train" or compound else [0,2]):
                qs=([compound_question(facts,b) for b in (False,True)] if compound else
                    [query(s,r,variant if variant==2 else i%2) for s,r,o in facts])
                for j,q in enumerate(qs):
                    chosen=(facts[::-1] if j else facts) if compound else [facts[j]]
                    y="".join(statement(*f) for f in chosen)
                    if len(y)>48 and compound:continue
                    assert y not in p and y not in q
                    key=f"{board['kind']}/{i}/{edit}/{variant}/{j}"
                    rows.append(Row(board["kind"],key,face(p),face(q),face(y),face(qs[1-j])))
                    meta.append(dict(board=i,edit=edit,variant=variant,request=j,facts=chosen,all_facts=facts,p=p,q=q,y=y))
    return rows,meta


def prepare():
    assert sha(ARCHIVE)==ARCHIVE_HASH
    WORK.mkdir(exist_ok=True)
    train,(heads,pages)=select("train",(set(),set()),850)
    dev,_=select("dev",(heads,pages),112)
    record(WORK/"boards.json",dict(train=train,development=dev))
    record(PUBLIC/"corpus.json",dict(source="Baidu DuIE 2.0",archive_sha256=ARCHIVE_HASH,
        boards_sha256=sha(WORK/"boards.json"),train=dict(Counter(x["kind"] for x in train)),
        development=dict(Counter(x["kind"] for x in dev)),training_subjects=len(heads),training_singlesources=len(pages)))
    print("train",Counter(x["kind"] for x in train),"development",Counter(x["kind"] for x in dev),flush=True)


if __name__=="__main__":prepare()
