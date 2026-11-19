"""Score full, stopped bitmap answers after reversing the evidence, claim, or both."""
from pathlib import Path

import numpy as np
import torch

from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_same_passage_writer import generate, image_archive, reference_sets
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.prepare_visual_inference_pairs import RESPONSES
from scripts.probe_visual_inference_switches import cases
from scripts.train_general_entry_writer import Entries, PairCorpus, small
from scripts.train_visual_inference_writer import Corpus, DECIDE, DEVICE, PRIVATE, WORK as PREVIOUS, load_visual_writer
from scripts.train_visual_role_writer import PUBLIC, START, Tiles, WORK, load, memory_bank, physical_tables, sha, write_json
from scripts.visual_role_examples import TEST_NAMES, predicates, situation

# New wording and two-clause contrasts, written before either fit.
NEW_CASES = (
    ("红车","白车","{}已经修好，{}还在等待维修。","{}已经修好。","{}还没修好。"),
    ("一组","二组","{}领取了奖品，{}没有领取。","{}拿到了奖品。","{}还没领奖。"),
    ("西厅","东厅","{}装有空调，{}至今没有空调。","{}有空调。","{}没有空调。"),
    ("铜罐","铁罐","{}里装满了盐，{}里面是空的。","{}里有盐。","{}是空的。"),
    ("周六","周日","{}这班船正常运行，{}取消了这班船。","{}有这班船。","{}这班船停航。"),
    ("南区","北区","供水通知说，{}今晚供水，{}今晚停水。","{}今晚有水。","{}今晚停水。"),
    ("青队","黄队","这场比赛{}获胜，{}落败。","{}赢了这场比赛。","{}输了这场比赛。"),
    ("前门","后门","{}已经上锁，{}敞开着。","{}锁着。","{}开着。"),
)


def role_tests():
    blocks = {}
    for label,source in (("prior",cases()),("new",fresh_cases())):
        rows=[(p,DECIDE+claim,RESPONSES[0 if yes else 1]) for p,_,_,claims in source for claim,yes in claims]
        # Source iterator nests orientation then predicate then subject. Reorder so each
        # four-row square has a fixed predicate and contains both orientations and subjects.
        blocks[label]=[rows[8*c+4*o+2*f+s] for c in range(len(rows)//8)
                       for f in range(2) for o in range(2) for s in range(2)]
    items=predicates()["validation"]
    rng=np.random.default_rng(50923)
    for layout in ("familiar","new_layout"):
        for task in ("decision","write"):
            blocks[layout+"_"+task]=[]
    for n,pred in enumerate(items):
        names=tuple(rng.choice(TEST_NAMES,2,replace=False))
        first=bool(rng.integers(2))
        for key,layout in (("familiar",n%2),("new_layout",2)):
            for task in ("decision","write"):
                blocks[key+"_"+task].extend(situation(pred,names,first,layout,task,n%2==1))
    return blocks


def fresh_cases():
    for x,y,passage,pos,neg in NEW_CASES:
        for a,b in ((x,y),(y,x)):
            for form,want in ((pos,a),(neg,b)):
                yield passage.format(a,b),form,want,[(form.format(k),k==want) for k in (x,y)]


def square_scores(ok, outputs):
    result=ok.reshape(-1,2,2)
    pixels=np.array(outputs,dtype=object).reshape(-1,2,2)
    return dict(squares=len(result),all_four=int(result.all(axis=(1,2)).sum()),
                both_sources=int(result.all(axis=1).sum()),both_queries=int(result.all(axis=2).sum()),
                identical_sources=int((pixels[:,0]==pixels[:,1]).sum()),pairs=2*len(result))


@torch.inference_mode()
def evaluate(work):
    report=work/"result.json"
    if report.exists():raise FileExistsError(report)
    bank=memory_bank()
    train,dev=[Corpus(bank,k) for k in ("train","development")]
    tile=Tiles(bank)
    blocks=role_tests()
    sets={}
    pixels=lambda y:b"".join(bank.tiles[v] for v in y)
    clusters={}
    for key,rows in blocks.items():
        p,q,y=zip(*rows,strict=True)
        ps,qs=[[tile(x) for x in seq] for seq in (p,q)]
        sets[key]=(ps,qs,[{pixels(tile(x))} for x in y])
        if key.endswith("_write"):
            # Cross-cell content must be compared on cell boundaries, not as Unicode.
            for a,b,c in rows:
                assert c not in a and c not in b
    with np.load(PRIVATE) as split:
        for j,(name,ds) in enumerate((("validation",train),("official",dev))):
            ix=split[name]
            sets[name]=([ds.p[i] for i in ix],[ds.q[i] for i in ix],[{pixels(ds.y[i])} for i in ix])
            clusters[name]=[ds.meta[i]["premise"] for i in ix]
            valid,holes,targets=ds.holes(ix,np.random.default_rng(73121+j))
            assert np.array_equal(valid,ix)
            sets[name+"_cloze"]=([ds.p[i] for i in valid],holes,[{pixels(y)} for y in targets])
        triples={k:np.searchsorted(split[k],split[k+"_triples"]) for k in ("official","validation")}
    entry=Entries(bank,"development")
    bounds=tile("【】")
    for key,(qs,ids) in prompts(bank,entry).items():
        if key in ("english_free","chinese_free"):
            ps=[select_visible_record(entry.cards[entry.card_of[i]],q,*bounds)[0] for q,i in zip(qs,ids,strict=True)]
            sets[key]=(ps,qs,[{pixels(entry.y[i])} for i in ids])
    du,_=small(bank,"development")
    sets["dureader"]=(du.p,du.q,[{pixels(y)} for y in du.y])
    dr=PairCorpus(bank,"development")
    ids=np.unique(dr.rows[:,1:3]); pairs=np.searchsorted(ids,dr.rows[:,1:3])
    sets["drcd"]=([dr.passages[dr.groups[i]] for i in ids],[dr.questions[i] for i in ids],reference_sets(dr,ids,bank))
    feat,mem,_=physical_tables(bank,DEVICE)
    arms={};flags={}
    for arm in ("parent","matched","randomized"):
        out=work/arm
        if arm=="parent":out.mkdir()
        net=load_visual_writer(START if arm=="parent" else out,DEVICE)
        ink=net.ink(feat);keys=net.write_key(ink[mem])
        arms[arm]={};flags[arm]={}
        for name,(ps,qs,ys) in sets.items():
            imgs,stops=generate(net,ink,mem,keys,ps,qs,bank)
            ok=np.array([s and x in y for x,s,y in zip(imgs,stops,ys,strict=True)])
            path=out/(name+".npz");image_archive(path,imgs,stops)
            rec=dict(exact=int(ok.sum()),total=len(ok),stopped=sum(stops),archive_sha256=sha(path))
            if name in blocks:rec.update(square_scores(ok,imgs))
            if name in triples:
                tri=triples[name]
                rec.update(triples=int(ok[tri].all(1).sum()),triple_rows=int(ok[tri].sum()),families=len(tri))
            if name=="official_cloze":rec["triple_rows"]=int(ok[triples["official"]].sum())
            if name=="drcd":rec["both"]=int(ok[pairs].all(1).sum())
            arms[arm][name]=rec
            flags[arm][name]=ok
        print(arm,{k:v["exact"] for k,v in arms[arm].items()},flush=True)
    effects={}
    for name in ("prior","new","familiar_decision","new_layout_decision","familiar_write","new_layout_write","official","validation"):
        for control in ("parent","randomized"):
            if name in blocks:
                a,b=[flags[key][name].reshape(-1,2,2).all(1).ravel() for key in ("matched",control)]
                group=np.arange(len(a))//(4 if name in ("prior","new") else 2)
            else:
                a,b=[flags[key][name] for key in ("matched",control)]
                group=clusters[name]
            effects[name+"_"+control]=paired_interval(a,b,group)
    p=predicates()
    write_json(report,dict(arms=arms,effects=effects,fit_predicates=len(p["fit"]),validation_predicates=len(p["validation"]),
              registration_sha256=sha(PUBLIC/"registration.json"),fits={key:load(work/key/"fit.json") for key in ("matched","randomized")}))
