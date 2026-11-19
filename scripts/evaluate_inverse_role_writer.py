"""Read the model's stopped responses on historical, ordinary, and crossed pictured questions."""
import hashlib
import numpy as np
import torch

from ilm.visual_lm.continuous_glyph_codec_data import render_historic_svg
from ilm.visual_lm.pictured_record_selection import select_visible_record
from scripts.audit_traditional_passage_training import archived
from scripts.check_role_writer_historical import FORMS, NOUNS
from scripts.check_stance_role_switch import CARDS
from scripts.evaluate_english_request_writer import prompts
from scripts.evaluate_same_passage_writer import generate, image_archive, reference_sets
from scripts.evaluate_visual_inference_writer import paired_interval
from scripts.evaluate_visual_role_writer import NEW_CASES, role_tests, square_scores
from scripts.infer_pictured_entry import page
from scripts.measure_natural_role_questions import BASE, PUBLIC as WHO_EVIDENCE
from scripts.train_general_entry_writer import Entries, PairCorpus, small
from scripts.train_inverse_role_writer import (DEVICE, FORWARD, PARENT, PARENT_FIT, PRIVATE, PUBLIC, ROOT,
    WHO, WORK, Corpus, Tiles, load, load_visual_writer, memory_bank, physical_tables, sha, write_json)
from scripts.visual_role_examples import Predicate, situation

REQUESTS=(
    ("哪一天能进阅览室？","哪一天阅览室不开放？"),("拿哪种票能进去？","哪种票不能用于入场？"),
    ("车辆从哪个门可以进去？","哪个门只能步行通过？"),("哪个班已经交了作业？","哪个班还欠着作业？"),
    ("现在能坐火车去哪个城市？","去哪座城市的车已经不开了？"),("谁已经学会游泳？","谁目前还不会游泳？"),
    ("哪个季节能看展览？","什么时候展览在维修？"),("钥匙在哪个盒子里？","哪个盒子没有东西？"),
    ("什么时候可以看门诊？","什么时段医生不上班？"),("去机场该搭哪条线？","哪条线不到机场？"),
    ("哪栋楼可以乘电梯？","哪栋楼只能走楼梯？"),("谁考试合格了？","谁没考过？"),
    ("哪台车已修复？","哪台车还在等修理？"),("哪组已拿到奖品？","哪组还没领奖？"),
    ("哪个厅装了空调？","哪个厅仍无空调？"),("哪个罐装着盐？","哪个罐没装东西？"),
    ("哪天还有这趟船？","哪天的船被取消了？"),("今晚哪个区有水？","今晚哪个区没水？"),
    ("赢的是哪支队？","落败的是哪支队？"),("哪扇门锁好了？","哪扇门还开着？"),
)
OBJECTS=(("玉环","铜匙","石斧","木勺"),("金册","漆碗","陶豆","竹篓"),("铜盘","玉佩","木尺","石印"),
         ("石鼓","竹弓","漆盒","陶瓶"),("铜钟","玉璧","竹笼","木剑"),("石锤","竹篮","陶灯","木盘"))


def history(bank):
    result={}
    for c in "秦楚晋燕蔡魏":
        meta=load(FORMS)[c];path=ROOT.parent/"incoder"/meta["source_local_path"]
        if sha(path)!=meta["sha256"]:raise ValueError(f"Historical source changed for {c}")
        raw=np.bitwise_not(np.packbits(render_historic_svg(path).numpy().reshape(-1))).tobytes()
        assert hashlib.sha256(raw).hexdigest()==meta["ink_sha256"]
        result[c]=bank.merge([raw])
    return result


def challenge():
    cases={}
    for kind in ("describe","identify"):
        records=[]
        for noun in NOUNS:
            for i,(p,q,y) in enumerate(situation(Predicate("有","没有",noun),("秦","楚"),True,0,"write")):
                if kind=="identify":
                    q="根据材料，谁"+("有" if i%2==0 else "没有")+noun+"？请只写名称。"
                    y="秦楚"[(i%2)^(i//2)]
                records.append((p,q,y))
        cases[kind]=records
    records=[]
    for objects in OBJECTS:
        for rot in range(4):
            ns="晋燕蔡魏"[rot:]+"晋燕蔡魏"[:rot]
            p="；".join(n+"收存"+obj for n,obj in zip(ns,objects,strict=True))+"。"
            for obj,n in zip(objects,ns,strict=True):
                records.append((p,"保管"+obj+"的是哪一方？",n))
    cases["four"]=records
    events=[(x,y,p) for x,y,p,_ in CARDS]+[(x,y,p) for x,y,p,_,_ in NEW_CASES]
    cases["events"]=[(p.format(a,b),q,answer) for (x,y,p),qs in zip(events,REQUESTS,strict=True)
                     for a,b in ((x,y),(y,x)) for q,answer in zip(qs,(a,b),strict=True)]
    return cases


def available_sets(bank):
    tile=Tiles(bank);hist=history(bank)
    def enc(s,style):
        return np.concatenate([hist[c] if style=="historic" and c in hist else tile(c) for c in s])
    px=lambda y:b"".join(bank.tiles[i] for i in y)
    data={}
    for mode,rows in challenge().items():
        for style in (("print",) if mode=="events" else ("print","historic")):
            p,q,y=zip(*rows,strict=True)
            data[style+"_"+mode]=([enc(s,style) for s in p],[enc(s,style) for s in q],[{px(enc(s,style))} for s in y])
    for tag,rows in role_tests().items():
        if tag not in ("familiar_write","new_layout_write","new","prior"):continue
        p,q,y=zip(*rows,strict=True)
        data[tag]=([tile(s) for s in p],[tile(s) for s in q],[{px(tile(s))} for s in y])
    dr=PairCorpus(bank,"development")
    ix=np.arange(len(dr.questions))
    data["drcd"]=([dr.passages[g] for g in dr.groups],dr.questions,reference_sets(dr,ix,bank))
    dev=Corpus(bank,"development")
    with np.load(PRIVATE) as z: ix=z["official"];trip=np.searchsorted(ix,z["official_triples"])
    data["inference"]=([dev.p[i] for i in ix],[dev.q[i] for i in ix],[{px(dev.y[i])} for i in ix])
    valid,holes,ys=dev.holes(ix,np.random.default_rng(73122))
    assert np.array_equal(valid,ix)
    data["spelling"]=([dev.p[i] for i in valid],holes,[{px(y)} for y in ys])
    du,_=small(bank,"development");data["dureader"]=(du.p,du.q,[{px(y)} for y in du.y])
    ents=Entries(bank,"development");bounds=tile("【】")
    for tag,(qs,ids) in prompts(bank,ents).items():
        if tag.endswith("_free"):
            data[tag]=([select_visible_record(ents.cards[ents.card_of[i]],q,*bounds)[0] for i,q in zip(ids,qs,strict=True)],
                       qs,[{px(ents.y[i])} for i in ids])
    return data,dr,trip


@torch.inference_mode()
def evaluate(work):
    bank=memory_bank();data,dr,trip=available_sets(bank);feat,mem,_=physical_tables(bank,DEVICE)
    with np.load(WHO/"cohort.npz") as split:who=split["development"];pairs=split["development_pairs"]
    who_groups=[dr.groups[i] for i in who];pair_groups=[who_groups[a] for a,_ in pairs]
    report={};points={}
    for arm in ("matched","randomized","parent"):
        dest=work/arm
        if arm=="parent":dest.mkdir()
        net=load_visual_writer(PARENT_FIT if arm=="parent" else dest,DEVICE)
        ink=net.ink(feat);keys=net.write_key(ink[mem]);stats={};points[arm]={}
        for tag,(ps,qs,refs) in data.items():
            # Baseline outputs already exist on every retained older test. Only the new
            # event questions and four-participant circles need a new parent generation.
            if arm=="parent" and tag not in ("print_four","historic_four","print_events"):continue
            imgs,stops=generate(net,ink,mem,keys,ps,qs,bank)
            out=dest/(tag+".npz");image_archive(out,imgs,stops)
            ok=np.array([s and p in aliases for s,p,aliases in zip(stops,imgs,refs,strict=True)])
            rec=dict(exact=int(ok.sum()),total=len(ok),stopped=sum(stops),sha256=sha(out))
            if tag=="drcd":
                mask=ok[who];paired=mask[pairs].all(1)
                rec.update(who=int(mask.sum()),who_total=len(who),who_pairs=int(paired.sum()),pairs=len(pairs))
                points[arm].update(who=mask,who_pairs=paired)
            elif tag.endswith("_four"):
                cube=ok.reshape(-1,4,4)
                rec.update(all_rotations=int(cube.all(1).sum()),rotation_sets=4*len(cube),all_sixteen=int(cube.all((1,2)).sum()))
            elif tag in ("inference","spelling"):
                rec.update(on_qwen=int(ok[trip].sum()),all_three=int(ok[trip].all(1).sum()))
            elif tag not in ("dureader","chinese_free","english_free"):
                rec.update(square_scores(ok,imgs))
                if tag=="print_events":points[arm]["events"]=ok.reshape(-1,2,2).all(1).ravel()
            if arm=="matched" and tag=="historic_identify":
                for i in (0,2):
                    for n,raw in (("premise",b"".join(bank.tiles[j] for j in ps[i])),("answer",imgs[i]),("query",b"".join(bank.tiles[j] for j in qs[i]))):
                        page(raw).save(dest/f"bronze{i//2}_{n}.png")
            stats[tag]=rec
        report[arm]=stats
        print(arm,{k:v["exact"] for k,v in stats.items()},flush=True)
    # Score comparator systems on the same natural records and aliases, with paired uncertainty.
    for sys in ("parent","passage_reader"):
        prev=load(WHO_EVIDENCE/"baseline.json")["arms"][sys if sys=="passage_reader" else "matched"]
        filename="passage_reader.npz" if sys=="passage_reader" else "matched.npz"
        imgs,stops=archived(FORWARD/"natural_who"/filename,prev["sha256"])
        ok=np.array([end and pix in data["drcd"][2][i] for i,pix,end in zip(who,imgs,stops,strict=True)])
        assert int(ok.sum())==prev["exact"]
        points.setdefault(sys,{}).update(who=ok,who_pairs=ok[pairs].all(1))
    effects={}
    for metric,groups in (("who",who_groups),("who_pairs",pair_groups),("events",np.arange(40)//2)):
        for control in ("parent","randomized","passage_reader"):
            if metric in points[control]:
                effects[metric+"_"+control]=paired_interval(points["matched"][metric],points[control][metric],groups)
    write_json(work/"result.json",dict(arms=report,effects=effects,
        fits={a:load(work/a/"fit.json") for a in ("matched","randomized")},registration=sha(PUBLIC/"registration.json")))
