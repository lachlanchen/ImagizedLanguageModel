"""Compare free pictured answers on untouched roles, fresh scenes and prior language tasks."""

import numpy as np
from safetensors.torch import load_file
import torch

from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_inverse_role_writer import available_sets
from scripts.evaluate_same_passage_writer import image_archive
from scripts.explicit_participant_cases import cases
from scripts.infer_pictured_entry import page
from scripts.pictured_participant_scenes import fresh_cases
from scripts.train_interval_writer import WHO
from scripts.train_pictured_participants import (
    DEVICE, GRAMMAR, PREVIOUS, PREVIOUS_PUBLIC, PUBLIC, START, WORK, SentenceCorpus, Tiles,
    load, load_visual_writer, make_model, memory_bank, physical_tables, sha, to_batch, treebank, write_json,
)

HELD_VERBS = (
    ("攙扶","伸手扶人","在旁人的攙扶下行走"), ("解僱","辭退別人","丟了工作"),
    ("擁抱","主動張開雙臂把對方抱住","被對方抱在懷裏"), ("逮捕","動手抓人","遭到拘捕"),
    ("斥責","出言訓斥另一人","遭受訓斥"), ("資助","出錢幫忙","得到金錢上的援助"),
    ("推舉","提名對方","被人提名"), ("營救","實施營救","被人從險境中救出"),
)


def held_cases():
    for verb, agent, patient in HELD_VERBS:
        for wording in ("{a}的確{v}了{b}。", "{b}那天被{a}{v}了。"):
            for a, b in (("陸芷","范熙"),("范熙","陸芷")):
                p = wording.format(v=verb, a=a, b=b)
                yield p, "誰"+agent+"？", a
                yield p, "誰"+patient+"？", b


def effect(candidate, control, groups):
    _, ix = np.unique(groups, return_inverse=True)
    n = np.bincount(ix)
    delta = np.bincount(ix, weights=candidate.astype(float)-control)
    sample = np.random.default_rng(190101).integers(len(n), size=(5000,len(n)))
    diffs = delta[sample].sum(1) / n[sample].sum(1)
    return dict(delta=float(delta.sum()/n.sum()), ci95=np.quantile(diffs,[.025,.975]).tolist())


@torch.inference_mode()
def evaluate():
    bank = memory_bank()
    tile = Tiles(bank)
    tests, dr, _ = available_sets(bank)
    keep = ("drcd", "familiar_write", "new_layout_write", "historic_describe", "historic_four", "dureader")
    tests = {k:tests[k] for k in keep}
    raw = lambda seq: b"".join(bank.tiles[i] for i in seq)
    groups = {}
    unseen = {}
    training_verbs = {m["verb"] for m in load(GRAMMAR/"train/rows.json")}
    for split in ("dev","test"):
        pairs, meta = treebank(bank, split)
        ps, qs, ys = [], [], []
        for p, q1, q2, a, b in pairs:
            ps.extend((p,p))
            qs.extend((q1,q2))
            ys.extend(({raw(p[slice(*a)])}, {raw(p[slice(*b)])}))
        tag = "gsd_"+split
        tests[tag] = ps, qs, ys
        groups[tag] = [m["id"] for m in meta]
        unseen[tag] = np.array([m["verb"] not in training_verbs for m in meta])

    for tag, records in (("explicit", cases()), ("fresh", fresh_cases()), ("held_verbs", held_cases())):
        rows = list(records)
        tests[tag] = ([tile(p) for p,_,_ in rows], [tile(q) for _,q,_ in rows],
                      [{raw(tile(y))} for _,_,y in rows])
    corpus = SentenceCorpus(dr, bank, "development")
    with np.load(WHO/"cohort.npz") as z:
        who, who_pairs = z["development"], z["development_pairs"]
    marks = set(tile("。！？!?"))
    passages = []
    for i in who:
        g = dr.groups[i]
        p = dr.passages[g]
        q = set(dr.questions[i])-marks
        def overlap(region):
            a,b = region
            cells = set(p[a:b])-marks
            return len(cells&q)/max(1,len(cells|q))
        a,b = max(corpus.bounds[g], key=overlap)
        passages.append(p[a:b])
    tests["overlap_who"] = passages, [dr.questions[i] for i in who], [tests["drcd"][2][i] for i in who]
    feat, mem, _ = physical_tables(bank, DEVICE)
    writer = load_visual_writer(START, DEVICE).requires_grad_(False)
    ink = writer.ink(feat)
    keys = writer.write_key(ink[mem])
    parent = load(PREVIOUS_PUBLIC/"calibrated.json")["scores"]
    report, flags = {}, {}
    for arm in ("parent","matched","exchanged"):
        folder = WORK/arm
        folder.mkdir(exist_ok=True)
        net = make_model(53231).eval()
        if arm != "parent":
            path = folder/"reader.safetensors"
            assert sha(path) == load(folder/"fit.json")["weights"]
            net.load_state_dict(load_file(str(path)))
        report[arm], flags[arm] = {}, {}
        for tag,(ps,qs,ys) in tests.items():
            imgs, done = [], []
            for i in range(0,len(qs),24):
                p, q = to_batch(ps[i:i+24], qs[i:i+24])
                out, stop = net.generate(writer, ink, p, q, keys, mem)
                imgs.extend(raw(row[row!=0]) for row in out.cpu().numpy())
                done.extend(stop.tolist())
            path = folder/(tag+".npz")
            image_archive(path, imgs, done)
            ok = np.array([s and y in targets for s,y,targets in zip(done,imgs,ys,strict=True)])
            if arm == "parent" and tag in parent:
                old, stops = archived(PREVIOUS/"calibrated"/(tag+".npz"), parent[tag]["archive"])
                assert imgs == old and list(stops)==done, tag
            rec = dict(exact=int(ok.sum()),total=len(ok),stopped=sum(done),archive=sha(path))
            if tag in groups:
                both = ok.reshape(-1,2).all(1)
                rec.update(both=int(both.sum()), predicates=len(both), unseen_both=int(both[unseen[tag]].sum()),
                           unseen_predicates=int(unseen[tag].sum()))
                flags[arm][tag] = both
            if tag in ("fresh","explicit","held_verbs"):
                sq=ok.reshape(-1,2,2)
                all_four=sq.all((1,2))
                rec.update(both_questions=int(sq.all(2).sum()),all_four=int(all_four.sum()),scenes=len(all_four))
                flags[arm][tag] = ok
                if tag=="fresh":
                    for j in range(4):
                        for k, px in (("passage",raw(ps[j])),("question",raw(qs[j])),("answer",imgs[j])):
                            page(px).save(folder/f"fresh{j}_{k}.png")
            if tag=="overlap_who":
                rec["both_questions"]=int(ok[who_pairs].all(1).sum())
            report[arm][tag]=rec
        print(arm,{k:v['exact'] for k,v in report[arm].items()},flush=True)
    effects={}
    for tag in ("gsd_test","fresh","explicit","held_verbs"):
        grouped = groups.get(tag)
        if grouped is None:
            size = 8 if tag=="held_verbs" else 4
            grouped = np.repeat(np.arange(len(flags["matched"][tag])//size),size)
        effects[tag]={arm:effect(flags["matched"][tag],flags[arm][tag],grouped)
                      for arm in ("exchanged","parent")}
    result = dict(arms=report,effects=effects,fits={a:load(WORK/a/"fit.json") for a in ("matched","exchanged")},
                  registration=sha(PUBLIC/"registration.json"))
    write_json(WORK/"result.json", result)
    write_json(PUBLIC/"result.json", result)
