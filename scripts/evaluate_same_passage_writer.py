"""Evaluate complete responses, changed-question behavior, reference preference and composition retention."""
from collections import Counter
from time import monotonic

import numpy as np
import torch

from scripts.audit_composed_writer_outputs import distance
from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_grounded_glyph_writer import LIMIT, load_writer, padded, small
from scripts.train_same_passage_writer import (ARMS, DEVICE, PARENT, ROOT, WORK, PairCorpus, mean_logp,
                                              memory_bank, physical_tables)

READER = ROOT/"artifacts/traditional_passage_training_v1"


def compare_cluster(left,right,groups):
    stats = clustered_delta(np.asarray(left,bool),np.asarray(right,bool),groups)
    stats["passages"] = stats.pop("articles")
    return stats


def image_archive(path, pixels, stops):
    offsets = np.r_[0,np.cumsum([len(p)//128 for p in pixels])]
    raw = np.frombuffer(b"".join(pixels),np.uint8).reshape(-1,128)
    np.savez_compressed(path, pixels=raw, offsets=offsets, stopped=np.asarray(stops,bool))


def generate(net, ink, memory, keys, ps, qs, bank):
    images, stops = [], []
    for j in range(0,len(qs),24):
        p,q = [torch.as_tensor(padded(s[j:j+24]),device=DEVICE) for s in (ps,qs)]
        out,ended = net.generate(ink,p,q,keys,memory,limit=LIMIT)
        for row in out.tolist():
            images.append(b"".join(bank.tiles[k] for k in row if k))
        stops.extend(ended.tolist())
    return images,stops


def grade(pixels, stops, refs, groups=None):
    ok = np.asarray([bool(s and x in ys) for x,s,ys in zip(pixels,stops,refs,strict=True)])
    counts = Counter()
    for i,(x,s,ys) in enumerate(zip(pixels,stops,refs,strict=True)):
        counts["stopped"] += int(s)
        counts["empty"] += int(not x)
        counts["first"] += int(bool(x) and any(x[:128]==y[:128] for y in ys))
        if ok[i]:
            counts["exact"]+=1
        elif x in ys:
            counts["matches_without_stop"]+=1
        elif x and any(y.startswith(x) for y in ys):
            counts["ends_early"]+=1
        elif any(x.startswith(y) for y in ys):
            counts["surplus"]+=1
        else:
            counts["different_content_or_empty"]+=1
    if groups is not None:
        counts.update({g+"_exact":int(ok[np.asarray(groups)==g].sum()) for g in set(groups)})
    return dict(records=len(refs),**counts),ok


def reference_sets(ds, ids, bank):
    refs = []
    for i in ids:
        p = ds.passages[ds.groups[i]]
        refs.append({b"".join(bank.tiles[k] for k in p[a:b]) for a,b in ds.spans[i]})
    return refs


@torch.inference_mode()
def evaluate():
    started = monotonic()
    bank = memory_bank()
    dev = PairCorpus(bank,"development")
    du, du_groups = small(bank,"development")
    features,memory,lookup = physical_tables(bank,DEVICE)
    blank = bank.lookup[bytes(128)]
    requested = np.unique(dev.rows[:,1:3])
    inputs = [dev.passages[dev.groups[i]] for i in requested]
    questions = [dev.questions[i] for i in requested]
    refs = reference_sets(dev,requested,bank)
    pos = {v:i for i,v in enumerate(requested)}
    pairs = np.array([[pos[a],pos[b]] for a,b in dev.rows[:,1:3]])
    clusters = dev.rows[:,0]
    du_refs = [{b"".join(bank.tiles[k] for k in y)} for y in du.y]
    report = {}
    writes, preferences = {},{}
    for arm in ("parent",*ARMS):
        directory = WORK/arm
        if arm=="parent":
            directory.mkdir()
        net = load_writer(PARENT/"paired" if arm=="parent" else directory,DEVICE)
        ink = net.ink(features)
        keys = net.write_key(ink[memory])
        group = {}
        for label,qs in (("original",questions),("erased",[np.array([blank])]*len(questions))):
            pixels,ended = generate(net,ink,memory,keys,inputs,qs,bank)
            image_archive(directory/(label+".npz"),pixels,ended)
            stats,ok = grade(pixels,ended,refs)
            joint = ok[pairs].all(1)
            both_same = sum(pixels[a]==pixels[b] for a,b in pairs)
            other = sum(ended[a] and pixels[a] in refs[b] for ab in pairs for a,b in (ab,ab[::-1]))
            group[label] = dict(**stats,pairs=len(pairs),both=int(joint.sum()),same_output=both_same,
                                sibling_reference=int(other))
            if label=="original":
                writes[arm] = joint
        packed,done = generate(net,ink,memory,keys,du.p,du.q,bank)
        image_archive(directory/"dureader.npz",packed,done)
        group["dureader"],_ = grade(packed,done,du_refs,du_groups)
        all_scores = []
        for j in range(0,len(pairs),20):
            ix = np.arange(j,min(j+20,len(pairs)))
            batch = dev.batch(ix,DEVICE)
            all_scores.append(mean_logp(net,ink,keys,lookup,batch).view(-1,4).cpu().numpy())
        scores = np.concatenate(all_scores)
        assert np.isfinite(scores).all()
        np.save(directory/"candidate_mean_logp.npy",scores)
        signs = scores[:,[0,3]] > scores[:,[1,2]]
        preferences[arm] = signs.all(1)
        group["provided_answers"] = dict(individual=int(signs.sum()),both=int(signs.all(1).sum()),
                                        pairs=len(pairs),ties=int((scores[:,[0,3]]==scores[:,[1,2]]).sum()))
        report[arm] = group
        print(arm,group,flush=True)
    metadata=load(READER/"mixed_result.json")["scores"]["drcd"]
    pix,end = archived(READER/"mixed_drcd.npz",metadata["archive_sha256"])
    stats,ok = grade([pix[i] for i in requested],[end[i] for i in requested],refs)
    report["released_span_reader"] = dict(**stats,both=int(ok[pairs].all(1).sum()),pairs=len(pairs))
    contrasts = {f"contrast_vs_{arm}":dict(generation=compare_cluster(writes["contrast"],writes[arm],clusters),
                                    recognition=compare_cluster(preferences["contrast"],preferences[arm],clusters))
                 for arm in ("parent","likelihood")}
    np.save(WORK/"question_rows.npy",requested)
    r = dict(conditions=report,contrasts=contrasts,seconds=monotonic()-started,
             fits={arm:load(WORK/arm/"fit.json") for arm in ARMS},
             archives={str(p.relative_to(WORK)):sha(p) for p in WORK.rglob("*.np*")})
    write_json(WORK/"result.json",r)
    print(contrasts,flush=True)
