"""Compare whole passages and three fixed evidence-sentence choices on real who-questions."""
import fcntl

import numpy as np
import torch

from scripts.audit_traditional_passage_training import archived, clustered_delta
from scripts.evaluate_same_passage_writer import generate, image_archive, reference_sets
from scripts.train_inverse_role_writer import (
    DEVICE, PUBLIC, ROOT, WHO, WORK, PairCorpus, Tiles, load, load_visual_writer,
    memory_bank, physical_tables, sha, write_json,
)


def run():
    # Group passages into sentences using the visual forms of delimiters.
    torch.set_num_threads(2)
    bank = memory_bank()
    ds = PairCorpus(bank, "development")
    boundaries = set(Tiles(bank)("。！？!?"))
    with np.load(WHO/"cohort.npz") as src:
        who = src["development"]
    ids, groups, positive, negative, retrieved, questions = [], [], [], [], [], []
    contains_answer = 0
    for i in who:
        passage = ds.passages[ds.groups[i]]
        q = ds.questions[i]
        ends = sorted({j+1 for j,v in enumerate(passage) if v in boundaries} | {len(passage)})
        spans = list(zip([0]+ends[:-1], ends, strict=True))
        def covers(span):
            a,b = span
            return any(a<=l and r<=b for l,r in ds.spans[i])
        good = [s for s in spans if covers(s)]
        bad = [s for s in spans if not covers(s)]
        if not good or not bad:
            continue
        oracle = min(good, key=lambda s:(s[1]-s[0],s[0]))
        distractor = min(bad, key=lambda s:(abs((s[1]-s[0])-(oracle[1]-oracle[0])),s[0]))
        qt = set(q)-boundaries
        def overlap(s):
            terms = set(passage[s[0]:s[1]])-boundaries
            return len(terms & qt)/max(1,len(terms | qt))
        best = max(spans, key=overlap)
        contains_answer += int(covers(best))
        for dest,(a,b) in zip((positive,negative,retrieved),(oracle,distractor,best),strict=True):
            dest.append(passage[a:b])
        ids.append(int(i));groups.append(int(ds.groups[i]));questions.append(q)
    dest = WORK/"evidence_sentences"
    dest.mkdir()
    refs = reference_sets(ds,ids,bank)
    score = lambda imgs,stops: np.array([bool(end and pix in ref) for pix,end,ref in zip(imgs,stops,refs,strict=True)])
    full,stops = archived(WORK/"matched/drcd.npz",load(WORK/"result.json")["arms"]["matched"]["drcd"]["sha256"])
    intact = score([full[i] for i in ids],stops[ids])
    measurements = {}
    with torch.inference_mode():
        net = load_visual_writer(WORK/"matched",DEVICE)
        feat,mem,_ = physical_tables(bank,DEVICE)
        ink = net.ink(feat);keys = net.write_key(ink[mem])
        for tag,ps in (("contains_reference",positive),("reference_absent",negative),("visual_overlap",retrieved)):
            imgs,ends = generate(net,ink,mem,keys,ps,questions,bank)
            path = dest/(tag+".npz")
            image_archive(path,imgs,ends)
            ok = score(imgs,ends)
            measurements[tag] = dict(exact=int(ok.sum()),stopped=sum(ends),archive=sha(path),
                                      versus_intact=clustered_delta(ok,intact,groups))
    write_json(PUBLIC/"evidence_sentences.json",dict(rows=len(ids),whole=int(intact.sum()),overlap_contains_reference=contains_answer,
               conditions=measurements, question_indices=ids,checkpoint=sha(WORK/"matched/final.safetensors"),
               program=sha(__file__)))


if __name__ == "__main__":
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
