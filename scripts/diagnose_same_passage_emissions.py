"""Score original stopped emissions and corresponding references under the frozen writer."""
from collections import Counter
import fcntl
import os
import numpy as np
import torch

from scripts.audit_same_passage_writer import EVIDENCE
from scripts.audit_traditional_passage_training import archived
from scripts.evaluate_same_passage_writer import reference_sets
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_same_passage_writer import (DEVICE, PARENT, WORK, PairCorpus, load_writer, memory_bank,
                                              padded, physical_tables, small)


@torch.inference_mode()
def score(net, ink, keys, lookup, passages, questions, ys):
    out = []
    for j in range(0,len(ys),16):
        batch = [torch.as_tensor(padded(v[j:j+16]),device=DEVICE) for v in (passages,questions,ys)]
        ll, valid = net.likelihood(ink,*batch,keys,lookup)
        lengths=valid.sum(1)
        total=(ll*valid).sum(1)
        out.extend(torch.stack((total,total/lengths),dim=1).cpu().tolist())
    return np.array(out)


def diagnose():
    torch.set_num_threads(2)
    bank=memory_bank()
    ds=PairCorpus(bank,"development")
    du,groups=small(bank,"development")
    features,memory,lookup=physical_tables(bank,DEVICE)
    ids=np.load(WORK/"question_rows.npy")
    cases=[
        ("original", [ds.passages[ds.groups[i]] for i in ids], [ds.questions[i] for i in ids],
         [ds.passages[ds.groups[i]][slice(*ds.spans[i][0])] for i in ids],reference_sets(ds,ids,bank),
         ["same_passage"]*len(ids)),
        ("dureader",du.p,du.q,du.y,[{b"".join(bank.tiles[j] for j in y)} for y in du.y],groups),
    ]
    registry=load(WORK/"result.json")["archives"]
    record={}
    for arm in ("likelihood","contrast"):
        net=load_writer(WORK/arm,DEVICE)
        with torch.no_grad():
            ink=net.ink(features)
            keys=net.write_key(ink[memory])
        record[arm]={}
        for label,ps,qs,ys,refs,kinds in cases:
            path=f"{arm}/{label}.npz"
            outputs,ends=archived(WORK/path,registry[path])
            as_ink=[np.array([bank.lookup[s[k:k+128]] for k in range(0,len(s),128)],np.int64) for s in outputs]
            ref_scores=score(net,ink,keys,lookup,ps,qs,ys)
            own_scores=score(net,ink,keys,lookup,ps,qs,as_ink)
            groups_out={}
            for i,(out,stop) in enumerate(zip(outputs,ends,strict=True)):
                if not stop or out in refs[i]:
                    continue
                delta=ref_scores[i]-own_scores[i]
                for group in {"all",kinds[i]}:
                    counts=groups_out.setdefault(group,Counter())
                    counts["stopped_misses"]+=1
                    counts["sum_prefers_reference"]+=int(delta[0]>1e-5)
                    counts["mean_prefers_reference"]+=int(delta[1]>1e-5)
                    counts["both_prefer_reference"]+=int(min(delta)>1e-5)
                    counts["mean_ties"]+=int(abs(delta[1])<=1e-5)
            np.savez_compressed(WORK/arm/(label+"_score_gap.npz"),reference=ref_scores,emission=own_scores,stopped=ends)
            record[arm][label]=groups_out
    write_json(EVIDENCE/"emission_scores.json",record)
    print(record)


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("Expected GPU 0")
    with (PARENT.parent/"grounded_glyph_writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        diagnose()
