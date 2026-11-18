"""Measure preference for two authentic pictured answers under exchanged questions and evidence order."""
from collections import Counter, defaultdict
import fcntl
import os
from time import monotonic

import numpy as np
import torch

from scripts.prepare_pictured_entry_answers import sha, write_json
from scripts.train_grounded_glyph_writer import ROOT, WORK, ShapeBank, Split, load_writer, padded

SRC = ROOT / "references/evidence/grounded_glyph_writer_20261119"


def choose_pairs(ds):
    rng = np.random.default_rng(19431)
    pairs = []
    for group in sorted(set(ds.groups)):
        ix = rng.permutation([i for i, g in enumerate(ds.groups) if g == group])
        ix = sorted(ix, key=lambda i: len(ds.y[i]))
        for a, b in zip(ix[::2], ix[1::2]):
            if any(np.array_equal(s[a], s[b]) for s in (ds.p, ds.q, ds.y)):
                continue
            pairs.append((a,b))
    return pairs


@torch.inference_mode()
def probe(ds, pairs, arm, device):
    net = load_writer(WORK / arm, device)
    memory = np.load(WORK / "memory.npy")
    bank = ShapeBank([bytes(128), *memory])
    ids = bank.merge(ds.tiles)
    blank = bank.merge([bytes(128)])[0]
    ink = net.ink(bank.features(device))
    keys = net.write_key(ink[1:len(memory)+1])
    classes = torch.full((len(ink),), -1, device=device)
    classes[1:len(memory)+1] = torch.arange(len(memory), device=device)
    result = defaultdict(Counter)
    choices = []
    for a,b in pairs:
        # Two evidence orders, two questions, two proposed completions, scored in one batch.
        ps, qs, ys = [], [], []
        for first, second in ((a,b),(b,a)):
            both = np.r_[ids[ds.p[first]], blank, ids[ds.p[second]]]
            for q in (a,b):
                for y in (a,b):
                    ps.append(both)
                    qs.append(ids[ds.q[q]])
                    ys.append(ids[ds.y[y]])
        tensors = [torch.as_tensor(padded(seqs),device=device) for seqs in (ps,qs,ys)]
        ll, valid = net.likelihood(ink,*tensors,keys,classes)
        scores = ((ll*valid).sum(-1)/valid.sum(-1)).view(2,2,2).cpu().numpy()
        assert np.isfinite(scores).all()
        correct = np.stack((scores[:,0,0] > scores[:,0,1],scores[:,1,1] > scores[:,1,0]),axis=1)
        stats = dict(pairs=1, individual=int(correct.sum()),both_questions_both_layouts=int(correct.all()),
                     individual_total=4,ties=int((scores[:,:,0]==scores[:,:,1]).sum()))
        for name in ("all", ds.groups[a]):
            result[name].update(stats)
        choices.append(scores)
    np.savez_compressed(WORK/arm/"answer_choices.npz", pairs=np.asarray(pairs),mean_logp=np.asarray(choices))
    return result


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("The comparison uses GPU 0 only")
    with (WORK.parent/"grounded_glyph_writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        ds = Split("development")
        pairs = choose_pairs(ds)
        start = monotonic()
        counts = {arm:probe(ds,pairs,arm,"cuda:0") for arm in ("paired","deranged")}
        report = dict(counts=counts, seconds=monotonic()-start, peak_bytes=torch.cuda.max_memory_allocated(),
                      excluded_cases=len(ds.q)-2*len(pairs), script_sha256=sha(__file__),
                      protocol_sha256=sha(ROOT/"references/grounded_answer_choice_protocol.md"))
        write_json(SRC/"answer_choices.json",report)
        print(report)


if __name__=="__main__":
    main()
