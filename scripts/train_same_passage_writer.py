"""Fit two visual writers with equal examples and work, changing only sibling-answer ranking."""
import argparse
import fcntl
import math
import os
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from scripts.compare_traditional_passage_training import TraditionalPassages
from scripts.prepare_overlap_controlled_reading import DEST as PAIRS, SOURCE
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_grounded_glyph_writer import DATA, ROOT, WORK as PARENT, ShapeBank, load_writer, padded, small

WORK = ROOT / "artifacts/same_passage_writer_v1"
REG = ROOT / "references/evidence/same_passage_writer_registration.json"
SEED, BATCH, EPOCHS, REPLAY = 19461, 24, 8, 12
ARMS = ("likelihood", "contrast")
DEVICE = "cuda:0"


def bindings():
    paths = [Path(__file__), ROOT/"scripts/evaluate_same_passage_writer.py", ROOT/"ilm/visual_lm/grounded_glyph_decoder.py",
             ROOT/"scripts/train_grounded_glyph_writer.py", ROOT/"scripts/prepare_overlap_controlled_reading.py",
             ROOT/"references/same_passage_writer_protocol.md", DATA/"preparation.json", SOURCE/"preparation.json",
             PAIRS/"receipt.json", PARENT/"paired/final.safetensors", PARENT/"memory.npy"]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


class PairCorpus:
    def __init__(self, bank, split):
        parent = TraditionalPassages(split)
        remap = bank.merge(parent.bank)
        self.passages = [remap[s] for s in parent.passages]
        self.questions = [remap[s] for s in parent.questions]
        self.groups = parent.groups
        self.spans = parent.intervals
        self.rows = np.load(PAIRS/(split+".npy"))
        receipt = load(PAIRS/"receipt.json")
        assert sha(PAIRS/(split+".npy")) == receipt["files"][split+".npy"]
        self.widths = np.array([len(p) for p in self.passages])[self.rows[:,0]]

    def batch(self, indices, device):
        """Score AA, AB, BA, BB in that order for each question pair."""
        passages, questions, targets = [], [], []
        for g, a, b, start_a, end_a, start_b, end_b in self.rows[indices]:
            p = self.passages[g]
            passages.extend((p,)*4)
            questions.extend((self.questions[a], self.questions[a], self.questions[b], self.questions[b]))
            targets.extend((p[start_a:end_a], p[start_b:end_b])*2)
        return [torch.as_tensor(padded(seqs),device=device) for seqs in (passages, questions, targets)]


def memory_bank():
    assert sha(PARENT/"memory.npy") == load(ROOT/"references/evidence/grounded_glyph_writer_20261119/audit.json")["memory_sha256"]
    return ShapeBank([bytes(128), *np.load(PARENT/"memory.npy")])


def physical_tables(bank, device):
    memory_count = len(np.load(PARENT/"memory.npy",mmap_mode="r"))
    memory = torch.arange(1, memory_count+1, device=device)
    lookup = torch.full((len(bank.tiles),), -1, device=device)
    lookup[memory] = torch.arange(memory_count,device=device)
    return bank.features(device), memory, lookup


def pair_batches(data, epoch):
    rng = np.random.default_rng(SEED + epoch)
    shuffled = rng.permutation(len(data.rows))
    for i in range(0, len(shuffled), BATCH*16):
        block = shuffled[i:i+BATCH*16]
        block = block[np.argsort(data.widths[block],kind="stable")]
        batches = [block[j:j+BATCH] for j in range(0,len(block),BATCH)]
        for j in rng.permutation(len(batches)):
            yield batches[j]


def mean_logp(net, ink, keys, lookup, batch):
    p, q, y = batch
    ll, mask = net.likelihood(ink,p,q,y,keys,lookup)
    return (ll*mask).sum(-1)/mask.sum(-1)


def fit(arm, data, replay, tables):
    target = WORK/arm
    target.mkdir()
    features, memory, lookup = tables
    torch.manual_seed(SEED)
    net = load_writer(PARENT/"paired",DEVICE).train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.00045)
    rank_weight = .5 if arm=="contrast" else 0.
    started = monotonic()
    torch.cuda.reset_peak_memory_stats()
    updates = 0
    epochs = []
    scheduled = EPOCHS*math.ceil(len(data.rows)/BATCH)
    for epoch in range(EPOCHS):
        rng = np.random.default_rng(SEED+200+epoch)
        # Balanced replay; the two arms get exactly the same indices and ordering.
        need = REPLAY*math.ceil(len(data.rows)/BATCH)
        replay_order = np.concatenate([rng.permutation(len(replay.y)) for _ in range(math.ceil(need/len(replay.y)))])
        totals = torch.zeros(3,device=DEVICE,dtype=torch.float64)
        seen = 0
        for j, indices in enumerate(pair_batches(data,epoch)):
            updates += 1
            rate = .0001 + .00035*(1+math.cos(math.pi*updates/scheduled))/2
            optimizer.param_groups[0]["lr"] = min(1.,updates/60)*rate
            given = replay_order[j*REPLAY:(j+1)*REPLAY]
            rb = [*replay.inputs(given,given,DEVICE), replay.targets(given,DEVICE)]
            batch = data.batch(indices,DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                ink = net.ink(features)
                keys = net.write_key(ink[memory])
                scores = mean_logp(net,ink,keys,lookup,batch).reshape(-1,4)
                preferred = scores[:,[0,3]]
                sibling = scores[:,[1,2]]
                positive_loss = -preferred.mean()
                ranking = F.softplus((sibling-preferred)/.35).mean()
                replay_loss = -mean_logp(net,ink,keys,lookup,rb).mean()
                loss = positive_loss + rank_weight*ranking + .3*replay_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            totals += len(indices)*torch.stack((positive_loss.detach(),ranking.detach(),replay_loss.detach()))
            seen += len(indices)
            if torch.cuda.max_memory_allocated()>12*1024**3 or monotonic()-started>1800:
                raise RuntimeError("Registered training resources exceeded")
        assert seen==len(data.rows)
        row = dict(epoch=epoch+1, pairs=seen, updates=updates, losses=(totals/seen).tolist(),
                   seconds=monotonic()-started)
        print(arm,row,flush=True)
        epochs.append(row)
    assert updates == scheduled
    path = target/"final.safetensors"
    save_file({k:v.detach().cpu().contiguous().clone() for k,v in net.state_dict().items()},str(path))
    write_json(target/"fit.json",dict(epochs=epochs,sha256=sha(path),seconds=monotonic()-started,
               peak_bytes=torch.cuda.max_memory_allocated(), updates=updates, replay_records=updates*REPLAY,
               paired_presentations=EPOCHS*len(data.rows), initial_sha256=sha(PARENT/"paired/final.safetensors")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    args = parser.parse_args()
    if args.register:
        if REG.exists():
            raise FileExistsError(REG)
        write_json(REG,bindings())
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0" or load(REG)!=bindings():
        raise ValueError("Registered sources or GPU changed")
    with (PARENT.parent/"grounded_glyph_writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        WORK.mkdir()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        bank = memory_bank()
        train = PairCorpus(bank,"train")
        replay,_ = small(bank,"train")
        tables = physical_tables(bank,DEVICE)
        for arm in ARMS:
            fit(arm,train,replay,tables)
        from scripts.evaluate_same_passage_writer import evaluate
        evaluate()


if __name__=="__main__":
    main()
