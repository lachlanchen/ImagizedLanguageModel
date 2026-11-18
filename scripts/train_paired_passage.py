"""Fit a pictured question planner and refine passage boundaries on aligned answer images."""
import fcntl
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_model
from torch.nn import functional as F

from ilm.visual_lm.passage_composer import PassageComposer, IGNORE, READ, JOIN, READ_STOP
from scripts.fit_mixed_ink_reader import batches, collate
from scripts.prepare_paired_passage_questions import DEST
from scripts.prepare_pictured_entry_answers import sha, load, write_json
from scripts.train_visual_extractive_reader import load_cases

ROOT = DEST.parents[1]
WORK = ROOT / "artifacts/paired_passage_reader_v1"
SEED = 19303
REG = ROOT / "references/evidence/paired_passage_registration_20261119.json"
MODEL = ROOT / "models/pictured_idiom_zh"


def unpack_sequences(path):
    with np.load(path) as z:
        return np.split(z["cells"], z["offsets"][1:-1])


class Corpus:
    def __init__(self, name, cases):
        directory = DEST / name
        for filename, expected in load(DEST / "report.json")[name]["files"].items():
            assert sha(directory / filename) == expected
        self.bank = np.load(directory / "pixels.npy")
        self.pairs = np.load(directory / "pairs.npy")
        self.passages, self.questions, self.answers, instruction = [
            unpack_sequences(directory / (s + ".npz"))
            for s in ("passages", "questions", "answers", "instruction_cells")]
        self.prefix, self.join = instruction
        assert len(self.join) == 1
        ids = load(directory / "source_ids.json")
        groups = {name: i for i,name in enumerate(load(directory / "passage_groups.json"))}
        self.groups = []
        for i,(record,c) in enumerate(zip(ids, cases, strict=True)):
            g = groups[c.group]
            assert (record == c.label and np.array_equal(self.bank[self.passages[g]], c.passage)
                    and np.array_equal(self.bank[self.questions[i]], c.question)
                    and np.array_equal(self.bank[self.answers[i]], c.answer))
            self.groups.append(g)

    def requests(self):
        prompts, tags = [], []
        for _,a,b in self.pairs:
            first, second = self.questions[a], self.questions[b]
            prompts.append(np.concatenate((self.prefix, first, self.join, second)))
            tags.append(np.array([IGNORE]*len(self.prefix) + [READ]*len(first) + [JOIN] +
                                 [READ]*(len(second)-1) + [READ_STOP], dtype=np.int8))
        for q in self.questions:
            prompts.append(q)
            tags.append(np.array([READ]*(len(q)-1) + [READ_STOP], dtype=np.int8))
        return prompts, tags

    def target(self, i):
        if i >= len(self.pairs):
            ids = self.answers[i-len(self.pairs)]
        else:
            _,a,b = self.pairs[i]
            ids = np.concatenate((self.answers[a], self.join, self.answers[b]))
        return self.bank[ids].tobytes()


def pad(arrays, device, fill=0):
    lengths = [len(v) for v in arrays]
    out = np.full((len(arrays),max(lengths)),fill,np.int32)
    for row, values in zip(out, arrays, strict=True):
        row[:len(values)] = values
    return torch.as_tensor(out, device=device), torch.as_tensor(lengths, device=device)


def fit_planner(net, features, corpus, rng):
    prompts, labels = corpus.requests()
    opt = torch.optim.AdamW(net.planner.parameters(), lr=.001)
    weights = features.new_tensor([1.,1.,5.,5.])
    history = []
    for epoch in range(6):
        numerator = denominator = 0.
        for ids in np.array_split(rng.permutation(len(prompts)), (len(prompts)+127)//128):
            q, lengths = pad([prompts[i] for i in ids], features.device)
            y, _ = pad([labels[i] for i in ids], features.device, -100)
            logits = net.plan(features, q, lengths)
            summed = F.cross_entropy(logits.transpose(1,2), y.long(), weight=weights, reduction="sum")
            norm = weights[y[y>=0].long()].sum()
            opt.zero_grad(set_to_none=True)
            (summed/norm).backward()
            opt.step()
            numerator += float(summed.detach())
            denominator += float(norm)
        history.append(numerator/denominator)
        print("planner",epoch+1,history[-1],flush=True)
    return history


def fit_reader(net, features, corpus, rng):
    rows = []
    for g,q,answer in zip(corpus.groups,corpus.questions,corpus.answers,strict=True):
        p = corpus.passages[g]
        starts = [i for i in np.flatnonzero(p==answer[0]) if np.array_equal(p[i:i+len(answer)],answer)]
        assert starts
        rows.append((p,q,starts,len(answer)))
    opt = torch.optim.AdamW(net.reader.parameters(), lr=.00025)
    history = []
    for epoch in range(8):
        total = 0.
        for ids in batches(rows,32,rng):
            batch = [rows[i] for i in ids]
            p,q,lens,qlens = collate(batch, features.device)
            a, n = pad([r[2] for r in batch],features.device)
            lengths = torch.as_tensor([r[3] for r in batch], device=features.device)
            _, start, end, duration = net.reader(features,p,q,lens,qlens)
            log_end = net.reader.end_scores(end,duration,a,lens).log_softmax(-1)
            joint = start.log_softmax(-1).gather(1,a.long()) + log_end.gather(
                2,(a+lengths[:,None]-1).unsqueeze(-1)).squeeze(-1)
            valid = torch.arange(a.shape[1],device=a.device) < n[:,None]
            loss = -joint.masked_fill(~valid,-torch.inf).logsumexp(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.reader.parameters(),2.)
            opt.step()
            total += float(loss.detach())*len(batch)
        history.append(total/len(rows))
        print("reader",epoch+1,history[-1],flush=True)
    return history


def train():
    began = monotonic()
    for name, digest in load(REG)["bound"].items():
        assert sha(ROOT/name)==digest, name
    WORK.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(.86)
    net = PassageComposer().cuda()
    net.reader.load_state_dict(load_file(str(MODEL/"mixed_reader.safetensors")))
    net.landmarks.copy_(load_file(str(MODEL/"reader_writer.safetensors"))["landmarks"])
    raw_train,raw_dev = load_cases(began)
    train = Corpus("train",raw_train)
    dev = Corpus("development",raw_dev)
    del raw_train,raw_dev
    features = net.features(train.bank)
    rng = np.random.default_rng(SEED)
    planner = fit_planner(net,features,train,rng)
    save_model(net,str(WORK/"planner_with_original_reader.safetensors"))
    reader = fit_reader(net,features,train,rng)
    save_model(net,str(WORK/"passage_composer.safetensors"))
    write_json(WORK/"fit.json",dict(planner_loss=planner,reader_loss=reader,seconds=monotonic()-began,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(), parameters=sum(p.numel() for p in net.parameters()),
               weights_sha256=sha(WORK/"passage_composer.safetensors"),
               split_sha256=sha(DEST/"report.json"), registration_sha256=sha(REG)))
    from scripts.score_paired_passage import evaluate
    evaluate(net.eval(), dev)


if __name__ == "__main__":
    with (ROOT/"artifacts/paired_passage_owner.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        train()
