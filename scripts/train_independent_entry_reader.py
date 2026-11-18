"""Compare candidate-specific and shared reading updates under identical wording diversity."""

import argparse
import fcntl
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.conditioned_entry_reader import ConditionedEntryReader
from ilm.visual_lm.entry_question_reader import anchors, offsets, retina
from ilm.visual_lm.entry_span_writer import parse_card
from scripts.audit_pictured_entry_answers import sequences
from scripts.author_entry_requests import OUT
from scripts.prepare_entry_wordings import DEST
from scripts.prepare_pictured_entry_answers import write_json
from scripts.train_entry_question_reader import (
    MODELS as PARENT,
    Requests,
    assets,
    corpus,
    load,
    sha,
    ROOT,
)

MODELS = ROOT / "models/independent_entry_reader"
SEED = 19305
ARMS = ("shared", "candidate")
EPOCHS = 8
BATCH = 128


class WordingRequests(Requests):
    def __init__(self):
        super().__init__("train", torch.device("cuda:0"))
        report = load(DEST / "report.json")
        assert all(sha(DEST / n) == h for n, h in report["files"].items())
        extra = np.load(DEST / "target.npy")
        tiles = np.load(DEST / "pixels.npy")
        rows = sequences(DEST, "queries", tiles)
        base_rows = self.strips.pop("ordered")
        self.strips.clear()
        marks, literals = assets()
        labels = [
            anchors(parse_card(c, marks), len(literals))
            for c in corpus("train")["cards"]
        ]
        loc = [offsets(q, labels[t // 8]) for q, t in zip(rows, extra, strict=True)]
        with np.load(DEST / "queries.npz") as z:
            indices, cuts = z["cells"], z["offsets"]
        width = max(base_rows.shape[1], int(np.diff(cuts).max())) + 4
        new = np.zeros((len(extra), width), np.int32)
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:], strict=True)):
            new[i, : b - a] = indices[a:b] + len(self.bank)
        covered = torch.nn.functional.pad(base_rows, (0, width - base_rows.shape[1]))
        self.strips["ordered"] = torch.cat((covered, self.tensor(new)))
        self.bank = torch.cat(
            (self.bank, self.tensor(np.stack([retina(bytes(g)) for g in tiles])))
        )
        self.anchors = torch.cat((self.anchors, self.tensor(np.stack(loc))))
        self.target = np.r_[self.target, extra]
        self.options_by_row = self.target // 8
        self.card = self.tensor(self.options_by_row.astype(np.int64))
        self.answer_id = self.tensor(self.target.astype(np.int64))


def binding():
    files = [
        Path(__file__),
        ROOT / "ilm/visual_lm/conditioned_entry_reader.py",
        ROOT / "scripts/prepare_entry_wordings.py",
        ROOT / "scripts/evaluate_independent_entry_reader.py",
        ROOT / "references/independent_entry_protocol.md",
        OUT / "questions.jsonl",
        OUT / "review.json",
        DEST / "report.json",
        PARENT / "ordered.safetensors",
        ROOT / "references/evidence/entry_question_reader_20261119/registration.json",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def fit():
    assert binding() == load(OUT / "registration.json")["sources"]
    MODELS.mkdir()
    start = monotonic()
    data = WordingRequests()
    prep = monotonic() - start
    assert len(data.target) == 89200
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = ConditionedEntryReader(arm == "candidate").cuda()
        missing, excess = model.load_state_dict(
            load_file(str(PARENT / "ordered.safetensors")), strict=False
        )
        assert missing == ["gain"] and not excess
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008)
        rng = np.random.default_rng(SEED)
        losses = []
        updates = 0
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        for epoch in range(EPOCHS):
            order = rng.permutation(len(data.target))
            steps, total = 0, 0.0
            for begin in range(0, len(order), BATCH):
                idx = data.tensor(order[begin : begin + BATCH])
                y = data.answer_id[idx]
                keys = data.keys(model, "ordered", idx)
                scores, _ = model(keys, data.history[y])
                allowed = data.gold[y]
                active = allowed.any(-1)
                scores, allowed = scores[active], allowed[active]
                loss = (
                    scores.logsumexp(-1)
                    - scores.masked_fill(~allowed, -torch.inf).logsumexp(-1)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += loss.item() * len(scores)
                steps += len(scores)
                updates += 1
            losses.append(total / steps)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 8 * 2**30
            ):
                raise RuntimeError("Resource ceiling reached")
            print(arm, epoch + 1, losses[-1], flush=True)
        path = MODELS / (arm + ".safetensors")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(path),
        )
        write_json(
            OUT / (arm + "_fit.json"),
            dict(
                seconds=monotonic() - start,
                preprocessing_seconds=prep,
                loss=losses,
                updates=updates,
                exposures=len(data.target) * EPOCHS,
                parameters=sum(p.numel() for p in model.parameters()),
                gain=model.gain.item(),
                peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                weight_sha=sha(path),
            ),
        )
        del model, optimizer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("register", "fit"))
    if p.parse_args().action == "register":
        if (OUT / "registration.json").exists():
            raise FileExistsError("Already registered")
        write_json(
            OUT / "registration.json",
            dict(sources=binding(), seed=SEED, epochs=EPOCHS, batch=BATCH),
        )
    else:
        torch.set_num_threads(2)
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fit()
