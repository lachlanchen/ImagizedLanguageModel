"""Fit two matched visual readers on pictured questions requiring clause order."""

import argparse
import fcntl
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.entry_question_reader import QuestionReader, anchors, offsets, retina
from ilm.visual_lm.entry_span_writer import (
    VIS,
    alignment,
    alternatives,
    cells,
    parse_card,
)
from scripts.audit_pictured_entry_answers import sequences
from scripts.prepare_entry_request_order import DEST
from scripts.train_pictured_entry_writer import (
    MODELS as BASE_MODELS,
    ROOT,
    assets,
    corpus,
    load,
    sha,
)
from scripts.prepare_pictured_entry_answers import write_json

OUT = ROOT / "references/evidence/entry_question_reader_20261119"
MODELS = ROOT / "models/entry_question_reader"
BATCH, EPOCHS, SEED = 128, 8, 77351
ARMS = ("canonical", "ordered")
BASE = BASE_MODELS / "matched.safetensors"


def registration():
    files = [
        BASE,
        BASE_MODELS / "appearance.npz",
        Path(__file__),
        ROOT / "ilm/visual_lm/entry_question_reader.py",
        ROOT / "ilm/visual_lm/entry_span_writer.py",
        ROOT / "scripts/evaluate_entry_question_reader.py",
        DEST / "report.json",
        ROOT / "references/pictured_entry_request_order_protocol.md",
    ]
    return dict(
        sources={str(p.relative_to(ROOT)): sha(p) for p in files},
        epochs=EPOCHS,
        batch=BATCH,
        seed=SEED,
        steps_per_epoch=349,
        training_examples=44600,
        seconds_per_arm=1200,
        gpu_gib=8,
    )


class Requests:
    def __init__(self, split, device, condition="questions"):
        self.device = device
        source = corpus(split)
        marks, literal = assets()
        records = [parse_card(raw, marks) for raw in source["cards"]]
        self.options = [alternatives(r, b"", literal) for r in records]
        labels = [anchors(r, len(literal)) for r in records]
        self.reference = source["answers"]
        if condition == "ordinary":
            questions = source["questions"]
            target = np.arange(len(questions))
        else:
            directory = DEST / split
            report = load(DEST / "report.json")["splits"][split]
            for name, expected in report["files"].items():
                if sha(directory / name) != expected:
                    raise ValueError("Changed question corpus")
            questions = sequences(
                directory, condition, np.load(directory / "query_pixels.npy")
            )
            target = np.load(directory / "target_row.npy")
            if split == "train":
                questions.extend(source["questions"])
                target = np.r_[target, np.arange(len(self.reference))]

        self.target = target
        self.options_by_row = target // 8
        catalog = sorted({g for question in questions for g in cells(question)})
        index = {g: i + 1 for i, g in enumerate(catalog)}
        bank = np.stack([np.zeros(64, np.float32)] + [retina(g) for g in catalog])
        rows = np.zeros((len(questions), max(map(len, questions)) // 128), np.int32)
        sorted_rows = np.zeros_like(rows)
        locations = []
        for i, q in enumerate(questions):
            inventory = [index[g] for g in cells(q)]
            rows[i, : len(inventory)] = inventory
            sorted_rows[i, : len(inventory)] = sorted(inventory)
            locations.append(offsets(q, labels[self.options_by_row[i]]))
        self.tensor = lambda array: torch.as_tensor(array, device=device)
        self.bank = self.tensor(bank)
        self.strips = {
            "ordered": self.tensor(rows),
            "canonical": self.tensor(sorted_rows),
        }
        self.anchors = self.tensor(np.stack(locations))
        self.surfaces = self.tensor(np.stack([o.features for o in self.options]))
        self.card = self.tensor(self.options_by_row.astype(np.int64))
        self.answer_id = self.tensor(target.astype(np.int64))
        if split == "train":
            paths = [
                alignment(self.options[i // 8], y) for i, y in enumerate(self.reference)
            ]
            steps = max(len(y) for _, y in paths)
            history = np.zeros((len(paths), steps, VIS), np.float32)
            gold = np.zeros((len(paths), steps, len(self.options[0].payload) + 1), bool)
            for i, (h, alignment_choices) in enumerate(paths):
                history[i, : len(h)] = h
                for j, choices in enumerate(alignment_choices):
                    gold[i, j, choices] = True
            self.history, self.gold = self.tensor(history), self.tensor(gold)

    def keys(self, model, arm, indices):
        strip = self.strips[arm][indices]
        return model.read(
            self.surfaces[self.card[indices]],
            self.anchors[indices],
            self.bank[strip.long()],
            strip != 0,
        )


def fit():
    if load(OUT / "registration.json") != registration():
        raise ValueError("Study differs from registration")
    MODELS.mkdir(exist_ok=False)
    device = torch.device("cuda:0")
    start = monotonic()
    data = Requests("train", device)
    preprocessing = monotonic() - start
    assert len(data.target) == 44600
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = QuestionReader().to(device)
        missing, unexpected = model.load_state_dict(load_file(str(BASE)), strict=False)
        assert not unexpected and all(
            k.startswith(("retina_conv.", "context.")) for k in missing
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        rng = np.random.default_rng(SEED)
        torch.cuda.reset_peak_memory_stats()
        start, losses = monotonic(), []
        for epoch in range(EPOCHS):
            order = rng.permutation(len(data.target))
            total, steps = 0.0, 0
            for begin in range(0, len(order), BATCH):
                i = data.tensor(order[begin : begin + BATCH])
                y = data.answer_id[i]
                keys = data.keys(model, arm, i)
                logits, _ = model(keys, data.history[y])
                allowed = data.gold[y]
                valid = allowed.any(-1)
                logits, allowed = logits[valid], allowed[valid]
                loss = (
                    logits.logsumexp(-1)
                    - logits.masked_fill(~allowed, -torch.inf).logsumexp(-1)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                total += loss.item() * len(logits)
                steps += len(logits)
            losses.append(total / steps)
            print(arm, epoch + 1, losses[-1], flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 8 * 2**30
            ):
                raise RuntimeError("Fit exceeded ceiling")
        weights = MODELS / (arm + ".safetensors")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            str(weights),
        )
        write_json(
            OUT / (arm + "_fit.json"),
            dict(
                parameters=sum(p.numel() for p in model.parameters()),
                seconds=monotonic() - start,
                preprocessing_seconds=preprocessing,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                losses=losses,
                updates=349 * EPOCHS,
                weight_sha=sha(weights),
                examples=len(data.target) * EPOCHS,
                registration_sha=sha(OUT / "registration.json"),
            ),
        )
        del model, optimizer


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["register", "fit"])
    if p.parse_args().action == "register":
        OUT.mkdir(exist_ok=False)
        write_json(OUT / "registration.json", registration())
    else:
        torch.set_num_threads(2)
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fit()
