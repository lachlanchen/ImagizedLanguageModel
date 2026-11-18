"""Fit matched autoregressive writers using whole answer images and document fields."""

import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import save_file, load_file

from ilm.visual_lm.entry_span_writer import (
    EntryWriter,
    alignment,
    alternatives,
    cells,
    parse_card,
    VIS,
)
from scripts.audit_pictured_entry_answers import sequences
from scripts.prepare_pictured_entry_answers import (
    DEFAULT,
    Rasterizer,
    load,
    sha,
    write_json,
)

ROOT = DEFAULT.parents[1]
OUT = ROOT / "references/evidence/pictured_entry_writer_20261119"
MODELS = ROOT / "models/pictured_entry_writer"
SEED, EPOCHS, BATCH = 92113, 8, 64


def corpus(split):
    folder = DEFAULT / split
    report = load(DEFAULT / "report.json")["splits"][split]
    for name, ref in report["files"].items():
        if sha(folder / name) != ref:
            raise ValueError("Changed corpus: " + name)
    tiles = np.load(folder / "pixels.npy")
    names = ["cards", "questions", "answers"]
    if split == "development":
        names.append("question_paraphrases")
    return {name: sequences(folder, name, tiles) for name in names}


def assets():
    path = MODELS / "appearance.npz"
    with np.load(path) as f:
        return tuple(map(bytes, f["marks"])), list(map(bytes, f["literals"]))


def restored(arm, device):
    ckpt = MODELS / (arm + ".safetensors")
    if sha(ckpt) != load(OUT / (arm + "_fit.json"))["weight_sha"]:
        raise ValueError("Mismatched weights")
    m = EntryWriter().to(device)
    m.load_state_dict(load_file(str(ckpt)))
    return m.eval()


def schedule(n):
    rng = np.random.default_rng(SEED)
    return [rng.permutation(n) for _ in range(EPOCHS)]


def train():
    report = load(OUT / "registration.json")
    assert report == registration()
    train = corpus("train")
    marks = tuple(Rasterizer().tile(c) for c in "【】：；")
    freq = Counter(g for y in train["answers"] for g in cells(y))
    literals = [
        g for g, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))[:64]
    ]
    MODELS.mkdir()
    np.savez_compressed(
        MODELS / "appearance.npz",
        marks=np.asarray([np.frombuffer(x, np.uint8) for x in marks]),
        literals=np.asarray([np.frombuffer(x, np.uint8) for x in literals]),
    )
    cards = [parse_card(s, marks) for s in train["cards"]]
    options = [
        alternatives(cards[i // 8], q, literals)
        for i, q in enumerate(train["questions"])
    ]
    n = len(options)
    assert n == 8 * len(cards)
    feature = torch.from_numpy(np.stack([o.features for o in options])).cuda()
    acts = feature.shape[1] + 1
    order = schedule(n)
    rng = np.random.default_rng(SEED + 1)
    rotated = []
    for card in range(len(cards)):
        p = rng.permutation(8)
        while (p == np.arange(8)).any():
            p = rng.permutation(8)
        rotated.extend(8 * card + p)
    for arm, ids in (("deranged", rotated), ("matched", range(n))):
        p = MODELS / (arm + ".safetensors")
        if p.exists():
            raise FileExistsError(p)
        paths = [alignment(o, train["answers"][j]) for o, j in zip(options, ids)]
        steps = max(len(y) for _, y in paths)
        ys = np.zeros((n, steps, acts), bool)
        history = np.zeros((n, steps, VIS), np.float32)
        for i, (h, labels) in enumerate(paths):
            history[i, : len(h)] = h
            for t, choices in enumerate(labels):
                ys[i, t, choices] = True
        ys, history = (
            torch.as_tensor(ys, device="cuda"),
            torch.as_tensor(history, device="cuda"),
        )
        torch.manual_seed(SEED)
        m = EntryWriter().cuda()
        optimizer = torch.optim.AdamW(m.parameters(), lr=0.003)
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        losses = []
        for indices in order:
            epoch_loss, count = 0.0, 0
            for batch in np.array_split(indices, (n + BATCH - 1) // BATCH):
                b = torch.as_tensor(batch, device="cuda")
                logits, _ = m(m.encode(feature[b]), history[b])
                labels = ys[b]
                active = labels.any(-1)
                scores = logits[active]
                targets = labels[active]
                loss = (
                    scores.logsumexp(-1)
                    - scores.masked_fill(~targets, -torch.inf).logsumexp(-1)
                ).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    m.parameters(), 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                epoch_loss += loss.item() * len(scores)
                count += len(scores)
            losses.append(epoch_loss / count)
            print(arm, len(losses), losses[-1], flush=True)
            if (
                monotonic() - start > 1200
                or torch.cuda.max_memory_allocated() > 8 * 2**30
            ):
                raise RuntimeError("Fit exceeded registered resource ceiling")
        save_file({k: v.cpu().contiguous() for k, v in m.state_dict().items()}, str(p))
        write_json(
            OUT / (arm + "_fit.json"),
            dict(
                weight_sha=sha(p),
                assets_sha=sha(MODELS / "appearance.npz"),
                parameters=sum(v.numel() for v in m.parameters()),
                epochs=EPOCHS,
                examples=n * EPOCHS,
                seconds=monotonic() - start,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                losses=losses,
                registration_sha=sha(OUT / "registration.json"),
            ),
        )
        del ys, history, optimizer, m


def registration():
    return dict(
        data_sha=sha(DEFAULT / "report.json"),
        sources={
            p.name: sha(p)
            for p in (Path(__file__), ROOT / "ilm/visual_lm/entry_span_writer.py")
        },
        epochs=EPOCHS,
        batch=BATCH,
        seed=SEED,
        max_seconds_per_fit=1200,
        max_allocated_gib=8,
        primary_exact_rate=0.80,
        primary_all_eight_rate=0.65,
        matched_advantage=0.50,
        relocated_payload_rate=0.80,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("register", "fit"))
    args = p.parse_args()
    if args.action == "register":
        OUT.mkdir()
        write_json(OUT / "registration.json", registration())
    else:
        torch.set_num_threads(2)
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train()
