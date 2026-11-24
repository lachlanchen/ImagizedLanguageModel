"""Fit equally exposed endpoint and whole-span rankers over frozen optical candidates."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ilm.visual_lm.fact_ranking import SpanRanker, candidate_loss, candidates
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_fact_channel import (
    MODEL as ORIGINAL,
    PUB as DATA,
    ROOT,
    dump,
    read,
    sha,
)
from scripts.prepare_fact_ranking_confirmation import PUB, WORK
from scripts.train_fact_channel import Data, load, join_regions, pack, SEED, gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

MODEL = ROOT / "models/fact_span_ranking"
MODES = ("endpoints", "whole")


def binding():
    sources = [
        "ilm/visual_lm/fact_ranking.py",
        "scripts/train_fact_ranking.py",
        "formal/span_list/FactRankMass.lean",
        "references/fact_span_reranking_protocol.md",
    ]
    return dict(
        reader=sha(ORIGINAL / "reader/weights.safetensors"),
        writer=sha(ORIGINAL / "human/weights.safetensors"),
        corpus=sha(DATA / "corpus.json"),
        reservation=sha(PUB / "reservation.json"),
        sources={p: sha(ROOT / p) for p in sources},
    )


def atomic_save(save, path):
    temp = path.with_suffix(".tmp")
    save(str(temp))
    temp.replace(path)


def check_budget(start, elapsed=0):
    if (
        monotonic() - start + elapsed > 10800
        or torch.cuda.max_memory_allocated() > 18 * 2**30
    ):
        raise RuntimeError("Registered resource ceiling exceeded")


def prepare():
    cache = WORK / "train-candidates.pt"
    receipt = PUB / "candidates.json"
    bound = binding()
    if receipt.exists() and cache.exists():
        rc = read(receipt)
        if rc["binding"] != bound or rc["sha256"] != sha(cache):
            raise ValueError("Candidate cache mismatch")
        return torch.load(cache, weights_only=True, mmap=True)
    data = Data("reader", "cuda")
    reader = load("reader", "cuda")
    rows = data.rows
    order = sorted(
        range(len(rows)), key=lambda i: sum(len(data.seq[k]) for k in rows[i][1])
    )
    shape = (len(rows), 32)
    bank = dict(
        features=torch.zeros((*shape, 3, 128), dtype=torch.float16),
        base=torch.full(shape, -torch.inf),
        lengths=torch.ones(shape, dtype=torch.long),
        correct=torch.zeros(shape, dtype=torch.bool),
    )
    counts = defaultdict(Counter)
    meter = snapshot()
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    for j in range(0, len(order), 24):
        ix = order[j : j + 24]
        pg = [join_regions(p) for p in data.pages(ix)]
        q = pack([data.seq[rows[i][0]] for i in ix], "cuda")
        feat, base, starts, sizes = candidates(
            reader, data.features, q, pack(pg, "cuda")
        )
        k = base.shape[1]
        bank["features"][ix, :k] = feat.cpu()
        bank["base"][ix, :k] = base.cpu()
        bank["lengths"][ix, :k] = sizes.cpu()
        for b, (i, page) in enumerate(zip(ix, pg, strict=True)):
            count = counts[rows[i][-2]]
            accepted = set(map(tuple, rows[i][2]))
            valid = bank["base"][i, :k].isfinite().numpy()
            locations = list(zip(starts[b].tolist(), sizes[b].tolist(), strict=True))
            copies = [
                tuple(page[s : s + n]) for (s, n), v in zip(locations, valid) if v
            ]
            hit = torch.tensor(
                [v and loc in accepted for loc, v in zip(locations, valid)],
                dtype=torch.bool,
            )
            bank["correct"][i, :k] = hit
            count.update(
                rows=1,
                supported=int(hit.any()),
                correct_positions=int(hit.sum()),
                positions=len(copies),
                duplicate_positions=len(copies) - len(set(copies)),
            )
        check_budget(start)
    torch.cuda.synchronize()
    seconds = monotonic() - start
    energy = consumed(meter, snapshot())["joules"]
    atomic_save(lambda p: torch.save(bank, p), cache)
    # Hash the actual tensors, independent of archive metadata on regeneration.
    hashes = {k: sha256(v.numpy().tobytes()).hexdigest() for k, v in bank.items()}
    if receipt.exists():
        old = read(receipt)
        if old["binding"] != bound or old["tensor_sha256"] != hashes:
            raise ValueError("Rebuilt candidates differ")
    dump(
        receipt,
        dict(
            binding=bound,
            sha256=sha(cache),
            tensor_sha256=hashes,
            counts=counts,
            seconds=seconds,
            joules=energy,
            peak_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    print("candidate coverage", dict(counts), flush=True)
    return bank


def train(mode, bank):
    out = MODEL / mode
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "weights.safetensors"
    receipt = out / "fit.json"
    bound = dict(
        **binding(),
        mode=mode,
        tensors=read(PUB / "candidates.json")["tensor_sha256"],
        seed=SEED,
    )
    if receipt.exists():
        rc = read(receipt)
        if rc["binding"] != bound or rc["weights"] != sha(dest):
            raise ValueError("Fit mismatch")
        return
    torch.manual_seed(SEED)
    net = SpanRanker(mode).cuda()
    initial = sha256(
        b"".join(v.detach().cpu().numpy().tobytes() for v in net.parameters())
    ).hexdigest()
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, fused=True)
    n = len(bank["base"])
    schedule = np.random.default_rng(SEED).permuted(
        np.tile(np.arange(n), (4, 1)), axis=1
    )
    resume = WORK / f"{mode}-resume.pt"
    history = []
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["binding"] != bound:
            raise ValueError("Resume mismatch")
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        history = state["history"]
    for epoch in range(len(history), 4):
        loss_sum = 0.0
        correct = seen = 0
        meter = snapshot()
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        for ix in np.array_split(schedule[epoch], (n + 127) // 128):
            inputs = [bank[k][ix].cuda() for k in ("features", "base", "lengths")]
            y = bank["correct"][ix].cuda()
            logits = net(*inputs)
            loss = candidate_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            supported = int(y.any(1).sum())
            seen += supported
            loss_sum += loss.item() * supported
            correct += int(y.gather(1, logits.argmax(1, keepdim=True)).sum())
            check_budget(start, sum(z["seconds"] for z in history))
        torch.cuda.synchronize()
        history.append(
            dict(
                epoch=epoch + 1,
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
                supported=seen,
                train_loss=loss_sum / seen,
                top_correct=correct,
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    binding=bound,
                    net=net.state_dict(),
                    opt=opt.state_dict(),
                    history=history,
                ),
                p,
            ),
            resume,
        )
        print(mode, history[-1], flush=True)
    atomic_save(
        lambda p: save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
        ),
        dest,
    )
    dump(
        receipt,
        dict(
            binding=bound,
            weights=sha(dest),
            initial=initial,
            order=sha256(schedule.tobytes()).hexdigest(),
            presentations=int(schedule.size),
            parameters=sum(p.numel() for p in net.parameters()),
            epochs=history,
        ),
    )


def restore(mode, device):
    root = MODEL / mode
    rc = read(root / "fit.json")
    if (
        rc["weights"] != sha(root / "weights.safetensors")
        or rc["binding"]["mode"] != mode
    ):
        raise ValueError("Ranker checkpoint mismatch")
    for arm, key in (("reader", "reader"), ("human", "writer")):
        if rc["binding"][key] != sha(ORIGINAL / arm / "weights.safetensors"):
            raise ValueError("Frozen model mismatch")
    ranker = SpanRanker(mode)
    ranker.load_state_dict(load_file(str(root / "weights.safetensors")))
    return ranker.to(device).eval()


if __name__ == "__main__":
    configure_cuda_cli()
    torch.set_num_threads(3)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modes", nargs="*", default=MODES)
    modes = parser.parse_args().modes
    if any(m not in MODES for m in modes):
        parser.error("Unknown ranker")
    with gpu_owner():
        bank = prepare()
        for mode in modes:
            train(mode, bank)
