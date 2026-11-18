"""Fit matched visual writers with positive phrase rehearsal and crossed-context ranking."""

import argparse
import fcntl
from hashlib import sha256
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional as F

from ilm.visual_lm.context_phrase import phrase_scores
from ilm.visual_lm.anchor_gap_writer import AnchorGapWriter
from ilm.visual_lm.idiom_ink import visual_input
from scripts.reserve_context_phrase_trial import EVIDENCE, ROOT, load, write_new
from scripts.train_visual_translation_coverage import (
    WEIGHTS as PREVIOUS,
    DEST as PRIOR,
    Rasterizer,
    sha,
    DATA,
)
from scripts.train_anchor_gap_writer import prepare

START = PREVIOUS / "broad.safetensors"
MODELS = PREVIOUS.parent / "context_phrases"
SEED = 712711
ARMS = ("rehearsal", "crossed")


def restore(path, device):
    state = load_file(str(path))
    m = AnchorGapWriter(state["landmarks"], state["exemplars"])
    m.load_state_dict(state)
    return m.to(device)


def pack(rows, raw):
    bank = list(map(bytes, raw))
    ids = {v: i for i, v in enumerate(bank)}
    render = Rasterizer()

    def encode(s):
        seq = []
        for c in s:
            px = render.tile(c)
            if px not in ids:
                ids[px] = len(bank)
                bank.append(px)
            seq.append(ids[px])
        return seq

    records = []
    for r in rows:
        src = list(map(encode, r["source"]))
        prefix = encode(r["prefix"])
        branches = list(map(encode, r["phrase"]))
        records.append((src, [prefix + v for v in branches], len(prefix)))
    return np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128), records


def four_scores(model, features, bank_ids, records):
    sources = []
    targets = []
    prefix = []
    for xs, ys, n in records:
        for x in xs:
            for y in ys:
                sources.append(x)
                targets.append(y)
                prefix.append(n)

    def padded(seq):
        a = np.zeros((len(seq), max(map(len, seq))), dtype=np.int64)
        for i, x in enumerate(seq):
            a[i, : len(x)] = x
        return torch.as_tensor(a, device=features.device)

    def tensor(v):
        return torch.tensor(v, device=features.device)

    return phrase_scores(
        model,
        features,
        bank_ids,
        padded(sources),
        tensor(list(map(len, sources))),
        padded(targets),
        tensor(list(map(len, targets))),
        tensor(prefix),
    ).reshape(-1, 2, 2)


def schedules(n):
    rng = np.random.default_rng(SEED)
    original = rng.permutation(55872).reshape(-1, 64)
    total = len(original) * 4
    supplemental = np.concatenate(
        [rng.permutation(n) for _ in range((total + n - 1) // n)]
    )[:total].reshape(-1, 4)
    return original, supplemental


def binding():
    files = [
        "training_pairs.json",
        "test_pairs.json",
        "holdout.json",
        "reading_roster.json",
    ]
    train = load(EVIDENCE / files[0])
    b, p = schedules(len(train))
    h = lambda a: sha256(a.astype("<i8").tobytes()).hexdigest()
    assert sha(START) == load(PRIOR / "broad_fit.json")["checkpoint_sha"]
    return dict(
        start_sha=sha(START),
        training_sha=sha(DATA / "training.npz"),
        files={k: sha(EVIDENCE / k) for k in files},
        book_schedule=h(b),
        contrast_schedule=h(p),
        seed=SEED,
        updates=len(b),
        pairs_per_update=4,
        lr=0.0001,
        clip=1,
        positive_weight=0.5,
        contrast_weight=0.5,
        time_limit_seconds=1800,
        vram_limit_gib=10,
        host_reserve_gib=24,
    )


def train():
    reg = binding()
    assert reg == load(EVIDENCE / "registration.json")
    packed, book = prepare(DATA / "training.npz")
    raw, records = pack(load(EVIDENCE / "training_pairs.json"), packed)
    idx = {bytes(v): i for i, v in enumerate(raw)}
    batches, extra = schedules(len(records))
    MODELS.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        target = MODELS / (arm + ".safetensors")
        if target.exists():
            raise FileExistsError(target)
        torch.manual_seed(SEED)
        m = restore(START, "cuda")
        lookup = torch.full((len(raw),), -1, dtype=torch.long)
        for j, b in enumerate(np.packbits(m.exemplars.cpu().numpy(), axis=1)):
            lookup[idx[bytes(b)]] = j
        keys = sorted(
            {v for _, ys, _ in records for y in ys for v in y}
            | set((lookup >= 0).nonzero().flatten().tolist())
        )
        lookup = lookup.cuda()
        bank = torch.tensor(keys, device="cuda")
        features = visual_input(
            torch.from_numpy(np.unpackbits(raw, axis=1)).cuda().bool(), m.landmarks
        )
        x, y, n, k, keep, l, r, o, c = (torch.as_tensor(a, device="cuda") for a in book)
        opt = torch.optim.AdamW(
            (v for v in m.parameters() if v.requires_grad), lr=reg["lr"]
        )
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        logs = []
        acc = np.zeros(3)
        for step, (batch, subset) in enumerate(zip(batches, extra), 1):
            ix = torch.as_tensor(batch, device="cuda")
            nx, ny = book[2][batch].max(), book[3][batch].max()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base = m.loss(
                    features,
                    x[ix, :nx].long(),
                    y[ix, :ny].long(),
                    n[ix],
                    k[ix],
                    keep[ix, :nx],
                    l[ix, : ny + 1].long(),
                    r[ix, : ny + 1].long(),
                    o[ix, : ny + 1].long(),
                    c[ix, : ny + 1],
                    lookup,
                )
                matrix = four_scores(m, features, bank, [records[i] for i in subset])
                pos = matrix.diagonal(dim1=1, dim2=2)
                neg = matrix.flip(-1).diagonal(dim1=1, dim2=2)
                rehearsal = -pos.mean()
                rank = F.softplus(neg - pos).mean()
                loss = (
                    base
                    + reg["positive_weight"] * rehearsal
                    + (reg["contrast_weight"] if arm == "crossed" else 0) * rank
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            acc += (
                torch.stack((base.detach(), rehearsal.detach(), rank.detach()))
                .float()
                .cpu()
                .numpy()
            )
            if step % 97 == 0:
                logs.append((acc / 97).tolist())
                acc.fill(0)
                mem = int(
                    next(
                        s.split()[1]
                        for s in Path("/proc/meminfo").read_text().splitlines()
                        if s.startswith("MemAvailable:")
                    )
                )
                if (
                    mem < 24 * 2**20
                    or monotonic() - start > 1800
                    or torch.cuda.max_memory_allocated() > 10 * 2**30
                ):
                    raise RuntimeError("Resource ceiling")
                print(arm, step, logs[-1], flush=True)
        save_file(
            {k: v.cpu().contiguous() for k, v in m.state_dict().items()}, str(target)
        )
        write_new(
            EVIDENCE / (arm + "_fit.json"),
            dict(
                sha=sha(target),
                registration_sha=sha(EVIDENCE / "registration.json"),
                steps=step,
                pairs_presented=extra.size,
                seconds=monotonic() - start,
                peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                losses_by_97=logs,
                bank_size=len(bank),
            ),
        )
        del m, opt, features, x, y, n, k, keep, l, r, o, c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "train"))
    args = parser.parse_args()
    if args.action == "register":
        write_new(EVIDENCE / "registration.json", binding())
    else:
        with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch.set_num_threads(2)
            torch.set_float32_matmul_precision("high")
            train()
