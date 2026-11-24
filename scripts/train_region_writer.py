"""Fit the shared region reader and matched bilingual full-answer writers."""

import argparse
from collections import defaultdict
from hashlib import sha256
import math
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.region_writer import RegionSelector, RegionWriter
from scripts.prepare_region_writer import ROOT, WORK, PUB, MODEL
from scripts.prepare_open_exam import sha, read, dump
from scripts.measure_writer_energy import snapshot, consumed
from scripts.train_paired_evidence import gpu_owner

SEED = 4261125


def pack(rows, device):
    a = np.zeros((len(rows), max(map(len, rows))), np.int64)
    for i, r in enumerate(rows):
        a[i, : len(r)] = r
    return torch.from_numpy(a).to(device)


def selector_input(qs, regions, device):
    owners = np.repeat(np.arange(len(qs)), list(map(len, regions)))
    off = np.full((len(qs), max(map(len, regions))), -1, np.int64)
    flat = []
    for i, rows in enumerate(regions):
        off[i, : len(rows)] = np.arange(len(flat), len(flat) + len(rows))
        flat.extend(rows)
    return (
        pack(qs, device),
        pack(flat, device),
        torch.from_numpy(owners).to(device),
        torch.from_numpy(off).to(device),
    )


def join_regions(rows):
    # Padding is a physical boundary in the attention mask. It is never copied.
    return np.concatenate(
        [r if i == 0 else np.r_[np.int32(0), r] for i, r in enumerate(rows)]
    )


class Corpus:
    def __init__(self, split, device):
        spec = read(PUB / "corpus.json")
        for f in ("tiles.npy", "strings.npz", split + ".json", "palette.npy"):
            if sha(WORK / f) != spec["files"][f]:
                raise ValueError(f)
        self.rows = read(WORK / (split + ".json"))
        with np.load(WORK / "strings.npz") as z:
            idx = z["ptr"]
            ids = z["ids"]
        self.seq = [ids[i:j] for i, j in zip(idx[:-1], idx[1:])]
        tiles = np.load(WORK / "tiles.npy")
        self.features = torch.from_numpy(observations({"tiles": tiles}, "ink")).to(
            device
        )
        self.n = spec["output_bitmaps"]
        if not np.array_equal(tiles[: self.n], np.load(WORK / "palette.npy")):
            raise ValueError("Inconsistent physical bank")

    def lengths(self, i):
        q, p, _, _, _, _ = self.rows[i]
        return len(self.seq[q]) + sum(len(self.seq[x]) for x in p)


def batches(data, kind):
    rng = np.random.default_rng(SEED)
    records = (
        [(i, -1) for i in range(len(data.rows))]
        if kind == "selector"
        else [(i, y) for i, r in enumerate(data.rows) for y in r[3]]
    )
    bins = defaultdict(list)
    lengths = [data.lengths(i) for i in range(len(data.rows))]
    for i, y in records:
        bins[lengths[i] // 240, 0 if y < 0 else len(data.seq[y]) // 48].append((i, y))
    schedule = []
    for _ in range(2 if kind == "selector" else 4):
        epoch = []
        for (plen, _), values in bins.items():
            size = (
                (20 if plen >= 3 else 40)
                if kind == "selector"
                else (24 if plen >= 3 else 48)
            )
            order = rng.permutation(len(values))
            epoch.extend(
                [values[j] for j in order[k : k + size]]
                for k in range(0, len(values), size)
            )
        schedule.extend(epoch[j] for j in rng.permutation(len(epoch)))
    return schedule


def selection(net, data, indices, device):
    qs = [data.seq[data.rows[i][0]] for i in indices]
    blocks = [[data.seq[j] for j in data.rows[i][1]] for i in indices]
    x = selector_input(qs, blocks, device)
    return net(data.features, *x)


@torch.inference_mode()
def select_all(net, data, device):
    out = np.zeros(len(data.rows), np.int32)
    ordered = sorted(range(len(out)), key=data.lengths)
    for j in range(0, len(out), 24):
        indices = ordered[j : j + 24]
        out[indices] = selection(net, data, indices, device).argmax(-1).cpu().numpy()
    return out


def fit(arm):
    out = MODEL / arm
    run = WORK / arm
    out.mkdir(parents=True, exist_ok=True)
    run.mkdir(parents=True, exist_ok=True)
    select = arm == "selector"
    with gpu_owner():
        device = "cuda"
        torch.manual_seed(SEED)
        torch.set_float32_matmul_precision("high")
        data = Corpus("train", device)
        schedule = batches(data, arm)
        prior = MODEL / "selector" / "weights.safetensors"
        binding = dict(
            corpus=sha(PUB / "corpus.json"),
            arm=arm,
            seed=SEED,
            model=sha(ROOT / "ilm/visual_lm/region_writer.py"),
            train=sha(Path(__file__)),
            protocol=sha(ROOT / "references/region_writer_protocol.md"),
            selector=None if select else sha(prior),
            order=sha256(
                np.asarray([r for b in schedule for r in b], dtype="<i4").tobytes()
            ).hexdigest(),
        )
        weight = out / "weights.safetensors"
        receipt = out / "fit.json"
        if receipt.exists():
            old = read(receipt)
            if old["binding"] != binding or sha(weight) != old["weights"]:
                raise ValueError("Changed completed run")
            return
        picks = None
        if not select:
            m = read(MODEL / "selector" / "fit.json")
            f = WORK / "selector" / "train-regions.npy"
            if sha(f) != m["train_region_hash"] or sha(prior) != m["weights"]:
                raise ValueError("Changed trained region selector")
            picks = np.load(f)
        net = (RegionSelector() if select else RegionWriter()).cuda()
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.001, weight_decay=0.01, fused=True
        )
        resume = run / "resume.pt"
        start = 0
        sec0 = energy0 = 0.0
        peak = 0
        trace = []
        if resume.exists():
            z = torch.load(resume, map_location="cpu", weights_only=True)
            if z["binding"] != binding:
                raise ValueError("Changed resume binding")
            net.load_state_dict(z["net"])
            opt.load_state_dict(z["opt"])
            start = z["step"]
            sec0 = z["seconds"]
            energy0 = z["joules"]
            trace = z["trace"]
            peak = z["peak"]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = monotonic()
        meter = snapshot()
        for j in range(start, len(schedule)):
            step = j + 1
            records = schedule[j]
            ix = [i for i, _ in records]
            rows = [data.rows[i] for i in ix]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if select:
                    scores = selection(net, data, ix, device).float()
                    valid = torch.zeros_like(scores, dtype=torch.bool)
                    jj, kk = zip(*[(k, p) for k, r in enumerate(rows) for p in r[2]])
                    valid[list(jj), list(kk)] = True
                    loss = (
                        scores.logsumexp(1)
                        - scores.masked_fill(~valid, -1e4).logsumexp(1)
                    ).mean()
                else:
                    qs = [data.seq[r[0]] for r in rows]
                    ps = [
                        join_regions([data.seq[k] for k in r[1]])
                        if arm == "direct"
                        else data.seq[r[1][picks[i]]]
                        for i, r in zip(ix, rows)
                    ]
                    ys = [np.r_[data.seq[y], np.int32(1)] for _, y in records]
                    ctx, h = net.encode(
                        data.features, data.n, pack(qs, device), pack(ps, device)
                    )
                    loss = net.loss(ctx, h, pack(ys, device)).mean()
            opt.param_groups[0]["lr"] = (
                (0.0012 if select else 0.001)
                * min(1, step / 120)
                * (0.18 + 0.82 * (1 + math.cos(math.pi * step / len(schedule))) / 2)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            if step == 1 or step % 250 == 0 or step == len(schedule):
                trace.append([step, float(loss.detach()), sec0 + monotonic() - t0])
                print(arm, step, len(schedule), trace[-1], flush=True)
            if step % 400 == 0 or step == len(schedule):
                torch.cuda.synchronize()
                sec = sec0 + monotonic() - t0
                energy = energy0 + consumed(meter, snapshot())["joules"]
                peak = max(peak, torch.cuda.max_memory_allocated())
                tmp = resume.with_suffix(".tmp")
                torch.save(
                    dict(
                        binding=binding,
                        step=step,
                        seconds=sec,
                        joules=energy,
                        peak=peak,
                        trace=trace,
                        net=net.state_dict(),
                        opt=opt.state_dict(),
                    ),
                    tmp,
                )
                tmp.replace(resume)
                if sec > 10800 or peak > 18 * 2**30:
                    raise RuntimeError("Resource ceiling exceeded")
        if start == len(schedule):
            sec, energy = sec0, energy0
        extra = {}
        if select:
            picks = select_all(net.eval(), data, device)
            f = run / "train-regions.npy"
            np.save(f, picks)
            extra = dict(
                train_region_hash=sha(f),
                train_region_correct=sum(
                    int(p) in r[2] for p, r in zip(picks, data.rows)
                ),
            )
        tmp = weight.with_suffix(".tmp")
        save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(tmp)
        )
        tmp.replace(weight)
        dump(
            receipt,
            dict(
                binding=binding,
                weights=sha(weight),
                steps=len(schedule),
                presentations=sum(map(len, schedule)),
                seconds=sec,
                joules=energy,
                peak_bytes=peak,
                parameters=sum(t.numel() for t in net.parameters()),
                trace=trace,
                **extra,
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=("selector", "direct", "selected"))
    fit(p.parse_args().arm)
