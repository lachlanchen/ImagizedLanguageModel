"""Fit and release the independent span reader and three matched visual answer writers."""

import argparse
from collections import defaultdict
from hashlib import sha256
import math
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.fact_channel import OpticalFactReader
from ilm.visual_lm.region_writer import RegionWriter
from scripts.prepare_fact_channel import ROOT, WORK, PUB, MODEL
from scripts.prepare_open_exam import read, dump, sha
from scripts.train_region_writer import pack, join_regions, SEED, gpu_owner
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


class Data:
    def __init__(self, split, device):
        spec = read(PUB / "corpus.json")
        for file in (split + ".json", "tiles.npy", "strings.npz", "palette.npy"):
            if sha(WORK / file) != spec["files"][file]:
                raise ValueError(file)
        self.rows = read(WORK / (split + ".json"))
        with np.load(WORK / "strings.npz") as z:
            ids, ptr = z["ids"], z["ptr"]
        self.seq = [ids[a:b] for a, b in zip(ptr[:-1], ptr[1:])]
        self.n = spec["output_bitmaps"]
        self.features = torch.from_numpy(
            observations({"tiles": np.load(WORK / "tiles.npy")}, "ink")
        ).to(device)

    def pages(self, indices):
        return [[self.seq[b] for b in self.rows[i][1]] for i in indices]


def schedule(data, reader):
    bins = defaultdict(list)
    for i, r in enumerate(data.rows):
        length = sum(len(data.seq[b]) for b in r[1])
        for y in [-1] if reader else r[3]:
            bins[(length // 180, 0 if reader else len(data.seq[y]) // 48)].append(
                (i, y)
            )
    rng = np.random.default_rng(SEED)
    result = []
    for epoch in range(4):
        batches = []
        for (length, _), rs in bins.items():
            size = 24 if length > 2 else 48
            perm = rng.permutation(len(rs))
            batches.extend(
                [[rs[k] for k in perm[j : j + size]] for j in range(0, len(rs), size)]
            )
        result.extend(batches[i] for i in rng.permutation(len(batches)))
    return result


def load(arm, device, root=MODEL):
    out = root / arm
    rc = read(out / "fit.json")
    if sha(out / "weights.safetensors") != rc["weights"]:
        raise ValueError("Changed weights")
    net = OpticalFactReader() if arm == "reader" else RegionWriter()
    net.load_state_dict(load_file(str(out / "weights.safetensors")))
    return net.to(device).eval()


def split_span(regions, start, length):
    for ix, region in enumerate(regions):
        if start < len(region):
            if start < 0 or length <= 0 or start + length > len(region):
                raise ValueError("Span crossed a region edge")
            return ix, start, np.asarray(region[start : start + length], dtype=np.int64)
        start -= len(region) + 1
    raise ValueError("Missing selected span")


@torch.inference_mode()
def predict(reader, features, queries, regions, device):
    pages = [join_regions(rs) for rs in regions]
    scores = reader(features, pack(queries, device), pack(pages, device))
    k = scores.shape[-1]
    arg = scores.flatten(1).argmax(1).tolist()
    return [
        split_span(rs, code // k, code % k + 1)
        for rs, code in zip(regions, arg, strict=True)
    ]


def predictions(data, device):
    cache = WORK / "predictions.npy"
    receipt = PUB / "predictions.json"
    binding = dict(
        corpus=sha(PUB / "corpus.json"),
        reader=sha(MODEL / "reader/weights.safetensors"),
        code=sha(Path(__file__)),
    )
    if receipt.exists() and cache.exists():
        rc = read(receipt)
        if rc["binding"] != binding or rc["sha256"] != sha(cache):
            raise ValueError("Changed fact predictions")
        return np.load(cache)
    reader = load("reader", device)
    order = sorted(
        range(len(data.rows)),
        key=lambda i: sum(len(data.seq[k]) for k in data.rows[i][1]),
    )
    vals = np.zeros((len(order), 3), np.int32)
    torch.cuda.synchronize()
    meter = snapshot()
    start = monotonic()
    for j in range(0, len(order), 36):
        ix = order[j : j + 36]
        qs = [data.seq[data.rows[i][0]] for i in ix]
        spans = predict(reader, data.features, qs, data.pages(ix), device)
        vals[ix] = [(b, s, len(v)) for b, s, v in spans]
    torch.cuda.synchronize()
    rc = dict(
        binding=binding,
        seconds=monotonic() - start,
        joules=consumed(meter, snapshot())["joules"],
    )
    tmp = WORK / "predictions.tmp.npy"
    np.save(tmp, vals)
    rc["sha256"] = sha(tmp)
    if receipt.exists() and rc["sha256"] != read(receipt)["sha256"]:
        raise ValueError("Regenerated facts differ")
    tmp.replace(cache)
    if not receipt.exists():
        dump(receipt, rc)
    return vals


def writer_source(arm, regions, gold, prediction):
    if arm == "direct":
        return join_regions(regions)
    if arm == "human":
        return gold
    b, s, n = prediction
    fact = regions[b][s : s + n]
    return join_regions([fact, regions[b]])


def fit(arm):
    reader = arm == "reader"
    device = "cuda"
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    data = Data("reader" if reader else "writer", device)
    picks = predictions(data, device) if arm == "predicted" else None
    batches = schedule(data, reader)
    order_hash = sha256(
        np.asarray([x for b in batches for x in b], dtype="<i4").tobytes()
    ).hexdigest()
    out = MODEL / arm
    run = WORK / arm
    for folder in (out, run):
        folder.mkdir(exist_ok=True, parents=True)
    binding = dict(
        arm=arm,
        order=order_hash,
        corpus=sha(PUB / "corpus.json"),
        seed=SEED,
        reader_predictions=sha(PUB / "predictions.json") if picks is not None else None,
        source={
            f: sha(ROOT / f)
            for f in (
                "scripts/train_fact_channel.py",
                "scripts/prepare_fact_channel.py",
                "ilm/visual_lm/fact_channel.py",
                "ilm/visual_lm/region_writer.py",
                "formal/span_list/PicturedFactCollision.lean",
                "references/fact_channel_protocol.md",
            )
        },
    )
    if (out / "fit.json").exists():
        if read(out / "fit.json")["binding"] != binding:
            raise ValueError("Changed fit")
        load(arm, device)
        return
    net = (OpticalFactReader() if reader else RegionWriter()).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01, fused=True)
    first = 0
    elapsed = energy = 0.0
    peak = 0
    trace = []
    state = run / "resume.pt"
    if state.exists():
        z = torch.load(state, map_location="cpu", weights_only=True)
        if z["binding"] != binding:
            raise ValueError("Changed restart")
        net.load_state_dict(z["net"])
        opt.load_state_dict(z["opt"])
        first, elapsed, energy, peak, trace = [
            z[k] for k in ("step", "seconds", "joules", "peak", "trace")
        ]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    meter = snapshot()
    start = monotonic()
    elapsed0, energy0 = elapsed, energy
    for j in range(first, len(batches)):
        step = j + 1
        records = batches[j]
        ix = [i for i, _ in records]
        rs = [data.rows[i] for i in ix]
        q = pack([data.seq[r[0]] for r in rs], device)
        regions = data.pages(ix)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if reader:
                scores = net(
                    data.features, q, pack([join_regions(b) for b in regions], device)
                )
                k = scores.shape[-1]
                aliases = np.full((len(rs), max(len(r[2]) for r in rs)), -1, np.int64)
                for idx, r in enumerate(rs):
                    aliases[idx, : len(r[2])] = [s * k + n - 1 for s, n in r[2]]
                loss = net.loss(scores, torch.from_numpy(aliases).to(device)).mean()
            else:
                ps = [
                    writer_source(
                        arm, bs, data.seq[r[2]], None if picks is None else picks[i]
                    )
                    for i, r, bs in zip(ix, rs, regions)
                ]
                ctx, h = net.encode(data.features, data.n, q, pack(ps, device))
                ys = [np.r_[data.seq[y], np.int32(1)] for _, y in records]
                loss = net.loss(ctx, h, pack(ys, device)).mean()
        opt.param_groups[0]["lr"] = (
            0.001
            * min(1, step / 120)
            * (0.18 + 0.82 * (1 + math.cos(math.pi * step / len(batches))) / 2)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 250 == 0 or step == len(batches):
            trace.append([step, float(loss.detach())])
            print(arm, step, len(batches), trace[-1][1], flush=True)
        if step % 500 == 0 or step == len(batches):
            torch.cuda.synchronize()
            elapsed = elapsed0 + monotonic() - start
            energy = energy0 + consumed(meter, snapshot())["joules"]
            peak = max(peak, torch.cuda.max_memory_allocated())
            tmp = state.with_suffix(".tmp")
            torch.save(
                dict(
                    binding=binding,
                    net=net.state_dict(),
                    opt=opt.state_dict(),
                    step=step,
                    seconds=elapsed,
                    joules=energy,
                    peak=peak,
                    trace=trace,
                ),
                tmp,
            )
            tmp.replace(state)
            if step % 2500 == 0:
                save_file(
                    {k: v.cpu().contiguous() for k, v in net.state_dict().items()},
                    str(run / f"step-{step}.safetensors"),
                )
            if elapsed > 10800 or peak > 18 * 2**30:
                raise RuntimeError("Resource ceiling")
    f = out / "weights.safetensors"
    tmp = f.with_suffix(".tmp")
    save_file({k: v.cpu().contiguous() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(f)
    dump(
        out / "fit.json",
        dict(
            binding=binding,
            weights=sha(f),
            steps=len(batches),
            presentations=sum(map(len, batches)),
            seconds=elapsed,
            joules=energy,
            peak_bytes=peak,
            parameters=sum(p.numel() for p in net.parameters()),
            trace=trace,
        ),
    )


if __name__ == "__main__":
    configure_cuda_cli()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "arms", nargs="+", choices=["reader", "direct", "human", "predicted"]
    )
    with gpu_owner():
        for arm in p.parse_args().arms:
            fit(arm)
