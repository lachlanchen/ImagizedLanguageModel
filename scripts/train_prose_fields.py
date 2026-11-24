"""Fit optical and byte segment readers to the same human Chinese descriptions."""

import hashlib
from collections import defaultdict
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.prose_fields import ProseFields, interval_loss
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_chinese_card_reply import (
    Rasterizer,
    FIELDS,
    DOMAINS,
    read,
    sha,
    dump,
)
from scripts.prepare_prose_fields import BASE, PUBLIC, WEIGHTS, LIMIT, joined
from scripts.train_semantic_page import gpu_owner, snapshot, consumed


def train_rows():
    data = read(BASE / "train-passages.json")
    by = defaultdict(list)
    for i, r in enumerate(data):
        by[r["domain"], r["name"]].append(i)
    rng = np.random.default_rng(31277)
    rows = list(data)
    seen = set(r["text"] for r in rows)
    for i, r in enumerate(data):
        group = by[r["domain"], r["name"]]
        if len(group) < 2:
            continue
        for j in rng.choice(group, size=min(12, len(group)), replace=False):
            other = data[j]
            if r["dialog"] == other["dialog"] or set(r["spans"]) == set(other["spans"]):
                continue
            pair = joined((r, other))
            if pair and pair["text"] not in seen:
                rows.append(pair)
                seen.add(pair["text"])
                break
    return rows, len(data)


def glyph_bank(chars, raster):
    tiles = np.stack([np.frombuffer(raster.tile(c), np.uint8) for c in chars])
    by = np.zeros((len(chars), 4), np.uint8)
    for i, c in enumerate(chars):
        v = list(c.encode())
        by[i, : len(v)] = v
    return dict(tiles=tiles, bytes=by)


def load_model(mode, device, folder=WEIGHTS, receipts=PUBLIC):
    path = folder / (mode + ".safetensors")
    if sha(path) != read(receipts / (mode + "-fit.json"))["weights"]:
        raise ValueError("Changed span reader")
    net = ProseFields(len(FIELDS), len(DOMAINS)).to(device)
    net.load_state_dict(load_file(str(path), device=str(device)))
    return net.eval()


def prepare(rows):
    alphabet = [" "] + sorted(set("".join(r["text"] for r in rows)) - {" "})
    bank = glyph_bank(alphabet, Rasterizer())
    chars = {c: i for i, c in enumerate(alphabet)}
    x = np.zeros((len(rows), LIMIT), np.int16)
    y = np.full((len(rows), len(FIELDS), 2), -1, np.int16)
    sizes = []
    domains = []
    for i, r in enumerate(rows):
        s = r["text"]
        sizes.append(len(s))
        domains.append(list(DOMAINS).index(r["domain"]))
        x[i, : len(s)] = [chars[c] for c in s]
        for k, span in r["spans"].items():
            y[i, FIELDS.index(k)] = span
    return bank, [torch.as_tensor(v, device="cuda") for v in (x, y, sizes, domains)]


def fit(mode, bank, data, natural, model_dir, receipts):
    dest = receipts / (mode + "-fit.json")
    if dest.exists():
        load_model(mode, "cpu", model_dir, receipts)
        return
    torch.manual_seed(4181)
    rng = np.random.default_rng(66311)
    net = ProseFields(len(FIELDS), len(DOMAINS)).cuda()
    feats = torch.as_tensor(observations(bank, mode), device="cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=0.0012, weight_decay=0.01, fused=True)
    x, y, sizes, dom = data
    digest = hashlib.sha256()
    seen = set()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e = snapshot()
    start = monotonic()
    for step in range(2400):
        a = rng.integers(0, natural, size=24)
        b = rng.integers(natural, len(x), size=24)
        ix = np.concatenate((a, b))
        seen.update(ix.tolist())
        digest.update(ix.tobytes())
        lengths = sizes[ix]
        target = y[ix].long()
        pred = net(feats[x[ix, : int(lengths.max())].long()], lengths, dom[ix])
        loss = interval_loss(pred, lengths, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 1600:
            opt.param_groups[0]["lr"] = 0.0003
        if step % 200 == 0:
            print(mode, step, float(loss), flush=True)
            if (
                not torch.isfinite(loss)
                or monotonic() - start > 480
                or torch.cuda.max_memory_allocated() > 2 * 2**30
            ):
                raise RuntimeError("Fit exceeded registered ceiling")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    cost = consumed(e, snapshot())
    weight = model_dir / (mode + ".safetensors")
    tmp = weight.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(weight)
    dump(
        dest,
        dict(
            weights=sha(weight),
            seconds=elapsed,
            energy=cost,
            parameters=sum(p.numel() for p in net.parameters()),
            peak=torch.cuda.max_memory_allocated(),
            schedule=digest.hexdigest(),
            distinct=len(seen),
            updates=2400,
            natural=natural,
            assembled=len(x) - natural,
        ),
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str)
    args = ap.parse_args()
    from pathlib import Path

    model, receipts = (
        (Path(args.run) / "weights", Path(args.run) / "scores")
        if args.run
        else (WEIGHTS, PUBLIC)
    )
    model.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    rows, n = train_rows()
    with gpu_owner():
        bank, data = prepare(rows)
        for mode in ("ink", "byte"):
            fit(mode, bank, data, n, model, receipts)
    if (
        read(receipts / "ink-fit.json")["schedule"]
        != read(receipts / "byte-fit.json")["schedule"]
    ):
        raise ValueError("Unequal schedule")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
