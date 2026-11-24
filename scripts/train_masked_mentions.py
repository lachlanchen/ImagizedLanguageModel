"""Fit attribution on the visually anonymized context, keeping source locations separate."""

import hashlib
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.masked_mentions import MaskedMentions
from scripts.prepare_story_mentions import MODEL, PUBLIC, LANGS, CELLS, read, dump, sha
from scripts.train_story_mentions import data, load
from scripts.story_mention_options import symbols, proposals, collapse
from scripts.train_semantic_page import gpu_owner, snapshot, consumed


@torch.inference_mode()
def prepare(split, mode, device="cuda"):
    rows, bank, (src, query, lengths, _, _) = data(split, device)
    obs = torch.as_tensor(observations(bank, mode), device=device)
    locator = load(mode, device)
    compressed = []
    output = {}
    for first in range(0, len(rows), 32):
        ix = slice(first, first + 32)
        vals = locator(obs[src[ix].long()], query[ix], lengths[ix])[0].cpu().numpy()
        codes = src[ix].cpu().numpy()
        for j, row in enumerate(rows[ix]):
            n = len(row["text"])
            q = row["query"]
            raw = codes[j, :n]
            physical = symbols(bank["tiles"][raw])
            good = {physical[a:b] for a, b, e in row["names"] if e == row["entity"]}
            for route in ("free", "cast"):
                candidates = (
                    proposals(vals[j], q[0])
                    if route == "free"
                    else [(a, b) for a, b, e in row["names"]]
                )
                indices, slots, spans, marked = collapse(physical, q, candidates)
                earlier = [s >= 0 and i < marked[0] for i, s in enumerate(slots)]
                truth = [
                    ok and physical[a:b] in good for ok, (a, b) in zip(earlier, spans)
                ]
                compressed.append((raw[indices], slots, truth, marked))
                output[first + j, route] = (slots, spans, marked)
    m = len(compressed)
    ids = np.zeros((m, CELLS), np.int32)
    slots = np.full((m, CELLS), -1, np.int16)
    truth = np.zeros((m, CELLS + 1), bool)
    bounds = np.empty((m, 2), np.int32)
    size = np.empty(m, np.int32)
    for i, (raw, s, t, q) in enumerate(compressed):
        n = len(raw)
        ids[i, :n] = raw
        slots[i, :n] = s
        truth[i, :n] = t
        truth[i, -1] = not any(t)
        bounds[i] = q
        size[i] = n
    packs = tuple(
        torch.as_tensor(x, device=device) for x in (ids, slots, truth, bounds, size)
    )
    return rows, bank, obs, packs, output


def net_for(arm, device):
    file = MODEL / ("masked-" + arm + ".safetensors")
    if sha(file) != read(PUBLIC / ("masked-" + arm + "-fit.json"))["sha256"]:
        raise ValueError("Changed masked reader")
    net = MaskedMentions().to(device)
    net.load_state_dict(load_file(str(file), device=device))
    return net.eval()


def train(arm, rows, obs, packs):
    name = "masked-" + arm
    path = MODEL / (name + ".safetensors")
    receipt = PUBLIC / (name + "-fit.json")
    if receipt.exists():
        net_for(arm, "cpu")
        return
    ids, slots, targets, bounds, lengths = packs
    pools = [np.array([i for i, r in enumerate(rows) if r["lang"] == l]) for l in LANGS]
    torch.manual_seed(6199)
    rng = np.random.default_rng(52491)
    model = MaskedMentions().cuda()
    opt = torch.optim.AdamW(
        model.parameters(), lr=0.0006, weight_decay=0.05, fused=True
    )
    dig = hashlib.sha256()
    trace = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    energy = snapshot()
    start = monotonic()
    for step in range(1, 1201):
        ix = np.concatenate(
            [2 * rng.choice(pool, 16) + np.tile([0, 1], 8) for pool in pools]
        )
        dig.update(ix.tobytes())
        feats = obs[ids[ix].long()]
        if arm == "geometry":
            feats = torch.zeros_like(feats)
        score = model(feats, slots[ix], bounds[ix], lengths[ix])
        loss = (
            score.logsumexp(1) - score.masked_fill(~targets[ix], -1e4).logsumexp(1)
        ).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 800:
            opt.param_groups[0]["lr"] = 0.00015
        if step == 1 or step % 200 == 0:
            x = loss.item()
            trace.append((step, x))
            print(name, step, x, flush=True)
            if (
                not np.isfinite(x)
                or monotonic() - start > 720
                or torch.cuda.max_memory_allocated() > 4 * 2**30
            ):
                raise RuntimeError("Resource or numerical limit exceeded")
    torch.cuda.synchronize()
    elapsed = monotonic() - start
    cost = consumed(energy, snapshot())
    tmp = path.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in model.state_dict().items()}, str(tmp))
    tmp.replace(path)
    dump(
        receipt,
        dict(
            sha256=sha(path),
            data=read(PUBLIC / "corpus.json")["splits"]["train"]["sha256"],
            proposer=read(
                PUBLIC / (("byte" if arm == "byte" else "ink") + "-fit.json")
            )["sha256"],
            steps=1200,
            batch=32,
            schedule=dig.hexdigest(),
            seconds=elapsed,
            energy=cost,
            peak=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
            trace=trace,
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner():
        for mode in ("ink", "byte"):
            rows, _, obs, packs, _ = prepare("train", mode)
            train(mode, rows, obs, packs)
            if mode == "ink":
                train("geometry", rows, obs, packs)
