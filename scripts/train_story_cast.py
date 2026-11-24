"""Train cast-specific optical readers with authentic Chinese and English coreference questions."""

from collections import defaultdict
import hashlib
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from ilm.visual_lm.masked_mentions import MaskedMentions
from ilm.visual_lm.book_prefix import observations
from scripts.check_mention_transfer import datasets
from scripts.prepare_story_mentions import (
    LANGS,
    MODEL,
    PUBLIC,
    RUN,
    CELLS,
    read,
    dump,
    sha,
)
from scripts.prepare_visual_sequence_corpus import Rasterizer
from scripts.train_prose_fields import glyph_bank
from scripts.train_semantic_page import gpu_owner, snapshot, consumed
from scripts.story_mention_options import collapse, symbols


def partitions():
    older, _ = datasets()
    incoming, exclusions = datasets(training=True)
    out = {}
    evidence = {}
    for name, new in incoming.items():
        old = older[name]
        rows = old + new
        parents = list(range(len(rows)))

        def root(i):
            while i != parents[i]:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        seen = {}
        for i, r in enumerate(rows):
            if name == "gap":
                keys = [r["article"]]
            else:
                s = "".join(r["original"].casefold().split())
                keys = (
                    [s] if len(s) <= 32 else [s[k : k + 32] for k in range(len(s) - 31)]
                )
            for k in keys:
                if k in seen:
                    parents[root(i)] = root(seen[k])
                else:
                    seen[k] = i
        prior = {root(i) for i in range(len(old))}
        groups = defaultdict(list)
        for i, r in enumerate(new, len(old)):
            groups[root(i)].append(r)
        training = []
        reserved = []
        excluded = 0
        ids = {}
        for g, rs in groups.items():
            if g in prior:
                excluded += len(rs)
                continue
            key = min(r.get("article") or "".join(r["original"].split()) for r in rs)
            digest = hashlib.sha256(key.encode()).hexdigest()
            part = reserved if int(digest, 16) % 5 == 0 else training
            part.extend(rs)
            for r in rs:
                ids[r["id"]] = dict(
                    group=digest, split="reserved" if part is reserved else "train"
                )
        out[name] = (training, reserved)
        evidence[name] = dict(
            official_overlap_excluded=excluded,
            ineligible=exclusions[name],
            train=len(training),
            reserved=len(reserved),
            ids=ids,
        )
    return out, evidence


def story_rows(split):
    out = {l: [] for l in LANGS}
    for r in read(RUN / (split + ".json")):
        noms = [(a, b) for a, b, e in r["names"]]
        good = [(a, b) for a, b, e in r["names"] if e == r["entity"]]
        out[r["lang"]].append(
            dict(text=r["text"], query=r["query"], cast=noms, good=good)
        )
    return out


def prepare(rows, mode, device):
    chars = sorted(set(" " + "".join(r["text"] for r in rows)))
    lookup = {c: i for i, c in enumerate(chars)}
    bank = glyph_bank(chars, Rasterizer())
    obs = torch.as_tensor(observations(bank, mode), device=device)
    n = len(rows)
    ids = np.zeros((n, CELLS), np.int32)
    slots = np.full_like(ids, -1)
    bounds = np.empty((n, 2), np.int32)
    size = np.empty(n, np.int32)
    truth = np.zeros((n, CELLS + 1), bool)
    items = []
    for i, r in enumerate(rows):
        raw = np.array([lookup[c] for c in r["text"]])
        s = symbols(bank["tiles"][raw])
        index, slot, spans, q = collapse(s, r["query"], r["cast"])
        m = len(slot)
        good = r.get("good")
        if good is None:
            good = [] if r["truth"] is None else [r["cast"][r["truth"]]]
        gold = {s[a:b] for a, b in good}
        ids[i, :m] = raw[index]
        slots[i, :m] = slot
        bounds[i] = q
        size[i] = m
        truth[i, :m] = [
            j < q[0] and v >= 0 and s[a:b] in gold
            for j, (v, (a, b)) in enumerate(zip(slot, spans))
        ]
        truth[i, -1] = not truth[i, :m].any()
        items.append((slot, spans, q))
    return (
        obs,
        tuple(
            torch.as_tensor(x, device=device) for x in (ids, slots, bounds, size, truth)
        ),
        items,
    )


def load_cast(arm, device):
    path = MODEL / ("cast-" + arm + ".safetensors")
    if sha(path) != read(PUBLIC / ("cast-" + arm + "-fit.json"))["sha256"]:
        raise ValueError("Changed cast checkpoint")
    net = MaskedMentions().to(device)
    net.load_state_dict(load_file(str(path), device=device))
    return net.eval()


def fit(arm, obs, packs, groups):
    mode = "cast-" + arm
    receipt = PUBLIC / (mode + "-fit.json")
    path = MODEL / (mode + ".safetensors")
    if receipt.exists():
        if (
            sha(path) != read(receipt)["sha256"]
            or sha(PUBLIC / "cast-partitions.json") != read(receipt)["partition"]
        ):
            raise ValueError("Changed cast fit")
        return
    ids, slots, bounds, size, truth = packs
    rng = np.random.default_rng(52491)
    torch.manual_seed(6199)
    net = MaskedMentions().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0006, weight_decay=0.05, fused=True)
    digest = hashlib.sha256()
    trace = []
    draws = hashlib.sha256()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e = snapshot()
    start = monotonic()
    for step in range(1, 1201):
        u = rng.random((4, 8))
        draws.update(u.tobytes())
        parts = []
        for j, (x, group) in enumerate(zip(u, groups)):
            if arm == "control" and j % 2:
                group = groups[j - 1]
            parts.append(group[(x * len(group)).astype(int)])
        ix = np.concatenate(parts)
        digest.update(ix.tobytes())
        features = obs[ids[ix].long()]
        if arm == "geometry":
            features = torch.zeros_like(features)
        logits = net(features, slots[ix], bounds[ix], size[ix])
        loss = (
            logits.logsumexp(1) - logits.masked_fill(~truth[ix], -1e4).logsumexp(1)
        ).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 800:
            opt.param_groups[0]["lr"] = 0.00015
        if step == 1 or step % 200 == 0:
            value = loss.item()
            print(mode, step, value, flush=True)
            trace.append((step, value))
            if (
                not np.isfinite(value)
                or monotonic() - start > 720
                or torch.cuda.max_memory_allocated() > 4 * 2**30
            ):
                raise RuntimeError("Fit exceeded limits")
    torch.cuda.synchronize()
    seconds = monotonic() - start
    cost = consumed(e, snapshot())
    tmp = path.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(path)
    dump(
        receipt,
        dict(
            sha256=sha(path),
            partition=sha(PUBLIC / "cast-partitions.json"),
            steps=1200,
            batch=32,
            draws=draws.hexdigest(),
            schedule=digest.hexdigest(),
            seconds=seconds,
            energy=cost,
            peak=torch.cuda.max_memory_allocated(),
            trace=trace,
            parameters=sum(p.numel() for p in net.parameters()),
        ),
    )


def main():
    other, partition = partitions()
    dump(PUBLIC / "cast-partitions.json", partition)
    books = story_rows("train")
    rows = []
    groups = []
    for lang, external in zip(LANGS, ("clue", "gap")):
        for sample in (books[lang], other[external][0]):
            groups.append(np.arange(len(rows), len(rows) + len(sample)))
            rows.extend(sample)
    with gpu_owner():
        for mode in ("ink", "byte"):
            obs, packs, _ = prepare(rows, mode, "cuda")
            for arm in ["control", "ink", "geometry"] if mode == "ink" else ["byte"]:
                fit(arm, obs, packs, groups)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
