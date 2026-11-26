"""Equal-exposure visual translation from narrow and broad parallel reading."""

from collections import defaultdict
from time import monotonic
import math

import numpy as np
from safetensors.torch import save_file
import torch

from scripts.prepare_book_breadth import (
    BASE,
    DOMAINS,
    MODEL,
    PUB,
    ROOT,
    WORK,
    read,
    dump,
    sha,
)
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.train_ordered_translation import pack
from scripts.train_paired_ink import conditional, material
from scripts.train_span_choice import gpu_owner
from scripts.write_ordered_translation import restore
from scripts.verse_gpu_runtime import configure_cuda_cli


def matched_pool(lengths, domains):
    by_shape = defaultdict(list)
    for i, (domain, pair) in enumerate(
        zip(domains.tolist(), lengths.tolist(), strict=True)
    ):
        by_shape[(domain, pair[0], pair[1])].append(i)
    rng = np.random.default_rng(581229)
    mapping = np.arange(len(lengths))
    for group in by_shape.values():
        indices = rng.permutation(group)
        k = math.ceil(len(group) / 8)
        mapping[indices] = np.resize(indices[:k], len(group))
    if not np.array_equal(domains, domains[mapping]) or not np.array_equal(
        lengths, lengths[mapping]
    ):
        raise ValueError("Length or domain mismatch")
    return mapping


def schedule(lengths, epoch):
    rng = np.random.default_rng(92174 + epoch)
    buckets = (lengths[:, 0] // 12).astype(int) * 32 + lengths[:, 1] // 8
    order = rng.permutation(len(lengths))
    order = order[np.argsort(buckets[order], kind="stable")]
    batches = []
    for group in np.split(order, np.flatnonzero(np.diff(buckets[order])) + 1):
        batches.extend(group[j : j + 128] for j in range(0, len(group), 128))
    return [batches[i] for i in rng.permutation(len(batches))]


@torch.inference_mode()
def development(net, rowset, material_):
    features, inventory, local = material_
    net.eval()
    metrics = {}
    for domain in DOMAINS:
        rows = [r["cells"] for r in rowset if r["domain"] == domain]
        loss = 0.0
        n = 0
        for t in (0, 1):
            y = pack([r[t] + [1] for r in rows], "cuda")
            result = conditional(
                net,
                features,
                pack([r[1 - t] for r in rows], "cuda"),
                y,
                inventory[t],
                local[t],
                t,
            )
            loss -= float(result.sum())
            n += int(y.count_nonzero())
        metrics[domain] = loss / n
    return metrics


def fit(mode, data, lengths, mapping, dev, optical, binding):
    folder = MODEL / mode
    folder.mkdir(parents=True, exist_ok=True)
    receipt, weights = folder / "fit.json", folder / "weights.safetensors"
    if receipt.exists():
        existing = read(receipt)
        if existing["binding"] != binding or existing["weights"] != sha(weights):
            raise ValueError("Published book fit differs")
        return existing
    f, inv, local = optical
    torch.manual_seed(581291)
    net = restore("paraphrase-varied", "cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=0.00045, fused=True)
    ck = WORK / (mode + "-resume.pt")
    initial = development(net, dev, optical)
    best_score = np.mean(list(initial.values()))
    best_epoch = 0
    best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
    history = []
    if ck.exists():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        if state["binding"] != binding or state["mode"] != mode:
            raise ValueError("Resume mismatch")
        net.load_state_dict(state["current"])
        opt.load_state_dict(state["optimizer"])
        history, best, best_epoch, best_score = (
            state[k] for k in ("history", "best", "best_epoch", "best_score")
        )
        del state
    for ep in range(len(history), 6):
        net.train()
        start = monotonic()
        torch.cuda.reset_peak_memory_stats()
        opt.param_groups[0]["lr"] = [
            0.00045,
            0.00038,
            0.00032,
            0.00026,
            0.00020,
            0.00014,
        ][ep]
        exposure = 0
        loss_sum = 0
        batches = schedule(lengths, ep)
        for idx in batches:
            mx = lengths[idx].max(0)
            chosen = mapping[idx] if mode == "narrow" else idx
            pair = data[torch.as_tensor(chosen, device="cuda")].long()
            n = torch.as_tensor(lengths[idx].astype(np.int64), device="cuda")
            sides = [pair[:, t, : int(mx[t])] for t in (0, 1)]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = []
                for t in (0, 1):
                    y = torch.nn.functional.pad(sides[t], (0, 1)).scatter_(
                        1, n[:, t : t + 1], 1
                    )
                    losses.append(
                        -conditional(net, f, sides[1 - t], y, inv[t], local[t], t).sum()
                    )
                count = int(lengths[idx].sum()) + 2 * len(idx)
                loss = sum(losses) / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            opt.step()
            exposure += count
            loss_sum += float(loss.detach()) * count
        if exposure != int(lengths.sum()) + 2 * len(lengths):
            raise ValueError("Target exposure mismatch")
        d = development(net, dev, optical)
        score = float(np.mean(list(d.values())))
        if score < best_score:
            best_score, best_epoch = score, ep + 1
            best = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        seconds = monotonic() - start
        check_budget(start, sum(v["seconds"] for v in history))
        history.append(
            dict(
                pass_number=ep + 1,
                updates=len(batches),
                positions=exposure,
                nll=loss_sum / exposure,
                dev_nll=d,
                seconds=seconds,
                peak_bytes=torch.cuda.max_memory_allocated(),
            )
        )
        atomic_save(
            lambda p: torch.save(
                dict(
                    binding=binding,
                    mode=mode,
                    current=net.state_dict(),
                    optimizer=opt.state_dict(),
                    history=history,
                    best=best,
                    best_epoch=best_epoch,
                    best_score=float(best_score),
                ),
                p,
            ),
            ck,
        )
        print(mode, history[-1], "best", best_epoch, flush=True)
    atomic_save(lambda p: save_file(best, p), weights)
    counts = (
        np.bincount(mapping) if mode == "narrow" else np.ones(len(lengths), dtype=int)
    )
    probs = counts[counts > 0] / counts.sum()
    result = dict(
        binding=binding,
        weights=sha(weights),
        initial_dev_nll=initial,
        selected_pass=best_epoch,
        epochs=history,
        parameters=sum(p.numel() for p in net.parameters()),
        distinct_pairs=len(probs),
        presentation_entropy_bits=float(-(probs * np.log2(probs)).sum()),
        pair_presentations_per_pass=len(lengths),
    )
    dump(receipt, result)
    return result


def main():
    configure_cuda_cli()
    corpus = read(PUB / "corpus.json")
    for n, h in corpus["files"].items():
        if sha(WORK / n) != h:
            raise ValueError(n)
    binding = {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            ROOT / "scripts/train_book_breadth.py",
            ROOT / "scripts/prepare_book_breadth.py",
            ROOT / "references/book_breadth_protocol.md",
            ROOT / "ilm/visual_lm/ordered_translation.py",
            PUB / "corpus.json",
            ROOT / "models/translation_paraphrases/varied/weights.safetensors",
            BASE / "palette.npy",
            BASE / "inventories.json",
        )
    }
    cells = np.load(WORK / "cells.npy", mmap_mode="r")
    domains = np.load(WORK / "domains.npy")
    lengths = np.count_nonzero(cells, axis=2).astype(np.uint8)
    mapping = matched_pool(lengths, domains)
    dev = read(WORK / "dev.json")
    with gpu_owner():
        if torch.cuda.mem_get_info()[0] < 6 * 2**30:
            raise RuntimeError("Book writer requires 6 GiB free on GPU0")
        optical = material("cuda")
        data = torch.as_tensor(cells.astype(np.int32), device="cuda")
        results = [
            fit(mode, data, lengths, mapping, dev, optical, binding)
            for mode in ("narrow", "broad")
        ]
    for key in ("positions", "updates"):
        assert [r[key] for r in results[0]["epochs"]] == [
            r[key] for r in results[1]["epochs"]
        ]


if __name__ == "__main__":
    main()
