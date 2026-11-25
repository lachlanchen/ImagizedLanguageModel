"""Fit a matched likelihood control and a source-conditioned first-divergence translator."""

from time import monotonic
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_proposition_translation import (
    ROOT,
    BASE,
    WORK,
    PUB,
    MODEL,
    read,
    dump,
    sha,
)
from scripts.train_ordered_translation import pack
from scripts.write_ordered_translation import restore
from scripts.train_span_choice import gpu_owner
from scripts.train_fact_ranking import atomic_save, check_budget
from scripts.measure_writer_energy import snapshot, consumed
from scripts.verse_gpu_runtime import configure_cuda_cli


def divergence(left, right):
    """Both streams include the physical stop action, so a proper prefix diverges at stop."""
    return next(k for k, (a, b) in enumerate(zip(left, right)) if a != b)


def schedule(contrasts, replay, epoch):
    rng = np.random.default_rng(87311 + epoch)
    events = []
    for t in (0, 1):
        order = rng.permutation(len(contrasts))
        for low in range(0, len(order), 32):
            records = [contrasts[i]["cells"] for i in order[low : low + 32]]
            events.append((t, True, [f for pair in records for f in pair]))
        items = []
        for pool in replay:
            items.extend(
                pool[i] for i in rng.choice(len(pool), len(contrasts), replace=False)
            )
        # Bucket replay by length to avoid wasting memory on padded images.
        items.sort(key=lambda p: max(map(len, p)) // 12)
        events.extend((t, False, items[i : i + 64]) for i in range(0, len(items), 64))
    return [events[i] for i in rng.permutation(len(events))]


def main():
    configure_cuda_cli()
    files = read(PUB / "design.json")["files"]
    for k, h in files.items():
        if sha(WORK / (k + ".json")) != h:
            raise ValueError("Corpus changed: " + k)
    binds = {
        p: sha(ROOT / p)
        for p in (
            "scripts/train_proposition_translation.py",
            "scripts/prepare_proposition_translation.py",
            "references/proposition_translation_protocol.md",
            "references/evidence/proposition_translation_v1/design.json",
            "models/ordered_translation/news/weights.safetensors",
            "ilm/visual_lm/ordered_translation.py",
        )
    }
    contrasts = read(WORK / "contrasts.json")
    replay = read(WORK / "replay.json")
    with gpu_owner():
        torch.set_num_threads(2)
        f = torch.tensor(
            observations({"tiles": np.load(BASE / "palette.npy")}, "ink"), device="cuda"
        )
        inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
        loc = [torch.full((len(f),), -100, device="cuda") for _ in inv]
        for a, v in zip(loc, inv):
            a[v] = torch.arange(len(v), device="cuda")
        for mode in ("likelihood", "contrast"):
            path = MODEL / mode
            path.mkdir(parents=True, exist_ok=True)
            receipt = path / "fit.json"
            weights = path / "weights.safetensors"
            if receipt.exists():
                if (
                    read(receipt)["binding"] != binds
                    or sha(weights) != read(receipt)["weights"]
                ):
                    raise ValueError("Fit differs")
                continue
            torch.manual_seed(632075)
            net = restore("news", "cuda").train()
            opt = torch.optim.AdamW(net.parameters(), lr=0.00023, fused=True)
            ck = WORK / (mode + "-resume.pt")
            hist = []
            if ck.exists():
                d = torch.load(ck, weights_only=True, map_location="cpu")
                if d["binds"] != binds:
                    raise ValueError("Checkpoint differs")
                net.load_state_dict(d["net"])
                opt.load_state_dict(d["opt"])
                hist = d["hist"]
            for ep in range(len(hist), 6):
                data = schedule(contrasts, replay, ep)
                start = monotonic()
                meter = snapshot()
                torch.cuda.reset_peak_memory_stats()
                total = divtotal = 0.0
                positions = examples = 0
                observations_count = 0
                for t, paired, rows in data:
                    opt.zero_grad(set_to_none=True)
                    src = pack([p[1 - t] for p in rows], "cuda")
                    targets = [p[t] + [1] for p in rows]
                    y = pack(targets, "cuda")
                    local = loc[t][y]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(f, src, y, inv[t], t)
                        ce = F.cross_entropy(
                            logits.flatten(0, 1), local.flatten(), ignore_index=-100
                        )
                        loss = ce
                        if paired:
                            first = [
                                divergence(a, b)
                                for a, b in zip(targets[::2], targets[1::2])
                            ]
                            k = torch.tensor(first, device="cuda").repeat_interleave(2)
                            nr = torch.arange(len(rows), device="cuda")
                            alt = nr ^ 1
                            head = logits[nr, k].float()
                            margins = head[nr, local[nr, k]] - head[nr, local[alt, k]]
                            pairloss = F.softplus(1 - margins).mean()
                            if mode == "contrast":
                                loss = ce + 0.15 * pairloss
                            divtotal += float(pairloss.detach()) * len(rows)
                            observations_count += len(rows)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        net.parameters(), 1, error_if_nonfinite=True
                    )
                    opt.step()
                    n = int((y != 0).sum())
                    total += float(ce.detach()) * n
                    positions += n
                    examples += len(rows)
                torch.cuda.synchronize()
                check_budget(start, sum(d["seconds"] for d in hist))
                hist.append(
                    dict(
                        epoch=ep + 1,
                        examples=examples,
                        positions=positions,
                        nll=total / positions,
                        contrast_loss=divtotal / observations_count,
                        seconds=monotonic() - start,
                        joules=consumed(meter, snapshot())["joules"],
                        peak=torch.cuda.max_memory_allocated(),
                    )
                )
                atomic_save(
                    lambda p: torch.save(
                        dict(
                            binds=binds,
                            net=net.state_dict(),
                            opt=opt.state_dict(),
                            hist=hist,
                        ),
                        p,
                    ),
                    ck,
                )
                print(mode, hist[-1], flush=True)
            atomic_save(
                lambda p: save_file(
                    {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, p
                ),
                weights,
            )
            dump(
                receipt,
                dict(
                    binding=binds,
                    weights=sha(weights),
                    epochs=hist,
                    parameters=sum(p.numel() for p in net.parameters()),
                ),
            )


if __name__ == "__main__":
    main()
