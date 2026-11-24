"""Fit ordinal-reference odds on visual cues with the cast scorer frozen."""

import hashlib
from time import monotonic
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from ilm.visual_lm.reference_order import ReferenceOrder, cue
from scripts.ordered_reference_data import teaching, ordinary
from scripts.train_story_cast import prepare
from scripts.train_story_cast import load_cast
from scripts.prepare_story_mentions import PUBLIC, MODEL, read, dump, sha
from scripts.train_semantic_page import gpu_owner, snapshot, consumed


def sides(slots, q):
    last = {s: i for i, s in enumerate(slots[:q]) if s >= 0}
    if len(last) != 2:
        return None
    older, newer = sorted(last, key=last.get)
    return older, newer


def load_order(arm, device):
    file = MODEL / ("order-" + arm + ".safetensors")
    if sha(file) != read(PUBLIC / ("order-" + arm + "-fit.json"))["sha256"]:
        raise ValueError("Changed order reader")
    net = ReferenceOrder().to(device)
    net.load_state_dict(load_file(str(file), device=device))
    return net.eval()


@torch.no_grad()
def features(rows, mode, erase):
    obs, (ids, slots, bounds, sizes, truth), meta = prepare(rows, mode, "cuda")
    base = load_cast(mode, "cuda")
    patches = []
    logits = []
    targets = []
    for a in range(0, len(rows), 48):
        ix = slice(a, a + 48)
        x = obs[ids[ix].long()]
        score = base(x, slots[ix], bounds[ix], sizes[ix])
        patch = cue(x, bounds[ix], slots[ix], sizes[ix], erase)
        for j, (tag, _, q) in enumerate(meta[ix]):
            order = sides(tag, q[0])
            if order is None:
                continue
            earlier = torch.arange(slots.size(1), device="cuda") < q[0]
            masks = [(slots[a + j] == v) & earlier for v in order]
            l = [score[j, :-1].masked_fill(~m, -1e4).logsumexp(0) for m in masks]
            t = [(truth[a + j, :-1] & m).any() for m in masks]
            logits.append(torch.stack([*l, score[j, -1]]))
            targets.append(torch.stack([*t, truth[a + j, -1]]))
            patches.append(patch[j].clone())
    return tuple(torch.stack(v) for v in (patches, logits, targets))


def fit(arm, known, ordered):
    receipt = PUBLIC / ("order-" + arm + "-fit.json")
    if receipt.exists():
        load_order(arm, "cpu")
        return
    torch.manual_seed(745919)
    rng = np.random.default_rng(4901)
    net = ReferenceOrder().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.01, fused=True)
    h = hashlib.sha256()
    trace = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    e = snapshot()
    start = monotonic()
    for step in range(1, 801):
        old = rng.integers(len(known[0]), size=24)
        new = rng.integers(len(ordered[0]), size=24)
        h.update(old.tobytes())
        h.update(new.tobytes())
        delta = net(known[0][old])
        x = ordered[1][new] + net(ordered[0][new])
        loss = (
            x.logsumexp(1) - x.masked_fill(~ordered[2][new], -1e4).logsumexp(1)
        ).mean() + 0.15 * delta.square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step == 550:
            opt.param_groups[0]["lr"] = 0.00035
        if step % 100 == 0:
            val = loss.item()
            trace.append((step, val))
            if (
                not np.isfinite(val)
                or monotonic() - start > 240
                or torch.cuda.max_memory_allocated() > 2 * 2**30
            ):
                raise RuntimeError("Resource limit")
    torch.cuda.synchronize()
    seconds = monotonic() - start
    cost = consumed(e, snapshot())
    p = MODEL / ("order-" + arm + ".safetensors")
    tmp = p.with_suffix(".tmp")
    save_file({k: v.cpu() for k, v in net.state_dict().items()}, str(tmp))
    tmp.replace(p)
    dump(
        receipt,
        dict(
            sha256=sha(p),
            schedule=h.hexdigest(),
            steps=800,
            seconds=seconds,
            energy=cost,
            parameters=sum(p.numel() for p in net.parameters()),
            peak=torch.cuda.max_memory_allocated(),
            eligible_ordinary=len(known[0]),
            eligible_constructed=len(ordered[0]),
            trace=trace,
        ),
    )
    print(arm, len(ordered[0]), seconds, cost, flush=True)


def main():
    ordrows = teaching()
    real = ordinary()
    with gpu_owner():
        for arm in ("ink", "erased", "byte"):
            mode = "byte" if arm == "byte" else "ink"
            x = features(real, mode, arm == "erased")
            y = features(ordrows, mode, arm == "erased")
            fit(arm, x, y)
            del x, y


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
