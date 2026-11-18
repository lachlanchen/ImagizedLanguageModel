"""Compare likelihood of one path and marginal likelihood of physically identical ink."""

import fcntl
import hashlib
import time
import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F
from ilm.visual_lm.causal_glyph_translator import COPY, DRAW, END, EXEMPLAR
from ilm.visual_lm.idiom_ink import visual_input, nibbles
from scripts.train_causal_glyph_translator import prepare, read_rows, tiles
from scripts.train_paired_student import (
    INITIAL,
    PARENT,
    ROOT,
    load,
    require,
    restore,
    sha,
    write_new,
)

OUT = ROOT / "artifacts/visible_image_likelihood_v1"


def emission_loss(
    model, feat, source, target, lens, sizes, aligned, labels, addr, lookup, marginal
):
    memory, keys, mask, h = model.read(feat[source], lens)
    q, _ = model.history(
        torch.cat((model.start.expand(len(lens), 1, -1), model.embed(feat[target])), 1),
        h,
    )
    state, attention, center = model.plan_steps(
        q, memory, keys, mask, lens, lens.new_full(lens.shape, -0.65, dtype=torch.float)
    )
    pos = torch.arange(state.size(1), device=source.device)
    live = pos[:-1] < sizes[:, None]
    valid = pos <= sizes[:, None]
    ids = torch.arange(len(lens), device=source.device)[:, None].expand_as(live)[live]
    a = attention[:, :-1][live].float()
    col = aligned[live]
    location = -a.gather(1, col[:, None]).mean()
    progress = F.smooth_l1_loss(center[:, :-1][live], col.float())
    route = model.route(state).float().log_softmax(-1)
    nr = route[:, :-1][live]
    st = state[:, :-1][live]
    tgt = target[live]
    mode = labels[:, :-1][live]
    proto = model.prototype_scores(st, model.image_keys()).float().log_softmax(-1)
    exemplar = lookup[tgt]
    same = (source[ids] == tgt[:, None]) & (~mask[ids])
    available = same.any(-1) | (exemplar >= 0)
    if marginal:
        # Mask strictly impossible paths. A finite sentinel avoids undefined
        # gradients from logsumexp(-inf, -inf) on draw-only rows.
        lp_copy = nr[:, COPY] + a.masked_fill(~same, -1e9).logsumexp(-1)
        lp_ex = nr[:, EXEMPLAR] + proto.gather(
            1, exemplar.clamp_min(0)[:, None]
        ).squeeze(1)
        score = torch.logaddexp(lp_copy, lp_ex.masked_fill(exemplar < 0, -1e9))
        paint = ~available
    else:
        lp_copy = nr[:, COPY] + a.gather(1, col[:, None]).squeeze(1)
        lp_ex = nr[:, EXEMPLAR] + proto.gather(1, addr[live][:, None]).squeeze(1)
        score = torch.where(mode == COPY, lp_copy, lp_ex)
        paint = mode == DRAW
    nll = (
        -score[~paint].sum()
        - route[torch.arange(len(lens), device=lens.device), sizes, END].sum()
    )
    if paint.any():
        bits = feat[tgt[paint], 64:]
        logits = model.draw.teacher(st[paint], bits).float()
        nll += (
            F.cross_entropy(logits.transpose(1, 2), nibbles(bits), reduction="sum")
            - nr[paint, DRAW].sum()
        )
    return nll / valid.sum() + 0.65 * location + 0.08 * progress


def run():
    OUT.mkdir()
    torch.set_num_threads(2)
    torch.manual_seed(20813)
    torch.set_float32_matmul_precision("high")
    require(
        sha(PARENT) == load(INITIAL / "fit.json")["checkpoint_sha256"], "Parent changed"
    )
    rows = read_rows("train")
    packed, canonical = np.unique(tiles(), axis=0, return_inverse=True)
    examples, arrays = prepare(rows, canonical)
    model = restore(PARENT).cuda()
    require(
        np.array_equal(
            np.packbits(model.exemplars.cpu().numpy(), axis=1), packed[examples]
        ),
        "Exemplar identity changed",
    )
    bank = torch.from_numpy(np.unpackbits(packed, axis=1)).cuda().bool()
    feat = visual_input(bank, model.landmarks)
    del model
    lookup = torch.full((len(feat),), -1, device="cuda", dtype=torch.long)
    lookup[torch.as_tensor(examples, device="cuda")] = torch.arange(
        len(examples), device="cuda"
    )
    x, y, col, route, addr, n, m = [torch.as_tensor(q, device="cuda") for q in arrays]
    stats = {}
    for arm in ("single", "marginal"):
        model = restore(PARENT).cuda()
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=0.0003
        )
        rng = np.random.default_rng(20813)
        schedule = hashlib.sha256()
        steps = 0
        losses = []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(4):
            batches = []
            for bucket in np.array_split(rng.permutation(len(rows)), 16):
                bucket = bucket[np.argsort(arrays[-1][bucket], kind="stable")]
                batches.extend(bucket[i : i + 96] for i in range(0, len(bucket), 96))
            rng.shuffle(batches)
            total = 0.0
            for items in batches:
                schedule.update(items.astype("<i4").tobytes())
                idx = torch.as_tensor(items, device="cuda")
                w, l = int(n[idx].max()), int(m[idx].max())
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = emission_loss(
                        model,
                        feat,
                        x[idx, :w].long(),
                        y[idx, :l].long(),
                        n[idx],
                        m[idx],
                        col[idx, :l].long(),
                        route[idx, : l + 1],
                        addr[idx, :l].long(),
                        lookup,
                        arm == "marginal",
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1, error_if_nonfinite=True
                )
                opt.step()
                steps += 1
                total += loss.item() * len(items)
            losses.append(total / len(rows))
            with open("/proc/meminfo") as f:
                reserve = int(
                    next(v for v in f if v.startswith("MemAvailable:")).split()[1]
                )
            require(
                reserve >= 24 * 2**20
                and torch.cuda.max_memory_allocated() < 10 * 2**30,
                "Memory limit reached",
            )
            print(arm, epoch + 1, losses[-1], flush=True)
        path = OUT / (arm + ".safetensors")
        save_file(
            {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(path)
        )
        stats[arm] = dict(
            loss=losses,
            steps=steps,
            seconds=time.monotonic() - start,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            sha256=sha(path),
            schedule=schedule.hexdigest(),
        )
    require(
        stats["single"]["schedule"] == stats["marginal"]["schedule"],
        "Unmatched schedule",
    )
    write_new(
        OUT / "fit.json",
        dict(
            arms=stats,
            rows=len(rows),
            target_cells=int(m.sum()),
            epochs=4,
            parent=sha(PARENT),
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run()
