"""Compare equal full-corpus fits, adding a context-dependent visual contrast to one."""

import fcntl
import hashlib
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from ilm.visual_lm.causal_glyph_translator import COPY, EXEMPLAR
from ilm.visual_lm.idiom_ink import visual_input
from scripts.prepare_predicate_likelihood import (
    OUT,
    ROOT,
    PREDICATES,
    Rasterizer,
    load,
    sha,
    write_new,
)
from scripts.train_causal_glyph_translator import prepare, read_rows, tiles
from scripts.train_paired_student import restore
from scripts.train_visible_image_likelihood import emission_loss
from scripts.train_visual_event_roles import PARENT


def choice_log_mass(route, attention, proto, source, mask, choices, lookup):
    """Marginalize every copy and exemplar route to each of two physical images."""
    match = (source[:, None, :] == choices[:, :, None]) & ~mask[:, None, :]
    copy = (
        attention[:, None, :].masked_fill(~match, -1e9).logsumexp(-1)
        + route[:, COPY, None]
    )
    ex = lookup[choices]
    if (ex < 0).any():
        raise ValueError("Contrastive images must belong to the frozen exemplar bank")
    return torch.logaddexp(copy, proto.gather(1, ex) + route[:, EXEMPLAR, None])


def contrast(model, feat, source, target, lens, at, choices, lookup):
    memory, keys, mask, h = model.read(feat[source], lens)
    width = int(at.max())
    prefix = model.embed(feat[target[:, :width]])
    q, _ = model.history(
        torch.cat((model.start.expand(len(lens), 1, -1), prefix), 1), h
    )
    st, att, _ = model.plan_steps(
        q, memory, keys, mask, lens, torch.full_like(lens, -0.65, dtype=torch.float)
    )
    rows = torch.arange(len(lens), device=lens.device)
    st = st[rows, at]
    mass = choice_log_mass(
        model.route(st).float().log_softmax(-1),
        att[rows, at].float(),
        model.prototype_scores(st, model.image_keys()).float().log_softmax(-1),
        source,
        mask,
        choices,
        lookup,
    )
    return F.softplus(mass[:, 1] - mass[:, 0]).mean()


def fit():
    torch.set_num_threads(2)
    torch.manual_seed(73619)
    torch.set_float32_matmul_precision("high")
    assert (
        sha(PARENT)
        == load(ROOT / "references/evidence/visible_likelihood_20261119/fit.json")[
            "arms"
        ]["marginal"]["sha256"]
    )
    seal = load(OUT / "admission.json")
    assert (
        sha(OUT / "test.json") == seal["test_sha"]
        and sha(OUT / "train.json") == seal["train_sha"]
    )
    full = read_rows("train")
    focus = load(OUT / "train.json")
    base = restore(PARENT)
    ex = np.packbits(base.exemplars.numpy(), axis=1)
    bank = list(map(bytes, ex))
    ids = {v: i for i, v in enumerate(bank)}
    renderer = Rasterizer()

    def index(v):
        if v not in ids:
            ids[v] = len(bank)
            bank.append(v)
        return ids[v]

    raw = np.array([index(bytes(c)) for c in tiles()])
    encoded = [dict(x=raw[r["x"]], z=raw[r["z"]]) for r in full]
    pos = []
    choices = []
    buckets = {g: [] for gs in PREDICATES.values() for g in gs}
    for i, r in enumerate(focus):
        encoded.append(
            {
                field: [index(renderer.tile(c)) for c in r[name] + "。"]
                for field, name in (("x", "source"), ("z", "target"))
            }
        )
        pos.append(r["target"].index(r["gloss"]))
        g = r["gloss"]
        (other,) = set(PREDICATES[r["predicate"]]) - {g}
        choices.append([ids[renderer.tile(v[0])] for v in (g, other)])
        buckets[g].append(i)
    _, a = prepare(encoded, np.arange(len(bank)), examples=np.arange(len(ex)))
    x, y, col, route, addr, n, m = [torch.as_tensor(v, device="cuda") for v in a]
    bits = (
        torch.from_numpy(
            np.unpackbits(
                np.frombuffer(b"".join(bank), np.uint8).reshape(-1, 128), axis=1
            )
        )
        .cuda()
        .bool()
    )
    feat = visual_input(bits, base.landmarks.cuda())
    lookup = torch.full((len(bank),), -1, dtype=torch.long, device="cuda")
    lookup[: len(ex)] = torch.arange(len(ex), device="cuda")
    pos = torch.tensor(pos, device="cuda")
    choices = torch.tensor(choices, device="cuda")
    assert (lookup[choices] >= 0).all()
    rng = np.random.default_rng(73619)
    order = rng.permutation(len(full))
    batches = [order[i : i + 96] for i in range(0, len(full), 96)]
    reservoirs = [
        np.concatenate(
            [rng.permutation(v) for _ in range((2 * len(batches)) // len(v) + 1)]
        )
        for v in buckets.values()
    ]
    schedule = [
        (ordinary, np.concatenate([v[2 * i : 2 * i + 2] for v in reservoirs]))
        for i, ordinary in enumerate(batches)
    ]
    ledger = hashlib.sha256()
    steps = len(schedule)
    total = 0
    for ordinary, special in schedule:
        seq = np.r_[ordinary, len(full) + special]
        ledger.update(seq.astype("<i4").tobytes())
        total += int(a[-1][seq].sum())
    report = {}
    for arm in ("replay", "context"):
        model = restore(PARENT).cuda()
        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=0.0001
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        losses = []
        aux = []
        for ordinary, special in schedule:
            idx = torch.tensor(np.r_[ordinary, len(full) + special], device="cuda")
            f = torch.tensor(special, device="cuda")
            xi, yi = (
                x[idx, : int(n[idx].max())].long(),
                y[idx, : int(m[idx].max())].long(),
            )
            width = yi.shape[1]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = emission_loss(
                    model,
                    feat,
                    xi,
                    yi,
                    n[idx],
                    m[idx],
                    col[idx, :width].long(),
                    route[idx, : width + 1],
                    addr[idx, :width].long(),
                    lookup,
                    True,
                )
                if arm == "context":
                    c = contrast(
                        model,
                        feat,
                        xi[-len(f) :],
                        yi[-len(f) :],
                        n[idx][-len(f) :],
                        pos[f],
                        choices[f],
                        lookup,
                    )
                    loss = loss + c
                    aux.append(c.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            losses.append(loss.item())
            if len(losses) % 25 == 0:
                mem = int(
                    next(
                        s
                        for s in open("/proc/meminfo")
                        if s.startswith("MemAvailable:")
                    ).split()[1]
                )
                if mem < 24 * 2**20 or torch.cuda.max_memory_allocated() >= 10 * 2**30:
                    raise RuntimeError("Resource ceiling")
        path = OUT / (arm + ".safetensors")
        if path.exists():
            raise FileExistsError(path)
        save_file(
            {k: v.cpu().contiguous() for k, v in model.state_dict().items()}, str(path)
        )
        report[arm] = dict(
            steps=steps,
            seconds=time.monotonic() - start,
            peak_mib=torch.cuda.max_memory_allocated() / 2**20,
            sha256=sha(path),
            first_loss=float(np.mean(losses[:32])),
            last_loss=float(np.mean(losses[-32:])),
            binary_loss_ends=[float(np.mean(aux[:32])), float(np.mean(aux[-32:]))]
            if aux
            else [],
        )
        print(arm, report[arm], flush=True)
    write_new(
        OUT / "fit.json",
        dict(
            arms=report,
            training=seal["train_sha"],
            parent=sha(PARENT),
            schedule=ledger.hexdigest(),
            ordinary=len(full),
            focus_exposures=8 * steps,
            target_cells=total,
        ),
    )


if __name__ == "__main__":
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fit()
