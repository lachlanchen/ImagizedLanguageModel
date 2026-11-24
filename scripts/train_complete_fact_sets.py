"""Continue the visual list reader with complete corpus coverage and competing answer sets."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import monotonic
from math import cos, pi

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.complete_fact_sets import complete_loss, source_runs
from ilm.visual_lm.run_survival import supervised
from scripts.prepare_complete_fact_sets import WORK, PUB, MODEL
from scripts.prepare_multiple_ink_runs import (
    WORK as OLD,
    PUB as OLD_PUBLIC,
    ROOT,
    read,
    sha,
    dump,
)
from scripts.train_multiple_ink_runs import load
from scripts.train_paired_evidence import gpu_owner
from scripts.train_visual_cursor import pad
from scripts.measure_writer_energy import consumed, snapshot

STEPS = 7182
SEED = 317101


def schedule(langs):
    rng = np.random.default_rng(SEED)
    draws = []
    for group, batch in enumerate((4, 12)):
        pool = np.flatnonzero(langs == group)
        n = STEPS * batch
        stream = np.concatenate(
            [rng.permutation(pool) for _ in range((n + len(pool) - 1) // len(pool))]
        )
        draws.append(stream[:n].reshape(STEPS, batch))
    return np.concatenate(draws, axis=1)


def layout(arr):
    bounds = np.zeros((len(arr["lang"]), 16, 2), np.int32)
    mask = np.zeros(len(arr["p"]), np.bool_)
    for i, (lo, hi) in enumerate(zip(arr["po"][:-1], arr["po"][1:], strict=True)):
        start, end, reserved = source_runs(arr["p"][lo:hi], arr["y"][lo:hi])
        if not 1 <= len(start) <= 16:
            raise ValueError("Training path outside decoder support")
        bounds[i, : len(start), 0] = start
        bounds[i, : len(start), 1] = end
        mask[lo:hi] = reserved
    return bounds, mask


def train(arm, run=None):
    out = run or MODEL / arm
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "survival.json"
    weight = out / "survival.safetensors"
    if receipt.exists():
        if read(receipt)["weights"] != sha(weight):
            raise ValueError("Changed fit")
        if read(receipt)["bindings"]["arm"] != arm:
            raise ValueError("Run directory belongs to other arm")
        return
    src = OLD / "training.npz"
    if sha(src) != read(OLD_PUBLIC / "corpus.json")["files"][src.name]:
        raise ValueError("Changed archive")
    with np.load(src) as f:
        arr = dict(f)
    ids = schedule(arr["lang"])
    bind = dict(
        arm=arm,
        corpus=sha(src),
        schedule=sha256(ids.tobytes()).hexdigest(),
        init=read(OLD_PUBLIC / "survival.json")["weights"],
        fresh=sha(PUB / "fresh.json"),
        trainer=sha(Path(__file__)),
        loss=sha(ROOT / "ilm/visual_lm/complete_fact_sets.py"),
        protocol=sha(ROOT / "references/complete_fact_sets_protocol.md"),
    )
    path = (run or WORK) / f"{arm}-resume.pt"
    limits = dict(seconds=1800, gib=12)

    def batch(field, offset, i):
        return arr[field][arr[offset][i] : arr[offset][i + 1]]

    if arm == "contrast":
        bounds, reserved = layout(arr)
    with gpu_owner():
        torch.manual_seed(SEED)
        model = load("cuda", True).train()
        model.ink.requires_grad_(False)
        bank = model.ink(torch.from_numpy(observations(arr, "ink")).cuda()).detach()
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.00025,
            weight_decay=0.02,
            fused=True,
        )
        first = 0
        elapsed = joules = 0.0
        trace = []
        if path.exists():
            st = torch.load(path, map_location="cpu", weights_only=True)
            if st["bindings"] != bind:
                raise ValueError("Checkpoint binding mismatch")
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            torch.set_rng_state(st["cpu"])
            torch.cuda.set_rng_state(st["cuda"])
            first, elapsed, joules, trace = (
                st["step"],
                st["seconds"],
                st["joules"],
                st["trace"],
            )
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        meter = snapshot()
        for step in range(first, STEPS):
            ix = ids[step]
            q, p, y = [
                torch.as_tensor(pad([batch(f, o, i) for i in ix]), device="cuda").long()
                for f, o in (("q", "qo"), ("p", "po"), ("y", "po"))
            ]
            keep = p >= 2
            opt.param_groups[0]["lr"] = (
                0.00025
                * min(1, (step + 1) / 120)
                * (0.15 + 0.85 * (1 + cos(pi * (step + 1) / STEPS)) / 2)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                role, c = model(bank, q, p)
                count = F.cross_entropy(c.float(), (y == 1).sum(1))
                path_loss = supervised(role, y, keep)
                loss = path_loss + 0.18 * count
                if arm == "contrast":
                    pairs = bounds[ix]
                    num = (pairs[:, :, 1] > 0).sum(1)
                    # Select one annotated length per page; uniform over its marked runs.
                    chosen = np.random.default_rng([SEED, step]).integers(0, num)
                    b = torch.as_tensor(pairs, device="cuda").long()
                    excluded = torch.as_tensor(
                        pad([reserved[arr["po"][i] : arr["po"][i + 1]] for i in ix]),
                        device="cuda",
                    ).bool()
                    ranking = complete_loss(
                        role.float(),
                        c.float(),
                        keep,
                        b[:, :, 0],
                        b[:, :, 1],
                        excluded,
                        torch.as_tensor(chosen, device="cuda"),
                    )
                    loss = loss + 0.20 * ranking
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            opt.step()
            n = step + 1
            if n == 1 or n % 500 == 0 or n == STEPS:
                rec = dict(step=n, likelihood=path_loss.item(), count=count.item())
                if arm == "contrast":
                    rec["ranking"] = ranking.item()
                trace.append(rec)
                print(arm, rec, flush=True)
            if n % 1000 == 0 or n == STEPS:
                torch.cuda.synchronize()
                seconds = elapsed + monotonic() - start
                energy = joules + consumed(meter, snapshot())["joules"]
                if (
                    seconds > limits["seconds"]
                    or torch.cuda.max_memory_allocated() > limits["gib"] * 2**30
                ):
                    raise RuntimeError("Fit exceeded resource budget")
                tmp = path.with_suffix(".tmp")
                torch.save(
                    dict(
                        bindings=bind,
                        step=n,
                        seconds=seconds,
                        joules=energy,
                        trace=trace,
                        model=model.state_dict(),
                        opt=opt.state_dict(),
                        cuda=torch.cuda.get_rng_state(),
                        cpu=torch.get_rng_state(),
                    ),
                    tmp,
                )
                tmp.replace(path)
        torch.cuda.synchronize()
        seconds = elapsed + monotonic() - start
        energy = joules + consumed(meter, snapshot())["joules"]
        tmp = weight.with_suffix(".tmp")
        save_file(model.cpu().state_dict(), str(tmp))
        tmp.replace(weight)
        dump(
            receipt,
            dict(
                weights=sha(weight),
                bindings=bind,
                steps=STEPS,
                presentations=int(ids.size),
                english_distinct=int(len(np.unique(ids[:, :4]))),
                chinese_distinct=int(len(np.unique(ids[:, 4:]))),
                parameters=sum(p.numel() for p in model.parameters()),
                peak_bytes=torch.cuda.max_memory_allocated(),
                seconds=seconds,
                joules=energy,
                trace=trace,
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("control", "contrast"))
    parser.add_argument("--run-dir", type=Path)
    a = parser.parse_args()
    train(a.arm, a.run_dir)
