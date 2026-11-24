"""Intersect every competitor's exact route constraint at an actual wrong prefix."""

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_region_writer import MODEL, WORK, PUB, Ink
from scripts.prepare_open_exam import read, dump
from scripts.write_region_writer import bitmap_ids, model
from scripts.train_region_writer import pack, join_regions
from scripts.train_paired_evidence import gpu_owner
from scripts.evaluate_visual_discourse import unpack


def interval(s, c, target, weak=False):
    lo, hi = Fraction(0), Fraction(1)
    closed_lo = closed_hi = True
    # Only actions with mass in either channel can beat a nonnegative target;
    # include one all-zero rival to preserve strict feasibility at zero mass.
    active = np.flatnonzero((s > 0) | (c > 0))
    zeros = np.flatnonzero((s == 0) & (c == 0) & (np.arange(len(s)) != target))
    for j in np.r_[active, zeros[:1]]:
        if j == target:
            continue
        b = int(c[target]) - int(c[j])
        a = int(s[target]) - int(s[j]) - b
        if a == 0:
            if b < 0 or (not weak and b == 0):
                return None
        else:
            cut = Fraction(-b, a)
            if a > 0 and cut >= lo:
                closed_lo = weak if cut > lo else weak and closed_lo
                lo = cut
            if a < 0 and cut <= hi:
                closed_hi = weak if cut < hi else weak and closed_hi
                hi = cut
        if lo > hi or (lo == hi and not (closed_lo and closed_hi)):
            return None
    return (lo + hi) / 2


def common_units(left, right):
    ratios = [float(v).as_integer_ratio() for v in (*left, *right)]
    denominator = max(d for _, d in ratios)
    values = np.asarray([v * (denominator // d) for v, d in ratios], dtype=object)
    return np.split(values, 2)


def main(dyadic=False):
    rows = read(WORK / "dev-source.json")
    report = {}
    bound = 100_000_000
    palette = np.load(MODEL / "palette.npy")
    n = len(palette)
    with gpu_owner(), torch.inference_mode():
        for arm in ("direct", "selected"):
            net = model(arm, "cuda")
            ink = Ink(palette)
            cases = []
            saved = unpack(WORK / (arm + "-predictions.json.gz"))["original"]
            for r, z in zip(rows, saved, strict=True):
                q = ink.encode(r["q"])
                parts = [ink.encode(b) for b in r["blocks"]]
                p = parts[z["selected"]] if arm == "selected" else join_regions(parts)
                refs = [ink.encode(t) + [1] for t in r["answers"]]
                raw = bytes.fromhex(z["ink_hex"])
                assert sha256(raw).hexdigest() == z["sha256"]
                emitted = bitmap_ids(
                    ink, np.frombuffer(raw, np.uint8).reshape(-1, 128)
                ) + ([1] if z["stop"] else [])
                keep = refs
                for step, pick in enumerate(emitted):
                    tail = [t for t in keep if step < len(t) and t[step] == pick]
                    if not tail:
                        break
                    keep = tail
                else:
                    cases.append(
                        dict(task=r["task"], exact=emitted in refs, no_divergence=True)
                    )
                    continue
                wanted = {t[step] for t in keep if step < len(t)}
                assert wanted and pick not in wanted
                x = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
                ctx, h = net.encode(x, n, pack([q], "cuda"), pack([p], "cuda"))
                hist = emitted[:step]
                states = torch.cat([net.bos[None, :], ctx[0][hist]], 0)[None, :, :]
                last, _ = net.gru(states, h)
                past = (hist[::-1] + [0, 0, 0])[:3]
                lp, ptr, g = net.distribution(
                    ctx, last[:, -1:], torch.tensor([[past]], device="cuda")
                )
                spell = np.zeros(len(ink.tiles))
                spell[1:n] = lp.exp()[0, 0].cpu().numpy()
                copy = np.bincount(
                    ctx[2][0].cpu().numpy(),
                    weights=ptr[0, 0].cpu().numpy(),
                    minlength=len(spell),
                )
                gate = float(g.sigmoid())
                native = gate * spell + (1 - gate) * copy
                s, c = (
                    np.rint(spell * bound).astype("int64"),
                    np.rint(copy * bound).astype("int64"),
                )
                if dyadic:
                    s, c = common_units(spell, copy)
                strict = False
                possible = False
                confirmed = False
                for a in wanted:
                    mid = interval(s[1:], c[1:], a - 1)
                    if mid is not None:
                        strict = True
                        mass = float(mid) * spell + (1 - float(mid)) * copy
                        confirmed |= bool(mass[a] > np.max(np.delete(mass[1:], a - 1)))
                    possible |= interval(s[1:], c[1:], a - 1, True) is not None
                cases.append(
                    dict(
                        task=r["task"],
                        exact=False,
                        no_divergence=False,
                        step=step,
                        native_agrees=int(native.argmax()) == pick,
                        quantized_agrees=int((gate * s + (1 - gate) * c).argmax())
                        == pick,
                        strict=strict,
                        weak=possible,
                        confirmed=confirmed,
                    )
                )
            groups = defaultdict(Counter)
            for z in cases:
                g = groups[z["task"]]
                g["cases"] += 1
                if z["no_divergence"]:
                    g["exact" if z["exact"] else "prefix_only"] += 1
                    continue
                g["wrong"] += 1
                g.update(
                    {
                        k: int(z[k])
                        for k in (
                            "native_agrees",
                            "quantized_agrees",
                            "strict",
                            "weak",
                            "confirmed",
                        )
                    }
                )
            report[arm] = dict(groups=groups, cases=cases)
    dump(
        PUB / ("route-native.json" if dyadic else "route.json"),
        dict(
            units="exact binary probability ratios" if dyadic else bound, results=report
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dyadic", action="store_true")
    main(p.parse_args().dyadic)
