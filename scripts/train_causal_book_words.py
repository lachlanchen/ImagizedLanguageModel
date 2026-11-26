"""Train matched optical models with consecutive or endpoint-only supervision."""

from hashlib import sha256
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from scripts import prepare_passage_selector as old
from scripts.prepare_causal_book_words import PUB, WORK, MODEL, RULE, SEED, ROOT
from scripts.train_passage_selector import (
    RATES,
    load_bank,
    atomic_save,
    gpu_owner,
    configure_cuda_cli,
)
from ilm.visual_lm.causal_book_words import CausalBookWords
from ilm.visual_lm.passage_selector import VisualBank, Print


def bindings():
    files = [
        PUB / "data.json",
        RULE,
        Path(__file__),
        ROOT / "scripts/prepare_causal_book_words.py",
        ROOT / "ilm/visual_lm/causal_book_words.py",
        ROOT / "ilm/visual_lm/ordered_translation.py",
        ROOT / "ilm/visual_lm/passage_selector.py",
        old.MODEL / "document_frequency.json",
    ]
    return {str(path.relative_to(ROOT)): old.base.sha(path) for path in files}


def extended_bank(device):
    face = Print()
    return VisualBank(
        face, [face.ids(w) for w in old.base.read(WORK / "forms.json")], device
    )


def check_causality(bank, net, src, marks):
    """Full forward/motor likelihood and gradients must agree with physically truncated input."""
    row = torch.arange(len(src), device=src.device)
    last = src.new_full((len(src),), 80)
    task = src.new_zeros(len(src))
    ids = src[:, 81]

    def trace(s, m, last):
        states = net.encode_page(bank, s, m)
        q, a, g = net.read_at(bank, s, states, net.key(states), row, last, task)
        logspell = net.spell_logprob(q, bank.codes[ids], bank.lengths[ids])
        mass = (a * s.eq(ids[:, None])).sum(1) * g
        value = torch.logaddexp(mass.clamp_min(1e-30).log(), torch.log1p(-g) + logspell)
        return q, a, g, value

    base = trace(src, marks, last)
    short = trace(src[:, :81], marks[:, :81], last)
    changed = src.clone()
    changed[:, 81:] = src[:, 81:].roll(1, 0)
    other = marks.clone()
    other[:, 81:] = (marks[:, 81:] + 3) % 8
    perturbed = trace(changed, other, last)
    assert not base[1][:, 81:].any()
    worst = 0.0
    for values in (short, perturbed):
        for x, y in zip(base, values, strict=True):
            if x.ndim == 2 and x.shape[1] != y.shape[1]:
                x = x[:, : y.shape[1]]
            torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-5)
            worst = max(worst, float((x - y).abs().max().detach()))
    vars = list(net.parameters())
    gradients = [
        torch.autograd.grad(v[3].sum(), vars, allow_unused=True) for v in (base, short)
    ]
    max_grad = 0.0
    for x, y in zip(*gradients, strict=True):
        if x is None:
            assert y is None
            continue
        torch.testing.assert_close(x, y, rtol=5e-4, atol=5e-5)
        max_grad = max(max_grad, float((x - y).abs().max()))
    return dict(
        cases=len(src), max_forward_absolute=worst, max_gradient_absolute=max_grad
    )


def fit(arm, bank, data, receipt):
    dest = MODEL / arm
    weights = dest / "weights.safetensors"
    if (dest / "fit.json").exists():
        prior = old.base.read(dest / "fit.json")
        assert prior["bindings"] == bindings() and prior["weights"] == old.base.sha(
            weights
        )
        return prior
    torch.manual_seed(SEED)
    net = CausalBookWords(bank.face.coarse)
    init = sha256(
        b"".join(x.detach().numpy().tobytes() for x in net.parameters())
    ).hexdigest()
    net.cuda().train()
    pre = check_causality(
        bank, net, data["source"][:12].long(), data["marks"][:12].long()
    )
    print("causal check", arm, pre, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=RATES[0], fused=True)
    rng = np.random.default_rng(SEED)
    position = torch.arange(96, device="cuda")[None]
    history = []
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch, rate in enumerate(RATES):
        order = rng.permutation(len(data["source"])).astype("<i4")
        opt.param_groups[0]["lr"] = rate
        totals = torch.zeros(5, device="cuda", dtype=torch.float64)
        for step, idx in enumerate(
            torch.as_tensor(order, device="cuda").long().split(128), 1
        ):
            s = data["source"][idx].long()
            m = data["marks"][idx].long()
            t = data["task"][idx].long()
            y = data["target"][idx]
            end = data["end"][idx]
            rows = torch.arange(len(idx), device="cuda")
            last = torch.full_like(rows, 95)
            first = 96 - s.ne(0).sum(1, keepdim=True) + 16
            qlo = first + (96 - first) * epoch // 4
            qhi = first + (96 - first) * (epoch + 1) // 4
            rr, wordpos = (
                (position >= qlo) & (position < qhi) & (bank.lengths[s] > 0)
            ).nonzero(as_tuple=True)
            aux_count = len(rr)
            # Both arms use an equal number of auxiliary ordinary prediction events.
            # The endpoint control repeats ordinary endpoints uniformly in batch order.
            if arm == "dense":
                yy = s[rr, wordpos]
                ee = m[rr, wordpos] == 1
                prev = wordpos - 1
            else:
                ordinary = rows[t == 0]
                assert len(ordinary)
                rr = ordinary[torch.arange(aux_count, device="cuda") % len(ordinary)]
                yy = y[rr]
                ee = end[rr]
                prev = last[rr]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = net.encode_page(bank, s, m)
                keys = net.key(h)
                a, _, _ = net.objective(bank, s, h, keys, rows, last, t, y, end)
                b, logp, present = net.objective(
                    bank, s, h, keys, rr, prev, torch.zeros_like(rr), yy, ee
                )
                loss = (a + b) * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            totals += torch.stack(
                (
                    a.detach(),
                    b.detach(),
                    -logp.sum(),
                    present.sum(),
                    torch.as_tensor(aux_count, device="cuda"),
                )
            )
            if (
                monotonic() - start > 2700
                or torch.cuda.max_memory_allocated() > 16 * 2**30
            ):
                raise RuntimeError("Fit exceeds registered resource budget")
            if step % 480 == 0:
                print(arm, epoch + 1, step, round(monotonic() - start, 1), flush=True)
        x, b, lp, p, n = totals.tolist()
        history.append(
            dict(
                epoch=epoch + 1,
                steps=step,
                windows=len(order),
                auxiliary=int(n),
                present=int(p),
                main_loss=x / step,
                auxiliary_loss=b / step,
                auxiliary_nll=lp / n,
                seconds=monotonic() - start,
                schedule=sha256(order.tobytes()).hexdigest(),
            )
        )
        print(history[-1], flush=True)
    assert sum(e["auxiliary"] for e in history) == receipt["interior"]
    dest.mkdir(parents=True, exist_ok=True)
    atomic_save(
        lambda f: save_file(
            {k: v.cpu().contiguous() for k, v in net.state_dict().items()}, f
        ),
        weights,
    )
    result = dict(
        arm=arm,
        initial=init,
        epochs=history,
        seconds=monotonic() - start,
        peak_bytes=torch.cuda.max_memory_allocated(),
        parameters=sum(x.numel() for x in net.parameters()),
        bindings=bindings(),
        weights=old.base.sha(weights),
        causal_check=pre,
        device=torch.cuda.get_device_name(),
    )
    old.base.dump(dest / "fit.json", result)
    return result


def main():
    configure_cuda_cli()
    d = old.base.read(PUB / "data.json")
    assert d["protocol"] == old.base.sha(RULE) and d["previous"] == old.base.sha(
        old.PUB / "data.json"
    )
    assert all(old.base.sha(WORK / f) == h for f, h in d["files"].items())
    assert (
        old.base.sha(old.WORK / "train.npz")
        == old.base.read(old.PUB / "data.json")["files"]["train.npz"]
    )
    keep = np.load(WORK / "keep.npy")
    with gpu_owner():
        bank = load_bank("cuda")
        with np.load(old.WORK / "train.npz") as a:
            data = {
                k: torch.as_tensor(a[k][keep], device="cuda")
                for k in ("source", "marks", "task", "end")
            }
            data["target"] = torch.as_tensor(a["target"][keep, 0], device="cuda").long()
        reports = [fit(arm, bank, data, d) for arm in ("dense", "endpoint")]
        assert reports[0]["initial"] == reports[1]["initial"] and all(
            a["schedule"] == b["schedule"] and a["auxiliary"] == b["auxiliary"]
            for a, b in zip(reports[0]["epochs"], reports[1]["epochs"], strict=True)
        )


if __name__ == "__main__":
    main()
