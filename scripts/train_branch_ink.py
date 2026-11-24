"""Fit a complete stopped optical writer with additional next-glyph supervision on sibling questions."""

import argparse
from hashlib import sha256
from math import cos, pi
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_branch_ink import PUB, WORK, MODEL, EXAM
from scripts.prepare_open_exam import ROOT, read, dump, sha
from scripts.train_visual_cursor import base_batch, pad
from scripts.train_open_exam import corpus, likelihood, SEED
from scripts.write_recurrent_evidence import load as initial
from scripts.write_visual_discourse import Writer
from scripts.measure_writer_energy import consumed, snapshot
from scripts.train_paired_evidence import gpu_owner


def branch_log_prob(net, base, x, bk, n, rows, cuts):
    """Log M(prefix + requested cell) minus log M(prefix); M(empty) = 1."""

    def prefix_mass(indices, extra):
        sel = [rows[i] for i in indices]
        ys = [rows[i]["y"][: int(cuts[i]) + extra] for i in indices]
        qs, ps = [[r[key] for r in sel] for key in ("qi", "pi")]
        q, p, y = [torch.as_tensor(pad(v), device=x.device) for v in (qs, ps, ys)]
        ids, pos = [
            torch.as_tensor(v, device=x.device)
            for v in base_batch([a + b for a, b in zip(ps, qs, strict=True)], ys)
        ]
        with torch.no_grad():
            prior = base.log_prob(bk, ids, n, pos)
        log_mass, _ = net.observed(x, q, p, y, prior, n)
        return log_mass

    eligible = np.flatnonzero(cuts >= 0)
    if not len(eligible):
        return x.new_empty(0)
    joint = prefix_mass(eligible, 1)
    nonempty = np.flatnonzero(cuts[eligible] > 0)
    if not len(nonempty):
        return joint
    prior = prefix_mass(eligible[nonempty], 0)
    return joint.index_add(0, torch.as_tensor(nonempty, device=x.device), -prior)


def fit(dest):
    rows, tiles, n = corpus("train")
    control = read(EXAM / "matched/fit.json")
    receipt = read(PUB / "corpus.json")
    assert sha(WORK / "cuts.npy") == receipt["files"]["cuts.npy"]
    cuts = np.load(WORK / "cuts.npy")
    rng = np.random.default_rng(SEED)
    batches = [
        order[i : i + 16]
        for order in [rng.permutation(len(rows)) for _ in range(4)]
        for i in range(0, len(rows), 16)
    ]
    digest = sha256(np.concatenate(batches).astype("<i8").tobytes()).hexdigest()
    assert digest == control["bindings"]["schedule"]
    keys = dict(
        schedule=digest,
        control=sha(EXAM / "matched/fit.json"),
        corpus=sha(PUB / "corpus.json"),
        trainer=sha(Path(__file__)),
        protocol=sha(ROOT / "references/branch_ink_protocol.md"),
        preflight=sha(PUB / "preflight.json"),
    )
    if not read(PUB / "preflight.json")["passed"]:
        raise ValueError("Conditional likelihood preflight failed")
    dest.mkdir(parents=True, exist_ok=True)
    state = dest / "resume.pt"
    final = dest / "writer.safetensors"
    if (dest / "fit.json").exists():
        previous = read(dest / "fit.json")
        if previous["bindings"] != keys or sha(final) != previous["weights"]:
            raise ValueError("Mismatched completed run")
        return
    torch.manual_seed(SEED)
    writer = Writer(device="cuda")
    assert np.array_equal(writer.palette, tiles[:n])
    base = writer.model.requires_grad_(False)
    x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
    bk = base.encode(x).detach()
    net = initial("dynamic", "cuda").train()
    opt = torch.optim.AdamW(
        net.parameters(), lr=0.00045, weight_decay=0.015, fused=True
    )
    start_step = 0
    trace = []
    elapsed = energy = 0.0
    if state.exists():
        saved = torch.load(state, map_location="cpu", weights_only=True)
        if saved["bindings"] != keys:
            raise ValueError("Checkpoint inputs differ")
        net.load_state_dict(saved["net"])
        opt.load_state_dict(saved["opt"])
        torch.set_rng_state(saved["cpu"])
        torch.cuda.set_rng_state(saved["cuda"])
        start_step, elapsed, energy, trace = (
            saved[k] for k in ("step", "seconds", "joules", "trace")
        )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    meter, start = snapshot(), monotonic()
    for j in range(start_step, len(batches)):
        idx = batches[j]
        group = [rows[i] for i in idx]
        whole = likelihood(
            net,
            base,
            x,
            bk,
            n,
            [r["qi"] for r in group],
            [r["pi"] for r in group],
            [r["y"] for r in group],
        )
        conditional = branch_log_prob(net, base, x, bk, n, group, cuts[idx])
        loss = -whole.mean() - conditional.sum() / len(group)
        opt.param_groups[0]["lr"] = (
            0.00045
            * min(1, (j + 1) / 100)
            * (0.15 + 0.85 * (1 + cos((j + 1) * pi / len(batches))) / 2)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        t = j + 1
        if t == 1 or t % 250 == 0 or t == len(batches):
            trace.append(
                dict(
                    step=t,
                    whole_nll=-whole.mean().item(),
                    extra=-conditional.sum().item() / len(group),
                )
            )
            print(trace[-1], flush=True)
        if t % 750 == 0 or t == len(batches):
            torch.cuda.synchronize()
            seconds = elapsed + monotonic() - start
            joules = energy + consumed(meter, snapshot())["joules"]
            peak = torch.cuda.max_memory_allocated()
            if seconds > 2700 or peak > 16 * 2**30:
                raise RuntimeError("Fit exceeded registered ceiling")
            record = dict(
                bindings=keys,
                step=t,
                seconds=seconds,
                joules=joules,
                peak_bytes=peak,
                trace=trace,
            )
            tmp = state.with_suffix(".tmp")
            torch.save(
                dict(
                    record,
                    net=net.state_dict(),
                    opt=opt.state_dict(),
                    cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(),
                ),
                tmp,
            )
            tmp.replace(state)
    if start_step == len(batches):
        record = {
            k: saved[k]
            for k in ("bindings", "step", "seconds", "joules", "peak_bytes", "trace")
        }
    tmp = final.with_suffix(".tmp")
    save_file(net.cpu().state_dict(), str(tmp))
    tmp.replace(final)
    dump(
        dest / "fit.json",
        dict(
            record,
            weights=sha(final),
            presentations=4 * len(rows),
            eligible_presentations=4 * int((cuts >= 0).sum()),
            parameters=sum(p.numel() for p in net.parameters())
            + sum(p.numel() for p in base.parameters()),
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=WORK / "fit")
    args = ap.parse_args()
    with gpu_owner():
        fit(args.run_dir)
