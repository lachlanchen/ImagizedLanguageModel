"""Fit an independent optical answer writer to human Chinese reading exams."""

import argparse
from hashlib import sha256
from math import cos, pi
from pathlib import Path
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file

from ilm.visual_lm.book_prefix import observations
from scripts.prepare_open_exam import WORK, PUB, MODEL, ROOT, read, dump, sha
from scripts.prepare_recurrent_evidence import MODEL as INIT
from scripts.write_visual_discourse import Writer, DEFAULT
from scripts.write_recurrent_evidence import load as original
from scripts.train_visual_cursor import base_batch, pad
from scripts.train_paired_evidence import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

SEED = 731231


def corpus(split):
    r = read(PUB / "corpus.json")
    for name in (split + ".json", "tiles.npy"):
        if sha(WORK / name) != r["files"][name]:
            raise ValueError("Changed exam corpus")
    return read(WORK / (split + ".json")), np.load(WORK / "tiles.npy"), r["canonical"]


def likelihood(net, base, features, bk, n, qs, ss, ys):
    device = features.device
    q, s, y = [torch.as_tensor(pad(a), device=device) for a in (qs, ss, ys)]
    ids, positions = [
        torch.as_tensor(v, device=device)
        for v in base_batch([p + q for p, q in zip(ss, qs, strict=True)], ys)
    ]
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        lp = base.log_prob(bk, ids, n, positions)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        score, _ = net.observed(features, q, s, y, lp, n)
    return score.float() / (y > 0).sum(-1)


def fit(arm, run=None):
    rows, tiles, n = corpus("train")
    rng = np.random.default_rng(SEED)
    batches = [
        order[i : i + 16]
        for order in [rng.permutation(len(rows)) for _ in range(4)]
        for i in range(0, len(rows), 16)
    ]
    source = dict(
        corpus=sha(PUB / "corpus.json"),
        init=sha(INIT / "dynamic.safetensors"),
        prior=sha(DEFAULT / "ink-pair.safetensors"),
        arm=arm,
        schedule=sha256(np.concatenate(batches).astype("<i8").tobytes()).hexdigest(),
        trainer=sha(Path(__file__)),
        protocol=sha(ROOT / "references/open_exam_protocol.md"),
    )
    dst = run or MODEL / arm
    dst.mkdir(parents=True, exist_ok=True)
    weight = dst / "writer.safetensors"
    receipt = dst / "fit.json"
    state = (run or WORK) / (arm + "-resume.pt")
    with gpu_owner():
        if receipt.exists():
            r = read(receipt)
            if r["bindings"] != source or r["weights"] != sha(weight):
                raise ValueError("Changed fitted writer")
            return
        torch.manual_seed(SEED)
        writer = Writer(device="cuda")
        if not np.array_equal(writer.palette, tiles[:n]):
            raise ValueError("Wrong glyph prior")
        base = writer.model.requires_grad_(False)
        x = torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda()
        bk = base.encode(x).detach()
        net = original("dynamic", "cuda").train()
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.00045, weight_decay=0.015, fused=True
        )
        trace = []
        begin = 0
        elapsed = joules = 0.0
        if state.exists():
            s = torch.load(state, weights_only=True, map_location="cpu")
            if s["bindings"] != source:
                raise ValueError("Resume binding differs")
            net.load_state_dict(s["net"])
            opt.load_state_dict(s["opt"])
            torch.set_rng_state(s["cpu"])
            torch.cuda.set_rng_state(s["cuda"])
            begin, elapsed, joules, trace = (
                s["step"],
                s["seconds"],
                s["joules"],
                s["trace"],
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, meter = monotonic(), snapshot()
        for j in range(begin, len(batches)):
            batch = [rows[i] for i in batches[j]]
            qs = [(r if arm == "matched" else rows[r["swap"]])["qi"] for r in batch]
            lp = likelihood(
                net,
                base,
                x,
                bk,
                n,
                qs,
                [r["pi"] for r in batch],
                [r["y"] for r in batch],
            )
            loss = -lp.mean()
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
                trace.append(dict(step=t, nll=loss.item()))
                print(arm, trace[-1], flush=True)
            if t % 750 == 0 or t == len(batches):
                torch.cuda.synchronize()
                sec = elapsed + monotonic() - start
                energy = joules + consumed(meter, snapshot())["joules"]
                if sec > 2700 or torch.cuda.max_memory_allocated() > 16 * 2**30:
                    raise RuntimeError("Resource ceiling exceeded")
                tmp = state.with_suffix(".tmp")
                torch.save(
                    dict(
                        net=net.state_dict(),
                        opt=opt.state_dict(),
                        step=t,
                        seconds=sec,
                        joules=energy,
                        trace=trace,
                        cpu=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state(),
                        bindings=source,
                    ),
                    tmp,
                )
                tmp.replace(state)
        tmp = weight.with_suffix(".tmp")
        save_file(net.cpu().state_dict(), str(tmp))
        tmp.replace(weight)
        dump(
            receipt,
            dict(
                bindings=source,
                weights=sha(weight),
                steps=len(batches),
                presentations=4 * len(rows),
                trainable=sum(p.numel() for p in net.parameters()),
                prior_parameters=sum(p.numel() for p in base.parameters()),
                seconds=sec,
                joules=energy,
                peak_bytes=torch.cuda.max_memory_allocated(),
                trace=trace,
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=("exchanged", "matched"))
    ap.add_argument("--run-dir", type=Path)
    a = ap.parse_args()
    fit(a.arm, a.run_dir)
