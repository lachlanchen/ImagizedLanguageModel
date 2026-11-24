"""Fit an independent visual writer on complete bilingual passages and evidence views."""

import argparse
from hashlib import sha256
from math import cos, pi
from pathlib import Path
from time import monotonic

import numpy as np
from safetensors.torch import save_file
import torch

from ilm.visual_lm.book_prefix import observations
from scripts.measure_writer_energy import snapshot, consumed
from scripts.prepare_evidence_views import WORK, PUB, ROOT, read, sha
from scripts.prepare_open_exam import dump
from scripts.prepare_recurrent_evidence import MODEL as INIT
from scripts.train_open_exam import likelihood
from scripts.train_paired_evidence import gpu_owner
from scripts.write_recurrent_evidence import load as original
from scripts.write_visual_discourse import Writer, DEFAULT

SEED = 902615


def schedule(meta):
    rng = np.random.default_rng(SEED)
    batches = []
    for cropped in (0, 1):
        for lang in (0, 1):
            order = rng.permutation(np.flatnonzero(meta[:, 3] == lang))
            batches.extend(
                (cropped, order[i : i + 16]) for i in range(0, len(order), 16)
            )
    return [batches[i] for i in rng.permutation(len(batches))]


def fit(arm, dest):
    spec = read(PUB / "corpus.json")
    path = WORK / "train.npz"
    if sha(path) != spec["files"]["train.npz"]:
        raise ValueError("Changed training pages")
    with np.load(path) as store:
        arr = {k: store[k] for k in store.files}

    def sequence(k, i):
        lo, hi = arr[k + "ptr"][i : i + 2]
        return arr[k][lo:hi].tolist()

    batches = schedule(arr["meta"])
    order = sha256()
    for v, idx in batches:
        order.update(bytes([v, len(idx)]))
        order.update(idx.astype("<i8").tobytes())
    binding = dict(
        arm=arm,
        order=order.hexdigest(),
        corpus=sha(PUB / "corpus.json"),
        initial=sha(INIT / "dynamic.safetensors"),
        prior=sha(DEFAULT / "ink-pair.safetensors"),
        trainer=sha(Path(__file__)),
        protocol=sha(ROOT / "references/evidence_views_protocol.md"),
        likelihood=sha(ROOT / "scripts/train_open_exam.py"),
        writer=sha(ROOT / "ilm/visual_lm/recurrent_evidence.py"),
    )
    dest.mkdir(parents=True, exist_ok=True)
    state = dest / "resume.pt"
    weight = dest / "writer.safetensors"
    receipt = dest / "fit.json"
    with gpu_owner():
        if receipt.exists():
            old = read(receipt)
            if old["bindings"] != binding or old["weights"] != sha(weight):
                raise ValueError("Changed completed fit")
            return
        torch.manual_seed(SEED)
        writer = Writer(device="cuda")
        base = writer.model.requires_grad_(False)
        n = spec["canonical"]
        if not np.array_equal(writer.palette, arr["tiles"][:n]):
            raise ValueError("Changed optical palette")
        x = torch.from_numpy(observations({"tiles": arr["tiles"]}, "ink")).cuda()
        bk = base.encode(x).detach()
        net = original("dynamic", "cuda").train()
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.0003, weight_decay=0.015, fused=True
        )
        begin = 0
        trace = []
        elapsed = energy = 0.0
        peak = 0
        if state.exists():
            saved = torch.load(state, map_location="cpu", weights_only=True)
            if saved["bindings"] != binding:
                raise ValueError("Incompatible optimizer checkpoint")
            net.load_state_dict(saved["net"])
            opt.load_state_dict(saved["opt"])
            torch.set_rng_state(saved["cpu"])
            torch.cuda.set_rng_state(saved["cuda"])
            begin, trace, elapsed, energy, peak = [
                saved[k] for k in ("step", "trace", "seconds", "joules", "peak")
            ]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = monotonic()
        meter = snapshot()
        for j in range(begin, len(batches)):
            cropped, idx = batches[j]
            qs = []
            ps = []
            ys = []
            for i in idx:
                doc, lo, hi, zh = arr["meta"][i]
                p = sequence("p", doc)
                if arm == "highlight" and cropped and not zh:
                    p = p[lo:hi]
                qs.append(sequence("q", i))
                ps.append(p)
                ys.append(sequence("y", i))
            loss = -likelihood(net, base, x, bk, n, qs, ps, ys).mean()
            t = j + 1
            opt.param_groups[0]["lr"] = (
                0.0003
                * min(1, t / 200)
                * (0.15 + 0.85 * (1 + cos(t * pi / len(batches))) / 2)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            if t == 1 or t % 250 == 0 or t == len(batches):
                trace.append(
                    dict(step=t, nll=loss.item(), seconds=elapsed + monotonic() - start)
                )
                print(arm, t, len(batches), trace[-1], flush=True)
            if t % 500 == 0 or t == len(batches):
                torch.cuda.synchronize()
                sec = elapsed + monotonic() - start
                joules = energy + consumed(meter, snapshot())["joules"]
                peak = max(peak, torch.cuda.max_memory_allocated())
                if sec > 10800 or peak > 16 * 2**30:
                    raise RuntimeError("Fit exceeded resource ceiling")
                tmp = state.with_suffix(".tmp")
                torch.save(
                    dict(
                        bindings=binding,
                        net=net.state_dict(),
                        opt=opt.state_dict(),
                        step=t,
                        trace=trace,
                        seconds=sec,
                        joules=joules,
                        peak=peak,
                        cpu=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state(),
                    ),
                    tmp,
                )
                tmp.replace(state)
        saved = torch.load(state, map_location="cpu", weights_only=True)
        tmp = weight.with_suffix(".tmp")
        save_file(saved["net"], str(tmp))
        tmp.replace(weight)
        dump(
            receipt,
            dict(
                bindings=binding,
                weights=sha(weight),
                steps=len(batches),
                presentations=2 * len(arr["meta"]),
                trainable=sum(p.numel() for p in net.parameters()),
                prior_parameters=sum(p.numel() for p in base.parameters()),
                seconds=saved["seconds"],
                joules=saved["joules"],
                peak_bytes=saved["peak"],
                trace=saved["trace"],
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("whole", "highlight"))
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    fit(args.arm, args.run_dir or WORK / args.arm)
