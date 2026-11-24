"""Fit matched writing models on real bilingual answers and general writing."""

import argparse
from pathlib import Path
import hashlib
import math
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.recurrent_evidence import ComposingWriter
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_recurrent_evidence import WORK, PUBLIC, MODEL, dump, read, sha
from scripts.train_paired_evidence import (
    Pairs,
    Corpus,
    byte_features,
    load_model,
    batch as paired_batch,
)
from scripts.write_paired_evidence import reader
from scripts.train_visual_cursor import pad, base_batch
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

STEPS = 3200
SEED = 851037


class Data:
    def __init__(self):
        self.spec = read(PUBLIC / "corpus.json")
        for f, h in self.spec["files"].items():
            if sha(WORK / f) != h:
                raise ValueError("Changed composition corpus " + f)
        with np.load(WORK / "train.npz") as z:
            self.a = {k: z[k] for k in z.files}
        rows = read(WORK / "rows.json")
        self.ids = {
            lang: [i for i, r in enumerate(rows) if r[0] == lang]
            for lang in ("en", "zh")
        }

    def seq(self, k, i):
        l, r = self.a[k + "o"][i : i + 2]
        return self.a[k][l:r]


def batch(data, paired, old, step):
    phase = (step - 1) % 10
    if phase >= 7:
        # Draw paired human QA examples and open writing with the original layout.
        # Map cycles across all four QA sources; the open batch contains all tasks.
        seq = 10 * (step // 10) + ((step // 10 + phase) % 8 + 1 if phase < 9 else 10)
        q, s, y, body, loc = paired_batch(paired, old, seq)
        # Shuffle before subsampling so all open-writing tasks remain eligible.
        idx = np.random.default_rng([SEED, step]).choice(len(q), 20, replace=False)
        return tuple([[a[i] for i in idx] for a in (q, s, y, body)]) + (loc[idx],)
    lang = "zh" if phase < 4 else "en"
    ids = np.random.default_rng([SEED, step]).choice(data.ids[lang], 20)
    q, s, y = ([data.seq(k, i) for i in ids] for k in ("q", "s", "y"))
    return (
        q,
        s,
        y,
        [np.r_[a, b] for a, b in zip(s, q, strict=True)],
        np.full((len(q), 2), -1),
    )


def fit(arm, data, paired, old, run_dir=None):
    path = (run_dir or MODEL) / (arm + ".safetensors")
    rc = (run_dir or PUBLIC) / (arm + ".json")
    if rc.exists():
        if sha(path) != read(rc)["weights"]:
            raise ValueError("Changed weights")
        return
    byte = arm == "byte"
    x = (
        byte_features(data.a["chars"])
        if byte
        else torch.from_numpy(observations({"tiles": data.a["tiles"]}, "ink"))
    ).cuda()
    base = load_model("byte-pair" if byte else "ink-pair", "cuda").requires_grad_(False)
    bk = base.encode(x)
    prior = reader("byte-evidence" if byte else "evidence", "cpu")
    torch.manual_seed(SEED)
    net = ComposingWriter(dynamic=arm != "fixed")
    diff = net.load_state_dict(prior.state_dict(), strict=False)
    if diff.unexpected_keys or set(diff.missing_keys) != {
        k for k in net.state_dict() if k not in prior.state_dict()
    }:
        raise ValueError("Checkpoint mismatch")
    net.cuda().train()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0007, weight_decay=0.01, fused=True)
    trace = []
    digest = bytes(32)
    total = 0
    meter = snapshot()
    torch.cuda.synchronize()
    start = monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, STEPS + 1):
        qs, ss, ys, body, loc = batch(data, paired, old, step)
        qa, sa, ya = map(pad, (qs, ss, ys))
        total += sum(map(len, ys))
        digest = hashlib.sha256(
            digest + qa.tobytes() + sa.tobytes() + ya.tobytes() + loc.tobytes()
        ).digest()
        q, s, y = (torch.from_numpy(v).cuda() for v in (qa, sa, ya))
        bi, bp = [torch.from_numpy(v).cuda() for v in base_batch(body, ys)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            blp = base.log_prob(bk, bi, old.n, bp)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logp, ev = net.observed(x, q, s, y, blp, old.n)
            loss = -(logp / (y > 0).sum(-1)).mean()
            l, r = torch.from_numpy(loc).cuda().T
            pos = torch.arange(s.size(1), device=s.device)
            visible = (
                (pos[None] >= l[:, None]) & (pos[None] < r[:, None]) & (l[:, None] >= 0)
            )
            aux = -(ev.log_softmax(-1) * visible).sum(-1) / visible.sum(-1).clamp_min(1)
            loss = loss + 0.45 * aux.sum() / visible.any(-1).sum().clamp_min(1)
        opt.param_groups[0]["lr"] = (
            0.0007
            * min(1, step / 80)
            * (0.2 + 0.8 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 100 == 0:
            trace.append([step, float(loss.detach())])
            print(arm, step, trace[-1][-1], round(monotonic() - start, 1), flush=True)
            if torch.cuda.max_memory_allocated() > 12 * 2**30:
                raise RuntimeError("GPU ceiling exceeded")
    torch.cuda.synchronize()
    secs = monotonic() - start
    energy = consumed(meter, snapshot())
    path.parent.mkdir(exist_ok=True, parents=True)
    save_file(net.cpu().state_dict(), str(path))
    dump(
        rc,
        dict(
            arm=arm,
            updates=STEPS,
            parameters=sum(p.numel() for p in net.parameters()),
            target_cells=total,
            schedule=digest.hex(),
            seconds=secs,
            joules=energy["joules"],
            peak_bytes=torch.cuda.max_memory_allocated(),
            source=sha(PUBLIC / "corpus.json"),
            weights=sha(path),
            trace=trace,
        ),
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, help="Store separate fitted heads and their receipts."
    )
    args = parser.parse_args()
    configure_cuda_cli()
    data = Data()
    paired = Pairs()
    old = Corpus()
    with gpu_owner():
        for arm in ("fixed", "dynamic", "byte"):
            fit(arm, data, paired, old, args.run_dir)
