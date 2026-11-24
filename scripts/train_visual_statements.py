"""Fit the visual sentence-writing stage on human declarations."""

import argparse
from pathlib import Path
import hashlib
import math
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_visual_statements import WORK, PUB, MODEL, read, dump, sha
from scripts.write_recurrent_evidence import load
from scripts.train_paired_evidence import load_model
from scripts.train_visual_cursor import pad, base_batch
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

STEPS = 6400


class Statements:
    def __init__(self):
        for f, h in read(PUB / "corpus.json")["files"].items():
            if sha(WORK / f) != h:
                raise ValueError("Changed data " + f)
        with np.load(WORK / "train.npz") as z:
            self.a = {k: z[k] for k in z.files}
        self.space = next(
            i for i, v in enumerate(self.a["tiles"]) if i >= 2 and not v.any()
        )
        self.n = len(np.load(MODEL.parent / "visual_discourse/glyphs.npy"))

    def seq(self, k, i):
        l, r = self.a[k + "o"][i : i + 2]
        return self.a[k][l:r].tolist()


def fact_page(q, f, space):
    return f + [space, space] + q


def main(run_dir=None):
    data = Statements()
    out = (run_dir or MODEL) / "statement.safetensors"
    rc = (run_dir or PUB) / "training.json"
    if rc.exists():
        if sha(out) != read(rc)["weights"]:
            raise ValueError("Changed statement model")
        return
    with gpu_owner():
        torch.manual_seed(475211)
        net = load("dynamic", "cuda").train()
        x = torch.from_numpy(observations({"tiles": data.a["tiles"]}, "ink")).cuda()
        base = load_model("ink-pair", "cuda").requires_grad_(False)
        bk = base.encode(x)
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.0008, weight_decay=0.01, fused=True
        )
        torch.cuda.synchronize()
        start = monotonic()
        meter = snapshot()
        torch.cuda.reset_peak_memory_stats()
        trace = []
        total = 0
        digest = bytes(32)
        n = len(data.a["qo"]) - 1
        for step in range(1, STEPS + 1):
            ids = np.random.default_rng([475211, step]).integers(n, size=24)
            qs, fs, ys = ([data.seq(k, i) for i in ids] for k in ("q", "f", "y"))
            ss = [fact_page(q, f, data.space) for q, f in zip(qs, fs, strict=True)]
            a, b, c = map(pad, (qs, ss, ys))
            total += sum(map(len, ys))
            digest = hashlib.sha256(
                digest + a.tobytes() + b.tobytes() + c.tobytes()
            ).digest()
            q, s, y = [torch.from_numpy(v).cuda() for v in (a, b, c)]
            bi, bp = [torch.from_numpy(v).cuda() for v in base_batch(ss, ys)]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                blp = base.log_prob(bk, bi, data.n, bp)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mass, _ = net.observed(x, q, s, y, blp, data.n)
                loss = -(mass / (y > 0).sum(-1)).mean()
            opt.param_groups[0]["lr"] = (
                0.0008
                * min(1, step / 80)
                * (0.15 + 0.85 * (1 + math.cos(math.pi * step / STEPS)) / 2)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1, error_if_nonfinite=True)
            opt.step()
            if step == 1 or step % 200 == 0:
                trace.append([step, float(loss.detach())])
                print(step, trace[-1][-1], round(monotonic() - start, 1), flush=True)
                if torch.cuda.max_memory_allocated() > 12 * 2**30:
                    raise RuntimeError("GPU memory ceiling")
        torch.cuda.synchronize()
        seconds = monotonic() - start
        energy = consumed(meter, snapshot())
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(net.cpu().state_dict(), str(out))
        dump(
            rc,
            dict(
                steps=STEPS,
                examples=STEPS * 24,
                target_cells=total,
                seconds=seconds,
                joules=energy["joules"],
                peak_bytes=torch.cuda.max_memory_allocated(),
                parameters=sum(p.numel() for p in net.parameters()),
                schedule=digest.hex(),
                weights=sha(out),
                trace=trace,
                corpus=sha(PUB / "corpus.json"),
            ),
        )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Store a new fit and its receipt together in this directory.",
    )
    args = parser.parse_args()
    configure_cuda_cli()
    main(args.run_dir)
