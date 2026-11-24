"""Fit two answer-image densities with identical question pairs and ordinary writing."""

import hashlib
import math
from time import monotonic
import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.paired_evidence import EvidenceWriter
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_paired_evidence import WORK, PUB, MODEL, read, dump, sha
from scripts.train_visual_discourse import Corpus, load_model
from scripts.train_visual_cursor import pad, base_batch
from scripts.train_span_choice import gpu_owner
from scripts.measure_writer_energy import snapshot, consumed

STEPS = 4800
SEED = 854019


class Pairs:
    def __init__(self):
        spec = read(PUB / "corpus.json")
        for f, h in spec["sha256"].items():
            if sha(WORK / f) != h:
                raise ValueError("Changed corpus " + f)
        with np.load(WORK / "corpus.npz") as z:
            self.a = {k: z[k] for k in z.files}
        groups = read(WORK / "groups.json")
        self.groups = {}
        for k, items in groups.items():
            grouped = []
            for ids in items:
                labels = {}
                for i in ids:
                    labels.setdefault(self.seq("y", i).tobytes(), []).append(i)
                if len(labels) > 1:
                    grouped.append(list(labels.values()))
            self.groups[k] = grouped

    def seq(self, field, i):
        l, r = self.a[field + "o"][i : i + 2]
        return self.a[field][l:r]


def batch(data, old, step):
    rng = np.random.default_rng([SEED, step])
    phase = (step - 1) % 10
    qs = []
    ss = []
    ys = []
    loc = []
    body = []
    if phase < 8:
        k = ("squad", "coqa", "cmrc", "drcd")[phase % 4]
        groups = data.groups[k]
        for ix in rng.integers(0, len(groups), 12):
            g = groups[ix]
            for j in rng.choice(len(g), 2, replace=False):
                i = rng.choice(g[j])
                doc, l, r = data.a["labels"][i]
                q = data.seq("q", i)
                s = data.seq("p", doc)
                qs.append(q)
                ss.append(s)
                ys.append(data.seq("y", i)[:49])
                loc.append((l, r))
                body.append(np.r_[s, q])
    else:
        for k in (
            "book-en",
            "book-zh",
            "instruction-en",
            "instruction-zh",
            "reference",
        ):
            for i in rng.choice(
                old.groups[k], 4 if k == "reference" else 5, p=old.books.get(k)
            ):
                seq = old.seq(i)
                cut = (
                    old.rows[i]["cut"]
                    if not k.startswith("book")
                    else rng.integers(1, len(seq) - 1)
                )
                x = seq[max(0, cut - 600) : cut]
                qs.append(x[-160:])
                ss.append(x)
                body.append(x)
                ys.append(seq[cut : cut + 49])
                loc.append((-1, -1))
    return qs, ss, ys, body, np.asarray(loc)


def fit(data, old, features, base, bk, arm):
    fp = MODEL / (arm + ".safetensors")
    rc = PUB / (arm + ".json")
    if rc.exists():
        if read(rc)["weights"] != sha(fp):
            raise ValueError("Changed weights")
        return
    torch.manual_seed(SEED)
    model = EvidenceWriter().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02, fused=True)
    digest = bytes(32)
    trace = []
    total = 0
    marked = 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = monotonic()
    meter = snapshot()
    for step in range(1, STEPS + 1):
        q, s, y, body, loc = batch(data, old, step)
        a, b, c = pad(q), pad(s), pad(y)
        digest = hashlib.sha256(
            digest + a.tobytes() + b.tobytes() + c.tobytes() + loc.tobytes()
        ).digest()
        total += sum(map(len, y))
        marked += int((loc[:, 0] >= 0).sum())
        inputs = [torch.as_tensor(v, device="cuda") for v in (a, b, c)]
        left, right = torch.as_tensor(loc, device="cuda").T
        bi, bp = (torch.as_tensor(v, device="cuda") for v in base_batch(body, y))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lp = base.log_prob(bk, bi, old.n, bp)
        opt.param_groups[0]["lr"] = (
            0.001
            * min(1, step / 80)
            * (0.15 + 0.85 * (1 + math.cos(math.pi * step / STEPS)) / 2)
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(
                features,
                *inputs,
                lp,
                old.n,
                left,
                right,
                0.45 if arm in ("evidence", "byte-evidence") else 0.0,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        opt.step()
        if step == 1 or step % 100 == 0:
            trace.append([step, float(loss.detach())])
            print(
                arm,
                step,
                round(trace[-1][1], 3),
                round(monotonic() - start, 2),
                flush=True,
            )
            if torch.cuda.max_memory_allocated() > 16 * 2**30:
                raise RuntimeError("Memory ceiling")
    torch.cuda.synchronize()
    cost = consumed(meter, snapshot())
    secs = monotonic() - start
    save_file(model.cpu().state_dict(), str(fp))
    dump(
        rc,
        dict(
            weights=sha(fp),
            schedule=digest.hex(),
            steps=STEPS,
            supervised_positions=int(total),
            visible_evidence_exposures=marked,
            corpus=sha(PUB / "corpus.json"),
            seconds=secs,
            joules=cost["joules"],
            peak_allocated=torch.cuda.max_memory_allocated(),
            parameters=sum(p.numel() for p in model.parameters()),
            trace=trace,
        ),
    )


def byte_features(chars):
    by = np.zeros((len(chars), 4), np.uint8)
    for i, ch in enumerate(chars):
        raw = str(ch).encode()
        by[i, : len(raw)] = list(raw)
    return torch.from_numpy(observations({"bytes": by}, "byte"))


def main():
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    data = Pairs()
    old = Corpus()
    MODEL.mkdir(exist_ok=True, parents=True)
    with gpu_owner():
        features = torch.from_numpy(
            observations({"tiles": data.a["tiles"]}, "ink")
        ).cuda()
        base = load_model("ink-pair", "cuda").requires_grad_(False)
        bk = base.encode(features)
        for arm in ("answer", "evidence"):
            fit(data, old, features, base, bk, arm)


if __name__ == "__main__":
    main()
