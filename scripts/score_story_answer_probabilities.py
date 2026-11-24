"""Measure held stopped-answer likelihood and candidate geometry under each trained selector."""

from collections import defaultdict
import numpy as np
import torch
from ilm.visual_lm.answer_evidence import answer_marginal
from ilm.visual_lm.book_prefix import observations
from scripts.check_story_answer_objective import likelihoods
from scripts.evaluate_story_answer_fit import ARMS
from scripts.prepare_story_answer_fit import WORK, PUB, read, dump, sha
from scripts.prepare_fact_channel import MODEL as BASE, Ink
from scripts.write_directional_reading import DirectionalAnswer
from scripts.train_story_answer_fit import restore
from scripts.train_fact_channel import gpu_owner, join_regions
from scripts.verse_gpu_runtime import configure_cuda_cli


def main():
    configure_cuda_cli()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    f = WORK / "confirmation.json"
    assert sha(f) == read(PUB / "reservation.json")["files"][f.name]
    ink = Ink(np.load(BASE / "palette.npy"))
    rows = read(f)
    items = [
        (
            ink.encode(r["q"]),
            join_regions([ink.encode(s) for s in r["blocks"]]),
            list(dict.fromkeys(tuple(ink.encode(a)) + (1,) for a in r["answers"])),
        )
        for r in rows
    ]
    by = defaultdict(list)
    traces = []
    with gpu_owner(), torch.inference_mode():
        net = DirectionalAnswer("signed", "cuda")
        heads = {m: restore(m, "cuda") for m in ARMS[1:]}
        feat = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
        for r, (q, p, refs) in zip(rows, items, strict=True):
            im, b, l, v = likelihoods(net, feat, q, p, refs, "cuda", return_inputs=True)
            if not v.isfinite().any():
                traces.append(dict(source_id=r["source_id"], supported=False))
                continue
            record = {}
            for m in ARMS:
                logits = b if m == "signed" else heads[m](im[None], b[None], l[None])[0]
                nll = float(-answer_marginal(logits, v))
                record[m] = dict(nll=nll, choice=int(logits.argmax()))
                by[m].append(nll)
            traces.append(dict(source_id=r["source_id"], supported=True, models=record))
    means = {m: float(np.mean(v)) for m, v in by.items()}
    intervals = {}
    for m in ARMS[:2]:
        pairs = []
        for i in range(0, len(traces), 2):
            group = [
                v["models"][m]["nll"] - v["models"]["marginal"]["nll"]
                for v in traces[i : i + 2]
                if v["supported"]
            ]
            if group:
                pairs.append(group)
        rng = np.random.default_rng(41261)
        sums = np.array([sum(v) for v in pairs])
        counts = np.array([len(v) for v in pairs])
        ix = rng.integers(0, len(pairs), (5000, len(pairs)))
        d = sums[ix].sum(1) / counts[ix].sum(1)
        intervals[m] = list(np.quantile(d, [0.025, 0.975]))
    dump(
        PUB / "held-probabilities.json",
        dict(
            n=len(rows),
            supported=len(by["signed"]),
            mean_nll=means,
            interval_nll_reduction=intervals,
            rows=traces,
        ),
    )


if __name__ == "__main__":
    main()
