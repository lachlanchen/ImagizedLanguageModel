"""Contrast full answer likelihoods while fixing evidence and varying question order."""

from collections import defaultdict
import numpy as np
import torch
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_sentence_context import WORK, PUB, MODEL, Ink, read, dump
from scripts.evaluate_visual_discourse import unpack
from scripts.evaluate_sentence_context import models
from scripts.train_sentence_context import extended_page
from scripts.train_region_writer import pack, gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def case_scores(net, bank, ink, rows, replies, mode):
    cases = defaultdict(list)
    for i, r in enumerate(rows):
        cases[(r["task"], r["family"])].append(i)
    values = defaultdict(list)
    for (task, family), indices in cases.items():
        a, b = [rows[i] for i in indices]
        if a["blocks"] != b["blocks"]:
            raise ValueError("Different pages")
        page = [ink.encode(s) for s in a["blocks"]]
        y = [np.r_[ink.encode(r["answers"][0]), np.int32(1)] for r in (a, b)]
        original = [ink.encode(r["q"]) for r in (a, b)]
        full = []
        qs = []
        targets = []
        for idx in indices:
            o = replies[idx]
            start = o["offset"]
            fact = page[o["selected"]][start : start + o["fact_cells"]]
            assert b"".join(ink.tiles[i] for i in fact).hex() == o["fact_hex"]
            source = fact if mode == "original" else extended_page(page, fact)
            for questions in (
                original,
                [q[::-1] for q in original],
                [original[0], original[0]],
            ):
                for q in questions:
                    for ref in y:
                        qs.append(q)
                        targets.append(ref)
                        full.append(source)
        ctx, h = net.writer.encode(bank, net.n, pack(qs, "cuda"), pack(full, "cuda"))
        sc = (
            -net.writer.loss(ctx, h, pack(targets, "cuda"))
            .cpu()
            .numpy()
            .reshape(2, 3, 2, 2)
            .mean(0)
        )
        margin = sc[:, 0, 0] - sc[:, 0, 1] + sc[:, 1, 1] - sc[:, 1, 0]
        if abs(margin[2]) > 1e-5:
            raise ValueError("Duplicated question yields a spurious preference")
        values[task].append(
            dict(
                family=family,
                intact=float(margin[0]),
                reversed=float(margin[1]),
                duplicate=float(margin[2]),
                both=int(sc[0, 0, 0] > sc[0, 0, 1] and sc[0, 1, 1] > sc[0, 1, 0]),
            )
        )
    return values


def main():
    rows = read(WORK / "confirmation.json")
    ink = Ink(np.load(MODEL / "palette.npy"))
    for r in rows:
        for s in (r["q"], *r["blocks"], *r["answers"]):
            ink.encode(s)
    bank = torch.from_numpy(observations({"tiles": ink.array()}, "ink")).cuda()
    out = {}
    with gpu_owner(), torch.inference_mode():
        for k, ctor in models().items():
            result = case_scores(
                ctor(), bank, ink, rows, unpack(WORK / (k + ".json.gz"))["original"], k
            )
            summary = {}
            for task, recs in result.items():
                scores = {
                    s: np.array([r[s] for r in recs])
                    for s in ("intact", "reversed", "duplicate")
                }
                summary[task] = {
                    s: dict(
                        positive=int((v > 1e-6).sum()),
                        ties=int((abs(v) <= 1e-6).sum()),
                        mean=float(v.mean()),
                    )
                    for s, v in scores.items()
                }
                delta = scores["intact"] - scores["reversed"]
                draws = (
                    np.random.default_rng(3471)
                    .choice(delta, (5000, len(delta)), replace=True)
                    .mean(1)
                )
                summary[task].update(
                    n=len(recs),
                    both=int(sum(r["both"] for r in recs)),
                    margin_change=float(delta.mean()),
                    change_ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                )
            out[k] = dict(summary=summary, records=result)
    dump(PUB / "assignment.json", out)
    print({k: v["summary"] for k, v in out.items()})


if __name__ == "__main__":
    configure_cuda_cli()
    main()
