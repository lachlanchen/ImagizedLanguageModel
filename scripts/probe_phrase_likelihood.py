"""Compute exact stopped-image likelihood with every compatible segmentation."""

import math
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from ilm.visual_lm.phrase_region import source_runs, LENGTHS, extend_matches
from ilm.visual_lm.book_prefix import observations
from scripts.prepare_phrase_confirmation import WORK, PUB
from scripts.train_phrase_region import MODEL, PARENT
from scripts.prepare_region_writer import Ink
from scripts.evaluate_visual_discourse import unpack
from scripts.prepare_open_exam import read, dump
from scripts.write_region_writer import phrase_model, model, bitmap_ids
from scripts.train_region_writer import pack, gpu_owner


def path_scores(probabilities):
    size = len(probabilities)
    alpha = np.full(size + 1, -np.inf)
    alpha[0] = 0.0
    for t, choices in enumerate(probabilities):
        for k, (length, mass) in enumerate(zip(LENGTHS, choices)):
            if t + length <= size and mass > 0:
                alpha[t + length] = np.logaddexp(
                    alpha[t + length], alpha[t] + math.log(mass)
                )
    pos = 0
    greedy = 0.0
    while pos < size:
        valid = [
            (l, p)
            for l, p in zip(LENGTHS, probabilities[pos])
            if p > 0 and pos + l <= size
        ]
        if not valid:
            return float(alpha[-1]), -math.inf
        length, prob = valid[-1]
        greedy += math.log(prob)
        pos += length
    if greedy > alpha[-1] + 1e-7:
        raise ValueError("Path sum below a compatible path")
    return float(alpha[-1]), float(greedy)


def masses(net, ctx, h, seq, phrase):
    ids = torch.tensor([seq], device=h.device)
    inputs = torch.cat([net.bos[None, None, :], ctx[0][ids[:, :-1]]], 1)
    state, _ = net.gru(inputs, h)
    history = torch.stack([F.pad(ids, (i, 0))[:, : ids.size(1)] for i in (1, 2, 3)], -1)
    logp, ptr, gate = net.distribution(ctx, state, history)
    synth = (
        logp.exp()
        .gather(-1, (ids - 1).clamp(0, logp.shape[-1] - 1).unsqueeze(-1))
        .squeeze(-1)
    )
    synth = synth.masked_fill(ids > logp.shape[-1], 0) * gate.sigmoid()
    matches = ids[:, :, None].eq(ctx[2][:, None, :]) & ctx[2][:, None, :].ge(2)
    if not phrase:
        return (
            (synth + (ptr * matches).sum(-1) * (-gate).sigmoid())[0]
            .cpu()
            .double()
            .numpy()[:, None]
        )
    valid = [r.ge(2).all(-1) for r in source_runs(ctx[2])]
    lengths = net.length_logprobs(state, valid).exp()
    mass = []
    for k, l in enumerate(LENGTHS):
        if k:
            matches = extend_matches(matches, l // 2)
        v = (ptr * matches).sum(-1) / (ptr * valid[k][:, None, :]).sum(-1).clamp_min(
            1e-30
        )
        v = v * lengths[:, :, k] * (-gate).sigmoid()
        mass.append(v + synth if k == 0 else v)
    return torch.stack(mass, -1)[0].cpu().double().numpy()


def main():
    rows = read(WORK / "confirmation.json")
    replies = unpack(WORK / "confirmation-phrase.json.gz")["original"]
    palette = np.load(PARENT / "palette.npy")
    ink = Ink(palette)
    results = defaultdict(list)
    with gpu_owner(), torch.inference_mode():
        torch.set_float32_matmul_precision("highest")
        one = model("selected", "cuda")
        multi = phrase_model(MODEL, "cuda")
        for r, reply in zip(rows, replies, strict=True):
            q = pack([ink.encode(r["q"])], "cuda")
            p = pack([ink.encode(r["blocks"][reply["selected"]])], "cuda")
            ref = ink.encode(r["answers"][0]) + [1]
            emitted = bitmap_ids(
                ink,
                np.frombuffer(bytes.fromhex(reply["ink_hex"]), np.uint8).reshape(
                    -1, 128
                ),
            ) + [1]
            features = torch.from_numpy(
                observations({"tiles": ink.array()}, "ink")
            ).cuda()
            ctx, h = multi.encode(features, len(palette), q, p)
            total, path = path_scores(masses(multi, ctx, h, ref, True))
            written, _ = (
                path_scores(masses(multi, ctx, h, emitted, True))
                if reply["stop"]
                else (-math.inf, -math.inf)
            )
            c, b = one.encode(features, len(palette), q, p)
            base, _ = path_scores(masses(one, c, b, ref, False))
            row = dict(
                cells=len(ref),
                supported=math.isfinite(total),
                single_supported=math.isfinite(base),
                reply_stopped=reply["stop"],
                different=ref != emitted,
            )
            for k, n in (
                ("reference", total),
                ("canonical", path),
                ("single", base),
                ("emitted", written),
            ):
                row[k] = n if math.isfinite(n) else None
            results[r["task"]].append(row)
    scores = {}
    for t, g in results.items():
        common = [r for r in g if r["supported"] and r["single_supported"]]
        different = [r for r in common if r["different"] and r["emitted"] is not None]
        scores[t] = dict(
            n=len(g),
            common_support=len(common),
            mean_nats_per_glyph={
                k: float(np.mean([-r[k] / r["cells"] for r in common]))
                for k in ("reference", "single", "canonical")
            },
            comparison_different_stopped=len(different),
            prefer_human=sum(r["reference"] > r["emitted"] for r in different),
            positive_canonical_slack=sum(
                r["reference"] - r["canonical"] > 1e-6 for r in common
            ),
        )
    dump(PUB / "likelihood.json", dict(scores=scores, examples=results))
    print(scores)


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
