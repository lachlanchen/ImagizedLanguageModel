"""Compare free name proposals and a supplied cast on the same held source images."""

import gzip
import json
from collections import defaultdict
import torch

from scripts.prepare_story_mentions import LANGS, PUBLIC, dump
from scripts.train_masked_mentions import prepare, net_for
from scripts.train_semantic_page import gpu_owner
from scripts.evaluate_story_mentions import grade, metrics, switches
from scripts.story_mention_options import select
from scripts.write_story_mentions import tiles
from scripts.prepare_visual_sequence_corpus import Rasterizer


@torch.inference_mode()
def evaluate(split):
    dec = defaultdict(list)
    raster = Rasterizer()
    for mode in ("ink", "byte"):
        rows, _, obs, (ids, slots, _, bounds, sizes), cases = prepare(split, mode)
        for arm in [mode, "geometry"] if mode == "ink" else [mode]:
            net = net_for(arm, "cuda")
            for start in range(0, len(ids), 64):
                ix = slice(start, start + 64)
                x = obs[ids[ix].long()]
                if arm == "geometry":
                    x = torch.zeros_like(x)
                logits = net(x, slots[ix], bounds[ix], sizes[ix]).cpu().numpy()
                for k, line in enumerate(logits):
                    i, r = divmod(start + k, 2)
                    row = rows[i]
                    route = ("free", "cast")[r]
                    source = tiles(row["text"], raster)
                    s, spans, q = cases[i, route]
                    pred = select(line, s, spans, q[0])
                    key = arm + "-" + route
                    dec[key].append(grade(row, source, pred))
                    if arm == "ink":
                        cand = [
                            tuple(ab) for j, ab in enumerate(spans[: q[0]]) if s[j] >= 0
                        ]
                        near = cand[-1] if cand else None
                        dec["near-" + route].append(grade(row, source, near))
                        origin = [
                            (a, b) for a, b, e in row["names"] if e == row["entity"]
                        ]
                        universe = {source[slice(*t)].tobytes() for t in origin}
                        good = next(
                            (
                                ab
                                for ab in cand
                                if source[slice(*ab)].tobytes() in universe
                            ),
                            None,
                        )
                        dec["ceiling-" + route].append(grade(row, source, good))
    out = {}
    for lang in LANGS:
        idx = [i for i, row in enumerate(rows) if row["lang"] == lang]
        pairs = switches(rows, idx)
        stats = {}
        for key, s in dec.items():
            score = metrics([s[i] for i in idx])
            near = dec["near-" + key.split("-")[-1]]
            for name, part in (
                ("switched", pairs),
                ("same_nearest", [p for p in pairs if p[2]]),
                (
                    "same_route_nearest",
                    [
                        p
                        for p in pairs
                        if near[p[0]]["bitmap_sha256"] == near[p[1]]["bitmap_sha256"]
                    ],
                ),
            ):
                score[name] = [
                    sum(s[i]["ink"] and s[j]["ink"] for i, j, _ in part),
                    len(part),
                ]
            stats[key] = score
        out[lang] = dict(
            arms=stats,
            works={
                d: {
                    k: metrics([v[i] for i in idx if rows[i]["doc"] == d])
                    for k, v in dec.items()
                }
                for d in sorted({rows[i]["doc"] for i in idx})
            },
        )
    dump(PUBLIC / ("masked-" + split + "-scores.json"), out)
    (PUBLIC / ("masked-" + split + "-decisions.json.gz")).write_bytes(
        gzip.compress(json.dumps(dec, separators=(",", ":")).encode(), mtime=0)
    )
    print(
        split,
        {l: {k: v["correct"] for k, v in t["arms"].items()} for l, t in out.items()},
        flush=True,
    )


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    with gpu_owner():
        for split in ("dev", "test"):
            evaluate(split)
