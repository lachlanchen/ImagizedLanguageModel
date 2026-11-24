"""Compare locally generated optical and quantized visual replies on the same printed input."""

from concurrent.futures import ThreadPoolExecutor
from time import monotonic
import numpy as np
import torch
from scripts.benchmark_competing_qwen import png, MODELS, STORE, server
from scripts.benchmark_open_exam import VLM, ask
from scripts.evaluate_region_writer import contains_fact
from scripts.train_region_writer import gpu_owner
from scripts.prepare_fact_channel import WORK, PUB, MODEL, read, dump, sha, Ink
from scripts.write_fact_channel import VisualAnswer
from scripts.measure_writer_energy import snapshot, consumed
from scripts.audit_region_writer import wording
from scripts.verse_gpu_runtime import configure_cuda_cli


def main(work=WORK, public=PUB, models=None):
    allrows = read(work / "confirmation.json")
    rows = [
        r
        for t in ("statements", "exam")
        for r in [x for x in allrows if x["task"] == t][:16]
    ]
    ink = Ink(np.load(MODEL / "palette.npy"))
    qs = [ink.encode(r["q"]) for r in rows]
    bs = [[ink.encode(s) for s in r["blocks"]] for r in rows]
    tiles = ink.array()
    chars = dict(enumerate(read(MODEL / "readback.json")))
    for c, i in ink.chars.items():
        chars.setdefault(i, c)
    messages = []
    for q, blocks in zip(qs, bs):
        grid = []
        for j, b in enumerate(blocks):
            if j:
                grid.extend([0] * ((-len(grid)) % 32 + 32))
            grid.extend(b)
        messages.append(
            dict(
                role="user",
                content="The first image is the question. The second is the source passage. Reply in the language of the question. For English write one complete statement; for Chinese give the short answer requested. Output only the answer.",
                images=[png(tiles, q), png(tiles, grid)],
            )
        )
    if sha(STORE / "blobs" / ("sha256-" + MODELS[VLM])) != MODELS[VLM]:
        raise ValueError("Changed comparator")
    runs = {}
    costs = {}
    with gpu_owner(), torch.inference_mode():
        for arm in models or ("direct", "human", "predicted"):
            actor = models[arm]() if models else VisualAnswer(arm, "cuda")
            actor.answer(tiles, [qs[0]], [bs[0]])
            torch.cuda.synchronize()
            meter = snapshot()
            start = monotonic()
            vals = []
            for q, p in zip(qs, bs):
                raw, _ = actor.answer(tiles, [q], [p])
                ids, stop = raw[0]
                vals.append(("".join(chars[k] for k in ids), stop))
            torch.cuda.synchronize()
            costs[arm] = dict(
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
            )
            runs[arm] = vals
            del actor
        with server() as url:
            ask(url, VLM, dict(role="user", content="只回答好。"))
            meter = snapshot()
            start = monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                runs["qwen"] = list(pool.map(lambda m: ask(url, VLM, m), messages))
            costs["qwen"] = dict(
                seconds=monotonic() - start,
                joules=consumed(meter, snapshot())["joules"],
            )
    score = {}
    for k, vals in runs.items():
        sc = {}
        for t in ("statements", "exam"):
            matched = [
                (r, a, e)
                for r, (a, e) in zip(rows, vals, strict=True)
                if r["task"] == t
            ]
            sc[t] = dict(
                n=len(matched),
                normalized_exact=sum(
                    e and any(wording(a) == wording(y) for y in r["answers"])
                    for r, a, e in matched
                ),
                fact=sum(
                    e
                    and contains_fact(
                        a, r["facts"]["answer"], "catalogue" if t == "exam" else t
                    )
                    for r, a, e in matched
                ),
            )
        score[k] = sc
    dump(
        public / "comparison.json",
        dict(
            score=score,
            cost=costs,
            vision_weights=MODELS[VLM],
            outputs=runs,
            families=[r["family"] for r in rows],
        ),
    )
    print(score, flush=True)


if __name__ == "__main__":
    configure_cuda_cli()
    main()
